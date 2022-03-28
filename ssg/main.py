import numpy as np
import torch
import torch.nn as nn
# import qpth
import scipy
import copy
import argparse
import time
import pandas as pd

from collections import OrderedDict, namedtuple

import torch.optim as optim
from knitro import *

import sys
sys.path.insert(1, '../')
from model import MAS, MAS_forward, block_diagonal
from utils import project

softmax = nn.Softmax(dim=0)
omega = 5
strategy_perturbation = 0.0

def generate_instance(n, number_of_agents, number_of_actions, max_payoff=10):
    target_payoffs = torch.rand(2,n) * max_payoff
    target_payoffs[0] = 0 # 0 -> reward
    target_payoffs[1] = - target_payoffs[1] # 1 -> penalty

    masks = []
    for agent in range(number_of_agents):
        choice = np.random.choice(n, size=number_of_actions[agent], replace=False)
        masks.append(choice)

    payoffs = - torch.rand(number_of_agents,n) * max_payoff # penalty
    for agent in range(number_of_agents):
        payoffs[agent][list(set(range(n)) - set(masks[agent]))] = 0
    # payoffs = torch.einsum('ijk,ik->ijk', payoffs, masks)
    attractiveness = torch.randn(n)
    number_of_resources = torch.ones(number_of_agents) * 10  # individual budget = 10 # torch.rand(number_of_agents) * torch.Tensor(number_of_actions) / 3

    return target_payoffs, payoffs, masks, attractiveness, number_of_resources

def strategy_initialization(number_of_actions):
    strategies = [torch.zeros(num) for num in number_of_actions]
    return strategies

def attacker_response(attractiveness, coverage):
    # print(attractiveness.shape, coverage.shape)
    # print(coverage)
    prob = softmax(attractiveness - omega * coverage)
    return prob

def attacker_response_derivative(attractiveness, coverage):
    prob = softmax(attractiveness - omega * coverage)
    return torch.diag(- prob * omega) + omega * torch.ger(prob, prob) # prob * prob * omega

def get_social_utility(attractiveness, masks, strategies, target_payoffs):
    n = target_payoffs.shape[-1]
    vulnerability = torch.ones(n)
    for i in range(len(strategies)):
        agent_coverage = torch.zeros(n)
        agent_coverage[masks[i]] = strategies[i]
        vulnerability = vulnerability * (1 - agent_coverage)

    coverage = 1 - vulnerability
    prob = attacker_response(attractiveness, coverage)
    return torch.sum(prob * coverage * target_payoffs[0] + prob * (1 - coverage) * target_payoffs[1]) # reward + penalty

def get_cooperated_social_utility(attractiveness, masks, strategies, target_payoffs):
    # assuming agents have perfect communication (to avoid overlap of coverage)
    # coverage is additive instead of multiplicative with non-covered probability
    n = target_payoffs.shape[-1]
    coverage = torch.zeros(n)
    for i in range(len(strategies)):
        agent_coverage = torch.zeros(n)
        agent_coverage[masks[i]] = strategies[i]
        coverage += agent_coverage

    coverage = torch.clamp(coverage, max=1, min=0)
    prob = attacker_response(attractiveness, coverage)
    return torch.sum(prob * coverage * target_payoffs[0] + prob * (1 - coverage) * target_payoffs[1])

def get_individual_utility(attractiveness, masks, strategies, payoffs, agent):
    n = payoffs.shape[-1]
    vulnerability = torch.ones(n)
    for i in range(len(strategies)):
        agent_coverage = torch.zeros(n)
        agent_coverage[masks[i]] = strategies[i]
        vulnerability = vulnerability * (1 - agent_coverage)

    coverage = 1 - vulnerability
    prob = attacker_response(attractiveness, coverage)
    penalty = torch.sum(torch.log(strategies[agent]) * strategies[agent])
    return torch.sum(prob * (1 - coverage) * payoffs[agent])
    # return torch.sum(prob * coverage * payoffs[agent][0] + prob * (1 - coverage) * payoffs[agent][1])


# def get_individual_derivative_manual(attractiveness, masks, strategies, payoffs, agent, retain_graph=True, create_graph=True):
#     n = payoffs.shape[-1]
#     vulnerability = torch.ones(n)
#     vulnerability_derivative = torch.ones(n)
#     for i in range(len(strategies)):
#         agent_coverage = torch.zeros(n)
#         agent_coverage[masks[i]] = strategies[i]
#         vulnerability = vulnerability * (1 - agent_coverage)
#         if i != agent:
#             vulnerability_derivative = vulnerability_derivative * (1 - agent_coverage)
#
#     coverage =  1 - vulnerability
#     prob = attacker_response(attractiveness, coverage)
#     prob_derivative = attacker_response_derivative(attractiveness, coverage)
#     grad = (prob * payoffs[agent][0] - prob * payoffs[agent][1]) * vulnerability_derivative + \
#         ((coverage * payoffs[agent][0] + (1 - coverage) * payoffs[agent][1]) @ prob_derivative) * vulnerability_derivative
#
#     penalty_grad = torch.log(strategies[agent]) + 1
#
#     return grad[masks[agent]]

def get_individual_derivative(attractiveness, masks, strategies, payoffs, agent, retain_graph=True, create_graph=True):
    x = torch.autograd.Variable(strategies[agent], requires_grad=True)
    strategies[agent] = x

    utility = get_individual_utility(attractiveness, masks, strategies, payoffs, agent)
    du_dx = torch.autograd.grad(utility, x, retain_graph=retain_graph, create_graph=create_graph)[0]
    return du_dx

def get_individual_hessian(attractiveness, masks, strategies, payoffs, agent, retain_graph=True, create_graph=True):
    xs = [torch.autograd.Variable(strategy, requires_grad=True) for strategy in strategies]

    # jac = get_individual_derivative_manual(attractiveness, masks, xs, payoffs, agent)
    jac = get_individual_derivative(attractiveness, masks, xs, payoffs, agent)
    hessian = torch.cat([torch.cat([torch.autograd.grad(jac[i], x, retain_graph=True, create_graph=True)[0] for x in xs]).view(1,-1) for i in range(len(jac))])

    return hessian

def get_violation(masks, strategies, initial_payoffs, payoffs, budget):
    n = payoffs.shape[-1]
    payoff_difference = torch.clamp(payoffs - initial_payoffs, min=0)
    total_payment = 0
    for i in range(len(strategies)):
        agent_coverage = torch.zeros(n)
        agent_coverage[masks[i]] = strategies[i]
        total_payment += payoff_difference[i] @ (1 - agent_coverage)
        # total_payment += payoff_difference[i][0] @ agent_coverage + payoff_difference[i][1] @ (1 - agent_coverage)
    return total_payment - budget

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SSG with Multiple Defenders')

    parser.add_argument('--agents', type=int, default=5, help='number of agents')
    parser.add_argument('--targets', type=int, default=100, help='number of targets')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--budget', type=float, default=1, help='budget')
    parser.add_argument('--constrained', default=False, action='store_true', help='Adding incentive compatible constraints')
    parser.add_argument('--no-constrained', dest='constrained', action='store_false', help='Adding incentive compatible constraints')
    parser.add_argument('--method', type=str, default='', help='optimization method (diffmulti, SLSQP, trust)')
    parser.add_argument('--forward-iterations', type=int, default=100, help='maximum iterations the equilibrium finding oracle uses')
    parser.add_argument('--gradient-iterations', type=int, default=5000, help='maximum iterations of gradient descent')
    parser.add_argument('--actions', type=int, default=50, help='number of available actions per agent. Here it refers to how many targets an agent can choose to cover')
    parser.add_argument('--prefix', type=str, default='', help='prefix of the filename')
    parser.add_argument('--init', default=False, action='store_true', help='Using dynamic initialization')
    parser.add_argument('--alg', type=str, default='', help='algorithm to use for knitro, default auto')

    args = parser.parse_args()

    number_of_targets = args.targets
    number_of_agents = args.agents
    constrained = args.constrained
    budget = args.budget # intervention designer's budget
    method = args.method
    max_iterations = args.forward_iterations
    number_of_iterations = args.gradient_iterations
    actions = args.actions
    prefix = args.prefix
    initialization = args.init
    assert(method in ['diffmulti', 'trust', 'SLSQP', 'annealing', 'knitro'])
    knitro_alg = args.alg
    assert(knitro_alg in ['auto', 'direct', 'cg', 'active', 'sqp', 'multi'])

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    GAMMA = 0.5

    print('Constrained:', constrained)
    if constrained:
        filename = 'exp/' + prefix + 'constrained_{}.csv'.format(method)
    else:
        filename = 'exp/' + prefix + 'unconstrained_{}.csv'.format(method)

    print('Budget {}, random seed {}...'.format(budget, seed))

    number_of_actions = [actions] * number_of_agents
    target_payoffs, initial_payoffs, masks, attractiveness, number_of_resources = generate_instance(number_of_targets, number_of_agents, number_of_actions)

    strategies = [torch.zeros(number_of_actions[agent]) for agent in range(number_of_agents)]
    # ========================= debugging =========================
    # print('debugging...')
    # print(strategies)
    # social_utility = get_social_utility(attractiveness, masks, strategies, target_payoffs)
    # print('social utility', social_utility)

    # start_time = time.time()
    # for count in range(100):
    #     manual_grads = [get_individual_derivative_manual(attractiveness, masks, strategies, payoffs=initial_payoffs, agent=agent) for agent in range(number_of_agents)]
    # print('elapsed time for manual computation:', time.time() - start_time)

    # start_time = time.time()
    # for count in range(100):
    #     grads = [get_individual_derivative(attractiveness, masks, strategies, payoffs=initial_payoffs, agent=agent, retain_graph=True, create_graph=True) for agent in range(number_of_agents)]
    # print('elapsed time for autograd computation:', time.time() - start_time)

    # print('manual gradient:', manual_grads)
    # print('autograd:', grads)

    # =================== function definition ====================
    print('testing multi-agent model...')
    eps = 1e-4
    bounds = [[(0.0, 1.0)] * number_of_actions[agent] for agent in range(number_of_agents)]
    constraints = [[torch.ones(1,len(masks[agent])), torch.ones(1) * number_of_resources[agent], # A, b
        torch.cat([-torch.eye(len(masks[agent])), torch.eye(len(masks[agent]))]), torch.cat([torch.zeros(len(masks[agent])), torch.ones(len(masks[agent]))])] for agent in range(number_of_agents)] # G, h list

    fs = [lambda x,y,agent=agent: get_individual_utility(attractiveness=attractiveness, masks=masks, strategies=y, payoffs=x, agent=agent) for agent in range(number_of_agents)]
    f_jacs = [lambda x,y,agent=agent: get_individual_derivative(attractiveness=attractiveness, masks=masks, strategies=copy.deepcopy(y), payoffs=x, agent=agent) for agent in range(number_of_agents)]

    # ================= testing initial utility ==================
    initial_strategies = strategy_initialization(number_of_actions) # [torch.zeros(number_of_actions[agent]) for agent in range(number_of_agents)]
    precomputed_initial_strategies = MAS_forward(initial_payoffs, fs, constraints, bounds, initial_strategies, f_jacs, eps=eps, max_iterations=max_iterations, gamma=GAMMA)
    # precomputed_initial_strategies = [strategy.detach() for strategy in precomputed_initial_strategies]
    initial_social_utility = get_social_utility(attractiveness, masks, precomputed_initial_strategies, target_payoffs)
    # initial_strategies = copy.deepcopy(precomputed_initial_strategies)
    print('initial social utility', initial_social_utility)

    # ================== dictatorship utility ====================
    dictatorship_A = block_diagonal([constraints[agent][0] for agent in range(number_of_agents)])
    dictatorship_b = torch.cat([constraints[agent][1] for agent in range(number_of_agents)])
    dictatorship_constraints = [{'type': 'eq', 'fun': lambda x: dictatorship_A.numpy() @ x - dictatorship_b.numpy()}]
    print('Computing the optimal dictatorship utility...')
    def get_dictatorship_objective(x):
        tmp_strategies = []
        tmp_index = 0
        for agent in range(number_of_agents):
            tmp_strategies.append(torch.Tensor(x[tmp_index:tmp_index + number_of_actions[agent]]))
            tmp_index += number_of_actions[agent]
        tmp_social_welfare = get_cooperated_social_utility(attractiveness, masks, tmp_strategies, target_payoffs)
        return -tmp_social_welfare.item()

    dictatorship_options = {'disp': True, 'maxiter': 1000, 'eps': 1e-5, 'ftol': 1e-8}
    dictatorship_bounds = [(0,1) for _ in range(sum(number_of_actions))]
    initial_dictatorship_strategies = torch.cat(precomputed_initial_strategies).numpy() # np.zeros(sum(number_of_actions))
    dictatorship_optimization_result = scipy.optimize.minimize(get_dictatorship_objective, initial_dictatorship_strategies, method='trust-constr', constraints=dictatorship_constraints, bounds=dictatorship_bounds)

    dictatorship_objective = - dictatorship_optimization_result['fun']
    print('dictatorship objective:', dictatorship_objective)
    print('dictatorship solution:', dictatorship_optimization_result['x'])

    # ================= optimize social utility ==================
    all_index = pd.MultiIndex.from_product([[method], ['obj', 'violation', 'forward time', 'backward time']], names=['method', 'measure'])
    df = pd.DataFrame(np.zeros((1, all_index.size)), columns=all_index).T

    # ============= differentiable multi-agent systems ===========
    if method == 'diffmulti':
        lr = 1e-2
        payoffs = initial_payoffs.clone()
        payoffs.requires_grad = True

        if constrained:
            mu = 10
            multipliers = torch.rand(1)
            slack_variables = torch.zeros(1)
            optimizer = optim.Adam([payoffs, slack_variables], lr=lr)
        else:
            mu = 10
            multipliers = torch.rand(1)
            slack_variables = torch.zeros(1)
            optimizer = optim.Adam([payoffs, slack_variables], lr=lr)

        counter = 0
        social_utility_list = []
        violation_list = []
        forward_time, backward_time = 0, 0
        for iteration in range(-1,number_of_iterations):
            start_time = time.time()
            optimizer.zero_grad()

            presolved_strategies = list(MAS_forward(payoffs.detach(), fs, constraints, bounds, initial_strategies, f_jacs, eps=eps, max_iterations=max_iterations, gamma=GAMMA))

            forward_time += time.time() - start_time
            start_time = time.time()

            grads = torch.cat([f_jacs[agent](payoffs, presolved_strategies) for agent in range(number_of_agents)])
            # grads = torch.cat([get_individual_derivative_manual(attractiveness, masks, presolved_strategies, payoffs=payoffs, agent=agent) for agent in range(number_of_agents)])
            hessians = [get_individual_hessian(attractiveness, masks, presolved_strategies, payoffs=payoffs, agent=agent).detach() for agent in range(number_of_agents)]

            mas_model = MAS(fs, number_of_actions, constraints, bounds, presolved_strategies, hessians)
            strategies = mas_model(*grads)

            social_utility = get_social_utility(attractiveness, masks, strategies, target_payoffs)
            social_utility_list.append(social_utility.item())

            # ======================== augmented lagrangian method =========================
            if constrained:
                violation = get_violation(masks, strategies, initial_payoffs, payoffs, budget) + slack_variables
                total_violation = multipliers @ violation + 0.5 * mu * torch.sum(violation * violation)
                (-social_utility + total_violation).backward()
            else:
                violation = torch.norm(payoffs - initial_payoffs, p=1) - budget + slack_variables
                total_violation = multipliers @ violation + 0.5 * mu * torch.sum(violation * violation)
                (-social_utility + total_violation).backward()

            payoffs.grad.data = payoffs.grad.data / torch.norm(payoffs.grad.data)
            if iteration >= 0:
                optimizer.step()

            if constrained:
                slack_variables.data = torch.clamp(slack_variables, min=0)
                if iteration % 100 == 0:
                    multipliers.data = multipliers.data + mu * violation
                payoffs.data = initial_payoffs.data + torch.clamp(payoffs.data - initial_payoffs.data, min=0)

                true_violation = get_violation(masks, strategies, initial_payoffs, payoffs, budget).detach()
                total_violation = torch.sum(torch.clamp(true_violation, min=0)).item()
            else:
                slack_variables.data = torch.clamp(slack_variables, min=0)
                if iteration % 100 == 0:
                    multipliers.data = multipliers.data + mu * violation
                payoffs.data = initial_payoffs.data + torch.clamp(payoffs.data - initial_payoffs.data, min=0)
                # payoffs.data = initial_payoffs.data + project(torch.clamp(payoffs.data - initial_payoffs.data, min=0), budget)
                total_violation = torch.clamp(torch.norm(payoffs.detach() - initial_payoffs, p=1) - budget, min=0).item()
            backward_time += time.time() - start_time

            violation_list.append(total_violation)
            print('iteration #{} with social utility: {}, difference: {}, violation: {}'.format(iteration, social_utility, torch.norm(initial_payoffs - payoffs, p=1), total_violation))
            print(forward_time, backward_time)

            if initialization:
                initial_strategies = [strategy.detach() for strategy in presolved_strategies] # updating initialization

        best_idx = -1 # np.argmax(np.array(social_utility_list) - 1000 * np.array(violation_list))
        optimal_social_utility = social_utility_list[best_idx]
        total_violation = violation_list[best_idx]

        df.loc[('diffmulti', 'obj')] = optimal_social_utility
        df.loc[('diffmulti', 'violation')] = total_violation
        df.loc[('diffmulti', 'forward time')] = forward_time
        df.loc[('diffmulti', 'backward time')] = backward_time

        print('optimal social utility: {}, violation: {}'.format(optimal_social_utility, total_violation))

    # ========================= scipy optimization ============================
    if method in ['SLSQP', 'trust', 'annealing']:
        print('Scipy optimization with blackbox constraints...')
        start_time = time.time()
        # dynamic initialization
        scipy_initial_strategies = strategy_initialization(number_of_actions)
        def get_objective(input_payoffs):
            if initialization:
                global scipy_initial_strategies
            else:
                scipy_initial_strategies = strategy_initialization(number_of_actions)

            torch_input_payoffs = initial_payoffs.clone() # initialize
            tmp_index = 0
            assert(len(input_payoffs) == sum(number_of_actions)) # make sure the input size matches to the size of changeable variables
            for i in range(number_of_agents):
                torch_input_payoffs[i,masks[i]] += torch.Tensor(input_payoffs[tmp_index:tmp_index+number_of_actions[i]])
                tmp_index += number_of_actions[i]

            strategies = list(MAS_forward(torch_input_payoffs, fs, constraints, bounds, scipy_initial_strategies, f_jacs, eps=eps, max_iterations=max_iterations, gamma=GAMMA))
            violation = get_violation(masks, strategies, initial_payoffs, torch_input_payoffs, budget).detach().item()
            # strategies = list(MAS_forward(torch.Tensor(input_payoffs).reshape(number_of_agents,2,number_of_targets), fs, constraints, bounds, scipy_initial_strategies, f_jacs, eps=eps, max_iterations=max_iterations, gamma=GAMMA))
            social_utility = get_social_utility(attractiveness, masks, strategies, target_payoffs)
            scipy_initial_strategies = [strategy.detach() + strategy_perturbation * torch.randn(strategy.shape) for strategy in strategies]
            if method in ['SLSQP', 'trust']:
                # print(social_utility, violation, sum(input_payoffs))
                return -social_utility.detach().item() # maximize so adding an additional negation
            elif method in ['annealing']:
                if constrained:
                    violation = get_violation(masks, strategies, initial_payoffs, torch_input_payoffs, budget).detach().item()
                else:
                    distance = torch.norm(torch.Tensor(input_payoffs), p=1).detach().item()
                    violation = distance - budget
                obj = -social_utility.detach().item() if violation <= 0 else np.inf
                # print(social_utility, violation, sum(input_payoffs), obj)
                return obj

        if constrained:
            def get_constraint(input_payoffs):
                if initialization:
                    global scipy_initial_strategies
                else:
                    scipy_initial_strategies = strategy_initialization(number_of_actions)

                torch_input_payoffs = initial_payoffs.clone() # initialize
                tmp_index = 0
                assert(len(input_payoffs) == sum(number_of_actions)) # make sure the input size matches to the size of changeable variables
                for i in range(number_of_agents):
                    torch_input_payoffs[i,masks[i]] += torch.Tensor(input_payoffs[tmp_index:tmp_index+number_of_actions[i]])
                    tmp_index += number_of_actions[i]

                strategies = list(MAS_forward(torch_input_payoffs, fs, constraints, bounds, scipy_initial_strategies, f_jacs, eps=eps, max_iterations=max_iterations, gamma=GAMMA))
                violation = get_violation(masks, strategies, initial_payoffs, torch_input_payoffs, budget).detach().item()
                # subsidy = - initial_payoffs.detach().flatten().numpy() + torch_input_payoffs.flatten().numpy()
                scipy_initial_strategies = [strategy.detach() + strategy_perturbation * torch.randn(strategy.shape) for strategy in strategies]
                return -violation
                # return np.concatenate([subsidy, [-violation]])
        else:
            def get_constraint(input_payoffs):
                distance = torch.norm(torch.Tensor(input_payoffs), p=1).detach().item()
                return budget - distance
                # subsidy = - initial_payoffs.detach().flatten().numpy() + input_payoffs
                # return np.concatenate([subsidy, [budget - distance]])

        violation_constraint = [{'type': 'ineq', 'fun': get_constraint}]
        if method == 'SLSQP':
            print('SLSQP optimization')
            optimization_method = 'SLSQP'
            options = {'disp': True, 'maxiter': 500, 'eps': 1e-4}
        elif method == 'trust':
            print('trust region optimization')
            optimization_method = 'trust-constr'
            options = {'verbose': 2, 'maxiter': 500, 'initial_tr_radius': 1e-4, 'initial_constr_penalty': 1e2}

        policy_bounds = [(0, budget) for _ in range(sum(number_of_actions))]
        scipy_initial_payoffs = torch.zeros(sum(number_of_actions))
        if method in ['SLSQP', 'trust']:
            optimization_result = scipy.optimize.minimize(get_objective, scipy_initial_payoffs.flatten().numpy(), method=optimization_method, constraints=violation_constraint, options=options, bounds=policy_bounds)
        elif method in ['annealing']:
            optimization_result = scipy.optimize.dual_annealing(get_objective, bounds=policy_bounds, x0=scipy_initial_payoffs.flatten().numpy())

        print(optimization_result)
        tmp_scipy_payoffs = torch.Tensor(optimization_result['x']) #.reshape(number_of_agents,2,number_of_targets)
        scipy_payoffs = initial_payoffs.clone() # initialize
        tmp_index = 0
        assert(len(tmp_scipy_payoffs) == sum(number_of_actions)) # make sure the input size matches to the size of changeable variables
        for i in range(number_of_agents):
            scipy_payoffs[i,masks[i]] += tmp_scipy_payoffs[tmp_index:tmp_index+number_of_actions[i]]
            tmp_index += number_of_actions[i]

        scipy_strategies = list(MAS_forward(scipy_payoffs, fs, constraints, bounds, scipy_initial_strategies, f_jacs, eps=eps, max_iterations=max_iterations, gamma=GAMMA))

        if constrained:
            scipy_true_violation = get_violation(masks, scipy_strategies, initial_payoffs, scipy_payoffs, budget).detach()
            scipy_total_violation = torch.sum(torch.clamp(scipy_true_violation, min=0)).item()
        else:
            scipy_total_violation = torch.clamp(torch.norm(scipy_payoffs, p=1) - budget, min=0).item()

        scipy_social_utility = get_social_utility(attractiveness, masks, scipy_strategies, target_payoffs).item()
        print('scipy social utility: {}, violation: {}'.format(scipy_social_utility, scipy_total_violation))

        df.loc[(method, 'obj')] = scipy_social_utility
        df.loc[(method, 'violation')] = scipy_total_violation
        df.loc[(method, 'forward time')] = time.time() - start_time

    # ========================== knitro solver ================================
    if method == 'knitro':
        start_time = time.time()

        # scipy_initial_strategies = strategy_initialization(number_of_actions)
        initial_strategies = strategy_initialization(number_of_actions)

        print('n targets', number_of_targets, 'n agents', number_of_agents)
        print('n actions', number_of_actions)

        strategy_indices = number_of_actions
        strategy_size = sum(number_of_actions)

        print('strategy size', strategy_size)

        subsidy_size = len(initial_payoffs.flatten()) # reimbursement subsidy
        subsidy_shape = list(initial_payoffs.shape)

        def callbackEval(kc, cb, evalRequest, evalResult, userParams):
            if evalRequest.type != KN_RC_EVALFC:
                print("*** callbackEvalF incorrectly called with eval type %d" % evalRequest.type)
                return -1

            assert len(evalRequest.x) == subsidy_size + strategy_size * 2 + strategy_size * 2 + number_of_agents
            assert len(evalResult.c) == strategy_size + strategy_size + (number_of_agents + 1)

            x = torch.Tensor(evalRequest.x)
            subsidy = torch.reshape(x[:subsidy_size], subsidy_shape) # pi - reimbursement
            strategies = list(torch.split(x[subsidy_size:subsidy_size+strategy_size], strategy_indices)) # x
            strategies_complement = list(torch.split(x[subsidy_size+strategy_size:subsidy_size+strategy_size*2], strategy_indices)) # y
            lamb = torch.split(x[subsidy_size+strategy_size*2:subsidy_size+strategy_size*3], strategy_indices) # lamb
            mu   = torch.split(x[subsidy_size+strategy_size*3:subsidy_size+strategy_size*4], strategy_indices) # mu
            nu   = x[-number_of_agents:] # nu

            # Evaluate nonlinear objective
            # strategies = torch.autograd.Variable(x[subsidy_size:subsidy_size+strategy_size].view(-1,1), requires_grad=True) # x
            obj = get_social_utility(attractiveness, masks, strategies, target_payoffs)  # all is fixed except 'strategies'
            evalResult.obj = obj.item()

            # Evaluate constraints
            # constraint - lagrangian
            # c_lagrangian = - strategy_gradient - lamb + mu + nu # nu times 1?
            # evalResult.c[:number_of_agents] = c_lagrangian# lagrangian

            # strategy_gradient = torch.autograd.grad(obj, strategies)[0].detach().flatten().numpy()
            cur = 0
            for agent in range(number_of_agents): # lagrangian
                f_jac_vals = f_jacs[agent](initial_payoffs + subsidy, strategies)
                tmp_lagrangian = (-f_jac_vals - lamb[agent] + mu[agent] + nu[agent]).detach().numpy()
                evalResult.c[cur:cur+number_of_actions[agent]] = tmp_lagrangian
                cur += number_of_actions[agent]

            # constraint - slack
            evalResult.c[strategy_size:strategy_size+strategy_size] = (torch.cat(strategies) + torch.cat(strategies_complement) - 1).flatten().detach().numpy()  # slack constraints

            # constraint - budget
            # individual budget
            for agent in range(number_of_agents):
                evalResult.c[strategy_size+strategy_size+agent] = sum(strategies[agent]).item() - number_of_resources[agent]

            # total budget
            evalResult.c[-1] = get_violation(masks, strategies, initial_payoffs, initial_payoffs + subsidy, budget).detach().item()

            # print('--------')
            # print('subsidy', subsidy)
            #
            # print('constraints lagrangian', evalResult.c[:strategy_size])
            # print('constraints slack', evalResult.c[strategy_size:strategy_size+strategy_size])
            # print('constraints individual budget', evalResult.c[strategy_size+strategy_size:strategy_size+strategy_size+number_of_agents])
            # print('constraints total budget', evalResult.c[-1])

            return 0

        # Function evaluating the components of the first derivatives/gradients
        # of the objective and the constraint involved in this callback.
        def callbackEvalGA(kc, cb, evalRequest, evalResult, userParams):
            if evalRequest.type != KN_RC_EVALGA:
                print ("*** callbackEvalGA incorrectly called with eval type %d" % evalRequest.type)
                return -1

            x = torch.Tensor(evalRequest.x)
            assert len(x) == subsidy_size + strategy_size*4 + number_of_agents

            strategies = list(torch.split(x[subsidy_size:subsidy_size+strategy_size], strategy_indices)) # x
            strategies_var = []
            for strategy in strategies:
                strategy_var = torch.autograd.Variable(strategy, requires_grad=True)
                strategies_var.append(strategy_var)

            # subsidy    = torch.autograd.Variable(torch.reshape(x[:subsidy_size], subsidy_shape), requires_grad=True)

            obj = get_social_utility(attractiveness, masks, strategies_var, target_payoffs)  # all is fixed except 'strategies'

            # subsidy_gradient  = torch.autograd.grad(obj, subsidy, retain_graph=True, create_graph=True)[0].detach().flatten().numpy() # pi gradient
            # evalResult.objGrad[:subsidy_size] = subsidy_gradient # pi

            cur = subsidy_size
            for i in range(number_of_agents):
                strategy_gradient = torch.autograd.grad(obj, strategies_var[i], retain_graph=True, create_graph=True)[0].detach().flatten().numpy()

                evalResult.objGrad[cur:cur+number_of_actions[i]] = strategy_gradient # x
                cur += number_of_actions[i]

            return 0


        # pi, x, y, lamb, mu, nu respectively
        knitro_initial_strategies = torch.cat(list(MAS_forward(initial_payoffs, fs, constraints, bounds, initial_strategies, f_jacs, eps=eps, max_iterations=max_iterations, gamma=GAMMA)))
        variable_size = subsidy_size + strategy_size * 2 + strategy_size * 2 + number_of_agents
        variables = Variables(nV=variable_size,
                xLoBnds=[0]*(variable_size),
                xUpBnds=[KN_INFINITY]*(subsidy_size) + [1] * (strategy_size * 2) + [KN_INFINITY] * (strategy_size * 2 + number_of_agents),
                # xInitVals=[0]*(subsidy_size + strategy_size * 2 + strategy_size * 2 + number_of_agents)
                xInitVals=torch.cat([torch.zeros(subsidy_size), knitro_initial_strategies, 1-knitro_initial_strategies, torch.zeros(strategy_size*2+number_of_agents)]).detach().numpy()
                )

        # lagrangian, slack, budget constraints
        constraint_size = strategy_size + strategy_size + (number_of_agents + 1)
        eq_eps = 0 # epsilon allowance for the equality constraint
        knitro_constraints = Constraints(nC=constraint_size,
                # cEqBnds=[0] * constraint_size, # equality constraint
                cLoBnds=[-eq_eps] * (strategy_size + strategy_size) + (-number_of_resources).tolist() + [-KN_INFINITY] * 1, # inequality constraint
                cUpBnds=[eq_eps] * (strategy_size + strategy_size) + [0] * number_of_agents + [0] * 1,
                cNames=['eqCon{}'.format(i) for i in range(constraint_size)])

        print('variable size', variable_size, 'constraint size', constraint_size)

        complementarity_size = strategy_size * 2
        complementarity = ComplementarityConstraints( #nCc=complementarity_size,
                indexComps1=list(range(subsidy_size, subsidy_size+strategy_size*2)), # x, y
                indexComps2=list(range(subsidy_size+strategy_size*2, subsidy_size+strategy_size*4)), # lamb, mu
                ccTypes=[KN_CCTYPE_VARVAR]*(strategy_size*2),
                cNames=['compCon{}'.format(i) for i in range(strategy_size*2)]
                )

        callback = Callback(funcCallback=callbackEval,
            # evalObj=True,
            # evalFCGA=False, # If evalFCGA is equal to True, the callback should also evaluate the relevant first derivatives/gradients.

            # evaluate the components of the first derivatives/gradients of the objective
            gradCallback=callbackEvalGA,
            objGradIndexVars=KN_DENSE, # KN_DENSE for all variables
            # jacIndexCons=KN_DENSE_ROWMAJOR, # if evaluating jacobian of constraints
            # hessCallback=callbackEvalH, # if evaluating hessian
            # hessIndexVars1=KN_DENSE_ROWMAJOR,
            # hessianNoFAllow=True,
            )

        objective = Objective(objGoal=KN_OBJGOAL_MAXIMIZE)

        # Solve the problem
        knitro_result = optimize(variables=variables,
                objective=objective,
                constraints=knitro_constraints,
                compConstraints=complementarity,
                callbacks=callback,
                options={
                    'ms_enable': True,              # multistart. NOTE: we can set multistart to run in parallel
                    'ms_maxsolves': 100,
                    'bar_feasible': 'get',     # get, stay - prioritze finding feasible point
                    'algorithm': knitro_alg,         # direct, cg, active, sqp, multi # which algorithm to use
                    # 'act_lpsolver': 'cplex',      # use licensed LP solver in active or sqp mode
                    # 'act_qppenalty': 'all',       # Constraint penalization for QP subproblems.
                    # 'bar_initpt': 'convex',       # convex, nearbnd, central - selecting first initial point (if not specified)
                    # 'bar_penaltycons': 'all',     # Whether or not to penalize constraints in the barrier algorithms.
                    # 'debug': 'problem', 'debug',  # problem, debug - output debugging
                    # 'derivcheck': 'all',          # Whether to perform a derivative check on the model
                    # 'honorbnds': 'always',        # auto, no, always, initpt - Whether to enforce satisfaction of simple bounds at all iterations
                    # 'mip_branchrule': 'most_frac', # Specifies the MIP branching rule for choosing a variable.
                    }
                )

        # print('knitro result', knitro_result.x)

        # Processing result
        knitro_perturbation = torch.reshape(torch.Tensor(knitro_result.x[:subsidy_size]), subsidy_shape) # pi
        # print('knitro used budget:', torch.sum(knitro_perturbation), 'real budget:', budget)
        # if torch.sum(knitro_perturbation) > budget: # rounding
        #     knitro_perturbation = knitro_perturbation * budget / torch.sum(knitro_perturbation)

        # print('initial_payoff_matrices', initial_payoffs)

        initial_strategies = strategy_initialization(number_of_actions)
        perturbed_payoffs = initial_payoffs.detach().clone() + knitro_perturbation
        strategies = list(MAS_forward(perturbed_payoffs, fs, constraints, bounds, initial_strategies, f_jacs, eps=eps, max_iterations=max_iterations, gamma=GAMMA))

        # print('knitro strategies', strategies)
        # print('knitro perturbations', knitro_perturbation)

        knitro_true_violation = get_violation(masks, strategies, initial_payoffs, perturbed_payoffs, budget).detach()
        knitro_total_violation = torch.sum(torch.clamp(knitro_true_violation, min=0)).item()
        knitro_social_utility = get_social_utility(attractiveness, masks, strategies, target_payoffs).item()

        print('amount of true violation', knitro_true_violation.item())
        print('amount of total violation', knitro_total_violation)
        print('amount of social utility', knitro_social_utility)

        knitro_runtime = time.time() - start_time
        print('knitro runtime', knitro_runtime)

        # Recording result
        df.loc[(method, 'obj')] = knitro_social_utility
        df.loc[(method, 'violation')] = knitro_total_violation
        # df.loc[(method, 'forward time')] = knitro_result['TimeReal']
        df.loc[(method, 'forward time')] = knitro_runtime



    # ================ random social utility =================
    start_time = time.time()
    random_perturbation = torch.rand(initial_payoffs.shape)
    random_perturbation = random_perturbation / torch.sum(random_perturbation) * budget
    perturbed_payoffs = initial_payoffs + random_perturbation

    strategies = list(MAS_forward(perturbed_payoffs, fs, constraints, bounds, precomputed_initial_strategies, f_jacs, eps=eps, max_iterations=max_iterations, gamma=GAMMA))

    if constrained:
        random_true_violation = get_violation(masks, strategies, initial_payoffs, perturbed_payoffs, budget).detach()
        random_total_violation = torch.sum(torch.clamp(random_true_violation, min=0)).item()
    else:
        random_total_violation = torch.clamp(torch.norm(perturbed_payoffs - initial_payoffs, p=1) - budget, min=0).item()

    random_social_utility = get_social_utility(attractiveness, masks, strategies, target_payoffs)
    print('random social utility: {}, violation: {}'.format(random_social_utility, random_total_violation))

    f = open(filename, 'a')
    f.write('{}, {}, {}, {}, {}, {},'.format(seed, budget, initial_social_utility, dictatorship_objective, random_social_utility, random_total_violation) +
            '{}\n'.format(', '.join([str(x) for x in df.values[:,0].tolist()]))
        )

    f.close()
    print('total runtime:', time.time() - start_time)
