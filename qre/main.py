import numpy as np
import torch
import torch.nn as nn
# import qpth
import scipy
import copy
import time
import argparse

import pandas as pd
import torch.optim as optim
from knitro import *

import sys
sys.path.insert(1, '../')
from model import MAS, MAS_forward
from utils import project

qr_function = nn.Softmax(dim=0)
lamb = 1
strategy_perturbation = 0.0

def generate_payoff_matrices(n, number_of_actions, max_payoff):
    # n: number of players
    # number_actions: an n-dim array where each entry represents how many actions the corresponding player has
    payoff_matrices = torch.rand(n, *number_of_actions) * max_payoff
    return payoff_matrices

def best_response(strategies, agent, payoff_matrix):
    permutation = list(range(len(payoff_matrix.shape)))
    permutation[agent], permutation[-1] = permutation[-1], permutation[agent]

    permuted_payoff_matrix = payoff_matrix.permute(permutation)
    # flatten_payoff_matrix = permuted_payoff_matrix.flatten(0,-2)

    for i, strategy in enumerate(strategies):
        if i == agent:
            continue
        permuted_payoff_matrix = torch.einsum('i,i...->...', strategy, permuted_payoff_matrix)

    response = qr_function(lamb * permuted_payoff_matrix)

    return response, permuted_payoff_matrix

def strategy_initialization(number_of_actions):
    strategies = [qr_function(torch.rand(i)) for i in number_of_actions]
    return strategies

def solve(payoff_matrices, max_iterations=1000, eps=1e-8):
    strategies = strategy_initialization(payoff_matrices.shape[1:])
    for iteration in range(max_iterations):
        new_strategies = copy.deepcopy(strategies)
        for agent, payoff_matrix in enumerate(payoff_matrices):
            # agent = np.random.randint(len(payoff_matrix.shape)) # randomly choosing an agent to update the response
            response, _ = best_response(strategies, agent, payoff_matrix)
            new_strategies[agent] = response
        difference = sum([torch.norm(new_strategy - strategy) for new_strategy, strategy in zip(new_strategies, strategies)])
        # print('Iteartion #{} with difference {}'.format(iteration, difference))
        if difference < eps:
            break
        else:
            strategies = new_strategies
    return strategies

def get_social_utility(payoff_matrices, strategies):
    payoff_matrix = torch.sum(payoff_matrices, dim=0)
    for strategy in strategies:
        payoff_matrix = torch.einsum('i,i...->...', strategy, payoff_matrix)
    return payoff_matrix

def get_individual_utility(payoff_matrices, strategies, agent):
    for i, strategy in enumerate(strategies):
        if i == 0:
            matrix = torch.einsum('i,i...->...', strategy, payoff_matrices[agent])
        else:
            matrix = torch.einsum('i,i...->...', strategy, matrix)
    penalty = torch.sum(torch.log(strategies[agent]) * strategies[agent])
    return matrix - penalty / lamb

def get_individual_derivative(payoff_matrices, strategies, agent, retain_graph=True, create_graph=True):
    x = torch.autograd.Variable(strategies[agent], requires_grad=True)

    tmp_strategies = strategies # copy.deepcopy(strategies)
    tmp_strategies[agent] = x # TODO BUG BUG!! NO INPLACE ASSIGNMENT IS ALLOWED!!

    utility = get_individual_utility(payoff_matrices, tmp_strategies, agent)
    du_dx = torch.autograd.grad(utility, x, retain_graph=retain_graph, create_graph=create_graph)[0]
    return du_dx

def get_individual_derivative_manual(payoff_matrices, strategies, agent):
    matrix = payoff_matrices[agent].clone()
    for i, strategy in enumerate(strategies):
        if i < agent:
            matrix = torch.einsum('i,i...->...', strategy, matrix)
        elif i > agent:
            matrix = torch.einsum('i,ji...->j...', strategy, matrix)
    penalty_grad = torch.log(strategies[agent]) + 1
    return matrix - penalty_grad / lamb

def get_individual_daction_dx(payoff_matrices, strategies, agent):
    # the second order derivative of agent's utility function with respect to agent's action and then with respect to input x
    x = torch.autograd.Variable(payoff_matrices, requires_grad=True)
    # print('payoff matrices dimension', x.shape)
    jac = get_individual_derivative(x, strategies, agent, retain_graph=True, create_graph=True)
    hessian = torch.cat([torch.autograd.grad(jac[i], x, retain_graph=True, create_graph=True)[0].unsqueeze(0) for i in range(len(jac))])
    return hessian

def get_individual_hessian(payoff_matrices, strategies, agent):
    # the second order derivative of agent's utility function with respect to agent's action and then with respect to all other agents' actions
    xs = [torch.autograd.Variable(strategy, requires_grad=True) for strategy in strategies]
    jac = get_individual_derivative(payoff_matrices, xs, agent, retain_graph=True, create_graph=True)
    hessian = torch.cat([torch.cat([torch.autograd.grad(jac[i], x, retain_graph=True, create_graph=True)[0] for x in xs]).view(1,-1) for i in range(len(jac))])
    return hessian

def get_violation(payoff_matrices, strategies, initial_payoff_matrices, budget):
    subsidy = torch.sum(torch.clamp(payoff_matrices - initial_payoff_matrices, min=0), dim=0) # subsidy per entry
    for i, strategy in enumerate(strategies):
        subsidy = torch.einsum('i,i...->...', strategy, subsidy)
    return subsidy - budget
    # surplus = initial_payoff_matrices - payoff_matrices
    # return torch.cat([subsidy.view(1) - budget, surplus.flatten()])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QRE experiment')

    parser.add_argument('--n', type=int, default=3, help='number of agents')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--budget', type=float, default=1, help='budget')
    parser.add_argument('--constrained', default=False, action='store_true', help='Adding incentive compatible constraints')
    parser.add_argument('--no-constrained', dest='constrained', action='store_false', help='Adding incentive compatible constraints')
    parser.add_argument('--method', type=str, default='', help='optimization method (diffmulti, SLSQP, trust)')
    parser.add_argument('--forward-iterations', type=int, default=100, help='maximum iterations the equilibrium finding oracle uses')
    parser.add_argument('--gradient-iterations', type=int, default=5000, help='maximum iterations of gradient descent')
    parser.add_argument('--actions', type=int, default=10, help='number of available actions per agent. Here it refers how many pure strategies each agent can choose')
    parser.add_argument('--prefix', type=str, default='', help='prefix of the filename')
    parser.add_argument('--init', default=False, action='store_true', help='Using dynamic initialization')

    args = parser.parse_args()

    n = args.n
    seed = args.seed
    constrained = args.constrained
    budget = args.budget
    method = args.method
    max_iterations = args.forward_iterations
    number_of_iterations = args.gradient_iterations
    actions = args.actions
    prefix = args.prefix
    initialization = args.init
    assert(method in ['diffmulti', 'SLSQP', 'trust', 'annealing', 'knitro', 'DO'])

    torch.manual_seed(seed)
    np.random.seed(seed)

    number_of_actions = [actions] * n
    max_payoff = 10
    GAMMA = 0.5
    eps = 1e-6

    print('Constrained:', constrained)
    if constrained:
        filename = 'exp/' + prefix + 'constrained_{}.csv'.format(method)
    else:
        filename = 'exp/' + prefix + 'unconstrained_{}.csv'.format(method)

    print('Budget {}, random seed {}...'.format(budget, seed))

    initial_payoff_matrices = generate_payoff_matrices(n, number_of_actions, max_payoff)
    test_payoff_matrices = initial_payoff_matrices.clone()
    bounds = [[(0.0,1.0)] * num for num in number_of_actions]
    constraints = [[torch.ones(1,num), torch.ones(1), -torch.eye(num), torch.zeros(num)] for num in number_of_actions] # A, b, G, h list

    fs = [lambda x,y,agent=agent: get_individual_utility(payoff_matrices=x, strategies=y, agent=agent) for agent in range(n)]
    f_jacs = [lambda x,y,agent=agent: get_individual_derivative(payoff_matrices=x, strategies=copy.deepcopy(y), agent=agent) for agent in range(n)]
    f_hessians = [lambda x,y,agent=agent: get_individual_hessian(payoff_matrices=x, strategies=y, agent=agent) for agent in range(n)]

    initial_strategies = strategy_initialization(number_of_actions)

    # ================= initial social utility ===================
    test_strategies = list(MAS_forward(initial_payoff_matrices.clone(), fs, constraints, bounds, initial_strategies, f_jacs, eps=eps, max_iterations=max_iterations, gamma=GAMMA))
    initial_social_utility = get_social_utility(initial_payoff_matrices.clone(), test_strategies)
    # initial_strategies = copy.deepcopy(test_strategies)
    print('initial social utility: {}'.format(initial_social_utility))

    dictatorship_objective = torch.max(torch.sum(initial_payoff_matrices, dim=0)).item()
    print('dictatorship objective:', dictatorship_objective)

    # ========================== recording =======================
    all_index = pd.MultiIndex.from_product([[method], ['obj', 'violation', 'forward time', 'backward time']], names=['method', 'measure'])
    df = pd.DataFrame(np.zeros((1, all_index.size)), columns=all_index).T

    # ================= optimize social utility ==================
    if method == 'diffmulti':
        mu = 10
        lr = 1e-2
        payoff_matrices = initial_payoff_matrices.clone()
        payoff_matrices.requires_grad = True

        multipliers = torch.rand(1) # + payoff_matrices.numel())
        slack_variables = torch.zeros(1) # + payoff_matrices.numel())
        optimizer = optim.Adam([payoff_matrices, slack_variables], lr=lr)

        counter = 0
        social_utility_list, violation_list = [], []
        start_time = time.time()
        forward_time, backward_time = 0, 0
        for iteration in range(-1, number_of_iterations):
            optimizer.zero_grad()

            forward_start_time = time.time()
            presolved_strategies = list(MAS_forward(payoff_matrices.detach(), fs, constraints, bounds, initial_strategies, f_jacs, eps=eps, max_iterations=max_iterations, gamma=GAMMA))
            forward_time += time.time() - forward_start_time

            backward_start_time = time.time()
            # grads = torch.cat([get_individual_derivative_manual(payoff_matrices, presolved_strategies, agent=agent) for agent in range(n)])
            grads = torch.cat([f_jacs[agent](payoff_matrices, presolved_strategies) for agent in range(n)])
            hessians = [f_hessians[agent](payoff_matrices, presolved_strategies).detach() for agent in range(n)]

            mas_model = MAS(fs, number_of_actions, constraints, bounds, presolved_strategies, hessians)
            strategies = mas_model(*grads)

            social_utility = get_social_utility(initial_payoff_matrices.clone(), strategies)
            social_utility_list.append(social_utility.item())

            # ======================== augmented lagrangian method =========================
            if constrained:
                violation = get_violation(payoff_matrices, strategies, initial_payoff_matrices.clone(), budget) + slack_variables
                total_violation = multipliers @ violation + 0.5 * mu * torch.sum(violation * violation)
                (-social_utility + total_violation).backward()
            else:
                violation = torch.norm(payoff_matrices - initial_payoff_matrices.clone(), p=1) - budget + slack_variables
                total_violation = multipliers @ violation + 0.5 * mu * torch.sum(violation * violation)
                (-social_utility + total_violation).backward()

            # projected_gradient = project(payoff_matrices.data - payoff_matrices.grad.data - initial_payoff_matrices.data, budget) + initial_payoff_matrices.data
            # payoff_matrices.grad.data = torch.sign(payoff_matrices.grad.data)
            # payoff_matrices.grad.data = torch.clamp(payoff_matrices.grad.data, min=-1e-7, max=1e-7)
            payoff_matrices.grad.data = payoff_matrices.grad.data / torch.norm(payoff_matrices.grad.data)
            if iteration >= 0:
                optimizer.step()
            backward_time += time.time() - backward_start_time

            slack_variables.data = torch.clamp(slack_variables, min=0)
            if iteration % 100 == 0 and iteration >= 0:
                multipliers.data = multipliers.data + mu * violation
                # mu = mu + 1
            payoff_matrices.data = initial_payoff_matrices.data + torch.clamp(payoff_matrices.data - initial_payoff_matrices.data, min=0)

            if constrained:
                true_violation = get_violation(payoff_matrices, strategies, initial_payoff_matrices.clone(), budget)
                total_violation = torch.sum(torch.clamp(true_violation, min=0)).item()
            else:
                total_violation = torch.clamp(torch.norm(payoff_matrices.detach() - initial_payoff_matrices.clone(), p=1) - budget, min=0).item()

            violation_list.append(total_violation)
            print('iteration #{} with social utility: {}, violation: {}'.format(iteration, social_utility, total_violation))
            print(forward_time, backward_time)

            if initialization:
                initial_strategies = [strategy.detach() for strategy in presolved_strategies]

            # if np.max(social_utility_list) - 1e-4 > social_utility.item() and torch.abs(budget - torch.norm(initial_payoff_matrices - payoff_matrices, p=1)) < 1e-4:
            #     counter += 1
            # else:
            #     counter = 0

            # if counter >= 3:
            #     break

        optimal_payoff_matrices = payoff_matrices.detach().clone()
        best_idx = -1 # np.argmax(np.array(social_utility_list) - 1000 * np.array(violation_list))
        optimal_social_utility = social_utility_list[best_idx]
        total_violation = violation_list[best_idx]

        df.loc[('diffmulti', 'obj')] = optimal_social_utility
        df.loc[('diffmulti', 'violation')] = total_violation
        df.loc[('diffmulti', 'forward time')] = forward_time
        df.loc[('diffmulti', 'backward time')] = backward_time

        print('optimal social utility: {}, violation: {}'.format(optimal_social_utility, total_violation))
        print('forward time: {}, backward time: {}'.format(forward_time, backward_time))

    # ========================= scipy optimization ============================
    if method in ['SLSQP', 'trust', 'annealing']:
        print('Scipy optimization with blackbox constraints...')
        start_time = time.time()
        # dynamic initialization
        scipy_initial_strategies = strategy_initialization(number_of_actions)
        def get_objective(input_payoff_matrices):
            # print('function evaluating...')
            if initialization:
                global scipy_initial_strategies
            else:
                scipy_initial_strategies = strategy_initialization(number_of_actions)
            tmp_strategies = list(MAS_forward(torch.Tensor(input_payoff_matrices).reshape(n, *number_of_actions), fs, constraints, bounds, scipy_initial_strategies, f_jacs, eps=eps, max_iterations=max_iterations, gamma=GAMMA))
            scipy_initial_strategies = [strategy.detach() + strategy_perturbation * torch.randn(strategy.shape) for strategy in tmp_strategies]
            social_utility = get_social_utility(initial_payoff_matrices.clone(), tmp_strategies)
            # print(social_utility)
            if method in ['SLSQP', 'trust']:
                return -social_utility.detach().item() # maximize so adding an additional negation
            elif method in ['annealing']:
                if constrained:
                    violation = get_violation(torch.Tensor(input_payoff_matrices).reshape(n, *number_of_actions), tmp_strategies, initial_payoff_matrices.clone(), budget).detach().numpy()
                else:
                    distance = torch.norm(torch.Tensor(input_payoff_matrices).reshape(n, *number_of_actions) - initial_payoff_matrices.clone(), p=1).detach().item()
                    violation = distance - budget
                obj = -social_utility.detach().item() if violation <= 0 else np.inf
                print(social_utility, violation, sum(input_payoffs), obj)
                return obj

        if constrained:
            def get_constraint(input_payoff_matrices):
                if initialization:
                    global scipy_initial_strategies
                else:
                    scipy_initial_strategies = strategy_initialization(number_of_actions)
                tmp_strategies = list(MAS_forward(torch.Tensor(input_payoff_matrices).reshape(n, *number_of_actions), fs, constraints, bounds, scipy_initial_strategies, f_jacs, eps=eps, max_iterations=max_iterations, gamma=GAMMA))
                scipy_initial_strategies = [strategy.detach() + strategy_perturbation * torch.randn(strategy.shape) for strategy in tmp_strategies]
                violation = get_violation(torch.Tensor(input_payoff_matrices).reshape(n, *number_of_actions), tmp_strategies, initial_payoff_matrices.clone(), budget).detach().numpy()
                subsidy = - initial_payoff_matrices.clone().flatten().numpy() + input_payoff_matrices
                return np.concatenate([subsidy, [-violation]])
        else:
            def get_constraint(input_payoff_matrices):
                distance = torch.norm(torch.Tensor(input_payoff_matrices).reshape(n, *number_of_actions) - initial_payoff_matrices.clone(), p=1).detach().item()
                subsidy = - initial_payoff_matrices.clone().flatten().numpy() + input_payoff_matrices
                return np.concatenate([subsidy, [budget - distance]])

        violation_constraint = [{'type': 'ineq', 'fun': get_constraint}]

        # policy_bounds = [(x, np.inf) for x in initial_payoff_matrices.flatten().numpy()]
        if method == 'SLSQP':
            print('SLSQP optimization')
            optimization_method = 'SLSQP'
            options = {'disp': True, 'maxiter': 100, 'eps': 1e-4}
        elif method == 'trust':
            print('trust region optimization')
            optimization_method = 'trust-constr'
            options = {'verbose': 2, 'maxiter': 100}

        policy_bounds = [(x, x+budget) for x in initial_payoff_matrices.clone().flatten().numpy()]
        if method in ['SLSQP', 'trust']:
            optimization_result = scipy.optimize.minimize(get_objective, initial_payoff_matrices.clone().flatten().numpy(), method=optimization_method, constraints=violation_constraint, options=options, bounds=policy_bounds)
        elif method in ['annealing']:
            optimization_result = scipy.optimize.dual_annealing(get_objective, bounds=policy_bounds, x0=initial_payoff_matrices.clone().flatten().numpy())

        print(optimization_result)
        scipy_payoff_matrices = torch.Tensor(optimization_result['x']).reshape(n, *number_of_actions)

        scipy_strategies = list(MAS_forward(scipy_payoff_matrices, fs, constraints, bounds, scipy_initial_strategies, f_jacs, eps=eps, max_iterations=max_iterations, gamma=GAMMA))

        if constrained:
            scipy_true_violation = get_violation(scipy_payoff_matrices, scipy_strategies, initial_payoff_matrices.clone(), budget)
            scipy_total_violation = torch.sum(torch.clamp(scipy_true_violation, min=0)).item()
        else:
            scipy_total_violation = torch.clamp(torch.norm(scipy_payoff_matrices - initial_payoff_matrices.clone(), p=1) - budget, min=0).item()

        scipy_social_utility = get_social_utility(initial_payoff_matrices.clone(), scipy_strategies).item()
        print('scipy social utility: {}, violation: {}'.format(scipy_social_utility, scipy_total_violation))

        df.loc[(method, 'obj')] = scipy_social_utility
        df.loc[(method, 'violation')] = scipy_total_violation
        df.loc[(method, 'forward time')] = time.time() - start_time

    # ========================== double oracle ================================
    if method == 'DO':
        subsidy = torch.zeros_like(initial_payoff_matrices)
        subsidy_shape, subsidy_size   = initial_payoff_matrices.shape, initial_payoff_matrices.numel()
        for iteration in range(number_of_iterations):
            # best response (follower)
            perturbed_payoff_matrices = initial_payoff_matrices.clone() + subsidy
            strategies = list(MAS_forward(perturbed_payoff_matrices, fs, constraints, bounds, initial_strategies, f_jacs, eps=eps, max_iterations=max_iterations, gamma=GAMMA))

            # evaluation
            DO_true_violation = get_violation(perturbed_payoff_matrices, strategies, initial_payoff_matrices.clone(), budget)
            DO_total_violation = torch.sum(torch.clamp(DO_true_violation, min=0)).item()
            DO_social_utility = get_social_utility(initial_payoff_matrices.clone(), strategies).item()
            print('Iteration: {}, DO obj: {}, DO violation: {}'.format(iteration, DO_social_utility, DO_total_violation))
            if initialization:
                initial_strategies = [strategy.detach() for strategy in strategies]

            # best response (leader)
            def get_objective(subsidy):
                tmp_payoff_matrices = initial_payoff_matrices.clone() + torch.Tensor(subsidy).reshape(subsidy_shape)
                return -get_social_utility(tmp_payoff_matrices.clone(), strategies).item()

            DO_bounds = [(0, np.Inf)] * subsidy_size
            DO_constraints = [{'type': 'ineq', 'fun': lambda x: get_violation(initial_payoff_matrices + torch.Tensor(x).reshape(subsidy_shape), strategies, initial_payoff_matrices, budget).item() }]

            opt_result = scipy.optimize.minimize(fun=get_objective, x0=np.zeros(subsidy_size), bounds=DO_bounds, constraints=DO_constraints)
            print(opt_result)
            subsidy = torch.Tensor(opt_result['x']).reshape(subsidy_shape)


    # ========================== knitro solver ================================
    if method == 'knitro':
        start_time = time.time()

        subsidy_shape, subsidy_size = initial_payoff_matrices.shape, initial_payoff_matrices.numel()
        strategy_indices = number_of_actions
        strategy_size = sum(number_of_actions)
        start_time = time.time()

        def callbackEval(kc, cb, evalRequest, evalResult, userParams):
            if evalRequest.type != KN_RC_EVALFC:
                print("*** callbackEvalF incorrectly called with eval type %d" % evalRequest.type)
                return -1

            x = torch.Tensor(evalRequest.x)
            subsidy = torch.reshape(x[:subsidy_size], subsidy_shape) # pi
            strategies = list(torch.split(x[subsidy_size:subsidy_size+strategy_size], strategy_indices)) # x
            strategies_complement = list(torch.split(x[subsidy_size+strategy_size:subsidy_size+strategy_size*2], strategy_indices)) # y
            lamb = torch.split(x[subsidy_size+strategy_size*2:subsidy_size+strategy_size*3], strategy_indices) # lamb
            mu   = torch.split(x[subsidy_size+strategy_size*3:subsidy_size+strategy_size*4], strategy_indices) # mu
            nu   = x[-n:] # nu

            # Evaluate nonlinear objective
            obj = get_social_utility(initial_payoff_matrices.clone(), strategies).item()
            evalResult.obj = obj

            # Evaluate constraints
            cur = 0
            for agent in range(n): # lagrangian
                tmp_lagrangian = (-f_jacs[agent](initial_payoff_matrices + subsidy, strategies) - lamb[agent] + mu[agent] + nu[agent]).detach().numpy()
                for j in range(number_of_actions[agent]):
                    evalResult.c[cur+j] = tmp_lagrangian[j]
                cur += number_of_actions[agent]

            evalResult.c[strategy_size:strategy_size*2] = (torch.cat(strategies) + torch.cat(strategies_complement) - 1).flatten().detach().numpy()  # slack constraints
            for agent in range(n):
                evalResult.c[strategy_size*2+agent] = sum(strategies[agent]) - 1
            budget_violation = get_violation(initial_payoff_matrices + subsidy, strategies, initial_payoff_matrices, budget).item() # torch.sum(subsidy).item() - budget
            # budget_violation = torch.sum(subsidy).item() - budget
            evalResult.c[-1] = budget_violation

            return 0

        # Function evaluating the components of the first derivatives/gradients
        # of the objective and the constraint involved in this callback.
        def callbackEvalGA(kc, cb, evalRequest, evalResult, userParams):
            if evalRequest.type != KN_RC_EVALGA:
                print ("*** callbackEvalGA incorrectly called with eval type %d" % evalRequest.type)
                return -1

            x = torch.Tensor(evalRequest.x)
            subsidy = torch.reshape(x[:subsidy_size], subsidy_shape) # pi
            strategies = list(torch.split(x[subsidy_size:subsidy_size+strategy_size], strategy_indices)) # x
            strategies_complement = list(torch.split(x[subsidy_size+strategy_size:subsidy_size+strategy_size*2], strategy_indices)) # y
            lamb = torch.split(x[subsidy_size+strategy_size*2:subsidy_size+strategy_size*3], strategy_indices) # lamb
            mu   = torch.split(x[subsidy_size+strategy_size*3:subsidy_size+strategy_size*4], strategy_indices) # mu
            nu   = x[-n:] # nu

            # ----------------------------------
            # gradient of objective (knitro .objGrad)
            # ----------------------------------

            # gradient - pi
            # subsidy = subsidy.detach().clone()
            # subsidy.requires_grad = True

            # curr_payoff_matrices = initial_payoff_matrices.detach().clone() + subsidy
            # obj = get_social_utility(curr_payoff_matrices, strategies)
            # obj_grad = torch.autograd.grad(obj, curr_payoff_matrices)[0].detach().numpy().flatten()
            # assert subsidy_size == len(obj_grad)
            # evalResult.objGrad[:subsidy_size] = obj_grad

            # There is no gradient d obj / d subsidy
            # evalResult.objGrad[:subsidy_size] = np.zeros(subsidy_size) # obj_grad

            # gradient - x
            for agent in range(n):
                agent_deriv = get_individual_derivative(initial_payoff_matrices, strategies, agent)
                start_idx = subsidy_size + agent*number_of_actions[agent]
                end_idx = subsidy_size + (agent+1)*number_of_actions[agent]

                evalResult.objGrad[start_idx:end_idx] = agent_deriv.detach().numpy()

            # gradient of pi, y, lamb, mu, and nu are all 0


            # # ----------------------------------
            # # gradient of the constraints (NOT USING!)
            # # constraint gradient (knitro .jac)
            # # - not added, but would all be linear
            # # - derivatives would be coefficients A and G
            # # ----------------------------------
            # variable_size = subsidy_size + strategy_size * 2 + strategy_size * 2 + n
            # constraint_size = strategy_size + strategy_size + n + 1
            # assert len(evalResult.jac) == variable_size * constraint_size
            #
            # # there are strategy_size=10 lagrangian constraints
            # #           strategy_size=10 slack constraints
            # #           n+1 = 2          budget constraints
            #
            # cons_i = 0
            #
            # # [lagrangian] lagrangian constraints
            # # pi, x, y, lamb, mu, nu
            # for i in range(strategy_size):  # go through all Lagrangian constraints
            #     for agent in range(n):
            #         # other strategies
            #         other_strategies = strategies[0:agent]
            #         if agent < n:
            #             other_strategies += strategies[agent+1:n]
            #         other_strategies = torch.stack(other_strategies)
            #         print('other strategies', other_strategies.shape, other_strategies)
            #         start_idx = i*variable_size + subsidy_size # skip earlier Lagrangian constraints, pi
            #         evalResult.jac[i*variable_size + agent] = torch.sum(other_strategies[agent) #strategies[0:agent] + strategies[agent+1:n] # x_{-i}
            #
            #     for j in range(strategy_size):
            #         # x, y - in f_jacs
            #         evalResult.jac[i*variable_size + subsidy_size + j] = # sum pi_j for j != i # jac_of_x
            #         evalResult.jac[i*variable_size + subsidy_size + strategy_size] = # jac_of_y
            #
            #     cons_i += variable_size
            #
            # # [lagrangian] partial derivative of constraint w.r.t. lambda, mu, nu
            # for i in range(strategy_size):
            #     evalResult.jac[i*variable_size + subsidy_size + strategy_size*2 : i*variable_size + subsidy_size + strategy_size*3] = [-1] * strategy_size # lambda
            #     evalResult.jac[i*variable_size + subsidy_size + strategy_size*3 : i*variable_size + subsidy_size + strategy_size*4] = [1] * strategy_size # mu
            #     evalResult.jac[i*variable_size + subsidy_size + strategy_size*4 : i*variable_size + subsidy_size + strategy_size*4 + n] = [1] * n # nu
            #
            #
            # # partial derivative of slack constraints
            # # [slack] x + complement_x = 1
            # cons_i = variable_size * strategy_size
            # for i in range(strategy_size):
            #     start_idx = cons_i + i*variable_size + subsidy_size  # skip Lagrangian, earlier slack constraints, pi variables
            #     end_idx = cons_i + i*variable_size + subsidy_size + strategy_size  # skip Lagrangian, earlier slack constraints, pi, x variables
            #     evalResult.jac[start_idx:end_idx] = [1] * strategy_size # x
            #     evalResult.jac[end_idx:end_idx+strategy_size] = [-1] * strategy_size # complement x
            #
            # # partial derivative of budget constraints
            # # [budget] sum of x_i = 1 for each agent -- num constraints: strategy_size
            # # jac = 0 for pi, lambda, mu, nu
            # cons_i = variable_size * (2 * strategy_size)
            # for agent in range(n):
            #     idx = cons_i + agent*variable_size + subsidy_size  # cons_i + i_agent constraint num + pi variables
            #
            #     # other agent strategies
            #     other_strategies = strategies[0:agent]
            #     if agent < n:
            #         other_strategies += strategies[agent+1:n]
            #
            #     for action_i in range(number_of_actions[agent]):
            #         jac_sum = sum([other_strategy[action_i] for other_strategy in other_strategies])#.item()    # sum_{j != i} x_j
            #         evalResult.jac[idx+action_i] = jac_sum # x
            #         # complement x
            #         evalResult.jac[idx+strategy_size+action_i] = -jac_sum # x complement
            #
            # # [budget] subsidy sum -- num constraints: 1
            # # TODO

            return 0

        # # The signature of this function matches KN_eval_callback in knitro.py.
        # # Only "hess" or "hessVec" are set in the KN_eval_result structure.
        # def callbackEvalH(kc, cb, evalRequest, evalResult, userParams):
        #     # Hessian evaluated at “x”, “lambda”, “sigma” for EVALH or EVALH_NO_F request (hessCallback)
        #     if evalRequest.type != KN_RC_EVALH and evalRequest.type != KN_RC_EVALH_NO_F:
        #         print ("*** callbackEvalH incorrectly called with eval type %d" % evalRequest.type)
        #         return -1
        #     # x = evalRequest.x
        #     x = torch.Tensor(evalRequest.x)
        #     subsidy = torch.reshape(x[:subsidy_size], subsidy_shape) # pi
        #     strategies = list(torch.split(x[subsidy_size:subsidy_size+strategy_size], strategy_indices)) # x
        #     strategies_complement = list(torch.split(x[subsidy_size+strategy_size:subsidy_size+strategy_size*2], strategy_indices)) # y
        #     lamb = torch.split(x[subsidy_size+strategy_size*2:subsidy_size+strategy_size*3], strategy_indices) # lamb
        #     mu   = torch.split(x[subsidy_size+strategy_size*3:subsidy_size+strategy_size*4], strategy_indices) # mu
        #     nu   = x[-n:] # nu
        #
        #     lambda_ = evalRequest.lambda_
        #     # Scale objective component of hessian by sigma
        #     sigma = evalRequest.sigma
        #     print('lambda', len(lambda_), lambda_) # len = num variables + num constraints
        #     print('sigma', sigma) # 1
        #     print('hess', len(evalResult.hess))  # lenth 1326
        #
        #     # 2nd order derivative of agent's utility function w.r.t. agent's action and then with respect to all other agents' actions
        #     for agent in range(n):
        #         hess = get_individual_hessian(initial_payoff_matrices.clone(), strategies, agent)
        #         print('agent', agent, hess.shape, hess) # shape: strategies x strategies
        #
        #     # 2nd order derivative of agent's utility function w.r.t. agent's action and then with respect to input x
        #     for agent in range(n):
        #         daction = get_individual_daction_dx(initial_payoff_matrices.clone(), strategies, agent)
        #         print('agent', agent, daction.shape, daction) # shape:
        #
        #     hessian = sigma * hess_obj # QUESTION: there's a lambda for every variable and constraints? why not just constraints?
        #     for c in range(num_constraints):
        #         hess_constraint
        #         hessian += lambda_[c] * hess_constraint
        #
        #     return 0

        variable_size = subsidy_size + strategy_size * 2 + strategy_size * 2 + n # pi, x, y, lamb, mu, nu respectively
        variables = Variables(nV=variable_size,
                xLoBnds=[0]*(subsidy_size + strategy_size * 2 + strategy_size * 2) + [-KN_INFINITY] * n,
                xUpBnds=[budget]*(subsidy_size) + [1] * (strategy_size * 2) + [KN_INFINITY] * (strategy_size * 2 + n),
                # xInitVals=[0]*(subsidy_size) + [0] * strategy_size + [1] * strategy_size + [1] * strategy_size + [0] * strategy_size + [0] * n, # set initial value
                )

        constraint_size = strategy_size + strategy_size + (n + 1) # lagrangian, slack, budget constraints
        eq_eps = 1e-2# epsilon allowance for the equality constraint
        knitro_constraints = Constraints(nC=constraint_size,
                # cEqBnds=[0] * constraint_size, # equality constraint
                cLoBnds=[-eq_eps] * (strategy_size * 2) + [-eq_eps] * n + [-KN_INFINITY], # inequality constraint
                cUpBnds=[eq_eps] * (strategy_size * 2) + [eq_eps] * n + [eq_eps],
                cNames=['eqCon{}'.format(i) for i in range(constraint_size)])

        print('variable size', variable_size, 'constraint size', constraint_size)

        complementarity = ComplementarityConstraints(
                indexComps1=list(range(subsidy_size, subsidy_size+strategy_size*2)), # x, y
                indexComps2=list(range(subsidy_size+strategy_size*2, subsidy_size+strategy_size*4)), # lamb, mu
                ccTypes=[KN_CCTYPE_VARVAR]*(strategy_size*2),
                cNames=['compCon{}'.format(i) for i in range(strategy_size*2)]
                )

        print('complementarity', complementarity)

        # https://www.artelys.com/docs/knitro/2_userGuide/callbacks.html
        # .obj .c .objGrad .jac .hess .hessVec .rsd .rsdJac
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
                    'ms_maxsolves': 20,
                    'bar_feasible': 'get',     # get, stay - prioritze finding feasible point
                    'algorithm': 'direct',         # direct, cg, active, sqp, multi # which algorithm to use
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


        print('subsidy', 'len', len(knitro_result.x[:subsidy_size]), np.round(knitro_result.x[:subsidy_size], 4))
        print('subsidy sum {:.5f}'.format(np.sum(knitro_result.x[:subsidy_size])))


        # Processing result
        knitro_perturbation = torch.reshape(torch.Tensor(knitro_result.x[:subsidy_size]), subsidy_shape) # pi
        print('knitro used budget:', torch.sum(knitro_perturbation), 'real budget:', budget)
        # if torch.sum(knitro_perturbation) > budget: # rounding
        #     knitro_perturbation = knitro_perturbation * budget / torch.sum(knitro_perturbation)

        perturbed_payoff_matrices = initial_payoff_matrices.clone() + knitro_perturbation

        knitro_strategies = knitro_result.x[subsidy_size:subsidy_size+strategy_size] # x
        print('knitro strategies', np.round(knitro_strategies, 3))


        initial_strategies = strategy_initialization(number_of_actions)
        strategies = list(MAS_forward(perturbed_payoff_matrices, fs, constraints, bounds, initial_strategies, f_jacs, eps=eps, max_iterations=max_iterations, gamma=GAMMA))

        knitro_true_violation = get_violation(perturbed_payoff_matrices, strategies, initial_payoff_matrices.clone(), budget)
        knitro_total_violation = torch.sum(torch.clamp(knitro_true_violation, min=0)).item()
        knitro_social_utility = get_social_utility(initial_payoff_matrices.clone(), strategies).item()

        print('amount of true violation', knitro_true_violation)
        print('amount of total violation', knitro_total_violation)
        print('amount of social utility', knitro_social_utility)

        knitro_runtime = time.time() - start_time
        print('knitro runtime', knitro_runtime)

        # Recording result
        df.loc[(method, 'obj')] = knitro_social_utility
        df.loc[(method, 'violation')] = knitro_total_violation
        # df.loc[(method, 'forward time')] = knitro_result['TimeReal']
        df.loc[(method, 'forward time')] = knitro_runtime

    # ========================== random policy ================================
    start_time = time.time()
    random_perturbation = torch.rand(initial_payoff_matrices.shape)
    random_perturbation = random_perturbation / torch.sum(random_perturbation) * budget
    perturbed_payoff_matrices = initial_payoff_matrices.clone() + random_perturbation

    initial_strategies = strategy_initialization(number_of_actions)
    strategies = list(MAS_forward(perturbed_payoff_matrices, fs, constraints, bounds, initial_strategies, f_jacs, eps=eps, max_iterations=max_iterations, gamma=GAMMA))

    if constrained:
        random_true_violation = get_violation(perturbed_payoff_matrices, strategies, initial_payoff_matrices.clone(), budget)
        random_total_violation = torch.sum(torch.clamp(random_true_violation, min=0)).item()
    else:
        random_total_violation = torch.clamp(torch.norm(perturbed_payoff_matrices - initial_payoff_matrices.clone(), p=1) - budget, min=0).item()

    random_social_utility = get_social_utility(initial_payoff_matrices.clone(), strategies).item()
    print('random social utility: {}, violation: {}'.format(random_social_utility, random_total_violation))

    f = open(filename, 'a')
    f.write('{}, {}, {}, {}, {}, {},'.format(seed, budget, initial_social_utility, dictatorship_objective, random_social_utility, random_total_violation) +
            '{}\n'.format(', '.join([str(x) for x in df.values[:,0].tolist()]))
        )

    f.close()
    print('total runtime:', time.time() - start_time)

    # ================= verifying backward pass ==================
    # strategies = list(mas_model(payoff_matrices))
    # dl_dx = torch.autograd.grad(strategies[0][0], payoff_matrices)
    # print(dl_dx)

    # ================ verifying the correctness =================
    # print('verifying correctness...')
    # print('strategies:', strategies)
    # print('jac:', f_jacs[0](payoff_matrices, strategies))
    # print('manual jac:', f_jacs_manual[0](payoff_matrices, strategies))
    # for agent in range(n):
    #     print('hessian:', f_hessians[agent](payoff_matrices, strategies))
    #     print('payoff matrix:', payoff_matrices[agent])

    # social_utility = get_social_utility(payoff_matrices, strategies)
    # print('social utility:', social_utility)
