import numpy as np
import torch
import torch.nn as nn
# import qpth
import scipy
import copy
import time
import argparse
import pandas as pd
from knitro import *

import torch.optim as optim

import sys
sys.path.insert(1, '../')
from model import MAS, MAS_forward
from utils import project

lamb = 1
lamb_L = 0.01
strategy_perturbation = 0.001

activation = nn.Sigmoid()
# activation = lambda x: torch.clamp(torch.exp(x), min=0, max=1)
# activation = nn.Softmax(dim=0)
# risk_aversion = 1

def generate_parameters(n):
    W = torch.rand(n,n) + torch.eye(n) * 1
    L = torch.rand(n) * 50 + 50
    c = torch.rand(n) * 5 + 5
    return (W, L, c)

def generate_policies(n):
    # n: number of players
    I = torch.zeros(n,1)
    rho = torch.zeros(n,1)
    policies = torch.cat([I, rho], dim=1)
    return policies

def strategy_initialization(n):
    x = torch.rand(n,1)
    # d = torch.rand(n,1)
    # strategies = torch.cat([x, d], dim=1)
    return x

def get_social_utility(policies, strategies, parameters):
    n = len(strategies)
    I, rho = policies[:,0], policies[:,1]
    W, L, c = parameters
    x = torch.cat([strategies[i][0].view(1) for i in range(n)])
    q = activation(- lamb * W @ x + lamb_L * L)
    # revenue = torch.sum(- L * q - c * x)
    revenue = torch.sum(- I * q + rho)
    return revenue

def get_individual_utility(policies, strategies, parameters, risk_aversion, agent):
    # parameters: D, L, c
    n = len(strategies)
    I, rho = policies[:,0], policies[:,1]
    W, L, c = parameters
    x = torch.cat([strategies[i][0].view(1) for i in range(n)])
    q = activation(- lamb * W @ x + lamb_L * L)
    payoff = - (L[agent] - I[agent]) * q[agent] - c[agent] * x[agent] - rho[agent] - risk_aversion * torch.abs(L[agent] - I[agent]) * torch.sqrt(q[agent] * (1 -q[agent]))
    return payoff # + risk

def get_individual_derivative(policies, strategies, parameters, risk_aversion, agent, retain_graph=True, create_graph=True):
    x = torch.autograd.Variable(strategies[agent], requires_grad=True)
    strategies[agent] = x
    payoff = get_individual_utility(policies, strategies, parameters, risk_aversion, agent)
    jac = torch.autograd.grad(payoff, x, retain_graph=True, create_graph=True)[0]
    return jac

# def get_individual_derivative_manual(policies, strategies, parameters, agent):
#     # parameters: D, L, c
#     n = len(strategies)
#     I, rho = policies[:,0], policies[:,1]
#     W, L, c = parameters
#     x = torch.cat([strategies[i][0].view(1) for i in range(n)])
#     # d = torch.cat([strategies[i][1].view(1) for i in range(n)])
#     q = activation(- lamb * W @ x + lamb_L * L)
#
#     # ------------ varaince version -----------
#     jac_x = (L[agent] - I[agent] * L[agent]) * lamb * W[agent,agent] * q[agent] - c[agent]
#     # risk_jac_x = - risk_aversion * (1 - d[agent] * I[agent]) * L[agent] * (1 - 2 * q[agent]) * q[agent] * W[agent,agent]
#     # jac_d = I[agent] * L[agent] * q[agent] - rho[agent]
#     # risk_jac_d = - risk_aversion * (- I[agent] * L[agent]) * q[agent] * (1 - q[agent])
#     # ------------ penalty version ------------
#     # payoff = - (L[agent] - d[agent] * I[agent] * L[agent]) * q[agent] - c[agent] * x[agent] - rho[agent] * d[agent]
#     # jac_x = (L[agent] - d[agent] * I[agent] * L[agent]) * lamb * W[agent,agent] * q[agent] - c[agent]
#     # jac_d = I[agent] * L[agent] * q[agent] - rho[agent] + risk_aversion * (1 - 2 * d[agent]) / (d[agent] * (1 - d[agent]))
#     # jac = torch.cat([(jac_x + risk_jac_x).view(1), (jac_d + risk_jac_d).view(1)])
#     jac = jac_x.view(1)
#     return jac

def get_individual_hessian(policies, strategies, parameters, risk_aversion, agent):
    # the second order derivative of agent's utility function with respect to agent's action and then with respect to all other agents' actions
    xs = [torch.autograd.Variable(strategy, requires_grad=True) for strategy in strategies]
    # jac = get_individual_derivative_manual(policies, xs, parameters, agent)
    jac = get_individual_derivative(policies, xs, parameters, risk_aversion, agent, retain_graph=True, create_graph=True)
    hessian = torch.cat([torch.cat([torch.autograd.grad(jac[i], x, retain_graph=True, create_graph=True)[0] for x in xs]).view(1,-1) for i in range(len(jac))])
    return hessian

def get_violation(policies, strategies, parameters, risk_aversion):
    n = len(strategies)
    I, rho = policies[:,0], policies[:,1]
    W, L, c = parameters
    x = torch.cat([strategies[i][0].view(1) for i in range(n)])
    q = activation(- lamb * W @ x + lamb_L * L)
    payoff_with_insurance = - (L - I) * q - c * x - rho - risk_aversion * torch.abs(L - I) * torch.sqrt(q * (1 -q))
    payoff_without_insurance = - L * q - c * x - risk_aversion * L * torch.sqrt(q * (1 - q))
    # violation = torch.cat([payoff_without_insurance - payoff_with_insurance, payment.view(1)])
    violation = payoff_without_insurance - payoff_with_insurance
    # We want the payoff with insurance is higher than the payoff without insurance, i.e., payoff_without_insurance <= payoff_with_insurance
    # We want the insurance company to have a non-negative net profit, i.e., payment <= 0
    return violation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cyber experiment')

    parser.add_argument('--n', type=int, default=10, help='number of agents')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--risk', type=float, default=0.1, help='risk')
    parser.add_argument('--constrained', default=True, action='store_true', help='Adding incentive compatible constraints')
    # parser.add_argument('--no-constrained', dest='constrained', action='store_false', help='Adding incentive compatible constraints')
    parser.add_argument('--method', type=str, default='', help='optimization method (diffmulti, SLSQP, trust)')
    parser.add_argument('--forward-iterations', type=int, default=100, help='maximum iterations the equilibrium finding oracle uses')
    parser.add_argument('--gradient-iterations', type=int, default=5000, help='maximum iterations of gradient descent')
    parser.add_argument('--prefix', type=str, default='', help='prefix of the filename')
    parser.add_argument('--init', default=False, action='store_true', help='Using dynamic initialization')

    args = parser.parse_args()

    n = args.n
    seed = args.seed
    constrained = args.constrained
    risk_aversion = args.risk
    method = args.method
    max_iterations = args.forward_iterations
    number_of_iterations = args.gradient_iterations
    prefix = args.prefix
    initialization = args.init
    assert(method in ['diffmulti', 'SLSQP', 'trust', 'knitro'])
    assert(constrained == True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    action_dimensions = [1] * n
    GAMMA = 0.5
    eps = 1e-4

    print('Constrained:', constrained)
    if constrained:
        filename = 'exp/' + prefix + 'constrained_{}.csv'.format(method)
    else:
        filename = 'exp/' + prefix + 'unconstrained_{}.csv'.format(method)

    print('Risk aversion {}, random seed {}...'.format(risk_aversion, seed))

    # ================= parameter generation ======================
    parameters = generate_parameters(n)
    initial_policies = generate_policies(n)

    bounds = [[(0.0,np.inf)] for _ in range(n)]
    constraints = [[torch.Tensor(), torch.Tensor(), torch.cat([-torch.eye(1)]), torch.cat([torch.zeros(1)])] for _ in range(n)] # A, b, G, h list

    # ========================== recording =======================
    all_index = pd.MultiIndex.from_product([[method], ['obj', 'violation', 'forward time', 'backward time']], names=['method', 'measure'])
    df = pd.DataFrame(np.zeros((1, all_index.size)), columns=all_index).T

    # ================= optimize social utility ==================
    mu = 100
    lr = 1e-1
    policies = initial_policies.clone()
    policies.requires_grad = True

    # =================== function definition =====================
    fs = [lambda x,y,agent=agent: get_individual_utility(policies=x, strategies=y, parameters=parameters, risk_aversion=risk_aversion, agent=agent) for agent in range(n)]
    f_jacs = [lambda x,y,agent=agent: get_individual_derivative(policies=x, strategies=copy.deepcopy(y), parameters=parameters, risk_aversion=risk_aversion, agent=agent, retain_graph=True, create_graph=True) for agent in range(n)]
    # f_jacs_manual = [lambda x,y,agent=agent: get_individual_derivative_manual(policies=x, strategies=copy.deepcopy(y), parameters=parameters, agent=agent) for agent in range(n)]
    f_hessians = [lambda x,y,agent=agent: get_individual_hessian(policies=x, strategies=y, parameters=parameters, risk_aversion=risk_aversion, agent=agent) for agent in range(n)]

    initial_strategies = strategy_initialization(n)
    print('autograd:', f_jacs[0](initial_policies, initial_strategies))
    # print('manual grad:', f_jacs_manual[0](initial_policies, initial_strategies))

    # ================= initial social utility ===================
    strategies = list(MAS_forward(initial_policies, fs, constraints, bounds, initial_strategies, f_jacs=f_jacs, eps=eps, max_iterations=max_iterations, gamma=GAMMA))
    initial_social_utility = get_social_utility(initial_policies, strategies, parameters)
    # initial_strategies = strategies
    print('initial social utility: {}'.format(initial_social_utility))

    # ================= optimizer and variables ==================
    if method == 'diffmulti':
        multipliers = torch.rand(n)
        slack_variables = torch.zeros(n)
        optimizer = optim.Adam([policies, slack_variables], lr=lr)

        counter = 0
        social_utility_list, violation_list = [], []
        forward_time, backward_time = 0, 0
        for iteration in range(-1, number_of_iterations):
            start_time = time.time()
            optimizer.zero_grad()

            presolved_strategies = list(MAS_forward(policies.detach(), fs, constraints, bounds, initial_strategies, f_jacs=f_jacs, eps=eps, max_iterations=max_iterations, gamma=GAMMA))

            forward_time += time.time() - start_time
            start_time = time.time()

            grads = torch.cat([f_jacs[agent](policies, presolved_strategies) for agent in range(n)])
            hessians = [f_hessians[agent](policies, presolved_strategies).detach() for agent in range(n)]

            mas_model = MAS(fs, action_dimensions, constraints, bounds, presolved_strategies, hessians)
            strategies = mas_model(*grads)

            social_utility = get_social_utility(policies, strategies, parameters)
            social_utility_list.append(social_utility.item())

            # ===================== augmented lagrangian method ===================
            if constrained:
                violation = get_violation(policies, strategies, parameters, risk_aversion) + slack_variables
                total_violation = multipliers @ violation + 0.5 * mu * torch.sum(violation * violation)
                (-social_utility + total_violation).backward()
            else:
                (-social_utility).backward()

            backward_time += time.time() - start_time

            # ======================== back-propagation ===========================
            if iteration >= 0:
                optimizer.step()

            if constrained:
                slack_variables.data = torch.clamp(slack_variables, min=0)
                if iteration % 100 == 0:
                    multipliers.data = multipliers.data + mu * violation

            #     policies.data[:,0] = torch.min(torch.clamp(policies.data[:,0], min=0), parameters[1])
            # else:
            #     policies.data[:,0] = torch.min(torch.clamp(policies.data[:,0], min=0), parameters[1])

            true_violation = get_violation(policies, strategies, parameters, risk_aversion)
            total_violation = torch.sum(torch.clamp(true_violation, min=0)).item()
            violation_list.append(total_violation)
            print('iteration:', iteration, 'social utility:', social_utility.item(), 'total violation:', total_violation)

            if initialization:
                initial_strategies = [strategy.detach() for strategy in presolved_strategies]

        best_idx = -1 # np.argmax(np.array(social_utility_list) - 1000 * np.array(violation_list))
        optimal_social_utility = social_utility_list[best_idx]
        total_violation = violation_list[best_idx]

        df.loc[('diffmulti', 'obj')] = optimal_social_utility
        df.loc[('diffmulti', 'violation')] = total_violation
        df.loc[('diffmulti', 'forward time')] = forward_time
        df.loc[('diffmulti', 'backward time')] = backward_time

        print('optimal social utility: {}, violation: {}'.format(optimal_social_utility, total_violation))

        print('policy', policies)
        print('strategies', strategies)
        print('slack_variables', slack_variables)

    # ========================= scipy optimization ============================
    if method in ['SLSQP', 'trust']:
        print('Scipy optimization with blackbox constraints...')
        start_time = time.time()
        policies = initial_policies.clone()
        initial_strategies = strategy_initialization(n)
        def get_objective(input_policies):
            if initialization:
                global initial_strategies
            else:
                initial_strategies = strategy_initialization(n)
            strategies = list(MAS_forward(torch.Tensor(input_policies).reshape(-1,2), fs, constraints, bounds, initial_strategies, f_jacs=f_jacs, eps=eps, max_iterations=max_iterations, gamma=GAMMA))
            social_utility = get_social_utility(torch.Tensor(input_policies).reshape(-1,2), strategies, parameters)
            initial_strategies = [strategy.detach() + strategy_perturbation * torch.randn(strategy.shape) for strategy in strategies]
            return -social_utility.detach().item() # maximize so adding an additional negation

        def get_constraint(input_policies):
            if initialization:
                global initial_strategies
            else:
                initial_strategies = strategy_initialization(n)
            strategies = list(MAS_forward(torch.Tensor(input_policies).reshape(-1,2), fs, constraints, bounds, initial_strategies, f_jacs=f_jacs, eps=eps, max_iterations=max_iterations, gamma=GAMMA))
            initial_strategies = [strategy.detach() + strategy_perturbation * torch.randn(strategy.shape) for strategy in strategies]
            violation = get_violation(torch.Tensor(input_policies).reshape(-1,2), strategies, parameters, risk_aversion).detach().numpy()
            return - violation

        violation_constraint = [{'type': 'ineq', 'fun': get_constraint}]
        if method == 'SLSQP':
            print('SLSQP optimization')
            optimization_method = 'SLSQP'
            options = {'disp': True, 'maxiter': 100, 'eps': 1e-2}
        elif method == 'trust':
            optimization_method = 'trust-constr'
            options = {'verbose': 2, 'maxiter': 100}

        optimization_result = scipy.optimize.minimize(get_objective, initial_policies.flatten().numpy(), method=optimization_method, constraints=violation_constraint, options=options)

        print(optimization_result)
        scipy_policies = torch.Tensor(optimization_result['x']).reshape(-1,2)

        initial_strategies = strategy_initialization(n)
        scipy_strategies = list(MAS_forward(scipy_policies, fs, constraints, bounds, initial_strategies, f_jacs=f_jacs, eps=eps, max_iterations=max_iterations, gamma=GAMMA))

        scipy_social_utility = get_social_utility(scipy_policies, scipy_strategies, parameters).item()

        scipy_true_violation = get_violation(scipy_policies, scipy_strategies, parameters, risk_aversion)
        scipy_total_violation = torch.sum(torch.clamp(scipy_true_violation, min=0)).item()
        print('scipy social utility: {}, violation: {}'.format(scipy_social_utility, scipy_total_violation))

        df.loc[(method, 'obj')] = scipy_social_utility
        df.loc[(method, 'violation')] = scipy_total_violation
        df.loc[(method, 'forward time')] = time.time() - start_time


    # ========================== knitro solver ================================
    if method == 'knitro':
        start_time = time.time()

        policy_shape, policy_size = initial_policies.shape, initial_policies.numel()
        strategy_size = n

        def callbackEval(kc, cb, evalRequest, evalResult, userParams):
            if evalRequest.type != KN_RC_EVALFC:
                print("*** callbackEvalF incorrectly called with eval type %d" % evalRequest.type)
                return -1

            assert len(evalRequest.x) == policy_size + strategy_size*2
            assert len(evalResult.c) == strategy_size*2

            x = torch.Tensor(evalRequest.x)

            policies   = torch.reshape(x[:policy_size], policy_shape) # pi
            strategies = torch.autograd.Variable(x[policy_size:policy_size+strategy_size].view(-1,1), requires_grad=True) # x
            lamb       = x[policy_size+strategy_size:policy_size+strategy_size*2] # lamb

            # budget_violation = get_violation(policies, strategies, parameters, risk_aversion)
            # total_violation = torch.sum(torch.clamp(budget_violation, min=0)).item()
            #
            # # discard policies with very large constraint violation
            # if total_violation > 100:
            #     # print('discard')
            #     discard += 1
            #     evalRequest.x[:policy_size] = [0] * policy_size
            #     policies = torch.reshape(x[:policy_size], policy_shape) # pi
            # else:
            #     no_discard += 1

            # Evaluate nonlinear objective
            obj = get_social_utility(policies, strategies, parameters)
            strategy_gradient = torch.autograd.grad(obj, strategies)[0].detach().flatten().numpy()
            evalResult.obj = obj.item()

            # Evaluate constraints
            c_lagrangian = - strategy_gradient - lamb.detach().numpy()
            evalResult.c[:strategy_size] = c_lagrangian# lagrangian

            budget_violation = get_violation(policies, strategies, parameters, risk_aversion) # NOTE: this must be an inequality (and not equality) constraint
            evalResult.c[strategy_size:] = budget_violation.detach().numpy() # individual rationality?
            # total_violation = torch.sum(torch.clamp(budget_violation, min=0)).item()
            # print(total_violation)

            return 0

        # Function evaluating the components of the first derivatives/gradients
        # of the objective and the constraint involved in this callback.
        def callbackEvalGA(kc, cb, evalRequest, evalResult, userParams):
            if evalRequest.type != KN_RC_EVALGA:
                print ("*** callbackEvalGA incorrectly called with eval type %d" % evalRequest.type)
                return -1

            x = torch.Tensor(evalRequest.x)
            assert len(x) == policy_size + strategy_size*2
            policies   = torch.autograd.Variable(torch.reshape(x[:policy_size], policy_shape), requires_grad=True) # pi
            strategies = torch.autograd.Variable(x[policy_size:policy_size+strategy_size].view(-1,1), requires_grad=True) # x
            lamb       = x[policy_size+strategy_size:policy_size+strategy_size*2] # lamb

            # Evaluate nonlinear objective
            obj = get_social_utility(policies, strategies, parameters)
            policy_gradient   = torch.autograd.grad(obj, policies, retain_graph=True, create_graph=True)[0] # pi gradient
            strategy_gradient = torch.autograd.grad(obj, strategies, retain_graph=True, create_graph=True)[0]

            evalResult.objGrad[:policy_size] = policy_gradient.detach().flatten().numpy() # pi
            evalResult.objGrad[policy_size:policy_size+strategy_size] = strategy_gradient.detach().flatten().numpy() # x
            # evalResult.objGrad[policy_size+strategy_size:] = np.zeros(strategy_size) # lamb

            budget_violation = get_violation(policies, strategies, parameters, risk_aversion) # NOTE: this must be an inequality (and not equality) constraint
            constraint_values = torch.cat([-strategy_gradient.flatten() - lamb, budget_violation])
            gradients = []
            for i in range(strategy_size*2):
                grad1 = torch.autograd.grad(constraint_values[i], policies, retain_graph=True)[0].flatten()
                grad2 = torch.autograd.grad(constraint_values[i], strategies, retain_graph=True)[0].flatten()
                gradients.append(torch.cat([grad1, grad2]).detach())

            evalResult.jac[:] = torch.cat(gradients).flatten().numpy()

            return 0

        initial_strategies = MAS_forward(initial_policies, fs, constraints, bounds, initial_strategies, f_jacs, eps=eps, max_iterations=max_iterations, gamma=GAMMA)
        variable_size = policy_size + strategy_size + strategy_size # pi, x, lamb
        variables = Variables(nV=variable_size,
                xLoBnds=[0]*(policy_size + strategy_size + strategy_size),
                xInitVals=torch.cat([initial_policies.flatten(), initial_strategies.flatten(), torch.zeros(strategy_size)]).flatten().detach().numpy()
                # xInitVals=[0]*(policy_size) + [0] * strategy_size + [0] * strategy_size,
                )

        constraint_size = strategy_size + strategy_size # lagrangian, individual rationality constraints
        eq_eps = 1e-2 # epsilon allowance for the equality constraint
        knitro_constraints = Constraints(nC=constraint_size,
                # cEqBnds = [list(range(0, strategy_size)), [0]*strategy_size], # equality for lagrangian
                # cLoBnds = [list(range(strategy_size, strategy_size*2)), [-KN_INFINITY] * strategy_size], # inequality for individual rationality
                # cUpBnds = [list(range(strategy_size, strategy_size*2)), [0] * strategy_size],
                # cEqBnds=[0] * constraint_size, # equality constraint
                cLoBnds=[-eq_eps] * strategy_size + [-KN_INFINITY] * strategy_size, # inequality constraint
                cUpBnds=[eq_eps] * strategy_size + [0] * strategy_size,
                cType=[KN_CONTYPE_GENERAL]*constraint_size, #KN_CONTYPE_GENERAL, , KN_CONTYPE_QUADRATIC
                cNames=['eqCon{}'.format(i) for i in range(constraint_size)])

        print('variable size', variable_size, 'constraint size', constraint_size)

        complementarity = ComplementarityConstraints(
                indexComps1=list(range(policy_size, policy_size+strategy_size)), # x
                indexComps2=list(range(policy_size+strategy_size, policy_size+strategy_size*2)), # lamb
                ccTypes=[KN_CCTYPE_VARVAR]*(strategy_size),
                cNames=['compCon{}'.format(i) for i in range(strategy_size)]
                )

        print('complementarity', complementarity)

        # https://www.artelys.com/docs/knitro/2_userGuide/callbacks.html
        # .obj .c .objGrad .jac .hess .hessVec .rsd .rsdJac
        callback = Callback(
            # evalObj=True,
            funcCallback=callbackEval,
            # evalFCGA=False, # If evalFCGA is equal to True, the callback should also evaluate the relevant first derivatives/gradients.

            # evaluate the components of the first derivatives/gradients of the objective
            gradCallback=callbackEvalGA,
            objGradIndexVars=KN_DENSE, # KN_DENSE for all variables
            jacIndexCons=KN_DENSE_ROWMAJOR, # if evaluating jacobian of constraints
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
                    # 'ms_maxsolves': 50, #200,
                    # 'ms_maxbndrange': 100, # maximum range that an unbounded variable can vary over when multistart computes new start points
                    # 'ms_startptrange': 100, # #maximum range that any variable can vary over when multistart computes new start points.
                    # 'opttol': 1e-03, # default 1e-06
                    # 'bar_feasible': 'get_stay',     # stay, get, get_stay - prioritze finding feasible point

                    # Specifies the final absolute stopping tolerance for the feasibility error.
                    # Smaller values of feastol_abs result in a higher degree of accuracy in the
                    # solution with respect to feasibility.
                    # 'feastol_abs': 10, # default 1e-03

                    # 'algorithm': 'multi',         # direct, cg, active, sqp, multi # which algorithm to use
                    # 'act_lpsolver': 'cplex',      # use licensed LP solver in active or sqp mode
                    # 'act_qppenalty': 'all',       # Constraint penalization for QP subproblems.
                    # 'bar_initpt': 'convex',       # convex, nearbnd, central - selecting first initial point (if not specified)
                    # 'bar_penaltycons': 'all',     # Whether or not to penalize constraints in the barrier algorithms.
                    # 'debug': 'execution', #'problem', 'debug',  # problem, execution - output debugging
                    # 'derivcheck': 'all',          # Whether to perform a derivative check on the model
                    # 'honorbnds': 'always',        # auto, no, always, initpt - Whether to enforce satisfaction of simple bounds at all iterations
                    # 'mip_branchrule': 'most_frac', # Specifies the MIP branching rule for choosing a variable.
                    }
                )


        print('policy', np.round(knitro_result.x[:policy_size], 3))
        print('strategies', np.round(knitro_result.x[policy_size:policy_size+strategy_size], 3))
        print('lambda', np.round(knitro_result.x[-strategy_size:], 3))

        # Processing result
        knitro_policy = torch.reshape(torch.Tensor(knitro_result.x[:policy_size]), policy_shape) # pi
        # if torch.sum(knitro_perturbation) > budget: # rounding
        #     knitro_perturbation = knitro_perturbation * budget / torch.sum(knitro_perturbation)

        initial_strategies = strategy_initialization(n)
        strategies = list(MAS_forward(knitro_policy, fs, constraints, bounds, initial_strategies, f_jacs, eps=eps, max_iterations=max_iterations, gamma=GAMMA))

        knitro_true_violation = get_violation(knitro_policy, strategies, parameters, risk_aversion)
        knitro_total_violation = torch.sum(torch.clamp(knitro_true_violation, min=0)).item()
        knitro_social_utility = get_social_utility(knitro_policy, strategies, parameters).item()

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
    perturbed_policies = initial_policies + torch.rand(initial_policies.shape)

    strategies = list(MAS_forward(perturbed_policies, fs, constraints, bounds, initial_strategies, f_jacs=f_jacs, eps=eps, max_iterations=max_iterations, gamma=GAMMA))

    random_true_violation = get_violation(perturbed_policies, strategies, parameters, risk_aversion)
    random_total_violation = torch.sum(torch.clamp(random_true_violation, min=0)).item()

    random_social_utility = get_social_utility(perturbed_policies, strategies, parameters).item()
    print('random social utility: {}, violation: {}'.format(random_social_utility, random_total_violation))

    f = open(filename, 'a')
    f.write('{}, {}, {}, {}, {},'.format(seed, risk_aversion, initial_social_utility, random_social_utility, random_total_violation) +
            '{}\n'.format(', '.join([str(x) for x in df.values[:,0].tolist()]))
        )

    f.close()
    print('total runtime:', time.time() - start_time)

    # ================= verifying backward pass ==================
    # strategies = list(mas_model(policies))
    # dl_dx = torch.autograd.grad(strategies[0][0], policies)
    # print(dl_dx)

    # ================ verifying the correctness =================
    # print('verifying correctness...')
    # print('strategies:', strategies)
    # print('jac:', f_jacs[0](policies, strategies))
    # print('manual jac:', f_jacs_manual[0](policies, strategies))
    # for agent in range(n):
    #     print('hessian:', f_hessians[agent](policies, strategies))
    #     print('payoff matrix:', policies[agent])

    # social_utility = get_social_utility(policies, strategies)
    # print('social utility:', social_utility)
