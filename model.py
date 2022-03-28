import copy
import scipy.optimize
import torch
import torch.nn as nn
import cvxpy as cp

EPSILON = 1e-6

def best_response(agent, x, strategies, fs, f_jacs, constraints, bounds):
    def f(y):
        tmp_strategies = copy.deepcopy(strategies)
        tmp_strategies[agent] = torch.Tensor(y)
        return -fs[agent](x,tmp_strategies).detach().numpy()
        # since we assume agents maximize their objective while scipy only supports minimization, so we have to take a negation here.

    if f_jacs is not None:
        def f_jac(y):
            tmp_strategies = copy.deepcopy(strategies)
            tmp_strategies[agent] = torch.Tensor(y)
            return -f_jacs[agent](x,tmp_strategies).detach().numpy()

    initial_point = strategies[agent].detach().numpy()
    options = {'eps': 1e-5}
    A_torch, b_torch, G_torch, h_torch = constraints[agent]
    A, b, G, h = A_torch.detach().numpy(), b_torch.detach().numpy(), G_torch.detach().numpy(), h_torch.detach().numpy()
    if A.shape[0] == 0 and b.shape[0] == 0 and G.shape[0] == 0 and h.shape[0] == 0:
        agent_constraints = []
    elif A.shape[0] == 0 and b.shape[0] == 0:
        agent_constraints = [{'type': 'ineq', 'fun': lambda x: h - G @ x}]
    elif G.shape[0] == 0 and h.shape[0] == 0:
        agent_constraints = [{'type': 'eq', 'fun': lambda x: b - A @ x}]
    else:
        agent_constraints = [{'type': 'eq', 'fun': lambda x: b - A @ x}, {'type': 'ineq', 'fun': lambda x: h - G @ x}]

    if f_jacs is not None:
        optimization_result = scipy.optimize.minimize(f, initial_point, method='SLSQP', jac=f_jac, constraints=agent_constraints, bounds=bounds[agent], options=options)
    else:
        optimization_result = scipy.optimize.minimize(f, initial_point, method='SLSQP', constraints=agent_constraints, bounds=bounds[agent], options=options)

    return torch.Tensor(optimization_result['x'])

def norm(strategies, new_strategies): # norm of the strategy space. This is used to determine the stopping criteria only
    return sum([torch.norm(strategy - new_strategy) for strategy, new_strategy in zip(strategies, new_strategies)])

def retrieve_dual_solution(A, G, slack, grad):
    # solving the following linear system:
    # [ AT   GT    [ nu     = [ - grad 
    #   0  slack ]   lamb ]       0    ]

    B = torch.cat([A.t(), torch.zeros(slack.shape[0], A.shape[0])])
    C = torch.cat([G.t(), torch.diag(slack)])
    BC = torch.cat([B,C], dim=1)
    r = torch.cat([-grad, torch.zeros(slack.shape[0])])

    solution, LU = torch.lstsq(r.view(-1,1), BC)

    nu, lamb = solution[:A.shape[0],0], solution[A.shape[0]:A.shape[0]+slack.shape[0],0]
    return nu, lamb

def retrieve_dual_solution_cvxpy(A, G, slack, grad):
    lamb = cp.Variable(G.shape[0])
    eps = cp.Variable(1)
    if A.shape[0] != 0:
        nu = cp.Variable(A.shape[0])
        # soc_constraint = [cp.SOC(eps, grad.detach().numpy() + G.t().detach().numpy() @ lamb + A.t().detach().numpy() @ nu), slack.detach().numpy() @ lamb >= 0, lamb >= 0] # eps-KKT conditions
        soc_constraint = [cp.SOC(eps, grad.detach().numpy() + G.t().detach().numpy() @ lamb + A.t().detach().numpy() @ nu), slack.detach().numpy() @ lamb >= - eps, lamb >= 0] # modified eps-KKT conditions
    else:
        soc_constraint = [cp.SOC(eps, grad.detach().numpy() + G.t().detach().numpy() @ lamb), slack.detach().numpy() @ lamb >= - eps, lamb >= 0] # modified eps-KKT conditions
    
    prob = cp.Problem(cp.Minimize(eps), soc_constraint)
    prob.solve()

    # Print result.
    # print("The optimal value is", prob.value)
    # print(nu.value, lamb.value)

    if A.shape[0] != 0:
        return torch.Tensor(nu.value), torch.Tensor(lamb.value)
    else:
        return torch.Tensor(), torch.Tensor(lamb.value)

def block_diagonal(As):
    n, m = 0, 0
    for A in As:
        if A.shape[0] != 0:
            n += A.shape[0]
            m += A.shape[1]
    B = torch.zeros(n,m)
    i, j = 0, 0
    for A in As:
        if A.shape[0] != 0:
            B[i:i+A.shape[0],j:j+A.shape[1]] = A
            i += A.shape[0]
            j += A.shape[1]

    return B

def MAS_forward(x, fs, constraints, bounds, initial_strategies, f_jacs=None, eps=EPSILON, max_iterations=100, gamma=1):
    n = len(fs)
    strategies = copy.deepcopy(initial_strategies)
    for iteration in range(max_iterations):
        new_strategies = [[] for _ in range(n)] # copy.deepcopy(strategies)
        for agent in range(n):
            new_strategies[agent] = best_response(agent, x, strategies, fs, f_jacs, constraints, bounds)

        difference = norm(strategies, new_strategies)
        print('iteration #{} with difference {}'.format(iteration, difference))
        if difference < eps:
            break
        else:
            for i in range(len(strategies)):
                decayed_gamma = gamma # / (iteartion + 1)
                strategies[i] = strategies[i] * (1 - decayed_gamma) + new_strategies[i] * decayed_gamma

    return strategies

def MAS(fs, number_of_actions, constraints, bounds, strategies, Qs, eps=EPSILON, max_iterations=1000):
    class MASFn(torch.autograd.Function):
        # def __init__(self, n, fs, constraints, bounds, strategy_initiator, eps=1e-8):
        #     super(MAS, self).__init__()
        #     self.n = n
        #     self.fs = fs # objective functions
        #     # fs: array of f
        #     # f: x, actions -> real value, representing the agent i's utility function
        #     # x is the parameter governing the utility function
        #     # actions is an array of all agents' actions, following the order
    
        #     self.constraints = constraints # Constraint dictionary. Currently only support linear constraints.
        #     self.bounds = bounds
        #     self.max_iterations = 1000
        #     self.strategy_initiator = strategy_initiator
        #     self.eps = eps
    
        @staticmethod
        def forward(ctx, *grads_sparse):
            n = len(fs)
            grads = torch.cat([grad.view(1) for grad in grads_sparse])
            assert(len(grads) == sum(number_of_actions) and len(Qs) == n)

            ctx.fs, ctx.constraints, ctx.bounds, ctx.eps = fs, constraints, bounds, eps
            # ================================ solving forward pass ===============================
            # strategies = equilibrium_solver(x, fs, constraints, bounds, strategies, eps=eps, max_iterations=max_iterations)
            # strategies = initial_strategies
            # for iteration in range(max_iterations):
            #     new_strategies = copy.deepcopy(strategies)
            #     for agent in range(n):
            #         new_strategies[agent] = best_response(agent, x, strategies, ctx.fs, ctx.constraints, ctx.bounds)
    
            #     difference = norm(strategies, new_strategies)
            #     # print('iteration #{} with difference {}'.format(iteration, difference))
            #     if difference < ctx.eps:
            #         break
            #     else:
            #         strategies = new_strategies

            # ============================ constructing the KKT matrix ============================
            G_list, A_list, slack_list, lamb_list, nu_list = [], [], [], [], []
            action_count = 0
            for agent in range(n):
                # print('resolving kkt of agent #{}'.format(agent))
                gradi = - grads[action_count:action_count+number_of_actions[agent]] # + EPSILON * strategies[agent] # since f is being maximized but the formulation here is for minimization
                Qi = - Qs[agent] # since f is being maximized but the formulation here is for minimization
                # Qi[:,action_count:action_count+number_of_actions[agent]] += EPSILON * torch.eye(number_of_actions[agent])
                # print('eigenvalues:', torch.eig(Qi[:,action_count:action_count+number_of_actions[agent]])[0])
                strategy = strategies[agent]
                Ai, bi, Gi, hi = constraints[agent]
                si = Gi @ strategy - hi
                si = torch.clamp(si, max=-EPSILON)
                nui, lambi = retrieve_dual_solution_cvxpy(Ai, Gi, si, gradi)
                # nui, lambi = retrieve_dual_solution(Ai, Gi, si, gradi)
                lambi = torch.clamp(lambi, min=0)
                action_count += number_of_actions[agent]

                # verification
                diff  = Ai.t() @ nui + Gi.t() @ lambi + gradi
                diff2 = si * lambi
                # print('difference:', diff, diff2)

                G_list.append(Gi)
                A_list.append(Ai)
                slack_list.append(si)
                lamb_list.append(lambi)
                nu_list.append(nui)

            Q     = torch.cat(Qs)
            G     = block_diagonal(G_list)
            A     = block_diagonal(A_list)
            slack = torch.cat(slack_list)
            lamb  = torch.cat(lamb_list)
            nu    = torch.cat(nu_list)

            ctx.save_for_backward(*strategies, Q, G, A, slack, lamb, nu, grads)
            return tuple(strategies)
    
        @staticmethod
        def backward(ctx, *dl_dstrategies):
            # dl_dstrategies is a tuple of dl_dstrategy, the derivative of the loss with respect to action ai of agent i
            # print('start of backward')
            n = len(fs)
            strategies = list(ctx.saved_tensors[:n])
            Q, G, A, slack, lamb, nu, grads = ctx.saved_tensors[n:]

            l1, l2 = G.shape[0], A.shape[0]
            strategy_size, slack_size, lamb_size, nu_size = Q.shape[0], slack.shape[0], lamb.shape[0], nu.shape[0]

            # ------------- with inequality constraints ------------
            if A.size()[0] != 0: # with equality constraints
                KKT_mat = torch.cat([torch.cat([Q, G.t(), A.t()], dim=1),
                        torch.cat([lamb.view(-1,1) * G, torch.diag(slack), torch.zeros(l1,l2)], dim=1),
                        torch.cat([A, torch.zeros(l2, l1+l2)], dim=1)])
            else: # no equality constraint
                KKT_mat = torch.cat([torch.cat([Q, G.t()], dim=1),
                        torch.cat([lamb.view(-1,1) * G, torch.diag(slack)], dim=1)])

            # KKT_mat = KKT_mat + EPSILON * torch.eye(KKT_mat.shape[0])
            dl_da = - torch.cat([torch.cat(list(dl_dstrategies)), torch.zeros(lamb_size + nu_size)])

            # ----------- without inequality constraints -----------
            # KKT_mat = torch.cat([torch.cat([Q, A.t()], dim=1),
            #         torch.cat([A, torch.zeros(l2, l2)], dim=1)])

            # dl_da = - torch.cat([torch.cat(list(dl_dstrategies)), torch.zeros(nu_size)])

            sol, _ = torch.solve(dl_da.view(-1,1), KKT_mat)
            dgrad = sol[:strategy_size,0]

            return tuple(dgrad)

    return MASFn.apply
