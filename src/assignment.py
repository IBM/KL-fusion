import ot
import numpy as np
import cvxpy as cp
import time

def nonparam_assignment(priors, subset_posteriors, dist_fn, reg=0.1, verbose=False):
    n = len(subset_posteriors)
    G = len(priors)
    L = len(subset_posteriors[0])
    costs = []
    flat_costs = []
    t_s = time.time()
    for k, dataset in enumerate(subset_posteriors):
        L = len(dataset)
        cost_m = np.zeros((G, L))
        for j, mu in enumerate(dataset):
            for i, nu in enumerate(priors):
                cost_m[i, j] = dist_fn(mu, nu)
                flat_costs.append(cost_m[i, j])
        costs.append(cost_m)
    if verbose:
        print('Computing costs took', time.time()-t_s, 'seconds')
        
    t_s = time.time()
    obj = 0.0
    variables = []
    constraints = []
    for k in range(n):
        L = len(subset_posteriors[k])
        P = cp.Variable((G, L))
        variables.append(P)
        obj += cp.sum(cp.multiply(P, costs[k]))
        constraints.extend([cp.sum(P, axis=0) == 1,
                            cp.sum(P, axis=1) <= 1,
                            P >= 0])
    
    size = np.max([v.shape[1] for v in variables])
    for g in range(G):
        cost = [0.0] * size
        for i, P in enumerate(variables):
            for l in range(P.shape[1]):
                cost[l] += P[g, l]
        obj += reg * np.sqrt(max(0,g+1-size)) * cp.pnorm(cp.vstack(cost))

    problem = cp.Problem(cp.Minimize(obj), constraints)
    
    if verbose:
        print('Building objective took', time.time()-t_s, 'seconds')
        
    t_s = time.time()
    
    problem.solve(solver=cp.SCS)
    
    if verbose:
        print('Solving took', time.time()-t_s, 'seconds')
    
    assignments = []
    for P in variables:
        assignments.append(P.value)

    return assignments


def sinkhorn(priors, subset_posteriors, dist_fn, reg=10.):
    """Solve an assignment problem.

     We want to solve for
       \min_B \sum B[i, j, k] * cost[i, j, k] + reg * H(B)
     subject to constraints
       \sum_i B[i, j, k] = 1
       \sum_{j, k} B[i, j, k] >= 1

     The first constraint guarantees that no subset posterior in each dataset matches to more
     than one posterior distribution. The second constraint guarantees that each posterior
     matches at least one subset posterior.

     H refers to entropy of the transport plan, and `reg` is a regularization parameter`.

     Args:
       priors: A list of (mu, Sigma) pairs representing a Gaussian.
       subset_posteriors: A list of subsets each of which contains a list of (mu, Sigma) pairs.

     Returns:
       An assignment B where B[i, j, k] = 1 says that component i in `priors` matches component
       `j` in the `k`-th subset.
    """

    n = len(subset_posteriors)
    G = len(priors)
    costs = []
    for k, dataset in enumerate(subset_posteriors):
        L = len(dataset)
        cost_m = np.zeros((G, L))
        for j, mu in enumerate(dataset):
            for i, nu in enumerate(priors):
                cost_m[i, j] = dist_fn(mu, nu)
        costs.append(cost_m)

    assignments = []
    for k in range(n):
        L = len(subset_posteriors[k])
        p = np.ones(G) / G
        q = np.ones(L) / L
        assignments.append(ot.bregman.sinkhorn_epsilon_scaling(p, q, costs[k], epsilon0=1e10, reg=reg))
        if not np.allclose(assignments[k].sum(), 1.):
            print('Assignment issues: summing to %f instead of 1' % assignments[k].sum())

    return assignments
