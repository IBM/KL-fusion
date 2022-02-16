import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import linear_sum_assignment
from scipy.stats import wishart
import ot

def gen_gaus_mixture(centers, sigmas, mixing_prop=None, M=5000, alpha=10):
    """
    sample from Gaussian mixture
    """
    K, D = centers.shape
    if len(np.array(sigmas[0]).shape) == 2:
        multivar = True
    else:
        multivar = False
        
    if mixing_prop is None:
        mixing_prop = np.random.dirichlet(alpha*np.ones(K))
    assignments = np.random.choice(K, size=M, p=mixing_prop)
    data_j = np.zeros((M,D))
    for k in range(K):
        
        if multivar:
            data_j[assignments==k] = np.random.multivariate_normal(mean=centers[k], cov=sigmas[k], size=(assignments==k).sum())
        else:
            data_j[assignments==k] = np.random.normal(loc=centers[k], scale=sigmas[k], size=((assignments==k).sum(),D))

    return data_j

def make_spd_matrix(n_dim, diag=1.):
    
    A = np.random.normal(size=(n_dim,n_dim))
    U, s, V = np.linalg.svd(np.dot(A.T, A))
    s = np.minimum(s, 1)
    X = np.dot(np.dot(U, diag + np.diag(s)), V)

    return X

def gen_partition(centers, sigmas, J, M=5000, local_mean_sd=0., local_sigma_df=0, a=1, b=1, nonparam=False):
    
    L, D = centers.shape
    
    data = []
    used_components = set()
    data_centers = []
    data_sigmas = []
    
    if nonparam:
        global_p = np.random.beta(a=a, b=b, size=L)
    else:
        global_p = None
        
    for j in range(J):
        
        if nonparam:
            atoms_j_idx = []
            while len(atoms_j_idx) < 2:
                atoms_j_idx = [l for l in range(L) if np.random.binomial(1, global_p[l])]

            used_components.update(atoms_j_idx)
            centers_j = centers[atoms_j_idx]
            sigmas_j = sigmas[atoms_j_idx]

        else:
            centers_j = centers
            sigmas_j = sigmas
            
        if local_mean_sd > 0:
            centers_j = np.random.normal(centers_j, scale=local_mean_sd)
            
        if local_sigma_df > 0:
            sigmas_j = np.array([wishart.rvs(local_sigma_df, s_jl)/local_sigma_df for s_jl in sigmas_j])
        
        
            
        data_j = gen_gaus_mixture(centers_j, sigmas_j, M=M)
            
        data.append(data_j)
        data_sigmas.append(sigmas_j)
        data_centers.append(centers_j)
    
    if nonparam:
        if len(used_components)<L:
            L = len(used_components)
            L_idx = list(used_components)
            global_p = global_p[L_idx]
            centers = centers[L_idx]
            sigmas = sigmas[L_idx]
            print('Removing unused components; new L is %d' % L)
        
    return centers, sigmas, global_p, data_centers, data_sigmas, data

    
def hungarian_match(est_atoms, true_atoms, dist_fn=None, transport=True):
    n_true = len(true_atoms)
    n_est = len(est_atoms)
        
    if dist_fn is None:
        dist = euclidean_distances(true_atoms, est_atoms)
    else:
        dist = np.zeros((n_true, n_est))
        for i, true in enumerate(true_atoms):
            for j, est in enumerate(est_atoms):
                dist[i,j] = dist_fn(true,est)
    
    if transport:
        obj = ot.emd2(np.ones(n_true)/n_true, np.ones(n_est)/n_est, dist)
    else:
        row_ind, col_ind = linear_sum_assignment(dist)
        obj = []
        for r, c in zip(row_ind, col_ind):
            obj.append(dist[r,c])
        obj = np.mean(obj)
        
    return obj

def min_match(est_atoms, true_atoms, dist_fn=None):
    if dist_fn is None:
        dist = euclidean_distances(true_atoms, est_atoms)
    else:
        dist = np.zeros((len(true_atoms), len(est_atoms)))
        for i, true in enumerate(true_atoms):
            for j, est in enumerate(est_atoms):
                dist[i,j] = dist_fn(true,est)
    
    e_to_t = dist.T         
    return max([max(np.min(e_to_t, axis=0)), max(np.min(e_to_t, axis=1))])

def matching_metrics(est_atoms, true_atoms, dist_fn=None, transport=True):
    n_true = len(true_atoms)
    n_est = len(est_atoms)
        
    if dist_fn is None:
        dist = euclidean_distances(true_atoms, est_atoms)
    else:
        dist = np.zeros((n_true, n_est))
        for i, true in enumerate(true_atoms):
            for j, est in enumerate(est_atoms):
                dist[i,j] = dist_fn(true,est)

    if transport:
        hm = ot.emd2(np.ones(n_true)/n_true, np.ones(n_est)/n_est, dist)
    else:
        row_ind, col_ind = linear_sum_assignment(dist)
        hm = []
        for r, c in zip(row_ind, col_ind):
            hm.append(dist[r,c])
        hm = np.mean(hm)
    
    e_to_t = dist.T
    mm = max([max(np.min(e_to_t, axis=0)), max(np.min(e_to_t, axis=1))])
    
    return mm, hm