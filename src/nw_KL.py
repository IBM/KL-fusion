import numpy as np
from sklearn.datasets import make_spd_matrix
from scipy.special import multigammaln
from scipy.special import digamma
from w2_gaussians import gaussian_dist, gaussian_bary
from scipy.stats import wishart

def digamma_mvar(a, d):
    res = np.sum(digamma([(a - (j - 1.)/2) for j in range(1, d+1)]), axis=0)
    return res

def gaus_kl(m0, m1, k0, k1, S):
    """
    Two gaussians have same precision S, but with different scales k0 and k1
    """
    D = S.shape[0]
    const = D*(np.log(k0/k1) - 1 + k0/k1)
    diff = m1 - m0
    q_form = (np.dot(diff,S)*diff).sum()
    return 0.5*(const + k1*q_form)
    
def wishart_kl(nu0, nu1, S0, S1_inv):
    D = S0.shape[0]
    prod = np.matmul(S1_inv,S0)
    _, logdet = np.linalg.slogdet(prod)
    m_part = -(nu1/2)*logdet + (nu0/2)*(np.trace(prod) - D)
    log_gamma = multigammaln(nu1/2, D) - multigammaln(nu0/2, D)
    dig = (nu0 - nu1)*digamma_mvar(nu0/2, D)/2
    return m_part + log_gamma + dig

   
def nw_kl(nw_0, nw_1):
    cond_gaus_kl = gaus_kl(nw_0['mu'], nw_1['mu'], 
                           nw_0['kappa'], nw_1['kappa'],
                           nw_0['nu']*nw_0['S'])
    
    wi_kl = wishart_kl(nw_0['nu'], nw_1['nu'],
                       nw_0['S'], nw_1['S_inv'])
    
    return cond_gaus_kl + wi_kl

def nw_to_natural(nw):
    nw_natural = {}
    nw_natural['kappa'] = nw['kappa']
    nw_natural['nu'] = nw['nu'] # we can ignore d + 2 term
    nw_natural['mu'] = nw['kappa']*nw['mu']
    nw_natural['sigma'] = nw['S_inv'] + nw['kappa']*np.outer(nw['mu'], nw['mu'])
    return nw_natural

def nw_to_raw(nw_natural):
    nw = {}
    nw['kappa'] = nw_natural['kappa']
    nw['nu'] = nw_natural['nu']
    nw['mu'] = nw_natural['mu']/nw['kappa']
    nw['S_inv'] = nw_natural['sigma'] - nw['kappa']*np.outer(nw['mu'], nw['mu'])
    nw['S'] = np.linalg.inv(nw['S_inv'])
    return nw

def nw_to_gaus(nw):
    return nw['mu'], nw['S_inv']/nw['kappa']

def nw_kl_barycenter(nw_list, weights=None):
    nw_natural_list = [nw_to_natural(nw) for nw in nw_list]
    barycenter_natural = {}

    if weights is None:
        n = len(nw_natural_list)
        weights = np.ones(n) / n
    for p in ['kappa', 'nu', 'mu', 'sigma']:
        barycenter_natural[p] = np.average([nw_natural[p] for nw_natural in nw_natural_list], weights=weights, axis=0)

    barycenter_raw = nw_to_raw(barycenter_natural)

    return barycenter_raw

def nw_logpart(nw):
    D = nw['S'].shape[0]
    const = np.log(2*np.pi)*D/2
    k_part = -np.log(nw['kappa'])*D/2
    nu_part = nw['nu']*np.log(2)*D/2
    _, logdet = np.linalg.slogdet(nw['S'])
    S_part = nw['nu']*logdet/2
    gamma_part = multigammaln(nw['nu']/2, D)
    return const + k_part + nu_part + S_part + gamma_part

def nw_w_sample_dist(nw, gaus):
    nw_var = np.linalg.inv(wishart.rvs(nw['nu'], nw['S']))/nw['kappa']
    return gaussian_dist(gaus, [nw['mu'], nw_var])

def nw_w_sample_barycenter(nw_list, weights=None):
    gaus_mus = [nw['mu'] for nw in nw_list]
    gaus_var = [np.linalg.inv(wishart.rvs(nw['nu'], nw['S']))/nw['kappa'] for nw in nw_list]

    return gaussian_bary(zip(gaus_mus, gaus_var), weights)

def gen_nw(d, seed=None):
    if seed is not None:
        np.random.seed(seed)
    nw = {}
    nw['kappa'] = np.random.uniform(0.5,2.)
    nw['nu'] = d - 1 + np.random.uniform(0.5, 10.)
    nw['mu'] = np.random.normal(size=d)
    nw['S'] = make_spd_matrix(d)
    nw['S_inv'] = np.linalg.inv(nw['S'])
    return nw

def barycenter_objective(barycenter, nw_list):
    obj = [nw_kl(barycenter, nw) for nw in nw_list]
    return np.mean(obj)

def nw_from_VI(VI, thr=0.):
    nw_list = []
    K = len(VI.mean_precision_)
    for k in range(K):
        if VI.weights_[k] > thr:
            nw = {}
            nw['kappa'] = VI.mean_precision_[k]
            nw['nu'] = VI.degrees_of_freedom_[k]
            nw['mu'] = VI.means_[k]
            if len(VI.precisions_[k].shape) == 1:
                nw['S'] = np.diag(VI.precisions_[k])/VI.degrees_of_freedom_[k]
                nw['S_inv'] = np.diag(VI.covariances_[k])*VI.degrees_of_freedom_[k]
            else:
                nw['S'] = VI.precisions_[k]/VI.degrees_of_freedom_[k]
                nw['S_inv'] = VI.covariances_[k]*VI.degrees_of_freedom_[k]
    
            nw_list.append(nw)
    return nw_list

def nw_from_bnpy(bn):
    nw_list = []
    K = bn['m'].shape[0]
    for k in range(K):
        nw = {}
        nw['kappa'] = bn['kappa'][k]
        nw['nu'] = bn['nu'][k]
        nw['mu'] = bn['m'][k]
        nw['S'] = bn['B'][k]
        nw['S_inv'] = np.linalg.inv(bn['B'][k])
        nw_list.append(nw)
    return nw_list
    