import numpy as np
import scipy.linalg as la

#https://arxiv.org/pdf/1511.05355.pdf

def gaussian_bary(mus, Lambda=None, use_cov=True):
    """ Wasserstein barycenter of Gaussians.

    Uses a fixed point iteration scheme for the barycenter covariance.

    Args:
      mus: List of (mu, Sigma) pairs representing a Gaussian distribution.
      Lambda: Weight vector.
    """
    Mu, Sigma = [], []
    for gaussian in mus:
        Mu.append(gaussian[0])
        Sigma.append(gaussian[1])
    Mu = np.array(Mu)
    Sigma = np.array(Sigma)

    n = len(Mu)
    mu = np.average(Mu, axis=0, weights=Lambda)
    if Lambda is None:
        Lambda = 1 / n * np.ones(n)
        
    if use_cov:
        S = fixed_point(Sigma, Lambda)
    else:
        S = None
        
    return mu, S


def fixed_point(Sigma, Lambda, maxiters=100):
    diag = False
    if len(Sigma[0].shape) == 1:
        diag = True

    dim = Sigma[0].shape[0]
    S = np.eye(dim)
    prevS = np.copy(S)
    for i in range(maxiters):
        iS = la.inv(S)
        sS = la.sqrtm(S)
        siS = la.sqrtm(iS)
        P = np.zeros(S.shape)
        for sigma_i, sigma in enumerate(Sigma):
            if diag:
                sigma = np.diag(sigma)
            P += np.real(Lambda[sigma_i] * la.sqrtm(sS @ sigma @ sS))
        P = P @ P
        S = siS @ P @ siS
        if la.norm(S - prevS) < 1e-6:
            break
        prevS = np.copy(S)
    return S


def gaussian_dist(mu, nu, use_cov=True):
    """ Wasserstein distance between two Gaussians.
    """
    mu1, Sigma1 = mu
    mu2, Sigma2 = nu
    if Sigma1 is None or Sigma2 is None:
        return la.norm(mu1 - mu2) ** 2
    else:
        if len(Sigma1.shape) == 1:
            Sigma1 = np.diag(Sigma1)
        if len(Sigma2.shape) == 1:
            Sigma2 = np.diag(Sigma2)
    
        sS1 = la.sqrtm(Sigma1)
        cross_cov_term = np.real(la.sqrtm(sS1 @ Sigma2 @ sS1))
        cov_term = np.trace(Sigma1 + Sigma2 - 2*cross_cov_term)
        
        return la.norm(mu1 - mu2) ** 2 + cov_term
