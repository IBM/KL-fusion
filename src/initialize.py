import numpy as np
from copy import deepcopy

def random_parametric(generator, k):
    return [generator() for _ in range(k)]


def random_choice(subset_posteriors, k):
    subset_posteriors = deepcopy(subset_posteriors)
    result = []
    for i in range(k):
        sub = np.random.choice(len(subset_posteriors))
        pick = np.random.choice(len(subset_posteriors[sub]))
        result.append(subset_posteriors[sub][pick])
    return result


def kmeanspp(posteriors, dist_fn, k, compute_reg=False, use_sk=True):        
    posteriors = deepcopy(posteriors)
    num = len(posteriors)
    if num <= k:
        result = posteriors
    else:
        result = [np.random.choice(posteriors)]
        min_dists = np.array([max(0,dist_fn(p, result[0])) for p in posteriors])
        if use_sk:
            n_local_trials = 2 + int(np.log(k))
        else:
            n_local_trials = 1
        for i in range(1, k):
            prob = np.copy(min_dists)
            prob = prob/prob.sum()
            if np.sum(prob) > 0:
                candidates = np.random.choice(posteriors, size=n_local_trials, replace=False, p=prob)
                potentials = np.zeros((n_local_trials,num))
                for j, posterior in enumerate(posteriors):
                    for c_idx, cand in enumerate(candidates):
                        potentials[c_idx,j] = max(0,dist_fn(posterior, cand))
                best_candidate_idx = np.argmax(potentials.sum(axis=1))
                result.append(candidates[best_candidate_idx])
                min_dists = np.minimum(min_dists, potentials[best_candidate_idx])
            else:
                print('Warning: all probabilities are 0: adding nothing')
    
    if compute_reg:
        reg = np.std([dist_fn(g, l) for g in result for l in posteriors])
        return result, reg
    else:
        return result
