import sys
import numpy as np
from assignment import sinkhorn, nonparam_assignment
from copy import deepcopy
from data_sampling import min_match, hungarian_match
from plot_gmm import plot_gmm, plot_estimation
import matplotlib.pyplot as plt
from nw_KL import nw_to_gaus
    
def fusion_alternating(subset_posteriors, priors, dist_fn, bary_fn,
                     num_priors=5, maxiters=10, reg=1., name='KL',
                     true_centers = None, true_covs=None,
                     margins=None, parametric=True, bnn_eval=None, verbose=False):
    """ Alternating procedure for fitting a parametric mixture to subset posteriors.
    """
    
    if bnn_eval is not None:
        verbose = True
        from compute_metrics import compute_metrics
        test_data, biases = bnn_eval
        
    if name in ['gaus', 'nw-sample'] and (not parametric):
        sys.exit('Nonparametric version of ' + name + ' needs to be updated.')

    subset_posteriors = deepcopy(subset_posteriors)
    priors = deepcopy(priors)

    if margins is not None:
        plot_datasets = [[np.array([s_p['mu'] for s_p in s_posterior]), np.array([s_p['S_inv']/s_p['nu'] for s_p in s_posterior])] for s_posterior in subset_posteriors]
        if name in ['gaus', 'nw-sample']:
            cov_scale = np.mean([[s_p['kappa'] for s_p in s_posterior] for s_posterior in subset_posteriors], axis=0)

    if name == 'gaus':
        subset_posteriors = [[nw_to_gaus(VI_k) for VI_k in VI] for VI in subset_posteriors]

    if name in ['gaus', 'nw-sample']:
        priors = [nw_to_gaus(VI_k) for VI_k in priors]

    K = len(subset_posteriors)
    arcs = [(j, k) for k in range(K) for j in range(len(subset_posteriors[k]))]
    mean_fusion_estimate = []

    for it in range(maxiters):
        # Solve assignment problem between current estimate and subset posteriors
        if parametric:
            assign = sinkhorn(priors, subset_posteriors, dist_fn=dist_fn, reg=reg)
        else:
            assign = nonparam_assignment(priors, subset_posteriors, dist_fn=dist_fn, reg=reg, verbose=verbose)
        # Allocate priors to new positions
        for i, prior in enumerate(priors):
            bary_comps, weights = [], []
            for j, k in arcs:
                if assign[k][i, j] > 0:
                    bary_comps.append(subset_posteriors[k][j])
                    weights.append(assign[k][i, j])
            weights /= np.sum(weights)
            if bary_comps != []:
                barycenter = bary_fn(bary_comps, weights)
                priors[i] = barycenter

            if it == maxiters - 1:
                if name in ['KL', 'nw-sample']:
                    mean_fusion_estimate.append(np.sum([w*m_comp['mu'] for w,m_comp in zip(weights,bary_comps)], axis=0))
                elif name == 'gaus':
                    mean_fusion_estimate.append(np.sum([w*m_comp[0] for w,m_comp in zip(weights,bary_comps)], axis=0))
                else:
                    mean_fusion_estimate.append(None)

        if true_centers is not None:
            active_id = np.sum([a.sum(axis=1) for a in assign], axis=0) >= 1.0
            if name == 'KL':
                est_centers = np.array([prior['mu'] for prior in priors])
            elif name in ['gaus', 'nw-sample']:
                est_centers = np.array([prior[0] for prior in priors])
            est_centers = np.array([p for i, p in enumerate(est_centers) if active_id[i] >= 1.0])
            if margins is None:
                print('Running', name, 'at iteration', it, ': MM is %f; HM is %f' % (min_match(est_centers, true_centers), hungarian_match(est_centers, true_centers)))
            if not parametric:
                print('Estimated number of global componets is', est_centers.shape[0], 'true is', true_centers.shape[0])

        if margins is not None:
            active_id = np.sum([a.sum(axis=1) for a in assign], axis=0) >= 1.0
            if name == 'KL':
                est_covs = np.array([prior['S_inv']/prior['nu'] for prior in priors])
            elif name in ['gaus', 'nw-sample']:
                if priors[0][1] is not None:
                    est_covs = np.array([prior[1]*cov_scale[k] for k,prior in enumerate(priors)])
                else:
                    est_covs = None
            if est_covs is not None:
                est_covs = np.array([p for i,p in enumerate(est_covs) if active_id[i]])
            print('Assignments.', 'Running', name, 'at iteration', it, ': MM is %f; HM is %f' % (min_match(est_centers, true_centers), hungarian_match(est_centers, true_centers)))
            if not parametric:
                print('Estimated number of global componets is', est_centers.shape[0], 'true is', true_centers.shape[0], 'regularization:', reg)

            props = [('red','*'), ('green','o'), ('blue','v'), ('orange', 'x'),
                     ('purple', 'v'), ('pink', '^'),
                     ('magenta', 's'), ('brown', 'd')]


            for i in range(len(priors)):
                if active_id[i]:
                    comp_means = []
                    comp_covs = []
                    for j, k in arcs:
                        if assign[k][i, j] > 1/(len(priors)**2):
                            comp_means.append(plot_datasets[k][0][j])
                            comp_covs.append(plot_datasets[k][1][j])
                    comp_means = np.array(comp_means)
                    comp_covs = np.array(comp_covs)
                    plot_gmm(comp_means, comp_covs, margins, marker=props[i][1], color = [props[i][0]]*len(comp_means))

            plt.show()
            print('Estimation.', 'Running', name, 'at iteration', it, ': MM is %f; HM is %f' % (min_match(est_centers, true_centers), hungarian_match(est_centers, true_centers)))
            if not parametric:
                print('Estimated number of global componets is', est_centers.shape[0], 'true is', true_centers.shape[0], 'regularization:', reg)
            plot_estimation([true_centers, true_covs], [est_centers, est_covs], margins)

        # Update centers every iteration
        active_id = np.sum([a.sum(axis=1) for a in assign], axis=0)
        priors = [prior for prior, value in sorted(zip(priors, active_id), key=lambda x: -x[1])]
        if verbose:
            print('Iteration %d number of active components is %d' % (it, (active_id>0.999).sum()))
        if bnn_eval is not None:
            fused_weights = gaus_to_weights(np.array([p for i,p in enumerate(priors) if active_id[i] >= 0.999]), biases, n_classes = 10)
            print(compute_metrics(fused_weights, test_data['Xtest'], test_data['Ytest'], n_samples=100),'\n')
        sys.stdout.flush()
        
    # Remove empty components
    if not parametric:
        active_id = np.sum([a.sum(axis=1) for a in assign], axis=0) >= 0.999
        assign = [a_j[active_id] for a_j in assign]
        priors = np.array([p for i,p in enumerate(priors) if active_id[i]])
        mean_fusion_estimate = np.array(mean_fusion_estimate)[active_id]
        
    return priors, assign, mean_fusion_estimate

def gaus_to_weights(priors, biases, n_classes = 10):
    priors = deepcopy(priors)
    gaus_means = np.array([e['mu'] for e in priors])
    gaus_covs = np.array([e['Sigma'] for e in priors])
    weights = {}
    weights['means'] = [[],[]]
    weights['logvar'] = [[],[]]
    weights['means'][0] = gaus_means[:,:-n_classes].T
    weights['means'][1] = np.vstack([gaus_means[:,-n_classes:], biases['mu']])
    weights['logvar'][0] = np.log(gaus_covs[:,:-n_classes]).T
    weights['logvar'][1] = np.log(np.vstack([gaus_covs[:,-n_classes:], biases['Sigma']]))
    return weights
