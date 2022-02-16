import sys
sys.path.append("./src/")
from data_sampling import gen_partition, make_spd_matrix, min_match, hungarian_match
from nw_KL import nw_from_VI, nw_kl, nw_kl_barycenter
from fusion import fusion_alternating
from plot_gmm import scale_margins, plot_estimation
from initialize import kmeanspp

from sklearn.mixture import BayesianGaussianMixture as VI_mixture
from sklearn.cluster import KMeans
import numpy as np

np.set_printoptions(precision=3, suppress=True)

np.random.seed(17)

# Data generating parameters
L = 3 # number of global atoms
J = 10 # number of groups
D = 2 # data dimension
M = 300 # data points per group (not relevant if we use true local atoms)
mu0 = 0.
local_mean_sd= 0.1 # 0.5; 0.5; 2.
separation_scale = 0.1 #1.; 0.1; 0.5
local_sigma_df = D*100
parametric = False

# Make global components
centers = np.random.normal(loc=mu0, scale=np.sqrt(separation_scale*L), size=(L,D))
sigmas = np.array([make_spd_matrix(D, diag=np.sqrt(1+l)) for l in range(L)])

# Generate data; set nonparam=True for unkn
centers, sigmas, global_p, data_centers, data_sigmas, datasets = gen_partition(centers, sigmas, J, M=M, nonparam=not parametric, local_mean_sd=local_mean_sd, local_sigma_df=local_sigma_df)
L = centers.shape[0]

margins = scale_margins([np.min(datasets, axis=(0,1)), np.max(datasets, axis=(0,1))])

for j in range(J):
    print('Plotting noisy dataset %d' % j)
    mm_error = min_match(data_centers[j], centers)
    print('Hausdorff distance between dataset %d and global means is %f\n' % (j,mm_error))
    plot_estimation([centers, sigmas], [data_centers[j], data_sigmas[j]], margins, X=datasets[j])
print('-----------------------------------')

# Fit VI
VI_centers = []
VI_sigmas = []
VI_nw = []
for j, data in enumerate(datasets):
    Lj = data_centers[j].shape[0]
    VI_j = VI_mixture(n_components=Lj, covariance_type='full', n_init=16, weight_concentration_prior_type='dirichlet_distribution')
    VI_j.fit(data)
    if VI_j.converged_:
        print('VI converged on dataset %d' % j)
    else:
        print('VI did NOT converge on dataset %d' % j)

    centers_j = VI_j.means_
    sigmas_j = VI_j.covariances_

    mm_error = min_match(centers_j, data_centers[j])

    print('Hausdorff distance between fitted means for dataset %d and its means is %f' % (j,mm_error))

    VI_centers.append(centers_j)
    VI_sigmas.append(sigmas_j)
    VI_nw.append(nw_from_VI(VI_j))

    plot_estimation([data_centers[j], data_sigmas[j]], [centers_j, sigmas_j], margins, X=datasets[j])

flat_VI = [VI for subset in VI_nw for VI in subset]

print('-----------------------------------')
result = {'MM':{}, 'HM':{}}

## K-means for sanity check.
print('RUNNING K-means on combined data\n')
all_data = np.vstack(datasets)
kmeans_centers_all = KMeans(n_clusters=L, n_init=8, n_jobs=-1).fit(all_data).cluster_centers_
result['MM']['K-means'] = min_match(kmeans_centers_all, centers)
result['HM']['K-means'] = hungarian_match(kmeans_centers_all, centers)
print('Hausdorff distance between K-means fit on all data and true means is %f; Hungarian is %f' % (result['MM']['K-means'], result['HM']['K-means']))
plot_estimation([centers, sigmas], [kmeans_centers_all, None], margins)
print('-----------------------------------')

## Test fusion strategy
print('RUNNING KL Fusion\n')
reg_scale = 0.5
if parametric:
    init, reg = VI_nw[0], 0.
else:
    init, reg = kmeanspp(flat_VI, nw_kl, 8, compute_reg=True)
priors, _, mean_fusion_centers = fusion_alternating(VI_nw, init, dist_fn=nw_kl, bary_fn=nw_kl_barycenter, reg=reg*reg_scale, true_centers=centers, true_covs=sigmas, margins=margins, parametric=parametric)
fusion_centers = np.array([prior['mu'] for prior in priors])
result['MM']['KL-fusion'] = min_match(fusion_centers, centers)
result['HM']['KL-fusion'] = hungarian_match(fusion_centers, centers)
print('Hausdorff distance between fusion averaging of VI fits and true means is %f; Hungarian is %f\n' % (result['MM']['KL-fusion'], result['HM']['KL-fusion']))

print('-----------------------------------')
for m_dist in result:
    print(m_dist + ':')
    for r in result[m_dist]:
        print(r, result[m_dist][r])
    print('\n')