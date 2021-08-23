import numpy as np

SEED = 0

# create samples
XY_rng = np.random.default_rng(SEED)
rho = 1 - 0.19 * XY_rng.random()
mean, cov, n = [0, 0], [[1, rho], [rho, 1]], 1000
XY = XY_rng.multivariate_normal(mean, cov, n)

XY_ref_rng = np.random.default_rng(SEED)
cov_ref, n_ = [[1, 0], [0, 1]], n
XY_ref = XY_ref_rng.multivariate_normal(mean, cov_ref, n_)