from tensorly import kruskal_to_tensor
import tensortools as tt
import numpy as np

# make factors
N, R, ndim = 200, 5, 3
true_factors = [np.random.randn(N,R) for _ in range(ndim)]

# make data
data = kruskal_to_tensor(true_factors)

# fit model
est_factors, info = tt.cp_rand(data, R)

score = tt.align_kruskal(true_factors, est_factors)[2]

# if score < 0.98:
#     raise Exception('cp rand did not converge to correct solution')
