from tensorly import kruskal_to_tensor
import tensortools as tt
import numpy as np

# make factors
dims = [199, 200, 201]
ndim = len(dims)
R = 5
true_factors = [np.random.randn(n,R) for n in dims]

# make data
data = kruskal_to_tensor(true_factors)

# fit model
def check_score(method):
    est_factors, info = method(data, R)
    return tt.align_factors(true_factors, est_factors)[2]

print('CP-RAND')
print('-'*30)
print(check_score(tt.cp_rand))

print('CP-MIX-RAND')
print('-'*30)
print(check_score(tt.cp_mixrand))

# if score < 0.98:
#     raise Exception('cp rand did not converge to correct solution')
