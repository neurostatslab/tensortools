"""
CP Decomposition via Randomized ALS
===================================
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools as itr
from tensorly.decomposition import parafac
from tensorly.kruskal import plot_kruskal, align_kruskal, standardize_kruskal

# N, size of matrix. R, rank of data
N = 50
R = 4

# make fake data
factors = list(np.random.randn(3,N,R).astype(np.float32))
data = np.zeros((N,N,N))

# make data
for i,j,k,r in itr.product(range(N), range(N), range(N), range(R)):
    data[i,j,k] += factors[0][i,r]*factors[1][j,r]*factors[2][k,r]

# fit CP decomposition
print('-'*50)
print('RAND ALS')
print('-'*50)
Xr, info_r = parafac(data, R, init='randn', exact=False, sample_frac=0.8,
                     tol=1e-7, verbose=1, n_iter_max=1000, print_every=10)

print('-'*50)
print('EXACT ALS')
print('-'*50)
Xe, info_e = parafac(data, R, init='randn', tol=1e-7, verbose=1,
                     print_every=10, n_iter_max=1000)
