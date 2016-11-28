"""
Canonical Polyadic Decomposition
================================
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
X1, info1 = parafac(data, R, init='random', tol=1e-7, verbose=1, n_iter_max=1000)
X2, info2 = parafac(data, R, init='random', tol=1e-7, verbose=1, n_iter_max=1000)

ax = plot_kruskal(X1, width_ratios=[1,2,1])
plot_kruskal(X2, color='r', ax=ax)

align1, align2, score = align_kruskal(X1, X2, greedy=False)
ax = plot_kruskal(align1, width_ratios=[1,2,1])
plot_kruskal(align2, color='r', ax=ax)

newmod1 = standardize_kruskal(align1, lam_ratios=[1,0,1])
ax = plot_kruskal(align1, width_ratios=[1,2,1])
plot_kruskal(newmod1, color='r', ax=ax, suptitle='Supertitle', link_yaxis=True)

plt.show()
