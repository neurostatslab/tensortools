"""
Canonical Polyadic Decomposition
================================
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools as itr
from tensorly.decomposition import parafac
from tensorly.kruskal import plot_kruskal, align_kruskal, redistribute_kruskal

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
model1 = parafac(data, R, init='random', verbose=1)
model2 = parafac(data, R, init='random', verbose=1)

plt.figure()
gs = plot_kruskal(model1, width_ratios=[1,2,1])
plot_kruskal(model2, color='r', gs=gs)

align1, align2, score = align_kruskal(model1, model2, greedy=False)
plt.figure()
gs = plot_kruskal(align1, width_ratios=[1,2,1])
plot_kruskal(align2, color='r', gs=gs)

newmod1 = redistribute_kruskal(align1,ratios=[1,0,0])
plt.figure()
gs = plot_kruskal(align1, width_ratios=[1,2,1])
plot_kruskal(newmod1, color='r', gs=gs)

plt.show()
