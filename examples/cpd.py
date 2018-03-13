import tensortools as tt
from tensortools.optimize import cp_als
import numpy as np
import matplotlib.pyplot as plt
rs = 1234  # random seed

# dimensions
I, J, K, R = 25, 25, 25, 4

# make tensor
X = tt.randn_tensor((I, J, K), rank=R)

# add noise
X += np.random.randn(I, J, K)

# fit cp decomposition twice
U = cp_als(X, rank=R, trace=False)
V = cp_als(X, rank=R, trace=False)

# compare results
fig, ax, po = tt.plot_factors(U.factors)
tt.plot_factors(V.factors, fig=fig)

sim = tt.kruskal_align(U.factors, V.factors, permute_U=True, permute_V=True)
print(sim)

# compare results
fig, ax, po = tt.plot_factors(U.factors)
tt.plot_factors(V.factors, fig=fig)
plt.show()
