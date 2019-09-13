import tensortools as tt
import numpy as np

# Make synthetic dataset.
I, J, K, R = 25, 25, 25, 4  # dimensions and rank
X = tt.randn_ktensor((I, J, K), rank=R).full()
X += np.random.randn(I, J, K)

# Fit CP tensor decomposition to first 20 trials.
U = tt.cp_als(X[:, :, :20], rank=R, verbose=True)

# Extend and re-initialize the factors along the final mode.
Uext = U.factors.copy()
Uext.factors[-1] = np.random.randn(K, R)
Uext.shape = (I, J, K)

# Fit model to the full dataset, only fitting the final set of factors.
V = tt.cp_als(X, rank=R, init=Uext, skip_modes=[0, 1], verbose=True)
