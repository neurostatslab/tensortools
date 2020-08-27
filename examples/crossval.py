import numpy as np
import tensortools as tt

# Make synthetic dataset (nonnegative data).
I, J, K, R = 25, 25, 25, 4  # dimensions and rank
X = tt.randexp_ktensor((I, J, K), rank=R).full()
X += np.random.randn(I, J, K) * .5
X = np.maximum(X, 0.0)

# Create random mask to holdout ~10% of the data at random.
mask = np.random.rand(I, J, K) > .1

# Fit nonnegative tensor decomposition.
U = tt.ncp_hals(X, rank=R, mask=mask, verbose=False)

# Compute model prediction for full tensor.
Xhat = U.factors.full()

# Compute norm of residuals on training and test sets.
train_error = np.linalg.norm(Xhat[mask] - X[mask]) / np.linalg.norm(X[mask])
test_error = np.linalg.norm(Xhat[~mask] - X[~mask]) / np.linalg.norm(X[~mask])

# Print result.
print("TRAINING ERROR:", train_error / np.linalg.norm(X[mask]))
print("TESTING ERROR: ", test_error / np.linalg.norm(X[~mask]))


### NOTES ###

# Currently, ncp_bcd does not support masking, use ncp_hals instead.

# For unconstrained (i.e. not nonnegative) tensor decompositions, use
# the `mcp_als` to optimize with a mask.

# You can also use the `mask` to handle missing data.
