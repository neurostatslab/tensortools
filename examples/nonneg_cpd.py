import tensortools as tt
import numpy as np
import matplotlib.pyplot as plt

# Make synthetic dataset.
I, J, K, R = 25, 25, 25, 4  # dimensions and rank
X = tt.randexp_ktensor((I, J, K), rank=R).full()
X += np.random.randn(I, J, K) * .5
X = np.maximum(X, 0.0)

# Fit CP tensor decomposition (two times).
U = tt.ncp_bcd(X, rank=R, verbose=True)
V = tt.ncp_bcd(X, rank=R, verbose=True)

# Compare the low-dimensional factors from the two fits.
fig, ax, po = tt.plot_factors(U.factors)
tt.plot_factors(V.factors, fig=fig)
fig.suptitle("raw models")
fig.tight_layout()

# Align the two fits and print a similarity score.
sim = tt.kruskal_align(U.factors, V.factors, permute_U=True, permute_V=True)
print(sim)

# Plot the results again to see alignment.
fig, ax, po = tt.plot_factors(U.factors)
tt.plot_factors(V.factors, fig=fig)
fig.suptitle("aligned models")
fig.tight_layout()

# Show plots.
plt.show()
