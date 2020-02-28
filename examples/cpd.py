import tensortools as tt
import numpy as np
import matplotlib.pyplot as plt

# Make synthetic dataset.
I, J, K, R = 25, 30, 35, 4  # dimensions and rank
X = tt.randn_ktensor((I, J, K), rank=R).full()
X += np.random.randn(I, J, K)

# Fit CP tensor decomposition (two times).
U = tt.cp_als(X, rank=R, verbose=True)
V = tt.cp_als(X, rank=R, verbose=True)

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


pred = U.predict()

fig, ax = plt.subplots()
ax.scatter(pred.ravel(), X.ravel())
plt.plot([pred.min(), pred.max()], [pred.min(), pred.max()], 'k--')

# Show plots.
plt.show()
