import tensortools as tt
import numpy as np
import matplotlib.pyplot as plt
rs = 1234  # random seed

# dimensions
I, J, K, R = 100, 100, 100, 4

# make tensor
X = tt.randn_tensor((I, J, K), rank=R)

# add noise
Xn = X + np.random.randn(I, J, K)

# fit cp decomposition across a range of ranks
options = {
    'tol': 1e-6,
}
ensemble = tt.Ensemble(options=options)
ensemble.fit(Xn, ranks=range(1, 9), replicates=3)

# plot similarity and error plots
plt.figure()
tt.plot_objective(ensemble)
plt.axhline(np.linalg.norm(X - Xn) / np.linalg.norm(Xn),
            dashes=[2, 2], color='k', zorder=-1, alpha=.7)

plt.figure()
tt.plot_similarity(ensemble)

plt.show()
