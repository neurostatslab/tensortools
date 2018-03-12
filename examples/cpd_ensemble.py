import tensortools as tt
import numpy as np
import matplotlib.pyplot as plt
rs = 1234  # random seed

# dimensions
I, J, K, R = 100, 100, 100, 4

# make tensor
X = tt.randn_tensor((I, J, K), rank=R)

# add noise
X += np.random.randn(I, J, K)

# fit cp decomposition across a range of ranks
options = {
    'tol': 1e-6,
}
ensemble = tt.Ensemble(options=options)
ensemble.fit(X, ranks=range(1, 9), replicates=3)

# plot similarity and error plots
plt.figure()
tt.plot_objective(ensemble)

plt.figure()
tt.plot_similarity(ensemble)

plt.show()
