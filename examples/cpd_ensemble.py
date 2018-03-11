import tensortools as tt
import numpy as np
import matplotlib.pyplot as plt
rs = 1234  # random seed

# dimensions
I, J, K, R = 25, 25, 25, 4

# make tensor
X = tt.randn_tensor((I, J, K), rank=R)

# add noise
X += np.random.randn(I, J, K)

# fit cp decomposition across a range of ranks
ensemble = tt.Ensemble()
ensemble.fit(X, ranks=range(1, 9), replicates=3)

# plot similarity and error plots
plt.figure()
tt.plot_objective(ensemble)

plt.figure()
tt.plot_similarity(ensemble)

plt.show()
