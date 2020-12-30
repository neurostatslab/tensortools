"""
Shifted tensor decomposition with per-dimension shift
parameters along only axis=1.
"""
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from tensortools.cpwarp import ShiftedCP, fit_shifted_cp, shifted_align
from tensortools.visualization import plot_factors

from scipy.ndimage import gaussian_filter1d
import itertools
from time import time

# Generate random low-rank factors and shifts.
I, J, K = 100, 101, 102
max_shift = 0.1
rank = 3
npr.seed(1234)

u = npr.exponential(1.0, size=(rank, I))
v = npr.exponential(1.0, size=(rank, J))
w = gaussian_filter1d(
    npr.exponential(1.0, size=(rank, K)), 3, axis=-1)

v_s = npr.uniform(-max_shift * K, max_shift * K, (rank, J))

# Store ground truth factors and generate noisy data.
ground_truth = ShiftedCP(u, v, w, u_s=None, v_s=v_s, boundary="edge")

noise_scale = 0.1
data = np.maximum(
    0., ground_truth.predict() + noise_scale * npr.randn(I, J, K))

# Fit model.
t0 = time()
model = fit_shifted_cp(
    data, rank, n_restarts=3,
    boundary="edge",
    max_shift_axis0=None,
    max_shift_axis1=max_shift,
    max_iter=60)

print("time per iteration: {}".format(
    (time() - t0) / len(model.loss_hist)))

# Plot loss history.
fig, ax = plt.subplots(1, 1)
ax.plot(model.loss_hist)
ax.set_ylabel("Normalized Error")
ax.set_xlabel("Iteration")

# Plot factors before alignment.
fig, axes, _ = plot_factors(model)
plot_factors(ground_truth, fig=fig)
axes[-1, -1].legend(("estimate", "ground truth"))
fig.suptitle("Factors before alignment")
fig.tight_layout()

# Permute and align components.
shifted_align(model, ground_truth, permute_U=True)

# Plot factors after alignment.
fig, axes, _ = plot_factors(model)
plot_factors(ground_truth, fig=fig)
axes[-1, -1].legend(("estimate", "ground truth"))
fig.suptitle("Factors after alignment")
fig.tight_layout()
fig.subplots_adjust(top=.92)

# Plot shifts along axis=1.
fig, axes = plt.subplots(rank, rank, sharey=True, sharex=True)
for r1, r2 in itertools.product(range(rank), range(rank)):
    axes[r1, r2].scatter(
        model.v_s[r1],
        ground_truth.v_s[r2],
        color="k", lw=0, s=20,
    )
for r in range(rank):
    axes[r, 0].set_ylabel("true shifts,\ncomponent {}".format(r))
    axes[-1, r].set_xlabel("est shifts,\ncomponent {}".format(r))
axes[0, 0].set_xlim(-max_shift * K, max_shift * K)
axes[0, 0].set_ylim(-max_shift * K, max_shift * K)
fig.suptitle("Recovery of ground truth shifts (axis=1)")
fig.tight_layout()
fig.subplots_adjust(top=.92)

plt.show()
