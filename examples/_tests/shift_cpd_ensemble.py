"""
Shifted tensor decomposition with per-dimension shift
parameters along only axis=0.
"""
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from tensortools.cpwarp import ShiftedCP, fit_shifted_cp, shifted_align
from tensortools.visualization import plot_factors

from scipy.ndimage import gaussian_filter1d
from collections import defaultdict
import itertools
from time import time
from tqdm import tqdm

# Generate random low-rank factors and shifts.
I, J, K = 30, 31, 102  # 100, 101, 102
max_shift = 0.1
rank = 3
npr.seed(1234)

u = npr.rand(rank, I)
v = npr.rand(rank, J)
w = gaussian_filter1d(
    npr.exponential(1.0, size=(rank, K)), 3, axis=-1)

u_s = npr.uniform(-max_shift * K, max_shift * K, (rank, I))

# Store ground truth factors and generate noisy data.
ground_truth = ShiftedCP(u, v, w, u_s, v_s=None, boundary="edge")

noise_scale = 0.02
data = np.maximum(
    0., ground_truth.predict() + noise_scale * npr.randn(I, J, K))

# Fit model.
t0 = time()

ranks = [1, 2, 3, 4, 5]
shifts = np.linspace(1e-6, .2, 6)
repeats = range(3)

model_errors = [[[] for s in shifts] for r in ranks]
prod_iter = itertools.product(
    range(len(ranks)), range(len(shifts)), repeats)

for i, j, k in tqdm(list(prod_iter)):
    model = fit_shifted_cp(
        data, ranks[i], boundary="edge",
        max_shift_axis0=shifts[j],
        max_shift_axis1=None,
        max_iter=100)

    model_errors[i][j].append(model.loss_hist[-1])

plt.figure()
for i in range(len(ranks)):
    plt.scatter(
        np.repeat(shifts, len(repeats)),
        np.concatenate(model_errors[i]),
        lw=0, s=10
    )
    plt.plot(
        shifts, np.min(model_errors[i], axis=1),
        label=ranks[i])

plt.ylabel("Normalized RMSE")
plt.xlabel("maximal shift (fraction of trial duration)")
plt.legend(bbox_to_anchor=[1, 0, 0, 1], title="rank")
plt.tight_layout()
plt.show()
