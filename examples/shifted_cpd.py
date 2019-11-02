import numpy as np
import numpy.random as npr
from tensortools.cpwarp import ShiftedCP, fit_shifted_cp
from scipy.ndimage import gaussian_filter1d

I, J, K = 7, 8, 20
rank = 3

u = npr.exponential(1.0, size=(rank, I))
v = npr.exponential(1.0, size=(rank, J))
w = gaussian_filter1d(
    npr.exponential(1.0, size=(rank, K)), 0.5, axis=-1)

u_s = npr.uniform(-1.5, 1.5, (rank, I))
v_s = npr.uniform(-1.5, 1.5, (rank, J))

ground_truth = ShiftedCP(u, v, w, u_s, v_s, boundary="edge")

noise_scale = 0.1
data = np.maximum(
    0., ground_truth.predict() + noise_scale * npr.randn(I, J, K))

model = fit_shifted_cp(data, rank, boundary="edge", max_shift=0.1)
