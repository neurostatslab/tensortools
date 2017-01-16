import numpy as np
from tensorly.kruskal import kruskal_to_tensor
from tensorly.tenalg import norm
from tensortools.cpfit import _compute_squared_recon_error_naive, _compute_squared_recon_error

# make factors
dims = [20, 30, 40]
ndim = len(dims)
rank = 5
factors = [np.random.randn(n,rank) for n in dims]

# make data
tensor = kruskal_to_tensor(factors)
norm_tensor = norm(tensor, 2)

err1 = _compute_squared_recon_error_naive(tensor, factors, norm_tensor)
err2 = _compute_squared_recon_error(tensor, factors, norm_tensor)

f2 = [np.random.randn(n,rank) for n in dims]

err3 = _compute_squared_recon_error_naive(tensor, f2, norm_tensor)
err4 = _compute_squared_recon_error(tensor, f2, norm_tensor)

assert(np.abs(err1 - err2) < 1e-6)
assert(np.abs(err3 - err4) < 1e-6)
