import tensortools as tt
import numpy as np

# make factors
N, R, ndim = 50, 2, 3
true_factors = [np.random.randn(N,R) for _ in range(ndim)]

# simulate data
data = np.einsum('ir,jr,kr->ijk', *true_factors)

est_factors, info = tt.cpfit(data, R, exact=False, sample_frac=0.1, verbose=True, print_every=10)

score = tt.align_kruskal(true_factors, est_factors)[2]
print(score)
