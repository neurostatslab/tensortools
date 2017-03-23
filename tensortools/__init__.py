from .kruskal import normalize_factors, standardize_factors, align_factors
from .cpfit import cp_als, cp_rand, cp_sparse, cp_batch_fit
from .plots import plot_factors, plot_scree, plot_similarity
from .tensor import coarse_grain_1d, coarse_grain
from tensorly import unfold
from tensorly.tenalg import norm

__version__ = '0.0.1'
