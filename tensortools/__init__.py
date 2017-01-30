from .kruskal import normalize_factors, standardize_factors, align_factors
from .cpfit import cp_als, cp_rand, cp_mixrand, cp_batch_fit
from .plots import plot_factors, plot_scree, plot_decode, plot_similarity
from .tensor import coarse_grain_1d, coarse_grain

__version__ = '0.0.1'
