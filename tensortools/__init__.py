# core functions ported from tensorly
from .tensor_utils import unfold, norm

# core functions
from .kruskal import normalize_factors, standardize_factors, align_factors
from .cpfit import cp_als, fit_ensemble
from .plots import plot_factors, plot_error, plot_similarity
from .crossval import cp_crossval

# useful, non-critical functions
from . import utils

__version__ = '0.0.1'
