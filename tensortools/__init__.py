# core functions
from .kruskal import normalize_factors, standardize_factors, align_factors
from .cpfit import cp_als, fit_ensemble
from .plots import plot_factors, plot_scree, plot_similarity

# core functions imported from other packages
from tensorly import unfold
from tensorly.tenalg import norm

# useful, non-critical functions
from . import utils

__version__ = '0.0.1'
