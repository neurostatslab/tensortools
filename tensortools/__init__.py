# main fitting function
from .ensemble import fit_ensemble

# solvers for fitting a single tensor decomposition
from .cpdirect import cp_direct
from .cprand import cp_rand

# visualization tools
from .plots import plot_factors, plot_error, plot_similarity

# functions for manipulating tensor decompositions
from .kruskal import normalize_factors, standardize_factors, align_factors

# useful, non-critical functions
from . import utils

__version__ = '0.0.1'
