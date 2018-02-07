# main fitting function
from .cp_decomposition.ensemble import fit_ensemble

# solvers for fitting a single tensor decomposition
from .cp_decomposition.cpdirect import cp_direct
from .cp_decomposition.cprand import cp_rand
from .cp_decomposition.cp_als import cp_als
from .cp_decomposition.cp_opt import cp_opt
from .cp_decomposition.ncp_bcd import ncp_bcd
from .cp_decomposition.ncp_hals import ncp_hals

# randomized tensor QB decomposition
from .compress import compress

# visualization tools
from .plots import plot_factors, plot_error, plot_similarity

# functions for manipulating tensor decompositions
from .kruskal import normalize_factors, standardize_factors, align_factors

# useful, non-critical functions
from . import utils
