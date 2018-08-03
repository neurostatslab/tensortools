"""
Tensortools
"""

from .ensemble import Ensemble
from .tensors import KTensor

from .diagnostics import kruskal_align

from .visualization import plot_factors, plot_objective, plot_similarity

<<<<<<< HEAD
from .data.random_tensor import randn_tensor, rand_tensor
=======
# useful, non-critical functions
from .tensor_utils import unfold
from . import utils
>>>>>>> alex

from .optimize.cp_als import cp_als
from .optimize.cp_opt import cp_opt
from .optimize.ncp_hals import ncp_hals
from .optimize.ncp_bcd import ncp_bcd

from .utils import soft_cluster, tsp_linearize
