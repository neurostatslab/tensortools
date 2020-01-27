"""Utilities for summarizing and setting up optimization."""

import numpy as np
from scipy import linalg
from tensortools.tensors import KTensor
import timeit
from tensortools.data.random_tensor import randn_ktensor, rand_ktensor


def _check_cpd_inputs(X, rank):
    """Checks that inputs to optimization function are appropriate.

    Parameters
    ----------
    X : ndarray
        Tensor used for fitting CP decomposition.
    rank : int
        Rank of low rank decomposition.

    Raises
    ------
    ValueError: If inputs are not suited for CP decomposition.
    """
    if X.ndim < 3:
        raise ValueError("Array with X.ndim > 2 expected.")
    if rank <= 0 or not isinstance(rank, int):
        raise ValueError("Rank is invalid.")


def _get_initial_ktensor(init, X, rank, random_state, scale_norm=True):
    """
    Parameters
    ----------
    init : str
        Specifies type of initializations ('randn', 'rand')
    X : ndarray
        Tensor that the decomposition is fit to.
    rank : int
        Rank of decomposition
    random_state : RandomState or int
        Specifies seed for random number generator
    scale_norm : bool
        If True, norm is scaled to match X (default: True)

    Returns
    -------
    U : KTensor
        Initial factor matrices used optimization.
    normX : float
        Frobenious norm of tensor data.
    """
    normX = linalg.norm(X) if scale_norm else None

    if init == 'randn':
        # TODO - match the norm of the initialization to the norm of X.
        U = randn_ktensor(X.shape, rank, norm=normX, random_state=random_state)

    elif init == 'rand':
        # TODO - match the norm of the initialization to the norm of X.
        U = rand_ktensor(X.shape, rank, norm=normX, random_state=random_state)

    elif isinstance(init, KTensor):
        U = init.copy()

    else:
        raise ValueError("Expected 'init' to either be a KTensor or a string "
                         "specifying how to initialize optimization. Valid "
                         "strings are ('randn', 'rand').")

    return U, normX


class FitResult(object):
    """
    Holds result of optimization.

    Attributes
    ----------
    total_time: float
        Number of seconds spent before stopping optimization.
    obj : float
        Objective value of optimization (at current parameters).
    obj_hist : list of floats
        Objective values at each iteration.
    """

    def __init__(self, factors, method, tol=1e-5, verbose=True, max_iter=500,
                 min_iter=1, max_time=np.inf):
        """Initializes FitResult.

        Parameters
        ----------
        factors : KTensor
            Initial guess for tensor decomposition.
        method : str
            Name of optimization method (used for printing).
        tol : float
            Stopping criterion.
        verbose : bool
            Whether to print progress of optimization.
        max_iter : int
            Maximum number of iterations before quitting early.
        min_iter : int
            Minimum number of iterations before stopping due to convergence.
        max_time : float
            Maximum number of seconds before quitting early.
        """
        self.factors = factors
        self.obj = np.inf
        self.obj_hist = []
        self.method = method

        self.tol = tol
        self.verbose = verbose
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.max_time = max_time

        self.iterations = 0
        self.converged = False
        self.t0 = timeit.default_timer()
        self.total_time = None

    @property
    def still_optimizing(self):
        """True unless converged or maximum iterations/time exceeded."""

        # Check if we need to give up on optimizing.
        if (self.iterations > self.max_iter) or (self.time_elapsed() > self.max_time):
            return False

        # Always optimize for at least 'min_iter' iterations.
        elif not hasattr(self, 'improvement') or (self.iterations < self.min_iter):
            return True

        # Check convergence.
        else:
            self.converged = self.improvement < self.tol
            return False if self.converged else True

    def time_elapsed(self):
        return timeit.default_timer() - self.t0

    def update(self, obj):

        # Keep track of iterations.
        self.iterations += 1

        # Compute improvement in objective.
        self.improvement = self.obj - obj
        self.obj = obj
        self.obj_hist.append(obj)

        # If desired, print progress.
        if self.verbose:
            p_args = self.method, self.iterations, self.obj, self.improvement
            s = '{}: iteration {}, objective {}, improvement {}.'
            print(s.format(*p_args))

    def finalize(self):

        # Set final time, final print statement
        self.total_time = self.time_elapsed()

        if self.verbose:
            s = 'Converged after {} iterations, {} seconds. Objective: {}.'
            print(s.format(self.iterations, self.total_time, self.obj))

        return self
