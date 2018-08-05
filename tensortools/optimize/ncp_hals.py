"""
CP decomposition by classic alternating least squares (ALS).

Author: N. Benjamin Erichson <erichson@uw.edu>
"""

import numpy as np
import scipy as sci
from scipy import linalg

from tensortools.operations import unfold, khatri_rao
from tensortools.tensors import KTensor
from tensortools.optimize import FitResult, optim_utils

from .._hals_update import _hals_update


def ncp_hals(X, rank, random_state=None, init='rand', **options):
    """
    Fits nonnegtaive CP Decomposition using the Hierarcial Alternating Least
    Squares (HALS) Method.

    Parameters
    ----------
    X : (I_1, ..., I_N) array_like
        A real array with nonnegative entries and ``X.ndim >= 3``.

    rank : integer
        The `rank` sets the number of components to be computed.

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.

    init : str, or KTensor, optional (default ``'rand'``).
        Specifies initial guess for KTensor factor matrices.
        If ``'randn'``, Gaussian random numbers are used to initialize.
        If ``'rand'``, uniform random numbers are used to initialize.
        If KTensor instance, a copy is made to initialize the optimization.

    options : dict, specifying fitting options.

        tol : float, optional (default ``tol=1E-5``)
            Stopping tolerance for reconstruction error.

        max_iter : integer, optional (default ``max_iter = 500``)
            Maximum number of iterations to perform before exiting.

        min_iter : integer, optional (default ``min_iter = 1``)
            Minimum number of iterations to perform before exiting.

        max_time : integer, optional (default ``max_time = np.inf``)
            Maximum computational time before exiting.

        verbose : bool ``{'True', 'False'}``, optional (default ``verbose=True``)
            Display progress.


    Returns
    -------
    result : FitResult instance
        Object which holds the fitted results. It provides the factor matrices
        in form of a KTensor, ``result.factors``.


    Notes
    -----
    This implemenation is using the Hierarcial Alternating Least Squares Method.


    References
    ----------
    Cichocki, Andrzej, and P. H. A. N. Anh-Huy. "Fast local algorithms for
    large scale nonnegative matrix and tensor factorizations."
    IEICE transactions on fundamentals of electronics, communications and
    computer sciences 92.3: 708-721, 2009.

    Examples
    --------


    """

    # Check inputs.
    optim_utils._check_cpd_inputs(X, rank)

    # Initialize problem.
    U, normX = optim_utils._get_initial_ktensor(init, X, rank, random_state)
    result = FitResult(U, 'NCP_HALS', **options)

    # Store problem dimensions.
    normX = linalg.norm(X)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Iterate the HALS algorithm until convergence or maxiter is reached
    # i)   compute the N gram matrices and multiply
    # ii)  Compute Khatri-Rao product
    # iii) Update component U_1, U_2, ... U_N
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    while result.still_optimizing:

        violation = 0.0

        for n in range(X.ndim):

            # Select all components, but U_n
            components = [U[j] for j in range(X.ndim) if j != n]

            # i) compute the N-1 gram matrices
            grams = sci.multiply.reduce([arr.T.dot(arr) for arr in components])

            # ii)  Compute Khatri-Rao product
            kr = khatri_rao(components)
            p = unfold(X, n).dot(kr)

            # iii) Update component U_n
            violation += _hals_update(U[n], grams, p)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the optimization result, checks for convergence.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute objective function
        # grams *= U[X.ndim - 1].T.dot(U[X.ndim - 1])
        # obj = np.sqrt( (sci.sum(grams) - 2 * sci.sum(U[X.ndim - 1] * p) + normX**2)) / normX
        result.update(linalg.norm(X - U.full()) / normX)

    # end optimization loop, return result.
    return result.finalize()
