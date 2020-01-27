"""
CP decomposition by classic alternating least squares (ALS).

Author: N. Benjamin Erichson <erichson@uw.edu> and Alex H. Williams
"""

import numpy as np

from tensortools.operations import unfold, khatri_rao
from tensortools.tensors import KTensor
from tensortools.optimize import FitResult, optim_utils


def ncp_bcd(
        X, rank, random_state=None, init='rand',
        skip_modes=[], negative_modes=[], **options):
    """
    Fits nonnegative CP Decomposition using the Block Coordinate Descent (BCD)
    Method.

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

    skip_modes : iterable, optional (default ``[]``).
        Specifies modes of the tensor that are not fit. This can be
        used to fix certain factor matrices that have been previously
        fit.

    negative_modes : iterable, optional (default ``[]``).
        Specifies modes of the tensor whose factors are not constrained
        to be nonnegative.

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
    This implemenation is using the Block Coordinate Descent Method.


    References
    ----------
    Xu, Yangyang, and Wotao Yin. "A block coordinate descent method for
    regularized multiconvex optimization with applications to
    negative tensor factorization and completion."
    SIAM Journal on imaging sciences 6.3 (2013): 1758-1789.


    Examples
    --------

    """

    # Check inputs.
    optim_utils._check_cpd_inputs(X, rank)

    # Store norm of X for computing objective function.
    N = X.ndim

    # Initialize problem.
    U, normX = optim_utils._get_initial_ktensor(init, X, rank, random_state)
    result = FitResult(U, 'NCP_BCD', **options)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Block coordinate descent
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Um = U.copy()  # Extrapolations of compoenents
    extraw = 1  # Used for extrapolation weight update
    weights_U = np.ones(N)  # Extrapolation weights
    L = np.ones(N)  # Lipschitz constants
    obj_bcd = 0.5 * normX**2  # Initial objective value

    # Main optimization loop.
    while result.still_optimizing:
        obj_bcd_old = obj_bcd  # Old objective value
        U_old = U.copy()
        extraw_old = extraw

        for n in range(N):

            # Skip modes that are specified as fixed.
            if n in skip_modes:
                continue

            # Select all components, but U_n
            components = [U[j] for j in range(N) if j != n]

            # i) compute the N-1 gram matrices
            grams = np.prod([arr.T.dot(arr) for arr in components], axis=0)

            # Update gradient Lipschnitz constant
            L0 = L  # Lipschitz constants
            L[n] = np.linalg.norm(grams, 2)

            # ii)  Compute Khatri-Rao product
            kr = khatri_rao(components)
            p = unfold(X, n).dot(kr)

            # Compute Gradient.
            grad = Um[n].dot(grams) - p

            # Enforce nonnegativity (project onto nonnegative orthant).
            U[n] = Um[n] - grad / L[n]
            if n not in negative_modes:
                U[n] = np.maximum(0.0, U[n])

        # Compute objective function and update optimization result.
        # grams *= U[X.ndim - 1].T.dot(U[X.ndim - 1])
        # obj = np.sqrt(np.sum(grams) - 2 * np.sum(U[X.ndim - 1] * p) + normX**2) / normX
        obj = np.linalg.norm(X - U.full()) / normX
        result.update(obj)

        # Correction and extrapolation.
        n = np.setdiff1d(np.arange(X.ndim), skip_modes).max()
        grams *= U[n].T.dot(U[n])
        obj_bcd = 0.5 * (np.sum(grams) - 2 * np.sum(U[n] * p) + normX**2)

        extraw = (1 + np.sqrt(1 + 4 * extraw_old**2)) / 2.0

        if obj_bcd >= obj_bcd_old:
            # restore previous A to make the objective nonincreasing
            Um = U_old

        else:
            # apply extrapolation
            w = (extraw_old - 1.0) / extraw  # Extrapolation weight
            for n in range(N):
                if n not in skip_modes:
                    weights_U[n] = min(w, 1.0 * np.sqrt(L0[n] / L[n]))  # choose smaller weights for convergence
                    Um[n] = U[n] + weights_U[n] * (U[n] - U_old[n])  # extrapolation

    # Finalize and return the optimization result.
    return result.finalize()
