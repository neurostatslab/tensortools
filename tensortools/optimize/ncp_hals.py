"""
Nonnegative CP decomposition by Hierarchical alternating least squares (HALS).

Author: N. Benjamin Erichson <erichson@uw.edu>
"""

import numpy as np
import numba

from tensortools.operations import unfold, khatri_rao
from tensortools.tensors import KTensor
from tensortools.optimize import FitResult, optim_utils


def ncp_hals(
        X, rank, mask=None, random_state=None, init='rand',
        skip_modes=[], negative_modes=[], **options):
    """
    Fits nonnegtaive CP Decomposition using the Hierarcial Alternating Least
    Squares (HALS) Method.

    Parameters
    ----------
    X : (I_1, ..., I_N) array_like
        A real array with nonnegative entries and ``X.ndim >= 3``.

    rank : integer
        The `rank` sets the number of components to be computed.

    mask : (I_1, ..., I_N) array_like
        Binary tensor, same shape as X, specifying censored or missing data values
        at locations where (mask == 0) and observed data where (mask == 1).

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

    # Mask missing elements.
    if mask is not None:
        X = np.copy(X)
        X[~mask] = np.mean(X[mask])

    # Check inputs.
    optim_utils._check_cpd_inputs(X, rank)

    # Initialize problem.
    U, normX = optim_utils._get_initial_ktensor(init, X, rank, random_state)
    result = FitResult(U, 'NCP_HALS', **options)

    # Store problem dimensions.
    normX = np.linalg.norm(X)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Iterate the HALS algorithm until convergence or maxiter is reached
    # i)   compute the N gram matrices and multiply
    # ii)  Compute Khatri-Rao product
    # iii) Update component U_1, U_2, ... U_N
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    while result.still_optimizing:

        for n in range(X.ndim):

            # Skip modes that are specified as fixed.
            if n in skip_modes:
                continue

            # Select all components, but U_n
            components = [U[j] for j in range(X.ndim) if j != n]

            # i) compute the N-1 gram matrices
            grams = np.prod([arr.T @ arr for arr in components], axis=0)

            # ii)  Compute Khatri-Rao product
            kr = khatri_rao(components)
            Xmkr = unfold(X, n).dot(kr)

            # iii) Update component U_n
            _hals_update(U[n], grams, Xmkr, n not in negative_modes)

            # iv) Update masked elements.
            if mask is not None:
                pred = U.full()
                X[~mask] = pred[~mask]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the optimization result, checks for convergence.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if mask is None:

            # Determine mode that was fit last.
            n = np.setdiff1d(np.arange(X.ndim), skip_modes).max()

            # Add contribution of last fit factors to gram matrix.
            grams *= U[n].T @ U[n]
            residsq = np.sum(grams) - 2 * np.sum(U[n] * Xmkr) + (normX ** 2)
            result.update(np.sqrt(residsq) / normX)

        else:
            result.update(np.linalg.norm(X - pred) / normX)

    # end optimization loop, return result.
    return result.finalize()


@numba.jit(nopython=True)
def _hals_update(factors, grams, Xmkr, nonneg):

    dim = factors.shape[0]
    rank = factors.shape[1]
    indices = np.arange(rank)

    # Handle special case of rank-1 model.
    if rank == 1:
        if nonneg:
            factors[:] = np.maximum(0.0, Xmkr / grams[0, 0])
        else:
            factors[:] = Xmkr / grams[0, 0]

    # Do a few inner iterations.
    else:
        for itr in range(3):
            for p in range(rank):
                idx = (indices != p)
                Cp = factors[:, idx] @ grams[idx][:, p]
                r = (Xmkr[:, p] - Cp) / np.maximum(grams[p, p], 1e-6)

                if nonneg:
                    factors[:, p] = np.maximum(r, 0.0)
                else:
                    factors[:, p] = r
