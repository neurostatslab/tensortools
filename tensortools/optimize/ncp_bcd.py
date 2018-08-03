"""
CP decomposition by classic alternating least squares (ALS).

Author: N. Benjamin Erichson <erichson@uw.edu> and Alex H. Williams
"""

import numpy as np
import scipy as sci

from tensortools.operations import unfold, khatri_rao
from tensortools.tensors import KTensor
from tensortools.data.random_tensor import rand_tensor
from tensortools.optimize import FitResult, optim_utils


def ncp_bcd(X, rank=None, random_state=None, **options):
    """
    Nonnegative CP Decomposition using the Block Coordinate Descent (BCD) Method.

    The CP (CANDECOMP/PARAFAC) method  is a decomposition for higher order
    arrays (tensors). The CP decomposition can be seen as a generalization
    of PCA, yet there are some important conceptual differences: (a) the CP
    decomposition allows to extract pure spectra from multi-way spectral data;
    (b) the data do not need to be unfolded. Hence, the resulting
    factors are easier to interpret and more robust to noise.

    When `X` is a N-way array, it is factorized as ``[U_1, ...., U_N]``,
    where `U_i` are 2D arrays of rank R.


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
    P : FitResult object
        Object which returens the fited results. It provides the factor matrices
        in form of a Kruskal operator.


    Notes
    -----
    This implemenation is using the Block Coordinate Descent Method.


    References
    ----------



    Examples
    --------
    Xu, Yangyang, and Wotao Yin. "A block coordinate descent method for
    regularized multiconvex optimization with applications to
    negative tensor factorization and completion."
    SIAM Journal on imaging sciences 6.3 (2013): 1758-1789.


    """

    # Check inputs.
    optim_utils._check_cpd_inputs(X, rank)

    # Store norm of X for computing objective function.
    normX = sci.linalg.norm(X)

    # Initialize problem.
    U = optim_utils._get_initial_ktensor(init, X, rank, random_state)
    result = FitResult(U, 'CP_ALS', **options)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Block coordinate descent
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

            # Select all components, but U_n
            components = [U[j] for j in range(N) if j != n]

            # i) compute the N-1 gram matrices
            grams = sci.multiply.reduce([arr.T.dot(arr) for arr in components])

            # Update gradient Lipschnitz constant
            L0 = L  # Lipschitz constants
            L[n] = sci.linalg.norm(grams, 2)

            # ii)  Compute Khatri-Rao product
            kr = khatri_rao(components)
            p = unfold(X, n).dot( kr )

            # Compute Gradient.
            grad = Um[n] .dot(grams) - p

            # Enforce nonnegativity (project onto nonnegative orthant).
            U[n] = sci.maximum(0.0, Um[n] - grad / L[n])

        # Compute objective function and update optimization result.
        # grams *= U[X.ndim - 1].T.dot(U[X.ndim - 1])
        # obj = np.sqrt(sci.sum(grams) - 2 * sci.sum(U[X.ndim - 1] * p) + normX**2) / normX
        obj = sci.linalg.norm(X - U.full()) / normX
        result.update(obj)

        # Correction and extrapolation.
        grams *= U[N - 1].T.dot(U[N - 1])
        obj_bcd = 0.5 * (sci.sum(grams) - 2 * sci.sum(U[N-1] * p) + normX**2 )

        extraw = (1 + sci.sqrt(1 + 4 * extraw_old**2)) / 2.0

        if obj_bcd >= obj_bcd_old:
            # restore previous A to make the objective nonincreasing
            Um = sci.copy(U_old)

        else:
            # apply extrapolation
            w = (extraw_old - 1.0) / extraw # Extrapolation weight
            for n in range(N):
                weights_U[n] = min(w, 1.0 * sci.sqrt( L0[n] / L[n] )) # choose smaller weights for convergence
                Um[n] = U[n] + weights_U[n] * (U[n] - U_old[n]) # extrapolation

    # Finalize and return the optimization result.
    return result.finalize()
