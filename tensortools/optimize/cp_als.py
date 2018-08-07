"""
CP decomposition by classic alternating least squares (ALS).

Author: N. Benjamin Erichson <erichson@uw.edu> and Alex H. Williams
"""

import numpy as np
import scipy as sci
from scipy import linalg

from tensortools.operations import unfold, khatri_rao
from tensortools.tensors import KTensor
from tensortools.optimize import FitResult, optim_utils


def cp_als(X, rank, random_state=None, init='randn', **options):
    """Fits CP Decomposition using the Alternating Least Squares (ALS).

    Parameters
    ----------
    X : (I_1, ..., I_N) array_like
        A real array with ``X.ndim >= 3``.

    rank : integer
        The `rank` sets the number of components to be computed.

    random_state : integer, ``RandomState``, or ``None``, optional (default ``None``)
        If integer, sets the seed of the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, use the RandomState instance used by ``numpy.random``.

    init : str, or KTensor, optional (default ``'randn'``).
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
    This implemenation uses the Alternating Least Squares Method.


    References
    ----------
    Kolda, T. G. & Bader, B. W.
    "Tensor Decompositions and Applications."
    SIAM Rev. 51 (2009): 455-500
    http://epubs.siam.org/doi/pdf/10.1137/07070111X

    Comon, Pierre & Xavier Luciani & Andre De Almeida.
    "Tensor decompositions, alternating least squares and other tales."
    Journal of chemometrics 23 (2009): 393-405.
    http://onlinelibrary.wiley.com/doi/10.1002/cem.1236/abstract


    Examples
    --------

    ```
    import tensortools as tt
    I, J, K, R = 20, 20, 20, 4
    X = tt.randn_tensor(I, J, K, rank=R)
    tt.cp_als(X, rank=R)
    ```
    """

    # Check inputs.
    optim_utils._check_cpd_inputs(X, rank)

    # Initialize problem.
    U, normX = optim_utils._get_initial_ktensor(init, X, rank, random_state)
    result = FitResult(U, 'CP_ALS', **options)

    # Main optimization loop.
    while result.still_optimizing:

        # Iterate over each tensor mode.
        for n in range(X.ndim):

            # i) Normalize factors to prevent singularities.
            U.rebalance()

            # ii) Compute the N-1 gram matrices.
            components = [U[j] for j in range(X.ndim) if j != n]
            grams = sci.multiply.reduce([sci.dot(u.T, u) for u in components])

            # iii)  Compute Khatri-Rao product.
            kr = khatri_rao(components)

            # iv) Form normal equations and solve via Cholesky
            # c = linalg.cho_factor(grams, overwrite_a=False)
            # p = unfold(X, n).dot(kr)
            # U[n] = linalg.cho_solve(c, p.T, overwrite_b=False).T
            U[n] = linalg.solve(grams, unfold(X, n).dot(kr).T).T

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the optimization result, checks for convergence.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute objective function
        # grams *= U[-1].T.dot(U[-1])
        # obj = np.sqrt(np.sum(grams) - 2*sci.sum(p*U[-1]) + normX**2) / normX
        obj = linalg.norm(U.full() - X) / normX

        # Update result
        result.update(obj)

    # Finalize and return the optimization result.
    return result.finalize()
