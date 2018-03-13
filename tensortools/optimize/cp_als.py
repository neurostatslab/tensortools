"""
CP decomposition by classic alternating least squares (ALS).

Author: N. Benjamin Erichson <erichson@uw.edu> and Alex H. Williams
"""

import numpy as np
import scipy as sci

from tensortools.operations import unfold, khatri_rao
from tensortools.tensors import KTensor
from tensortools.data.random_tensor import randn_tensor
from tensortools.optimize import FitResult


def cp_als(X, rank=None, random_state=None, **options):
    """
    CP Decomposition using the Alternating Least Squares (ALS) Method.

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
        A real array with ``X.ndim >= 3``.

    rank : integer
        The `rank` sets the number of components to be computed.

    random_state : integer, ``RandomState``, or ``None``, optional (default ``None``)
        If integer, sets the seed of the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, use the RandomState instance used by ``numpy.random``.

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
        Object which returns the fited results. It provides the factor matrices
        in form of a Kruskal operator.


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
    tt.optimize.cp_als(X, rank=R)


    """

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Error catching
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if X.ndim < 3:
        raise ValueError("Array with X.ndim > 2 expected.")

    if rank <= 0:
        raise ValueError("Rank is invalid.")

    # store dimensions and norm of X
    N = X.ndim
    normX = sci.linalg.norm(X)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize KTensor
    # Initialize components [U_1, U_2, ... U_N] using random standard normal
    # distributed entries.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # default options
    options.setdefault('init', None)

    if options['init'] is None:
        # TODO - match the norm of the initialization to the norm of X.
        U = randn_tensor(X.shape, rank, ktensor=True,
                         random_state=random_state)

    elif type(options['init']) is not KTensor:
        raise ValueError("Optional parameter 'init' is not a KTensor.")

    else:
        U = options['init']

    # initialize result
    result = FitResult(U, 'CP_ALS', **options)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Iterate the ALS algorithm until convergence or maxiter is reached
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    while result.converged is False:

        for n in range(N):

            # i) Normalize factors to prevent singularities
            U.rebalance()

            # ii) compute the N-1 gram matrices
            components = [U[j] for j in range(N) if j != n]
            grams = sci.multiply.reduce([sci.dot(u.T, u) for u in components])

            # iii)  Compute Khatri-Rao product
            kr = khatri_rao(components)

            # iv) Form normal equations and solve via Cholesky
            # c = sci.linalg.cho_factor(grams, overwrite_a=False)
            # p = unfold(X, n).dot(kr)
            # U[n] = sci.linalg.cho_solve(c, p.T, overwrite_b=False).T
            U[n] = sci.linalg.solve(grams, unfold(X, n).dot(kr).T).T

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the optimization result, checks for convergence.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute objective function
        # grams *= U[-1].T.dot(U[-1])
        # obj = np.sqrt(np.sum(grams) - 2*sci.sum(p*U[-1]) + normX**2) / normX
        obj = sci.linalg.norm(U.full() - X) / normX

        # Update result
        result.update(obj)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Prepares final version of the optimization result.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    result.finalize()

    return result
