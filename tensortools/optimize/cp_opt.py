"""
CP decomposition by classic alternating least squares (ALS).

Author: N. Benjamin Erichson <erichson@uw.edu> and Alex H. Williams
"""

import numpy as np
import scipy as sci

from scipy.optimize import minimize

from tensortools.operations import unfold, khatri_rao
from tensortools.tensors import KTensor
from tensortools.data.random_tensor import randn_tensor
from tensortools.optimize import FitResult


def cp_opt(X, rank=None, method='Newton-CG', random_state=None, **options):
    """
    CP Decomposition using optimization methods.

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

    method : str or callable, optional
            Type of solver.  Should be one of
                - 'CG'          :ref:`(see here) <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cg.html#optimize-minimize-cg>`
                - 'Newton-CG'   :ref:`(see here) <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-newtoncg.html#optimize-minimize-newtoncg>`
                - 'TNC'         :ref:`(see here) <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-tnc.html#optimize-minimize-tnc>`
                - 'BFGS'        :ref:`(see here) <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html#optimize-minimize-bfgs>`

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
        Object which returns the fited results. It provides the factor matrices
        in form of a Kruskal operator.


    Notes
    -----
    Experimental - This implemenation is generally slower than CP_ALS


    References
    ----------



    Examples
    --------



    """

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Error catching
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if X.ndim < 3:
        raise ValueError("Array with X.ndim > 2 expected.")

    if rank <= 0:
        raise ValueError("Rank is invalid.")

    # N-way array
    N = X.ndim

    # Norm of input array
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
        U = randn_tensor(X.shape, rank=rank, ktensor=True,
                         random_state=random_state)
        U = [U[n] / sci.linalg.norm(U[n]) * normX**(1.0/N) for n in range(N)]

    elif type(options['init']) is not KTensor:
        raise ValueError("Optional parameter 'init' is not a KTensor.")

    else:
        U = options['init']

    # initialize result
    result = FitResult(U, 'CP_OPT', **options)
    options = {'disp': result.verbose, 'maxiter': result.max_iter}

    U = sci.asarray(U)
    U_shape = U.shape

    success = False
    while success is False:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Optimize until convergence or maxiter is reached
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        res = minimize(cp_opt_fun, np.ravel(U),
                       args=(X, normX, U_shape, result), method=method,
                       jac=True, hess=None, hessp=None, bounds=None,
                       constraints=(), tol=result.tol, callback=None,
                       options=options)

        success = res.success

        if success is False:
            U = randn_tensor(X.shape, rank=rank, ktensor=True,
                             random_state=random_state)
            U = [U[n] / sci.linalg.norm(U[n]) * normX**(1.0/N) for n in range(N)]
            U = sci.asarray(U)

            result.iterations = 0
            result.obj_hist = []

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Prepares final version of the optimization result.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    U = np.reshape(res.x, U_shape)

    result.factors = KTensor(U)

    result.finalize()

    return result


def cp_opt_fun(U, X, normX, Ushape, result):

    # Dimensions of input tensor
    N = X.ndim

    # Reshape
    U = sci.reshape(U, Ushape)

    #U = KTensor(U)

    # i) Normalize factors to prevent singularities
    #U.rebalance()

    # Calculate gradient and F2
    grad = []
    for n in range(N):

        # ii) compute the N-1 gram matrices
        components = [U[j] for j in range(N) if j != n]
        grams = sci.multiply.reduce([sci.dot(u.T, u) for u in components])

        # iii)  Compute Khatri-Rao product
        kr = khatri_rao(components)

        # iv) Form normal equations
        p = unfold(X, n).dot(kr)

        # iv) Append gradient
        grad.append(-p + U[n].dot(grams))

    # Compute objective function
    grams *= U[-1].T.dot(U[-1])
    obj = np.sqrt(np.sum(grams) - 2*sci.sum(p*U[-1]) + normX**2)  # / normX

    # Update result
    result.update(obj / normX)

    return(obj, np.ravel(grad))
