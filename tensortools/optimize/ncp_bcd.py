"""
CP decomposition by classic alternating least squares (ALS).

Author: N. Benjamin Erichson <erichson@uw.edu> and Alex H. Williams
"""

import numpy as np
import scipy as sci

from tensortools.operations import unfold, khatri_rao
from tensortools.tensors import KTensor
from tensortools.data.random_tensor import rand_tensor
from tensortools.optimize import FitResult


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
    if X.ndim < 3:
        raise ValueError("Array with X.ndim > 2 expected.")
    if rank is None:
        raise ValueError("Rank is not specified.")
    if rank < 0:
        raise ValueError("Rank is invalid.")

    # Store tensor order/dimension and norm.
    N = X.ndim
    normX = sci.linalg.norm(X)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize KTensor

    # Initialize components [U_1, U_2, ... U_N] using random standard normal
    # distributed entries.
    # Note that only N-1 components are required for initialization
    # Hence, U_1 should be assigned as an empty list, i.e., U_1 = []
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # default options
    options.setdefault('init', None)

    if options['init'] is None:
        # TODO - match the norm of the initialization to the norm of X.
        U = rand_tensor(X.shape, rank=rank, ktensor=True, random_state=random_state)
        #U = KTensor([U[n] / sci.linalg.norm(U[n]) * normX**(1.0 / N ) for n in range(N)])


    elif type(options['init']) is not KTensor:
        raise ValueError("Optional parameter 'init' is not a KTensor.")

    else:
        U = options['init']

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Init
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    result = FitResult(U, 'NCP_BCD', **options)




    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Block coordinate descent
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Um = sci.copy(U.factors) # Extrapolations of compoenents
    #Um = KTensor(Um)

    extraw = 1 # Used for extrapolation weight update
    weights_U = np.ones(N) # Extrapolation weights
    L = np.ones(N) # Lipschitz constants
    obj_bcd = 0.5 * normX**2 # Initial objective value


    while result.converged == False:
        obj_bcd_old = obj_bcd # Old objective value
        U_old = sci.copy(U.factors) # Old updates
        extraw_old = extraw


        for n in range(N):

            # Select all components, but U_n
            components = [U[j] for j in range(N) if j != n]

            # i) compute the N-1 gram matrices
            grams = sci.multiply.reduce([ arr.T.dot(arr) for arr in components ])

            # Update gradient Lipschnitz constant
            L0 = L # Lipschitz constants
            L[n] = sci.linalg.norm(grams, 2)


            # ii)  Compute Khatri-Rao product
            kr = khatri_rao(components)
            p = unfold(X, n).dot( kr )

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute Gradient
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            grad = Um[n] .dot(grams) - p

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Maximum operator to enforce nonnegativity
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            U[n] = sci.maximum(0.0, Um[n] - grad / L[n])

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the optimization result, checks for convergence.
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Compute objective function
        #grams *= U[X.ndim - 1].T.dot(U[X.ndim - 1])
        #obj = np.sqrt(sci.sum(grams) - 2 * sci.sum(U[X.ndim - 1] * p) + normX**2) / normX
        obj = sci.linalg.norm(X - U.full()) / normX

        # Update
        result.update(obj)


        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Correction and extrapolation
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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




    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Prepares final version of the optimization result.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    result.finalize()

    return result
