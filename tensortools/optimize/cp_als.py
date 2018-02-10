"""
CP decomposition by classic alternating least squares (ALS).
"""

import numpy as np
import scipy as sci

from functools import reduce

from tensortools.operations import unfold, khatri_rao

from tensortools.tensor import Ktensor
from .optimize import FitResult

def cp_als(X, r=None, options=dict()):
    """
    Randomized CP Decomposition using the Alternating Least Squares Method.
    
    Given a tensor X, the best rank-R CP model is estimated using the 
    alternating least-squares method.
    If `c=True` the input tensor is compressed using the randomized 
    QB-decomposition.

    
    Parameters
    ----------
    X : array_like or dtensor
        Real tensor `X` with dimensions `(I, J, K)`.
    
    r : int
        `r` denotes the number of components to compute.     
        
    options : dict, specifying fitting options.

        tol : float, optional (default `tol=1E-5`)
            Stopping tolerance for reconstruction error.
            
        maxiter : int, optional (default `maxiter=500`)
            Maximum number of iterations to perform before exiting.

        trace : bool `{'True', 'False'}`, optional (default `trace=True`)
            Display progress.


    Returns
    -------
    P : ktensor
        Tensor stored in decomposed form as a Kruskal operator.

    
    Notes
    -----  
    
    
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

    """

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Error catching
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
    if X.ndim < 3:
        raise ValueError("Array with ndim > 2 expected.")

    if r is None:
        raise ValueError("Rank 'r' not given.")

    if r < 0 or r > np.min(X.shape):
        raise ValueError("Rank 'r' is invalid.")
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize Ktensor
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if init_factors is None:
        U = Ktensor(X.shape, init='randn')
    elif type(init_factors) is not Ktensor:
        raise ValueError("Parameter 'init_factors' is not a Ktensor.")
    else:
        U = init_factors

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Init
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    result = FitResult(X, U, 'CP_ALS', **options)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Iterate the ALS algorithm until convergence or maxiter is reached
    # i)   compute the N gram matrices and multiply   
    # ii)  Compute Khatri-Rao Pseudoinverse
    # iii) Update component U_1, U_2, ... U_N
    # iv) Normalize columns of U_1, U_2, ... U_N to length 1
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    while result.converged == False:

        for n in range(X.ndim):
            
            # Select all components, but U_n
            components = [U[j] for j in range(X.ndim) if j != n]

            # i) compute the N-1 gram matrices 
            grams = [ arr.T.dot(arr) for arr in components ]             

            # ii)  Compute Khatri-Rao Pseudoinverse
            p1 = khatri_rao(components)
            p2 = sci.linalg.pinv(reduce(sci.multiply, grams, 1.))

            # iii) Update component U_n            
            U[n] = unfold(X, n).dot( p1.dot(p2) )           

            # iv) normalize U_n to prevent singularities
            U.rebalance()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the optimization result, checks for convergence.
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        result.update(U)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Prepares final version of the optimization result.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    result.finalize()

    return result
