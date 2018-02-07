#*************************************************************************
#***        Author: N. Benjamin Erichson <nbe@st-andrews.ac.uk>        ***
#***                              <2016>                               ***
#*************************************************************************    

import numpy as np
import scipy as sci
import timeit

from functools import reduce

from tensortools.tensor_utils import fold, unfold
from tensortools.compress import compress
from tensortools.tensor_utils import khatri_rao
from tensortools.utilities import _normalization, _eiginit, _arrange
from tensortools.tensor_utils import kruskal_to_tensor

#import pyximport; pyximport.install()
from .._hals_update import _hals_update

def _randinit(X, r, mode):
    
    if mode==0: 
        return( np.zeros((X.shape[0], r)) )

    else:
    
        m, n = X.shape
            
        U = sci.maximum(0.0, sci.random.standard_normal((m, r))) 
            
        return U



def ncp_hals(X, r=None, tol=1E-5, maxiter=500, trace=True):
    """
    Nonnegtaive CP Decomposition using the Hierarcial Alternating Least Squares Method.
    
    Given a tensor X, the best rank-R CP model is estimated using the 
    alternating least-squares method.

    
    Parameters
    ----------
    X : array_like or dtensor
        Real tensor `X` with dimensions `(I, J, K)`.
    
    r : int
        `r` denotes the number of components to compute.
        
        
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
    # Init
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    t0 = timeit.default_timer() # start timer
    N = X.ndim


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize components [U_1, U_2, ... U_N] using the eig. decomposition
    # Note that only N-1 components are required for initialization
    # Hence, U_1 is assigned an empty list, i.e., U_1 = []
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #U = [_eiginit(unfold(X, n), r, n) for n in range(N)]    
    U = [_randinit(unfold(X, n), r, n) for n in range(N)]    
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Iterate the ALS algorithm until convergence or maxiter is reached
    # i)   compute the N gram matrices and multiply   
    # ii) Update component U_1, U_2, ... U_N
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for itr in range(maxiter):
        violation = 0.0

        for n in range(N):
            
            # Select all components, but U_n
            components = [U[j] for j in range(N) if j != n]

            # i) compute the N-1 gram matrices 
            grams = [ arr.T.dot(arr) for arr in components ]             
            grams = reduce(sci.multiply, grams, 1.)

            # ii) Update component U_n            
            XUs = unfold(X, n).dot(khatri_rao(components))           

            violation += _hals_update(U[n], grams, XUs )
            
            # iv) normalize U_n to prevent singularities
            #lmbda = _normalization(U[n], itr)
            #U[n] = U[n] / lmbda

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute stopping condition.
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

        if itr == 0:
            violation_init = violation
    
        if violation_init == 0:
            break       
    
        fitchange = violation / violation_init
            
        if trace == True:
            print('Iteration: %s fit: %s, fitchange: %s' %(itr, violation, fitchange))        
    
        if fitchange <= tol:
            break



    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Normalize and sort components and store as ktensor
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    #U, lmbda = _arrange(U, lmbda)

    P = lambda: None  
    P.factors = U
    P.lmbda = None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
    if trace==True: print('Compute time: %s seconds' %(timeit.default_timer()  - t0))
    return P

