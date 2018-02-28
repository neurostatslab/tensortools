"""
CP decomposition by classic alternating least squares (ALS).

Author: N. Benjamin Erichson <erichson@uw.edu> 
"""

import numpy as np
import scipy as sci

from tensortools.operations import unfold, khatri_rao
from tensortools.tensors import Ktensor
from tensortools.data.random_tensor import rand_tensor
from tensortools.optimize import FitResult


#import pyximport; pyximport.install()
from .._hals_update import _hals_update




def ncp_hals(X, rank=None, random_state=None, **options):
    """
    Nonnegtaive CP Decomposition using the Hierarcial Alternating Least Squares
    (HALS) Method.
    
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
            
        trace : bool ``{'True', 'False'}``, optional (default ``trace=True``)
            Display progress.


    Returns
    -------
    P : FitResult object
        Object which returens the fited results. It provides the factor matrices
        in form of a Kruskal operator. 

    
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

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Error catching
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
    
    if X.ndim < 3:
        raise ValueError("Array with ndim > 2 expected.")

    if rank is None:
        raise ValueError("Rank 'rank' not given.")

    if rank < 0 or rank > np.min(X.shape):
        raise ValueError("Rank 'rank' is invalid.")
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize Ktensor
    
    # Initialize components [U_1, U_2, ... U_N] using random standard normal 
    # distributed entries. 
    # Note that only N-1 components are required for initialization
    # Hence, U_1 should be assigned as an empty list, i.e., U_1 = []    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # default options
    options.setdefault('init', None)    
    
    
    if options['init'] is None:
        U = rand_tensor(X.shape, rank=rank, ktensor=True, random_state=random_state)
       
    elif type(options['init']) is not Ktensor:
        raise ValueError("Optional parameter 'init' is not a Ktensor.")
    
    else:
        U = options['init']
    
 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Init
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    result = FitResult(X, U, 'NCP_HALS', **options)    
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Iterate the HALS algorithm until convergence or maxiter is reached
    # i)   compute the N gram matrices and multiply   
    # ii)  Compute Khatri-Rao product
    # iii) Update component U_1, U_2, ... U_N
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    normX = sci.linalg.norm(X)
    
    
    while result.converged == False:
        violation = 0.0

        for n in range(X.ndim):
            
            # Select all components, but U_n
            components = [U[j] for j in range(X.ndim) if j != n]

            # i) compute the N-1 gram matrices 
            grams = sci.multiply.reduce([ arr.T.dot(arr) for arr in components ])

            # ii)  Compute Khatri-Rao product
            kr = khatri_rao(components)    
            p =  unfold(X, n).dot( kr )

            # iii) Update component U_n
            violation += _hals_update(U[n], grams, p )
            

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute stopping condition.
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

#        if itr == 0:
#            violation_init = violation
#    
#        if violation_init == 0:
#            break       
#    
#        fitchange = violation / violation_init
#            
#        if trace == True:
#            print('Iteration: %s fit: %s, fitchange: %s' %(itr, violation, fitchange))        
#    
#        if fitchange <= tol:
#            break


        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the optimization result, checks for convergence.
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute objective function
        grams *= U[X.ndim - 1].T.dot(U[X.ndim - 1])        
        obj = np.sqrt( (sci.sum(grams) - 2 * sci.sum(U[X.ndim - 1] * p) + normX**2)) / normX

        
        # Update
        result.update2(obj)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Prepares final version of the optimization result.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    result.finalize(X)

    return result    
    
    