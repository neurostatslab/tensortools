#*************************************************************************
#***                 Author: N. Benjamin Erichson                      ***
#***                              <2017>                               ***
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


def ncp_bcd(X, r=None, c=True, p=10, q=1, tol=1E-5, maxiter=500, trace=True):
    """
    Randomized Nonnegative CP Decomposition using the Block Coordinate Descent Method.
    
    Given a tensor X, the best rank-R CP model is estimated using the 
    block coordinate descent method.
    If `c=True` the input tensor is compressed using the randomized 
    QB-decomposition.
    
    Parameters
    ----------
    X : array_like or dtensor
        Real tensor `X` with dimensions `(I, J, K)`.
    
    r : int
        `r` denotes the number of components to compute.

    c : bool `{'True', 'False'}`, optional (default `c=True`)
        Whether or not to compress the tensor.         

    p : int, optional (default `p=10`)
        `p` sets the oversampling parameter.

    q : int, optional (default `q=2`)
        `q` sets the number of power iterations.        
        
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
    The algorithm is based on the following code:
    http://www.caam.rice.edu/%7Eoptimization/bcu/ncp/ncp.html
    
    
    References
    ----------
    Xu, Yangyang, and Wotao Yin. "A block coordinate descent method for 
    regularized multiconvex optimization with applications to 	
    negative tensor factorization and completion." 
    SIAM Journal on imaging sciences 6.3 (2013): 1758-1789.



    """
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Error catching
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~       
    if X.ndim < 3:
        raise ValueError("Array with ndim > 2 expected.")

    if r is None:
        raise ValueError("Rank 'r' is not specified.")

    if r < 0 or r > np.min(X.shape):
        raise ValueError("Rank 'r' is not valid.")
   
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Init
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    t0 = timeit.default_timer() # start timer
    hist_obj = [] 
    hist_relerr1 = []
    hist_relerr2 = []
    nstall = 0

    rdiff = 1 # Rel. difference between compressed and full tensor
    N = X.ndim # Dimensions of input array
    Xnrm = sci.linalg.norm(X) # norm

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compress Tensor
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    if trace==True: print('Shape of Tensor: ', X.shape ) 
        
    if c==True:    
        Q , X = compress(X, r=r, p=p, q=q, NN=True)

        # Compare fro norm between the compressed and full tensor  
        cXnrm = sci.linalg.norm(X) # Fro norm
        rdiff =  cXnrm/Xnrm   
        if trace==True: print('Shape of cTensor: ', X.shape ) 
        if trace==True: print('Fro. norm of Tensor: %s,  cTensor: %s' %(Xnrm, cXnrm) )
        if trace==True: print('Rel. difference of the Fro. norm: %s' %( round(1-rdiff,2) ))
        Xnrm = cXnrm     
        hU = [None] * N
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize [A,B,C] using the higher order SVD
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Rand init
    nWay = X.shape # Shape of input array
    U = [np.random.uniform(low=0.0, high=1.0, size=([nWay[n],r])) for n in range(N)]    
    # U = [sci.maximum(0, sci.random.standard_normal([nWay[n],r])) for n in range(N)]    

    # Normalize
    U = [U[n] / sci.linalg.norm(U[n]) * Xnrm**(1.0/N) for n in range(N)]    

    # Cache its squre
    Usq = [U[n].T.dot(U[n]) for n in range(N)]    
    
    Um = np.copy(U) # Extrapolations of compoenents    
                
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Block coordinate descent 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    obj = 0.5 * Xnrm**2 # Initial objective value                       
    extraw = 1 # Used for extrapolation weight update
    rw = 1      # Initial extrapolation weight
    wU = np.ones(N) # Extrapolation weights
    L = np.ones(N) # Lipschitz constants
    
    for itr in range(maxiter):
        obj0 = obj # Old objective value
        U0 = sci.copy(U) # Old updates
        #U0 = U
        extraw0 = extraw         
        

        for n in range(N):
            
            # i) Compute the N-1 Bsq matrix 
            Bsq = reduce(sci.multiply, [Usq[j] for j in range(N) if j != n], 1.0)  
            
                        
            # Update gradient Lipschnitz constant
            L0 = L # Lipschitz constants
            L[n] = sci.linalg.norm(Bsq, 2)

            # Compute unfolded tensor times Khatri-Rao product
            MB = sci.dot(unfold(X, n), khatri_rao(U, skip_matrix=n, reverse=False))
             
            # Compute gradient
            Gn = Um[n].dot(Bsq) - MB 
            
            if c==False:       
                U[n] = np.maximum(0, Um[n] - Gn/L[n])  
            
            elif c==True:
                hU[n] = np.array(Q[n]).dot(Um[n])
                hGn = np.array(Q[n]).dot(Gn)
                hU[n] = np.maximum(0, hU[n] - hGn/L[n])  
                U[n] = np.array(Q[n].T).dot(hU[n])
            
            
            Usq[n] =  U[n].T.dot(U[n])              


        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute fit of the approximation,

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
        
        # Update objective function    
        obj =  0.5 * (sci.sum(sci.sum(Usq[N-1] * Bsq)) - 2 * sci.sum(sci.sum(U[N-1] * MB)) + Xnrm**2 )  

        # Relative objective change
        relerr1 = sci.absolute(obj - obj0) / (obj0 + 1.0) 
        
        # Relative residual 
        relerr2 = (2.0 * obj)**0.5 / Xnrm       
        
        # History 
        hist_obj.append( obj )        
        hist_relerr1.append( relerr1 )
        hist_relerr2.append( relerr2 )

        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Correction and extrapolation
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    
        extraw = (1 + sci.sqrt(1 + 4 * extraw0**2)) / 2.0
        
        if obj >= obj0:
            # restore previous A to make the objective nonincreasing
            Um = sci.copy(U0)

        else: 
            # apply extrapolation
            w = (extraw0 - 1.0) / extraw # Extrapolation weight
            for n in range(N):
                wU[n] = min(w, rw * sci.sqrt( L0[n] / L[n] )) # choose smaller weights for convergence
                Um[n] = U[n] + wU[n] * (U[n] - U0[n]) # extrapolation
                #print(wU[n])
       
        
        
        if trace==True:
            print('Iteration: %s obj: %s, rel. error: %s' %(itr, obj0, relerr2))

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check stopping criterion
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    
        crit = relerr1 < tol
        
        if crit==True:
            nstall += 1
        
        else:
            nstall = 0
        
        if nstall>=3 or relerr2 < tol:
            break
        

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Recover full-state components 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

    if c==True:
        for n in range(len(U)):
              U[n] = hU[n] 
              
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Store
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    
    P = lambda: None  
    P.factors = U
    P.hist_obj = hist_obj
    P.hist_relerr1 = hist_relerr1
    P.hist_relresid = hist_relerr2
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
    if trace==True: print('Compute time: %s seconds' %(timeit.default_timer()  - t0))
    
    return P

