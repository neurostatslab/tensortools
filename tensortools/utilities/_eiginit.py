import numpy as np
import scipy as sci

from ._sign_flip import _sign_flip

def _eiginit(X, r, mode):
    
    if mode==0: 
        return( np.zeros((X.shape[0],r)) )

    else:
    
        XXt = X.dot(X.T)
            
        XXt = 0.5*(XXt+XXt.T) # ensure symmetry
            
        m = XXt.shape[0]
            
        _, U = sci.linalg.eigh(XXt, eigvals=(m-r, m-1))
            
        U = U[:, ::-1]  # reverse order 
            
        U = _sign_flip(U)
            
        return U
