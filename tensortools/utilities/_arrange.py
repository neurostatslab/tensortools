import numpy as np
import scipy as sci

def _arrange(factors, lmbda):
    # Normalize components   
    for n in range(len(factors)):
        norm = np.sqrt((factors[n] ** 2).sum(axis=0))
        factors[n] /= norm
        lmbda *= norm
    
    # Sort
    sort_idx = np.argsort(lmbda)[::-1]
    lmbda = lmbda[sort_idx]
    factors = [arr[ : , sort_idx] for arr in factors]
    
    # Flip signs
    # X = [_sign_flip(arr) for arr in X]

    return( factors, lmbda )
