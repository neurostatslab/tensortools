import numpy as np
import scipy as sci



def _normalization(X, itr):
        
    if itr == 0:
        normalization = sci.sqrt((X ** 2).sum(axis=0))
    
    else:
        normalization = sci.absolute(X).max(axis=0)
        normalization[normalization < 1] = 1
    
    		
    return normalization

