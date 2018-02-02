import numpy as np
import scipy as sci

def _sign_flip(A):
    """
    Flip the signs of A so that largest absolute value is positive.
    """
    signs = sci.sign(A[sci.argmax(sci.absolute(A), axis=0), list(range(A.shape[1]))])
    return signs * A

