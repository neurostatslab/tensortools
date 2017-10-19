import numpy as np
from .cpfit import cp_solver

def cp_crossval(tensor, rank, p_holdout=0.1, **kwargs):
    """Helper function for performing cross-validation.
    """

    # check inputs
    if 'M' in kwargs:
        raise ValueError('M cannot be specified as an argument to cp_crossval. Call cp_solver directly.')
    
    # create a random held out subset of the tensor
    M = np.random.rand(*tensor.shape) > p_holdout

    # fit CP decomposition
    factors, info = cp_solver(tensor, rank, M=M, **kwargs)

    # compute train and test error
    resid = tensor - np.einsum('ir,jr,kr->ijk', *factors)
    train_err = norm(resid[M], 2) / norm(tensor[M], 2)
    test_err = norm(resid[~M], 2) / norm(tensor[~M], 2)

    return train_err, test_err, factors, info

