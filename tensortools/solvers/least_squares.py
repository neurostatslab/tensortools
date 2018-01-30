"""Simple least squares solvers
"""
import numpy as np
from ..tensor_utils import unfold, khatri_rao
from .nnls import nnlsm_blockpivot
from numba import jit

def ls_solver(A, B, M=None, l1=None, l2=None, X0=None, nonneg=False):
    """Updates a factor matrix by least-squares.
    
    Note: Does not support regularization

   Parameters
    ----------
    A : ndarray (matrix)
        Transposed Khatri-Rao product of factor matrices
    B : ndarray(matrix)
        Tensor unfolded along mode that is being updated


    mode : int
        Specifies which factor matrix is updated
    M : None or ndarray
        If not None, specifies missing / held-out elements of B.
    X0 : ndarray (matrix)

    Returns
    -------
    new_factor : ndarray
        An updated factor matrix along specified mode
    """

    if l1 is not None or l2 is not None:
        raise NotImplementedError('l1 and l2 regularization is not available yet...')

    # # set up loss function: || X*A - B ||
    # A = khatri_rao(factors, skip_matrix=mode).T
    # B = unfold(tensor, mode)

    # Solvers without missing data
    if M is None:

        # Simple Least Squares
        if nonneg is False:
            return np.linalg.lstsq(A.T, B.T)[0].T

        # Nonnegative Least Squares 
        else:
            # catch singular matrix error, reset result
            # (this should not happen often)
            try:
                X = nnlsm_blockpivot(A.T, B.T, init=X0)[0].T
            except np.linalg.linalg.LinAlgError:
                X = np.random.rand(*X0.shape)

            # prevent a full column of X going to zero
            z  = np.isclose(X, 0).all(axis=0)
            if np.any(z):
                X[:,z] = np.random.rand(X.shape[0], np.sum(z))

            return X

    # Solvers with missing data
    else:
        # Astack is R x R x len(factor[mode])
        Astack = np.matmul(M[:,None,:] * A[None,:,:], A.T[None, :, :])
        # Bstack is len(factors[mode]) x R x 1
        Bstack = np.dot(B * M, A.T)[:,:,None]

        # Simple Least Squares
        if nonneg is False:
            # broadcasted solve, each solved chunk is len(R)
            x = np.linalg.solve(Astack, Bstack)

        # Nonnegative Least Squares
        else:
            x = np.array(_nnls_broadcast(Astack, Bstack))
        
        return x.reshape((B.shape[0], A.shape[0]))

def _nnls_broadcast(Astack, Bstack):
    return [nnlsm_blockpivot(A, b)[0] for A, b in zip(Astack, Bstack)]