"""Simple least squares solvers
"""
import numpy as np
from .tensor_utils import unfold, khatri_rao
from .nnls import nnlsm_blockpivot
import sys

def ls_solver(A, B, M=None, l1=None, l2=None, X0=None, nonneg=False, is_input_prod=False):
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
            if is_input_prod:
                # interpret input as AtA, AtB
                if np.linalg.cond(A) > 1/sys.float_info.epsilon:
                    A[np.diag_indices_from(A)] += 1e-3
                return np.linalg.solve(A, B.T).T
            else:
                return np.linalg.lstsq(A.T, B.T)[0].T

        # Nonnegative Least Squares 
        else:

            # prevent singular matrix
            if not is_input_prod:
                A = np.dot(A.T, A)
                B = np.dot(A.T, B)
            if np.linalg.cond(A) > 1/sys.float_info.epsilon:
                A[np.diag_indices_from(A)] += 1e-3

            # solve nonnegative least squares
            X = nnlsm_blockpivot(A, B.T, init=X0.T, is_input_prod=True)[0].T

            # prevent a full column of X going to zero
            idx = np.linalg.norm(X, axis=0) < sys.float_info.epsilon
            X[:, idx] = X0[:, idx]

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
            x = np.array(_nnls_broadcast(Astack, Bstack, X0))
        
        return x.reshape((B.shape[0], A.shape[0]))

def _nnls_broadcast(Astack, Bstack, X0):
    for i in range(len(Astack)):
        A, B = Astack[i], Bstack[i]
        if np.linalg.cond(A) > 1/sys.float_info.epsilon:
            A[np.diag_indices_from(A)] += 1e-3
        X0[i] = nnlsm_blockpivot(A, B, is_input_prod=True, init=X0[i,:,None])[0].T
    return X0.ravel()


def _add_to_diag(A, z):
    """Add z to diagonal of matrix A.
    """
    A[np.diag_indices_from(A)] += z

