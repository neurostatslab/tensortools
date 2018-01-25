least_squares.py

"""
"""

def ls_solver(X0, A, B):
    """Minimizes ||X*A - B||_F for X

    Parameters
    ----------
    X0 : ndarray
            m x k matrix, initial guess for X
    A : ndarray
            k x n matrix
    B : ndarray
            m x n matrix
    """
    # TODO - do conjugate gradient if n is too large    
    return np.linalg.lstsq(A.T, B.T)[0].T