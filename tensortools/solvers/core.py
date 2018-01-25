"""
Core algorithms
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

def solve_subproblem(tensor, factors, mode, M=None, l1=None, l2=None, nonneg=False, options=dict(maxiter=10000)):
    """Approximately solves least-squares with missing/censored data by L-BFGS-B
        Updates X to (approximately) minimize ||X*A - B|| using Frobenius
        norm. Also accommodates missing data entries and l1/l2 regularization
        on X.

    Parameters
    ----------
    tensor : ndarray
            Data tensor being approximated by CP decomposition
    factors : list
            List of factor matrices
    mode : int
            Specifies which factor matrix is updated
    M : ndarray (optional)
            Binary masking matrix (m x n), specifies missing data. Ignored if None (default).
    l1 : float (optional)
            Strength of l1 regularization. Ignored if None (default).
    l2 : float (optional)
            Strength of l2 regularization. Ignored if None (default).
    nonneg : bool (optional)
            If True, constrain X to be nonnegative. Ignored if False (default).
    options : dict
            optimization options passed to scipy.optimize.minimize

    Returns
    -------
    result : OptimizeResult
            returned by scipy.optimize.minimize
    """

    # set up loss function: || X*A - B ||
    A = khatri_rao(factors, skip_matrix=mode).T
    B = unfold(tensor, mode)

    # initial guess for X
    X0 = factors[mode]

    # if no missing data or regularization, exploit fast solvers
    if M is None:
        # reduce grammians
        rank = X0.shape[1]
        G = np.ones((rank, rank))
        for i, f in enumerate(factors):
            if i != mode:
                G *= np.dot(f.T, f)
    
        if l1 is None and l2 is None:
            # method for least squares or nonneg least squares
            solver = _nnls_solver if nonneg else _ls_solver
            return solver(X0.T, G, np.dot(B, A.T))
        else:
            l1 = 0 if l1 is None else l1
            l2 = 0 if l2 is None else l2
            return _elastic_net_solver(X0, G, np.dot(B, A.T), l1, l2, nonneg)
    else:
        # Missing data - use scipy.optimize
        M = unfold(M, mode)
        def fg(x):
            # computes objective and gradient with missing data
            X = x.reshape(*X0.shape)
            resid = np.dot(X, A) - B
            f1, g1 = _l1_reg(l1, X)
            f2, g2 = _l2_reg(l2, X)
            f = 0.5*np.sum(resid[M]**2) + f1 + f2
            g = np.dot((M * resid), A.T) + g1 + g2
            return f, g.ravel()
        # run optimization
        bounds = [(0,None) for _ in range(X0.size)] if nonneg else None
        result = minimize(fg, X0.ravel(), method='L-BFGS-B', jac=True, options=options, bounds=bounds)
        return result.x.reshape(*X0.shape)
