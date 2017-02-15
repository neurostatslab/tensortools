from sklearn.linear_model import Lasso
from ._nnls import nnlsm_blockpivot
import numpy as np

def _get_factor_solvers(ndim, nonneg=None, robust=None, sparsity_penalty=None, lasso_kw=None):
    return [_get_solver(nn, rb, sp, lkw) for nn, rb, sp, lkw in zip(nonneg, robust, sparsity_penalty, lasso_kw)]

def _get_solver(nonneg, robust, sparsity_penalty, lasso_kw):
    
    sparse = sparsity_penalty is not None

    # check for not implemented combinations
    if nonneg and robust:
        raise NotImplementedError
    if sparse and robust:
        raise NotImplementedError
    
    # LASSO solver (not robust, possibly nonneg)
    if sparse:
        lasso_kw['alpha'] = sparsity_penalty
        lasso_kw['positive'] = nonneg
        lasso_kw['warm_start'] = False
        
        lasso = Lasso(**lasso_kw)
        
        def _lasso_solver(A, B, warm_start=None):
            
            # fit factors
            result = lasso.fit(A, B).coef_

            # prevent all parameters going to zero
            for r in range(result.shape[1]):
                if np.allclose(result[:,r], 0):
                    result[:,r] = np.random.rand(result.shape[0])

            return result
        
        return _lasso_solver

    # median absolute deviations solver (not nonneg or sparse)
    elif robust:
        return _irls_solver

    # non-negative least squares solver (not robust or sparse)
    elif nonneg:
        return _nnls_solver

    # classic least squares solver (not nonneg, robust, or sparse)
    else:
        return lambda A, B, warm_start=None: np.linalg.lstsq(A, B)[0].T


def _nnls_solver(A, B, warm_start=None):
    
    # catch singular matrix error, reset result
    # (this should not happen often)
    try:
        result = nnlsm_blockpivot(A, B)[0].T
    except np.linalg.linalg.LinAlgError:
        result = np.random.rand(B.shape[1], A.shape[1])

    # prevent all parameters going to zero
    for r in range(result.shape[1]):
        if np.allclose(result[:,r], 0):
            result[:,r] = np.random.rand(result.shape[0])

    return result


def _irls_solver(A, b, warm_start=None, maxiter=20, d = 0.0001, tol=0.001):
    """Iteratively reweighted least squares

    Solves min_x |A*x - b|_1
    """
    x = warm_start

    if x is None:
        w = np.ones(x.shape[0])
    elif x.ndim > 1:
        x_new = np.empty(x.shape)
        for j in range(x.shape[1]):
            x_new[:,j] = _irls_solver(A, b[:,j], warm_start=x[:,j])
        return x_new.T
    else:
        w = 1 / np.maximum(d, np.abs(A.dot(x) - b))

    itr = 0
    variation = np.inf
    while True:
        # update
        x_last = x
        x = np.linalg.solve((A.T * w).dot(A), (A.T * w).dot(b))
        variation = np.linalg.norm(x - x_last)/np.linalg.norm(x)
        itr += 1

        # check convergence
        if variation < tol and itr >= maxiter:
            break
        else:
            w = 1 / np.maximum(d, np.abs(A.dot(x) - b))

    return x
