import scipy.optimize as opt
from scipy.special import huber
import numpy as np

def huber_regression(delta, A, B, x0=None, **opt_kw):
    """Solve A*X = B under a Huber loss function.
    """

    x_shape = (A.shape[1], B.shape[1])

    def _huber_loss(x):
        X = np.reshape(x, x_shape)
        resid = np.dot(A,X) - B
        resid = resid.ravel()
        loss =  np.sum(huber(delta, resid))
        idx = np.abs(resid) > delta
        resid[idx] = delta * np.sign(resid[idx])
        return loss, np.dot(A.T, resid.reshape(A.shape[0], -1)).ravel()

    if x0 is None:
        x0 = np.linalg.solve(A, B).ravel()

    f = lambda x: _huber_loss(x)[0]
    g = lambda x: _huber_loss(x)[1]

    result = opt.minimize(_huber_loss, x0, jac=True, **opt_kw)
    # if not result.success:
    #     print('warn - not converged.')
    return result.x.reshape(x_shape)

def irls(A, b, x=None, maxiter=20, d = 0.0001, tol=0.001):
    """Iteratively reweighted least squares

    Solves min_x |A*x - b|_1
    """

    if x is None:
        w = np.ones(x.shape[0])
    elif x.ndim > 1:
        x_new = np.empty(x.T.shape)
        for j in range(x.shape[1]):
            x_new[j] = irls(A, b[:,j], x=x[:,j])
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
