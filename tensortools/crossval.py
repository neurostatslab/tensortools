import numpy as np
from .cpfit import _cp_initialize, _default_options
from tqdm import trange
from .tensor_utils import norm, unfold, khatri_rao
from scipy.optimize import minimize
from .kruskal import standardize_factors
from time import time

def censored_least_squares(X0, A, B, M, options=dict(maxiter=20), **kwargs):
    """Approximately solves least-squares with missing/censored data by L-BFGS-B
        Updates X to minimize Frobenius norm of M .* (X*A - B), where
        M is a masking matrix (m x n filled with zeros and ones), A and
        B are constant matrices.
    Parameters
    ----------
    X0 : ndarray
            m x k matrix, initial guess for X
    A : ndarray
            k x n matrix
    B : ndarray
            m x n matrix
    M : ndarray
            m x n matrix, filled with zeros and ones
    options : dict
            optimization options passed to scipy.optimize.minimize
    
    Note: additional keyword arguments are passed to scipy.optimize.minimize

    Returns
    -------
    result : OptimizeResult
            returned by scipy.optimize.minimize
    """
    def fg(x):
        X = x.reshape(*X0.shape)
        resid = np.dot(X, A) - B
        f = 0.5*np.sum(resid[M]**2)
        g = np.dot((M * resid), A.T)
        return f, g.ravel()

    result = minimize(fg, X0.ravel(), method='L-BFGS-B', jac=True, options=options, **kwargs)
    return result.x.reshape(*X0.shape)

def cp_crossval(tensor, rank, tol=1e-3, p_holdout=0.1, nonneg=False, init=None, options=_default_options):

    # default initialization method
    if init is None:
        init = 'randn' if nonneg is False else 'rand'

    # initialize factors
    factors = _cp_initialize(tensor, rank, init)

    # setup optimization
    M = np.random.rand(*tensor.shape) > p_holdout
    norm_train = norm(tensor[M], 2)
    norm_test = norm(tensor[~M], 2)
    
    resid = tensor - np.einsum('ir,jr,kr->ijk', *factors)
    train_err = [norm(resid[M], 2) / norm_train]
    test_err = [norm(resid[~M], 2) / norm_test]
    t_elapsed = [0]

    # set box constraints for optimization
    if nonneg:
        bounds = []
        for factor in factors:
            bounds.append([(0, None) for _ in range(factor.size)])
    else:
        bounds = None

    # initial print statement
    verbose = options['print_every'] > 0
    print_counter = 0 # time to print next progress

    # main loop
    t0 = time()
    for iteration in range(options['n_iter_max']):

        # alternating optimization over modes
        for mode in range(tensor.ndim):

            # form unfolding and khatri-rao product
            unf = unfold(tensor, mode)
            kr = khatri_rao(factors, skip_matrix=mode)
            factors[mode] = censored_least_squares(factors[mode], kr.T, unf, unfold(M, mode), bounds=bounds)
            
        # renormalize factors
        factors = standardize_factors(factors, sort_factors=False)

        # compute train/test error
        resid = tensor - np.einsum('ir,jr,kr->ijk', *factors)
        train_err.append(norm(resid[M], 2) / norm_train)
        test_err.append(norm(resid[~M], 2) / norm_test)
        
        # check convergence
        t_elapsed.append(time() - t0)
        converged = abs(train_err[-2] - train_err[-1]) < tol

        # display progress
        if verbose and (time()-t0)/options['print_every'] > print_counter:
            prnt = options['prepend_print'] + 'iter={0:d}, error={1:.4f}, variation={2:.4f}'
            fprnt = prnt.format(iteration+1, train_err[-1], train_err[-2] - train_err[-1])
            print(fprnt, end=options['append_print'])
            print_counter += options['print_every']

        # break loop if converged
        if converged and (time()-t0) > options['min_time']:
            prnt = options['prepend_print'] + 'converged in {} iterations.'
            if verbose: print(prnt.format(iteration+1), end=options['append_print'])
            break

        # stop early if over time
        if (time()-t0) > options['max_time']:
            break

    # final print statement
    if not converged and verbose:
        prnt = 'gave up after {} iterations and {} seconds'
        print(prnt.format(iteration, time()-t0), end=options['append_print'])

    return factors, { 'err_hist' : train_err,
                      'test_err_hist' : test_err,
                      't_hist' : t_elapsed,
                      'err_final' : train_err[-1],
                      'test_err_final': test_err[-1],
                      'converged' : converged,
                      'iterations' : len(train_err) }

