"""
Fitting code for Canonical Polyadic (CP) Decompositions.
"""

import numpy as np
from .tensor_utils import unfold, kruskal_to_tensor, khatri_rao, norm
from numpy.random import randint
from time import time
from .kruskal import standardize_factors, align_factors, _validate_factors
from .nnls import nnlsm_blockpivot
from scipy.optimize import minimize
import scipy

# default options for cp_solver
OPTIONS = {
    'min_time': 0,
    'max_time': np.inf,
    'n_iter_max': 1000,
    'print_every': 0.3,
    'prepend_print': '\r',
    'append_print': ''    
}

# default stopping criterion for cp_solver
TOL = 1e-6

def _ls_solver(X0, A, B):
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

def _nnls_solver(X0, A, B):
    """Minimizes ||X*A - B||_F for X, subject to X >= 0

    Parameters
    ----------
    X0 : ndarray
            m x k matrix, initial guess for X
    A : ndarray
            k x n matrix
    B : ndarray
            m x n matrix
    """

    # catch singular matrix error, reset result
    # (this should not happen often)
    try:
        X = nnlsm_blockpivot(A.T, B.T, init=X0)[0].T
    except np.linalg.linalg.LinAlgError:
        X = np.random.rand(B.shape[0], A.shape[0])

    # prevent all parameters going to zero
    for r in range(X.shape[1]):
        if np.allclose(X[:,r], 0):
            X[:,r] = np.random.rand(X.shape[0])

    return X

def _add_to_diag(A, z):
    B = A.copy()
    B[np.diag_indices_from(B)] + z
    return B


def _elastic_net_solver(X0, A, B, gam1, gam2, nonneg, lam=1, iterations=1000):
    """Minimizes ||X*A - B||_F + elastic_net(X) for CP decomposition subproblem

    Parameters
    ----------
    X0 : ndarray
            n x r matrix, initial guess for X
    A : ndarray
            r x r, symmetric matrix holding reduced Grammians
    B : ndarray
            n x r matrix, unfolding times khatri-rao product
    """

    # admm penalty param
    lam1 = gam1*lam
    lam2 = gam2*lam

    # cache lu factorization for fast prox operator
    # add 1/lam to diagonal of AtA
    Afct = scipy.linalg.lu_factor(_add_to_diag(A, 1/lam))

    # proximal operators
    prox_f = lambda v: scipy.linalg.lu_solve(Afct, (B + v/lam).T).T
    if nonneg:
        prox_g = lambda v: np.maximum(0, v-lam1) / (1 + lam2)
    else:
        prox_g = lambda v: (np.maximum(0, v-lam1) - np.maximum(0, -v-lam1)) / (1 + lam2)

    # initialize admm
    x = X0.copy()
    z = prox_g(x)
    u = x - z

    # admm iterations
    for itr in range(iterations):
        # updates
        x1 = prox_f(z - u)
        z1 = prox_g(x1 + u)
        u1 = u + x1 - z1

        # primal resids (r) and dual resids (s)
        r = np.linalg.norm(x1 - z1)
        s = (1/lam) * np.linalg.norm(z - z1)

        # # keep primal and dual resids within factor of 10
        # if r > 10*s:
        #     lam = lam / 2
        #     # print('{} - {} - {}'.format(itr, r, s))
        #     lam1 = gam1*lam
        #     lam2 = gam2*lam
        #     Afct = scipy.linalg.lu_factor(_add_to_diag(A, 1/lam))

        # elif s > 10*r:
        #     lam = lam * 1.9
        #     # print('{} * {} * {}'.format(itr, r, s))
        #     lam1 = gam1*lam
        #     lam2 = gam2*lam
        #     Afct = scipy.linalg.lu_factor(_add_to_diag(A, 1/lam))

        # accept parameter update
        x, z, u = x1.copy(), z1.copy(), u1.copy()

        # quit if we've converged
        if r < np.sqrt(x.size)*1e-3 and s < np.sqrt(x.size)*1e-3:
            break

    return x

def _l1_reg(lam, X):
    """Returns value and gradient of l1 regularization term on X

    Parameters
    ----------
    lam : float
        scale of the regularization
    X : ndarray
        Array holding the optimized variables
    """
    if lam is None:
        return 0, 0
    else:
        f = lam * np.sum(np.abs(X))
        g = lam * np.sign(X)
        return f, g

def _l2_reg(lam, X):
    """Returns value and gradient of l2 regularization term on X

    Parameters
    ----------
    lam : float
        scale of the regularization
    X : ndarray
        Array holding the optimized variables
    """
    if lam is None:
        return 0, 0
    else:
        f = 0.5 * lam * np.sum(X**2)
        g = lam * X
        return f, g

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
        bounds = [(0,None) for _ in range(X.size())] if nonneg else None
        result = minimize(fg, X0.ravel(), method='L-BFGS-B', jac=True, options=options, bounds=bounds)
        return result.x.reshape(*X0.shape)

def cp_solver(tensor, rank, M=None, l1=None, l2=None, nonneg=False,
              init_factors=None, tol=TOL, options=OPTIONS):
    """ Fit CP decomposition by alternating least-squares.

    Args
    ----
    tensor : ndarray
            Data to be approximated by CP decomposition
    rank : int
            Number of components in the CP decomposition model
    M : ndarray (optional)
            Binary tensor (same size as data). If specified, indicates missing data.
            Ignored if None (default).
    l1 : float (optional)
            Strength of l1 regularization. Ignored if None (default).
    l2 : float (optional)
            Strength of l2 regularization. Ignored if None (default).
    nonneg : bool (optional)
            If True, constrain CP decomposition factors to be nonnegative.
            If False, CP factors are unconstrained (default).
    init_factors : str or list (optional)
            Either a list specifying the initial factor matrices or
            a string specifying how to initalize factors.
            Options are {'randn', 'rand', 'svd'}. If unspecified,
            either 'randn' (if `nonneg` is False) or 'rand' (if `nonneg`
            is True) is used.
    tol : float (optional)
            Sets convergence criteria. Optimization terminates when
            improvement in objective function is less than this
            tolerance (default value, 1e-6).

    Returns
    -------
    factors : list of ndarray
        estimated low-rank decomposition (in kruskal tensor format)
    info : dict
        information about the fit / optimization convergence
    """

    # ensure default options are present
    for k, v in OPTIONS.items():
        options.setdefault(k, v)

    # default initialization method
    if init_factors is None:
        init_factors = 'randn' if nonneg is False else 'rand'

    l1 = l1 if np.iterable(l1) else [l1 for _ in range(tensor.ndim)]
    l2 = l2 if np.iterable(l2) else [l2 for _ in range(tensor.ndim)]

    # intialize factor matrices
    factors = _cp_initialize(tensor, rank, init_factors)

    # setup optimization
    converged = False
    norm_tensor = norm(tensor, 2) if M is None else norm(M * tensor, 2)
    t_elapsed = [0]
    resid = (tensor - kruskal_to_tensor(factors))
    resid = M * resid if M is not None else resid
    rec_errors = [norm(resid, 2) / norm_tensor]

    penalty = sum([_l1_reg(lam, factor)[0] for lam, factor in zip(l1, factors)]) + \
            sum([_l2_reg(lam, factor)[0] for lam, factor in zip(l2, factors)])
    obj_hist = [rec_errors[-1] + penalty]

    # initial print statement
    verbose = options['print_every'] > 0
    print_counter = 0 # time to print next progress

    # main loop
    t0 = time()
    for iteration in range(options['n_iter_max']):

        # alternating optimization over modes
        for mode in range(tensor.ndim):
            factors[mode] = solve_subproblem(tensor, factors, mode, M=M, l1=l1[mode],
                                             l2=l2[mode], nonneg=nonneg)

        # renormalize factors
        factors = standardize_factors(factors, sort_factors=False)

        # check convergence
        resid = (tensor - kruskal_to_tensor(factors))
        resid = M * resid if M is not None else resid
        rec_errors.append(norm(resid, 2) / norm_tensor)

        penalty = sum([_l1_reg(lam, factor)[0] for lam, factor in zip(l1, factors)]) + \
            sum([_l2_reg(lam, factor)[0] for lam, factor in zip(l2, factors)])
        obj_hist.append(rec_errors[-1] + penalty)

        t_elapsed.append(time() - t0)

        converged = obj_hist[-2] - obj_hist[-1] < tol

        # display progress
        if verbose and (time()-t0)/options['print_every'] > print_counter:
            prnt = options['prepend_print'] + 'iter={0:d}, obj={1:.4f}, variation={2:.4f}'
            fprnt = prnt.format(iteration+1, obj_hist[-1], obj_hist[-2] - obj_hist[-1])
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

    if not converged and verbose:
        prnt = 'gave up after {} iterations and {} seconds'
        print(prnt.format(iteration, time()-t0), end=options['append_print'])

    # return optimized factors and info
    return factors, { 'err_hist' : rec_errors,
                      't_hist' : t_elapsed,
                      'err_final' : rec_errors[-1],
                      'converged' : converged,
                      'iterations' : len(rec_errors) }

def _cp_initialize(tensor, rank, init):
    """ Parameter initialization methods for CP decomposition
    """
    if rank <=0:
        raise ValueError('Trying to fit a rank-{} model. Rank must be a positive integer.'.format(rank))

    if isinstance(init, list):
        _validate_factors(init)
        factors = [fctr.copy() for fctr in init]
    elif init is 'randn':
        factors = [np.random.randn(tensor.shape[i], rank) for i in range(tensor.ndim)]
    elif init is 'rand':
        factors = [np.random.rand(tensor.shape[i], rank) for i in range(tensor.ndim)]
    elif init is 'svd':
        factors = []
        for mode in range(tensor.ndim):
            u, s, _ = np.linalg.svd(unfold(tensor, mode), full_matrices=False)
            factors.append(u[:, :rank]*np.sqrt(s[:rank]))
    else:
        raise ValueError('initialization method not recognized')

    return factors

def _hold_out_data(tensor, nanmask, p_holdout):
    """ Create cross-validation mask
    """
    cvmask = np.random.rand(*tensor.shape) > p_holdout if (p_holdout > 0) else None
    
    if nanmask is None and cvmask is None:
        M = None
    elif nanmask is not None and cvmask is not None:
        M = cvmask | nanmask
        cvmask[nanmask] = 0
    elif cvmask is None:
        M = nanmask.copy()
    else:
        M = cvmask.copy()

    return M, cvmask

def fit_ensemble(tensor, ranks, l1=None, l2=None, nonneg=False,
                 replicates=1, p_holdout=0, tol=TOL, options=OPTIONS):
    """ Helper function that fits a bunch of CP decomposition models

    Args
    ----
    tensor : ndarray
            Data to be approximated by CP decomposition
    ranks : iterable or int
            Specifies a range of model ranks to search over.
    l1 : float (optional)
            Strength of l1 regularization. Ignored if None (default).
    l2 : float (optional)
            Strength of l2 regularization. Ignored if None (default).
    nonneg : bool (optional)
            If True, constrain CP decomposition factors to be nonnegative.
            If False, CP factors are unconstrained (default).
    replicates : int (optional)
            Number of randomly initialized optimization runs for each
            model rank (default, 1).
    p_holdout : float (optional)
            Probability of holding out an element of the tensor (at random).
            Useful for cross-validation purposes (default, 0).
    tol : float (optional)
            Sets convergence criteria. Optimization terminates when
            improvement in objective function is less than this
            tolerance (default value, 1e-6).
    options : dict (optional)
            Dictionary specifying other options for cp_solver.
                'min_time' : optimize for at least this many seconds (default, 0)
                'max_time' : optimize for at most this many seconds (default, np.inf)
                'n_iter_max' : max number of iterations (default, 1000)
                'print_every' : how often to display progress (default, 0.3s)
                'prepend_print': a string preprended to display (default, '\r')
                'append_print' : a string appended to dispay (default, '')    

    Returns
    -------
    factors : list of ndarray
        estimated low-rank decomposition (in kruskal tensor format)
    info : dict
        information about the fit / optimization convergence
    """

    # Treat nans as missing data
    nanmask = np.isfinite(tensor)
    if nanmask.sum() == nanmask.size:
        nanmask = None

    # if rank is input as a single int, wrap it in a list
    if isinstance(ranks, int):
        ranks = [ranks]

    # compile optimization results into dict indexed by model rank
    keys = ['factors', 'ranks', 'err_hist', 'err_final',
            'test_err', 't_hist', 'converged', 'iterations']
    results = {r: {k: [] for k in keys} for r in ranks}

    # if true, print progress
    verbose = 'print_every' in options.keys() and options['print_every'] >= 0
    if verbose:
        t0 = time()

    for r in ranks:

        if verbose:
            print('Optimizing rank-{} models.'.format(r))
            t0_inner = time()

        for s in range(replicates):

            # create mask for held out data
            M, cvmask = _hold_out_data(tensor, nanmask, p_holdout)

            # fit cpd
            options['prepend_print'] = '\r   fitting replicate: {}/{}    '.format(s+1, replicates)
            factors, info = cp_solver(tensor, r, M=M, tol=tol, l1=l1, l2=l2, nonneg=nonneg, options=options)

            # store results
            results[r]['factors'].append(factors)
            results[r]['ranks'].append(r)
            for k in info.keys():
                results[r][k].append(info[k])

            # compute test error
            if cvmask is None:
                results[r]['test_err'].append(-1)
            else:
                resid = tensor - np.einsum('ir,jr,kr->ijk', *factors)
                test_err = norm(resid[~cvmask], 2) / norm(tensor[~cvmask], 2)
                results[r]['test_err'].append(test_err)

        # summarize the fits for rank-r models
        if verbose:
            summary = '\r   {0:d}/{1:d} converged, min error = {2:.4f}, max error = {3:.4f}, mean error = {4:.4f}, time to fit = {5:.4f}s'
            n_converged = np.sum(results[r]['converged'])
            min_err = np.min(results[r]['err_final'])
            max_err = np.max(results[r]['err_final'])
            mean_err = np.mean(results[r]['err_final'])
            print(summary.format(n_converged, replicates, min_err, max_err, mean_err, time()-t0_inner))

        # sort results by final reconstruction error
        idx = np.argsort(results[r]['err_final'])
        for k in results[r].keys():
            results[r][k] = [results[r][k][i] for i in idx]

        # calculate similarity score of each model to the best fitting model
        best_model = results[r]['factors'][0]
        results[r]['similarity'] = [1.0] + (replicates-1)*[None]
        for s in range(1, replicates):
            aligned_factors, _, score = align_factors(results[r]['factors'][s], best_model)
            results[r]['similarity'][s] = score
            results[r]['factors'][s] = aligned_factors

    if verbose:
        print('Total time to fit models: {0:.4f}s'.format(time()-t0))

    return results


# def _compute_squared_recon_error(tensor, kruskal_factors, norm_tensor):
#     """Prototype for more efficient reconstruction of squared recon error.
#     """
#     rank = kruskal_factors[0].shape[1]
#     # e.g. 'abc' for a third-order tensor
#     tnsr_idx = ''.join(chr(ord('a') + i) for i in range(len(kruskal_factors)))
#     # e.g. 'az,bz,cz' for a third-order tensor
#     kruskal_idx = ','.join(idx+'z' for idx in tnsr_idx)
#     # compute reconstruction error using einsum
#     innerprod = np.einsum(tnsr_idx+','+kruskal_idx+'->', tensor, *kruskal_factors)
#     G = np.ones((rank, rank))
#     for g in [np.dot(f.T, f) for f in kruskal_factors]:
#         G *= g
#     factors_sqnorm = np.sum(G)
#     return np.sqrt(norm_tensor**2 + factors_sqnorm - 2*innerprod) / norm_tensor
