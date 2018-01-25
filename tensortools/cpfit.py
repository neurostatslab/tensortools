"""
Fitting code for Canonical Polyadic (CP) Decompositions.
"""

import numpy as np
from .tensor_utils import unfold, kruskal_to_tensor, khatri_rao, norm
from numpy.random import randint
from time import time
from .kruskal import standardize_factors, align_factors, _validate_factors
from .solvers import solve_subproblem
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