"""
Fitting code for Canonical Polyadic (CP) Decompositions.
"""

import numpy as np
from .tensor_utils import unfold, kruskal_to_tensor, khatri_rao, norm
from numpy.random import randint
from time import time
from .kruskal import standardize_factors, align_factors, _validate_factors
from .nnls import nnlsm_blockpivot

def _ls_solver(A, B, warm_start=None):
    """Solves X*A - B for X using least-squares.
    """
    # TODO - do conjugate gradient if n is too large
    return np.linalg.lstsq(A.T, B.T)[0].T

def _nnls_solver(A, B, warm_start=None):
    """Solves X*A - B for X within nonnegativity constraint.
    """
    # catch singular matrix error, reset result
    # (this should not happen often)
    try:
        result = nnlsm_blockpivot(A.T, B.T, init=warm_start)[0].T
    except np.linalg.linalg.LinAlgError:
        result = np.random.rand(B.shape[0], A.shape[0])

    # prevent all parameters going to zero
    for r in range(result.shape[1]):
        if np.allclose(result[:,r], 0):
            result[:,r] = np.random.rand(result.shape[0])

    return result

def cp_als(tensor, rank, nonneg=False, init=None, tol=1e-6,
           min_time=0, max_time=np.inf, n_iter_max=1000, print_every=0.3,
           prepend_print='\r', append_print=''):
    """ Fit CP decomposition by alternating least-squares.

    Args
    ----
    tensor : ndarray
        data to be approximated by CP decomposition
    rank : int
        number of components in the CP decomposition model

    Keyword Args
    ------------
    nonneg : bool
        (default = False)
        if True, use alternating non-negative least squares to fit tensor
    init : str or ktensor
        specified initialization procedure for factor matrices
        {'randn','rand','svd'}
    tol : float
        convergence criterion
    n_iter_max : int
        maximum number of optimizations iterations before aborting
        (default = 1000)
    print_every : float
        how often (in seconds) to print progress. If <= 0 then don't print anything.
        (default = -1)

    Returns
    -------
    factors : list of ndarray
        estimated low-rank decomposition (in kruskal tensor format)
    info : dict
        information about the fit / optimization convergence
    """

    # default initialization method
    if init is None:
        init = 'randn' if nonneg is False else 'rand'

    # intialize factor matrices
    factors = _cp_initialize(tensor, rank, init)

    # setup optimization
    converged = False
    norm_tensor = norm(tensor, 2)
    t_elapsed = [0]
    rec_errors = [_compute_squared_recon_error(tensor, factors, norm_tensor)]

    # setup alternating solver
    solver = _nnls_solver if nonneg else _ls_solver

    # initial print statement
    verbose = print_every > 0
    print_counter = 0 # time to print next progress
    if verbose:
        print(prepend_print+'iter=0, error={0:.4f}'.format(rec_errors[-1]), end=append_print)

    # main loop
    t0 = time()
    for iteration in range(n_iter_max):

        # alternating optimization over modes
        for mode in range(tensor.ndim):

            # reduce grammians
            G = np.ones((rank, rank))
            for i, f in enumerate(factors):
                if i != mode:
                    G *= np.dot(f.T, f)

            # form unfolding and khatri-rao product
            unf = unfold(tensor, mode)
            kr = khatri_rao(factors, skip_matrix=mode)

            # update factor
            factors[mode] = solver(G, np.dot(unf, kr), warm_start=factors[mode].T)
        
        # renormalize factors
        factors = standardize_factors(factors, sort_factors=False)

        # check convergence
        rec_errors.append(_compute_squared_recon_error(tensor, factors, norm_tensor))
        t_elapsed.append(time() - t0)

        # break loop if converged
        converged = abs(rec_errors[-2] - rec_errors[-1]) < tol
        if converged and (time()-t0)>min_time:
            if verbose: print(prepend_print+'converged in {} iterations.'.format(iteration+1), end=append_print)
            break

        # display progress
        if verbose and (time()-t0)/print_every > print_counter:
            print_str = 'iter={0:d}, error={1:.4f}, variation={2:.4f}'.format(
                iteration+1, rec_errors[-1], rec_errors[-2] - rec_errors[-1])
            print(prepend_print+print_str, end=append_print)
            print_counter += print_every

        # stop early if over time
        if (time()-t0)>max_time:
            break

    if not converged and verbose:
        print('gave up after {} iterations and {} seconds'.format(iteration, time()-t0), end=append_print)

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

# TODO: optimize this computation
def _compute_squared_recon_error(tensor, kruskal_factors, norm_tensor):
    """ Computes norm of residuals divided by norm of data.
    """
    return norm(tensor - kruskal_to_tensor(kruskal_factors), 2) / norm_tensor

def fit_ensemble(tensor, ranks, replicates=1, method=cp_als, **kwargs):

    # if rank is input as a single int, wrap it in a list
    if isinstance(ranks, int):
        ranks = [ranks]

    # compile optimization results into dict indexed by model rank
    keys = ['factors', 'ranks', 'err_hist', 'err_final', 't_hist', 'converged', 'iterations']
    results = {r: {k: [] for k in keys} for r in ranks}

    # if true, print progress
    verbose = 'print_every' not in kwargs.keys() or kwargs['print_every'] >= 0
    if verbose:
        t0 = time()

    for r in ranks:

        if verbose:
            print('Optimizing rank-{} models.'.format(r))
            t0_inner = time()

        for s in range(replicates):
            # fit cpd
            kwargs['prepend_print'] = '\r   fitting replicate: {}/{}    '.format(s+1, replicates)
            factors, info = method(tensor, r, **kwargs)

            # store results
            results[r]['factors'].append(factors)
            results[r]['ranks'].append(r)
            for k in info.keys():
                results[r][k].append(info[k])

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
