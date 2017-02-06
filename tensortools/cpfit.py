import numpy as np
from ._nnls import nnlsm_blockpivot
import tensorly
from tensorly.base import unfold
from tensorly.kruskal import kruskal_to_tensor
from tensorly.tenalg import khatri_rao, mode_dot
from numpy.random import randint
from time import time
from scipy.fftpack import dct, idct
from scipy.optimize import least_squares
from .kruskal import standardize_factors, align_factors
from ._robust import irls

def cp_als(tensor, rank, nonneg=False, init=None, init_factors=None, tol=1e-6,
           min_time=0, max_time=np.inf, n_iter_max=1000, print_every=0.3, robust=False,
           huber_delta=0.1, prepend_print='\r', append_print=''):
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
    init : str
        specified initialization procedure for factor matrices
        {'randn','rand','svd'}
    init_factors : ktensor (list of ndarray)
        initial factor matrices (overrides "init" keyword arg)
    tol : float
        convergence criterion
    n_iter_max : int
        maximum number of optimizations iterations before aborting
        (default = 1000)
    print_every : float
        how often (in seconds) to print progress. If <= 0 then don't print anything.
        (default = -1)
    """

    # default initialization method
    if init is None:
        init = 'randn' if nonneg is False else 'rand'

    # intialize factor matrices
    factors = _cp_initialize(tensor, rank, init, init_factors)

    # setup optimization
    converged = False
    norm_tensor = tensorly.tenalg.norm(tensor, 2)
    gram = [np.dot(f.T, f) for f in factors]
    t_elapsed = [0]
    rec_errors = [_compute_squared_recon_error(tensor, factors, norm_tensor)]

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
            A = G.T
            B = np.dot(unf, kr).T
            if nonneg and robust:
                raise NotImplementedError()
            elif nonneg is True:
                factors[mode] = nnlsm_blockpivot(A, B)[0].T
            elif robust is True:
                A = kr
                X = factors[mode]
                B = unf
                factors[mode] = irls(A, B.T, x=X.T).T
            else:
                factors[mode] = np.linalg.solve(G.T, np.dot(unf, kr).T).T

            for r in range(rank):
                if np.allclose(factors[mode][:,r], 0):
                    factors[mode][:,r] = np.random.rand(tensor.shape[mode])
        
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

def cp_rand(tensor, rank, iter_samples=None, max_iter_samples=None, fit_samples=2**14, sample_increase=1.1,
            convergence_window=10, nonneg=False, init=None, init_factors=None, tol=1e-5, n_iter_max=1000,
            print_every=0.3, prepend_print='\r', append_print=''):

    # If iter_samples not specified, use heuristic
    if iter_samples is None:
        iter_samples = int(4 * rank * np.log(rank))

    if fit_samples >= len(tensor.ravel()):
        # TODO: warning here.
        fit_samples = len(tensor.ravel())

    # default initialization method
    if init is None:
        init = 'randn' if nonneg is False else 'rand'

    # intialize factor matrices
    factors = _cp_initialize(tensor, rank, init, init_factors)

    # use non-negative least squares for nncp
    if nonneg is True:
        ls_method = lambda A, B: nnlsm_blockpivot(A, B)[0]
    else:
        ls_method = lambda A, B: np.linalg.lstsq(A, B)[0]

    # setup convergence checking
    converged = False
    fit_ind = np.random.choice(range(len(tensor.ravel())), size=fit_samples, replace=False)
    fit_sub = np.array([list(np.unravel_index(i, tensor.shape)) for i in fit_ind])
    tensor_sample = tensor.ravel()[fit_ind]
    tensor_sample_norm = np.linalg.norm(tensor_sample)
    min_error = np.inf
    convergence_counter = 0

    # initial calculation of error
    est_sample = np.ones((fit_samples, rank))
    for i, f in enumerate(factors):
        est_sample *= f[fit_sub[:, i], :]
    est_sample = np.sum(est_sample, axis=1)
    rec_error = np.linalg.norm(tensor_sample - est_sample) / tensor_sample_norm
    rec_errors = [rec_error]
    t_elapsed = [0.0]

    # initial print statement
    verbose = print_every > 0
    print_counter = 0 # time to print next progress
    if verbose:
        print(prepend_print+'iter=0, error={0:.4f}'.format(rec_errors[-1]), end=append_print)

    # set threshold for calling cp-als
    if max_iter_samples is None:
        num_fibers = []
        for mode in range(tensor.ndim):
            mode_shape = [s for m,s in enumerate(tensor.shape) if m != mode]
            num_fibers.append(np.prod(mode_shape))
        max_iter_samples = np.max(num_fibers)

    # main loop
    t0 = time()
    for iteration in range(n_iter_max):

        if iter_samples > max_iter_samples:
            print('punting to cpals')
            return cp_als(tensor, rank, init_factors=best_factors, print_every=-1, nonneg=nonneg)

        # alternating optimization over modes
        for mode in range(tensor.ndim):
            # sample mode-n fibers uniformly with replacement
            idx = [tuple(randint(0, D, iter_samples)) if n != mode else slice(None) for n, D in enumerate(tensor.shape)]

            # unfold sampled tensor
            if mode == 0:
                unf = tensor[idx]
            else:
                unf = tensor[idx].T

            # sub-sampled khatri-rao
            rank = factors[0].shape[1]
            kr = np.ones((iter_samples, rank))
            for i, f in enumerate(factors):
                if i != mode:
                    kr *= f[idx[i], :]

            # update factor
            factors[mode] = ls_method(kr, unf.T).T

            for r in range(rank):
                if np.allclose(factors[mode][:,r], 0):
                    factors[mode][:,r] = np.random.rand(tensor.shape[mode])
        
        # renormalize factors to prevent singularities
        factors = standardize_factors(factors, sort_factors=False)

        # estimate randomized subset of full tensor
        est_sample = np.ones((fit_samples, rank))
        for i, f in enumerate(factors):
            est_sample *= f[fit_sub[:, i], :]
        est_sample = np.sum(est_sample, axis=1)

        # store reconstruction error
        rec_error = np.linalg.norm(tensor_sample - est_sample) / tensor_sample_norm
        rec_errors.append(rec_error)
        t_elapsed.append(time() - t0)

        # check if error went down
        if rec_error < min_error:
            min_error = rec_error
            best_factors = [fctr.copy() for fctr in factors]
        else:
            # TODO - CONSIDER:
            #   factors = [fctr.copy() for fctr in best_factors]
            #   rec_errors[-1] = min_error
            iter_samples = int(iter_samples*sample_increase)

        # check convergence
        if iteration > convergence_window:
            converged = abs(np.mean(np.diff(rec_errors[-convergence_window:]))) < tol
        else:
            converged = False

        # print convergence and break loop
        if converged and verbose:
            print('{}converged in {} iterations.'.format(prepend_print, iteration+1))
        if converged:
            break
            
        # display progress
        if verbose and (time()-t0)/print_every > print_counter:
            print_str = 'iter={0:d}, error={1:.4f}, variation={2:.4f}'.format(
                iteration+1, min_error, rec_errors[-2] - rec_errors[-1])
            print(prepend_print+print_str, end=append_print)
            print_counter += print_every

    # return optimized factors and info
    return best_factors, { 'err_hist' : rec_errors,
                          't_hist' : t_elapsed,
                          'err_final' : rec_errors[-1],
                          'converged' : converged,
                          'iterations' : len(rec_errors) }

def cp_mixrand(tensor, rank, **kwargs):
    """
    Performs mixing to decrease coherence amongst factors before applying randomized
    alternating-least squares to fit CP decomposition. Unmixes the factors before
    returning.
    """
    ndim = tensor.ndim

    # random orthogonal matrices for each tensor
    U = [np.linalg.qr(np.random.randn(s,s))[0] for s in tensor.shape]

    # mix tensor
    tensor_mix = tensor.copy()
    for mode, u in enumerate(U):
        tensor_mix = mode_dot(tensor_mix, u, mode)

    # call cp_rand as a subroutine
    factors_mix, info = cp_rand(tensor_mix, rank, **kwargs)

    # demix factors by inverting orthogonal matrices
    factors = [np.dot(u.T, fact) for u, fact in zip(U, factors_mix)]

    return factors, info

def _cp_initialize(tensor, rank, init, init_factors):
    """ Parameter initialization methods for CP decomposition
    """
    if init_factors is not None:
        factors = init_factors.copy()
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
    return tensorly.tenalg.norm(tensor - kruskal_to_tensor(kruskal_factors), 2) / norm_tensor

def cp_batch_fit(tensor, ranks, replicates=1, method=cp_als, **kwargs):

    # if rank is input as a single int, wrap it in a list
    if isinstance(ranks, int):
        ranks = [ranks]

    # compile optimization results into dict indexed by model rank
    keys = ['factors', 'ranks', 'err_hist', 'err_final', 't_hist', 'converged', 'iterations']
    results = {r: {k: [] for k in keys} for r in ranks}

    # if true, print progress
    verbose = 'print_every' not in kwargs.keys() or kwargs['print_every'] >= 0

    for r in ranks:

        if verbose:
            print('Optimizing rank-{} models.'.format(r))

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
            summary = '\r   {0:d}/{1:d} converged, min error = {2:.4f}, max error = {3:.4f}, mean error = {4:.4f}'
            n_converged = np.sum(results[r]['converged'])
            min_err = np.min(results[r]['err_final'])
            max_err = np.max(results[r]['err_final'])
            mean_err = np.mean(results[r]['err_final'])
            print(summary.format(n_converged, replicates, min_err, max_err, mean_err))

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
