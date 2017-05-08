import tensorly
import numpy as np
from time import time
from tensorly.base import unfold
from tensorly.tenalg import khatri_rao
from .cpfit import _ls_solver, _nnls_solver, _compute_squared_recon_error
from .kruskal import standardize_factors

def constrained_cp(tensor, factors, free_idx, nonneg=False, tol=1e-6,
                   min_time=0, max_time=np.inf, n_iter_max=1000, print_every=0.3,
                   prepend_print='\r', append_print=''):
    """ Fit CP decomposition with constrained factors by alternating least-squares.

    Example
    -------
    X = [array([ False, True])

    Args
    ----
    tensor : ndarray
        data to be approximated by CP decomposition
    factors : list of ndarray
        initial estimate of low-rank factors (in kruskal tensor format)
    free_idx : list of ndarray
        specifies which factors are free to be optimized. For example, 
        free_idx[mode] == array([ False, True, True]) sets the first factor
        as constant and the second two as free to be optimized.

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

    # setup optimization
    converged = False
    norm_tensor = tensorly.tenalg.norm(tensor, 2)
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

            # check number of constrained factors for this mode
            _rank = sum(free_idx[mode])
            if _rank == 0:
                continue

            # select factors that are free for this mode
            _factors = [f[:, free_idx[mode]] for f in factors]

            # reduce grammians
            G = np.ones((_rank, _rank))
            for i, f in enumerate(_factors):
                if i != mode:
                    G *= np.dot(f.T, f)

            # form unfolding and khatri-rao product
            unf = unfold(tensor, mode)
            kr = khatri_rao(_factors, skip_matrix=mode)

            # update factor
            factors[mode][:, free_idx[mode]] = solver(G, np.dot(unf, kr), warm_start=_factors[mode].T)
        
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
