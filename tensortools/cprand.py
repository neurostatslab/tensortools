import numpy as np
from .cpdirect import _cp_initialize
from .solvers import ls_solver
from time import time
from .kruskal import standardize_factors

# default options for randomized solver
OPTIONS = {
    'min_time': 0,
    'max_time': np.inf,
    'n_iter_max': 10000,
    'print_every': 1.0,
    'prepend_print': '\r',
    'append_print': '',
    'rs': None,
    'fs': 2**14,
    'rs_inc': 2,
    'rs_max': None,
    'patience': 100,
    'tol': 1e-3
}
            
def cp_rand(tensor, rank, M=None, l1=None, l2=None, nonneg=False, init=None,
            options=OPTIONS):
    
    # ensure default options are present
    for k, v in OPTIONS.items():
        options.setdefault(k, v)

    # extract options
    rs = options['rs']
    fs = options['fs']
    rs_inc = options['rs_inc']
    rs_max = options['rs_max']
    patience = options['patience']
    tol = options['tol']

    # heuristic for the number of samples
    if rs is None:
        rs = np.maximum(10, int(np.ceil(4 * rank * np.log(rank))))
    
    # search over an order of magnitude for fitting samples
    if rs_max is None:
        rs_max = rs * 10

    # default initialization method
    if init is None:
        init = 'rand' if nonneg is False else 'randn'

    # intialize factor matrices
    factors = _cp_initialize(tensor, rank, init)

    # initialize Polyak averaging with damping to stabilize our
    # parameter estimate
    damping = 1e-2 ** (1 / patience)
    averaged_factors = [f.copy() for f in factors]
    
    # set up error estimation
    fit_ind = np.random.randint(0, tensor.size, size=fs)
    fit_sub = np.array(np.unravel_index(fit_ind, tensor.shape)).T
    tensor_sample = tensor.ravel()[fit_ind]
    tensor_sample_norm = np.linalg.norm(tensor_sample)
    est_sample = np.ones((fs, rank))
    min_error = np.inf
    
    # we detect convergence by measuring the Pearson correlation of
    # reconstruction loss and iteration
    converged = False
    px = np.arange(patience)
    px = (px - np.mean(px)) / np.std(px)

    # initial calculation of error
    est_rnks = np.ones((fs, rank))
    kr = np.ones((rs_max, rank))
    est_sample = _cp_est_subset(est_rnks, factors, fit_sub)
    rec_error = np.linalg.norm(tensor_sample - est_sample) / tensor_sample_norm
    err_hist = [rec_error]
    t_elapsed = [0.0]
    Mm = None

    # initial print statement
    verbose = options['print_every'] > 0
    print_counter = 0 # time to print next progress
    if verbose:
        print(options['prepend_print']+'iter=0, error={0:.4f}'.format(err_hist[-1]), end=options['append_print'])

    # main loop
    t0 = time()
    for iteration in range(options['n_iter_max']):

        # alternating optimization over modes
        for mode in range(tensor.ndim):
            # sample mode-n fibers uniformly with replacement
            idx = [tuple(np.random.randint(0, D, rs)) if n != mode else slice(None) for n, D in enumerate(tensor.shape)]

            # unfold sampled tensor
            unf = tensor[idx] if mode == 0 else tensor[idx].T
            
            # if missing data, also unfold mask
            if M is not None:
                Mm = M[idx] if mode == 0 else M[idx].T
            
            # compute sampled khatri-rao
            _krprod_sampled(kr[:rs], factors, idx, mode)

            # update factor with trust region approach / damping
            factors[mode] = ls_solver(kr[:rs].T, unf, M=Mm, nonneg=nonneg, X0=factors[mode])

        # renormalize factors to prevent singularities
        factors = standardize_factors(factors, sort_factors=False)
        
        # Polyak averaging
        averaged_factors = [damping*f0+(1-damping)*f1 for f0, f1 in zip(averaged_factors, factors)]

        # store reconstruction error
        est_sample = _cp_est_subset(est_rnks, factors, fit_sub)
        rec_error = np.linalg.norm(tensor_sample - est_sample) / np.linalg.norm(tensor_sample)
        err_hist.append(rec_error)
        t_elapsed.append(time() - t0)

        # keep track of lowest error
        if rec_error < min_error:
            min_error = rec_error
        # increase sampling if error went up
        elif rec_error > min_error + tol:
            rs = np.minimum(rs*rs_inc, rs_max)
        
        # check convergence
        if iteration > 2*patience:
            # quit optimization if the improvement per iteration (as estimated by linear
            # regression) is less than the chosen tolerance
            converged = np.inner(err_hist[-patience:], px) > -tol

        # print convergence and break loop
        if converged and verbose:
            print('{}converged in {} iterations.'.format(options['prepend_print'], iteration+1), end=options['append_print'])
        if converged:
            break
            
        # display progress
        if verbose and (time()-t0)/options['print_every'] > print_counter:
            print_str = 'iter={0:d}, error={1:.4f}, variation={2:.4f}'.format(
                iteration+1, min_error, err_hist[-2] - err_hist[-1])
            print(options['prepend_print']+print_str, end=options['append_print'])
            print_counter += options['print_every']

    #
    est_sample = _cp_est_subset(est_rnks, averaged_factors, fit_sub)
    final_error = np.linalg.norm(tensor_sample - est_sample) / np.linalg.norm(tensor_sample)

    # return optimized factors and info
    return averaged_factors, { 'err_hist' : err_hist,
                               't_hist' : t_elapsed,
                               'err_final' : final_error,
                               'converged' : converged,
                               'iterations' : len(err_hist) }


def _krprod_sampled(kr, factors, idx, mode):
    """Forms sampled khatri-rao product of factors
    """
    kr.fill(1.0)
    for i, f in enumerate(factors):
        if i != mode:
            kr *= f[idx[i], :]

def _cp_est_subset(est, factors, fit_sub):
    """Forms low-rank estimate of tensor at indices fit_sub
    """
    est.fill(1.0)
    for i, f in enumerate(factors):
        est *= f[fit_sub[:, i], :]
    return np.sum(est, axis=1)
