"""
Main interface to solvers, fits an ensemble of tensor decompositions
over a specified range of tensor ranks.
"""

import numpy as np
from time import time
from .cpdirect import cp_direct
from .cprand import cp_rand
from .kruskal import align_factors
from .tensor_utils import norm
import pdb

# dictionary holding various optimization algorithms
fitting_methods = {
    'direct': cp_direct,
    'randomized': cp_rand
}

def fit_ensemble(tensor, ranks, l1=None, l2=None, nonneg=False,
                 replicates=1, p_holdout=0, align=True,
                 method='direct', options={}):
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
    method : str (optional)
            Specifies optimization algorithm. Current options are:
                'direct' (default) : alternating least squares (ALS), reviewed in ref [1]
                'randomized' (default) : randomized ALS, ref [2]

    options : dict (optional)
            Dictionary specifying other options for cp_solver.
                'min_time' : optimize for at least this many seconds (default, 0)
                'max_time' : optimize for at most this many seconds (default, np.inf)
                'n_iter_max' : max number of iterations (default, 1000)
                'print_every' : how often to display progress (default, 0.3s)
                'prepend_print': a string preprended to display (default, '\r')
                'append_print' : a string appended to dispay (default, '')
                'tol' : sets convergence criteria for optimization.

    Returns
    -------
    factors : list of ndarray
        estimated low-rank decomposition (in kruskal tensor format)
    info : dict
        information about the fit / optimization convergence


    References
    ----------
    [1] Kolda TG, Bader BW (2009). Tensor Decompositions and Applications. SIAM Review.
    [2] Battaglino C, Ballard G, Kolda TG (2017). A Practical Randomized CP Tensor Decomposition.
    """

    # Treat nans as missing data
    nanmask = np.isfinite(tensor)
    if nanmask.sum() == nanmask.size:
        nanmask = None

    # if rank is input as a single int, wrap it in a list
    if isinstance(ranks, int):
        ranks = [ranks]
    # make sure ranks are sorted
    ranks = np.sort(ranks)

    # compile optimization results into dict indexed by model rank
    keys = ['factors', 'ranks', 'err_hist', 'err_final',
            'test_err', 't_hist', 'converged', 'iterations']
    results = {r: {k: [] for k in keys} for r in ranks}

    # if true, print progress
    verbose = 'print_every' not in options.keys() or options['print_every'] >= 0
    if verbose:
        t0 = time()

    # determine optimization method
    if method not in fitting_methods:
        raise ValueError('Specified method ({}) not recognized. Available options are: {}'.format(method, list(fitting_methods.keys())))
    else:
        fit_cpd = fitting_methods[method]

    # fit ensemble of models for each rank
    for r in ranks:

        if verbose:
            print('Optimizing rank-{} models.'.format(r))
            t0_inner = time()

        for s in range(replicates):

            # create mask for held out data
            M, cvmask = _hold_out_data(tensor, nanmask, p_holdout)

            # fit cpd
            options['prepend_print'] = '\r   fitting replicate: {}/{}    '.format(s+1, replicates)
            factors, info = fit_cpd(tensor, r, M=M, l1=l1, l2=l2, nonneg=nonneg, options=options)

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

    if verbose:
        print('Total time to fit models: {0:.4f}s'.format(time()-t0))


    if align:
        # align factors across ranks
        results[ranks[-1]]['similarity'] = [np.nan] + (replicates-1)*[None]
        for r in reversed(ranks[:-1]):
            # align best rank-r model to the best rank-(r+1) model
            _, factors, score = align_factors(results[r+1]['factors'][0], results[r]['factors'][0])
            results[r]['factors'][0] = factors
            results[r]['similarity'] = [score] + (replicates-1)*[None]

        # align factors within ranks
        for r in ranks:
            best_model = results[r]['factors'][0]
            for s in range(1, replicates):
                aligned_factors, _, score = align_factors(results[r]['factors'][s], best_model)
                results[r]['similarity'][s] = score
                results[r]['factors'][s] = aligned_factors

    return results

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
