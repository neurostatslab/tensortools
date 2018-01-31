"""
Main interface for fitting tensor decompositions.
"""

import numpy as np
from .tensor_utils import unfold, kruskal_to_tensor, norm, khatri_rao
from numpy.random import randint
from time import time
from .kruskal import standardize_factors, _validate_factors
from .solvers import ls_solver

# default options for cp_direct
OPTIONS = {
    'min_time': 0,
    'max_time': np.inf,
    'n_iter_max': 1000,
    'print_every': 0.3,
    'prepend_print': '\r',
    'append_print': '',
    'tol': 1e-6
}

def cp_direct(tensor, rank, M=None, l1=None, l2=None, nonneg=False,
              init=None, options=OPTIONS):
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
    init : str or list (optional)
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
    if init is None:
        init = 'randn' if nonneg is False else 'rand'

    l1 = l1 if np.iterable(l1) else [l1 for _ in range(tensor.ndim)]
    l2 = l2 if np.iterable(l2) else [l2 for _ in range(tensor.ndim)]
    nonneg = nonneg if np.iterable(nonneg) else [nonneg for _ in range(tensor.ndim)]

    # intialize factor matrices
    factors = _cp_initialize(tensor, rank, init)

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

    # solvers for convex subproblems
    G = np.empty((rank, rank)) # storage for grammians

    # main loop
    t0 = time()
    for iteration in range(options['n_iter_max']):

        # alternating optimization over modes
        for mode in range(tensor.ndim):
            A = khatri_rao(factors, skip_matrix=mode).T
            B = unfold(tensor, mode)

            if M is None:
                # Since A is a khatri-rao product, its pseudoinverse has a special form
                #   see Kolda & Bader (2009), equation 2.2
                G.fill(1.0)
                for i in range(len(factors)):
                    if i != mode:
                        G *= np.dot(factors[i].T, factors[i])
                # compute update
                factors[mode] = ls_solver(G, np.dot(B, A.T), l1=l1[mode], l2=l2[mode], nonneg=nonneg[mode], X0=factors[mode], is_input_prod=True)
            else:
                factors[mode] = ls_solver(A, B, M=unfold(M, mode), l1=l1[mode], l2=l2[mode], nonneg=nonneg[mode], X0=factors[mode])

        # renormalize factors
        factors = standardize_factors(factors, sort_factors=False)

        # assess errors
        resid = (tensor - kruskal_to_tensor(factors))
        resid = M * resid if M is not None else resid
        rec_errors.append(norm(resid, 2) / norm_tensor)

        # assess regularization penalty
        penalty = sum([_l1_reg(lam, factor)[0] for lam, factor in zip(l1, factors)]) + \
            sum([_l2_reg(lam, factor)[0] for lam, factor in zip(l2, factors)])
        obj_hist.append(rec_errors[-1] + penalty)

        # check convergence
        converged = obj_hist[-2] - obj_hist[-1] < options['tol']

        # display progress
        t_elapsed.append(time() - t0)
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

    # initial factors were provided by user
    if isinstance(init, list):
        _validate_factors(init)
        factors = [fctr.copy() for fctr in init]
        return factors
    
    # initialize factors randomly
    if init is 'randn':
        factors = [np.random.randn(tensor.shape[i], rank) for i in range(tensor.ndim)]
    elif init is 'rand':
        factors = [np.random.rand(tensor.shape[i], rank) for i in range(tensor.ndim)]
    else:
        raise ValueError('initialization method not recognized')
    
    # make sure that the norm of the tensor is close to the norm of our initialization
    # first, draw a random sample of entries from the tensor
    idx = np.random.randint(0, tensor.size, size=2**10)
    sub = np.array(np.unravel_index(idx, tensor.shape)).T

    # second, find our initial estimate for those entries
    est = np.ones((len(idx), rank))
    for i, f in enumerate(factors):
        est *= f[sub[:,i]]

    # rescale factors so that the reconstruction matches norm of tensor
    scale = (np.linalg.norm(tensor.ravel()[idx]) / np.linalg.norm(est.sum(axis=1))) ** (1/tensor.ndim)
    return standardize_factors([f * scale for f in factors], sort_factors=False)


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
