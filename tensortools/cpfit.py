import numpy as np
from ._nnls import nnlsm_blockpivot
from tensorly.base import unfold
from tensorly.kruskal import kruskal_to_tensor
from tensorly.tenalg import khatri_rao, norm

def cpfit(tensor, rank, update_method=None, nonneg=False, exact=True,
          sample_frac=0.5, init=None, init_factors=None, tol=10e-7,
          n_iter_max=1000, verbose=False, print_every=1):

    # default initialization method
    if init is None:
        init = 'randn' if nonneg is False else 'rand'

    # intialize factor matrices
    if init_factors is not None:
        factors = init_factors.copy()
    elif init is 'randn':
        factors = [np.random.randn(tensor.shape[i], rank) for i in range(tensor.ndim)]
    elif init is 'rand':
        factors = [np.random.rand(tensor.shape[i], rank) for i in range(tensor.ndim)]
    else:
        raise ValueError('initialization method not recognized')

    # determine optimization method
    if update_method is None:
        # nonneg vs vanilla least-squares
        if nonneg is True:
            ls_method = lambda A, B: nnlsm_blockpivot(A, B)[0]
        else:
            ls_method = np.linalg.solve

        # exact vs randomized least-squares
        if exact is False:
            update_method = lambda t, f, m: _als_rand_update(t, f, m, ls_method=ls_method, frac=sample_frac)
        else:
            update_method = lambda t, f, m: _als_exact_update(t, f, m, ls_method=ls_method)

    # setup optimization
    rec_errors = []
    converged = False
    norm_tensor = norm(tensor, 2)

    # main loop
    for iteration in range(n_iter_max):

        # alternating optimization over modes
        for mode in range(tensor.ndim):
            factors[mode] = update_method(tensor, factors, mode)

        # store reconstruction errors
        rec_error = norm(tensor - kruskal_to_tensor(factors), 2) / norm_tensor
        rec_errors.append(rec_error)

        # check covergence
        if iteration > 1:

            # display progress
            if verbose and ((iteration+1)%print_every) == 0:
                print('iter={}, error={}, variation={}.'.format(
                    iteration+1, rec_errors[-1], rec_errors[-2] - rec_errors[-1]))

            # break loop if converged
            converged = abs(rec_errors[-2] - rec_errors[-1]) < tol
            if tol and converged:
                if verbose:
                    print('converged in {} iterations.'.format(iteration+1))
                break

        elif verbose and iteration==0:
            print('iter={}, error={}.'.format(iteration+1, rec_errors[-1]))


    # return optimized factors and info
    return factors, { 'rec_errors' : rec_errors,
                      'converged' : converged,
                      'iterations' : len(rec_errors) }

def _als_exact_update(tensor, factors, mode, ls_method=np.linalg.solve):

    # reduce grammians
    rank = factors[0].shape[1]
    pseudo_inverse = np.ones((rank, rank))
    for i, factor in enumerate(factors):
        if i != mode:
            pseudo_inverse *= np.dot(factor.T, factor)

    # solve least-squares
    factor = np.dot(unfold(tensor, mode), khatri_rao(factors, skip_matrix=mode))
    return ls_method(pseudo_inverse.T, factor.T).T


def _als_rand_update(tensor, factors, mode, ls_method=np.linalg.solve, frac=0.5):

    # reduce grammians
    rank = factors[0].shape[1]
    pseudo_inverse = np.ones((rank, rank))
    for i, factor in enumerate(factors):
        if i != mode:
            pseudo_inverse *= np.dot(factor.T, factor)

    # unfold tensor
    unf = unfold(tensor, mode)

    # sampled columns of unfolding
    samp_idx = np.random.rand(unf.shape[1]) <= frac

    # sub-sample
    unf = unf[:, samp_idx]
    kr = khatri_rao(factors, skip_matrix=mode)[samp_idx, :]

    # compute factor
    return ls_method(pseudo_inverse.T, np.dot(unf, kr).T).T
