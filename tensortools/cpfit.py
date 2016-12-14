import numpy as np
from ._nnls import nnlsm_blockpivot
from tensorly.base import unfold
from tensorly.kruskal import kruskal_to_tensor
from tensorly.tenalg import khatri_rao, norm
from numpy.random import randint

def cpfit(tensor, rank,  nonneg=False, exact_update=True, n_samples=None,
          init=None, init_factors=None, tol=10e-7, n_iter_max=1000,
          verbose=False, print_every=1):

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

    # use non-negative least squares for nncp
    if nonneg is True:
        ls_method = lambda A, B: nnlsm_blockpivot(A, B)[0]
    elif exact_update is False:
        ls_method = lambda A, B: np.linalg.lstsq(A, B)[0]
    else:
        ls_method = np.linalg.solve

    # exact vs randomized least-squares
    if exact_update is False:
        if n_samples is None:
            n_samples = int(4 * rank * np.log(rank))
        update_method = lambda *args: _als_rand_update(*args, n_samples, ls_method=ls_method)
    else:
        update_method = lambda *args: _als_exact_update(*args, ls_method=ls_method)

    # setup optimization
    rec_errors = []
    converged = False
    norm_tensor = norm(tensor, 2)
    gram = [np.dot(f.T, f) for f in factors]

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
    G = np.ones((rank, rank))
    for i, f in enumerate(factors):
        if i != mode:
            G *= np.dot(f.T, f)

    # form unfolding and khatri-rao product
    unf = unfold(tensor, mode)
    kr = khatri_rao(factors, skip_matrix=mode)
    
    # solve least-squares to update factor
    return ls_method(G.T, np.dot(unf, kr).T).T


def _als_rand_update(tensor, factors, mode, n_samples, ls_method=np.linalg.lstsq):

    # sample mode-n fibers uniformly with replacement
    idx = [tuple(randint(0, D, n_samples)) if n != mode else slice(None) for n, D in enumerate(tensor.shape)]

    # unfold sampled tensor
    if mode == 0:
        unf = tensor[idx]
    else:
        unf = tensor[idx].T

    # sub-sampled khatri-rao
    rank = factors[0].shape[1]
    kr = np.ones((n_samples, rank))
    for i, f in enumerate(factors):
        if i != mode:
            kr *= f[idx[i], :]

    # compute factor
    return ls_method(kr, unf.T).T

