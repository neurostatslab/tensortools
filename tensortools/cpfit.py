import numpy as np
from ._nnls import nnlsm_blockpivot
from tensorly.base import unfold
from tensorly.kruskal import kruskal_to_tensor
from tensorly.tenalg import khatri_rao, norm, mode_dot
from numpy.random import randint
from time import time
from scipy.fftpack import dct, idct
from .kruskal import standardize_kruskal

def cp_als(tensor, rank, nonneg=False, init=None, init_factors=None, tol=10e-7,
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

    # use non-negative least squares for nncp
    if nonneg is True:
        ls_method = lambda A, B: nnlsm_blockpivot(A, B)[0]
    else:
        ls_method = np.linalg.solve


    # setup optimization
    converged = False
    norm_tensor = norm(tensor, 2)
    gram = [np.dot(f.T, f) for f in factors]
    t_elapsed = [0]
    rec_errors = [norm(tensor - kruskal_to_tensor(factors), 2) / norm_tensor]

    # initial print statement
    if verbose:
        print('iter=0, error={}'.format(rec_errors[-1]))

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
            factors[mode] = ls_method(G.T, np.dot(unf, kr).T).T
        
        # renormalize factors
        factors = standardize_kruskal(factors, sort_factors=False)

        # check convergence
        rec_error = norm(tensor - kruskal_to_tensor(factors), 2) / norm_tensor
        rec_errors.append(rec_error)
        t_elapsed.append(time() - t0)

        # break loop if converged
        converged = abs(rec_errors[-2] - rec_errors[-1]) < tol
        if tol and converged:
            if verbose:
                print('converged in {} iterations.'.format(iteration+1))
            break

        # display progress
        if verbose and ((iteration+1)%print_every) == 0 and len(rec_errors) > 1:
            print('iter={}, error={}, variation={}'.format(
                iteration+1, rec_errors[-1], rec_errors[-2] - rec_errors[-1]))

    # return optimized factors and info
    return factors, { 'rec_errors' : rec_errors,
                      't_elapsed' : t_elapsed,
                      'converged' : converged,
                      'iterations' : len(rec_errors) }

def cp_rand(tensor, rank, iter_samples=None, fit_samples=2**14, nonneg=False, init=None,
            init_factors=None, tol=10e-7, n_iter_max=1000, verbose=False, print_every=1):

    # If iter_samples not specified, use heuristic
    if iter_samples is None:
        iter_samples = int(4 * rank * np.log(rank))

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
    else:
        ls_method = lambda A, B: np.linalg.lstsq(A, B)[0]

    # setup convergence checking
    converged = False
    fit_ind = np.random.choice(range(len(tensor.ravel())), size=fit_samples, replace=False)
    fit_sub = np.array([list(np.unravel_index(i, tensor.shape)) for i in fit_ind])
    tensor_sample = tensor.ravel()[fit_ind]
    tensor_sample_norm = np.linalg.norm(tensor_sample)

    # initial calculation of error
    est_sample = np.ones((fit_samples, rank))
    for i, f in enumerate(factors):
        est_sample *= f[fit_sub[:, i], :]
    est_sample = np.sum(est_sample, axis=1)
    rec_error = np.linalg.norm(tensor_sample - est_sample) / tensor_sample_norm
    rec_errors = [rec_error]
    t_elapsed = [0.0]

    # initial print statement
    if verbose:
        print('iter=0, error={}'.format(rec_errors[-1]))

    # main loop
    t0 = time()
    for iteration in range(n_iter_max):

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
        
        # renormalize factors to prevent singularities
        factors = standardize_kruskal(factors, sort_factors=False)

        # estimate randomized subset of full tensor
        est_sample = np.ones((fit_samples, rank))
        for i, f in enumerate(factors):
            est_sample *= f[fit_sub[:, i], :]
        est_sample = np.sum(est_sample, axis=1)

        # store reconstruction error
        rec_error = np.linalg.norm(tensor_sample - est_sample) / tensor_sample_norm
        rec_errors.append(rec_error)
        t_elapsed.append(time() - t0)

        # check convergence, break loop if converged
        converged = abs(rec_errors[-2] - rec_errors[-1]) < tol
        if tol and converged:
            if verbose:
                print('converged in {} iterations.'.format(iteration+1))
            break
            
        # display progress
        if verbose and ((iteration+1)%print_every) == 0 and len(rec_errors) > 1:
            print('iter={}, error={}, variation={}'.format(
                iteration+1, rec_errors[-1], rec_errors[-2] - rec_errors[-1]))

    # return optimized factors and info
    return factors, { 'rec_errors' : rec_errors,
                      't_elapsed' : t_elapsed,
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
