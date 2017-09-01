# """
# Experimental code for randomized Alternating-Least Squares for fitting CP Decompositions.
# """

# from cp_als import *

# def cp_rand(tensor, rank, nonneg=False, iter_samples=None,
#             fit_samples=2**14, window=10, init=None, tol=1e-5,
#             n_iter_max=1000, print_every=0.3, prepend_print='\r', append_print=''):

#     # If iter_samples not specified, use heuristic
#     if iter_samples is None:
#         iter_samples = lambda itr: max(200, int(np.ceil(4 * rank * np.log(rank))))

#     if fit_samples >= len(tensor.ravel()):
#         # TODO: warning here.
#         fit_samples = len(tensor.ravel())

#     # default initialization method
#     if init is None:
#         init = 'randn' if nonneg is False else 'rand'

#     # intialize factor matrices
#     factors = _cp_initialize(tensor, rank, init)

#     # setup convergence checking
#     converged = False
#     fit_ind = np.random.randint(0, np.prod(tensor.shape), size=fit_samples)
#     fit_sub = np.array([list(np.unravel_index(i, tensor.shape)) for i in fit_ind])
#     tensor_sample = tensor.ravel()[fit_ind]
#     tensor_sample_norm = np.linalg.norm(tensor_sample)
#     min_error = np.inf
#     convergence_counter = 0

#     # initial calculation of error
#     est_sample = np.ones((fit_samples, rank))
#     for i, f in enumerate(factors):
#         est_sample *= f[fit_sub[:, i], :]
#     est_sample = np.sum(est_sample, axis=1)
#     rec_error = np.linalg.norm(tensor_sample - est_sample) / tensor_sample_norm
#     rec_errors = [rec_error]
#     t_elapsed = [0.0]

#     # setup alternating solver
#     solver = _nnls_solver if nonneg else _ls_solver

#     # initial print statement
#     verbose = print_every > 0
#     print_counter = 0 # time to print next progress
#     if verbose:
#         print(prepend_print+'iter=0, error={0:.4f}'.format(rec_errors[-1]), end=append_print)

#     # main loop
#     t0 = time()
#     for iteration in range(n_iter_max):
#         s = iter_samples(iteration)
        
#         # alternating optimization over modes
#         for mode in range(tensor.ndim):
#             # sample mode-n fibers uniformly with replacement
#             idx = [tuple(randint(0, D, s)) if n != mode else slice(None) for n, D in enumerate(tensor.shape)]

#             # unfold sampled tensor
#             if mode == 0:
#                 unf = tensor[idx]
#             else:
#                 unf = tensor[idx].T

#             # sub-sampled khatri-rao
#             rank = factors[0].shape[1]
#             kr = np.ones((s, rank))
#             for i, f in enumerate(factors):
#                 if i != mode:
#                     kr *= f[idx[i], :]

#             # update factor
#             factors[mode] = solver(kr.T, unf, warm_start=factors[mode].T)

#         # renormalize factors to prevent singularities
#         factors = standardize_factors(factors, sort_factors=False)

#         # estimate randomized subset of full tensor
#         est_sample = np.ones((fit_samples, rank))
#         for i, f in enumerate(factors):
#             est_sample *= f[fit_sub[:, i], :]
#         est_sample = np.sum(est_sample, axis=1)

#         # store reconstruction error
#         rec_error = np.linalg.norm(tensor_sample - est_sample) / tensor_sample_norm
#         rec_errors.append(rec_error)
#         t_elapsed.append(time() - t0)

#         # check if error went down
#         if rec_error < min_error:
#             min_error = rec_error
#             best_factors = [fctr.copy() for fctr in factors]
#         else:
#             factors = [fctr.copy() for fctr in best_factors]
#             rec_errors[-1] = min_error

#         # check convergence
#         if iteration > window:
#             converged = abs(np.mean(np.diff(rec_errors[-window:]))) < tol
#         else:
#             converged = False

#         # print convergence and break loop
#         if converged and verbose:
#             print('{}converged in {} iterations.'.format(prepend_print, iteration+1), end=append_print)
#         if converged:
#             break
            
#         # display progress
#         if verbose and (time()-t0)/print_every > print_counter:
#             print_str = 'iter={0:d}, error={1:.4f}, variation={2:.4f}'.format(
#                 iteration+1, min_error, rec_errors[-2] - rec_errors[-1])
#             print(prepend_print+print_str, end=append_print)
#             print_counter += print_every

#     # return optimized factors and info
#     return best_factors, { 'err_hist' : rec_errors,
#                           't_hist' : t_elapsed,
#                           'err_final' : rec_errors[-1],
#                           'converged' : converged,
#                           'iterations' : len(rec_errors) }

# def cp_mixrand(tensor, rank, **kwargs):
#     """
#     Performs mixing to decrease coherence amongst factors before applying randomized
#     alternating-least squares to fit CP decomposition. Unmixes the factors before
#     returning.
#     """
#     ndim = tensor.ndim

#     # random orthogonal matrices for each tensor
#     U = [np.linalg.qr(np.random.randn(s,s))[0] for s in tensor.shape]

#     # mix tensor
#     tensor_mix = tensor.copy()
#     for mode, u in enumerate(U):
#         tensor_mix = mode_dot(tensor_mix, u, mode)

#     # call cp_rand as a subroutine
#     factors_mix, info = cp_rand(tensor_mix, rank, **kwargs)

#     # demix factors by inverting orthogonal matrices
#     factors = [np.dot(u.T, fact) for u, fact in zip(U, factors_mix)]

#     return factors, info