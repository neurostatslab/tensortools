"""
Optimization approach for multi-shift-model.
"""

import numba
from tensortools.cpwarp import periodic_shifts
from tensortools.cpwarp import padded_shifts


@numba.jit(nopython=True, parallel=True)
def multishift_hals(
        X, rank, trial_factors, templates, min_iter=10,
        max_iter=1000, tol=1e-4, warp_iterations=50, max_shift=.5,
        shift_reg=1e-6, periodic_boundaries=False, patience=5, seed=None):

    # Problem dimensions, norm of data.
    K, T, N = X.shape
    Xest = np.empty_like(X)
    Xnorm = np.linalg.norm(X.ravel())

    # Ensure at least two iterations for convergence check.
    min_iter = max(patience, min_iter)
    loss_hist = []
    reg_hist = []

    # Initialize model parameters.
    shifts = npr.uniform(-1.0, 1.0, size=(rank, K))

    # Allocate memory for intermediate computations.
    w_template = np.empty((T, N))
    WtX = np.empty((T, N))
    WtW = np.empty((2, T))

    # === main loop === #

    converged = False
    itercount = 0

    while (itercount < max_iter) and not converged:

        # Update parameters in random order.
        for z in npr.permutation(rank * 3):

            # Update component r.
            r = z // 3

            # Update residuals (skip component r).
            _multishift_predict(
                trial_factors, templates, shifts,
                periodic_boundaries, Xest, skip_dim=r)
            R = X - Xest

            # Update one of the templates, trial factors, or shifts.
            q = z % 3

            # Update trial factors
            if q == 0:

                for k in range(K):

                    if periodic_boundaries:
                        periodic_shifts.shift_one_trial(
                            templates[r], shifts[r, k], w_template)
                    else:
                        padded_shifts.shift_one_trial(
                            templates[r], shifts[r, k], w_template)

                    num = 0.0
                    denom = 0.0
                    for t in range(T):
                        for n in range(N):
                            num += R[k, t, n] * w_template[t, n]
                            denom += w_template[t, n] * w_template[t, n]

                    g = num / (1e-6 + denom)
                    trial_factors[r, k] = g if g > 0 else 0.0

            # Update templates
            elif q == 1:

                # Constrain shifts to be centered.
                shifts[r] = shifts[r] - np.mean(shifts[r])

                # Periodic boundary condition.
                if periodic_boundaries:

                    # Compute gram matrices.
                    c, a = periodic_shifts.sum_shift_grams(
                        trial_factors[r], shifts[r])

                    WtX.fill(0.0)
                    for k in range(K):
                        WtX += (
                            trial_factors[r, k] *
                            periodic_shifts.transpose_shift_one_trial(
                                R[k], shifts[r, k], w_template))

                    # Bound Lipshitz constant by Gersgorin Circle Theorem
                    L = c + 2 * a

                    # Update temporal factor by projected gradient descent.
                    ss = 0.95 / L

                    for itr in range(10):

                        # Update gradient with banded matrix multiply.
                        periodic_shifts.tri_sym_circ_matvec(
                            c, a, templates[r], w_template)
                        grad = w_template - WtX

                        # Projected gradient descent step.
                        for t in range(T):
                            for n in range(N):

                                templates[r, t, n] = \
                                    templates[r, t, n] - ss * grad[t, n]

                                if templates[r, t, n] < 0:
                                    templates[r, t, n] = 0.0

                # Padded boundary condition.
                else:

                    # Compute gram matrices.
                    padded_shifts.sum_shift_grams(
                        trial_factors[r], shifts[r], WtW)

                    WtX.fill(0.0)
                    for k in range(K):
                        WtX += (
                            trial_factors[r, k] *
                            padded_shifts.transpose_shift_one_trial(
                                R[k], shifts[r, k], w_template))

                    # Bound Lipshitz constant by Gersgorin Circle Theorem
                    L = max(WtW[1, 0] + WtW[0, 1], WtW[1, -1] + WtW[0, -1])
                    for i in range(1, T - 1):
                        Li = WtW[1, i] + WtW[0, i] + WtW[0, i + 1]
                        if Li > L:
                            L = Li

                    # Update temporal factor by projected gradient descent.
                    ss = 0.95 / L

                    for itr in range(10):

                        # Update gradient with banded matrix multiply.
                        padded_shifts.sym_bmat_mul(
                            WtW, templates[r], w_template)
                        grad = w_template - WtX

                        # Projected gradient descent step.
                        for t in range(T):
                            for n in range(N):

                                templates[r, t, n] = \
                                    templates[r, t, n] - ss * grad[t, n]

                                if templates[r, t, n] < 0:
                                    templates[r, t, n] = 0.0

            # Update shifts
            elif q == 2:
                for k in numba.prange(K):
                    shifts[r, k] = _fit_one_shift(
                        R[k], templates[r], shifts[r, k],
                        shift_reg, max_shift * T, periodic_boundaries,
                        warp_iterations)

        # Rebalance model.
        for r in range(rank):
            gnm = np.linalg.norm(trial_factors[r])
            trial_factors[r] *= np.sqrt(K) / gnm
            templates[r] *= gnm / np.sqrt(K)

        # Update model estimate for convergence check.
        _multishift_predict(
            trial_factors, templates, shifts, periodic_boundaries, Xest)

        # Test for convergence.
        itercount += 1

        loss = np.linalg.norm((X - Xest).ravel()) / Xnorm
        loss_hist.append(loss)

        reg = shift_reg * np.mean(np.abs(shifts.ravel()))
        reg_hist.append(reg)

        # Break loop if converged.
        if itercount > min_iter:
            c1 = abs(loss_hist[-patience] - loss_hist[-1]) < tol
            c2 = abs(reg_hist[-patience] - reg_hist[-1]) < tol
            converged = c1 and c2

    return trial_factors, templates, shifts, loss_hist, reg_hist


@numba.jit(nopython=True)
def _fit_one_shift(
        Rk, template, s0, sreg, max_shift, periodic, n_iter):

    w_template = np.empty_like(template)
    best_s = s0
    best_obj = np.inf
    best_loss = np.inf

    for i in range(n_iter):

        if i == 0:
            s = s0
        else:
            s = npr.uniform(-max_shift, max_shift)

        if periodic:
            periodic_shifts.shift_one_trial(template, s, w_template)
        else:
            padded_shifts.shift_one_trial(template, s, w_template)

        resid = (w_template - Rk).ravel()
        loss = np.dot(resid, resid)
        obj = loss + sreg * np.abs(s)

        if obj < best_obj:
            best_s = s
            best_loss = loss
            best_obj = obj

    return best_s
