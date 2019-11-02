"""
Fits third-order shifted CP decomposition with shifts along a
single mode. For shifts along two modes, see fit_shift_cp02.py
"""
import numba
import numpy as np
import numpy.random as npr

from tensortools.cpwarp import padded_shifts
from tensortools.cpwarp import periodic_shifts


@numba.jit(nopython=True, parallel=True)
def fit_shift_cp0(
        X, rank, init_g, init_u, init_v, min_iter=10,
        max_iter=1000, tol=1e-4, warp_iterations=50, max_shift=.5,
        shift_reg=1e-2, periodic_boundaries=False, patience=5,
        seed=None):
    """
    Fits shifted, semi-nonnegative CP-decomposition to a third-order tensor.
    Shifting occurs along axis=1. Axes 0 and 2 are constrained to be
    nonnegative. Shift parameters are introduced along only axis=0.

    For example, if the input is a (trial x timebin x unit/feature) tensor,
    then per-trial shift parameters are fit. To also fit per-unit shift
    parameters, see `fit_shift_cp02.py`.

    Uses coordinate descent updates (Hierarchical Alternating Least Squares;
    HALS) and a random search over the shift parameters.

    Parameters
    ----------
    X : ndarray
        (trials x timebins x units) tensor of data.
    rank : int
        Number of model components.
    min_iter : int
        Minimum number of optimization iterations.
    max_iter : int
        Maximum number of optimization iterations.
    tol : float
        Convergence tolerance.
    warp_iterations : int
        Number of inner iterations for warping function updates.
    max_shift : float
        Largest allowable shift expressed as a fraction of trial length.
    shift_reg : float
        Strength of penalty on shift parameters.
    periodic_boundaries : bool
        If True, use periodic boundary condition on shifts.
    seed : int
        Seeds random initialization.
    verbose : bool
        Whether to print output.
    """

    # Problem dimensions, norm of data.
    K, T, N = X.shape
    Xnorm = np.linalg.norm(X.ravel())

    # Ensure at least two iterations for convergence check.
    min_iter = max(patience, min_iter)
    loss_hist = []
    reg_hist = []

    # Initialize model parameters.
    g = np.copy(init_g)
    u = np.copy(init_u)
    v = np.copy(init_v)
    shifts = npr.uniform(-1.0, 1.0, size=(rank, K))

    # Compute model prediction.
    Xest = np.full_like(X, np.nan)
    _shiftcp3_predict(
        g, u, v, shifts, periodic_boundaries, Xest)
    Xest_norm = np.linalg.norm(Xest.ravel())

    alph = (Xnorm / Xest_norm) ** (1. / 3.)
    g *= alph
    u *= alph
    v *= alph

    # Preallocated space for intermediate computations
    WtW = np.empty((2, T))
    WtB = np.empty(T)
    ur = np.empty((K, T))
    ug = np.empty(T)
    vg = np.empty(N)

    # Set up progress bar.
    itercount = 0

    # === main loop === #

    converged = False

    while (itercount < max_iter) and not converged:

        # Update parameters in random order.
        for z in npr.permutation(rank * 4):

            # Update component r.
            r = z // 4

            # Update residuals.
            predict(
                g, u, v, shifts, periodic_boundaries, Xest, skip_dim=r)
            R = X - Xest

            # Update one of the low-rank factors or shifts.
            q = z % 4

            # Update trial factors
            if q == 0:

                if periodic_boundaries:
                    periodic_shifts.shift_all_trials(u[r], shifts[r], ur)
                else:
                    padded_shifts.shift_all_trials(u[r], shifts[r], ur)

                vtv = np.dot(v[r], v[r])
                for k in range(K):
                    num = np.dot(ur[k].T, np.dot(R[k], v[r]))
                    denom = np.dot(ur[k], ur[k]) * vtv
                    g[r, k] = num / (1e-6 + denom)

                if np.all(g[r] < 0):
                    g[r] = -g[r]
                    u[r] = -u[r]

                g[r] = np.maximum(0, g[r])

            # Update neuron factors
            elif q == 1:

                if periodic_boundaries:
                    periodic_shifts.shift_all_trials(u[r], shifts[r], ur)
                else:
                    padded_shifts.shift_all_trials(u[r], shifts[r], ur)

                vg.fill(0.0)
                denom = 0.0
                for k in range(K):
                    vg += g[r, k] * np.dot(R[k].T, ur[k])
                    denom += (g[r, k] ** 2) * np.dot(ur[k], ur[k])

                if np.all(v[r] < 0):
                    v[r] = -v[r]
                    u[r] = -u[r]

                v[r] = np.maximum(0, vg / denom)

            # Update temporal factors
            elif q == 2:
                # Constrain shifts to be centered...
                shifts[r] = shifts[r] - np.mean(shifts[r])

                for k in range(K):
                    ur[k] = np.dot(R[k], v[r])

                # Periodic boundary condition.
                if periodic_boundaries:

                    # Compute gram matrices.
                    c, a = periodic_shifts.sum_shift_grams(
                        g[r], shifts[r])
                    periodic_shifts.sum_transpose_shift_all_trials(
                        g[r], ur, shifts[r], WtB)
                    vtv = np.dot(v[r], v[r])
                    c *= vtv
                    a *= vtv

                    # Bound Lipshitz constant by Gersgorin Circle Theorem
                    L = c + 2 * a

                    # Update temporal factor by projected gradient descent.
                    ss = 0.95 / L

                    for itr in range(10):

                        # Update gradient with banded matrix multiply.
                        periodic_shifts.tri_sym_circ_matvec(
                            c, a, u[r], ug)
                        grad = ug - WtB

                        # Projected gradient descent step.
                        for t in range(T):
                            u[r, t] = u[r, t] - ss * grad[t]
                            if u[r, t] < 0:
                                u[r, t] = 0.0

                # Padded boundary condition.
                else:

                    # Compute gram matrices.
                    padded_shifts.sum_shift_grams(g[r], shifts[r], WtW)
                    padded_shifts.sum_transpose_shift_all_trials(
                        g[r], ur, shifts[r], WtB)
                    WtW *= np.dot(v[r], v[r])

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
                        padded_shifts.sym_bmat_mul(WtW, u[r], ug)
                        grad = ug - WtB

                        # Projected gradient descent step.
                        for t in range(T):
                            u[r, t] = u[r, t] - ss * grad[t]
                            if u[r, t] < 0:
                                u[r, t] = 0.0

            # Update shifts
            elif q == 3:
                for k in numba.prange(K):
                    shifts[r, k] = _fit_one_shift(
                        R[k], g[r, k], u[r], v[r], shifts[r, k],
                        shift_reg, max_shift * T, periodic_boundaries,
                        warp_iterations)

        # Rebalance model.
        for r in range(rank):
            gnm = np.linalg.norm(g[r])
            unm = np.linalg.norm(u[r])
            vnm = np.linalg.norm(v[r])
            lam = (gnm * unm * vnm) ** (1 / 3)

            g[r] *= (lam / gnm)
            u[r] *= (lam / unm)
            v[r] *= (lam / vnm)

        # Update model estimate for convergence check.
        predict(
            g, u, v, shifts, periodic_boundaries, Xest)
        R = X - Xest

        # Test for convergence.
        itercount += 1

        loss = np.linalg.norm(R.ravel()) / Xnorm
        loss_hist.append(loss)

        reg = shift_reg * np.mean(np.abs(shifts.ravel()))
        reg_hist.append(reg)

        # Break loop if converged.
        if itercount > min_iter:
            c1 = abs(loss_hist[-patience] - loss_hist[-1]) < tol
            c2 = abs(reg_hist[-patience] - reg_hist[-1]) < tol
            converged = c1 and c2

    return g, u, v, shifts, loss_hist, reg_hist


@numba.jit(nopython=True, parallel=True)
def predict(g, u, v, shifts, periodic, result, skip_dim=-1):
    K, T, N = result.shape
    rank = len(g)
    result.fill(0.0)

    for k in numba.prange(K):
        for r in range(rank):

            if r == skip_dim:
                continue

            if periodic:
                uw = periodic_shifts.shift_one_trial(
                    u[r], shifts[r, k], np.empty(T))
            else:
                uw = padded_shifts.shift_one_trial(
                    u[r], shifts[r, k], np.empty(T))

            for t in range(T):
                for n in range(N):
                    result[k, t, n] += uw[t] * g[r, k] * v[r, n]


@numba.jit(nopython=True)
def _fit_one_shift(Rk, gk, u, v, s0, sreg, max_shift, periodic, n_iter):
    """
    Fits shift parameter on one trial.

    Parameters
    ----------
    Rk : ndarray
        Residual tensor from trial k. Has shape (timebins, units).
    gk : float
        Trial factor for component r on trial k.
    u : ndarray
        Temporal factor for component r. Has shape (timebins,).
    v : ndarray
        Unit/neural factor for component r. Has shape (units,).
    shifts : ndarray
        Per-trial shifts for component (trials,).
    sreg : float
        Shift penalty / regularization strength.
    max_shift : float
        Maximum shift.
    periodic : bool
        If True, use periodic boundary conditions.
    n_iter : int
        Number of search iterations.

    Returns
    -------
    loss : float
        Squared reconstruction error on trial k.
    """
    us = np.empty_like(u)
    Rv = np.dot(Rk, v)
    vtv = np.dot(v, v)
    best_s = s0
    best_obj = np.inf
    best_loss = np.inf
    gk2 = gk ** 2

    for i in range(n_iter):

        if i == 0:
            s = s0
        else:
            s = npr.uniform(-max_shift, max_shift)

        if periodic:
            periodic_shifts.shift_one_trial(u, s, us)
        else:
            padded_shifts.shift_one_trial(u, s, us)

        utu = np.dot(us, us)
        loss = (gk2 * utu * vtv) - 2 * gk * np.dot(us, Rv)
        obj = loss + sreg * np.abs(s)

        if obj < best_obj:
            best_s = s
            best_loss = loss
            best_obj = obj

    return best_s
