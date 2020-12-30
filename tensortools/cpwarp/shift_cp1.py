"""
Fits third-order shifted CP decomposition with shift
parameters along the first and second modes. See
fit_shift_cp1.py for shift parameters only along
the first mode.
"""
import numba
import numpy as np
import numpy.random as npr

from scipy.linalg import solveh_banded

from tensortools.cpwarp import padded_shifts
from tensortools.cpwarp import periodic_shifts

# At the moment, using parallelization tends to be slower,
# with numba.__version__ == "0.46.0".
USE_PARALLEL = False


@numba.jit(nopython=True, parallel=USE_PARALLEL, cache=True)
def fit_shift_cp1(
        X, Xnorm, rank, u, v, w, u_s, mask, min_iter=10,
        max_iter=1000, tol=1e-4, warp_iterations=10, max_shift_axis0=.1,
        u_nonneg=True, v_nonneg=True, periodic=False, patience=5):
    """
    Fits shifted, semi-nonnegative CP-decomposition to a third-order tensor.
    Shifting occurs along axis=2, with per-dimension shift parameters
    along axis=1. That is:

        X[i, j, t] \approx sum_r  u[r, i] * v[r, j] * w[r, t + s[r, i]]

    Where s[r, i] are shift parameters.

    For example, suppose N-dimensional time series is collected over K
    trials. Then the input tensor could have shape (N x K x T) to implement
    per-feature time-shifting along axis=2. Or, the input tensor could be
    (K x N x T) to implement per-trial shifting along axis=2.

    The algorithm uses coordinate descent updates and a random search over
    the shift parameters.
    """

    # Problem dimensions, norm of data.
    N, K, T = X.shape
    Xest = np.empty_like(X)

    if mask.size == (N * K * T):
        masked = True
    else:
        masked = False

    # Ensure at least two iterations for convergence check.
    min_iter = max(patience, min_iter)
    loss_hist = []

    # Preallocated space for intermediate computations
    WtW = np.empty((2, T))
    _WtW = np.empty((2, T))
    Ww = np.empty(T)
    v_num = np.empty(K)
    rhs = np.empty(T)

    # Set up progress bar.
    itercount = 0

    # === main loop === #
    converged = False

    while (itercount < max_iter) and not converged:

        # Update groups of parameters in random order.
        for z in npr.permutation(rank * 4):

            # Update component r.
            r = z // 4

            # Update residual tensor.
            predict(
                u, v, w, u_s, periodic, Xest, skip_dim=r)
            Z = X - Xest

            # Update one of the low-rank factors or shifts.
            q = z % 4

            # === UPDATE FACTOR WEIGHTS FOR AXIS 0 === #
            if q == 0:

                vtv = v[r] @ v[r]

                for n in range(N):

                    # shift w[r], store result in Ww.
                    if periodic:
                        periodic_shifts.apply_shift(
                            w[r], u_s[r, n], Ww)
                    else:
                        padded_shifts.apply_shift(
                            w[r], u_s[r, n], Ww)

                    num = v[r] @ (Z[n] @ Ww)
                    denom = vtv * (Ww @ Ww)
                    u[r, n] = num / denom

                # If u is all negative, flip sign of temporal factor.
                if np.all(u[r] < 0):
                    u[r] = -u[r]
                    w[r] = -w[r]

                # Project u onto nonnegative orthant.
                if u_nonneg:
                    u[r] = np.maximum(0, u[r])
                _prevent_zeros(u[r])

            # === UPDATE FACTOR WEIGHTS FOR AXIS 1 === #
            elif q == 1:

                v_num.fill(0.0)
                denom = 0.0

                for n in range(N):

                    # shift w[r], store result in Ww.
                    if periodic:
                        periodic_shifts.apply_shift(
                            w[r], u_s[r, n], Ww)
                    else:
                        padded_shifts.apply_shift(
                            w[r], u_s[r, n], Ww)

                    v_num += u[r, n] * (Z[n] @ Ww)
                    denom += u[r, n] * u[r, n] * (Ww @ Ww)

                v[r] = v_num / denom

                # If v is all negative, flip sign of temporal factor.
                if np.all(v[r] < 0):
                    v[r] = -v[r]
                    w[r] = -w[r]

                # Project v onto nonnegative orthant.
                if v_nonneg:
                    v[r] = np.maximum(0, v[r])
                _prevent_zeros(v[r])

            # === UPDATE AXIS WEIGHTS FOR AXIS 2 (temporal factors) === #
            elif q == 2:

                # Periodic boundary condition.
                if periodic:

                    # Holds diagonal and off diagonal of gram matrix.
                    d, off_d = 0.0, 0.0

                    # Holds right hand side of normal equations.
                    rhs.fill(0.0)

                    for n in range(N):
                        for k in range(K):

                            # Shift and weighting factor.
                            shift = u_s[r, n]
                            u_v = u[r, n] * v[r, k]
                            u_v2 = u_v * u_v

                            # Contribution to gram matrix.
                            _d, _off_d = periodic_shifts.shift_gram(shift)
                            d += u_v2 * _d
                            off_d += u_v2 * _off_d

                            # Contribution to right hand side.
                            rhs += u_v * periodic_shifts.trans_shift(
                                Z[n, k], shift, Ww)

                    # Update temporal factor w[r] by tridiag, circulant solver.
                    periodic_shifts.rojo_solve(d, off_d, rhs, w[r], Ww)

                # Padded boundary condition.
                else:

                    # Holds gram matrix.
                    WtW.fill(0.0)
                    WtW[-1] += 1e-8

                    # Holds right hand side of normal equations.
                    rhs.fill(0.0)

                    for n in range(N):
                        for k in range(K):
                            # Shift and weighting factor.
                            shift = u_s[r, n]
                            u_v = u[r, n] * v[r, k]
                            u_v2 = u_v * u_v

                            # Contribution to gram matrix.
                            WtW += u_v2 * \
                                padded_shifts.shift_gram(shift, T, _WtW)

                            # Contribution to right hand side.
                            rhs += u_v * padded_shifts.trans_shift(
                                Z[n, k], shift, Ww)

                    # Bound Lipshitz constant by Gersgorin Circle Theorem
                    L = max(WtW[1, 0] + WtW[0, 1], WtW[1, -1] + WtW[0, -1])
                    for i in range(1, T - 1):
                        Li = WtW[1, i] + WtW[0, i] + WtW[0, i + 1]
                        if Li > L:
                            L = Li

                    # Update temporal factor by gradient descent.
                    ss = 0.95 / L
                    for itr in range(10):

                        # Update gradient with banded matrix multiply.
                        ug = padded_shifts.sym_bmat_mul(WtW, w[r], Ww)
                        grad = ug - rhs

                        # Gradient descent step.
                        for t in range(T):
                            w[r, t] = max(0.0, w[r, t] - ss * grad[t])

                # Prevent divide by zero.
                _prevent_zeros(w[r])

            # === UPDATE SHIFT PARAMS FOR AXIS 0 === #
            elif q == 3:
                for n in numba.prange(N):
                    u_s[r, n] = _fit_shift(
                        Z[n], u[r, n], v[r], w[r],
                        max_shift_axis0 * T, periodic,
                        warp_iterations, u_s[r, n])

        # Update model estimate for convergence check.
        predict(
            u, v, w, u_s, periodic, Xest)

        # Update masked entries, if applicable.
        if masked:
            for n in range(N):
                for k in range(K):
                    for t in range(T):
                        if not mask[n, k, t]:
                            X[n, k, t] = Xest[n, k, t]

        # Test for convergence.
        itercount += 1
        loss = np.linalg.norm((X - Xest).ravel()) / Xnorm
        loss_hist.append(loss)

        # Break loop if converged.
        if itercount > min_iter:
            converged = abs(loss_hist[-patience] - loss_hist[-1]) < tol

    return u, v, w, u_s, loss_hist


@numba.jit(nopython=True, parallel=USE_PARALLEL, cache=True)
def predict(u, v, w, u_s, periodic, result, skip_dim=-1):

    N, K, T = result.shape
    rank = u.shape[0]
    result.fill(0.0)

    for k in numba.prange(K):
        for n in range(N):
            for r in range(rank):

                if r == skip_dim:
                    continue

                if periodic:
                    wshift = periodic_shifts.apply_shift(
                        w[r], u_s[r, n], np.empty(T))
                else:
                    wshift = padded_shifts.apply_shift(
                        w[r], u_s[r, n], np.empty(T))

                for t in range(T):
                    result[n, k, t] += wshift[t] * u[r, n] * v[r, k]

    return result


@numba.jit(nopython=True, cache=True)
def _fit_shift(
        Z, un, v, w, max_shift, periodic, n_iter, init_shift):
    """
    Z   : matrix, K x T
    un  : float
    v   : vector, length K
    w   : vector, length T
    """

    K, T = Z.shape
    ws = np.empty_like(w)
    best_loss = np.inf
    best_shift = 0.0

    Ztv = Z.T @ v
    vtv = v @ v

    for i in range(n_iter):

        # Sample new shift.
        if i == 0:
            s = init_shift
        else:
            s = npr.uniform(-max_shift, max_shift)

        # Apply shift for m-th element.
        if periodic:
            periodic_shifts.apply_shift(w, s, ws)
        else:
            padded_shifts.apply_shift(w, s, ws)

        # Compute loss.
        loss = vtv * (ws @ ws) - (ws @ Ztv)

        # Save best loss.
        if loss < best_loss:
            best_shift = s
            best_loss = loss

    return best_shift


@numba.jit(nopython=True, cache=True)
def _prevent_zeros(x):
    for xi in x:
        if abs(xi) > 1e-9:
            return None
    x[:] = np.random.rand(x.size)
