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


# @numba.jit(nopython=True, parallel=True)
def fit_shift_cp2(
        X, Xnorm, rank, u, v, w, u_s, v_s, min_iter=10,
        max_iter=1000, tol=1e-4, warp_iterations=50, max_shift=.1,
        periodic=False, patience=5):
    """
    Fits shifted, semi-nonnegative CP-decomposition to a third-order tensor.
    Shifting occurs along axis=-1, with per-dimension shift parameters
    along axes 1 and 2. That is:

        X[i, j, t] \approx sum_r  u[r, i] * v[r, j] + w[r, t + s_i + z_j]

    Where s_i and z_j are shift parameters.

    For example, suppose N-dimensional time series is collected over K
    trials. Then the input tensor could have shape (N x K x T) to implement
    per-trial and per-feature time-shifting along the final axis.

    The algorihm uses coordinate descent updates and a random search over
    the shift parameters.

    Parameters
    ----------
    X : ndarray
        Third-order tensor of data.
    rank : int
        Number of low-dimensional components.
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
    periodic : bool
        If True, use periodic boundary condition on shifts.
    seed : int
        Seeds random initialization.
    verbose : bool
        Whether to print output.
    """

    # Problem dimensions, norm of data.
    N, K, T = X.shape
    Xest = np.empty_like(X)

    # Ensure at least two iterations for convergence check.
    min_iter = max(patience, min_iter)
    loss_hist = []

    # Preallocated space for intermediate computations
    WtW = np.empty((2, T))
    _WtW = np.empty((2, T))
    Ww = np.empty(T)
    rhs = np.empty(T)

    # Set up progress bar.
    itercount = 0

    # === main loop === #

    converged = False

    while (itercount < max_iter) and not converged:

        print("itercount: ", itercount)
        if itercount > 0:
            print("-> ", loss)

        # Update groups of parameters in random order.
        for z in npr.permutation(rank * 5):

            # Update component r.
            r = z // 5

            # Update residual tensor.
            predict(
                u, v, w, u_s, v_s, periodic, Xest, skip_dim=r)
            Z = X - Xest

            # Update one of the low-rank factors or shifts.
            q = z % 5

            # === UPDATE FACTOR WEIGHTS FOR AXIS 0 === #
            if q == 0:

                if periodic:
                    for n in range(N):

                        num, denom = 0.0, 0.0

                        for k in range(K):
                            # shift w[r], store result in Ww.
                            periodic_shifts.apply_shift(
                                w[r], u_s[r, n] + v_s[r, k], Ww)
                            num += v[r, k] * (Z[n, k] @ Ww)
                            denom += v[r, k] * v[r, k] * (Ww @ Ww)

                        u[r, n] = num / denom

                else:
                    for n in range(N):

                        num, denom = 0.0, 0.0

                        for k in range(K):
                            # shift w[r], store result in Ww.
                            padded_shifts.apply_shift(
                                w[r], u_s[r, n] + v_s[r, k], Ww)
                            num += v[r, k] * (Z[n, k] @ Ww)
                            denom += v[r, k] * v[r, k] * (Ww @ Ww)

                        u[r, n] = num / denom

                # If u is all negative, flip sign of temporal factor.
                if np.all(u[r] < 0):
                    u[r] = -u[r]
                    w[r] = -w[r]

                # Project u onto nonnegative orthant.
                u[r] = np.maximum(0, u[r])

            # === UPDATE FACTOR WEIGHTS FOR AXIS 1 === #
            elif q == 1:

                if periodic:
                    for k in range(K):

                        num, denom = 0.0, 0.0

                        for n in range(N):
                            # shift w[r], store result in Ww.
                            periodic_shifts.apply_shift(
                                w[r], u_s[r, n] + v_s[r, k], Ww)
                            num += u[r, n] * (Z[n, k] @ Ww)
                            denom += u[r, n] * u[r, n] * (Ww @ Ww)

                        v[r, k] = num / denom

                else:
                    for k in range(K):

                        num, denom = 0.0, 0.0

                        for n in range(N):
                            # shift w[r], store result in Ww.
                            padded_shifts.apply_shift(
                                w[r], u_s[r, n] + v_s[r, k], Ww)
                            num += u[r, n] * (Z[n, k] @ Ww)
                            denom += u[r, n] * u[r, n] * (Ww @ Ww)

                        v[r, k] = num / denom

                # If v is all negative, flip sign of temporal factor.
                if np.all(v[r] < 0):
                    v[r] = -v[r]
                    w[r] = -w[r]

                # Project v onto nonnegative orthant.
                v[r] = np.maximum(0, v[r])

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
                            shift = u_s[r, n] + v_s[r, k]
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
                            shift = u_s[r, n] + v_s[r, k]
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
                            w[r, t] = w[r, t] - ss * grad[t]

            # === UPDATE SHIFT PARAMS FOR AXIS 0 === #
            elif q == 3:
                for n in numba.prange(N):
                    u_s[r, n] = _fit_shift(
                        Z[n], u[r, n], v[r], v_s[r], w[r],
                        max_shift * T, periodic,
                        warp_iterations, u_s[r, n])

            # === UPDATE SHIFT PARAMS FOR AXIS 1 === #
            elif q == 4:
                for k in numba.prange(K):
                    v_s[r, k] = _fit_shift(
                        Z[:, k], v[r, k], u[r], u_s[r], w[r],
                        max_shift * T, periodic,
                        warp_iterations, v_s[r, k])

        # # Rebalance model.
        # for r in range(rank):
        #     unm = np.linalg.norm(u[r])
        #     vnm = np.linalg.norm(v[r])
        #     wnm = np.linalg.norm(w[r])
        #     lam = (unm * vnm * wnm) ** (1 / 3)

        #     u[r] *= (lam / unm)
        #     v[r] *= (lam / vnm)
        #     w[r] *= (lam / wnm)

        # Update model estimate for convergence check.
        predict(
            u, v, w, u_s, v_s, periodic, Xest)

        # Test for convergence.
        itercount += 1

        loss = np.linalg.norm((X - Xest).ravel()) / Xnorm
        loss_hist.append(loss)

        # Break loop if converged.
        if itercount > min_iter:
            converged = abs(loss_hist[-patience] - loss_hist[-1]) < tol

    return u, v, w, u_s, v_s, loss_hist


# @numba.jit(nopython=True, parallel=True)
def predict(u, v, w, u_s, v_s, periodic, result, skip_dim=-1):

    N, K, T = result.shape
    rank = u.shape[0]
    result.fill(0.0)

    for k in numba.prange(K):
        for n in range(N):
            for r in range(rank):

                if r == skip_dim:
                    continue

                shift = u_s[r, n] + v_s[r, k]

                if periodic:
                    wshift = periodic_shifts.apply_shift(
                        w[r], shift, np.empty(T))
                else:
                    wshift = padded_shifts.apply_shift(
                        w[r], shift, np.empty(T))

                for t in range(T):
                    result[n, k, t] += wshift[t] * u[r, n] * v[r, k]

    return result


# @numba.jit(nopython=True)
def _fit_shift(
        Z, y, f, f_s, w, max_shift, periodic, n_iter, init_shift):
    """
    Z   : matrix, M x T
    y   : float
    f   : vector, length M
    f_s : vector, length M
    w   : vector, length T
    """

    M, T = Z.shape
    ws = np.empty_like(w)
    best_loss = np.inf
    best_shift = 0.0

    for i in range(n_iter):

        # Sample new shift.
        if i == 0:
            s = init_shift
        else:
            s = npr.uniform(-max_shift, max_shift)

        # Apply shifts.
        loss = 0.0
        for m in range(M):

            # Apply shift for m-th element.
            if periodic:
                periodic_shifts.apply_shift(w, s + f_s[m], ws)
            else:
                padded_shifts.apply_shift(w, s + f_s[m], ws)

            # Compute loss for m-th element.
            for t in range(T):
                resid = Z[m, t] - (y * f[m] * ws[t])
                loss += resid * resid

            # Can stop early due to nonnegative loss.
            if loss > best_loss:
                break

        # Save best loss.
        if loss < best_loss:
            best_shift = s
            best_loss = loss

    return best_shift
