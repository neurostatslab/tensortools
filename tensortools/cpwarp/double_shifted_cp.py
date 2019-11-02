"""
Implements warping and basic tensor functions.
"""
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import numba
import scipy as sci

from tensortools.cpwarp import padded_shifts
from tensortools.cpwarp import periodic_shifts


class TrialShiftedCP(object):
    """
    Represents third-order shifted tensor decomposition, with shifted
    components along the second dimension.

    Attributes
    ----------
    factors : list of ndarray
        List of three factor matrices. Shapes are (rank x num_trials),
        (rank x num_timepoints), and (rank x num_units).
    shifts : ndarray
        (rank x num_trials) matrix holding per-trial shift
        for each component.
    shape : tuple
        Dimensions of full tensor.
    size : int
        Number of elements in full tensor.
    rank : int
        Dimensionality of low-rank factor matrices. Number of model
        components
    """

    def __init__(self, factors, shifts=None, periodic_boundaries=False):
        """Initializes KTensor.

        Parameters
        ----------
        factors : list of ndarray
            List of three factor matrices. Shapes are (rank x num_trials),
            (rank x num_timepoints), and (rank x num_units).
        shifts : ndarray
            (num_components x num_trials) matrix holding per-trial shift
            for each component.
        """

        self.factors = factors
        self.shape = tuple([f.shape[1] for f in factors])
        self.ndim = len(self.shape)
        self.size = np.prod(self.shape)
        self.rank = factors[0].shape[0]
        self.periodic_boundaries = periodic_boundaries

        for f in factors[1:]:
            if f.shape[0] != self.rank:
                raise ValueError('Tensor factors have inconsistent rank.')

        if self.ndim != 3:
            raise ValueError(
                "ShiftedDecomposition only supports order-three tensors.")

        if shifts is None:
            self.shifts = np.zeros((self.rank, self.shape[0]))

        elif ((shifts.shape[0] != self.rank)
                or (shifts.shape[1] != self.shape[0])):
            raise ValueError(
                "Shift parameters should be provided as a (num_components "
                "x num_trials) matrix.")
        else:
            self.shifts = shifts

    def predict(self, skip_dims=None):
        """
        Forms model prediction.

        Parameters
        ----------
        skip_dim : None or set
            If provided, skips specified components.

        Returns
        -------
        est : ndarray
            Model estimate. Has shape (trials, timebins, units).
        """
        result = np.zeros(self.shape)
        g, u, v = self.factors

        # Determine components to skip.
        ranks = np.setdiff1d(np.arange(self.rank), skip_dims)

        # Compute prediction.
        _shiftcp3_predict(
            g[ranks],
            u[ranks],
            v[ranks],
            self.shifts[ranks],
            self.periodic_boundaries,
            result
        )

        return result

    def rebalance(self):
        """Rescales factors across modes so that all norms match.
        """

        # Compute norms along columns for each factor matrix
        norms = [np.linalg.norm(f, axis=1) for f in self.factors]

        # Multiply norms across all modes
        lam = sci.multiply.reduce(norms) ** (1/self.ndim)

        # Update factors
        self.factors = \
            [f * (lam / n)[:, None] for f, n in zip(self.factors, norms)]
        return self

    def permute(self, idx):
        """
        Permutes the components (rows of factor matrices) inplace.

        Parameters
        ----------
        idx : ndarray
            Permutation of components.
        """

        # Check that input is a true permutation
        if set(idx) != set(range(self.rank)):
            raise ValueError('Invalid permutation specified.')

        # Update factors
        self.factors = [f[idx] for f in self.factors]
        self.shifts = self.shifts[idx]
        return self.factors

    def plot(self, fig=None, axes=None, figsize=(10, 5)):
        if axes is None:
            fig, axes = plt.subplots(self.rank, 3, figsize=figsize)

        if self.rank == 1:
            axes = axes[None, :]

        for r in range(self.rank):
            for f, ax in zip(self.factors, axes[r]):
                ax.plot(f[r])

        return fig, axes

    def copy(self):
        return deepcopy(self)

    def __getitem__(self, i):
        return self.factors[i]

    def __setitem__(self, i, factor):
        factor = np.array(factor)
        if factor.shape != (self.shape[i], self.rank):
            raise ValueError('Dimension mismatch in KTensor assignment.')
        self.factors[i] = factor


@numba.jit(nopython=True, parallel=True)
def trialshift_ncp_hals(
        X, rank, init_g, init_u, init_v, min_iter=10,
        max_iter=1000, tol=1e-4, warp_iterations=50, max_shift=.5,
        shift_reg=1e-2, periodic_boundaries=False, patience=5,
        seed=None):
    """
    Fits shifted, semi-nonnegative CP-decomposition to a third-order tensor.
    Shifting occurs along axis=1. Axes 0 and 2 are constrained to be
    nonnegative.

    Uses coordinate descent updates (Hierarchical Alternating Least Squares;
    HALS) and a random search over the warping functions.

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
            _shiftcp3_predict(
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
        _shiftcp3_predict(
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
def _shiftcp3_predict(g, u, v, shifts, periodic, result, skip_dim=-1):
    """Model prediction for shifted CP decomposition."""
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
