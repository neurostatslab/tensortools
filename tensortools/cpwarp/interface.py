"""
Implements warping and basic tensor functions.
"""
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import numba
import scipy as sci

from tensortools.cpwarp import shift_cp2  # shift params for axis=(0, 1).
from tensortools.cpwarp import shift_cp1  # shift params for axis=0.


def fit_shifted_cp(
        X, rank, init_u=None, init_v=None, init_w=None,
        max_shift_axis0=None, max_shift_axis1=None,
        boundary="edge", min_iter=10, max_iter=1000, tol=1e-4,
        warp_iterations=50, patience=5):

    # Check inputs.
    if X.ndim != 3:
        raise ValueError(
            "Only 3rd-order tensors are supported for "
            "shifted decompositions.")

    if (max_shift_axis0 is None) and (max_shift_axis1 is None):
        raise ValueError(
            "Either `max_shift_axis0` or `max_shift_axis1` "
            "should be specified.")

    N, K, T = X.shape
    periodic = True if boundary == "wrap" else False

    # Initialize model parameters.
    if init_u is None:
        u = npr.rand(rank, N)
    else:
        u = np.copy(init_u)

    if init_v is None:
        v = npr.rand(rank, K)
    else:
        v = np.copy(init_v)

    if init_u is None:
        w = npr.rand(rank, T)
    else:
        w = np.copy(init_w)

    # Shifts per-unit and per-trial.
    shifting_0 = (max_shift_axis0 is not None) and (max_shift_axis0 > 0)
    shifting_1 = (max_shift_axis1 is not None) and (max_shift_axis1 > 0)

    if shifting_0:
        u_s = npr.uniform(
            -max_shift_axis0 * T,
            max_shift_axis0 * T, size=(rank, N))
    else:
        u_s = np.zeros((rank, N))

    if shifting_1:
        v_s = npr.uniform(
            -max_shift_axis1 * T,
            max_shift_axis1 * T, size=(rank, K))
    else:
        v_s = np.zeros((rank, K))

    # Compute model prediction.
    X_norm = np.linalg.norm(X)
    Xest_norm = np.linalg.norm(shift_cp2.predict(
        u, v, w, u_s, v_s, periodic, np.empty_like(X)))

    # Rescale initialization.
    alph = (X_norm / Xest_norm) ** (1. / 3.)
    u *= alph
    v *= alph
    w *= alph

    # Fit model.
    if shifting_0 and shifting_1:
        u, v, w, u_s, v_s, loss_hist = \
            shift_cp2.fit_shift_cp2(
                X, X_norm, rank, u, v, w, u_s, v_s,
                min_iter=min_iter,
                max_iter=max_iter,
                tol=tol,
                warp_iterations=warp_iterations,
                max_shift_axis0=max_shift_axis0,
                max_shift_axis1=max_shift_axis1,
                periodic=periodic,
                patience=patience
            )

    elif shifting_0:
        v_s = None
        u, v, w, u_s, loss_hist = \
            shift_cp1.fit_shift_cp1(
                X, X_norm, rank, u, v, w, u_s,
                min_iter=min_iter,
                max_iter=max_iter,
                tol=tol,
                warp_iterations=warp_iterations,
                max_shift=max_shift_axis0,
                periodic=periodic,
                patience=patience
            )

    elif shifting_1:
        u_s = None
        v, u, w, v_s, loss_hist = \
            shift_cp1.fit_shift_cp1(
                X.transpose((1, 0, 2)),
                X_norm, rank, v, u, w, v_s,
                min_iter=min_iter,
                max_iter=max_iter,
                tol=tol,
                warp_iterations=warp_iterations,
                max_shift=max_shift_axis1,
                periodic=periodic,
                patience=patience
            )

    return ShiftedCP(u, v, w, u_s, v_s, boundary, loss_hist=loss_hist)


class ShiftedCP(object):
    """
    Represents third-order shifted tensor decomposition, with
    shifted components along axis=-1.
    """

    def __init__(
            self, u, v, w, u_s=None, v_s=None,
            boundary="edge", loss_hist=None):
        """
        Parameters
        ----------
        u : ndarray
            First factor matrix. Has shape (rank, I).
        v : ndarray
            Second factor matrix. Has shape (rank, J).
        w : ndarray
            Third factor matrix. Has shape (rank, K).
        u_s : ndarray or None
            Shift parameters for each dimension of
            axis=0. Has shape (rank, I). Ignored if None.
        v_s : ndarray or None
            Shift parameters for each dimension of
            axis=1. Has shape (rank, J). Ignored if None.
        boundary : str
            Specifies boundary condition of shifting. If
            "edge", then the edge values of the array
            are used. If "wrap", then periodic boundary
            conditions are used. These are analogous to
            the `mode` parameter for numpy.pad(...).
        loss_hist : None or list
            Optional, holds history of objective function
            during optimization.
        """

        # Check factor matrix dimensions.
        if u.ndim != 2:
            raise ValueError("Factor matrix 'u' is not 2-dimensional.")
        if v.ndim != 2:
            raise ValueError("Factor matrix 'v' is not 2-dimensional.")
        if w.ndim != 2:
            raise ValueError("Factor matrix 'w' is not 2-dimensional.")

        # Gather factor matrices.
        self.factors = (u, v, w)

        # Set model rank.
        ranks = [f.shape[0] for f in self.factors]
        if np.unique(ranks).size > 1:
            raise ValueError("Tensor factors have inconsistent rank.")
        else:
            self.rank = ranks[0]

        # Set tensor dimensions.
        self.shape = tuple([f.shape[1] for f in self.factors])
        self.ndim = 3
        self.size = np.prod(self.shape)

        # Other information / parameters.
        self.loss_hist = loss_hist

        # Boundary parameters.
        if boundary not in ("edge", "wrap"):
            raise ValueError("Did not recognize boundary condition setting.")
        else:
            self.boundary = boundary

        # Check shift parameters along axis=0.
        if u_s is not None:
            if u_s.shape[0] != self.rank:
                raise ValueError("Parameter u_s has inconsistent rank.")
            elif u_s.shape[1] != self.shape[0]:
                raise ValueError("Parameter u_s has inconsistent dimension.")

        # Check shift parameters along axis=1.
        if v_s is not None:
            if v_s.shape[0] != self.rank:
                raise ValueError("Parameter v_s has inconsistent rank.")
            elif v_s.shape[1] != self.shape[1]:
                raise ValueError("Parameter v_s has inconsistent dimension.")

        self.u_s = u_s
        self.v_s = v_s

    def predict(self, skip_dims=None):
        """
        Forms model prediction.

        Parameters
        ----------
        skip_dim : None or int
            If provided, skips specified components.

        Returns
        -------
        est : ndarray
            Model estimate. Has shape (trials, timebins, units).
        """
        periodic = True if self.boundary == "wrap" else False
        u, v, w = self.factors

        if skip_dims is not None:
            idx = np.setdiff1d(
                np.arange(self.rank), skip_dims)
            u, v, w = u[idx], v[idx], w[idx]
            u_s = None if self.u_s is None else self.u_s[idx]
            v_s = None if self.v_s is None else self.v_s[idx]

        else:
            u_s, v_s = self.u_s, self.v_s

        # No shifting.
        if (u_s is None) and (v_s is None):
            return np.einsum("ir,jr,kr->ijk", u.T, v.T, w.T)

        # Shift parameters along both axis=(0, 1)
        elif (u_s is not None) and (v_s is not None):
            return shift_cp2.predict(
                u, v, w, u_s, v_s, periodic,
                np.empty(self.shape), skip_dim=-1)

        # Shift parameters along only axis=0
        elif v_s is None:
            return shift_cp1.predict(
                u, v, w, u_s, periodic,
                np.empty(self.shape), skip_dim=-1)

        # Shift parameters along only axis=1
        elif u_s is None:
            shape = [
                self.shape[1],
                self.shape[0],
                self.shape[2],
            ]
            Xest = shift_cp1.predict(
                v, u, w, v_s, periodic,
                np.empty(shape), skip_dim=-1)
            return Xest.transpose((1, 0, 2))

        else:
            assert False

    def prune_(self):
        """Drops any factors with zero magnitude."""
        idx = self.component_lams() > 0
        self.factors = tuple([f[idx] for f in self.factors])
        self.rank = np.sum(idx)

    def pad_zeros_(self, n):
        """Adds n more factors holding zeros."""
        if n == 0:
            return
        self.factors = tuple(
            [np.row_stack((f, np.zeros((f.shape[1], n))))
                for f in self.factors])
        self.rank += n

    def component_lams(self):
        """Returns norm of each component."""
        fnrms = np.column_stack(
            [np.linalg.norm(f, axis=1) for f in self.factors])
        return np.prod(fnrms, axis=1)

    def permute(self, idx):
        """
        Permutes the components (rows of factor matrices) inplace.

        Parameters
        ----------
        idx : ndarray
            Permutation of components.
        """
        # Check that input is a true permutation.
        if set(idx) != set(range(self.rank)):
            raise ValueError("Invalid permutation specified.")
        # Permute factors and shifts.
        self.factors = tuple([f[idx] for f in self.factors])

        if self.u_s is not None:
            self.u_s = self.u_s[idx]
        if self.v_s is not None:
            self.v_s = self.v_s[idx]

    def copy(self):
        return deepcopy(self)

    def __iter__(self):
        return iter([f.T for f in self.factors])  # used for plot_factors() func.
