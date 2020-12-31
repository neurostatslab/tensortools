"""
Implements warping and basic tensor functions.
"""
import matplotlib.pyplot as plt
import numpy as np
import numbers
from copy import deepcopy

from tensortools.cpwarp import shift_cp2  # shift params for axis=(0, 1).
from tensortools.cpwarp import shift_cp1  # shift params for axis=0.



def fit_shifted_cp(X, rank, n_restarts=1, **kwargs):
    """
    Fits a time-shifted tensor decomposition.
    """

    # Check that `n_restarts` is at least one.
    if (n_restarts < 1) or not isinstance(n_restarts, numbers.Integral):
        raise ValueError(
            "Expected `n_restarts` to be a nonnegative integer, but " + 
            f"saw n_restarts={n_restarts}."
        )

    # Handle multiple restarts with recursive calls.
    elif n_restarts > 1:
        best_loss = np.inf
        for i in range(n_restarts):
            m = fit_shifted_cp(X, rank, n_restarts=1, **kwargs)
            if m.loss_hist[-1] < best_loss:
                best_model = m
        return best_model

    # === 
    # Fit a single model.

    DEFAULTS = {
        "init_u": None,
        "init_v": None,
        "init_w": None,
        "max_shift_axis0": None,
        "max_shift_axis1": None,
        "u_nonneg": True,
        "v_nonneg": True,
        "boundary": "edge",
        "min_iter": 10,
        "max_iter": 10000,
        "tol": 1e-4,
        "warp_iterations": 10,
        "patience": 5,
        "mask": None,
    }

    # Check for unexpected keywords.
    for k in kwargs:
        if k not in DEFAULTS:
            raise TypeError(
                f"fit_shifted_cp() got an unexpected keyword argument '{k}'")

    # Set default keyword args.
    for k, v in DEFAULTS.items():
        if k not in kwargs:
            kwargs[k] = v

    # Keyword args to `fit_shift_cp1` or `fit_shift_cp2` 
    fit_args = {
        "periodic": True if kwargs["boundary"] == "wrap" else False,
    }
    for k in (
            "u_nonneg", "v_nonneg", "min_iter", "max_iter", "tol",
            "warp_iterations", "max_shift_axis0", "max_shift_axis1",
            "patience"):
        fit_args[k] = kwargs[k]

    # Check inputs.
    X = np.ascontiguousarray(X)
    if X.ndim != 3:
        raise ValueError(
            "Only 3rd-order tensors are supported for "
            "shifted decompositions.")

    if (kwargs["max_shift_axis0"] is None) and (kwargs["max_shift_axis1"] is None):
        raise ValueError(
            "Either `max_shift_axis0` or `max_shift_axis1` "
            "should be specified.")

    # Get tensor dimensions.
    N, K, T = X.shape

    # Initialize model parameters.
    if kwargs["init_u"] is None:
        u = np.random.rand(rank, N)
    else:
        u = np.copy(kwargs["init_u"])

    if kwargs["init_v"] is None:
        v = np.random.rand(rank, K)
    else:
        v = np.copy(kwargs["init_v"])

    if kwargs["init_w"] is None:
        w = np.random.rand(rank, T)
    else:
        w = np.copy(kwargs["init_w"])

    # Shifts per-unit and per-trial.
    shifting_0 = (kwargs["max_shift_axis0"] is not None) and (kwargs["max_shift_axis0"] > 0)
    shifting_1 = (kwargs["max_shift_axis1"] is not None) and (kwargs["max_shift_axis1"] > 0)

    if shifting_0:
        u_s = np.random.uniform(-.5, .5, size=(rank, N))
    else:
        u_s = np.zeros((rank, N))

    if shifting_1:
        v_s = np.random.uniform(-.5, .5, size=(rank, K))
    else:
        v_s = np.zeros((rank, K))

    # Compute model prediction.
    X_norm = np.linalg.norm(X)
    Xest_norm = np.linalg.norm(shift_cp2.predict(
        u, v, w, u_s, v_s, fit_args["periodic"], np.empty_like(X)))

    # Rescale initialization.
    alph = (X_norm / Xest_norm) ** (1. / 3.)
    u *= alph
    v *= alph
    w *= alph

    # Encode mask as 3D tensor
    if kwargs["mask"] is None:
        mask = np.array([[[0]]]).astype(bool)
    else:
        X = np.copy(X)
        mask = kwargs["mask"].astype(bool)
        assert mask.shape == X.shape

    # Fit model.
    if shifting_0 and shifting_1:
        # Shift both axis=0 and axis=1.
        u, v, w, u_s, v_s, loss_hist = \
            shift_cp2.fit_shift_cp2(
                X, X_norm, rank, u, v, w,
                u_s, v_s, mask,
                **fit_args
                # u_nonneg=u_nonneg,
                # v_nonneg=v_nonneg,
                # min_iter=min_iter,
                # max_iter=max_iter,
                # tol=tol,
                # warp_iterations=warp_iterations,
                # max_shift_axis0=max_shift_axis0,
                # max_shift_axis1=max_shift_axis1,
                # periodic=periodic,
                # patience=patience,
                # verbose=verbose,
            )

    elif shifting_0:
        # Shift only axis=0.
        v_s = None
        fit_args.pop("max_shift_axis1")
        u, v, w, u_s, loss_hist = \
            shift_cp1.fit_shift_cp1(
                X, X_norm, rank, u, v, w,
                u_s, mask, **fit_args
                # u_nonneg=u_nonneg,
                # v_nonneg=v_nonneg,
                # min_iter=min_iter,
                # max_iter=max_iter,
                # tol=tol,
                # warp_iterations=warp_iterations,
                # max_shift=max_shift_axis0,
                # periodic=periodic,
                # patience=patience
            )

    elif shifting_1:
        # Shift only axis=1.
        # 
        # Since `fit_shift_cp1` assumes that axis=0 is shifted
        # we need to transpose the arguments into `fit_shift_cp1`
        # and the un-transpose them on the other side.
        u_s = None
        transposed_args = deepcopy(fit_args)
        transposed_args["u_nonneg"] = fit_args["v_nonneg"]
        transposed_args["v_nonneg"] = fit_args["u_nonneg"]
        transposed_args["max_shift_axis0"] = transposed_args.pop("max_shift_axis1")
        v, u, w, v_s, loss_hist = \
            shift_cp1.fit_shift_cp1(
                np.copy(X.transpose((1, 0, 2))),
                X_norm, rank,
                v, u, w, # transposed.
                v_s, mask,
                **transposed_args
                # u_nonneg=v_nonneg, # transposed
                # v_nonneg=u_nonneg, # transposed
                # min_iter=min_iter,
                # max_iter=max_iter,
                # tol=tol,
                # warp_iterations=warp_iterations,
                # max_shift=max_shift_axis1, # transposed
                # periodic=periodic,
                # patience=patience
            )

    return ShiftedCP(
        u, v, w, u_s, v_s, kwargs["boundary"], loss_hist=loss_hist
    )


class ShiftedCP(object):
    """
    Represents third-order shifted tensor decomposition, with
    shifts added to axis=-1.

    Given a tensor X[i, j, t], the model estimate is:

        X_hat[i, j, t] = sum_r ( u[r, i] * v[r, j] * w[r, t + u_s[r, i] + v_s[r, i]] )

    Here, u_s[r, i] and v_s[r, i] are the shift parameters, and 
    the low-dimensional factors are { u[r, i], v[r, j], w[r, t] }.
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
