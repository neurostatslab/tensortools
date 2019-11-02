"""
Implements warping and basic tensor functions.
"""
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import numba
import scipy as sci

from tensortools.cpwarp import shift_cp2  # shift params for axis=(0, 1).
# from tensortools.cpwarp import shift_cp1  # shift params for axis=0.


def fit_shifted_cp(
        X, rank, init_u=None, init_v=None, init_w=None,
        shift_axis=2, shift_params_along=(0, 1), boundary="edge",
        min_iter=10, max_iter=1000, tol=1e-4, warp_iterations=50,
        max_shift=.1, periodic=False, patience=5):

    # Check inputs.
    if X.ndim != 3:
        raise ValueError(
            "Only 3rd-order tensors are supported for "
            "shifted decompositions.")

    I, J, K = X.shape
    periodic = True if boundary == "wrap" else False

    # Initialize model parameters.
    if init_u is None:
        u = npr.rand(rank, I)
    else:
        u = np.copy(init_u)

    if init_v is None:
        v = npr.rand(rank, J)
    else:
        v = np.copy(init_v)

    if init_u is None:
        w = npr.rand(rank, K)
    else:
        w = np.copy(init_w)

    # Shifts per-unit and per-trial.
    u_s = npr.uniform(-1.0, 1.0, size=(rank, I))
    v_s = npr.uniform(-1.0, 1.0, size=(rank, J))

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
    u, v, w, u_s, v_s, loss_hist = \
        shift_cp2.fit_shift_cp2(
            X, X_norm, rank, u, v, w, u_s, v_s,
            min_iter=min_iter,
            max_iter=max_iter,
            tol=tol,
            warp_iterations=warp_iterations,
            max_shift=max_shift,
            periodic=periodic,
            patience=patience
        )

    return ShiftedCP(u, v, w, u_s, v_s, boundary)



class ShiftedCP(object):
    """
    Represents third-order shifted tensor decomposition, with
    shifted components along axis=-1.
    """

    def __init__(self, u, v, w, u_s=None, v_s=None, boundary="edge"):
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

        # Boundary parameters.
        if boundary not in ("edge", "wrap"):
            raise ValueError("Did not recognize boundary condition setting.")
        else:
            self.boundary = boundary

        # Shift parameters along axis=0.
        if u_s is not None:
            if u_s.shape[0] != self.rank:
                raise ValueError("Parameter u_s has inconsistent rank.")
            elif u_s.shape[1] != self.shape[0]:
                raise ValueError("Parameter u_s has inconsistent dimension.")
            else:
                self.u_s = u_s

        # Shift parameters along axis=1.
        if v_s is not None:
            if v_s.shape[0] != self.rank:
                raise ValueError("Parameter u_s has inconsistent rank.")
            elif v_s.shape[1] != self.shape[1]:
                raise ValueError("Parameter u_s has inconsistent dimension.")
            else:
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
        if (self.u_s is None) and (self.v_s is None):
            return np.einsum("ir,jr,kr->ijk", u.T, v.T, w.T)

        # Shift parameters along both axis=(0, 1)
        elif (self.u_s is not None) and (self.v_s is not None):
            return shift_cp2.predict(
                u, v, w, u_s, v_s, periodic,
                np.empty(self.shape), skip_dim=-1)

        # Shift parameters along only axis=0
        elif self.v_s is None:
            raise NotImplementedError()

        # Shift parameters along only axis=1
        elif self.u_s is None:
            raise NotImplementedError()

        else:
            assert False

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
        self.factors = [f[idx] for f in self.factors]
        self.u_s, self.v_s = self.u_s[idx], self.v_s[idx]

    def copy(self):
        return deepcopy(self)
