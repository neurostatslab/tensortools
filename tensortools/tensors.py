from tensortools.operations import khatri_rao
import numpy as np
from copy import deepcopy


class KTensor(object):
    """Kruskal tensor object.

    Attributes
    ----------
    factors : list of ndarray
        Factor matrices.
    shape : tuple
        Dimensions of full tensor.
    size : int
        Number of elements in full tensor.
    rank : int
        Dimensionality of low-rank factor matrices.
    """

    def __init__(self, factors):
        """Initializes KTensor.

        Parameters
        ----------
        factors : list of ndarray
            Factor matrices.
        """

        self.factors = factors
        self.shape = tuple([f.shape[0] for f in factors])
        self.ndim = len(self.shape)
        self.size = np.prod(self.shape)
        self.rank = factors[0].shape[1]

        for f in factors[1:]:
            if f.shape[1] != self.rank:
                raise ValueError('Tensor factors have inconsistent rank.')

    def full(self):
        """Converts KTensor to a dense ndarray."""

        # Compute tensor unfolding along first mode
        unf = self.factors[0] @ khatri_rao(self.factors[1:]).T

        # Inverse unfolding along first mode
        return np.reshape(unf, self.shape)

    def norm(self):
        """Efficiently computes Frobenius-like norm of the tensor."""
        C = np.prod([F.T @ F for F in self.factors], axis=0)
        return np.sqrt(np.sum(C))

    def rebalance(self):
        """Rescales factors across modes so that all norms match."""

        # Compute norms along columns for each factor matrix
        norms = [np.linalg.norm(f, axis=0) for f in self.factors]

        # Multiply norms across all modes
        lam = np.prod(norms, axis=0) ** (1/self.ndim)

        # Update factors
        self.factors = [f * (lam / fn) for f, fn in zip(self.factors, norms)]
        return self

    def prune_(self):
        """Drops any factors with zero magnitude."""
        idx = self.component_lams() > 0
        self.factors = [f[:, idx] for f in self.factors]
        self.rank = np.sum(idx)

    def pad_zeros_(self, n):
        """Adds n more factors holding zeros."""
        if n == 0:
            return
        self.factors = [np.column_stack((f, np.zeros((f.shape[0], n))))
                        for f in self.factors]
        self.rank += n

    def permute(self, idx):
        """Permutes the columns of the factor matrices inplace."""

        # Check that input is a true permutation
        if set(idx) != set(range(self.rank)):
            raise ValueError('Invalid permutation specified.')

        # Update factors
        self.factors = [f[:, idx] for f in self.factors]
        return self.factors

    def component_lams(self):
        fnrms = np.column_stack(
            [np.linalg.norm(f, axis=0) for f in self.factors])
        return np.prod(fnrms, axis=1)

    def factor_lams(self):
        import warnings
        warnings.warn(
            "KTensor.factor_lams() has been deprecated. Use "
            "KTensor.component_lams() instead.",
            DeprecationWarning)
        return self.component_lams()

    def copy(self):
        return deepcopy(self)

    def __getitem__(self, i):
        return self.factors[i]

    def __setitem__(self, i, factor):
        factor = np.array(factor)
        if factor.shape != (self.shape[i], self.rank):
            raise ValueError('Dimension mismatch in KTensor assignment.')
        self.factors[i] = factor

    def __iter__(self):
        return iter(self.factors)
