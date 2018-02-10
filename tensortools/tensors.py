from .operations import khatri_rao
import numpy as np
import scipy as sci

class Ktensor(object):
    """Kruskal tensor object
    """

    def __init__(self, factors):

        self.factors = factors
        self.shape = tuple([factor.shape[0] for factor in factors])
        self.ndim = len(self.shape)
        self.size = np.prod(self.shape)

    def full(self):
        
        # Compute tensor unfolding along first mode
        unf = np.dot(factors[0], khatri_rao(factors[1:]).T)

        # Inverse unfolding along first mode
        return fold(unf, 0, self.shape)

    def rebalance(self):
        
        # Compute norms along columns for each factor matrix
        factor_norms = [sci.linalg.norm(f, axis=0) for f in self.factors]
        
        # Multiply norms across all modes
        lam = reduce(sci.multiply, factor_norms) ** (1 / self.ndim)

        # Update factors
        self.factors = [f * lam for f in self.factors]
        return self.factors

    def __getitem__(self, key):
        return self.factors[key]

    def __iter__(self):
        return iter(self.factors)
