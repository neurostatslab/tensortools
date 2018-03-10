from tensortools.operations import khatri_rao
import numpy as np
import scipy as sci

class KTensor(object):
    """Kruskal tensor object
    """

    def __init__(self, factors):

        self.factors = factors
        self.shape = tuple([f.shape[0] for f in factors])
        self.ndim = len(self.shape)
        self.size = np.prod(self.shape)
        self.rank = factors[0].shape[1]

        for f in factors[1:]:
            if f.shape[1] != self.rank:
                raise ValueError('Tensor factors have inconsistent rank.')


    def full(self):
        
        # Compute tensor unfolding along first mode
        unf = sci.dot(self.factors[0], khatri_rao(self.factors[1:]).T)


        # Inverse unfolding along first mode
        return sci.reshape(unf, self.shape)

    def rebalance(self):
        """Rescales KTensor so that factors have equal magnitude across modes.
        """

        # Compute norms along columns for each factor matrix
        factor_norms = [sci.linalg.norm(f, axis=0) for f in self.factors]
        
        # Multiply norms across all modes
        lam = sci.multiply.reduce(factor_norms) ** (1/self.ndim)

        # Update factors
        self.factors = [f * (lam / fn) for f, fn in zip(self.factors, factor_norms)]
        return self.factors

    def permute(self, idx):
        """Permutes the columns of the factor matrices inplace
        """
        
        # Check that input is a true permutation
        if set(idx) != set(range(self.rank)):
            raise ValueError('Invalid permutation specified.')
        
        # Update factors
        self.factors = [f[:,idx] for f in self.factors]
        return self.factors


    def __getitem__(self, i):
        return self.factors[i]

    def __setitem__(self, i, factor):
        factor = sci.array(factor)
        if factor.shape != (self.shape[i], self.rank):
            raise ValueError('Dimension mismatch in KTensor assignment.')
        self.factors[i] = factor

    def __iter__(self):
        return iter(self.factors)
