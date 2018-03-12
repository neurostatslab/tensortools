from tensortools.optimize import cp_als
from tensortools.diagnostics import kruskal_align
from tqdm import trange
import numpy as np


class Ensemble(object):

    def __init__(self, nonneg=False, compress=False, sketching=False,
                 options=dict()):

        # model specification
        self.nonneg = nonneg
        self.compress = compress
        self.sketching = sketching

        # TODO - automatic selection of backend method
        self.fitting_method = cp_als

        # TODO - good defaults for optimization options
        options.setdefault('tol', 1e-5)
        options.setdefault('max_iter', 500)
        self.options = options

        # TODO - better way to hold all results...
        self.results = dict()

    def fit(self, X, ranks, replicates=1):
        """
        Fits CP tensor decompositions for different choices of rank.

        Parameters
        ----------
        X : array_like
            Real tensor

        ranks : int, or iterable
            iterable specifying number of components in each model

        replicates: int
            number of models to fit at each rank

        """

        # ~~~~~~~~~~~~~~~~~~~~~~~~
        # FIT MODELS FOR EACH RANK
        # ~~~~~~~~~~~~~~~~~~~~~~~~

        # make ranks iterable if necessary
        if not np.iterable(ranks):
            ranks = (ranks,)

        # fit models at each rank
        for r in ranks:

            # initialize result for rank-r
            if r not in self.results:
                self.results[r] = []

            # fit multiple replicates
            for i in trange(replicates):
                model_fit = self.fitting_method(X, r, **self.options)
                self.results[r].append(model_fit)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ALIGN FACTORS AND COMPUTER SIMILARITIES
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ranks = sorted(self.results, reverse=True)  # iterate in reverse order

        # sort models of same rank by objective function
        for rank in ranks:
            idx = np.argsort([r.obj for r in self.results[rank]])
            self.results[rank] = [self.results[rank][i] for i in idx]

        # align best model of rank r, to best model of next larger rank
        for r1, r2 in zip(ranks[1:], ranks):
            # note r1 < r2
            U = self.results[r1][0].factors
            V = self.results[r2][0].factors
            kruskal_align(U, V, permute_U=True)

        # for each rank, align everything to the best model
        for rank in ranks:
            # store best factors
            U = self.results[rank][0].factors       # best model factors
            self.results[rank][0].similarity = 1.0  # similarity to itself

            # align lesser fit models to best models
            for res in self.results[rank][1:]:
                res.similarity = kruskal_align(U, res.factors, permute_V=True)

    def objectives(self, rank):
        """Returns objective values of models with specified rank
        """
        self._check_rank(rank)
        return [r.obj for r in self.results[rank]]

    def similarities(self, rank):
        """Returns similarity scores for models with specified rank
        """
        self._check_rank(rank)
        return [r.similarity for r in self.results[rank]]

    def factors(self, rank):
        """Returns KTensor factors for models with specified rank
        """
        self._check_rank(rank)
        return [r.factors for r in self.results[rank]]

    def _check_rank(self, rank):
        """Checks if specified rank has been fit.
        """
        if rank not in self.results:
            raise ValueError('No models of rank-{} have been fit.' +
                             'Please call Ensemble.fit(...) first.')
