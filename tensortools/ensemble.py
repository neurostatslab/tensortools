from tensortools.optimize import cp_als
from tqdm import trange


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

    def fit(self, X, ranks, replicates):
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

        # make ranks iterable if necessary
        if not np.iterable(ranks):
            ranks = (ranks,)

        # iterate over ranks
        for r in self.ranks:

            if r not in self.results:
                self.results[r] = []

            for i in trange(self.replicates):
                results[r].append(self.fitting_method(data, r, **self.options))

    def arrange(self):
        # TODO - greedy alignment of factors across ranks
        pass
