from tensortools.optimize import cp_als

class Ensemble(object):

    def __init__(self, nonneg=False, compress=False, sketching=False):

        # model specification
        self.nonneg = nonneg
        self.compress = compress
        self.sketching = sketching

        # TODO - automatic selection of method
        self.fitting_method = cp_als

        # TODO - good defaults for optimization options
        self.options = {
            'tol': 1e-5,
            'max_iter': 500
        }

        # TODO - better way to hold all results...
        self.results = dict()

    def fit(self, data, ranks, replicates):

        for r in self.ranks:

            if r not in self.results:
                self.results[r] = []

            for i in range(self.replicates):
                results[r].append(self.fitting_method(data, r, **self.options))

    def arrange(self):
        # TODO - greedy alignment of factors across ranks
        pass

