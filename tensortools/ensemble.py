from tensortools import optimize
from tensortools.diagnostics import kruskal_align
from tqdm import trange
from collections.abc import Iterable
import numpy as np


class Ensemble(object):
    """
    Represents an ensemble of fitted tensor decompositions.
    """

    def __init__(self, nonneg=False, fit_method=None, fit_options=dict()):
        """Initializes Ensemble.

        Parameters
        ----------
        nonneg : bool
            If True, constrains low-rank factor matrices to be nonnegative.
        fit_method : None, str, callable, optional (default: None)
            Method for fitting a tensor decomposition. If input is callable,
            it is used directly. If input is a string then method is taken
            from tensortools.optimize using ``getattr``. If None, a reasonable
            default method is chosen.
        fit_options : dict
            Holds optional arguments for fitting method.
        """

        # Model parameters
        self._nonneg = nonneg

        # Determinine optimization method. If user input is None, try to use a
        # reasonable default. Otherwise check that it is callable.
        if fit_method is None:
            self._fit_method = optimize.ncp_bcd if nonneg else optimize.cp_als
        elif isinstance(fit_method, str):
            try:
                self._fit_method = getattr(optimize, fit_method)
            except AttributeError:
                raise ValueError("Did not recognize method 'fit_method' "
                                 "{}".format(fit_method))
        elif callable(fit_method):
            self._fit_method = fit_method
        else:
            raise ValueError("Expected 'fit_method' to be a string or "
                             "callable.")

        # Try to pick reasonable defaults for optimization options.
        fit_options.setdefault('tol', 1e-5)
        fit_options.setdefault('max_iter', 500)
        fit_options.setdefault('verbose', False)
        self._fit_options = fit_options

        # TODO - better way to hold all results...
        self.results = dict()

    def fit(self, X, ranks, replicates=1, verbose=True):
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
        verbose : bool
            If True, prints summaries and optimization progress.
        """

        # Make ranks iterable if necessary.
        if not isinstance(ranks, Iterable):
            ranks = (ranks,)

        # Iterate over model ranks, optimize multiple replicates at each rank.
        for r in ranks:

            # Initialize storage
            if r not in self.results:
                self.results[r] = []

            # Display fitting progress.
            if verbose:
                itr = trange(replicates,
                             desc='Fitting rank-{} models'.format(r),
                             leave=False)
            else:
                itr = range(replicates)

            # Fit replicates.
            for i in itr:
                model_fit = self._fit_method(X, r, **self._fit_options)
                self.results[r].append(model_fit)

            # Print summary of results.
            if verbose:
                itr.close()
                itr.refresh()
                min_obj = min([res.obj for res in self.results[r]])
                max_obj = max([res.obj for res in self.results[r]])
                elapsed = sum([res.total_time for res in self.results[r]])
                print('Rank-{} models:  min obj, {:.2f};  '
                      'max obj, {:.2f};  time to fit, '
                      '{:.1f}s'.format(r, min_obj, max_obj, elapsed), flush=True)

        # Sort results from lowest to largest loss.
        for r in ranks:
            idx = np.argsort([result.obj for result in self.results[r]])
            self.results[r] = [self.results[r][i] for i in idx]

        # Align best model within each rank to best model of next larger rank.
        # Here r0 is the rank of the lower-dimensional model and r1 is the rank
        # of the high-dimensional model.
        for i in reversed(range(1, len(ranks))):
            r0, r1 = ranks[i-1], ranks[i]
            U = self.results[r0][0].factors
            V = self.results[r1][0].factors
            kruskal_align(U, V, permute_U=True)

        # For each rank, align everything to the best model
        for r in ranks:
            # store best factors
            U = self.results[r][0].factors       # best model factors
            self.results[r][0].similarity = 1.0  # similarity to itself

            # align lesser fit models to best models
            for res in self.results[r][1:]:
                res.similarity = kruskal_align(U, res.factors, permute_V=True)

    def objectives(self, rank):
        """Returns objective values of models with specified rank.
        """
        self._check_rank(rank)
        return [result.obj for result in self.results[rank]]

    def similarities(self, rank):
        """Returns similarity scores for models with specified rank.
        """
        self._check_rank(rank)
        return [result.similarity for result in self.results[rank]]

    def factors(self, rank):
        """Returns KTensor factors for models with specified rank.
        """
        self._check_rank(rank)
        return [result.factors for result in self.results[rank]]

    def _check_rank(self, rank):
        """Checks if specified rank has been fit.

        Parameters
        ----------
        rank : int
            Rank of the models that were queried.

        Raises
        ------
        ValueError: If no models of rank ``rank`` have been fit yet.
        """
        if rank not in self.results:
            raise ValueError('No models of rank-{} have been fit.'
                             'Call Ensemble.fit(tensor, rank={}, ...) '
                             'to fit these models.'.format(rank))
