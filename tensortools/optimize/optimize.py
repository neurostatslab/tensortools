"""
Optimization objects
"""

import numpy as np
import scipy as sci
import timeit


class FitResult(object):
    """
    Holds result of optimization
    """

    def __init__(self, factors, method, tol=1e-5, verbose=True, max_iter=500,
                 min_iter=1, max_time=np.inf, **kwargs):
        """

        Parameters
        ----------
        factors : KTensor

        """

        self.factors = factors
        self.obj = np.inf
        self.obj_hist = []
        self.method = method

        self.tol = tol
        self.verbose = verbose
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.max_time = max_time

        self.iterations = 0
        self.converged = False
        self.t0 = timeit.default_timer()

    def time_elapsed(self):
        return timeit.default_timer() - self.t0

    def update(self, obj):

        # ~~~~~~~~~~~~~~~~~~~~~~~~
        # Keep track of iterations
        # ~~~~~~~~~~~~~~~~~~~~~~~~
        self.iterations += 1

        if self.iterations == 1:
            self.obj = np.inf

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute improvement in objective
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        improvement = self.obj - obj
        # assert improvement > 0
        self.obj = obj
        self.obj_hist.append(obj)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~
        # If desired, print progress
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.verbose:
            p_args = self.method, self.iterations, self.obj, improvement
            s = '{}: iteration {}, objective {}, improvement {}.'
            print(s.format(*p_args))

        # ~~~~~~~~~~~~~~~~~~~~~~
        # Check for convergence
        # ~~~~~~~~~~~~~~~~~~~~~~
        self.converged =\
            (self.iterations > self.min_iter and improvement < self.tol) or\
            (self.iterations > self.max_iter or self.time_elapsed() > self.max_time)

        return self

    def finalize(self):

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set final time, final print statement
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.total_time = self.time_elapsed()

        if self.verbose:
            s = 'Converged after {} iterations, {} seconds. Objective: {}.'
            print(s.format(self.iterations, self.total_time, self.obj))

        return self
