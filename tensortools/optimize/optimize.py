"""
Optimization objects
"""

import numpy as np
import scipy as sci
import timeit

class FitResult(object):

    def __init__(self, X, U, method, tol=1e-5, verbose=True, max_iter=500, min_iter=1):

        self.X = X
        self.normX = sci.linalg.norm(X)
        self.fit_history = []

        self.method = method

        self.tol = tol
        self.verbose = verbose
        self.max_iter = max_iter
        self.min_iter = min_iter

        self.iterations = 0

    def time_elapsed(self):
        return timeit.default_timer()  - self.t0

    def update(self, Unext):

        #~~~~~~~~~~~~~~~~~~~~~~~~~
        # Keep track of iterations
        #~~~~~~~~~~~~~~~~~~~~~~~~~
        self.iterations += 1
        self.time_elapsed = timeit.default_timer() - self.t0

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute improvement in fit
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~
        old_fit = self.fit
        self.U = Unext
        self.compute_fit() # updates fit
        fit_improvement = old_fit - self.fit

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # If desired, print progress
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.verbose:
            print('{}: iteration {}, fit {}, improvement {}.'.format(self.method, self.fit, fit_improvement))

        #~~~~~~~~~~~~~~~~~~~~~~
        # Check for convergence
        #~~~~~~~~~~~~~~~~~~~~~~
        self.converged =\
            ( self.iterations > min_iter and fit_improvement < self.tol ) or\
            ( self.iterations > max_iter or self.time_elapsed() > self.max_time )

        return self

    def compute_fit(self):

        # TODO: more efficient estimations.
        return 1 - (self.normX / sci.linalg.norm(self.X - self.U.full()))

    def finalize(self):
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set final time, final print statement
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.time_elapsed = np.round(timeit.default_timer() - self.t0, 1)

        if verbose:
            print('Converged after {} iterations, {} seconds.'.format(self.iterations, self.time_elapsed))

        return self
