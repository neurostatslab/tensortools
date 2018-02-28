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

    def __init__(self, X, factors, method, tol=1e-5, verbose=True, max_iter=500,
                 min_iter=1, max_time=np.inf, **kwargs):
        """

        Parameters
        ----------
        factors : Ktensor

        """

        self.normX = sci.linalg.norm(X)
        self.fit_history = []

        self.factors = factors

        self.method = method

        self.tol = tol
        self.verbose = verbose
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.max_time = max_time

        self.iterations = 0
        self.converged = False
        self.t0 = timeit.default_timer()

        # compute initial fit
        self.compute_fit(X)
        

    def time_elapsed(self):
        return timeit.default_timer()  - self.t0


    def update(self, factors_next, X):

        #~~~~~~~~~~~~~~~~~~~~~~~~~
        # Keep track of iterations
        #~~~~~~~~~~~~~~~~~~~~~~~~~
        self.iterations += 1

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute improvement in fit
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~
        old_fit = self.fit
        self.factors = factors_next
        new_fit = self.compute_fit(X)
        fit_improvement = np.abs(old_fit - new_fit) / (old_fit + 1.0) 



        #~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # If desired, print progress
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.verbose:
            p_args = self.method, self.iterations, self.fit, fit_improvement
            print('{}: iteration {}, fit {}, improvement {}.'.format(*p_args))

        #~~~~~~~~~~~~~~~~~~~~~~
        # Check for convergence
        #~~~~~~~~~~~~~~~~~~~~~~
        self.converged =\
            ( self.iterations > self.min_iter and fit_improvement < self.tol ) or\
            ( self.iterations > self.max_iter or self.time_elapsed() > self.max_time )

        return self
    
    
    def update2(self, obj):

        #~~~~~~~~~~~~~~~~~~~~~~~~~
        # Keep track of iterations
        #~~~~~~~~~~~~~~~~~~~~~~~~~
        self.iterations += 1

        if self.iterations == 1:
            self.obj = np.inf

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute improvement in fit
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~
        fit_improvement = np.abs(self.obj - obj) 
        
        # Update fit
        self.obj = obj


        #~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # If desired, print progress
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.verbose:
            p_args = self.method, self.iterations, self.obj, fit_improvement
            print('{}: iteration {}, objective {}, improvement {}.'.format(*p_args))

        #~~~~~~~~~~~~~~~~~~~~~~
        # Check for convergence
        #~~~~~~~~~~~~~~~~~~~~~~
        self.converged =\
            ( self.iterations > self.min_iter and fit_improvement < self.tol ) or\
            ( self.iterations > self.max_iter or self.time_elapsed() > self.max_time )

        return self    
    
    

    def compute_fit(self, X):
        """Updates quality of fit
        """
        self.fit = 1 - (sci.linalg.norm(X - self.factors.full()) / self.normX)
        return self.fit


    def finalize(self, X):
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set final time, final print statement
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.total_time = self.time_elapsed()
        self.fit = self.compute_fit(X)
        if self.verbose:
            print('Converged after {} iterations, {} seconds. Final fit {}.'.format(self.iterations, self.total_time, self.fit))

        return self
