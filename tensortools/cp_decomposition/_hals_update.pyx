# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# See https://github.com/erichson/ristretto
# And https://github.com/scikit-learn/scikit-learn
# License: BSD 3 clause

cimport cython
from libc.math cimport fabs

def _hals_update(double[:, ::1] U, double[:, :] grams, double[:, :] XU):
    
    cdef Py_ssize_t m = U.shape[0]  
    cdef Py_ssize_t n = U.shape[1]
    cdef Py_ssize_t i, r, t
    cdef double gradient, projected_gradient
    cdef double violation = 0
    
    with nogil:
        for t in xrange(n):
            for i in xrange(m):
                
                # Gradient
                gradient = -XU[i, t]

                for r in xrange(n):
                    gradient += grams[t, r] * U[i, r]

                # projected gradient
                projected_gradient = min(0.0, gradient) if U[i, t] == 0 else gradient
                violation += fabs(projected_gradient)    

                # Update Factor matrix 
                if grams[t, t] != 0:
                    U[i, t] = max(0.0, U[i, t] - gradient / grams[t, t])   

    return violation