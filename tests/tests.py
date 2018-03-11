from __future__ import division
import numpy as np
import scipy as sci

from tensortools.optimize import cp_als, ncp_hals, ncp_bcd, cp_opt
from tensortools.operations import khatri_rao
from tensortools.data import randn_tensor, rand_tensor
from tensortools.diagnostics import kruskal_align

from unittest import main, makeSuite, TestCase, TestSuite
from numpy.testing import assert_raises, assert_equal

from copy import deepcopy

atol_float32 = 1e-4
atol_float64 = 1e-8

random_state = 123

#
#******************************************************************************
#
class test_base(TestCase):
    def setUp(self):
        np.random.seed(123)

    def test_khatrirao(self):
        A = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
        ])

        B = np.array([
        [1, 4, 7],
        [2, 5, 8],
        [3, 6, 9]
        ])

        C = np.array([
        [1, 8, 21],
        [2, 10, 24],
        [3, 12, 27],
        [4, 20, 42],
        [8, 25, 48],
        [12, 30, 54],
        [7, 32, 63],
        [14, 40, 72],
        [21, 48, 81]
        ])

        assert np.allclose(khatri_rao((A, B)), C, atol_float64)


#
#******************************************************************************
#
class test_diagnostics(TestCase):
    def setUp(self):
        np.random.seed(123)

    def test_align(self):
        I,J,K = 15,15,15
        U = randn_tensor((I,J,K), rank=3, ktensor=True)
        V = deepcopy(U)

        assert (kruskal_align(U, V) - 1) < atol_float64


#
#******************************************************************************
#
class test_cp(TestCase):
    def setUp(self):
        np.random.seed(123)

    def test_cp_als_deterministic(self):
        I,J,K,R = 15,15,15,3
        X = randn_tensor((I,J,K), rank=R, random_state=random_state)
        P = cp_als(X, rank=R, trace=False, random_state=random_state)

        percent_error = sci.linalg.norm(P.factors.full() - X) / sci.linalg.norm(X)
        assert percent_error < atol_float32


    def test_cp_opt_deterministic(self):
        I,J,K,R = 10,10,10,2
        X = randn_tensor((I,J,K), rank=R, random_state=random_state)
        P = cp_opt(X, rank=R, trace=False, method='BFGS', random_state=random_state)

        percent_error = sci.linalg.norm(P.factors.full() - X) / sci.linalg.norm(X)
        assert percent_error < atol_float32


        P = cp_opt(X, rank=R, trace=False, method='Newton-CG', random_state=random_state)

        percent_error = sci.linalg.norm(P.factors.full() - X) / sci.linalg.norm(X)
        assert percent_error < atol_float32


#
#******************************************************************************
#
class test_nonnegative_cp(TestCase):
    def setUp(self):
        np.random.seed(123)

    def test_ncp_hals_deterministic(self):
        I,J,K,R = 15,15,15,3
        X = rand_tensor((I,J,K), rank=R, random_state=random_state)
        P = ncp_hals(X, rank=R, trace=False, random_state=random_state)

        NN = np.sum(P.factors.full() < 0)
        assert NN == 0

        percent_error = sci.linalg.norm(P.factors.full() - X) / sci.linalg.norm(X)
        assert percent_error < atol_float32



    def test_ncp_bcd_deterministic(self):
        I,J,K,R = 15,15,15,3
        X = rand_tensor((I,J,K), rank=R, random_state=random_state)
        P = ncp_bcd(X, rank=R, trace=False, random_state=random_state)

        NN = np.sum(P.factors.full() < 0)
        assert NN == 0

        percent_error = sci.linalg.norm(P.factors.full() - X) / sci.linalg.norm(X)
        assert percent_error < atol_float32



#
#******************************************************************************
#

def suite():
    s = TestSuite()
    s.addTest(test_base('test_khatrirao'))
    s.addTest(test_cp('test_cp_als_deterministic'))
    s.addTest(test_nonnegative_cp('test_ncp_hals_deterministic'))
    s.addTest(test_nonnegative_cp('test_ncp_bcd_deterministic'))



    return s

if __name__ == '__main__':
    main(defaultTest = 'suite')
