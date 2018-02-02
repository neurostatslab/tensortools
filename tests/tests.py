from __future__ import division
import numpy as np
import scipy as sci


from tensortools.cp_decomposition import cp_als, cp_opt
from tensortools.tensor_utils import khatri_rao, norm
from tensortools.tensor_utils import kruskal_to_tensor, kruskal_to_unfolded
from tensortools.data import cp_tensor


from unittest import main, makeSuite, TestCase, TestSuite
from numpy.testing import assert_raises, assert_equal

atol_float32 = 1e-4
atol_float64 = 1e-8

#
#******************************************************************************
#


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
class test_cp_als(TestCase):
    def setUp(self):
        np.random.seed(123)        
        
    def test_cp_als_deterministic(self):
        I,J,K,R = 15,15,15,3
        X = cp_tensor((I,J,K), R)         
        X = kruskal_to_tensor(X)	
        P = cp_als(X, r=R, c=False, tol=atol_float64, maxiter=150)  
                
        percent_error = sci.linalg.norm(kruskal_to_tensor(P.factors, lmbda=P.lmbda) - X) / sci.linalg.norm(X)
        assert percent_error < atol_float32   

    def test_cp_als_randomized(self):
        I,J,K,R = 15,15,15,3
        X = cp_tensor((I,J,K), R)         
        X = kruskal_to_tensor(X)	
        P = cp_als(X, r=R, c=True, tol=atol_float64, maxiter=150)
                
        percent_error = sci.linalg.norm(kruskal_to_tensor(P.factors, lmbda=P.lmbda) - X) / sci.linalg.norm(X)
        assert percent_error < atol_float32   

#
#******************************************************************************
#
class test_cp_bcd(TestCase):
    def setUp(self):
        np.random.seed(123)        
        
    def test_cp_bcd_deterministic(self):
        I,J,K,R = 15,15,15,3
        X = cp_tensor((I,J,K), R)         
        X = kruskal_to_tensor(X)	
        P = cp_bcd(X, r=R, c=False, tol=atol_float64, maxiter=150)  
                
        percent_error = sci.linalg.norm(kruskal_to_tensor(P.factors, lmbda=P.lmbda) - X) / sci.linalg.norm(X)
        assert percent_error <  0.01  

    def test_cp_bcd_randomized(self):
        I,J,K,R = 15,15,15,3
        X = cp_tensor((I,J,K), R)         
        X = kruskal_to_tensor(X)	
        P = cp_bcd(X, r=R, c=True, tol=atol_float64, maxiter=150)  
                
        percent_error = sci.linalg.norm(kruskal_to_tensor(P.factors, lmbda=P.lmbda) - X) / sci.linalg.norm(X)
        assert percent_error < 0.01


#
#******************************************************************************
#
class test_cp_opt(TestCase):
    def setUp(self):
        np.random.seed(123)        
        
    def test_cp_opt_deterministic(self):
        I,J,K,R = 15,15,15,3
        X = cp_tensor((I,J,K), R)         
        X = kruskal_to_tensor(X)	
        P = cp_opt(X, r=R, c=False, tol=atol_float64, maxiter=150)  
                
        percent_error = sci.linalg.norm(kruskal_to_tensor(P.factors, lmbda=P.lmbda) - X) / sci.linalg.norm(X)
        assert percent_error <  0.01  

    def test_cp_opt_randomized(self):
        I,J,K,R = 15,15,15,3
        X = cp_tensor((I,J,K), R)         
        X = kruskal_to_tensor(X)	
        P = cp_opt(X, r=R, c=True, tol=atol_float64, maxiter=150)  
                
        percent_error = sci.linalg.norm(kruskal_to_tensor(P.factors, lmbda=P.lmbda) - X) / sci.linalg.norm(X)
        assert percent_error < 0.01


#
#******************************************************************************
#
        
def suite():
    s = TestSuite()
    s.addTest(test_base('test_khatrirao'))

    s.addTest(test_cp_als('test_cp_als_deterministic'))
    s.addTest(test_cp_als('test_cp_als_randomized'))
    s.addTest(test_cp_bcd('test_cp_bcd_deterministic'))
    s.addTest(test_cp_bcd('test_cp_bcd_randomized'))
    
    return s

if __name__ == '__main__':
    main(defaultTest = 'suite')
