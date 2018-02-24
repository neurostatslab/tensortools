import tensortools as tt
import numpy as np
np.random.seed(1234)

def test_recovery():
    
    # create ground-truth factors
    shape = (20, 21, 22)
    R = 3
    factors = tt.Ktensor([np.random.randn(s, R) for s in shape])

    # create low-rank dense tensor
    X = factors.full()

    # fit cp-als to recover ground truth
    result = tt.optimize.cp_als(X, R)

    return tt.kruskal_similarity(factors, result.U)