import tensortools as tt
import numpy as np
import itertools
import ipdb
np.random.seed(111)

dims = (100, 101, 102)

for rank in range(1, 8):
    
    A = [np.random.randn(d, rank) for d in dims]

    for perm in itertools.permutations(range(rank)):
        print(perm)
        # ipdb.set_trace()
        FUCK = [a[:,perm] for a in A]
        B, C, score = tt.align_factors(FUCK, A)

        _, ndim_C, rank_C = tt.kruskal._validate_factors(C)
        _, ndim_B, rank_B = tt.kruskal._validate_factors(B)
        
        assert ndim_C == ndim_B == len(dims)
        assert rank_C == rank_B == rank
        assert [np.linalg.norm(a-b) < np.finfo(float).eps for a, b in zip(A, B)]
        assert [np.linalg.norm(a-c) < np.finfo(float).eps for a, c in zip(A, C)]
        assert score > 0.9999
