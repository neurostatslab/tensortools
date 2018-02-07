import tensortools as tt
import numpy as np
import itertools
import ipdb
np.random.seed(111)

dims = (100, 101, 102)
patience = 100

for rank in range(1, 8):
    
    A = [np.random.randn(d, rank) for d in dims]

    count = 0

    for perm in itertools.permutations(range(rank)):
        

        B, C, score = tt.align_factors([a[:,perm] for a in A], A)

        _, ndim_C, rank_C = tt.kruskal._validate_factors(C)
        _, ndim_B, rank_B = tt.kruskal._validate_factors(B)
        
        assert ndim_C == ndim_B == len(dims)
        assert rank_C == rank_B == rank
        assert [np.linalg.norm(a-b) < np.finfo(float).eps for a, b in zip(A, B)]
        assert [np.linalg.norm(a-c) < np.finfo(float).eps for a, c in zip(A, C)]
        assert score > 0.9999

        for r in range(1, rank-1):

            # truncated alignment
            D, E, score = tt.align_factors([a[:,perm[:r]] for a in A], A)

            _, ndim_D, rank_D = tt.kruskal._validate_factors(D)
            _, ndim_E, rank_E = tt.kruskal._validate_factors(E)
            
            assert ndim_D == ndim_E == len(dims)
            assert rank_E == rank
            assert rank_D == r
            assert score > 0.9999

        if count > patience:
            break
        else:
            count += 1