"""Test metrics and diagnostic tools."""
import pytest
import numpy as np
from scipy import linalg
import itertools

import tensortools as tt

atol_float32 = 1e-4
atol_float64 = 1e-8


def test_align():
    # Generate random KTensor.
    I, J, K, R = 15, 16, 17, 4
    U = tt.randn_ktensor((I, J, K), rank=R)
    X = U.full()   # Dense representation of U.

    # Enumerate all permutations of factors and test that
    # kruskal_align appropriately inverts this permutation.
    for prm in itertools.permutations(range(R)):
        V = U.copy()
        V.permute(prm)
        assert (tt.kruskal_align(U, V) - 1) < atol_float64
        assert linalg.norm(X - U.full()) < atol_float64
        assert linalg.norm(X - V.full()) < atol_float64

    # Test that second input to kruskal_align is correctly permuted.
    for prm in itertools.permutations(range(R)):
        V = U.copy()
        V.permute(prm)
        tt.kruskal_align(U, V, permute_V=True)
        for fU, fV in zip(U, V):
            assert linalg.norm(fU - fV) < atol_float64
            assert linalg.norm(X - U.full()) < atol_float64
            assert linalg.norm(X - V.full()) < atol_float64

    # Test that first input to kruskal_align is correctly permuted.
    for prm in itertools.permutations(range(R)):
        V = U.copy()
        V.permute(prm)
        tt.kruskal_align(V, U, permute_U=True)
        for fU, fV in zip(U, V):
            assert linalg.norm(fU - fV) < atol_float64
            assert linalg.norm(X - U.full()) < atol_float64
            assert linalg.norm(X - V.full()) < atol_float64
