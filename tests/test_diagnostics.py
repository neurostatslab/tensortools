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


def test_degenerate_align():

    # Tests for alignment between factors with zeroed entries.

    # Generate random KTensor.
    I, J, K, R = 15, 16, 17, 4
    U = tt.randn_ktensor((I, J, K), rank=R)
    V = tt.randn_ktensor((I, J, K), rank=R)

    V[0][:, -1] = 0.0
    V[1][:, -1] = 0.0
    V[2][:, -1] = 0.0

    sim_uv = tt.kruskal_align(U, V)
    sim_vu = tt.kruskal_align(V, U)
    assert abs(sim_uv - sim_vu) < atol_float64

    tt.kruskal_align(U.copy(), V.copy(), permute_U=True)
    tt.kruskal_align(V.copy(), U.copy(), permute_U=True)
    tt.kruskal_align(U.copy(), V.copy(), permute_V=True)
    tt.kruskal_align(V.copy(), U.copy(), permute_V=True)
    tt.kruskal_align(U.copy(), V.copy(), permute_U=True, permute_V=True)
    tt.kruskal_align(V.copy(), U.copy(), permute_U=True, permute_V=True)
