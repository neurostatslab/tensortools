"""Test core tensor operations and functionality."""
import pytest
import numpy as np

from tensortools.operations import khatri_rao

atol_float32 = 1e-4
atol_float64 = 1e-8


def test_khatrirao():
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
