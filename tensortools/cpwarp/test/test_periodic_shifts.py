"""
Test shifting utility functions.
"""

import pytest
import numpy as np
from tensortools.cpwarp import periodic_shifts
from scipy.linalg import circulant


@pytest.mark.parametrize(
    "shift", np.linspace(3, 3, 10)
)
def test_shifts_1d(shift):
    x = np.array([1, 2, 3, 4, 5], dtype="float")
    wx = [i - shift for i in range(5)]
    xs = np.empty_like(x)

    periodic_shifts.apply_shift(x, shift, xs)
    np.testing.assert_allclose(
        xs, np.interp(wx, np.arange(5), x, period=5))


@pytest.mark.parametrize(
    "shift", np.linspace(3, 3, 10)
)
def test_shifts_2d(shift):
    """
    Tests shift operation on a matrix, with shifting
    applied to the last dimension.
    """
    x = np.array([1, 2, 3, 4, 5], dtype="float")
    x = np.tile(x[None, :], (2, 1)).T

    wx = [i - shift for i in range(5)]
    xs = np.empty_like(x)

    periodic_shifts.apply_shift(x, shift, xs)

    y = np.interp(wx, np.arange(5), x[:, 0], period=5)
    np.testing.assert_allclose(xs, np.column_stack((y, y)))


def test_transpose_shifts():
    x = np.array([1, 2, 3, 4, 5], dtype="float")
    xs = np.empty_like(x)

    # test shift right by 1.0
    W = np.array([
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
    ])
    periodic_shifts.trans_shift(x, 1.0, xs)
    np.testing.assert_allclose(xs, np.dot(W.T, x))

    periodic_shifts.apply_shift(x, 1.0, xs)
    np.testing.assert_allclose(xs, np.dot(W, x))

    # test shift right by 1.1
    W = np.array([
        [0.0, 0.0, 0.0, 0.1, 0.9],
        [0.9, 0.0, 0.0, 0.0, 0.1],
        [0.1, 0.9, 0.0, 0.0, 0.0],
        [0.0, 0.1, 0.9, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.9, 0.0],
    ])
    periodic_shifts.trans_shift(x, 1.1, xs)
    np.testing.assert_allclose(xs, np.dot(W.T, x))

    periodic_shifts.apply_shift(x, 1.1, xs)
    np.testing.assert_allclose(xs, np.dot(W, x))

    # test shift right by 0.7
    W = np.array([
        [0.3, 0.0, 0.0, 0.0, 0.7],
        [0.7, 0.3, 0.0, 0.0, 0.0],
        [0.0, 0.7, 0.3, 0.0, 0.0],
        [0.0, 0.0, 0.7, 0.3, 0.0],
        [0.0, 0.0, 0.0, 0.7, 0.3],
    ])
    periodic_shifts.trans_shift(x, 0.7, xs)
    np.testing.assert_allclose(xs, np.dot(W.T, x))

    periodic_shifts.apply_shift(x, 0.7, xs)
    np.testing.assert_allclose(xs, np.dot(W, x))

    # test shift left by 1.0
    W = np.array([
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
    ])
    periodic_shifts.trans_shift(x, -1.0, xs)
    np.testing.assert_allclose(xs, np.dot(W.T, x))

    periodic_shifts.apply_shift(x, -1.0, xs)
    np.testing.assert_allclose(xs, np.dot(W, x))

    # test shift left by 1.3
    W = np.array([
        [0.0, 0.7, 0.3, 0.0, 0.0],
        [0.0, 0.0, 0.7, 0.3, 0.0],
        [0.0, 0.0, 0.0, 0.7, 0.3],
        [0.3, 0.0, 0.0, 0.0, 0.7],
        [0.7, 0.3, 0.0, 0.0, 0.0],
    ])
    periodic_shifts.trans_shift(x, -1.3, xs)
    np.testing.assert_allclose(xs, np.dot(W.T, x))

    periodic_shifts.apply_shift(x, -1.3, xs)
    np.testing.assert_allclose(xs, np.dot(W, x))

    # test shift left by 0.4
    W = np.array([
        [0.6, 0.4, 0.0, 0.0, 0.0],
        [0.0, 0.6, 0.4, 0.0, 0.0],
        [0.0, 0.0, 0.6, 0.4, 0.0],
        [0.0, 0.0, 0.0, 0.6, 0.4],
        [0.4, 0.0, 0.0, 0.0, 0.6],
    ])
    periodic_shifts.trans_shift(x, -0.4, xs)
    np.testing.assert_allclose(xs, np.dot(W.T, x))

    periodic_shifts.apply_shift(x, -0.4, xs)
    np.testing.assert_allclose(xs, np.dot(W, x))


@pytest.mark.parametrize(
    "c", np.linspace(3, 3, 3)
)
@pytest.mark.parametrize(
    "a", np.linspace(3, 3, 3)
)
def test_tri_sym_circ_matvec(c, a):
    x = np.random.randn(10, 5)
    M = (
        np.diag(np.full(10, c), 0) +
        np.diag(np.full(9, a), -1) +
        np.diag(np.full(9, a), 1)
    )
    M[0, -1] = a
    M[-1, 0] = a

    result = np.full_like(x, np.nan)
    periodic_shifts.tri_sym_circ_matvec(c, a, x, result)

    np.testing.assert_allclose(result, M @ x)
