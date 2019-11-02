"""
Test shifting utility functions.
"""

import pytest
import numpy as np
from tensortools.cpwarp import padded_shifts


@pytest.mark.parametrize(
    "shift", np.linspace(3, 3, 10)
)
def test_shifts_1d(shift):
    x = np.array([1, 2, 3, 4, 5], dtype="float")
    wx = [i - shift for i in range(5)]
    xs = np.empty_like(x)

    padded_shifts.apply_shift(x, shift, xs)
    np.testing.assert_allclose(
        xs, np.interp(wx, np.arange(5), x))


@pytest.mark.parametrize(
    "shift", np.linspace(3, 3, 10)
)
def test_shifts_2d(shift):
    """
    Tests shift operation on a matrix, with shifting
    applied to the first dimension.
    """
    x = np.array([1, 2, 3, 4, 5], dtype="float")
    x = np.tile(x[None, :], (2, 1)).T

    wx = [i - shift for i in range(5)]
    xs = np.empty_like(x)

    padded_shifts.apply_shift(x, shift, xs)

    y = np.interp(wx, np.arange(5), x[:, 0])
    np.testing.assert_allclose(xs, np.column_stack((y, y)))


def test_transpose_shifts():
    x = np.array([1, 2, 3, 4, 5], dtype="float")
    xs = np.empty_like(x)

    # test shift right by 1.0
    W = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
    ])
    padded_shifts.trans_shift(x, 1.0, xs)
    np.testing.assert_allclose(xs, np.dot(W.T, x))

    padded_shifts.apply_shift(x, 1.0, xs)
    np.testing.assert_allclose(xs, np.dot(W, x))

    # test shift right by 1.1
    W = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.9, 0.0, 0.0, 0.0],
        [0.0, 0.1, 0.9, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.9, 0.0],
    ])
    padded_shifts.trans_shift(x, 1.1, xs)
    np.testing.assert_allclose(xs, np.dot(W.T, x))

    padded_shifts.apply_shift(x, 1.1, xs)
    np.testing.assert_allclose(xs, np.dot(W, x))

    # test shift right by 0.7
    W = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.7, 0.3, 0.0, 0.0, 0.0],
        [0.0, 0.7, 0.3, 0.0, 0.0],
        [0.0, 0.0, 0.7, 0.3, 0.0],
        [0.0, 0.0, 0.0, 0.7, 0.3],
    ])
    padded_shifts.trans_shift(x, 0.7, xs)
    np.testing.assert_allclose(xs, np.dot(W.T, x))

    padded_shifts.apply_shift(x, 0.7, xs)
    np.testing.assert_allclose(xs, np.dot(W, x))

    # test shift left by 1.0
    W = np.array([
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ])
    padded_shifts.trans_shift(x, -1.0, xs)
    np.testing.assert_allclose(xs, np.dot(W.T, x))

    padded_shifts.apply_shift(x, -1.0, xs)
    np.testing.assert_allclose(xs, np.dot(W, x))

    # test shift left by 1.3
    W = np.array([
        [0.0, 0.7, 0.3, 0.0, 0.0],
        [0.0, 0.0, 0.7, 0.3, 0.0],
        [0.0, 0.0, 0.0, 0.7, 0.3],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ])
    padded_shifts.trans_shift(x, -1.3, xs)
    np.testing.assert_allclose(xs, np.dot(W.T, x))

    padded_shifts.apply_shift(x, -1.3, xs)
    np.testing.assert_allclose(xs, np.dot(W, x))

    # test shift left by 0.4
    W = np.array([
        [0.6, 0.4, 0.0, 0.0, 0.0],
        [0.0, 0.6, 0.4, 0.0, 0.0],
        [0.0, 0.0, 0.6, 0.4, 0.0],
        [0.0, 0.0, 0.0, 0.6, 0.4],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ])
    padded_shifts.trans_shift(x, -0.4, xs)
    np.testing.assert_allclose(xs, np.dot(W.T, x))

    padded_shifts.apply_shift(x, -0.4, xs)
    np.testing.assert_allclose(xs, np.dot(W, x))


def test_grams():
    probe_data = [
        # No shift.
        (0, np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ])),
        # Shift right by 1.0
        (1, np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ])),
        # Shift right by 2.0
        (2, np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ])),
        # Shift left by 1.0
        (-1, np.array([
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ])),
        # Shift left by 2.0
        (-2, np.array([
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ])),
        # Shift right by 0.3
        (0.3, np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.3, 0.7, 0.0, 0.0, 0.0],
            [0.0, 0.3, 0.7, 0.0, 0.0],
            [0.0, 0.0, 0.3, 0.7, 0.0],
            [0.0, 0.0, 0.0, 0.3, 0.7],
        ])),
        # Shift right by 1.3
        (1.3, np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.3, 0.7, 0.0, 0.0, 0.0],
            [0.0, 0.3, 0.7, 0.0, 0.0],
            [0.0, 0.0, 0.3, 0.7, 0.0],
        ])),
        # Shift left by 0.3
        (-0.3, np.array([
            [0.7, 0.3, 0.0, 0.0, 0.0],
            [0.0, 0.7, 0.3, 0.0, 0.0],
            [0.0, 0.0, 0.7, 0.3, 0.0],
            [0.0, 0.0, 0.0, 0.7, 0.3],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ])),
        # Shift left by 1.3
        (-1.3, np.array([
            [0.0, 0.7, 0.3, 0.0, 0.0],
            [0.0, 0.0, 0.7, 0.3, 0.0],
            [0.0, 0.0, 0.0, 0.7, 0.3],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ])),
    ]

    # Test each example.
    for shift, W in probe_data:
        WtW = np.dot(W.T, W)
        expected = np.row_stack(
            [np.concatenate(([0.0], np.diag(WtW, 1))), np.diag(WtW, 0)])

        actual = padded_shifts.shift_gram(shift, 5, np.empty((2, 5)))
        np.testing.assert_allclose(actual, expected)


def test_sym_bmat_mul():
    S = np.array([
        [1, 2, 1, 0, 0, 0, 0, 0],
        [2, 1, 1, 4, 0, 0, 0, 0],
        [1, 1, 5, 1, 1, 0, 0, 0],
        [0, 4, 1, 4, 1, 8, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 8, 1, 7, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 3],
        [0, 0, 0, 0, 0, 1, 3, 9],
    ]).astype('float')

    x = np.random.randn(8)

    Sb = np.full((3, 8), np.nan)
    Sb[-1] = np.diag(S)
    Sb[-2, 1:] = np.diag(S, 1)
    Sb[-3, 2:] = np.diag(S, 2)

    y = np.dot(S, x)

    z = np.empty_like(x)
    padded_shifts.sym_bmat_mul(Sb, x, z)

    np.testing.assert_allclose(z, y)
