"""
Test model class.
"""
import pytest
import numpy as np
from tensortools.cpwarp import ShiftedCP


def test_predict_no_shift():
    I, J, K, rank = 9, 10, 11, 3
    rs = np.random.RandomState(123)

    u = rs.rand(rank, I)
    v = rs.rand(rank, J)
    w = rs.rand(rank, K)
    u_s = np.zeros((rank, I))
    v_s = np.zeros((rank, J))

    model = ShiftedCP(u, v, w, u_s, v_s)

    actual = model.predict()
    expected = np.einsum("ir,jr,kr->ijk", u.T, v.T, w.T)

    np.testing.assert_allclose(actual, expected)
