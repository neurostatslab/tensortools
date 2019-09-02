"""Test KTensor functions."""
import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from tensortools import KTensor


def test_norm():

    rs = np.random.RandomState(123)
    U = KTensor([rs.randn(55, 3) for _ in range(3)])

    assert_almost_equal(U.norm(), np.linalg.norm(U.full()))
