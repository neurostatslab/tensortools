"""Test data generation routines."""
import pytest
import numpy as np
from scipy import linalg
import itertools
import tensortools
from numpy.testing import assert_almost_equal

funcnames = ["randn_ktensor", "rand_ktensor"]
shapes = [(10, 11, 12), (10, 11, 12, 13), (100, 101, 102)]
ranks = [1, 2, 5]
norms = [1.0, 10.0, 100.0]


@pytest.mark.parametrize(
    "funcname,shape,rank,norm",
    itertools.product(funcnames, shapes, ranks, norms)
)
def test_norm(funcname, shape, rank, norm):
    f = getattr(tensortools, funcname)
    kten = f(shape, rank, norm=norm)
    assert_almost_equal(linalg.norm(kten.full()), norm)
