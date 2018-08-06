"""Test optimization routines."""
import pytest
import numpy as np
from scipy import linalg
import itertools
import tensortools as tt

obj_decreased_tol = 1e-3
data_seed = 0
alg_seed = 100

algnames = ['cp_als', 'ncp_hals', 'ncp_bcd']
shapes = [(10, 11, 12), (10, 11, 12, 13), (100, 101, 102)]
ranks = [1, 2, 5]


@pytest.mark.parametrize(
    "algname,shape,rank",
    itertools.product(algnames, shapes, ranks)
)
def test_objective_decreases(algname, shape, rank):

    # Generate data. If algorithm is made for nonnegative tensor decomposition
    # then generate nonnegative data.
    if algname in ['ncp_hals, ncp_bcd']:
        X = tt.rand_ktensor(shape, rank=rank, random_state=data_seed).full()
    else:
        X = tt.randn_ktensor(shape, rank=rank, random_state=data_seed).full()

    # Fit model.
    f = getattr(tt, algname)
    result = f(X, rank=rank, verbose=False, tol=1e-6, random_state=alg_seed)

    # Test that objective function monotonically decreases.
    assert np.all(np.diff(result.obj_hist) < obj_decreased_tol)
