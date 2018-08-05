"""Test optimization routines."""
import pytest
import numpy as np
from scipy import linalg

import tensortools as tt

deterministic_tol = 1e-2


def _get_data(rank, order, nonneg):
    """Sets random seed and creates low rank tensor.

    Parameters
    ----------
    rank : int
        Rank of the synthetic tensor.
    nonneg : bool
        If True, returns nonnegative data. Otherwise data is not constrained.

    Returns
    -------
    X : ndarray
        Low-rank tensor
    """
    np.random.seed(123)
    shape = np.full(order, 15)
    if nonneg:
        X = tt.rand_ktensor(shape, rank=rank).full()
    else:
        X = tt.randn_ktensor(shape, rank=rank).full()
    return X, linalg.norm(X)


def test_cp_als_deterministic():
    rank, order = 4, 3
    # Create dataset.
    X, normX = _get_data(rank, order, nonneg=False)

    # Fit model.
    P = tt.cp_als(X, rank=rank, verbose=False, tol=1e-6)

    # Check that error is low.
    percent_error = linalg.norm(P.factors.full() - X) / normX
    assert percent_error < deterministic_tol


def test_ncp_hals_deterministic():
    rank, order = 4, 3
    # Create dataset.
    X, normX = _get_data(rank, order, nonneg=True)

    # Fit model.
    P = tt.ncp_hals(X, rank=rank, verbose=False, tol=1e-6)

    # Check that result is nonnegative.
    for factor in P.factors:
        assert np.all(factor >= 0)

    # Check that error is low.
    percent_error = linalg.norm(P.factors.full() - X) / normX
    assert percent_error < deterministic_tol


def test_ncp_bcd_deterministic():
    rank, order = 4, 3
    # Create dataset.
    X, normX = _get_data(rank, order, nonneg=True)

    # Fit model.
    P = tt.ncp_bcd(X, rank=rank, verbose=False, tol=1e-6)

    # Check that result is nonnegative.
    for factor in P.factors:
        assert np.all(factor >= 0)

    # Check that error is low.
    percent_error = linalg.norm(P.factors.full() - X) / normX
    assert percent_error < deterministic_tol
