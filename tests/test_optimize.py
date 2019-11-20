"""Test optimization routines."""
import pytest
import numpy as np
from scipy import linalg
import itertools
import tensortools as tt

obj_decreased_tol = 1e-3
data_seed = 0
alg_seed = 100

algnames = ['cp_als', 'ncp_hals', 'ncp_bcd', 'mcp_als']
shapes = [(10, 11, 12), (10, 11, 12, 13)]
ranks = [1, 2, 5]


@pytest.mark.parametrize(
    "algname,shape",
    itertools.product(algnames, shapes)
)
def test_deterministic(algname, shape):
    """Tests that random seed fully specifies initialization."""

    # Random tensor.
    X = np.random.rand(*shape)

    # Model options. Set to zero iterations.
    rank = 3
    options = dict(
        rank=rank, verbose=False, max_iter=-1, random_state=alg_seed)

    # Add special options for particular algorithms.
    if algname in ("mcp_als",):
        options["mask"] = np.ones_like(X).astype(bool)

    # Fit decomposition twice on the same data
    result_1 = getattr(tt, algname)(X, **options)
    result_2 = getattr(tt, algname)(X, **options)

    for u1, u2 in zip(result_1.factors, result_2.factors):
        assert np.allclose(u1, u2)


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

    # Algorithm fitting options.
    options = dict(rank=rank, verbose=False, tol=1e-6, random_state=alg_seed)

    # Add special options for particular algorithms.
    if algname == 'mcp_als':
        options['mask'] = np.ones_like(X).astype(bool)

    # Fit model.
    result = getattr(tt, algname)(X, **options)

    # Test that objective function monotonically decreases.
    assert np.all(np.diff(result.obj_hist) < obj_decreased_tol)


@pytest.mark.parametrize(
    "algname", ["mcp_als", "ncp_hals"]
)
def test_missingness(algname):

    # Random data tensor.
    shape = (15, 16, 17)
    rank = 3

    if algname == "mcp_als":
        X = tt.randn_ktensor(shape, rank=rank, random_state=data_seed).full()
    elif algname == "ncp_hals":
        X = tt.rand_ktensor(shape, rank=rank, random_state=data_seed).full()

    # Random missingness mask.
    mask = np.random.binomial(1, .5, size=X.shape).astype(bool)

    # Create second tensor with corrupted entries.
    Y = X.copy()
    Y[~mask] = 999.

    # Algorithm fitting options.
    options = dict(
        rank=rank,
        mask=mask,
        verbose=False,
        tol=1e-6,
        random_state=alg_seed
    )

    # Fit decompositions for both X and Y.
    resultX = getattr(tt, algname)(X, **options)
    resultY = getattr(tt, algname)(Y, **options)

    # Test that learning curves are identical.
    assert np.allclose(resultX.obj_hist, resultY.obj_hist)

    # Test that final factors are identical.
    for uX, uY in zip(resultX.factors, resultY.factors):
        assert np.allclose(uX, uY)


@pytest.mark.parametrize(
    "algname", ["ncp_hals", "ncp_bcd"]
)
@pytest.mark.parametrize(
    "neg_modes", [[], [0], [1], [2], [1, 2]]
)
def test_nonneg(algname, neg_modes):

    # Random data tensor.
    shape = (15, 16, 17)
    rank = 3

    X = tt.randn_ktensor(shape, rank=rank, random_state=data_seed).full()

    # Algorithm fitting options.
    options = dict(
        rank=rank,
        negative_modes=neg_modes,
        verbose=False,
        tol=1e-6,
        random_state=alg_seed
    )

    # Fit decomposition.
    result = getattr(tt, algname)(X, **options)

    for mode, factor in enumerate(result.factors):
        if mode in neg_modes:
            assert factor.min() < 0  # this should be true for most datasets
        else:
            assert factor.min() >= 0
