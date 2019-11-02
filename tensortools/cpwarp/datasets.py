import numpy as np
from scipy.ndimage import gaussian_filter1d

from tensortools.cpwarp.shifted_cp import ShiftedCP
from tensortools.cpwarp.multishift import MultiShiftModel


def simulate_shifted_cp(
        shape, rank, max_shift=.5, smoothness=2.0, noise_scale=.1, seed=None):
    """
    Generates a synthetic dataset from a shifted decomposition.

    Parameters
    ----------
    shape : tuple
        Tuple of three integers specifying num_trials, num_timepoints,
        num_units.
    max_shift : float
        Largest allowable shift expressed as a fraction of trial length.
    smoothness : float
        Specifies width of gaussian smoothing kernel applied to ground
        truth model along the temporal dimension.
    noise_scale : float
        Standard deviation of truncated Gaussian noise.
    seed : RandomState, int, or None
        Seeds random number generator.

    Returns
    -------
    X : ndarray
        Tensor of simulated date (num_trials x num_timepoints x num_units).
    true_model : ShiftedDecomposition
        Object holding the true factors.
    """

    rs = np.random.RandomState(seed)
    factors = [
        rs.rand(rank, shape[0]),
        rs.exponential(1.0, size=(rank, shape[1])),
        rs.rand(rank, shape[2]),
    ]

    # factors[0] *= (factors[0] > np.percentile(factors[1], 50))
    # factors[2] *= (factors[2] > np.percentile(factors[1], 50))

    factors[1] *= (factors[1] > np.percentile(factors[1], 90))
    factors[1] = gaussian_filter1d(factors[1], smoothness, axis=-1)

    b = max_shift * shape[1]
    shifts = rs.uniform(-b, b, size=(rank, shape[0]))

    true_model = ShiftedCP(factors, shifts)
    true_model.rebalance()

    X = true_model.predict()
    X += rs.randn(*shape) * noise_scale
    # X = np.maximum(0.0, X)

    return X, true_model


def simulate_multishift(
        shape, rank, max_shift=.5, trial_factor_sparsity=.5,
        smoothness=2.0, noise_scale=.1, seed=None):
    """
    Generates a synthetic dataset from a multi-warp model.

    Parameters
    ----------
    shape : tuple
        Tuple of three integers specifying num_trials, num_timepoints,
        num_units.
    max_shift : float
        Largest allowable shift expressed as a fraction of trial length.
    trial_factor_sparsity : float
        Dirichlet distribution parameter, smaller values correspond to
        more sparse (one-hot) loadings on the trial factors.
    smoothness : float
        Specifies width of gaussian smoothing kernel applied to ground
        truth model along the temporal dimension.
    noise_scale : float
        Standard deviation of truncated Gaussian noise.
    seed : RandomState, int, or None
        Seeds random number generator.

    Returns
    -------
    X : ndarray
        Tensor of simulated date (num_trials x num_timepoints x num_units).
    true_model : MultiShiftModel
        Object holding the true model.
    """

    K, T, N = shape
    rs = np.random.RandomState(seed)

    _tmp = rs.exponential(1.0, size=(rank, T, N))
    _tmp *= (_tmp > np.percentile(_tmp, 95))
    templates = gaussian_filter1d(_tmp, smoothness, axis=1)

    trial_factors = np.random.dirichlet(
        [trial_factor_sparsity for _ in range(rank)], size=K).T

    shifts = rs.uniform(
        -max_shift * T, max_shift * T, size=(rank, K))

    true_model = MultiShiftModel(
        templates, trial_factors, shifts=shifts, periodic=True)

    X = true_model.predict()
    X += rs.randn(*shape) * noise_scale

    return X, true_model
