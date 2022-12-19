import numpy as np

from tensortools.tensors import KTensor

# TODO - refactor this code to take an arbitrary random generator.


def _check_random_state(random_state):
    """Checks and processes user input for seeding random numbers.

    Parameters
    ----------
    random_state : int, RandomState instance or None
        If int, a RandomState instance is created with this integer seed.
        If RandomState instance, random_state is returned;
        If None, a RandomState instance is created with arbitrary seed.

    Returns
    -------
    numpy.random.RandomState instance

    Raises
    ------
    TypeError
        If ``random_state`` is not appropriately set.
    """
    if random_state is None or isinstance(random_state, int):
        return np.random.RandomState(random_state)
    elif isinstance(random_state, np.random.RandomState):
        return random_state
    else:
        raise TypeError('Seed should be None, int or np.random.RandomState')


def _rescale_tensor(factors, norm):
    # Rescale the tensor to match the specified norm.
    if norm is None:
        return factors.rebalance()
    else:
        # Compute rescaling factor for tensor
        factors[0] *= norm / factors.norm()
        return factors.rebalance()


def randn_ktensor(shape, rank, norm=None, random_state=None):
    """
    Generates a random N-way tensor with rank R, where the factors are
    generated from the standard normal distribution.

    Parameters
    ----------
    shape : tuple
        shape of the tensor

    rank : integer
        rank of the tensor

    norm : float or None, optional (defaults: None)
        If not None, the factor matrices are rescaled so that the Frobenius
        norm of the returned tensor is equal to ``norm``.

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.


    Returns
    -------
    X : (I_1, ..., I_N) array_like
        N-way tensor with rank R.

    Example
    -------
    >>> # Create a rank-2 tensor of dimension 5x5x5:
    >>> import tensortools as tt
    >>> X = tt.randn_tensor((5,5,5), rank=2)

    """

    # Check input.
    rns = _check_random_state(random_state)

    # Draw low-rank factor matrices with i.i.d. Gaussian elements.
    factors = KTensor([rns.standard_normal((i, rank)) for i in shape])
    return _rescale_tensor(factors, norm)


def rand_ktensor(shape, rank, norm=None, random_state=None):
    """
    Generates a random N-way tensor with rank R, where the factors are
    generated from the standard uniform distribution in the interval [0.0,1].

    Parameters
    ----------
    shape : tuple
        shape of the tensor

    rank : integer
        rank of the tensor

    norm : float or None, optional (defaults: None)
        If not None, the factor matrices are rescaled so that the Frobenius
        norm of the returned tensor is equal to ``norm``.

    ktensor : bool
        If true, a KTensor object is returned, i.e., the components are in factored
        form ``[U_1, U_2, ... U_N]``; Otherwise an N-way array is returned.

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.


    Returns
    -------
    X : (I_1, ..., I_N) array_like
        N-way tensor with rank R.

    Example
    -------
    >>> # Create a rank-2 tensor of dimension 5x5x5:
    >>> import tensortools as tt
    >>> X = tt.rand_tensor((5,5,5), rank=2)

    """

    # Check input.
    rns = _check_random_state(random_state)

    # Randomize low-rank factor matrices i.i.d. uniform random elements.
    factors = KTensor([rns.uniform(0.0, 1.0, size=(i, rank)) for i in shape])
    return _rescale_tensor(factors, norm)


def randexp_ktensor(shape, rank, scale=1.0, norm=None, random_state=None):
    """
    Generates a random N-way tensor with rank R, where the entries are
    drawn from an exponential distribution

    Parameters
    ----------
    shape : tuple
        shape of the tensor

    rank : integer
        rank of the tensor

    scale : float
        Scale parameter for the exponential distribution.

    norm : float or None, optional (defaults: None)
        If not None, the factor matrices are rescaled so that the Frobenius
        norm of the returned tensor is equal to ``norm``.

    ktensor : bool
        If true, a KTensor object is returned, i.e., the components are in factored
        form ``[U_1, U_2, ... U_N]``; Otherwise an N-way array is returned.

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.


    Returns
    -------
    X : (I_1, ..., I_N) array_like
        N-way tensor with rank R.

    Example
    -------
    >>> # Create a rank-2 tensor of dimension 5x5x5:
    >>> import tensortools as tt
    >>> X = tt.randexp_tensor((5,5,5), rank=2)

    """

    # Check input.
    rns = _check_random_state(random_state)

    # Randomize low-rank factor matrices i.i.d. uniform random elements.
    factors = KTensor(
        [rns.exponential(scale=scale, size=(i, rank)) for i in shape])
    return _rescale_tensor(factors, norm)
