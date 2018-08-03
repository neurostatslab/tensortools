"""
Simple tensor operations and utility functions.

Some of these functions were ported with minor modifications
from the tensorly package, https://tensorly.github.io/, distributed
under a BSD clause 3 license.
"""
import numpy as np


def unfold(tensor, mode=0):
    """Returns the mode-`mode` unfolding of `tensor`.

    Parameters
    ----------
    tensor : ndarray
    mode : int, default is 0
        Indexing starts at 0, therefore mode is in ``range(0, tensor.ndim)``

    Returns
    -------
    ndarray
        unfolded_tensor of shape ``(tensor.shape[mode], -1)``
    """
    return np.moveaxis(tensor, mode, 0).reshape((tensor.shape[mode], -1))


def fold(unfolded_tensor, mode, shape):
    """Refolds the mode-`mode` unfolding of a tensor back to `shape`.

    Parameters
    ----------
    unfolded_tensor : ndarray
        unfolded tensor of shape ``(shape[mode], -1)``
    mode : int
        the mode of the unfolding
    shape : tuple
        shape of the original tensor before unfolding

    Returns
    -------
    ndarray
        folded_tensor of shape `shape`
    """
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return np.moveaxis(unfolded_tensor.reshape(full_shape), 0, mode)


def kruskal_to_tensor(factors):
    """Converts canonical polyadic decomposition factor matrices to a full tensor.

    For a third order tensor this is equivalent to
    ``np.einsum('ir,jr,kr->ijk', *factors)``.

    Parameters
    ----------
    factors : list of ndarray
        list of factor matrices, all with the same number of columns.

    Returns
    -------
    ndarray
        tensor with shape ``(U[1].shape[0], ... U[-1].shape[0])``
    """
    shape = [factor.shape[0] for factor in factors]
    full_tensor = np.dot(factors[0], khatri_rao(factors[1:]).T)
    return fold(full_tensor, 0, shape)


def khatri_rao(matrices, skip_mode=None):
    """
    Khatri-Rao product of a list of matrices. This can be seen as a column-wise
    kronecker product.

    Parameters
    ----------
    matrices : ndarray list
        list of matrices with the same number of columns.
    skip_mode : None or int, optional, default is None
        if not None, index of a matrix to skip
    reverse : bool, optional
        if True, the order of the matrices is reversed

    Returns
    -------
    ndarray
        Khatri-Rao product of matrices.
    """
    if skip_mode is not None:
        matrices = [matrices[i] for i in range(len(matrices)) if i != skip_mode]

    n_columns = matrices[0].shape[1]
    n_factors = len(matrices)

    start = ord('a')
    common_dim = 'z'
    target = ''.join(chr(start + i) for i in range(n_factors))
    source = ','.join(i+common_dim for i in target)
    operation = source+'->'+target+common_dim
    return np.einsum(operation, *matrices).reshape((-1, n_columns))


def factor_weights(X):
    """
    Normalizes all factors to unit length, and returns factor weights\
    (higher-order generalization of singular values.)

    Example:
        >>> Y, lam = factor_weights(X)
        >>> [y * np.power(lam[None,:], 1/len(X)) for y in Y] # equal to X

    Parameters
    -----------
    X : list of ndarray
        list of factor matrices.

    Returns
    -------
    Y : list of ndarray
        same as factors, but with normalized columns (unit length)
    lam : ndarray
        vector of length R holding the weight for each factor
    """
    factors, ndim, rank = _validate_factors(X)

    # factor norms
    lam = np.ones(rank)

    # destination for new ktensor
    newfactors = []

    # normalize columns of factor matrices
    lam = np.ones(rank)
    for fact in factors:
        s = np.linalg.norm(fact, axis=0)
        lam *= s
        newfactors.append(fact/(s + 1e-20))

    return newfactors, lam


def reweight_factors(X, mode_weights=None, sort_factors=True):
    """Sorts factors by norm and distributes factor weights all modes

    Parameters
    ----------
    X : list of ndarray
        List of factor matrices (each matrix has R
        columns, corresponding to the rank of the model)
    mode_weights (optional) : ndarray
        If specified, determines how to distribute factors weights. For
        example, if lam_ratios = [1, 0, 0, ...] then all factors are unit
        length except the factors along the first mode which are multiplied
        by the weight of that component. Defaults to [1/M, 1/M, ... , 1/M]
        which evenly distributes the weights among the factors.
    sort_factors (optional) : bool
        If True, sort the factors by their weights. Defaults to True.

    Returns
    -------
    Y : ndarray list
        list of factor matrices after standardization
    """

    # Compute normalized factors and factor weights/singfular values.
    nrmfactors, lam = factor_weights(X)

    # Default to equally sized factors.
    if mode_weights is None:
        mode_weights = np.ones(X.ndim)  # later normalized to sum to one.

    # Check input is valid
    if len(lam_weights) != len(X):
        raise ValueError("Parameter 'lam_weights' must be a list equal to the "
                         "number of tensor modes/dimensions.")

    elif np.min(mode_weights) < 0:
        raise ValueError("Parameter 'mode_weights' must all be nonnegative.")

    # Normalize mode_weights to sum to one.
    mode_weights = np.asarray(mode_weights) / np.sum(mode_weights)

    # Sort columns of factor matrices by norm if desired.
    prm = np.argsort(lam)[::-1] if sort_factors else slice(None)

    # Distributed factor weights across modes. For example, when distributing
    # weight lam equally across M modes we multiply each factor by lam ** (1/M)
    # so that the M-way outer product of the factors is lam.
    new_factors = []
    for f, w in zip(nrmfactors, mode_weights):
        new_factors.append(f[:, prm] * np.power(lam[prm], w))


def _validate_factors(factors):
    """Checks that input is a valid kruskal tensor

    Returns
    -------
    ndim : int
        number of dimensions in tensor
    rank : int
        number of factors
    """
    ndim = len(factors)

    # if necessary, add an axis to factor matrices
    for i, f in enumerate(factors):
        if f.ndim == 1:
            factors[i] = f[:, np.newaxis]

    # check rank consistency
    rank = factors[0].shape[1]
    for f in factors:
        if f.shape[1] != rank:
            raise ValueError('KTensor has inconsistent rank along modes.')

    # return factors and info
    return factors, ndim, rank
