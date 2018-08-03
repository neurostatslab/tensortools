"""
Simple tensor operations and utility functions.

Many of these functions were ported with minor modifications
from the tensorly package, https://tensorly.github.io/, distributed
under a BSD clause 3 license.
"""
import numpy as np


def norm(tensor, order=2):
    """Computes vectorized tensor norms.

    Parameters
    ----------
    tensor : ndarray
    order : int, default is 2
        Order of the norm. Defaults to order=2, which is the Euclidean norm.

    Returns
    -------
    float
        l-`order` norm of tensor
    """
    if order == 1:
        return np.sum(np.abs(tensor))
    elif order == 2:
        return np.sqrt(np.dot(tensor.ravel(), tensor.ravel()))
    else:
        return np.sum(np.abs(tensor)**order)**(1/order)


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
