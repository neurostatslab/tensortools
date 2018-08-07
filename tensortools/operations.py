"""
Simple tensor operations and utility functions.

Some of these functions were ported with minor modifications
from the tensorly package, https://tensorly.github.io/, distributed
under a BSD clause 3 license.
"""
import numpy as np


def unfold(tensor, mode):
    """Returns the mode-`mode` unfolding of `tensor`.

    Parameters
    ----------
    tensor : ndarray
    mode : int

    Returns
    -------
    ndarray
        unfolded_tensor of shape ``(tensor.shape[mode], -1)``

    Author
    ------
    Jean Kossaifi <https://github.com/tensorly>
    """
    return np.moveaxis(tensor, mode, 0).reshape((tensor.shape[mode], -1))


def khatri_rao(matrices):
    """Khatri-Rao product of a list of matrices.

    Parameters
    ----------
    matrices : list of ndarray

    Returns
    -------
    khatri_rao_product: matrix of shape ``(prod(n_i), m)``
        where ``prod(n_i) = prod([m.shape[0] for m in matrices])``
        i.e. the product of the number of rows of all the matrices in the
        product.

    Author
    ------
    Jean Kossaifi <https://github.com/tensorly>
    """

    n_columns = matrices[0].shape[1]
    n_factors = len(matrices)

    start = ord('a')
    common_dim = 'z'
    target = ''.join(chr(start + i) for i in range(n_factors))
    source = ','.join(i+common_dim for i in target)
    operation = source+'->'+target+common_dim
    return np.einsum(operation, *matrices).reshape((-1, n_columns))
