"""
These functions are ported from the tensorly package, https://tensorly.github.io/

license - BSD clause 3
"""
import numpy as np

def norm(tensor, order):
    """Computes the l-`order` norm of tensor
    Parameters
    ----------
    tensor : ndarray
    order : int
    Returns
    -------
    float
        l-`order` norm of tensor
    """
    if order == 1:
        return np.sum(np.abs(tensor))
    elif order == 2:
        return np.sqrt(np.sum(tensor**2))
    else:
        return np.sum(np.abs(tensor)**order)**(1/order)

def fold(unfolded_tensor, mode, shape):
    """Refolds the mode-`mode` unfolding into a tensor of shape `shape`
        In other words, refolds the n-mode unfolded tensor
        into the original tensor of the specified shape.
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

def unfold(tensor, mode=0):
    """Returns the mode-`mode` unfolding of `tensor` with modes starting at `0`.
    Parameters
    ----------
    tensor : ndarray
    mode : int, default is 0
           indexing starts at 0, therefore mode is in ``range(0, tensor.ndim)``
    Returns
    -------
    ndarray
        unfolded_tensor of shape ``(tensor.shape[mode], -1)``
    """
    return np.moveaxis(tensor, mode, 0).reshape((tensor.shape[mode], -1))

def kruskal_to_tensor(factors):
    """Turns the Khatri-product of matrices into a full tensor
        ``factor_matrices = [|U_1, ... U_n|]`` becomes
        a tensor shape ``(U[1].shape[0], U[2].shape[0], ... U[-1].shape[0])``
    Parameters
    ----------
    factors : ndarray list
        list of factor matrices, all with the same number of columns
        i.e. for all matrix U in factor_matrices:
        U has shape ``(s_i, R)``, where R is fixed and s_i varies with i
    Returns
    -------
    ndarray
        full tensor of shape ``(U[1].shape[0], ... U[-1].shape[0])``
    """
    shape = [factor.shape[0] for factor in factors]
    full_tensor = np.dot(factors[0], khatri_rao(factors[1:]).T)
    return fold(full_tensor, 0, shape)

def khatri_rao(matrices, skip_matrix=None):
    """Khatri-Rao product of a list of matrices
        This can be seen as a column-wise kronecker product.
        (see [1]_ for more details).
    Parameters
    ----------
    matrices : ndarray list
        list of matrices with the same number of columns, i.e.::
            for i in len(matrices):
                matrices[i].shape = (n_i, m)
    skip_matrix : None or int, optional, default is None
        if not None, index of a matrix to skip
    reverse : bool, optional
        if True, the order of the matrices is reversed
    Returns
    -------
    khatri_rao_product: matrix of shape ``(prod(n_i), m)``
        where ``prod(n_i) = prod([m.shape[0] for m in matrices])``
        i.e. the product of the number of rows of all the matrices in the product.
    """
    if skip_matrix is not None:
        matrices = [matrices[i] for i in range(len(matrices)) if i != skip_matrix]

    n_columns = matrices[0].shape[1]
    n_factors = len(matrices)

    start = ord('a')
    common_dim = 'z'
    target = ''.join(chr(start + i) for i in range(n_factors))
    source = ','.join(i+common_dim for i in target)
    operation = source+'->'+target+common_dim
    return np.einsum(operation, *matrices).reshape((-1, n_columns))
