"""
These functions are ported from the tensorly package, https://tensorly.github.io/

license - BSD clause 3
"""
import numpy as np
import scipy as sci

def norm(tensor, order=2):
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



def kruskal_to_tensor(factors, lmbda=None):
    """
    Turns the Khatri-product of matrices into a full tensor
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
    
    Notes
    -----

    Author
    ------        
    Jean Kossaifi <https://github.com/tensorly>   

    """
    shape = [factor.shape[0] for factor in factors]
    
    if lmbda is not None:
        full_tensor = np.dot(factors[0]*lmbda, khatri_rao(factors[1:]).T)
    else:
        full_tensor = np.dot(factors[0], khatri_rao(factors[1:]).T)
    
    return fold(full_tensor, 0, shape)



def kruskal_to_unfolded(factors, mode):
    """
    Turns the khatri-product of matrices into an unfolded tensor
        turns ``factors = [|U_1, ... U_n|]`` into a mode-`mode`
        unfolding of the tensor
    
    Parameters
    ----------
    factors : ndarray list
        list of matrices, all with the same number of columns
        ie for all u in factor_matrices:
        u[i] has shape (s_u_i, R), where R is fixed
    mode: int
        mode of the desired unfolding
    
    Returns
    -------
    ndarray
        unfolded tensor of shape (tensor_shape[mode], -1)
    
    Notes
    -----
    Writing factors = [U_1, ..., U_n], we exploit the fact that
    ``U_k = U[k].dot(khatri_rao(U_1, ..., U_k-1, U_k+1, ..., U_n))``
    
    Author
    ------        
    Jean Kossaifi <https://github.com/tensorly>   
    
    """
    
    return factors[mode].dot(khatri_rao(factors, skip_matrix=mode).T)





def khatri_rao(matrices, skip_matrix=None, reverse=False):
    """
    
    Khatri-Rao product
    
    
    
    Parameters
    ----------
    matrices : tuple of ndarrays
        Matrices for which the columnwise Khatri-Rao product should be computedns  
    
    
    skip_matrix : None or int, optional, default is None
        if not None, index of a matrix to skip
    
    
    reverse : bool, optional
        if True, compute Khatri-Rao product in reverse order
    
    
    Returns
    -------
    khatri_rao_product: 
    
    
    Examples
    --------
    >>> A = np.random.randn(5, 2)
    >>> B = np.random.randn(4, 2)
    >>> C = khatrirao((A, B))
    >>> C.shape
    (20, 2)
    >>> (C[:, 0] == np.kron(A[:, 0], B[:, 0])).all()
    true
    >>> (C[:, 1] == np.kron(A[:, 1], B[:, 1])).all()
    true    
        
    
    
    Notes
    -----
    
    
    Author
    ------        
    Jean Kossaifi <https://github.com/tensorly>
    
        
    References
    ----------
    .. [1] T.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications",
       SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
    
    	"""
    
    if skip_matrix is not None:
        matrices = [matrices[i] for i in range(len(matrices)) if i != skip_matrix]
    
    n_columns = matrices[0].shape[1]
    
    # Optional part, testing whether the matrices have the proper size
    for i, matrix in enumerate(matrices):
        if matrix.ndim != 2:
            raise ValueError('All the matrices must have exactly 2 dimensions!'
                             'Matrix {} has dimension {} != 2.'.format(
                                 i, matrix.ndim))
            
        
        if matrix.shape[1] != n_columns:
            raise ValueError('All matrices must have same number of columns!'
                             'Matrix {} has {} columns != {}.'.format(
                                 i, matrix.shape[1], n_columns))
    
    n_factors = len(matrices)
    
    if reverse:
        matrices = matrices[::-1]
    
    
    start = ord('a')
    common_dim = 'z'
    target = ''.join(chr(start + i) for i in range(n_factors))
    source = ','.join(i+common_dim for i in target)
    operation = source+'->'+target+common_dim
    
    return sci.einsum(operation, *matrices).reshape((-1, n_columns))
          

