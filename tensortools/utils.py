"""
Useful helper functions, not critical to core functionality of tensortools.
"""

import numpy as np
import math

def coarse_grain_1d(tensor, factor, axis=0, reducer=np.sum,
                    pad_mode='constant', pad_kwargs=dict(constant_values=0)):
    """Coarse grains a large tensor along axis by factor

    Args
    ----
    tensor : ndarray
    factor : int
        multiplicat
    axis : int
        mode to coarse grain (default=0)
    reducer : function
        reducing function to implement coarse-graining
    """
    if not isinstance(factor, int):
        raise ValueError('coarse-graining factor must be an integer.')
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError('invalid axis for coarse-graining.')

    # compute reshaping dimensions
    new_shape = [s for s in tensor.shape]
    new_shape[axis] = math.ceil(new_shape[axis]/factor)
    new_shape.insert(axis+1, factor)

    # pad tensor if necessary
    pad_width = factor*new_shape[axis] - tensor.shape[axis]
    if pad_width > 0:
        pw = [pad_width if a==axis else 0 for a in range(tensor.ndim)]
        tensor = np.pad(tensor, pw, pad_mode, **pad_kwargs)

    # sanity check
    assert pad_width >= 0
    
    # coarse-grain
    return reducer(tensor.reshape(*new_shape), axis=axis+1)

def coarse_grain(tensor, factors, **kwargs):
    """Coarse grains a large tensor along all modes by specified factors
    """
    for axis, factor in enumerate(factors):
        tensor = coarse_grain_1d(tensor, factor, axis=axis, **kwargs)

    return tensor

def soft_cluster_factor(factor):
    """Returns soft-clustering of data based on CP decomposition results.

    Args
    ----
    factors : ndarray
        Matrix holding low-dimensional CP factors in columns

    Returns
    -------
    cluster_ids : ndarray of ints
        List of cluster assignments for each row of factor matrix
    perm : ndarray of ints
        Permutation that groups rows by clustering and factor magnitude
    """
    
    # copy factor of interest
    f = np.copy(factor)

    # cluster based on score of maximum absolute value
    cluster_ids = np.argmax(np.abs(f), axis=1)
    scores = f[range(f.shape[0]), cluster_ids]

    # resort based on cluster assignment
    #i0 = np.argsort(cluster_ids)
    #f, scores = f[i0], scores[i0]

    # resort within each cluster
    perm = []
    for cluster in np.unique(cluster_ids):
        idx = np.where(cluster_ids == cluster)[0]
        perm += list(idx[np.argsort(scores[idx])][::-1])

    return cluster_ids, perm
