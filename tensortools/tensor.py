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