import scipy as sci
import numpy as np

from tensortools.tensors import Ktensor


def random_tensor(shape, rank, full=False, random_state=None):
    """
    Generates a random rank-R tensor of dimension IxJxK.
    
    Parameters
    ----------
    shape : tuple
        shape of the tensor
    
    rank : int
        rank of the tensor
    
    full : bool, optional, default is False
        if True, a full tensor is returned
        otherwise, the decomposed tensor is returned
    
    random_state : 'np.random.RandomState'
        
    Returns
    -------
    cp_tensor : ND-array or 2D-array list
        ND-array : full tensor if `full` is True
        2D-array list : list of factors otherwise
        
    Example
    -------        
    # Create a rank-2 tensor of dimension 5x5x5:
    X = cp_tensor((5,5,5), rank=2, full=True)
 
       
    """

    if random_state is None or isinstance(random_state, int):
            rns = sci.random.RandomState(random_state)
    
    elif isinstance(random_state, sci.random.RandomState):
            rns = random_state
    
    else:
        raise ValueError('Seed should be None, int or np.random.RandomState')


    factors = [rns.standard_normal((i, rank)) for i in shape]
    
    
    if full:
        return Ktensor(factors).full()
    else:
        return factors
