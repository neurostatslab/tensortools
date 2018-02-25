import scipy as sci
import numpy as np

from tensortools.tensors import Ktensor


def randn_tensor(shape, rank, ktensor=False, random_state=None):
    """
    Generates a random rank-R tensor of shape `(IxJxK)`.
    
    Parameters
    ----------
    shape : tuple
        shape of the tensor
    
    rank : int
        rank of the tensor
        
    ktensor : bool
        If true, a Ktensor object is returned;
        Otherwise an array of shape `(IxJxK)` is returned.
    
    random_state : `np.random.RandomState`
        
    Returns
    -------
    X : ndarray
        Rank-R tensor of shape `(IxJxK)`.
        
    Example
    -------        
    # Create a rank-2 tensor of dimension 5x5x5:
    X = cp_tensor((5,5,5), rank=2)
 
       
    """

    if random_state is None or isinstance(random_state, int):
            rns = sci.random.RandomState(random_state)
    
    elif isinstance(random_state, sci.random.RandomState):
            rns = random_state
    
    else:
        raise ValueError('Seed should be None, int or np.random.RandomState')

    factors = [rns.standard_normal((i, rank)) for i in shape]
    
    if ktensor == False:
        return Ktensor(factors).full()
    
    else:
        return Ktensor(factors)

