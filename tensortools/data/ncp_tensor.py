import scipy as sci
import numpy as np

from tensortools.tensor_utils import kruskal_to_tensor

def ncp_tensor(shape, rank, full=False, random_state=None):
    """
    Generates a random Nonnegative CP tensor
    
    Parameters
    ----------
    shape : tuple
        shape of the tensor to generate
    rank : int
        rank of the CP decomposition
    full : bool, optional, default is False
        if True, a full tensor is returned
        otherwise, the decomposed tensor is returned
    random_state : `np.random.RandomState`
        
    Returns
    -------
    cp_tensor : ND-array or 2D-array list
        ND-array : full tensor if `full` is True
        2D-array list : list of factors otherwise
    """

    if random_state is None or isinstance(random_state, int):
            rns = sci.random.RandomState(random_state)
    
    elif isinstance(random_state, sci.random.RandomState):
            rns = random_state
    
    else:
        raise ValueError('Seed should be None, int or np.random.RandomState')


    factors = [ sci.maximum(0.0, rns.standard_normal((i, rank))) for i in shape]
    
    if full:
        return kruskal_to_tensor(factors)
    else:
        return factors
