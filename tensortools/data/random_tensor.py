import scipy as sci
import numpy as np

from tensortools.tensors import KTensor


def randn_tensor(shape, rank, nonnegative=False, ktensor=False, random_state=None):
    """
    Generates a random N-way tensor with rank R, where the entries are 
    drawn from the standard normal distribution. 
    
    Parameters
    ----------
    shape : tuple
        shape of the tensor
    
    rank : integer
        rank of the tensor
        
    nonnegative : bool 
        If ``True`` a nonnegative tensor is returned, otherwise the entries are
        standard normal distributed. 
        
    ktensor : bool
        If true, a KTensor object is returned, i.e., the components are in factored
        form ``[U_1, U_2, ... U_N]``; Otherwise an N-way array is returned.
    
    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator; 
        If RandomState instance, random_state is the random number generator; 
        If None, the random number generator is the RandomState instance used by np.random.

        
    Returns
    -------
    X : (I_1, ..., I_N) array_like
        N-way tensor with rank R.
        
    Example
    -------        
    >>> # Create a rank-2 tensor of dimension 5x5x5:
    >>> import tensortools as tt
    >>> X = tt.randn_tensor((5,5,5), rank=2)
 
       
    """

    if random_state is None or isinstance(random_state, int):
            rns = sci.random.RandomState(random_state)
    
    elif isinstance(random_state, sci.random.RandomState):
            rns = random_state
    
    else:
        raise ValueError('Seed should be None, int or np.random.RandomState')

    if nonnegative == False:
        factors = [rns.standard_normal((i, rank)) for i in shape]
    
    elif nonnegative == True:
        factors = [sci.maximum(0.0, rns.standard_normal((i, rank))) for i in shape]
        
    
    if ktensor == False:
        return KTensor(factors).full()
    
    else:
        return KTensor(factors)
    
    
    
    
    

    
def rand_tensor(shape, rank, nonnegative=False, ktensor=False, random_state=None):
    """
    Generates a random N-way tensor with rank R, where the entries are 
    drawn from the standard uniform distribution in the interval [0.0,1]. 
    
    Parameters
    ----------
    shape : tuple
        shape of the tensor
    
    rank : integer
        rank of the tensor
        
    nonnegative : bool 
        If ``True`` a nonnegative tensor is returned, otherwise the entries are
        standard normal distributed. 
        
    ktensor : bool
        If true, a KTensor object is returned, i.e., the components are in factored
        form ``[U_1, U_2, ... U_N]``; Otherwise an N-way array is returned.
    
    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator; 
        If RandomState instance, random_state is the random number generator; 
        If None, the random number generator is the RandomState instance used by np.random.

        
    Returns
    -------
    X : (I_1, ..., I_N) array_like
        N-way tensor with rank R.
        
    Example
    -------        
    >>> # Create a rank-2 tensor of dimension 5x5x5:
    >>> import tensortools as tt
    >>> X = tt.randn_tensor((5,5,5), rank=2)
 
       
    """

    if random_state is None or isinstance(random_state, int):
            rns = sci.random.RandomState(random_state)
    
    elif isinstance(random_state, sci.random.RandomState):
            rns = random_state
    
    else:
        raise ValueError('Seed should be None, int or np.random.RandomState')


    factors = [rns.uniform(0.0, 1.0, size=(i, rank)) for i in shape]
        
    
    if ktensor == False:
        return KTensor(factors).full()
    
    else:
        return KTensor(factors)        