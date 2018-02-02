import numpy as np
import scipy as sci



from tensortools.tensor_utils import fold, unfold



def compress(X, r, p, q, NN=False):
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Shape of input matrix 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~         
    dat_type =  X.dtype   
    
    if  dat_type == sci.float32: 
        isreal = True
    
    elif dat_type == sci.float64: 
        isreal = True
    
    else:
        raise ValueError("A.dtype is not supported")    
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compress each principal flattened version of tensor
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    Qlist = [] # init empty list to store Q
    
    if isinstance(p, list) == False: 
        p = sci.repeat(p, X.ndim)
        
    for mode in range(X.ndim):
        
        A = unfold(X, mode)
        m , n = A.shape 
        
        
        if p[mode] == None:  
            Qlist.append(None)

        elif (p[mode]+r) >= m:   
            Qlist.append(None)
        
        else:
        
            # Set number of samples
            k = min(p[mode] + r, m)
            if k < 0:
            	raise ValueError("The number of samples must be positive.")  
        
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #Generate a random sampling matrix O
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if NN==False:
                O = sci.array( sci.random.uniform( -1 , 1 , size=( n, k ) ) , dtype = dat_type ) 
                

            elif NN==True:
                O = sci.array( sci.random.uniform( 0 , 1 , size=( n, k ) ) , dtype = dat_type ) 
                         
                    
                
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #Build sample matrix Y : Y = A * O
            #Note: Y should approximate the range of A
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
            Y = A.dot(O)
            del(O)
            
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #Orthogonalize Y using economic QR decomposition: Y=QR
            #If q > 0 perfrom q subspace iterations
            #Note: check_finite=False may give a performance gain
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      
                       
            if q > 0:
                for i in np.arange( 1, q+1 ):
                    Y , _ = sci.linalg.lu( Y , permute_l=True, check_finite=False, overwrite_a=True )
                    Z , _ = sci.linalg.lu( sci.dot( A.T , Y ) , permute_l=True, check_finite=False, overwrite_a=True)
                    Y = sci.dot( A , Z )
                #End for
             #End if       
                
            Q , _ = sci.linalg.qr( Y ,  mode='economic' , check_finite=False, overwrite_a=True ) 
            del(Y)
            
            
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Project the data matrix a into a lower dimensional subspace
            # B = Q.T * A 
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
            B = Q.T.dot(A)   
        
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # fold matrix
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            shape = sci.array( X.shape )
            shape[mode] = k
            X = fold(B, mode, shape)

            Qlist.append( Q )
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return Q and B
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
    return ( Qlist, X )



