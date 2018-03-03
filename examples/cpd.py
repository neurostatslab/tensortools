from tensortools import randn_tensor
from tensortools.optimize import cp_als
import numpy as np

# dimensions
I, J, K, R = 15, 15, 15, 3

# make tensor
X = randn_tensor((I,J,K), rank=R)

# add noise
X += np.random.randn(I,J,K)

# fit cp decomposition
P = cp_als(X, rank=R, trace=False, random_state=random_state)
