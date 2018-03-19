import tensortools as tt
import numpy as np
import matplotlib.pyplot as plt

data = np.random.randn(100, 2)
x, y = data.T

print(data.shape)
prm = tt.tsp_linearize(data)

plt.plot(x, y, '.b')
plt.plot(x[prm], y[prm], '-r')
plt.show()
