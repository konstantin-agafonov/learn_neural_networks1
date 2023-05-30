import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

from LayerDense import LayerDense
from ActivationReLU import ActivationReLU

nnfs.init()
np.random.seed(0)

'''X = [
    [1, 2, 3, 2.5],
    [2, 5, -1, 2],
    [-1.5, 2.7, 3.3, -.8],
]'''

X, y = spiral_data(100, 3)

'''plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
plt.show()'''


layer1 = LayerDense(2, 5)
activation1 = ActivationReLU()
layer1.forward(X)
print(layer1.output)
activation1.forward(layer1.output)

print(activation1.output)
