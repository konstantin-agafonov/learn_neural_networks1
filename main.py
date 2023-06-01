import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

from LayerDense import LayerDense
from ActivationReLU import ActivationReLU
from ActivationSoftMax import ActivationSoftMax

nnfs.init()
np.random.seed(0)

X, y = spiral_data(100, 3)

layer1 = LayerDense(2, 3)
layer1.forward(X)

activation1 = ActivationReLU()
activation1.forward(layer1.output)

layer2 = LayerDense(3, 3)
layer2.forward(activation1.output)

activation2 = ActivationSoftMax()
activation2.forward(layer2.output)

print(activation2.output)

