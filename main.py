import numpy as np
from LayerDense import LayerDense

X = [
    [1, 2, 3, 2.5],
    [2, 5, -1, 2],
    [-1.5, 2.7, 3.3, -.8],
]

layer1 = LayerDense(4, 5)

layer1.forward(X)
print(layer1.output)
