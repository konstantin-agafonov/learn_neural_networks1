import numpy as np
np.random.seed(0)


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = .1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros(n_neurons)
        self.output = []

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
