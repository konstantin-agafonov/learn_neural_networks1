import numpy as np


class ActivationSoftMax:
    def __init__(self):
        self.output = []

    def forward(self, inputs):
        # exp_values = np.exp(inputs)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
