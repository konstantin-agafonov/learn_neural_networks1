import numpy as np


class ActivationReLU:
    def __init__(self):
        self.output = []

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
