# activations.py

import numpy as np

class Softmax:
    @staticmethod
    def forward(z):
        exp_values = np.exp(z - np.max(z, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities

    @staticmethod
    def backward(dA, cache):
        Z = cache['Z']
        S = Softmax.forward(Z)
        dZ = dA * S * (1 - S)
        return dZ