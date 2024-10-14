# dropout.py

import numpy as np

class Dropout:
    def __init__(self, rate):
        self.rate = rate
        self.mask = None

    def forward(self, A):
        """
        Apply dropout during training.

        Parameters:
        - A: Activations from the previous layer

        Returns:
        - A_dropout: Activations after applying dropout
        """
        if self.rate > 0:
            self.mask = np.random.binomial(1, 1 - self.rate, size=A.shape) / (1 - self.rate)
            return A * self.mask
        return A

    def backward(self, dA):
        """
        Backpropagate through dropout.

        Parameters:
        - dA: Gradient of the loss with respect to activations

        Returns:
        - dA_dropout: Gradient after applying dropout mask
        """
        if self.rate > 0:
            return dA * self.mask
        return dA