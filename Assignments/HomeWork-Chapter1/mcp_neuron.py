# mcp_neuron.py

import numpy as np

class MCPNeuron:
    def __init__(self, n_inputs, threshold=0):
        self.weights = np.zeros(n_inputs)  # Initialize weights as zero
        self.threshold = threshold  # Threshold for activation
    
    def predict(self, inputs):
        # Compute the weighted sum of inputs
        weighted_sum = np.dot(inputs, self.weights)
        # Apply activation function (binary step function)
        return 1 if weighted_sum >= self.threshold else 0
    
    def train(self, X, y, epochs=100, lr=0.1):
        # Training using simple weight update (Perceptron-like learning)
        for _ in range(epochs):
            print(self.weights)
            for inputs, label in zip(X, y):
                prediction = self.predict(inputs)
                # Update weights if prediction is incorrect
                error = label - prediction
                self.weights += lr * error * inputs