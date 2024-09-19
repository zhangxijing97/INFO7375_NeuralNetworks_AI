# mcp_neuron.py

import numpy as np

class MCPNeuron:
    def __init__(self, n_inputs, threshold=0):
        self.weights = np.zeros(n_inputs)  # Initialize weights as zero for each input feature
        self.threshold = threshold  # Threshold for activation
    
    def predict(self, features):
        # Compute the weighted sum of inputs
        weighted_sum = np.dot(features, self.weights)
        # Apply activation function (binary step function)
        return 1 if weighted_sum >= self.threshold else 0
    
    def train(self, X, y, epochs=10, lr=0.1):
        update_count = 0  # Counter for updates
        # Training using simple weight update (Perceptron-like learning)
        for _ in range(epochs):
            for features, label in zip(X, y):
                prediction = self.predict(features)
                # Update weights if prediction is incorrect
                error = label - prediction
                self.weights += lr * error * features