# neural_network.py

import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size, activation=None):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.activation = activation
        self.inputs = None
        self.outputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases
        if self.activation == 'relu':
            self.outputs = np.maximum(0, self.outputs)
        elif self.activation == 'sigmoid':
            self.outputs = 1 / (1 + np.exp(-self.outputs))
        return self.outputs

    def backward(self, dvalues):
        # Backward pass calculation
        if self.activation == 'relu':
            self.dinputs = dvalues * (self.inputs > 0)
        elif self.activation == 'sigmoid':
            self.dinputs = dvalues * (self.outputs * (1 - self.outputs))
        else:
            self.dinputs = dvalues  # For no activation or other types

        self.dweights = np.dot(self.inputs.T, self.dinputs)
        self.dbiases = np.sum(self.dinputs, axis=0, keepdims=True)
        return self.dinputs

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, y_true, y_pred):
        # Calculate the loss gradient
        dvalues = y_pred - y_true
        for layer in reversed(self.layers):
            dvalues = layer.backward(dvalues)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.weights -= learning_rate * layer.dweights
            layer.biases -= learning_rate * layer.dbiases

    def fit(self, X, y, epochs=50, learning_rate=0.001):
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)

            # Backward pass
            self.backward(y, y_pred)

            # Update weights
            self.update(learning_rate)

            # Optionally print loss for monitoring
            loss = np.mean((y_pred - y) ** 2)  # Example loss (mean squared error)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")