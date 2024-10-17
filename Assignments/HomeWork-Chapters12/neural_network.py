# neural_network.py

import numpy as np
from activations import Softmax

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.parameters = self.initialize_parameters()

    def initialize_parameters(self):
        parameters = {}
        L = len(self.layers)
        for l in range(1, L):
            parameters[f'W{l}'] = np.random.randn(self.layers[l-1], self.layers[l]) * 0.01
            parameters[f'b{l}'] = np.zeros((1, self.layers[l]))
        return parameters

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z))  # for numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # def forward(self, X):
    #     cache = {'A0': X}
    #     L = len(self.layers) - 1
    #     for l in range(1, L + 1):
    #         cache[f'Z{l}'] = np.dot(cache[f'A{l-1}'], self.parameters[f'W{l}']) + self.parameters[f'b{l}']
    #         cache[f'A{l}'] = self.sigmoid(cache[f'Z{l}'])
    #     return cache
    
    def forward(self, X):
        cache = {'A0': X}
        L = len(self.layers) - 1
        for l in range(1, L):
            cache[f'Z{l}'] = np.dot(cache[f'A{l-1}'], self.parameters[f'W{l}']) + self.parameters[f'b{l}']
            cache[f'A{l}'] = self.sigmoid(cache[f'Z{l}'])
        # Apply softmax to the output layer
        cache[f'Z{L}'] = np.dot(cache[f'A{L-1}'], self.parameters[f'W{L}']) + self.parameters[f'b{L}']
        cache[f'A{L}'] = self.softmax(cache[f'Z{L}'])
        return cache

    def compute_loss(self, Y, Y_hat):
        # Mean Squared Error
        return np.mean((Y - Y_hat) ** 2)

    def backward(self, cache, X, Y, learning_rate):
        gradients = {}
        L = len(self.layers) - 1
        m = X.shape[0]

        # Compute the gradient on the output layer
        dA = cache[f'A{L}'] - Y
        for l in reversed(range(1, L + 1)):
            dZ = dA * self.sigmoid_derivative(cache[f'A{l}'])
            dW = (1 / m) * np.dot(cache[f'A{l-1}'].T, dZ)
            db = (1 / m) * np.sum(dZ, axis=0, keepdims=True)
            if l > 1:
                dA = np.dot(dZ, self.parameters[f'W{l}'].T)
            
            gradients[f'dW{l}'] = dW
            gradients[f'db{l}'] = db

            # Update weights and biases
            self.parameters[f'W{l}'] -= learning_rate * gradients[f'dW{l}']
            self.parameters[f'b{l}'] -= learning_rate * gradients[f'db{l}']

    def train(self, X, Y, epochs, learning_rate):
        for epoch in range(epochs):
            cache = self.forward(X)
            Y_hat = cache[f'A{len(self.layers)-1}']
            loss = self.compute_loss(Y, Y_hat)
            self.backward(cache, X, Y, learning_rate)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')