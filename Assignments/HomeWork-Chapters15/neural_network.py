# neural_network.py

import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.parameters = self.initialize_parameters()

    def initialize_parameters(self):
        parameters = {}
        L = len(self.layers)
        for l in range(1, L):
            # He initialization: scales the random values by sqrt(2 / layer size)
            parameters[f'W{l}'] = np.random.randn(self.layers[l-1][0], self.layers[l][0]) * np.sqrt(2 / self.layers[l-1][0])
            parameters[f'b{l}'] = np.zeros((1, self.layers[l][0]))
        return parameters


    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def get_activation(self, func_name):
        if func_name == 'ReLU':
            return self.relu
        elif func_name == 'Sigmoid':
            return self.sigmoid
        else:
            raise ValueError(f"Unknown activation function: {func_name}")

    def get_activation_derivative(self, func_name):
        if func_name == 'ReLU':
            return self.relu_derivative
        elif func_name == 'Sigmoid':
            return self.sigmoid_derivative
        else:
            raise ValueError(f"Unknown activation function: {func_name}")

    def forward(self, X):
        cache = {'A0': X}
        L = len(self.layers)
        for l in range(1, L):
            Z = np.dot(cache[f'A{l-1}'], self.parameters[f'W{l}']) + self.parameters[f'b{l}']
            activation_func = self.get_activation(self.layers[l][1])
            cache[f'A{l}'] = activation_func(Z)
        return cache

    def compute_loss(self, Y, Y_hat):
        return np.mean((Y - Y_hat) ** 2)

    def backward(self, cache, X, Y, learning_rate):
        gradients = {}
        L = len(self.layers)
        m = X.shape[0]

        # Compute the gradient on the output layer
        dA = cache[f'A{L-1}'] - Y
        for l in reversed(range(1, L)):
            activation_derivative_func = self.get_activation_derivative(self.layers[l][1])
            dZ = dA * activation_derivative_func(cache[f'A{l}'])
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