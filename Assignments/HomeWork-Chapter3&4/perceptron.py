import numpy as np

class Perceptron:
    def __init__(self, input_size, num_classes=10, learning_rate=0.1, epochs=100):
        self.weights = np.zeros((input_size, num_classes))  # Weight matrix for multi-class
        self.bias = np.zeros(num_classes)  # One bias per class
        self.lr = learning_rate
        self.epochs = epochs
        self.losses = []  # To track the loss over epochs

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z))  # For numerical stability
        return exp_z / np.sum(exp_z)

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        probabilities = self.softmax(z)
        return np.argmax(probabilities)  # Return the class with the highest probability

    def train(self, X, y):
        for epoch in range(self.epochs):
            print(self.epochs)
            for inputs, label in zip(X, y):
                z = np.dot(inputs, self.weights) + self.bias
                probabilities = self.softmax(z)
                error = probabilities
                error[label] -= 1  # One-hot encoding of the label
                # Gradient descent update for each class
                self.weights -= self.lr * np.outer(inputs, error)
                self.bias -= self.lr * error