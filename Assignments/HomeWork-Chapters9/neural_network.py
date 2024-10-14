import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, num_classes, learning_rate=0.1, epochs=100):
        # Initialize weights
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros(hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, num_classes) * 0.01
        self.bias_output = np.zeros(num_classes)
        self.lr = learning_rate
        self.epochs = epochs

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z))  # For numerical stability
        return exp_z / np.sum(exp_z)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def tanh(self, z):
        return np.tanh(z)

    def tanh_derivative(self, z):
        return 1 - np.tanh(z) ** 2

    def linear(self, z):
        return z

    def predict(self, X):
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.relu(hidden_input)
        output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        output = self.softmax(output_input)
        return np.argmax(output, axis=1)

    def compute_loss(self, y_pred, y_true):
        # One-hot encoding of the true labels
        y_true_one_hot = np.zeros_like(y_pred)
        y_true_one_hot[np.arange(len(y_true)), y_true] = 1
        # Cross-entropy loss
        return -np.sum(y_true_one_hot * np.log(y_pred + 1e-12)) / y_pred.shape[0]

    def train(self, X, y):
        for epoch in range(self.epochs):
            for inputs, label in zip(X, y):
                # Forward pass
                hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
                hidden_output = self.relu(hidden_input)
                output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
                output = self.softmax(output_input)

                # Compute error
                error_output = output.copy()
                error_output[label] -= 1  # One-hot encoding of the label

                # Backpropagation
                delta_output = error_output
                delta_hidden = np.dot(self.weights_hidden_output, delta_output) * self.relu_derivative(hidden_output)

                # Update weights and biases
                self.weights_hidden_output -= self.lr * np.outer(hidden_output, delta_output)
                self.bias_output -= self.lr * delta_output
                self.weights_input_hidden -= self.lr * np.outer(inputs, delta_hidden)
                self.bias_hidden -= self.lr * delta_hidden

            # Print the loss for the current epoch
            # loss = self.compute_loss(output, y)
            print(f"Epoch {epoch+1} training in progress... Loss: {loss:.4f}")