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
            print(f"Epoch {epoch+1} is training in progress...")
            for inputs, label in zip(X, y):
                z = np.dot(inputs, self.weights) + self.bias
                probabilities = self.softmax(z)
                error = probabilities
                error[label] -= 1  # One-hot encoding of the label
                # Gradient descent update for each class
                self.weights -= self.lr * np.outer(inputs, error)
                self.bias -= self.lr * error

    def generate_digit_image(self, target_class, steps=10000, step_size=1.0):
        # Initialize random noise image
        generated_image = np.random.rand(self.weights.shape[0]) * 0.1  # Scaled small random noise

        for step in range(steps):
            z = np.dot(generated_image, self.weights) + self.bias
            probabilities = self.softmax(z)
            
            # Calculate the gradient of the target_class neuron with respect to the image
            gradient = np.dot(self.weights[:, target_class], probabilities[target_class] - 1)
            
            # Apply gradient ascent
            generated_image += step_size * gradient
            
            # Optional: apply some form of regularization to keep the pixel values valid
            generated_image = np.clip(generated_image, 0, 1)  # Ensure pixel values are between 0 and 1
            
            if step % 100 == 0:  # Print probability every 50 steps to monitor progress
                print(f"Step {step}: Probability of class {target_class} = {probabilities[target_class]}")
        
        return generated_image