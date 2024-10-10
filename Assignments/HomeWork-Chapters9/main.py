# main.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

from neural_network import NeuralNetwork

# Load the dataset
df = pd.read_csv('Datasets/breast_cancer_dataset.csv')

# Separate features and labels
X = df.drop('target', axis=1)  # Features
y = df['target']  # Labels (0 or 1)

# Normalize data between 0 and 1
X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Neural Network model
input_size = X_train.shape[1]
hidden_size = 50  # Adjust hidden_size as needed
num_classes = 2  # Since it's a binary classification problem
learning_rate = 0.01
epochs = 100

neural_network = NeuralNetwork(input_size, hidden_size, num_classes, learning_rate, epochs)

# Train the Neural Network model
neural_network.train(X_train.values, y_train.values)

# Make predictions on the test set
y_pred = neural_network.predict(X_test.values)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
