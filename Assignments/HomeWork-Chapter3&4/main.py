# main.py

# NOTE: Before running this script, make sure to run 'resize_mnist_images.py' to generate the 'resized_mnist.csv' file.
# This file is required to load the MNIST dataset resized to 20x20 pixels.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

from perceptron import Perceptron

# Load the dataset
df = pd.read_csv('Assignments/HomeWork-Chapter3&4/resized_mnist.csv')

# Separate features and labels
X = df.drop('label', axis=1)  # Features
y = df['label']  # Labels (0-9)

# Normalize data between 0 and 1
X = X / 255.0

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Perceptron model for multi-class classification
perceptron = Perceptron(input_size=X_train.shape[1], num_classes=10, learning_rate=0.01, epochs=100)

# Train the Perceptron model
perceptron.train(X_train.values, y_train.values)

# Make predictions on the test set
y_pred = [perceptron.predict(x) for x in X_test.values]

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")