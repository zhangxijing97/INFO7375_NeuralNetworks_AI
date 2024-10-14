# # main.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from neural_network import NeuralNetwork

# Load the dataset
df = pd.read_csv('Datasets/breast_cancer_dataset.csv')

# Separate features and labels
X = df.drop('target', axis=1).values  # Features
y = df['target'].values.reshape(-1, 1)  # Labels (0 or 1)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the network structure
input_size = X_train.shape[1]
hidden_layers = [6, 6]  # Example: two hidden layers with 30 and 30 units
output_size = 1
layers = [input_size] + hidden_layers + [output_size]

# Hyperparameters
epochs = 10000
learning_rate = 0.1

# Create and train the neural network
nn = NeuralNetwork(layers)
nn.train(X_train, y_train, epochs, learning_rate)

# Test the neural network
predictions = nn.forward(X_test)[f'A{len(layers)-1}']
predictions = (predictions > 0.5).astype(int)  # Convert probabilities to binary predictions

# Calculate accuracy
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")