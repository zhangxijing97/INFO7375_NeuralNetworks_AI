# main.py

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

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the network structure
input_size = X_train.shape[1]  # Adjust input size dynamically
layers = [
    (input_size, 'ReLU'),  # Input layer: size matches the input data
    (10, 'ReLU'),    # Layer 2: 10 neurons, ReLU
    (8, 'ReLU'),    # Layer 3: 8 neurons, ReLU
    (8, 'ReLU'),    # Layer 4: 8 neurons, ReLU
    (4, 'ReLU'),    # Layer 5: 4 neurons, ReLU
    (1, 'Sigmoid')  # Output layer: 1 neuron, Sigmoid
]

# Hyperparameters
epochs = 20000
learning_rate = 0.5

# Create and train the neural network
nn = NeuralNetwork(layers)
nn.train(X_train, y_train, epochs, learning_rate)

# Test the neural network
predictions = nn.forward(X_test)[f'A{len(layers)-1}']  # Adjust index for last layer
predictions = (predictions > 0.5).astype(int)  # Convert probabilities to binary predictions

# Calculate accuracy
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")