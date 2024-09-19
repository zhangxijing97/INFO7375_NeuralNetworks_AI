# main.py

import pandas as pd # Import Pandas for data manipulation
from sklearn.model_selection import train_test_split # For splitting data into training and testing sets
from sklearn.metrics import accuracy_score # For calculating the accuracy of the model
import numpy as np # Import NumPy for numerical operations

from mcp_neuron import MCPNeuron  # Import the class from the mcp_neuron file

# Load the dataset from the CSV file
df = pd.read_csv('Datasets/breast_cancer_dataset.csv')

# Separate features and labels
X = df.drop('target', axis=1)  # Features
y = df['target']  # Labels (0 or 1)

# Normalize data between 0 and 1
X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the McCulloch-Pitts neuron model with the number of features
mcp_neuron = MCPNeuron(n_inputs=X_train.shape[1], threshold=0.5)

# Train the model
mcp_neuron.train(X_train.values, y_train.values, epochs=100, lr=0.01)

# Make predictions on the test set
y_pred = [mcp_neuron.predict(x) for x in X_test.values]

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")