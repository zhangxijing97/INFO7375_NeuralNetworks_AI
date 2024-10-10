# # main.py

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from neural_network import NeuralNetwork, DenseLayer

# # Load the dataset
# df = pd.read_csv('Datasets/breast_cancer_dataset.csv')

# # Separate features and labels
# X = df.drop('target', axis=1)
# y = df['target'].values

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize the features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Create the neural network model
# model = NeuralNetwork()
# model.add(DenseLayer(input_size=X_train.shape[1], output_size=64, activation='relu'))
# model.add(DenseLayer(input_size=64, output_size=32, activation='relu'))
# model.add(DenseLayer(input_size=32, output_size=1, activation='sigmoid'))

# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(X_train, y_train, epochs=50, batch_size=32, learning_rate=0.001, validation_data=(X_test, y_test))

# # Evaluate the model on the test set
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print(f'Test accuracy: {test_accuracy:.4f}')

import pandas as pd
from neural_network import NeuralNetwork, DenseLayer

# Load the dataset
df = pd.read_csv('Datasets/breast_cancer_dataset.csv')

# Separate features and labels
X = df.drop('target', axis=1).values  # Features
y = df['target'].values.reshape(-1, 1)  # Labels (0 or 1) as a column vector

# Create the neural network model
model = NeuralNetwork()
model.add(DenseLayer(input_size=X.shape[1], output_size=64, activation='relu'))  # First layer
model.add(DenseLayer(input_size=64, output_size=32, activation='relu'))  # Second layer
model.add(DenseLayer(input_size=32, output_size=1, activation='sigmoid'))  # Output layer

# Train the model using the entire dataset
model.fit(X, y, epochs=50, learning_rate=0.001)