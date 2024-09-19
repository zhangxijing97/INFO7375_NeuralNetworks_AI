import csv
import os
import numpy as np
from tensorflow.keras.datasets import mnist
from PIL import Image

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Directory path where CSV file will be saved
output_dir = 'Assignments/HomeWork-Chapter3&4'
os.makedirs(output_dir, exist_ok=True)

# File path
csv_file_path = os.path.join(output_dir, 'resized_mnist.csv')

# Create CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write headers
    writer.writerow(['label'] + [f'pixel_{i}' for i in range(400)])  # 20x20 = 400 pixels
    
    # Resize and write pixel data
    for image, label in zip(x_train, y_train):
        img = Image.fromarray(image).resize((20, 20))  # Resize to 20x20
        img_data = np.array(img).flatten()  # Flatten to a 1D array
        writer.writerow([label] + img_data.tolist())  # Write label and pixel values

print(f"CSV file saved in '{output_dir}'")