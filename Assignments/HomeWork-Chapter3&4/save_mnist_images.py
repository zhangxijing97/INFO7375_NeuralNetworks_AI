import csv
import os
import numpy as np
from tensorflow.keras.datasets import mnist
from PIL import Image

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Directory paths
output_dir = 'Assignments/HomeWork-Chapter3&4'
image_dir = os.path.join(output_dir, 'resized_mnist_images')
os.makedirs(image_dir, exist_ok=True)

# CSV file path
csv_file_path = os.path.join(output_dir, 'mnist_image_data.csv')

# Create CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write headers
    writer.writerow(['label', 'image_file'])
    
    # Resize and save images, write metadata to CSV
    for i, (image, label) in enumerate(zip(x_train, y_train)):
        img = Image.fromarray(image).resize((20, 20))  # Resize to 20x20
        image_filename = f'mnist_image_{i}.png'
        image_path = os.path.join(image_dir, image_filename)
        
        # Save image to the folder
        img.save(image_path)
        
        # Write label and image file name to the CSV
        writer.writerow([label, image_filename])

print(f"CSV file saved in '{csv_file_path}'")
print(f"Images saved in '{image_dir}'")