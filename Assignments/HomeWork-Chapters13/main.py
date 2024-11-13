# main.py

import numpy as np

import numpy as np

# Function to perform convolution
def convolution2d(image, kernel):
    # Get dimensions of image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Calculate the dimensions of the output matrix (no padding, no striding)
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1
    
    # Initialize the output matrix
    output = np.zeros((output_height, output_width))
    
    # Perform the convolution operation
    for i in range(output_height):
        for j in range(output_width):
            # Extract the region of interest (ROI) from the image
            roi = image[i:i+kernel_height, j:j+kernel_width]
            # Perform element-wise multiplication and sum the result
            output[i, j] = np.sum(roi * kernel)
    
    return output

# Create 6x6 images
images = [
    np.array([[1, 2, 3, 4, 5, 6],
              [7, 8, 9, 10, 11, 12],
              [13, 14, 15, 16, 17, 18],
              [19, 20, 21, 22, 23, 24],
              [25, 26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35, 36]]),
    
    np.array([[2, 3, 4, 5, 6, 7],
              [8, 9, 10, 11, 12, 13],
              [14, 15, 16, 17, 18, 19],
              [20, 21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30, 31],
              [32, 33, 34, 35, 36, 37]]),
    
    np.array([[1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1]]),
    
    np.array([[1, 4, 7, 10, 13, 16],
              [2, 5, 8, 11, 14, 17],
              [3, 6, 9, 12, 15, 18],
              [4, 7, 10, 13, 16, 19],
              [5, 8, 11, 14, 17, 20],
              [6, 9, 12, 15, 18, 21]]),
    
    np.array([[3, 2, 1, 6, 5, 4],
              [6, 5, 4, 3, 2, 1],
              [1, 6, 5, 4, 3, 2],
              [4, 3, 2, 1, 6, 5],
              [5, 4, 3, 2, 1, 6],
              [2, 1, 6, 5, 4, 3]]),
]

# Define a 3x3 kernel for all images
kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])

# Apply convolution on each image
outputs = [convolution2d(image, kernel) for image in images]

# Print the results
for i, output in enumerate(outputs, 1):
    print(f"Output of Image {i}:")
    print(output)
    print("\n")