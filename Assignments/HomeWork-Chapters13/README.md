# HW to Chapter 13 “Convolutional Layer”

# Non-programming Assignment

## 1. What is convolution operation and how does it work?

The convolution operation is a mathematical process used primarily in convolutional neural networks (CNNs) to extract features from input data, such as images or time series. It is a fundamental operation that allows the network to learn spatial hierarchies of features.<br>

How Convolution Works:<br>
1. Input Image: The input is typically a multi-dimensional matrix (e.g., an image with height, width, and depth). For a grayscale image, the depth is 1, and for a color image (RGB), the depth is 3 (one channel per color).

2. Filter (Kernel): A filter (or kernel) is a smaller matrix that slides over the input image. It is also multi-dimensional, usually smaller than the input image. For example, a common filter size is 3x3 or 5x5. The filter is responsible for detecting specific features, such as edges, textures, or patterns in the image.

- Sliding the Filter: The filter slides across the input image, typically with a certain stride (step size). The filter is applied to different regions of the image. For each position:

- The filter's values are multiplied element-wise with the corresponding values in the image (a dot product).
The result is summed to produce a single value, which represents the feature extracted from that specific region.

4. Output Feature Map: The output of the convolution is a new matrix (often called a feature map or activation map) that represents the transformed image, containing the detected features from the input image.

Example of Convolution:<br>
Suppose you have a 5x5 image and a 3x3 filter:<br>

```
Input Image (5x5):
[1, 2, 3, 4, 5]
[6, 7, 8, 9, 10]
[11, 12, 13, 14, 15]
[16, 17, 18, 19, 20]
[21, 22, 23, 24, 25]

Filter (3x3):
[1, 0, -1]
[1, 0, -1]
[1, 0, -1]

Result (3x3):
[-6, -6, -6]
[-6, -6, -6]
[-6, -6, -6]
```

To apply the filter to the image, we place it over the top-left corner of the image and perform element-wise multiplication:<br>

- Multiply each element in the filter by the corresponding element in the image.
- Sum the products to get a single value for that position.

For the top-left corner:<br>
(1∗1)+(2∗0)+(3∗−1)+(6∗1)+(7∗0)+(8∗−1)+(11∗1)+(12∗0)+(13∗−1) = −6<br>

Key Points:<br>
- The filter extracts local patterns in the image, such as edges or textures, by emphasizing specific features in the image region.
- By using multiple filters in a convolutional layer, the network can learn to detect increasingly complex patterns as the image moves through deeper layers of the network.

If the output values from the convolution are very close to each other, it generally means that the image does not have strong vertical edges in the region being processed by the filter: <br>
```
# Image with Strong Vertical Edge
[  0,   0,   0, 255, 255]
[  0,   0,   0, 255, 255]
[  0,   0,   0, 255, 255]
[  0,   0,   0, 255, 255]
[  0,   0,   0, 255, 255]

# Vertical Edge Filter (Sobel Filter, 3x3)
[ 1,  0, -1]
[ 1,  0, -1]
[ 1,  0, -1]

# Convolution output with a vertical edge filter
[   0,  -510, -765]
[   0,  -510, -765]
[   0,  -510, -765]
[   0,  -510, -765]
[   0,  -510, -765]

# Image with Smooth Vertical Edge
[  0,   0,   0, 127, 127]
[  0,   0,   0, 127, 127]
[  0,   0,   0, 127, 127]
[  0,   0,   0, 127, 127]
[  0,   0,   0, 127, 127]

# Vertical Edge Filter (Sobel Filter, 3x3)
[ 1,  0, -1]
[ 1,  0, -1]
[ 1,  0, -1]

# Convolution output with a vertical edge filter
[   0,  -254, -381]
[   0,  -254, -381]
[   0,  -254, -381]
[   0,  -254, -381]
[   0,  -254, -381]
```

## 2. Why do we need convolutional layers in neural networks?

1. Feature Extraction
Convolutional layers help in automatically detecting features from the raw input, such as edges, textures, shapes, and more complex patterns. In an image, low-level features (like edges and corners) can be learned by applying small filters (kernels) to the image. As the network deepens, these features combine to form high-level features (like faces, objects, or other complex patterns).<br>

- First layers may learn simple patterns like edges, corners, or textures.
- Deeper layers combine these patterns to recognize more complex objects or structures.

2. Parameter Sharing
The same filter (kernel) is applied across different regions of the input image. This is known as parameter sharing. The same filter is reused across the entire image, meaning fewer parameters need to be learned compared to fully connected layers where each pixel is treated independently.

- This reduces the number of parameters significantly, making the network more efficient and reducing the risk of overfitting.

3. Translation Invariance
The network can recognize features regardless of where they appear in the image.<br>

For example, an edge in the top-left corner of an image can be detected by the same filter as an edge in the bottom-right corner.<br>

4. Reduction in Computational Complexity
Since convolutional layers use small filters and share weights across the image, they significantly reduce the computational complexity compared to fully connected layers. Instead of treating each pixel as a separate entity with its own set of weights, convolutions share weights and reduce the total number of calculations, making the model more efficient.<br>

This makes convolutional layers ideal for high-dimensional data like images, where fully connected layers would be too computationally expensive.<br>

5. Preserving Spatial Hierarchy
Unlike fully connected layers, which flatten the image into a one-dimensional vector, convolutional layers preserve the spatial hierarchy of the data. This means that the relative position of features within the image is maintained.<br>

In an image, nearby pixels are likely to be related in some way (e.g., in terms of color, texture, etc.), and convolutional layers respect this relationship, ensuring that the spatial arrangement is captured.<br>

6. Learning Filters
Instead of manually engineering features, convolutional layers automatically learn the most important features during training. Initially, filters might detect simple features like edges or textures, but as the network learns, it combines these features into more complex representations (like parts of objects or whole objects). This ability to learn relevant features from data is one of the main reasons why CNNs are powerful in tasks like image classification or object detection.<br>

7. Efficient for Images and Time-Series Data
Convolutional layers are particularly well-suited for data with a grid-like structure, such as images or time-series data. The local connectivity of neurons in a convolutional layer allows the network to focus on small patches of data at a time, learning the local patterns before combining them into larger features.<br>

8. Reduction of Overfitting
Since convolutional layers use fewer parameters (due to weight sharing and local receptive fields), they are less prone to overfitting compared to fully connected layers. This is particularly helpful when working with large datasets where the risk of overfitting is higher.<br>