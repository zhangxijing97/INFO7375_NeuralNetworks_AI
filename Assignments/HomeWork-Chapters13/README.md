# HW to Chapter 13 “Convolutional Layer”

# Non-programming Assignment

## 1. What is convolution operation and how does it work?

The convolution operation is a mathematical process used primarily in convolutional neural networks (CNNs) to extract features from input data, such as images or time series. It is a fundamental operation that allows the network to learn spatial hierarchies of features.<br>

1. How Convolution Works:
Input Image: The input is typically a multi-dimensional matrix (e.g., an image with height, width, and depth). For a grayscale image, the depth is 1, and for a color image (RGB), the depth is 3 (one channel per color).

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

If the output values from the convolution are very close to each other, it generally means that the image does not have strong vertical edges in the region being processed by the filter.<br>

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

## 3. How are sizes of the original image, the filter, and the resultant convoluted image are relted?

## 4. What is padding and why is it needed?

## 5. What is strided convolution and why is it needed?