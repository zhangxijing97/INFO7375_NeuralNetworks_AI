# HW to Chapter 13 “Convolutional Layer”

# Non-programming Assignment

## 1. What is convolution operation and how does it work?

The convolution operation is a mathematical process used primarily in convolutional neural networks (CNNs) to extract features from input data, such as images or time series. It is a fundamental operation that allows the network to learn spatial hierarchies of features.<br>

How Convolution Works:<br>
1. Input Image: The input is typically a multi-dimensional matrix (e.g., an image with height, width, and depth). For a grayscale image, the depth is 1, and for a color image (RGB), the depth is 3 (one channel per color).

2. Filter (Kernel): A filter (or kernel) is a smaller matrix that slides over the input image. It is also multi-dimensional, usually smaller than the input image. For example, a common filter size is 3x3 or 5x5. The filter is responsible for detecting specific features, such as edges, textures, or patterns in the image.

3. Sliding the Filter: The filter slides across the input image, typically with a certain stride (step size). The filter is applied to different regions of the image. For each position:

- The filter's values are multiplied element-wise with the corresponding values in the image (a dot product).
- The result is summed to produce a single value, which represents the feature extracted from that specific region.

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

1. Feature Extraction<br>
Convolutional layers help in automatically detecting features from the raw input, such as edges, textures, shapes, and more complex patterns. In an image, low-level features (like edges and corners) can be learned by applying small filters (kernels) to the image. As the network deepens, these features combine to form high-level features (like faces, objects, or other complex patterns).<br>

- First layers may learn simple patterns like edges, corners, or textures.
- Deeper layers combine these patterns to recognize more complex objects or structures.

2. Parameter Sharing<br>
The same filter (kernel) is applied across different regions of the input image. This is known as parameter sharing. The same filter is reused across the entire image, meaning fewer parameters need to be learned compared to fully connected layers where each pixel is treated independently.

- This reduces the number of parameters significantly, making the network more efficient and reducing the risk of overfitting.

3. Translation Invariance<br>
The network can recognize features regardless of where they appear in the image.<br>

For example, an edge in the top-left corner of an image can be detected by the same filter as an edge in the bottom-right corner.<br>

4. Reduction in Computational Complexity<br>
Since convolutional layers use small filters and share weights across the image, they significantly reduce the computational complexity compared to fully connected layers. Instead of treating each pixel as a separate entity with its own set of weights, convolutions share weights and reduce the total number of calculations, making the model more efficient.<br>

This makes convolutional layers ideal for high-dimensional data like images, where fully connected layers would be too computationally expensive.<br>

5. Preserving Spatial Hierarchy<br>
Unlike fully connected layers, which flatten the image into a one-dimensional vector, convolutional layers preserve the spatial hierarchy of the data. This means that the relative position of features within the image is maintained.<br>

In an image, nearby pixels are likely to be related in some way (e.g., in terms of color, texture, etc.), and convolutional layers respect this relationship, ensuring that the spatial arrangement is captured.<br>

6. Learning Filters<br>
Instead of manually engineering features, convolutional layers automatically learn the most important features during training. Initially, filters might detect simple features like edges or textures, but as the network learns, it combines these features into more complex representations (like parts of objects or whole objects). This ability to learn relevant features from data is one of the main reasons why CNNs are powerful in tasks like image classification or object detection.<br>

7. Efficient for Images and Time-Series Data<br>
Convolutional layers are particularly well-suited for data with a grid-like structure, such as images or time-series data. The local connectivity of neurons in a convolutional layer allows the network to focus on small patches of data at a time, learning the local patterns before combining them into larger features.<br>

8. Reduction of Overfitting<br>
Since convolutional layers use fewer parameters (due to weight sharing and local receptive fields), they are less prone to overfitting compared to fully connected layers. This is particularly helpful when working with large datasets where the risk of overfitting is higher.<br>

## 3. Why do we need convolutional layers in neural networks?

The sizes of the riginal image, the filter (or kernel), and the resultant convoluted image are closely related and depend on several factors: image size, filter size, stride, and padding. Here's how they are related:<br>

### Key Variables:
- **Input image size**: The dimensions of the original image are typically given by height `H` and width `W`.
- **Filter (kernel) size**: The filter is typically a smaller matrix, usually represented by the height `F` and width `F`, assuming a square filter (e.g., 3x3, 5x5).
- **Stride (S)**: The stride is the number of pixels the filter moves across the image at each step. A stride of 1 means the filter moves one pixel at a time.
- **Padding (P)**: Padding is the addition of extra pixels (usually zeros) around the borders of the image to control the output size. Padding can help maintain the spatial dimensions of the input after convolution, or to reduce the output size.

For a square input image of size `H x W` and a square filter of size `F x F`, the output size `H_out` (height) and `W_out` (width) are given by:<br>

`H_out = (H - F + 2P) / S + 1`<br>
`W_out = (W - F + 2P) / S + 1`<br>

Where:
- `H` and `W` are the height and width of the input image.
- `F` is the size of the filter (usually square, so `F x F`).
- `P` is the padding added to the input image.
- `S` is the stride (step size for the filter).

### Example 1: No Padding, Stride = 1
- **Input image size**: `H = 5, W = 5`
- **Filter size**: `F = 3` (3x3 filter)
- **Padding**: `P = 0` (no padding)
- **Stride**: `S = 1`

`H_out = (5 - 3 + 2(0)) / 1 + 1 = (2) / 1 + 1 = 3`<br>
`W_out = (5 - 3 + 2(0)) / 1 + 1 = (2) / 1 + 1 = 3`<br>

So, the output feature map will have a size of `3 x 3`.<br>

### Example 2: Padding = 1, Stride = 1
- **Input image size**: `H = 5, W = 5`
- **Filter size**: `F = 3` (3x3 filter)
- **Padding**: `P = 1`
- **Stride**: `S = 1`

`H_out = (5 - 3 + 2(1)) / 1 + 1 = (4) / 1 + 1 = 5`<br>
`W_out = (5 - 3 + 2(1)) / 1 + 1 = (4) / 1 + 1 = 5`<br>

So, the output feature map will have a size of `5 x 5`, the same as the input size due to the padding.<br>

## 4. What is padding and why is it needed?

Padding refers to adding extra pixels (usually zeros) around the input image before applying the convolution filter<br>

### Why Padding is Needed:
1. **Control Output Size**: Padding helps maintain the spatial dimensions of the input image or controls the reduction in size, ensuring the output feature map is not too small.
2. **Process Edge Information**: Without padding, the filter can't fully cover the edges of the image, leaving edge information unprocessed.
3. **Uniform Filter Application**: Padding allows the filter to be applied evenly across the entire image, including edges.
4. **Enable Larger Strides**: With padding, larger strides can be used without missing important regions of the image.

### Types of Padding:
- **Valid Padding (No Padding)**: No extra pixels added, resulting in a smaller output size.
- **Same Padding**: Padding is added so the output has the same size as the input.

### Example:
- **Input**: 5x5 image, 3x3 filter, Padding = 1
- **Output Size** (with padding): 5x5 (same as input)

## 5. What is strided convolution and why is it needed?

Strided Convolution refers to applying a convolution filter with a stride greater than 1, meaning the filter moves by more than one pixel at a time across the input image. Instead of sliding the filter pixel by pixel (stride = 1), a larger stride means the filter skips some pixels, effectively downsampling the image as it moves.<br>

1. Reduce Spatial Dimensions: Strided convolution helps reduce the size of the output feature map. A larger stride (e.g., 2 or 3) results in a smaller output, which reduces the computational load and memory requirements.

2. Downsampling: Strided convolution acts as a form of downsampling, summarizing the input information into a smaller representation, which helps focus on more important features and improves efficiency.

3. Improve Computation Efficiency: By skipping pixels, the filter does fewer computations, which speeds up the process, especially for large images or when using deeper networks.

4. Capture Larger Context: With larger strides, the convolutional layer captures a broader context from the image, allowing the network to process and summarize information over larger areas.