# HW to Chapter 13 “Convolutional Layer”

# Non-programming Assignment

## 1. What is pooling layer and how it works?

A pooling layer in Convolutional Neural Networks (CNNs) reduces the spatial dimensions (height and width) of the input feature map, helping decrease computational load, memory usage, and overfitting. It also makes the model more invariant to small shifts or distortions in the input data.<br>

Types of Pooling:<br>
- Max Pooling: Extracts the maximum value from each region of the feature map. Commonly used to retain prominent features.
- Average Pooling: Computes the average value from each region, providing a smoother feature map.
- Global Pooling: Reduces the entire feature map to a single value per channel (using max or average).

How It Works:<br>
- Input: Feature map from a convolution layer.
- Sliding Window: A small window (e.g., 2x2 or 3x3) slides over the feature map, and the pooling operation is applied (max or average).
- Output: A smaller feature map with reduced dimensions.

Advantages:<br>
- Reduces Overfitting: By decreasing the size, pooling prevents overfitting.
- Improves Efficiency: Less computation and fewer parameters.
- Translation Invariance: Helps the model recognize features despite small shifts.

Key Hyperparameters:<br>
- Kernel Size: Size of the pooling window (e.g., 2x2).
- Stride: Step size for moving the window.

Example (Max Pooling 2x2):<br>

```
# Input (4x4):
[1, 3, 2, 4]
[5, 6, 7, 8]
[9, 10, 11, 12]
[13, 14, 15, 16]

# Output:
[6, 8]
[14, 16]
```

## 2. What are three major types of layers in the convolutional neural network?

1. Convolutional Layer: This layer applies convolution operations to the input, using filters (or kernels) to extract features such as edges, textures, and shapes. Each filter scans the input image and produces feature maps, which are used for further processing.

2. Pooling Layer: Pooling layers reduce the spatial dimensions (width and height) of the feature maps, typically using operations like max pooling or average pooling. This helps to reduce computational complexity and overfitting, while retaining essential features.

3. Fully Connected (FC) Layer: In this layer, all neurons are connected to every neuron in the previous layer. It is typically used toward the end of the network to make predictions or classifications based on the features extracted by the convolutional and pooling layers.

## 3. What is the architecture of a convolutional network?

The architecture of a Convolutional Neural Network (CNN) is typically organized in a sequence of layers that work together to automatically extract features from input data (such as images) and make predictions. The general architecture of a CNN includes the following key components:

1. **Input Layer**:
   - The input layer takes the raw data, usually in the form of an image or a 2D matrix of pixel values (e.g., RGB values for color images).
   
2. **Convolutional Layers**:
   - The core building block of a CNN, where a filter (or kernel) is applied to the input image. This filter slides over the image to create feature maps that highlight specific features (e.g., edges, textures).
   - The convolution operation is repeated multiple times using different filters to detect various features in the input data.
   
3. **Activation Function (ReLU)**:
   - After convolution, the feature maps typically go through an activation function, commonly the Rectified Linear Unit (ReLU). This introduces non-linearity, enabling the network to learn complex patterns.

4. **Pooling Layers**:
   - Pooling layers (such as Max Pooling or Average Pooling) downsample the feature maps by reducing their spatial dimensions, which helps reduce computational complexity, prevents overfitting, and retains only the most important information.

5. **Normalization Layers** (Optional):
   - Some CNN architectures include normalization layers like Batch Normalization to stabilize training by normalizing the output of the previous layer, improving convergence speed, and potentially leading to better performance.

6. **Fully Connected (FC) Layers**:
   - After several convolutional and pooling layers, the network flattens the feature maps into a 1D vector and passes them through fully connected layers. These layers make final decisions based on the features extracted from the image.
   - FC layers are where the network learns high-level representations and makes predictions (such as classification or regression).

7. **Output Layer**:
   - The output layer is where the final prediction is made. For classification tasks, this often uses a softmax activation function to produce probability distributions over classes. For regression tasks, it may use a linear activation function.

8. **Softmax Activation** (for classification):
   - For multi-class classification tasks, a softmax activation function is typically applied to the output layer, turning raw output values (logits) into probabilities that sum to 1.

---

## Example of a CNN Architecture:
1. **Input Layer**: Image of size 224x224x3 (height, width, channels).
2. **Convolutional Layer 1**: 32 filters of size 3x3, stride 1.
3. **ReLU Activation**.
4. **Max Pooling Layer**: Pool size 2x2.
5. **Convolutional Layer 2**: 64 filters of size 3x3, stride 1.
6. **ReLU Activation**.
7. **Max Pooling Layer**: Pool size 2x2.
8. **Fully Connected Layer**: 128 neurons.
9. **Softmax Output Layer**: For classification with n classes.

This structure can vary depending on the complexity of the task and the size of the dataset, but this is a basic framework that forms the foundation of most convolutional neural networks.