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
// Input (4x4)
[1, 3, 2, 4]
[5, 6, 7, 8]
[9, 10, 11, 12]
[13, 14, 15, 16]

// Output
[6, 8]
[14, 16]
```

## 2. What are three major types of layers in the convolutional neural network?

## 3. What is the architecture of a convolutional network?
