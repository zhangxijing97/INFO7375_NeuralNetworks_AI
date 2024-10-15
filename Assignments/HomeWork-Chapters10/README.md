# HW to Chapter 10 “Normalization and Optimization Methods”

# Non-programming Assignment

## 1. What is normalization and why is it needed?

Normalization transforms the features of the dataset to a common scale, like [0, 1] or [-1, 1]. This process ensures that the different features contribute equally to improves the performance of machine learning algorithms.

### Common Normalization Techniques:

Original Data:<br>
 [[1 2 3]<br>
 [4 5 6]<br>
 [7 8 9]]<br>

Column 1 (Feature 1):<br>
Values: 1, 4, 7<br>
Column 2 (Feature 2):<br>
Values: 2, 5, 8<br>
Column 3 (Feature 3):<br>
Values: 3, 6, 9<br>

- **Min-Max Scaled Data**<br>

Min-Max Scaling X' = (X - Xmin)/(Xmax - Xmin)<br>

 [[0.  0.  0. ]<br>
 [0.5 0.5 0.5]<br>
 [1.  1.  1. ]]<br>

- **Z-Score Normalization**<br>

Z-Score Scaling X' = (X - μ)/σ<br>
μ is the mean of the feature<br>
σ is the standard deviation of the feature<br>
μ1 = 4<br>
μ2 = 5<br>
μ3 = 6<br>
σ1^2 = Σ(xi - μ)^2 / N<br>
= (1 - 4)^2 + (4 - 4)^2 + (7 - 5)^2 / 1<br>
= 6<br>
σ1 = 2.45<br>
σ2 = 2.45<br>
σ3 = 2.45<br>
X11' = (1-4)/2.45 = -1.22474487<br>

 [[-1.22474487 -1.22474487 -1.22474487]<br>
 [ 0.          0.          0.        ]<br>
 [ 1.22474487  1.22474487  1.22474487]]<br>

- **Robust Scaling**<br>

Min-Max Scaling X' = (X - median(X))/IQR(X)<br>
Interquartile Range (IQR): The range between the 25th percentile (Q1) and the 75th percentile (Q3).<br>
IQR(X1) = 5.5 - 2.5 = 3.0
IQR(X2) = 6.5 - 3.5 = 3.0
IQR(X3) = 7.5 - 4.5 = 3.0

X11' = (1-4)/3 = -1<br>

 [[-1. -1. -1.]<br>
 [ 0.  0.  0.]<br>
 [ 1.  1.  1.]]<br>

## 2. What are vanishing and exploding gradients?

### Vanishing Gradients

Vanishing gradients occur when the gradients (partial derivatives of the loss function with respect to the weights) become very small during backpropagation. This can cause the following issues:<br>

- **Slow Learning**: The weights update very slowly because the gradient is too small.<br>
- **Poor Performance**: The network might not learn properly.<br>

**Why Does It Happen?**<br>
This problem is common in deep networks with activation functions like the sigmoid or tanh, which squash their input into a small output range (0 to 1 for sigmoid and -1 to 1 for tanh).<br>

When gradients are backpropagated through many layers, they can be repeatedly multiplied by small numbers, causing them to shrink exponentially.<br>

**Solutions**<br>

**Activation Functions**: Using activation functions that do not squash the gradient as much, such as ReLU.<br>

**Weight Initialization**: Using better weight initialization methods like He initialization or Xavier initialization to maintain the scale of gradients.<br>

**Batch Normalization**: Normalizing the inputs of each layer to maintain a stable gradient flow.<br>

### Exploding Gradients

Exploding gradients occur when the gradients grow exponentially during backpropagation. This can cause the following issues:<br>

- **Unstable Training**: The weights can change dramatically, causing the training process to become unstable.<br>
- **Overflow**: The weights can become so large that they result in numerical overflow, causing the model to fail.<br>

**Why Does It Happen?**
Exploding gradients can occur in deep networks when the gradients are repeatedly multiplied by large numbers during backpropagation, leading to an exponential increase.<br>

This is often seen with certain weight initialization methods or when using activation functions that don't constrain their output.<br>

**Solutions**<br>

**Gradient Clipping**: Clipping the gradients during backpropagation to a maximum value to prevent them from growing too large.<br>

**Weight Regularization**: Adding regularization terms to the loss function to penalize large weights.<br>

## 3. What Adam algorithm and why is it needed?

## 4. How to choose hyperparameters?