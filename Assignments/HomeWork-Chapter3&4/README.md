# HW to Chapter 3 & 4 “The Perceptron for Logistic Regression“

# Non-programming Assignment

## 1. Describe the logistic regression
**Logistic Regression** is a type of regression used for binary classification problems. It predicts the probability of a binary outcome by applying the **sigmoid activation function** to the weighted sum of the input features. The output is a probability between 0 and 1, which is then used to classify the data into two categories (0 or 1).

## 2. How are grayscale and color (RGB) images presented as inputs for the perceptron?
- **Grayscale images**: Grayscale images are represented as 2D arrays where each pixel has a value between 0 and 255, representing the intensity of light. For the Perceptron, these values are typically flattened into a 1D array, with each pixel treated as a feature.
- **Color (RGB) images**: RGB images have three channels (Red, Green, Blue), each of which is a 2D array. The pixel values from all three channels are flattened into a single 1D array and combined as input features for the perceptron.

## 3. Is image recognition a logistic regression problem? Why?
**No**, image recognition is generally not a logistic regression problem. Logistic regression is designed for **binary classification**, while image recognition often involves **multi-class classification** (e.g., recognizing digits 0-9 in MNIST). Logistic regression can be extended to multi-class classification through techniques like **Softmax regression**, but for more complex problems, neural networks are more effective.

## 4. Is home prices prediction a logistic regression problem? Why?
**No**, home prices prediction is not a logistic regression problem. It is a **regression problem**, where the goal is to predict a continuous value (home prices). Logistic regression is used for **binary classification**, not predicting continuous variables.

## 5. Is image diagnostics a logistic regression problem? Why?
**It depends**. If image diagnostics involves distinguishing between two categories (e.g., whether an image contains a tumor or not), then it could be framed as a **logistic regression** problem. However, more complex diagnostic tasks often require multi-class classification or regression models, which go beyond the scope of logistic regression.

## 6. How does gradient descent optimization work?
**Gradient Descent** is an optimization algorithm used to minimize the loss function in a model. It works by calculating the gradient (slope) of the loss function with respect to the model’s parameters (weights and biases). The weights are then updated by moving in the direction that reduces the loss, proportional to the **learning rate**. This process continues iteratively until the model converges to the minimum loss.

## 7. How does image recognition work as a logistic regression classifier?
In **logistic regression**, image recognition can work by flattening the image into a 1D vector of pixel values (features) and then applying a weighted sum followed by a **sigmoid activation function** to predict a binary outcome (e.g., whether an image contains a specific object or not). However, logistic regression is limited to binary classification tasks and struggles with more complex multi-class image recognition.

## 8. Describe the logistic regression loss function and explain the reasons behind this choice.
The **logistic regression loss function** is the **binary cross-entropy** (also called log-loss). It measures the difference between the predicted probabilities and the actual binary labels (0 or 1). The formula is:

`L = - [ y * log(ŷ) + (1 - y) * log(1 - ŷ) ]`

Where:
- `y` is the actual label (0 or 1),
- `ŷ` is the predicted probability.

This loss function is chosen because it is well-suited for binary classification tasks, and it penalizes predictions that are confident but wrong (i.e., assigning a high probability to the wrong class).

## 9. Describe the sigmoid activation function and the reasons behind its choice.
The **sigmoid activation function** is used in logistic regression to map the weighted sum of inputs to a probability between 0 and 1. The sigmoid function is defined as:

`σ(z) = 1 / (1 + e^(-z))`

The reasons behind its choice include:
- It converts the linear output into a probability, making it suitable for binary classification.
- It provides a smooth gradient for use in **gradient descent**, allowing the model to learn efficiently.
