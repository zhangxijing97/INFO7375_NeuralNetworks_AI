# HW to Chapter 9 “Fitting, Bias, Regularization, and Dropout”

# Non-programming Assignment

### 1. What are underfitting and overfitting?

**Underfitting** occurs when a model is too simple to capture the underlying patterns in the data, leading to poor performance on both the training and validation datasets. It fails to learn effectively from the data.

**Overfitting** happens when a model learns the training data too well, including its noise and outliers. This results in excellent performance on the training set but poor generalization to new, unseen data.

### 2. What may cause an early stopping of the gradient descent optimization process?

Early stopping can be caused by:
- **Convergence**: When the loss function reaches a predefined threshold or shows minimal improvement over several iterations.
- **Validation Performance**: If the model's performance on the validation set starts to degrade, indicating potential overfitting.

### 3. Describe the recognition bias vs variance and their relationship.

**Bias** refers to the error introduced by approximating a real-world problem (which may be complex) using a simplified model. High bias can lead to underfitting.

**Variance** refers to the model's sensitivity to fluctuations in the training data. High variance can lead to overfitting.

The relationship between bias and variance is known as the **bias-variance tradeoff**: reducing bias increases variance and vice versa. The goal is to find a balance that minimizes total error.

### 4. Describe regularization as a method and the reasons for it.

**Regularization** is a technique used to prevent overfitting by adding a penalty term to the loss function. Common methods include L1 (Lasso) and L2 (Ridge) regularization.

**Reasons for Regularization**:
- **Reduces Overfitting**: Helps improve generalization by discouraging overly complex models.
- **Simplifies Models**: Encourages the model to focus on the most important features.

### 5. Describe dropout as a method and the reasons for it.

**Dropout** is a regularization technique where randomly selected neurons are ignored (dropped out) during training. This prevents the model from becoming too reliant on any one neuron.

**Reasons for Dropout**:
- **Prevents Overfitting**: Helps improve generalization by reducing co-adaptation of neurons.
- **Encourages Robust Features**: Forces the network to learn more robust features that are useful in conjunction with various combinations of neurons.