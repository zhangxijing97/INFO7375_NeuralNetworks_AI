# HW to Chapter 4 & 5 “Neural Network with one hidden layer”

# Non-programming Assignment

## What is Hadamard Matrix Product?
The **Hadamard product**, also known as the element-wise product, is an operation on two matrices of the same dimensions. The result is another matrix, where each element is the product of the corresponding elements from the two input matrices. If \( A \) and \( B \) are two matrices of the same size, the Hadamard product is defined as:

`C = A ⊙ B` where `C_ij = A_ij * B_ij`.

## Describe Matrix Multiplication
**Matrix multiplication** involves taking two matrices and producing a new matrix. For matrices \( A \) (of size `m × n`) and \( B \) (of size `n × p`), the resulting matrix \( C \) (of size `m × p`) is calculated as:

`C_ij = ∑(k=1 to n) (A_ik * B_kj)`.

The key requirement for matrix multiplication is that the number of columns in the first matrix must equal the number of rows in the second matrix.

## What is Transpose Matrix and Vector?
The **transpose** of a matrix \( A \), denoted as `A^T`, is formed by flipping the matrix over its diagonal. This means that the row and column indices are swapped. If \( A \) is of size `m × n`, then `A^T` will be of size `n × m`.

For a vector, the transpose converts a row vector into a column vector and vice versa. For example, if `v` is a column vector:

`v = [a, b, c]^T`

Then its transpose `v^T` is a row vector:

`v^T = [a, b, c]`.

## Describe the Training Set Batch
In machine learning, a **training set batch** refers to a subset of the training dataset used to train the model in one iteration of the training process. Rather than using the entire dataset at once, which can be computationally expensive and inefficient, the dataset is often divided into smaller batches. This approach allows for more manageable computations and helps in optimizing the learning process.

The size of the batch (often called the **batch size**) can significantly impact the performance and stability of the training process. Common strategies include:

- **Mini-batch Gradient Descent**: Using small batches (e.g., 32 or 64 samples).
- **Stochastic Gradient Descent**: Using a batch size of 1 (updating weights after each training example).
- **Full-batch Gradient Descent**: Using the entire dataset as a single batch.

## Describe the Entropy-Based Loss (Cost or Error) Function
The **entropy-based loss function**, commonly referred to as **cross-entropy loss**, measures the dissimilarity between the predicted probability distribution and the actual distribution (ground truth). It is especially used in classification tasks. The cross-entropy loss `L` for two probability distributions `p` (true distribution) and `q` (predicted distribution) is defined as:

`L(p, q) = -∑(i=1 to n) (p(i) * log(q(i)))`.

### Why Use It for Training Neural Networks?
Cross-entropy loss is used in training neural networks because it provides a clear gradient for optimization, particularly in the context of logistic regression and softmax outputs. It helps in effectively adjusting the weights during backpropagation, ensuring that the network learns to output probabilities that align closely with the actual labels.

## Describe Neural Network Supervised Training Process
The **supervised training process** for neural networks involves several key steps:

1. **Data Preparation**: The dataset is divided into training and validation sets. The training set is used to train the model, while the validation set is used to evaluate its performance.

2. **Forward Propagation**: Input data is fed into the network, and predictions are made based on current weights and biases. The output is computed as:

   `y = f(w_1 * x_1 + w_2 * x_2 + ... + w_n * x_n + b)`

3. **Loss Calculation**: The loss function (e.g., cross-entropy) computes the difference between predicted and actual outputs.

4. **Backward Propagation**: The gradients of the loss function are calculated concerning the model parameters using the chain rule.

5. **Weight Update**: The model's weights are updated using an optimization algorithm (e.g., Stochastic Gradient Descent) to minimize the loss.

6. **Iteration**: Steps 2-5 are repeated for multiple epochs until the model converges or achieves satisfactory performance.

## Describe in Detail Forward Propagation and Backpropagation
### Forward Propagation
**Forward propagation** is the process of passing input data through the network to obtain output predictions. Each layer of the network computes a weighted sum of its inputs, applies a bias, and then passes the result through an activation function. For a simple feedforward neural network, the forward propagation process can be described as follows:

1. **Input Layer**: The input features are fed into the network.
2. **Hidden Layers**: Each hidden layer computes its output as:

   `z^(l) = W^(l) * a^(l-1) + b^(l)`

   `a^(l) = f(z^(l))`

   Where `z^(l)` is the weighted sum of inputs, `W^(l)` is the weight matrix, `b^(l)` is the bias vector, `a^(l-1)` is the output from the previous layer, and `f` is the activation function (e.g., ReLU, sigmoid).

3. **Output Layer**: The final layer produces predictions using a suitable activation function, such as softmax for classification tasks.

### Backpropagation
**Backpropagation** is the process of computing the gradient of the loss function concerning the weights of the network by applying the chain rule. The steps involved are:

1. **Loss Calculation**: Compute the loss using the output from forward propagation and the true labels.
   
2. **Output Layer Gradients**: Calculate the gradients of the loss with respect to the outputs:

   `δ^(L) = ∇_a L * f'(z^(L))`.

3. **Hidden Layer Gradients**: For each hidden layer `l`:

   `δ^(l) = (W^(l+1))^T * δ^(l+1) * f'(z^(l))`.

4. **Weight Updates**: Update weights and biases for each layer using the calculated gradients:

   `W^(l) = W^(l) - α * ∂L/∂W^(l)`

   `b^(l) = b^(l) - α * ∂L/∂b^(l)`.

Where `α` is the learning rate.

This process is repeated iteratively over multiple epochs until the model converges to an optimal set of weights and biases, minimizing the loss function.
