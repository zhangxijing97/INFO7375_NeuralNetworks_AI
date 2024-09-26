# HW to Chapter 4 & 5 “Neural Network with one hidden layer”

# Non-programming Assignment

## 1. What is Hadamard Matrix Product?

The Hadamard matrix product, also known as the element-wise product or Schur product, is an operation that takes two matrices of the same dimensions and produces another matrix where each element \(i, j\) is the product of elements \(i, j\) of the original two matrices. If \(A\) and \(B\) are two matrices of the same dimension, then their Hadamard product \(C\) is defined as:

\[ C_{ij} = A_{ij} \cdot B_{ij} \]

For example, if 

\[ A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \]
\[ B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} \]

then their Hadamard product \(C\) is:

\[ C = \begin{pmatrix} 1 \cdot 5 & 2 \cdot 6 \\ 3 \cdot 7 & 4 \cdot 8 \end{pmatrix} = \begin{pmatrix} 5 & 12 \\ 21 & 32 \end{pmatrix} \]

## 2. Describe Matrix Multiplication

Matrix multiplication is an operation that produces a new matrix from two matrices. Given two matrices \(A\) of dimension \(m \times n\) and \(B\) of dimension \(n \times p\), the product \(C = A \times B\) will be a matrix of dimension \(m \times p\). The element at row \(i\) and column \(j\) of \(C\) is calculated as the dot product of the \(i\)-th row of \(A\) and the \(j\)-th column of \(B\):

\[ C_{ij} = \sum_{k=1}^{n} A_{ik} \cdot B_{kj} \]

For example, if 

\[ A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \]
\[ B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} \]

then their product \(C\) is:

\[ C = \begin{pmatrix} 1 \cdot 5 + 2 \cdot 7 & 1 \cdot 6 + 2 \cdot 8 \\ 3 \cdot 5 + 4 \cdot 7 & 3 \cdot 6 + 4 \cdot 8 \end{pmatrix} = \begin{pmatrix} 19 & 22 \\ 43 & 50 \end{pmatrix} \]

## 3. What is Transpose Matrix and Vector?

The transpose of a matrix is a new matrix whose rows are the columns of the original. Formally, the transpose of matrix \(A\), denoted as \(A^T\), is defined by:

\[ (A^T)_{ij} = A_{ji} \]

For example, if 

\[ A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \]

then the transpose of \(A\) is:

\[ A^T = \begin{pmatrix} 1 & 3 \\ 2 & 4 \end{pmatrix} \]

The concept is similar for vectors. For a column vector \(v\), its transpose \(v^T\) is a row vector, and vice versa.

## 4. Describe the Training Set Batch

In machine learning, especially in training neural networks, a batch is a subset of the training dataset. Instead of processing the entire training set at once, the training data is divided into smaller chunks called batches. Each batch is used to update the model's weights. This approach has several benefits:

1. It makes the training process more efficient by reducing memory requirements.
2. It helps in faster convergence as the weights are updated more frequently.
3. It allows for parallelism in computations.

A training set batch typically contains a fixed number of training examples, known as the batch size.

## 5. Describe the Entropy-Based Loss (Cost or Error) Function and Explain Why It is Used for Training Neural Networks

Entropy-based loss functions are used to measure the difference between the predicted probability distribution and the true distribution. The most common entropy-based loss function is the cross-entropy loss, especially used in classification problems. 

For a single instance, the cross-entropy loss is defined as:

\[ L(y, \hat{y}) = - \sum_{i} y_i \log(\hat{y}_i) \]

where \(y\) is the true distribution (usually a one-hot encoded vector), and \(\hat{y}\) is the predicted distribution (the output of the softmax function in classification problems).

The cross-entropy loss is used because it provides a robust way to penalize incorrect classifications by the model and encourages the model to output high probabilities for the correct classes.

## 6. Describe Neural Network Supervised Training Process

In supervised training of neural networks, the model learns to map input data to output labels using labeled training data. The process involves:

1. **Initialization**: Initialize the network's weights and biases.
2. **Forward Propagation**: Pass input data through the network to get predicted outputs.
3. **Loss Calculation**: Compute the loss (error) between the predicted output and the true output using a loss function.
4. **Backward Propagation**: Compute gradients of the loss with respect to each weight using backpropagation.
5. **Weight Update**: Update the weights using an optimization algorithm (e.g., gradient descent) to minimize the loss.
6. **Iteration**: Repeat steps 2-5 for a number of epochs or until convergence.

## 7. Describe in Detail Forward Propagation and Backpropagation

### Forward Propagation

Forward propagation is the process by which input data is passed through the neural network to obtain the output. It involves the following steps:

1. **Input Layer**: Input features are fed into the network.
2. **Hidden Layers**: Each hidden layer processes the input from the previous layer using weights, biases, and activation functions. The output of each neuron is calculated as:

   \[ z_i = w_i \cdot x + b_i \]
   \[ a_i = \sigma(z_i) \]

   where \(z_i\) is the weighted sum of inputs, \(w_i\) are the weights, \(b_i\) is the bias, \(x\) is the input, and \(\sigma\) is the activation function.
3. **Output Layer**: The final layer produces the output using a suitable activation function (e.g., softmax for classification).

### Backpropagation

Backpropagation is the process of computing the gradient of the loss function with respect to each weight by the chain rule, allowing for efficient computation of gradients.

1. **Loss Gradient**: Compute the gradient of the loss function with respect to the output of the network.
2. **Output Layer Gradient**: Calculate the gradient of the loss with respect to the output layer's weights and biases.
3. **Hidden Layers Gradient**: Propagate the gradient back through the network, layer by layer, using the chain rule to compute gradients for each layer's weights and biases.
4. **Weight Update**: Update the weights and biases using the computed gradients and an optimization algorithm (e.g., stochastic gradient descent).

Backpropagation ensures that each weight is adjusted to minimize the loss, leading to a trained model.

---

This document provides an overview of various concepts related to neural networks and their training processes, offering a foundation for further study and implementation.

