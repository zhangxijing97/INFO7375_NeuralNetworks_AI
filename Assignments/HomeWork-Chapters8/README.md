# HW to Chapters 6 “Deep Neural Networks” and 7 “Activation Functions”

# Non-programming Assignment

## Why are Multilayer (Deep) Neural Networks Needed?
**Multilayer (deep) neural networks** are necessary because they can learn complex patterns and representations from data. A single-layer network (or shallow network) may not be able to capture the intricate relationships within the data, leading to underfitting. Deep networks, on the other hand, utilize multiple layers of neurons to progressively extract higher-level features from the input. This hierarchical feature learning allows them to perform well on tasks like image recognition, natural language processing, and speech recognition, where the relationships among input features are highly non-linear.

## What is the Structure of Weight Matrix?
The **weight matrix** in a neural network layer has dimensions determined by the number of neurons in the current layer and the number of neurons in the previous layer. Specifically, if the previous layer has \( n \) neurons and the current layer has \( m \) neurons, the weight matrix \( W \) will be of size:

`W ∈ ℝ^(m × n)`

This means it has \( m \) rows (one for each neuron in the current layer) and \( n \) columns (one for each neuron in the previous layer).

## Describe the Gradient Descent Method
The **gradient descent method** is an optimization algorithm used to minimize the loss function in machine learning and neural networks. It iteratively adjusts the model parameters (weights and biases) to reduce the difference between the predicted outputs and the actual targets. The steps involved in gradient descent are as follows:

1. **Initialize Parameters**: Set initial values for the weights and biases (often randomly).
2. **Compute the Loss**: Use the current parameters to calculate the loss using a suitable loss function.
3. **Calculate Gradients**: Compute the gradients of the loss function with respect to each parameter by applying the chain rule.
4. **Update Parameters**: Adjust the parameters in the opposite direction of the gradients to minimize the loss:

   `W_new = W_old - α * ∂L/∂W`

   `b_new = b_old - α * ∂L/∂b`

   Where \( α \) is the learning rate that controls the step size of the updates.
5. **Iterate**: Repeat steps 2-4 until convergence or for a predetermined number of epochs.

## Describe in Detail Forward Propagation and Backpropagation for Deep Neural Networks
### Forward Propagation
In **forward propagation** for deep neural networks, the input data is passed through each layer of the network to obtain the final output. The process is as follows:

1. **Input Layer**: The input features are fed into the network.
2. **Hidden Layers**: Each hidden layer computes its output using:

   `z^(l) = W^(l) * a^(l-1) + b^(l)`

   `a^(l) = f(z^(l))`

   Where `z^(l)` is the weighted sum of inputs, `W^(l)` is the weight matrix, `b^(l)` is the bias vector, and `f` is the activation function applied to the layer's output.
3. **Output Layer**: The final layer produces the output predictions based on the chosen activation function (e.g., softmax for classification).

### Backpropagation
In **backpropagation** for deep neural networks, gradients are computed to update the weights and biases. The process involves:

1. **Loss Calculation**: Compute the loss using the output from forward propagation and the true labels.
2. **Output Layer Gradients**: Calculate gradients for the output layer:

   `δ^(L) = ∇_a L * f'(z^(L))`.

3. **Hidden Layer Gradients**: For each hidden layer \( l \):

   `δ^(l) = (W^(l+1))^T * δ^(l+1) * f'(z^(l))`.

4. **Weight Updates**: Update weights and biases for each layer using the gradients:

   `W^(l) = W^(l) - α * ∂L/∂W^(l)`

   `b^(l) = b^(l) - α * ∂L/∂b^(l)`.

5. **Iterate**: This process is repeated for multiple epochs until the model converges.

## Describe Linear, ReLU, Sigmoid, Tanh, and Softmax Activation Functions
### 1. Linear Activation Function
The **linear activation function** outputs the input directly, defined as:

`f(x) = x`.

**Use**: It is often used in regression tasks where the output is a continuous value.

### 2. ReLU Activation Function
The **Rectified Linear Unit (ReLU)** activation function is defined as:

`f(x) = max(0, x)`.

**Use**: It is widely used in hidden layers of deep networks due to its simplicity and effectiveness in alleviating the vanishing gradient problem.

### 3. Sigmoid Activation Function
The **sigmoid activation function** is defined as:

`f(x) = 1 / (1 + exp(-x))`.

**Use**: It squashes the input to a range between 0 and 1, making it suitable for binary classification tasks. However, it can suffer from the vanishing gradient problem.

### 4. Tanh Activation Function
The **tanh activation function** is defined as:

`f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`.

**Use**: It squashes the input to a range between -1 and 1, making it zero-centered. It is often preferred over sigmoid in hidden layers, though it can still experience vanishing gradients.

### 5. Softmax Activation Function
The **softmax activation function** is defined as:

`f(x_i) = exp(x_i) / ∑(j=1 to K) exp(x_j)`,

where \( K \) is the number of classes.

**Use**: It is typically used in the output layer of multi-class classification problems to produce a probability distribution over multiple classes, allowing for effective interpretation of model outputs.