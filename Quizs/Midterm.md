# Midterm Exam

## 1. (max points 4). Describe the artificial neuron model.

The artificial neuron model is a kind of model designed to simulate the functioning of a biological neuron in a simplified way.<br>

Input: The model receives several inputs from other neurons, representing signals from other neurons.<br>

Weights: Each input is associated with a weight, which determines the influence of that input on the output.<br>

Weighted Sum: Calculate the weighted sum z = w * x + b for each neuron.<br>

Activation Function: Apply an activation function f(z) to get the output y. We can use different kinds of activation functions to decide different kinds of output.<br>

## 2. (max points 4). What is the logistic regression problem?

Logistic Regression is a type of regression used for binary classification problems. It predicts the probability of a binary outcome by applying the sigmoid activation function to the weighted sum of the input features. The output is a probability between 0 and 1, which is then used to classify the data into two categories (0 or 1).<br>

## 3. (max points 4). Describe multilayer (deep) neural network.

A single-layer network (or shallow network) may not be able to capture the intricate relationships within the data, leading to underfitting. However multilayer (deep) neural networks can learn complex non-linear patterns and representations from data. And just like a single-layer network, we can use Forward Propagation and Backpropagation to train the model<br>

Forward propagation:<br>

1. **Input Layer**: The input features are fed into the network.
2. **Hidden Layers**: Each hidden layer computes its output using:

   `z^(l) = W^(l) * a^(l-1) + b^(l)`

   `a^(l) = f(z^(l))`

   Where `z^(l)` is the weighted sum of inputs, `W^(l)` is the weight matrix, `b^(l)` is the bias vector, and `f` is the activation function applied to the layer's output.
3. **Output Layer**: The final layer produces the output predictions based on the chosen activation function (e.g., softmax for classification).

Backpropagation<br>
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

**Formula**: 
`w = w - α * ∂J/∂w`

Multilayer (deep) neural network layers example by table:<br>

| Layer            | Number of Neurons | Weights per Neuron | Total Weights | Bias per Neuron | Total Biases |
|------------------|-------------------|--------------------|---------------|-----------------|--------------|
| Input Layer      | 50                | 0                  | 0             | 0               | 0            |
| Second Layer     | 100               | 50                 | 5000          | 1               | 100          |
| Third Layer      | 64                | 100                | 6400          | 1               | 64           |
| Output Layer     | 10                | 64                 | 640           | 1               | 10           |

## 4. (max points 4). Describe major activation functions: linear, ReLU, sigmoid, tanh, and softmax, and explain their usage.

1. Linear Activation Function
The linear activation function outputs the input directly, defined as:<br>

`f(x) = x`

Use: It is often used in regression tasks where the output is a continuous value.<br>

2. ReLU Activation Function
The Rectified Linear Unit (ReLU) activation function is defined as:<br>

`f(x) = max(0, x)`

Use: It is widely used in hidden layers of deep networks due to its simplicity and effectiveness in alleviating the vanishing gradient problem.<br>

3. Sigmoid Activation Function<
The sigmoid activation function is defined as:<br>

`f(x) = 1 / (1 + exp(-x))`

Use: It squashes the input to a range between 0 and 1, making it suitable for binary classification tasks. However, it can suffer from the vanishing gradient problem.<br>

4. Tanh Activation Function
The tanh activation function is defined as:<br>

`f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`

Use: It squashes the input to a range between -1 and 1, making it zero-centered. It is often preferred over sigmoid in hidden layers, though it can still experience vanishing gradients.<br>

5. Softmax Activation Function
The softmax activation function is defined as:<br>

`f(x_i) = exp(x_i) / ∑(j=1 to K) exp(x_j)`

where K is the number of classes.<br>

Use: It is typically used in the output layer of multi-class classification problems to produce a probability distribution over multiple classes, allowing for effective interpretation of model outputs.<br>

## 5. (max points 4). What is supervised learning?

Supervised Learning<br>
Common Approaches:<br>
- Classification (e.g., image recognition, spam detection).
- Regression (e.g., predicting house prices).

Unsupervised Learning<br>
Common Approaches:<br>
- Clustering (e.g., K-Means for customer segmentation).
- Dimensionality Reduction (e.g., PCA to reduce feature space).

Key Difference:<br>
- Supervised Learning uses labeled data to predict outputs.
- Unsupervised Learning discovers patterns in unlabeled data without predefined outputs.

## 6. (max points 4). Describe loss/cost function.

The cost function (also called the loss function) measures the error between the predicted output of the neural network and the actual target values. It helps determine how well the model is performing. Common cost functions include:<br>

Mean Squared Error (MSE): Typically used for regression tasks.<br>
Cross-Entropy Loss: Commonly used for classification tasks.<br>

The cross-entropy loss `L` for two probability distributions `p` (true distribution) and `q` (predicted distribution) is defined as:<br>
`L(p, q) = -∑(i=1 to n) (p(i) * log(q(i)))`

## 7. (max points 4). Describe forward and backward propagation for a multilayer (deep) neural network.

Forward propagation:<br>

1. **Input Layer**: The input features are fed into the network.
2. **Hidden Layers**: Each hidden layer computes its output using:

   `z^(l) = W^(l) * a^(l-1) + b^(l)`

   `a^(l) = f(z^(l))`

   Where `z^(l)` is the weighted sum of inputs, `W^(l)` is the weight matrix, `b^(l)` is the bias vector, and `f` is the activation function applied to the layer's output.
3. **Output Layer**: The final layer produces the output predictions based on the chosen activation function (e.g., softmax for classification).

Backpropagation<br>
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

## 8. (max points 4). What are parameters and hyperparameters in neural networks and what is the conceptual difference between them.

### Parameters
Definition: Parameters are the internal variables of the model that are learned from the training data. They are the weights and biases in a neural network.<br>


Examples: In a simple feedforward neural network, the parameters include the weights W and biases b associated with each neuron.<br>

### Hyperparameters
Definition: Hyperparameters are the external configurations set before the training process begins. They govern the training process and the structure of the neural network.<br>


Examples: learning rate, the number of hidden layers, the number of neurons in each layer, the batch size, the number of epochs, the activation functions, and regularization techniques (like dropout rate).<br>

### Key difference

Parameters are the internal values (weights and biases) learned by the neural network from the training data. They directly influence the network's output and are updated during training.<br>

Hyperparameters are external settings that configure the training process and network architecture. They are set before training and influence how the parameters are learned.<br>

## 9. (max points 4). How to set the initial values for the neural network training

Zero Initialization leads to symmetry; all neurons will learn the same features and thus fail to break symmetry during training.<br>

Random Initialization helps break symmetry, it can lead to exploding or vanishing gradients, especially in deep networks.<br>

Xavier (Glorot) Initialization helps maintain the variance of the outputs of each layer, which mitigates the vanishing gradient problem in networks using activation functions like sigmoid or tanh.<br>

For a layer with n int input units and n out output units, Xavier initialization sets the weights W of the layer using the Uniform Distribution and Normal Distribution.<br>

## 10. (max points 4). Why are mini-batches used instead of complete batches in training of neural networks.

A small, random subset of the training dataset used in each iteration of model training.<br>

- Training Efficiency:

Balances between Stochastic Gradient Descent (SGD) and Batch Gradient Descent.
Updates weights more frequently than Batch Gradient Descent.<br>

- Generalization:

Introduces noise, helping the model avoid overfitting and generalize better.<br>