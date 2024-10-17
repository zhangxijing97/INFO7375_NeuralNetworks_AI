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

A single-layer network (or shallow network) may not be able to capture the intricate relationships within the data, leading to underfitting. But multilayer (deep) neural networks can learn complex non-linear patterns and representations from data.<br>

Example:<br>

| Layer            | Number of Neurons | Weights per Neuron | Total Weights | Bias per Neuron | Total Biases |
|------------------|-------------------|--------------------|---------------|-----------------|--------------|
| Input Layer      | 50                | 0                  | 0             | 0               | 0            |
| Second Layer     | 100               | 50                 | 5000          | 1               | 100          |
| Third Layer      | 64                | 100                | 6400          | 1               | 64           |
| Output Layer     | 10                | 64                 | 640           | 1               | 10           |

## 4. (max points 4). Describe major activation functions: linear, ReLU, sigmoid, tanh, and softmax, and explain their usage.

1. Linear Activation Function
The linear activation function outputs the input directly, defined as:<br>

`f(x) = x`.

Use: It is often used in regression tasks where the output is a continuous value.<br>

2. ReLU Activation Function
The Rectified Linear Unit (ReLU) activation function is defined as:<br>

`f(x) = max(0, x)`.

Use: It is widely used in hidden layers of deep networks due to its simplicity and effectiveness in alleviating the vanishing gradient problem.<br>

3. Sigmoid Activation Function<
The sigmoid activation function is defined as:<br>

`f(x) = 1 / (1 + exp(-x))`.

Use: It squashes the input to a range between 0 and 1, making it suitable for binary classification tasks. However, it can suffer from the vanishing gradient problem.<br>

4. Tanh Activation Function
The tanh activation function is defined as:<br>

`f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`.

Use: It squashes the input to a range between -1 and 1, making it zero-centered. It is often preferred over sigmoid in hidden layers, though it can still experience vanishing gradients.<br>

5. Softmax Activation Function
The softmax activation function is defined as:<br>

`f(x_i) = exp(x_i) / ∑(j=1 to K) exp(x_j)`,

where K is the number of classes.<br>

Use: It is typically used in the output layer of multi-class classification problems to produce a probability distribution over multiple classes, allowing for effective interpretation of model outputs.<br>

## 5. (max points 4). What is supervised learning?

## 6. (max points 4). Describe loss/cost function.



## 7. (max points 4). Describe forward and backward propagation for a multilayer (deep) neural network.

## 8. (max points 4). What are parameters and hyperparameters in neural networks and what is the conceptual difference between them.

## 9. (max points 4). How to set the initial values for the neural network training

## 10. (max points 4). Why are mini-batches used instead of complete batches in training of neural networks.