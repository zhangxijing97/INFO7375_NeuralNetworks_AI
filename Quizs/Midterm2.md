# Midterm Exam

## 1. (max points 4). Describe the McCulloch & Pitts neuron model.

The McCulloch and Pitts model is one of the earliest mathematical models of a neuron, it is a binary model of an artificial neuron and operates as follows:

1. Input: The model receives several binary inputs (either 0 or 1), representing signals from other neurons.
2. Weights: Each input is associated with a weight, which determines the influence of that input on the output.
3. Summation: The weighted inputs are summed up to produce a total input value.
4. Threshold: The neuron fires if the total input exceeds a certain threshold value, The output is 1. If the input is below the threshold, the output is 0.

Mathematically, the McCulloch and Pitts neuron is represented as:
- Output = 1 if (Σ(weight × input) ≥ threshold)
- Output = 0 otherwise.

Usage:<br>
Introducing key neural network principles and logical circuit design. Its simplicity laid the groundwork for more advanced models that incorporate learning capabilities and non-linear functions.<br>

Advantages:<br>
Simplicity: The model is easy to understand and implement due to its binary structure.
Logical Computation: M&P neurons can represent basic logical operations (AND, OR, NOT), allowing for simple computational tasks.<br>

Disadvantages:<br>
Biological Inaccuracy: The model oversimplifies biological neurons, omitting important characteristics like continuous output, adaptability, and synaptic plasticity.<br>
Limited Functionality: It cannot model complex patterns like XOR or real-world data effectively due to its simplicity.<br>

Limitations:<br>
Lack of Learning: The M&P model cannot learn, weights and thresholds are fixed.<br>
Binary Output: Produces only binary outputs, limiting its application in real-world scenarios requiring more nuanced responses.<br>
Linear Separability: Only linearly separable functions can be represented, so it cannot solve problems like XOR.<br>

## 2. (max points 4). What is the logistic regression problem?

Logistic Regression is a type of regression used for binary classification problems. It predicts the probability of a binary outcome by applying the sigmoid activation function to the weighted sum of the input features.<br>

The output is a probability between 0 and 1, which is then used to classify the data into two categories (0 or 1).<br>

Usage:<br>
Logistic Regression is commonly used for binary classification tasks like spam detection, disease diagnosis, and customer churn prediction.<br>

Advantages:<br>
Interpretability: Provides clear insights into how features influence outcomes.<br>
Efficiency: Simple, fast, and effective for smaller datasets.<br>
Probability Outputs: Predicts probabilities, aiding in decision-making.<br>

Disadvantages:<br>
Linear Boundaries: Assumes a linear relationship, limiting complex data fit.<br>
Outlier Sensitivity: Prone to distortion from outliers.<br>
Binary Focus: Primarily for binary classification, needing adjustments for multi-class tasks.<br>

## 3. (max points 4). What is the reason for adding hidden layers to a neural network?

Before we talk about DNN, we learn Perceptron, and Perceptron computes a linear combination of inputs:<br>

`z = w_1 * x_1 + w_2 * x_2 + ... + w_n * x_n + b`<br>

This creates a linear decision boundary in the feature space, meaning it can only separate linearly separable data. For example, it can classify AND and OR, but fails with XOR, which is non-linearly separable. So we wanna improve the Perceptron:<br>

To solve non-linear problems like XOR, we can:<br>
1. Add hidden layers.
2. Use non-linear activation functions, for example ReLU or Sigmoid.

In a multi-layer perceptron (MLP), each layer computes: `z = W * x + b`, The non-linearity enables solving complex, non-linear problems.<br>

## 4. (max points 4). Describe major activation functions: step function, linear, ReLU, sigmoid, tanh, and softmax, and explain their usage.

1. Step Function
`f(x) = 1 if x >= 0`
`f(x) = 0 if x < 0`

Usage: Outputs 0 or 1 depending on whether input is below or above a threshold. Used in early neural networks but less common now due to its inability to handle non-linear problems and lack of gradient for backpropagation.<br>

2. Linear Activation Functions
`f(x) = x`

Usage: Outputs a scaled value of the input, usefull in simple tasks. It’s rarely used in hidden layers since it doesn’t capture non-linear relationships.<br>

3. ReLU Activation Function
`f(x) = max(0, x)`

Usage: Outputs 0 for negative inputs and the input itself for positive values. Widely used in hidden layers due to its simplicity and efficiency in handling non-linear problems, but it can lead to units stuck at zero.<br>

4. Sigmoid Activation Function
`f(x) = 1 / (1 + exp(-x))`

Usage: Squashes inputs to range (0, 1), making it useful for binary classification in the final layer. However, it has the vanishing gradient problem, which can slow down learning.<br>

5. Tanh Activation Function
`f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`

Usage: Similar to Sigmoid Activation Function. It squashes the input to a range between -1 and 1, making it zero-centered. It is often preferred over sigmoid in hidden layers, though it can still experience vanishing gradients.<br>

6. Softmax Activation Function
`f(x_i) = exp(x_i) / ∑(j=1 to K) exp(x_j)`

where K is the number of classes.<br>

Usage: Converts outputs into probabilities that sum to 1, making it ideal for multi-class classification. Usually applied in the output layer of multi-class classification networks.<br>

## 5. (max points 4). What is the difference between batch and mini-batch training? What are the cons and pros for using batch or mini-batch training?

Batch Training<br>
The entire dataset is passed through the model at once for each iteration to get the gradient and before updating the model's parameters.<br>

Pros:<br>
Stable convergence: Uses the full dataset, leading to a more accurate estimate of the gradient and typically more stable convergence.<br>
Efficient on small datasets: Works well when the dataset is small enough to fit into memory.<br>

Cons:<br>
High memory consumption: Not feasible for large datasets as it can exhaust memory.<br>
Slow convergence: Processing the whole dataset per update can slow down training.<br>

Mini-Batch Training<br>
The dataset is divided into smaller subsets (mini-batches), and model parameters are updated after each mini-batch.<br>

Pros:<br>
Efficient on big datasets: much faster when the datasets is big, and it allows training on larger datasets that don’t fit in memory at once.<br>
Faster convergence: Updates parameters more frequently than batch training, leading to faster and more gradual convergence.<br>
Balanced variance: Maintains some stability by averaging over a subset, reducing variance compared to stochastic gradient descent.<br>

Cons:<br>
Some noise in gradient estimates: Mini-batch updates can be noisier than full batch training, something mini-batch cannot accurate estimate of the gradient compare to batch training.<br>

## 6. (max points 4). Why is the enthropy-based loss (scost) function is typically used in training neural networks?

About Probabilistic: Cross-entropy loss treats the output of a neural network (often after a softmax activation) as a probability distribution over classes. This is ideal for classification tasks where we want the network to output probabilities that align closely with the true class probabilities.<br>

Penalty for Wrong Predictions: Entropy-based losses are particularly sensitive to incorrect predictions. They penalize confident yet incorrect predictions more heavily, guiding the network to adjust weights to improve accuracy.<br>

Good Gradient Properties: Entropy-based losses provide gradients that are well-suited for backpropagation. This helps the model learn faster and converge more effectively by providing meaningful updates, even when predictions are close but not quite accurate.<br>

Logarithmic Penalty: The logarithmic nature of cross-entropy makes it sensitive to misclassified examples, pushing the model harder to correctly classify difficult samples. This helps the network make predictions that align well with ground truth classes.<br>

## 7. (max points 4). Describe forward and backward propagation for a multilayer (deep) neural network.

Forward Propagation:<br>
Forward propagation is the process of passing input data through the network to produce an output.<br>

1. Input Layer: Data enters the network.<br>

2. Layer-by-Layer Computation: Each layer applies a linear transformation (using weights and biases) followed by an activation function, which introduces non-linearity, making it possible for the network to learn complex patterns.<br>

`z^(l) = W^(l) * a^(l-1) + b^(l)`<br>

`z^(l)` is the weighted sum of inputs<br>
`W^(l)` is the weight matrix<br>
`b^(l)` is the bias vector.<br>

`a^(l) = f(z^(l))`<br>

`f` is the activation function applied to the layer's output.<br>

3. Output Layer: The final layer produces the output, typically using a softmax or sigmoid activation for classification.<br>

4. Loss Calculation: The output is compared to the ground truth using a loss function (e.g., cross-entropy), which quantifies the error.<br>

Backward Propagation:<br>
Backward propagation is the process of updating weights to minimize the loss, using gradients computed from the loss with respect to each weight and bias.<br>

1. Loss Calculation: Compute the loss using the output from forward propagation and the true labels.<br>

2. Weight Updates: Update weights and biases for each layers<br>
`W^(l) = W^(l) - α * ∂L/∂W^(l)`<br>
`b^(l) = b^(l) - α * ∂L/∂b^(l)`<br>


## 8. (max points 4). Why do exploding and vanishing gradients problems occur during the training process and how to resolve these problems.

Vanishing Gradients:<br>
Vanishing gradients occur when the gradients (partial derivatives of the loss function with respect to the weights) become very small during backpropagation.<br>

Problems:<br>
Slow Learning: The weights update very slowly because the gradient is too small.<br>
Poor Performance: The network might not learn properly.<br>

This problem is common in deep networks with activation functions like the sigmoid or tanh, which squash their input into a small output range (0 to 1 for sigmoid and -1 to 1 for tanh).<br>
When gradients are backpropagated through many layers, they can be repeatedly multiplied by small numbers, causing them to shrink exponentially.<br>

Solutions:<br>
Activation Functions: Using activation functions that do not squash the gradient as much, such as ReLU.<br>
Weight Initialization: Using better weight initialization methods like He initialization or Xavier initialization to maintain the scale of gradients.<br>
Batch Normalization: Normalizing the inputs of each layer to maintain a stable gradient flow.<br>


Exploding Gradients:<br>
Exploding gradients occur when the gradients grow exponentially during backpropagation.<br>

Problems:<br>
Unstable Training: The weights can change dramatically, causing the training process to become unstable.<br>
Overflow: The weights can become so large that they result in numerical overflow, causing the model to fail.<br>

Exploding gradients can occur in deep networks when the gradients are repeatedly multiplied by large numbers during backpropagation, leading to an exponential increase.<br>
This is often seen with certain weight initialization methods or when using activation functions that don't constrain their output.<br>

Solutions:<br>
Gradient Clipping: Clipping the gradients during backpropagation to a maximum value to prevent them from growing too large.<br>
Weight Regularization: Adding regularization terms to the loss function to penalize large weights.<br>

## 9. (max points 4). What is overfitting and underfitting in a neural network?

Overfitting:<br>
Overfitting occurs when a neural network learns the training data too well, including noise and outliers, making it too specific to the training set.<br>
Symptoms: High accuracy on the training set but poor generalization to new, unseen data, resulting in low accuracy on validation/test data.<br>
Causes: Often happens with complex models or insufficient data for the complexity of the network.<br>
Solutions: Regularization techniques (like dropout, L2 regularization), simplifying the model architecture, or increasing the amount of training data.<br>

Underfitting:<br>
Definition: Underfitting occurs when a neural network is too simple to capture the patterns in the data, failing to learn from both the training data and the test data.<br>
Symptoms: Low accuracy on both training and validation/test sets, indicating that the model isn't capturing essential data patterns.<br>
Causes: Usually due to an overly simplistic model, insufficient training time, or inadequate feature representation.<br>
Solutions: Increase model complexity, improve feature selection, train longer, or use a more powerful network.<br>

## 10. (max points 4). What is the normalization technique, how does it work, and why is it needed?

Normalization transforms the features of the dataset to a common scale, like [0, 1] or [-1, 1]. This process ensures that the different features contribute equally to improves the performance of machine learning algorithms.

Common Normalization Techniques:

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

Min-Max Scaled Data:<br>

Min-Max Scaling X' = (X - Xmin)/(Xmax - Xmin)<br>

[[0.  0.  0. ]<br>
[0.5 0.5 0.5]<br>
[1.  1.  1. ]]<br>

Z-Score Normalization:<br>

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

Robust Scaling:<br>

Min-Max Scaling X' = (X - median(X))/IQR(X)<br>
Interquartile Range (IQR): The range between the 25th percentile (Q1) and the 75th percentile (Q3).<br>
IQR(X1) = 5.5 - 2.5 = 3.0
IQR(X2) = 6.5 - 3.5 = 3.0
IQR(X3) = 7.5 - 4.5 = 3.0

X11' = (1-4)/3 = -1<br>

[[-1. -1. -1.]<br>
[ 0.  0.  0.]<br>
[ 1.  1.  1.]]<br>