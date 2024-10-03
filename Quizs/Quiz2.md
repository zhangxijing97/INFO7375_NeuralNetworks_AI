# Quiz 2

## Describe the following activation functions and their usage: linear, ReLU, sigmoid, tanh, and softmax.

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