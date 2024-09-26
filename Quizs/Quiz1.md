# Quiz 1

Non-programming Assignment:
## Describe in detail forward- and backpropagation method for a neural network with one hidden layer including the expressions how to calculate the derivatives and update the parameters for a deep neural network.

#### Forward Propagation
In forward propagation, input data passes through each layer, computing a weighted sum, adding a bias, and applying an activation function to generate output.
1. **Input**: Pass the input data `x` into the network.
2. **Weighted Sum**: Calculate the weighted sum `z = w * x + b` for each neuron.
3. **Activation Function**: Apply an activation function `f(z)` to get the output `y`.
4. **Repeat**: Perform this process for each layer until reaching the output layer.
5. **Final Output**: The final output is the model’s prediction.

**Formula**: 
`y = f(w_1 * x_1 + w_2 * x_2 + ... + w_n * x_n + b)`

#### Backpropagation
Backpropagation computes the error and adjusts weights using gradient descent to minimize the loss.
1. **Error Calculation**: Compute the loss between the predicted output and the actual target.
2. **Gradient Calculation**: Compute the gradient of the loss with respect to weights using the chain rule.
3. **Update Weights**: Adjust weights using gradient descent: 
   `w = w - α * ∂J/∂w`
4. **Repeat**: Continue updating weights and biases across all layers to minimize the loss.

**Formula**: 
`w = w - α * ∂J/∂w`