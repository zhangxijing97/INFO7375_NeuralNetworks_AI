# HW to Chapter 12 “Softmax”

# Non-programming Assignment

## 1. What is the reason for softmax?

- Probability Distribution: Softmax converts logits into a range (0, 1) where the sum is 1, representing probabilities.
- Multi-Class Classification: It allows the model to assign probabilities to each class and pick the class with the highest probability.
- Interpretability: The output probabilities make the model's predictions easier to understand.
- Gradient-Based Optimization: When combined with a loss function like cross-entropy, it helps in efficiently training the model by providing useful gradients.

## 2. What is softmax and how does it works?

The softmax function is an activation function used in neural networks, particularly in the final layer of a classification model. It converts a vector of raw scores (logits) into probabilities that sum to 1, making it suitable for multi-class classification tasks.

### How Does Softmax Work?

1. **Input (Logits)**:
   - You start with a vector of raw scores `z = [z_1, z_2, ..., z_K]` from the neural network's output layer.

2. **Exponentiation**:
   - Each logit `z_i` is exponentiated to ensure all values are positive.
   - `e^{z_i}` for each `i`.

3. **Normalization**:
   - Each exponentiated value is divided by the sum of all exponentiated values to normalize them into a probability distribution.
   - `σ(z)_i = e^{z_i} / Σ_{j=1}^K e^{z_j}`

### Example

Suppose you have three logits: `z = [2.0, 1.0, 0.1]`.

1. **Exponentiation**:
   - `e^{2.0} ≈ 7.39`
   - `e^{1.0} ≈ 2.72`
   - `e^{0.1} ≈ 1.11`

2. **Sum of Exponentiated Values**:
   - `7.39 + 2.72 + 1.11 ≈ 11.22`

3. **Normalization**:
   - `σ(z)_1 = 7.39 / 11.22 ≈ 0.66`
   - `σ(z)_2 = 2.72 / 11.22 ≈ 0.24`
   - `σ(z)_3 = 1.11 / 11.22 ≈ 0.10`

So, the softmax probabilities are approximately `[0.66, 0.24, 0.10]`. This means the first class has a 66% probability, the second class has a 24% probability, and the third class has a 10% probability.