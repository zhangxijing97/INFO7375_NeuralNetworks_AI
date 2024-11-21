# HW to Chapter 19 “Recurrent Neural Networks”

# Non-programming Assignment

## 1. What are Recurrent Neural Networks (RNNs) and Why Are They Needed?
Recurrent Neural Networks (RNNs) are a type of neural network designed to process sequential data, such as time-series data, text, or speech. Unlike traditional feedforward neural networks, RNNs have loops that allow information to persist, making them ideal for tasks where the context from previous steps is important.

### Why Are They Needed?
- **Sequential Data**: Many real-world problems, such as language translation, stock prediction, and speech recognition, require understanding sequences of data.
- **Memory**: RNNs maintain a hidden state that acts as a memory, enabling them to capture dependencies in sequences.

---

## 2. What Role Do Time Steps Play in Recurrent Neural Networks?
Time steps in RNNs represent the sequential nature of the input data. At each time step, the network processes one element of the sequence while retaining information from previous steps in its hidden state. This sequential processing is what enables RNNs to model temporal or contextual dependencies.

---

## 3. What are the Types of Recurrent Neural Networks?
1. **Vanilla RNN**: The basic RNN structure with a single hidden layer.
2. **Long Short-Term Memory (LSTM)**: Designed to overcome the vanishing gradient problem with gating mechanisms (input, forget, and output gates).
3. **Gated Recurrent Unit (GRU)**: A simplified version of LSTM with fewer gates, making it computationally efficient.
4. **Bidirectional RNN (BRNN)**: Processes input sequences in both forward and backward directions to capture future and past context.
5. **Deep RNN (DRNN)**: Stacks multiple RNN layers to increase model capacity.

---

## 4. How is the Loss Function for RNN Defined?
The loss function for RNNs is typically the sum of losses across all time steps. For example:
- **Sequence Classification Tasks**: Cross-entropy loss is commonly used.
- **Regression Tasks**: Mean Squared Error (MSE) is often employed.

The loss at each time step \( t \) is calculated as:
\[
\text{Loss} = \sum_{t=1}^{T} L(y_t, \hat{y}_t)
\]
where:
- \( L \) is the loss function.
- \( y_t \) is the true output.
- \( \hat{y}_t \) is the predicted output.

---

## 5. How Do Forward and Backpropagation of RNN Work?

### Forward Propagation
1. At each time step \( t \):
   - The input \( x_t \) is combined with the hidden state \( h_{t-1} \) from the previous step.
   - The output and updated hidden state are computed using the activation function.

\[
h_t = f(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
\]

2. The output \( y_t \) is computed from the hidden state:
\[
y_t = g(W_{hy} h_t + b_y)
\]

### Backpropagation Through Time (BPTT)
1. The loss is propagated backward through time.
2. Gradients are computed for each time step, and parameters are updated using techniques like gradient descent.
3. Challenges like vanishing or exploding gradients may arise, which LSTMs and GRUs address effectively.

---

## 6. What are the Most Common Activation Functions for RNN?
1. **Sigmoid**: Used in gating mechanisms (e.g., in LSTMs and GRUs).
2. **Tanh**: Commonly used for hidden state updates.
3. **ReLU**: Occasionally used, though less common due to potential issues with exploding gradients.

---

## 7. What are Bidirectional Recurrent Neural Networks (BRNN)?
Bidirectional RNNs (BRNNs) extend standard RNNs by processing input sequences in both forward and backward directions. This means that at each time step, the model considers both past and future context.

### Why Are BRNNs Needed?
- Many tasks (e.g., speech recognition, where context from both sides is crucial) benefit from understanding both previous and upcoming information.
- BRNNs improve performance in tasks where the entire sequence is available before prediction.

---

## 8. What are Deep Recurrent Neural Networks (DRNN)?
Deep RNNs (DRNNs) stack multiple RNN layers on top of one another. The output of one layer serves as the input for the next layer, enabling the network to learn hierarchical representations.

### Why Are DRNNs Needed?
- Increased capacity to capture complex patterns and long-range dependencies.
- Improved feature extraction for more challenging tasks.

---