# HW to Chapter 20 “LM, LSTM, and GRU”

# Non-programming Assignment

## 1. How Does a Language Model (LM) Work?
A Language Model (LM) is designed to predict the probability distribution of the next word in a sequence, given the previous words. It learns the relationships and patterns in text data to generate coherent and contextually appropriate sequences.

### Key Steps:
1. **Tokenization**: Input text is split into smaller units like words or subwords.
2. **Embedding**: Tokens are mapped to dense vector representations.
3. **Sequential Processing**: The model processes the sequence step-by-step using architectures like RNNs, LSTMs, GRUs, or Transformers.
4. **Output Prediction**: For each step, the model predicts the likelihood of the next token.

---

## 2. How Does Word Prediction Work?
Word prediction involves estimating the probability of the next word in a sequence based on the words that precede it. For example, given the sequence "The cat is on the," the model predicts the next word might be "mat."

### Steps:
1. **Contextual Understanding**: The model uses previous words to establish context.
2. **Conditional Probability**: The probability of the next word \( w_t \) is calculated as:
   \[
   P(w_t | w_1, w_2, \dots, w_{t-1})
   \]
3. **Output**: The model outputs the word with the highest probability.

---

## 3. How to Train a Language Model?
### Steps:
1. **Data Preparation**:
   - Collect large datasets of text.
   - Tokenize and preprocess the data.
2. **Define the Model**:
   - Use architectures like RNNs, LSTMs, GRUs, or Transformers.
3. **Objective Function**:
   - Minimize the negative log-likelihood or cross-entropy loss:
   \[
   L = -\sum_{t=1}^{T} \log P(w_t | w_1, w_2, \dots, w_{t-1})
   \]
4. **Training**:
   - Use techniques like gradient descent with backpropagation to update parameters.
5. **Validation**:
   - Evaluate on a validation set to monitor overfitting.

---

## 4. Describe the Problem and the Nature of Vanishing and Exploding Gradients

### Vanishing Gradients:
- Occurs when gradients become too small during backpropagation through deep networks.
- Causes weights to update very slowly, preventing the model from learning long-term dependencies.
- Common in models like Vanilla RNNs.

### Exploding Gradients:
- Occurs when gradients grow exponentially during backpropagation.
- Causes instability in training, with weights becoming excessively large.

### Nature of the Problem:
- Both issues arise due to repeated multiplication of gradients during backpropagation through time.
- Mitigation Techniques:
  - **Vanishing Gradients**: Use LSTMs or GRUs with gating mechanisms.
  - **Exploding Gradients**: Gradient clipping to cap gradient values.

---

## 5. What is LSTM and the Main Idea Behind It?
Long Short-Term Memory (LSTM) is a type of RNN designed to address the vanishing gradient problem and model long-term dependencies in sequential data.

### Main Idea:
- LSTMs use **gates** (input, forget, output) to control the flow of information.
- A **cell state** acts as memory, retaining important information over long sequences.

### Key Equations:
1. **Forget Gate**:
   \[
   f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
   \]
2. **Input Gate**:
   \[
   i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
   \]
3. **Output Gate**:
   \[
   o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
   \]

---

## 6. What is GRU?
Gated Recurrent Unit (GRU) is a simplified version of LSTM that uses fewer gates, making it computationally more efficient.

### Key Features:
- Combines the forget and input gates into a **reset gate**.
- Uses an **update gate** to determine how much of the past information to retain.

### Key Equations:
1. **Reset Gate**:
   \[
   r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)
   \]
2. **Update Gate**:
   \[
   z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)
   \]
3. **New Memory**:
   \[
   \tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)
   \]
4. **Final Output**:
   \[
   h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t
   \]

### Why Use GRU?
- Fewer parameters than LSTM.
- Faster training while retaining the ability to model long-term dependencies.