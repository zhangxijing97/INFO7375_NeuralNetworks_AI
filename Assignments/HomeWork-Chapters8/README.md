# HW to Chapter 8 “Initialization and Training Sets”

# Non-programming Assignment

### 1. To which values initialize parameters (W, b) in a neural networks and why?

Zero Initialization leads to symmetry; all neurons will learn the same features and thus fail to break symmetry during training.

Random Initialization helps break symmetry, it can lead to exploding or vanishing gradients, especially in deep networks.

Xavier (Glorot) Initialization helps maintain the variance of the outputs of each layer, which mitigates the vanishing gradient problem in networks using activation functions like sigmoid or tanh.

### 2. Describe the problem of exploding and vanishing gradients?

### 3. What is Xavier initialization?

### 4. Describe training, validation, and testing data sets and explain their role and why all they are needed.

### 5. What is training epoch?

### 6. How to distribute training, validation, and testing sets?

### 7. What is data augmentation and why may it needed?