# HW to Chapter 8 “Initialization and Training Sets”

# Non-programming Assignment

### 1. To which values initialize parameters (W, b) in a neural networks and why?
Zero Initialization leads to symmetry; all neurons will learn the same features and thus fail to break symmetry during training.

Random Initialization helps break symmetry, it can lead to exploding or vanishing gradients, especially in deep networks.

Xavier (Glorot) Initialization helps maintain the variance of the outputs of each layer, which mitigates the vanishing gradient problem in networks using activation functions like sigmoid or tanh.

### 2. Describe the problem of exploding and vanishing gradients?
Vanishing gradients occur when the gradients of the loss function become very small as they are propagated backward through the layers of the network. This leads to very little or no update of the weights in the earlier layers.

Exploding gradients occur when the gradients become excessively large during backpropagation, leading to large updates to the weights.

### 3. What is Xavier initialization?
Xavier initialization, also known as Glorot initialization, is a weight initialization technique designed to keep the scale of the gradients roughly the same across all layers in a neural network.

The primary goal of Xavier initialization is to prevent the vanishing or exploding gradient problems that can occur in deep networks, allowing for better convergence during training.

For a layer with n int input units and n out output units, Xavier initialization sets the weights W of the layer using the following Uniform Distribution and Normal Distribution.

### 4. Describe training, validation, and testing data sets and explain their role and why all they are needed.

**Training Set**:  
The training set is the portion of the dataset used to train the model. It contains input-output pairs that the model learns from. The model adjusts its parameters based on this data to minimize the loss function.

**Validation Set**:  
The validation set is used to tune the model's hyperparameters and evaluate its performance during training. It helps to ensure that the model is not overfitting to the training data. The performance on the validation set can indicate how well the model will generalize to unseen data.

**Testing Set**:  
The testing set is a separate portion of the dataset that is used to evaluate the final model's performance after training is complete. It provides an unbiased assessment of how the model will perform in real-world scenarios.

**Why All Are Needed**:  
- **Training** is essential for learning the model parameters.
- **Validation** helps in fine-tuning and selecting the best model configuration.
- **Testing** assesses the final model's performance, ensuring it generalizes well to new, unseen data.

### 5. What is training epoch?

A training epoch refers to one complete pass through the entire training dataset during the training process. In each epoch, the model learns from all the training examples, updating its parameters to minimize the loss function. Multiple epochs are often required to ensure the model converges and learns effectively, as one pass may not be sufficient for the model to capture the underlying patterns in the data.

### 6. How to distribute training, validation, and testing sets?

The distribution of training, validation, and testing sets can vary depending on the size of the dataset and the problem domain. A common practice is:

- For small datasets (e.g., < 10,000 samples):
  - **Training**: 70%
  - **Validation**: 15%
  - **Testing**: 15%

- For medium datasets (e.g., 10,000 to 100,000 samples):
  - **Training**: 80-90%
  - **Validation**: 5-10%
  - **Testing**: 5-10%

- For large datasets (e.g., > 1,000,000 samples):
  - **Training**: 98%
  - **Validation**: 1%
  - **Testing**: 1%

This distribution ensures that the model has enough data to learn from while still allowing for effective validation and testing.

### 7. What is data augmentation and why may it needed?

**Data Augmentation**:  
Data augmentation is a technique used to artificially increase the size of a training dataset by creating modified versions of existing data points. This can include transformations such as rotation, scaling, flipping, cropping, and adding noise.

**Why It May Be Needed**:  
- **Improved Generalization**: Augmentation helps models generalize better by exposing them to a wider variety of training examples, which can prevent overfitting.
- **Limited Data**: In scenarios where data collection is expensive or time-consuming, augmentation allows for more training data without the need for additional samples.
- **Enhancing Robustness**: It can make models more robust to variations in input data, leading to better performance in real-world applications.