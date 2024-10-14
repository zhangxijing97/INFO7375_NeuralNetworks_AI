# HW to Chapter 8 “Initialization and Training Sets”

# Non-programming Assignment

## Neural Networks Initialization and Training Concepts

### Parameter Initialization (W, b)
- **Weights (W)**:
  - **Xavier Initialization**: `Variance = 2 / (n_in + n_out)`
  - **He Initialization**: `Variance = 2 / n_in`
  - **Random Initialization**: Small values to avoid exploding gradients.
- **Biases (b)**: Initialized to zero.

### Exploding and Vanishing Gradients
- **Exploding Gradients**: Large gradients causing unstable updates.
- **Vanishing Gradients**: Small gradients preventing significant updates.

### Xavier Initialization
- Weights from `N(0, 2 / (n_in + n_out))`
- Helps in maintaining variance and gradients across layers.

### Data Sets
- **Training Set**: For model learning.
- **Validation Set**: For tuning and preventing overfitting.
- **Testing Set**: For final performance evaluation.

### Training Epoch
- One complete pass through the entire training dataset.

### Data Set Distribution
- **Training Set**: 70-80%
- **Validation Set**: 10-15%
- **Testing Set**: 10-15%

### Data Augmentation
- **Purpose**: Increase dataset size, improve generalization, reduce overfitting.
- **Methods**: Rotations, flips, shifts, scaling, noise addition.