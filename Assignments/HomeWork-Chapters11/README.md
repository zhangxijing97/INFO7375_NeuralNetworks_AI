# HW to Chapter 11 “Learning Rates Decay and Hyperparameters”

# Non-programming Assignment

## 1. What is learning rate decay and why is it needed?

- Stability: High learning rates can cause the model to oscillate around the minimum. Decaying the learning rate helps the model settle into the minimum smoothly.
- Avoiding Local Minima: High initial learning rates help escape local minima. As training progresses, a lower learning rate allows finer adjustments.
- Improving Performance: Smaller updates later in training can help achieve lower loss and higher accuracy.
- Efficiency: It ensures efficient training by dynamically adjusting the learning rate.

Common Strategies for Learning Rate Decay

- 1/t Decay: Learning rate decreases proportionally to the inverse of the epoch number.
r = rₒ/1+γ
rₒ is the initial learning rate<br>
γ is the decay rate<br>
e is the epoch sequential number<br>
- Step Decay: Reduces the learning rate by a factor at specific intervals (e.g., halving every 10 epochs).
- Exponential Decay: Reduces the learning rate exponentially based on the epoch number.
- Cosine Annealing: Learning rate follows a cosine function for non-linear decay.

## 2. What are saddle and plateau problems?

- Saddle Points
Definition: Points where the gradient is zero but are not local minima. Loss increases in some directions and decreases in others. Which make the value of gradients close to zero<br>
Problem: Optimization can stall because the gradients are very small, slowing down or halting progress.<br>

- Plateaus
Definition: Flat regions of the loss function with very small or zero gradients over a large area.<br>
Problem: Training can stall, leading to slow convergence and inefficiency.<br>

## 3. Why should we avoid grid approach in hyperparameter choice?

- Computational Expense:

Exponential Growth: The number of evaluations increases exponentially with the number of hyperparameters.<br>
High Resource Usage: This makes grid search computationally expensive and time-consuming.<br>

- Inefficient Exploration:

Fixed Intervals: Grid search evaluates hyperparameters at fixed points, potentially missing optimal values between these points.<br>
Poor Coverage: This can lead to suboptimal exploration, especially in high-dimensional spaces.<br>

- Artificial Period Dependencies:
Pattern Bias: Fixed intervals can introduce artificial biases, leading to misleading results due to periodic patterns in hyperparameter interactions.<br>

## 4. What is mini batch and how is it used?