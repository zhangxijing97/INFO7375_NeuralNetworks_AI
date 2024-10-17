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
rₒ is the initial learning rate<bar>
γ is the decay rate<bar>
e is the epoch sequential number<bar>
- Step Decay: Reduces the learning rate by a factor at specific intervals (e.g., halving every 10 epochs).
- Exponential Decay: Reduces the learning rate exponentially based on the epoch number.
- Cosine Annealing: Learning rate follows a cosine function for non-linear decay.

## 2. What are saddle and plateau problems?

## 3. Why should we avoid grid approach in hyperparameter choice?

## 4. What is mini batch and how is it used?