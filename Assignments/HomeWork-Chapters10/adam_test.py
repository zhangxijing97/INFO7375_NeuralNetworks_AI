# adam_test.py

import numpy as np

# Initialize parameters
theta = np.random.randn(2, 2)
m = np.zeros_like(theta)
v = np.zeros_like(theta)
t = 0

# Hyperparameters
alpha = 0.001  # Initial learning rate
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Dummy gradient function
def compute_gradients(theta):
    return np.random.randn(2, 2)

# Function to compute effective learning rate
def effective_learning_rate(alpha, m_hat, v_hat, epsilon):
    return alpha / (np.sqrt(v_hat) + epsilon)

# Adam optimization step
for i in range(10):  # Use a small number of iterations for demonstration
    t += 1
    g = compute_gradients(theta)
    
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * (g ** 2)
    
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    
    # Update parameters
    theta -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)
    
    # Compute and print effective learning rate for each parameter
    eff_lr = effective_learning_rate(alpha, m_hat, v_hat, epsilon)
    print(f"Iteration {t}")
    print("Effective Learning Rate:")
    print(eff_lr)
    print()

print("Optimized Parameters:", theta)