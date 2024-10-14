# regularization.py

import numpy as np

def l2_regularization(weights, lambda_reg):
    """
    Compute the L2 regularization term and its gradients.

    Parameters:
    - weights: dictionary containing weights of the neural network
    - lambda_reg: regularization strength

    Returns:
    - reg_cost: L2 regularization cost
    - gradients: dictionary containing the gradients of the weights
    """
    reg_cost = 0
    gradients = {}

    for key in weights.keys():
        reg_cost += np.sum(np.square(weights[key]))
        gradients[key] = 2 * lambda_reg * weights[key]

    return reg_cost, gradients