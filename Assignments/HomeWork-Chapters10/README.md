# HW to Chapter 10 “Normalization and Optimization Methods”

# Non-programming Assignment

## 1. What is normalization and why is it needed?
Normalization transforms the features of the dataset to a common scale, like [0, 1] or [-1, 1]. This process ensures that the different features contribute equally to improves the performance of machine learning algorithms.

Common Normalization Techniques:

Original Data:
 [[1 2 3]
 [4 5 6]
 [7 8 9]]

Min-Max Scaled Data:
Min-Max Scaling X' = (X - Xmin)/(Xmax - Xmin)
 [[0.  0.  0. ]
 [0.5 0.5 0.5]
 [1.  1.  1. ]]


Z-Score Normalization
Min-Max Scaling X' = (X - Xmin)/(Xmax - Xmin)
 [[-1.22474487 -1.22474487 -1.22474487]
 [ 0.          0.          0.        ]
 [ 1.22474487  1.22474487  1.22474487]]

Robust Scaling
Min-Max Scaling X' = (X - Xmin)/(Xmax - Xmin)
 [[-1. -1. -1.]
 [ 0.  0.  0.]
 [ 1.  1.  1.]]

## 2. What are vanishing and exploding gradients?

## 3. What Adam algorithm and why is it needed?

## 4. How to choose hyperparameters?