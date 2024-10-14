# HW to Chapter 10 “Normalization and Optimization Methods”

# Non-programming Assignment

## 1. What is normalization and why is it needed?
Normalization transforms the features of the dataset to a common scale, like [0, 1] or [-1, 1]. This process ensures that the different features contribute equally to improves the performance of machine learning algorithms.

Common Normalization Techniques:

Original Data:<br>
 [[1 2 3]<br>
 [4 5 6]<br>
 [7 8 9]]<br>

Min-Max Scaled Data:<br>
Min-Max Scaling X' = (X - Xmin)/(Xmax - Xmin)<br>
 [[0.  0.  0. ]<br>
 [0.5 0.5 0.5]<br>
 [1.  1.  1. ]]<br>

Z-Score Normalization<br>
Min-Max Scaling X' = (X - Xmin)/(Xmax - Xmin)<br>
 [[-1.22474487 -1.22474487 -1.22474487]<br>
 [ 0.          0.          0.        ]<br>
 [ 1.22474487  1.22474487  1.22474487]]<br>

Robust Scaling<br>
Min-Max Scaling X' = (X - Xmin)/(Xmax - Xmin)<br>
 [[-1. -1. -1.]<br>
 [ 0.  0.  0.]<br>
 [ 1.  1.  1.]]<br>

## 2. What are vanishing and exploding gradients?

## 3. What Adam algorithm and why is it needed?

## 4. How to choose hyperparameters?