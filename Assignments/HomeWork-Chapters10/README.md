# HW to Chapter 10 “Normalization and Optimization Methods”

# Non-programming Assignment

## 1. What is normalization and why is it needed?

Normalization transforms the features of the dataset to a common scale, like [0, 1] or [-1, 1]. This process ensures that the different features contribute equally to improves the performance of machine learning algorithms.

### Common Normalization Techniques:

Original Data:<br>
 [[1 2 3]<br>
 [4 5 6]<br>
 [7 8 9]]<br>

Column 1 (Feature 1):<br>
Values: 1, 4, 7<br>
Column 2 (Feature 2):<br>
Values: 2, 5, 8<br>
Column 3 (Feature 3):<br>
Values: 3, 6, 9<br>

- **Min-Max Scaled Data**<br>

Min-Max Scaling X' = (X - Xmin)/(Xmax - Xmin)<br>

 [[0.  0.  0. ]<br>
 [0.5 0.5 0.5]<br>
 [1.  1.  1. ]]<br>

- **Z-Score Normalization**<br>

Z-Score Scaling X' = (X - μ)/σ<br>
μ is the mean of the feature<br>
σ is the standard deviation of the feature<br>
μ1 = 4<br>
μ2 = 5<br>
μ3 = 6<br>
σ1^2 = Σ(xi - μ)^2 / N<br>
= (1 - 4)^2 + (4 - 4)^2 + (7 - 5)^2 / 1<br>
= 6<br>
σ1 = 2.45<br>
σ2 = 2.45<br>
σ3 = 2.45<br>
X11' = (1-4)/2.45 = -1.22474487<br>

 [[-1.22474487 -1.22474487 -1.22474487]<br>
 [ 0.          0.          0.        ]<br>
 [ 1.22474487  1.22474487  1.22474487]]<br>

- **Robust Scaling**<br>

Min-Max Scaling X' = (X - median(X))/IQR(X)<br>
Interquartile Range (IQR): The range between the 25th percentile (Q1) and the 75th percentile (Q3).<br>
IQR(X1) = 5.5 - 2.5 = 3.0
IQR(X2) = 6.5 - 3.5 = 3.0
IQR(X3) = 7.5 - 4.5 = 3.0

X11' = (1-4)/3 = -1<br>

 [[-1. -1. -1.]<br>
 [ 0.  0.  0.]<br>
 [ 1.  1.  1.]]<br>

## 2. What are vanishing and exploding gradients?

## 3. What Adam algorithm and why is it needed?

## 4. How to choose hyperparameters?