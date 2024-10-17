# Quiz 3

High Bias (Underfitting): The model is too simple to capture the true patterns in the data, and does not fit properly. For example, using a linear regression for data with a quadratic relationship.<br>

High Variance (Overfitting): The model is too complex and captures noise in the training data as if it were a true pattern. This results in poor performance on new, unseen data. For example, using a high-degree polynomial regression that fits all the nuances and noise in the training data.<br>

Example: Predicting Housing Prices<br>
Scenario 1: High Bias<br>
Imagine we use a very simple model, such as a linear regression, to predict housing prices. The model assumes that the relationship between house size and price is strictly linear.<br>

Model: Linear Regression (a straight line).<br>
Result: The model might predict prices that are systematically off because it oversimplifies the relationship. For instance, if the true relationship is quadratic (prices increase faster as house size increases), the linear model won't capture this trend.<br>
Impact: The model has high bias. It underfits the data, leading to systematic errors in predictions. The decision boundary (straight line) is too rigid and does not capture the true patterns in the data.<br>

Scenario 2: High Variance<br>
Now, let's use a very complex model, such as a high-degree polynomial regression, to fit the same data.<br>

Model: Polynomial Regression (a very wavy line).<br>
Result: The model fits the training data very closely, capturing all the nuances and even the noise. On the training data, the predictions are very accurate.<br>
Impact: The model has high variance. It overfits the data, leading to poor generalization to new, unseen data. The decision boundary is too flexible, capturing random noise as if it were true patterns, resulting in large errors on new data points.<br>

Scenario 3: Just Right<br>
Finally, let's use a model that balances bias and variance, such as a second-degree polynomial regression.<br>

Model: Polynomial Regression (a smooth curve).<br>
Result: The model captures the general trend in the data without being too rigid or too flexible. It fits the training data well and generalizes better to new data.<br>
Impact: The model has a good balance of bias and variance. It adequately captures the relationship between house size and price without overfitting to noise or underfitting the true pattern.<br>