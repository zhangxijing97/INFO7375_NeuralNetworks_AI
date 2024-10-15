# scaler_test.py

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Original data
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Min-Max Scaling
scaler = MinMaxScaler()
X_minmax = scaler.fit_transform(X)

print("Original Data:\n", X)
print("Min-Max Scaled Data:\n", X_minmax)





import numpy as np
from sklearn.preprocessing import StandardScaler

# Original data
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Standardization
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

print("Original Data:\n", X)
print("Standardized Data:\n", X_standardized)
standard_deviations = scaler.scale_
print(standard_deviations)




import numpy as np
from sklearn.preprocessing import RobustScaler

# Original data
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Robust Scaling
scaler = RobustScaler()
X_robust = scaler.fit_transform(X)

print("Original Data:\n", X)
print("Robust Scaled Data:\n", X_robust)