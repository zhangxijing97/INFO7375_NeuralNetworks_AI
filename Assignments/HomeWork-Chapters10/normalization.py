# normalization.py

import numpy as np

class Normalization:
    def __init__(self):
        self.scaler = None
    
    def min_max_scaling(self, X):
        """Min-Max scaling to scale features to a range of [0, 1]."""
        self.scaler = (np.min(X, axis=0), np.max(X, axis=0))
        return (X - self.scaler[0]) / (self.scaler[1] - self.scaler[0])

    def z_score_scaling(self, X):
        """Z-Score normalization to scale features to have mean 0 and standard deviation 1."""
        self.scaler = (np.mean(X, axis=0), np.std(X, axis=0))
        return (X - self.scaler[0]) / self.scaler[1]

    def robust_scaling(self, X):
        """Robust scaling to reduce the influence of outliers by using median and interquartile range."""
        self.scaler = (np.median(X, axis=0), np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        return (X - self.scaler[0]) / self.scaler[1]