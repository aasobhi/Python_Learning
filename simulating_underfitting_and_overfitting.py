import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 1. Generate some noisy curved data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2.1, 3.9, 9.2, 16.1, 24.8]) # Follows x^2 roughly, with some noise

# --- SIMULATING UNDERFIT ---
# Forcing a simple straight line onto curved data
model_under = LinearRegression()
model_under.fit(X, y) 
# Result: High error on training data

# --- SIMULATING OVERFIT ---
# Transforming 1 feature into a 10th-degree polynomial matrix
poly_over = PolynomialFeatures(degree=10)
X_overfit = poly_over.fit_transform(X)

model_over = LinearRegression()
model_over.fit(X_overfit, y)
# Result: 0% error on training data, but will give wild, crazy predictions for X = 6