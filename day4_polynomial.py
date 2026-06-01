import numpy as np
import matplotlib.pyplot as plt

# 1. Non-linear training data (Imagine a curve, like engine wear over time)
x_raw = np.array([1, 2, 3, 4, 5])
y_train = np.array([2, 5, 10, 17, 26]) # Formed by y = x^2 + 1

# 2. Feature Engineering: Create a brand new feature by squaring x!
x_squared = x_raw ** 2

print("Raw Feature: ", x_raw)
print("Engineered Feature (x^2):", x_squared)

# 3. Stack them into a matrix to feed to a multiple linear regression model
X_mapped = np.stack((x_raw, x_squared), axis=1)

# Hardcoded optimal weights for our engineered features
# f(x) = 0*(x) + 1*(x^2) + 1
w_vec = np.array([0.0, 1.0])
b_scalar = 1.0

# 4. Calculate predictions simultaneously using our vector dot product
predictions = np.dot(X_mapped, w_vec) + b_scalar

print("---------------------------------------------")
print(f"AI Predictions: {predictions}")
print(f"Actual Targets: {y_train}")
print("Perfect fit achieved via feature engineering!")