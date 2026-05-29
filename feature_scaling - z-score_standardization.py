import numpy as np

# 1. Calculate Mean (mu) and Standard Deviation (sigma)
mu = np.mean(X_train, axis=0)
sigma = np.std(X_train, axis=0)

# 2. Apply Z-score normalization
X_norm = (X_train - mu) / sigma