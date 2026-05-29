import numpy as np

# Suppose X_train is your matrix: (rows, columns)
# 1. Calculate the mean of each column (axis=0)
mu = np.mean(X_train, axis=0)

# 2. Calculate the range (max - min) of each column
sigma = np.ptp(X_train, axis=0) # 'ptp' stands for "peak-to-peak" (max - min)

# 3. Apply the formula to the entire matrix at once!
X_norm = (X_train - mu) / sigma