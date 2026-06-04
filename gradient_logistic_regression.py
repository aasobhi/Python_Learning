import numpy as np

def compute_gradient_logistic(X, y, w, b):
    """
    Computes the gradient for logistic regression.
    X : Matrix of features (m rows, n features)
    y : Vector of true binary labels (m rows)
    w : Vector of weights (n features)
    b : Scalar bias
    """
    m, n = X.shape
    
    # 1. Compute the linear base (z)
    z = np.dot(X, w) + b
    
    # 2. Apply the Sigmoid Filter to get probability predictions
    f_wb = 1 / (1 + np.exp(-z))
    
    # 3. Calculate errors across all training examples
    errors = f_wb - y
    
    # 4. Vectorized Gradient Math (Computes all dj_dw and dj_db states in parallel)
    dj_dw = (1 / m) * np.dot(X.T, errors)  # Transpose X to align features with errors
    dj_db = (1 / m) * np.sum(errors)
    
    return dj_dw, dj_db