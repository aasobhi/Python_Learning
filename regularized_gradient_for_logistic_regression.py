import numpy as np

def compute_gradient_logistic_reg(X, y, w, b, lambda_):
    """
    Computes the regularized gradient for logistic regression.
    X : Feature matrix (m rows, n columns)
    y : Target vector (m rows)
    w : Current weights vector (n columns)
    b : Current bias scalar
    lambda_ : Regularization volume knob constant
    """
    m, n = X.shape
    
    # Step 1: Compute standard predictions and errors
    z = np.dot(X, w) + b
    f_wb = 1 / (1 + np.exp(-z)) # Sigmoid filter
    errors = f_wb - y
    
    # Step 2: Compute base gradients (The left side of the slide equations)
    dj_dw = (1 / m) * np.dot(X.T, errors)
    dj_db = (1 / m) * np.sum(errors)
    
    # Step 3: Compute the regularization penalty vector (The right side of the slide equation)
    # This automatically matches the length of your feature space
    regularization_vector = (lambda_ / m) * w
    
    # Step 4: Add the penalty to the weights gradient (Leave dj_db alone!)
    dj_dw = dj_dw + regularization_vector
    
    return dj_dw, dj_db