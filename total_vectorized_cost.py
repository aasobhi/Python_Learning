import numpy as np

def compute_logistic_cost(X, y, w, b):
    """
    Computes the total vectorized cost for logistic regression
    """
    m = X.shape[0]
    
    # 1. Get sigmoid probabilities for all rows simultaneously
    z = np.dot(X, w) + b
    f_wb = 1 / (1 + np.exp(-z))
    
    # 2. Prevent code from crashing due to log(0) errors (Machine Learning safety net)
    # np.clip forces values slightly away from pure 0.0 and 1.0
    f_wb = np.clip(f_wb, 1e-15, 1 - 1e-15)
    
    # 3. Calculate the single-line compressed cross-entropy cost matrix
    loss_vector = -y * np.log(f_wb) - (1 - y) * np.log(1 - f_wb)
    
    # 4. Average the losses over all m examples
    total_cost = np.sum(loss_vector) / m
    
    return total_cost