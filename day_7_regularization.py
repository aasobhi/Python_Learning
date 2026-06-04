import numpy as np

# 1. Targets and Predictions for 3 complex parameters
y_train = np.array([1, 0, 1])
predictions = np.array([0.9, 0.1, 0.85])

# Simulated massive weights from an overfitted, hyper-complex model
w_overfitted = np.array([10.5, -12.2, 15.1])
b = -2.0
lambda_penalty = 2.5  # The regularization parameter

# 2. Regularized Cost Function
def compute_regularized_cost(y, pred, w, reg_param):
    m = len(y)
    # Core Log Loss
    base_loss = -y * np.log(pred) - (1 - y) * np.log(1 - pred)
    base_cost = (1 / m) * np.sum(base_loss)
    
    # HYBRID TASK: Manually type these two lines to feel the regularization penalty math
    reg_penalty = (reg_param / (2 * m)) * np.sum(np.square(w))
    total_cost = base_cost + reg_penalty
    
    return base_cost, total_cost

# 3. Calculate the difference
standard_cost, regularized_cost = compute_regularized_cost(y_train, predictions, w_overfitted, lambda_penalty)

print("Analyzing Overfitting Penalization (L2 Regularization)...")
print("----------------------------------------------------------")
print(f"Base Error Cost (How close the predictions are):   {standard_cost:.4f}")
print(f"Regularized Total Cost (With complexity penalty): {regularized_cost:.4f}")
print("----------------------------------------------------------")
print("By forcing the total cost up due to huge weights, the math compels")
print("Gradient Descent to keep the weights small, smoothing out the model's curves!")