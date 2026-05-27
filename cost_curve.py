import numpy as np

# 1. Setup the 3 simple data points from your slide table
x_train = np.array([1.0, 2.0, 3.0])
y_train = np.array([1.0, 2.0, 3.0])
m = len(x_train)

# 2. Let's look at the 3 specific slope guesses from your slide
weight_guesses = [0.0, 0.5, 1.0, 1.5, 2.0]

print("--- Calculating the Scoreboard Curve J(w) ---")

for w in weight_guesses:
    # Calculate predictions for all 3 houses at once
    predictions = w * x_train
    
    # Calculate individual errors (Guess - Real)
    errors = predictions - y_train
    
    # Square the errors and sum them up
    sum_squared_errors = np.sum(errors ** 2)
    
    # Final scaling factor: 1 / (2 * m)
    cost_J = (1 / (2 * m)) * sum_squared_errors
    
    print(f"If Model tests Slope w = {w:<3} | Total Cost Score J(w) = {cost_J:.2f}")