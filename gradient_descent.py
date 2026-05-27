import numpy as np

# 1. Load our scaled housing vectors (m = 2 rows)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
m = len(x_train)

# 2. Set our starting unoptimized parameters
w = 0.0
b = 0.0

# 3. Tuning parameters: 10 steps with a safe learning rate (alpha)
iterations = 10
alpha = 0.1

print("--- Simulating Gradient Descent Steps ---")

for step in range(iterations):
    # Calculate all predictions (y-hat) simultaneously using numpy vectors
    predictions = w * x_train + b
    errors = predictions - y_train
    
    # Calculate the mathematical derivative slope for w and b
    dj_dw = (1 / m) * np.sum(errors * x_train)
    dj_db = (1 / m) * np.sum(errors)
    
    # SIMULTANEOUS UPDATE: Overwrite both variables at the exact same step
    w = w - (alpha * dj_dw)
    b = b - (alpha * dj_db)
    
    # Calculate the fresh cost score to check our optimization progress
    current_cost = (1 / (2 * m)) * np.sum((predictions - y_train) ** 2)
    
    print(f"Step [{step}]: Scoreboard Cost: {current_cost:<6.1f} | Updated w = {w:.1f}, b = {b:.1f}")