import numpy as np
import time

# 1. Training Data (Size vs Price)
x_train = np.array([1.0, 2.0, 3.0, 4.0])
y_train = np.array([300.0, 500.0, 680.0, 850.0])

# 2. Starting point (intentional bad guesses)
w = 0.0
b = 0.0

# 3. Hyperparameters
learning_rate = 0.01
iterations = 1000
m = len(x_train)

print("Starting training loop...")
print("--------------------------")

# 4. The Training Simulation Loop
for i in range(iterations):
    # Calculate current predictions and errors
    f_wb = w * x_train + b
    error = f_wb - y_train
    
    # Calculate gradients (how steep the slope is)
    dj_dw = (1/m) * np.sum(error * x_train)
    dj_db = (1/m) * np.sum(error)
    
    # Update parameters simultaneously by taking a step "downhill"
    w = w - learning_rate * dj_dw
    b = b - learning_rate * dj_db
    
    # Print the update every 100 steps so you can watch it learn
    if i % 100 == 0:
        # Calculate current average cost (Mean Squared Error)
        cost = (1/(2*m)) * np.sum(error**2)
        print(f"Iteration {i:4d}: Cost = {cost:8.2f} | w = {w:6.2f}, b = {b:6.2f}")
        time.sleep(0.1) # Deliberate slight pause so you can watch it stream in

print("--------------------------")
print(f"Training Complete! Final Model: f(x) = {w:.2f}x + {b:.2f}")