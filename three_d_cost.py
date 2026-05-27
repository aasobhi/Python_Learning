import numpy as np
import matplotlib.pyplot as plt

# 1. Setup our 2 simple data points from your housing notes
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
m = len(x_train)

# 2. Build a grid of weights (w) and biases (b) to evaluate
# This creates a meshgrid to plot our 3D surface
w_vals = np.linspace(50, 350, 100)
b_vals = np.linspace(-100, 300, 100)
W, B = np.meshgrid(w_vals, b_vals)

# 3. Compute the Cost Score J(w,b) for every point on the grid
# Instead of importing from a fake file, we do the math right here!
cost_grid = np.zeros(W.shape)
for i in range(100):
    for j in range(100):
        # Calculate predictions: (w * x) + b
        predictions = W[i,j] * x_train + B[i,j]
        # Calculate Squared Error Cost
        cost_grid[i,j] = (1 / (2 * m)) * np.sum((predictions - y_train) ** 2)

# 4. Initialize and render the 3D Projection Plot using Matplotlib
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Draw the wireframe surface
surface = ax.plot_surface(W, B, cost_grid, cmap='viridis', alpha=0.8)

# Add clear labels for your study guide notes
ax.set_title("3D Cost Function Surface Plot J(w,b)")
ax.set_xlabel("Weight (w) - Slope")
ax.set_ylabel("Bias (b) - Intercept")
ax.set_zlabel("Cost Score J(w,b)")

# 5. Stop building background components and show the interactive window
plt.show()