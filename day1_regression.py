import numpy as np
import matplotlib.pyplot as plt

# 1. Setup your raw data (Size in 1,000 sq ft vs Price in $1,000s)
x_train = np.array([1.0, 2.0, 3.0, 4.0])
y_train = np.array([300.0, 500.0, 680.0, 850.0])

# 2. Hardcode your model parameters (Weight and Bias)
# Try changing these numbers later to see how the line moves!
w = 200 
b = 100

# 3. Create the function that calculates predictions
def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

# 4. Generate predictions and plot the results
predictions = compute_model_output(x_train, w, b)

plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Data')
plt.plot(x_train, predictions, c='b', label='Our Prediction Line')
plt.title("Housing Price Predictor - Initial Test")
plt.xlabel("Size (1000 sqft)")
plt.ylabel("Price ($1000s)")
plt.legend()
plt.show()