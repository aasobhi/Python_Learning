import numpy as np

# A single house: [Size, Bedrooms, Floors, Age]
x_row = np.array([2104, 5, 1, 45])

# The model's learned weights for each feature
w_vector = np.array([0.1, 4.0, 10.0, -2.0])
b = 80.0

# Calculate the prediction instantly using np.dot()
prediction = np.dot(w_vector, x_row) + b

print(f"Predicted Price: ${prediction}k")