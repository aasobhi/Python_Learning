import numpy as np

# Original feature: x
x = np.array([2, 3, 4])

# Create a polynomial feature set (x, x^2)
x_poly = np.column_stack((x, x**2)) 

# Now your model sees: 
# [2, 4]
# [3, 9]
# [4, 16]
# Gradient descent will now find w1 and w2 for this curved data!