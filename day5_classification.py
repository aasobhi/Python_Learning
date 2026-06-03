import numpy as np
import matplotlib.pyplot as plt

# 1. Training Data: Hours Studied
X_train = np.array([1.0, 2.0, 3.0, 5.0, 6.0, 7.0])
# Targets: 0 = Failed, 1 = Passed
y_train = np.array([0, 0, 0, 1, 1, 1])

# 2. The Sigmoid Function Implementation
def sigmoid(z):
    """Squashes any input number to a value between 0 and 1"""
    return 1 / (1 + np.exp(-z))

# 3. Model Parameters (Pre-optimized for this example)
w = 1.5
b = -6.0

print("Running Logistic Regression Predictions...")
print("------------------------------------------")

# 4. Predict probabilities for each student
for i in range(len(X_train)):
    hours = X_train[i]
    
    # Calculate z (the linear part: w*x + b)
    z = w * hours + b
    # Pass it through the sigmoid function to get the probability
    probability = sigmoid(z)
    
    # Make a final decision using a 0.5 threshold (The Decision Boundary)
    prediction = 1 if probability >= 0.5 else 0
    
    status = "PASS" if prediction == 1 else "FAIL"
    print(f"Studied {hours} hours -> Probability of passing: {probability*100:5.1f}% | AI Decision: {status}")

print("------------------------------------------")
print("Notice how 4 hours would be the perfect 'Decision Boundary' fence line!")