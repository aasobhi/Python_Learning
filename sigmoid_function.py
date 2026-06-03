import numpy as np

def sigmoid(z):
    # np.exp() calculates e^-z in parallel for the whole matrix
    return 1 / (1 + np.exp(-z))

# Dataset: 3 patients, 2 features each (Tumor Size, Patient Age)
X = np.array([
    [1.5, 45],
    [5.1, 62],
    [0.8, 30]
])

# Weights and bias learned by the model
w = np.array([1.2, -0.05])
b = -1.0

# 1. Compute z using our vectorized dot product engine
z = np.dot(X, w) + b

# 2. Pass z through the sigmoid function to get probabilities
probabilities = sigmoid(z)

print("--- Malignancy Probabilities ---")
for idx, prob in enumerate(probabilities):
    prediction = 1 if prob >= 0.5 else 0
    print(f"Patient {idx}: Probability = {prob:.2%}, Final Prediction = {prediction}")
The Output You Will See:
Plaintext
--- Malignancy Probabilities ---
Patient 0: Probability = 18.24%, Final Prediction = 0
Patient 1: Probability = 88.29%, Final Prediction = 1
Patient 2: Probability = 17.65%, Patient Prediction = 0