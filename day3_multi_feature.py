import numpy as np

# 1. Training Data Matrix (4 houses, 4 features each)
# Features: [Size (sqft), Bedrooms, Floors, Age (years)]
X_train = np.array([
    [2104, 5, 1, 45],
    [1416, 3, 2, 40],
    [852, 2, 1, 35],
    [1985, 4, 2, 12]
])

# Actual prices in $1,000s
y_train = np.array([460, 232, 178, 520])

# 2. Model Parameters (Weights for each of the 4 features + 1 base bias)
w_vec = np.array([0.4, 15.0, -5.0, -1.2])
b_scalar = 80.0

print("Calculating predictions using vectorization...")
print("---------------------------------------------")

# 3. The Vectorized Prediction Function
def predict_single_house(x, w, b):
    """Calculates prediction for one house using dot product"""
    return np.dot(x, w) + b

# 4. Loop through our houses and print the AI's guesses vs Reality
for i in range(len(X_train)):
    single_house_features = X_train[i]
    prediction = predict_single_house(single_house_features, w_vec, b_scalar)
    actual = y_train[i]
    error = prediction - actual
    
    print(f"House {i+1}: Specs {single_house_features}")
    print(f"   -> AI Predicted Price: ${prediction:6.1f}K")
    print(f"   -> Actual Price:       ${actual:6.1f}K")
    print(f"   -> Error:              ${error:6.1f}K\n")

print("---------------------------------------------")
# 5. Matrix operation: Calculate ALL predictions at once in a single hardware operation!
all_predictions = np.dot(X_train, w_vec) + b_scalar
print(f"All Vectorized Predictions Matrix: {all_predictions}")