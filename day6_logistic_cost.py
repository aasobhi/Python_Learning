import numpy as np

# 1. True Labels (e.g., 0 = Not Spam, 1 = Spam)
y_train = np.array([1, 0, 1, 0])

# 2. AI Model's Predicted Probabilities (How confident the AI is)
# Case A: A well-trained model (Confident and mostly correct)
predictions_good = np.array([0.92, 0.05, 0.88, 0.12])

# Case B: A terrible model (Confident but completely wrong)
predictions_bad = np.array([0.05, 0.95, 0.10, 0.90])

# 3. The Combined Log Loss Cost Function
def compute_logistic_cost(y, pred):
    m = len(y)
    # This single line implements the combined log loss math for the whole matrix
    loss = -y * np.log(pred) + (1-y) * np.log(1 - pred)
    total_cost = (1 / m) * np.sum(loss)
    return total_cost

# 4. Calculate and compare the costs
cost_good = compute_logistic_cost(y_train, predictions_good)
cost_bad = compute_logistic_cost(y_train, predictions_bad)

print("Analyzing Classification Cost (Log Loss)...")
print("--------------------------------------------")
print(f"Good Model Predictions: {predictions_good}")
print(f"-> Total Calculated Cost: {cost_good:.4f} (Very low, close to 0)")
print("")
print(f"Bad Model Predictions:  {predictions_bad}")
print(f"-> Total Calculated Cost: {cost_bad:.4f} (Massive penalty!)")
print("--------------------------------------------")
print("The math heavily penalizes the model for being confidently wrong.")