# --- Quick Scale Check ---
# If your dataset X is a NumPy array:
print(f"Max size: {np.max(X_train[:, 0])}") # e.g., 2000
print(f"Min size: {np.min(X_train[:, 0])}") # e.g., 300

# If the gap between min and max is massive, 
# you MUST normalize before running the gradient loop!