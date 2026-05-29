# Check if the change is smaller than 0.001
if abs(previous_cost - current_cost) < 0.001:
    print("Converged! Stopping training.")
    break