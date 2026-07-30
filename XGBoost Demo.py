pip install xgboost scikit-learn matplotlib

import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 1. Load a standard binary classification dataset
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Split into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Initialize the XGBoost Classifier
# Key hyperparameters:
# - n_estimators: number of sequential trees built
# - learning_rate (eta): step size shrinkage used to prevent overfitting
# - max_depth: max depth per decision tree
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    eval_metric="logloss",
)

# 4. Train the model
model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=10,  # Prints evaluation progress every 10 trees
)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print("\n" + "=" * 40)
print(f"Test Accuracy: {accuracy:.4f}")
print("=" * 40 + "\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# 7. Plot Top 10 Feature Importances
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, max_num_features=10)
plt.title("XGBoost - Top 10 Feature Importances")
plt.tight_layout()
plt.show()