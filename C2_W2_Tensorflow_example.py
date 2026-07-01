import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Building the exact Demand Forecasting model you saw yesterday
model = Sequential([
    Dense(units=3, activation='sigmoid'), # Hidden Layer 1 (Affordability, Awareness, Quality)
    Dense(units=1, activation='sigmoid')  # Output Layer (Final Purchase Prediction)
])