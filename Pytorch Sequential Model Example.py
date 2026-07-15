import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. GENERATE SYNTHETIC COFFEE ROASTING DATA
# Features: [Temperature in Celsius, Duration in Minutes]
X_train = np.array([
    [185.0, 12.0], [200.0, 14.0], [210.0, 15.0], [170.0, 9.0],  # Good roasts
    [150.0, 5.0],  [240.0, 20.0], [190.0, 2.0],  [220.0, 6.0]   # Bad roasts (under/over)
])
# Targets: 1 = Delicious Roast, 0 = Bad Roast
y_train = np.array([1, 1, 1, 1, 0, 0, 0, 0])

# 2. NORMALIZATION LAYER (Crucial for neural networks to converge quickly)
# This scales features so temperature and duration are on similar scales
normalization_layer = tf.keras.layers.Normalization(axis=-1)
normalization_layer.adapt(X_train)
X_train_norm = normalization_layer(X_train)

# 3. DEFINE THE MODEL ARCHITECTURE (Step 1 of the training loop)
# Standard hidden layer structure with ReLU, and a 1-unit Sigmoid output for binary choice
model = Sequential([
    Dense(units=3, activation='relu', name='hidden_layer1'),
    Dense(units=1, activation='sigmoid', name='output_layer')
])

# 4. COMPILE THE MODEL (Step 2 of the training loop)
# 'adam' automated optimizer + 'binary_crossentropy' loss for 0/1 classification
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.BinaryCrossentropy()
)

# 5. FIT THE MODEL TO DATA (Step 3 of the training loop)
# Epochs = number of full passes through the dataset during gradient descent
print("--- Starting Training ---")
model.fit(X_train_norm, y_train, epochs=200, verbose=0) # verbose=0 keeps terminal clean
print("Training Complete!\n")

# 6. MAKE A PREDICTION ON NEW DATA
# Let's test a new batch: 205 degrees Celsius for 13.5 minutes
X_new = np.array([[205.0, 13.5]])
X_new_norm = normalization_layer(X_new)

prediction = model.predict(X_new_norm)
print(f"\nRaw Sigmoid Probability Output: {prediction[0][0]:.4f}")

# Threshold the probability to get a strict 0 or 1 decision
if prediction >= 0.5:
    print("Decision: This will be a delicious batch of coffee! ☕")
else:
    print("Decision: Alert! Bad batch (Under or Over-roasted).")