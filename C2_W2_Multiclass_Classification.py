"""
Multi-Class Classification with Neural Networks
------------------------------------------------
Demonstrates multi-class classification using TensorFlow/Keras,
the Adam Optimizer, numerical stability via linear logits, and a 
standalone NumPy Softmax implementation.

Course Reference: Deep Learning Specialization / Course 2 Week 2
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


def my_softmax(z):
    """
    Computes the softmax activation for a vector/matrix z in pure NumPy.
    Subtracts max(z) for numerical stability against exponent overflow.
    """
    z_stable = z - np.max(z, axis=-1, keepdims=True)
    ez = np.exp(z_stable)
    return ez / np.sum(ez, axis=-1, keepdims=True)


def main():
    # -------------------------------------------------------------------------
    # 1. Dataset Generation
    # -------------------------------------------------------------------------
    # Generate synthetic dataset with 4 distinct clusters (classes 0, 1, 2, 3)
    # 10 input features per sample
    NUM_CLASSES = 4
    X, y = make_blobs(
        n_samples=2000, 
        centers=NUM_CLASSES, 
        n_features=10, 
        cluster_std=1.5, 
        random_state=42
    )

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")
    print(f"Feature dimensions: {X_train.shape[1]} | Target classes: {NUM_CLASSES}\n")

    # -------------------------------------------------------------------------
    # 2. Model Architecture
    # -------------------------------------------------------------------------
    # Output layer uses 'linear' activation for numerical precision.
    # Softmax is computed directly inside the loss function.
    model = Sequential([
        Dense(units=32, activation='relu', name='hidden_layer_1'),
        Dense(units=16, activation='relu', name='hidden_layer_2'),
        Dense(units=NUM_CLASSES, activation='linear', name='output_layer')
    ], name="MultiClass_Classifier")

    model.summary()

    # -------------------------------------------------------------------------
    # 3. Model Compilation
    # -------------------------------------------------------------------------
    # SparseCategoricalCrossentropy handles integer target labels (0, 1, 2, 3).
    # from_logits=True optimizes calculation by avoiding small rounding errors.
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    # -------------------------------------------------------------------------
    # 4. Model Training
    # -------------------------------------------------------------------------
    print("\nStarting model training with Adam optimizer...")
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # -------------------------------------------------------------------------
    # 5. Model Evaluation
    # -------------------------------------------------------------------------
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n--- Evaluation Results ---")
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%\n")

    # -------------------------------------------------------------------------
    # 6. Inference & Custom Softmax Verification
    # -------------------------------------------------------------------------
    # Grab 3 sample test instances
    sample_X = X_test[:3]
    sample_y = y_test[:3]

    # Predict raw logits from the model
    raw_logits = model.predict(sample_X, verbose=0)

    # Convert logits to probabilities using TensorFlow
    tf_probs = tf.nn.softmax(raw_logits).numpy()

    # Convert logits to probabilities using custom NumPy Softmax
    numpy_probs = my_softmax(raw_logits)

    print("--- Inference Verification ---")
    for i in range(len(sample_X)):
        predicted_class = np.argmax(raw_logits[i])
        actual_class = sample_y[i]
        
        print(f"Sample {i + 1}:")
        print(f"  True Label:      {actual_class}")
        print(f"  Predicted Label: {predicted_class}")
        print(f"  Raw Logits (z):  {np.round(raw_logits[i], 3)}")
        print(f"  TF Probabilities:    {np.round(tf_probs[i], 4)}")
        print(f"  NumPy Probabilities: {np.round(numpy_probs[i], 4)}")
        print(f"  Probabilities Sum:   {np.sum(numpy_probs[i]):.1f}\n")


if __name__ == "__main__":
    main()