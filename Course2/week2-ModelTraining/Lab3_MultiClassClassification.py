import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

X_train = np.array([])
y_train = np.array([])

tf.random.set_seed(1234)  # applied to achieve consistent results
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(2, activation = 'relu',   name = "L1"),
        tf.keras.layers.Dense(4, activation = 'linear', name = "L2")
    ]
)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01),
)

model.fit(
    X_train,y_train,
    epochs=200
)

# After training, get the raw logits for the examples
logits = model.predict(X_train)

# Apply sigmoid activation to get probabilities
probabilities = tf.nn.sigmoid(logits).numpy()

# Threshold probabilities to get binary predictions
binary_predictions = (probabilities >= 0.5).astype(int)

print("Binary Predictions:", binary_predictions)
