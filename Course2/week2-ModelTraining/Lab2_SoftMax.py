import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Generate random data for testing
np.random.seed(42)

# Assuming X_train is a 2D array with 100 samples and 5 features
X_train = np.random.randn(100, 5)

# Assuming y_train is a 1D array with class labels (0, 1, 2, or 3)
y_train = np.random.randint(0, 4, size=100)

def my_softmax(z):
    ez = np.exp(z)              #element-wise exponenial
    sm = ez/np.sum(ez)
    return(sm)

model = tf.keras.Sequential(
    [ 
        tf.keras.layers.Dense(25, activation = 'relu'),
        tf.keras.layers.Dense(15, activation = 'relu'),
        tf.keras.layers.Dense(4, activation = 'softmax')    # < softmax activation here
    ]
)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

model.fit(
    X_train,y_train,
    epochs=10
)

model.summary()

p_nonpreferred = model.predict(X_train)
print(p_nonpreferred [:2])
print("largest value", np.max(p_nonpreferred), "smallest value", np.min(p_nonpreferred))

#  The output ranges from 1 to 0, ranging from near 1 to extremely small near 0 values