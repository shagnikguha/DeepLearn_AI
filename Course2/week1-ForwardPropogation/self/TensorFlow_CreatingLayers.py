import tensorflow as tf
import numpy as np

x = np.array([[2], [3], [4], [5], [6]])  # Convert to NumPy array

layer1 = tf.keras.layers.Dense(units=2, activation='sigmoid')
a1 = layer1(x.reshape(-1, 1))  # Reshape x to make it a 2D array

layer2 = tf.keras.layers.Dense(units=1, activation='sigmoid')
a2 = layer2(a1)

# creating sequence via declaring before sequence
model1 = tf.keras.Sequential([layer1, layer2])  

# declaring layers in the sequence
model2 = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(units=2, activation='sigmoid', name='Layer1'),
        tf.keras.layers.Dense(units=1, activation='sigmoid', name='Layer2')
    ]
)

f = model1.predict(x)  # Reshape x to make it a 2D array
f2 = model2.predict(x)
print("Output from model1:", f)
print("Output from model2:", f2)  


# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])         #sets up model for training
# model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))                 #trains the model
