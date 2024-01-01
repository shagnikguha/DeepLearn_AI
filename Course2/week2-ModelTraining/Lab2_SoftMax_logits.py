import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

X_train = np.array([])
y_train = np.array([])

def my_softmax(z):
    ez = np.exp(z)              #element-wise exponenial
    sm = ez/np.sum(ez)
    return(sm)

model = tf.keras.Sequential(
    [ 
        tf.keras.layers.Dense(25, activation = 'relu'),
        tf.keras.layers.Dense(15, activation = 'relu'),
        tf.keras.layers.Dense(4, activation = 'linear')    # <-- To facilitate logits
    ]
)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),    # more stable and accurate results can be obtained if the softmax and loss are combined during training
    optimizer=tf.keras.optimizers.Adam(0.001),                          
)

'''Note to self-->same can be done for sigmoid.  tf.keras.losses.BinaryCrossentropy(from_logits=True)'''

model.fit(
    X_train,y_train,
    epochs=10
)

p_preferred = model.predict(X_train)
print(p_preferred [:2])
print("largest value", np.max(p_preferred), "smallest value", np.min(p_preferred))

# ranges from extremely large +ve values to even -ve numbers

sm_preferred = tf.nn.softmax(p_preferred).numpy()               # sigmoid instead of softmax if doing bianry classification
print(f"two example output vectors:\n {sm_preferred[:2]}")
print("largest value", np.max(sm_preferred), "smallest value", np.min(sm_preferred))

predicted_classes = np.argmax(sm_preferred, axis=1)
print("Predicted Classes:", predicted_classes)