'''
This a very basic code for theory. There is a better version used in industry
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

model = tf.keras.Sequential(
    [
        # tf.keras.Input(shape = (X.shape[1],)),
        tf.keras.layers.Dense(25, activation="relu", name="layer1"),
        tf.keras.layers.dense(15, activation="relu", name="layer2"),
        tf.keras.layers.Dense(10, activation="softmax", name="layer3")
    ]
)

model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy() 
)

#model.fit(X, Y, epoch = 10)