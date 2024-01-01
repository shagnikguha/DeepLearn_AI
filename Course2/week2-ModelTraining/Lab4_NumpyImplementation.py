import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def my_softmax(z):  
    """ Softmax converts a vector of values to a probability distribution.
    Args:
      z (ndarray (N,))  : input data, N features
    Returns:
      a (ndarray (N,))  : softmax of z
    """    
    ### START CODE HERE ### 
    a = np.zeros_like(z)
    e_T = 0
    for zj in z:
        e_T += np.exp(zj)
    for i, Zin in enumerate(z):
        a[i] = np.exp(Zin)/e_T
    
    ### END CODE HERE ### 
    return a

z = np.array([1., 2., 3., 4.])
a = my_softmax(z)
atf = tf.nn.softmax(z)
print(f"my_softmax(z):         {a}")
print(f"tensorflow softmax(z): {atf}")

# UNQ_C2
# GRADED CELL: Sequential model
tf.random.set_seed(1234) # for consistent results
model = tf.keras.Sequential(
    [               
        ### START CODE HERE ### 
        tf.keras.layers.Dense(25, activation='relu', input_shape=(400,), name='layer1'),         # When specifing input size here itself, we are able to build to NN. If we don't know the input size, then we can use model.fit()
        tf.keras.layers.Dense(15, activation='relu', name='layer2'),
        tf.keras.layers.Dense(10, activation='linear', name='layer3'),
        ### END CODE HERE ### 
    ], name = "my_model" 
)

model.summary()