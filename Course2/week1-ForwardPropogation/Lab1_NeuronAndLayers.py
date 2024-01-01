import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.linear_model
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

X_train = np.array([[1.0], [2.0]])  # (size in 1000 square feet)
Y_train = np.array([[300.0], [500.0]])  # (price in 1000s of dollars)

##################    LINEAR REGRESSION MODEL    ##################

# Define a linear layer in TensorFlow
linear_layer = tf.keras.layers.Dense(units=1, activation='linear')                      # creating and working with a single layer. To stack/sequence multiple layers, check logisitc portion

# Perform a forward pass on all input examples
predictions_tf = linear_layer(X_train)                                                  # Before putting our custom weights, we need to build our layer

set_w = np.array([[200]])
set_b = np.array([100])

# set_weights takes a list of numpy arrays
linear_layer.set_weights([set_w, set_b])
print(linear_layer.get_weights())

# Perform a forward pass on all input examples
predictions_tf = linear_layer(X_train)

# Extract weights and bias from the linear layer
w, b = linear_layer.get_weights()
print(f"TensorFlow Model - w = {w}, b = {b}")

# Create and fit a scikit-learn SGDRegressor model
lr_model = sklearn.linear_model.SGDRegressor(max_iter=1000)
lr_model.fit(X_train, Y_train)
predictions_sklearn = lr_model.predict(X_train)

# Plot the results
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Plot TensorFlow predictions
ax.scatter(X_train, Y_train, marker='x', c='r', label="Data Points")
ax.plot(X_train, predictions_tf, label="TensorFlow Predictions")

# Plot scikit-learn SGDRegressor predictions
ax.plot(X_train, predictions_sklearn, label="scikit-learn Predictions")

ax.legend(fontsize='xx-large')
ax.set_ylabel('Price (in 1000s of dollars)', fontsize='xx-large')
ax.set_xlabel('Size (1000 sqft)', fontsize='xx-large')
ax.set_title('Comparison of TensorFlow and scikit-learn Predictions')
plt.show()




##################    LOGISTIC REGRESSION MODEL    ##################
X_train = np.array([[0., 1, 2, 3, 4, 5]]).reshape(-1,1)  # 2-D Matrix with one column and multiple rows
Y_train = np.array([[0,  0, 0, 1, 1, 1]]).reshape(-1,1)  # 2-D Matrix with one column and multiple rows

pos = Y_train == 1
neg = Y_train == 0

model = tf.keras.Sequential(                                                                        # Automatically builds the NN when initializing
    [
        tf.keras.layers.Dense(1, input_dim=1,  activation = 'sigmoid', name='L1')
    ]
)

model.summary()

logistic_layer = model.get_layer('L1')

'''
w,b = logistic_layer.get_weights()
print(w,b)
print(w.shape,b.shape)
'''

set_w = np.array([[2]])
set_b = np.array([-4.5])
# set_weights takes a list of numpy arrays
logistic_layer.set_weights([set_w, set_b])
print(logistic_layer.get_weights())

predictions_tf = model.predict(X_train)
# print(predictions_tf)

lr_model = sklearn.linear_model.LogisticRegression(max_iter = 1000)
lr_model.fit(X_train, Y_train)
predictions_sklearn = lr_model.predict(X_train)

print(predictions_tf)

for v in range (len(predictions_tf)):
    if predictions_tf[v] > 0.5:
        predictions_tf[v] = 1
    else:
        predictions_tf[v] = 0

fig,ax = plt.subplots(1,1,figsize=(4,3))
ax.scatter(X_train[pos], Y_train[pos], marker='x', s=80, c = 'red', label="y=1")
ax.scatter(X_train[neg], Y_train[neg], marker='o', s=100,c = 'blue', label="y=0")

ax.plot(X_train, predictions_tf, label="TensorFlow Predictions")
ax.plot(X_train, predictions_sklearn, label="scikit-learn Predictions")

ax.set_ylim(-0.08,1.1)
ax.set_ylabel('y', fontsize=12)
ax.set_xlabel('x', fontsize=12)
ax.set_title('one variable plot')
ax.legend(fontsize=12)
plt.show()
