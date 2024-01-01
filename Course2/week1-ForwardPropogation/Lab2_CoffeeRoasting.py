import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[185.32, 12.69],
              [259.92, 11.87],
              [231.01, 14.41],
              [175.37, 11.72],
              [187.12, 14.13],
              [225.91, 12.1 ],
              [208.41, 14.18],
              [207.08, 14.03],
              [280.6 , 14.23],
              [202.87, 12.25],
              [196.7 , 13.54],
              [270.31, 14.6 ],
              [192.95, 15.2 ],
              [213.57, 14.28],
              [164.47, 11.92],
              [177.26, 15.04],
              [241.77, 14.9 ],
              [237.  , 13.13],
              [219.74, 13.87],
              [266.39, 13.25]])

Y = np.array([[1.],
              [0.],
              [0.],
              [0.],
              [1.],
              [1.],
              [0.],
              [0.],
              [0.],
              [1.],
              [1.],
              [0.],
              [0.],
              [0.],
              [0.],
              [0.],
              [0.],
              [0.],
              [0.],
              [0.]])

print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
norm_l = tf.keras.layers.Normalization(axis=-1)    # Creating normalization layer.  Axis = -1 means to apply it on the last axis of data. Column in this case
norm_l.adapt(X)  # learns mean, variance of each feature 
Xn = norm_l(X)   # applies normalization on original list
print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")


# increasing size of data set by duplicating list to reduce number of epochs
Xt = np.tile(Xn,(1000,1))
Yt= np.tile(Y,(1000,1))   
print(Xt.shape, Yt.shape)   

tf.random.set_seed(1234)  # applied to achieve consistent results
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(2,)),                 # Defines input layer which will accept data set having 2 features
        tf.keras.layers.Dense(3, activation='sigmoid', name = 'layer1'),
        tf.keras.layers.Dense(1, activation='sigmoid', name = 'layer2')
    ]
)

'''
[layer1, layer2] = model.layers
print(model.layers[1].weights)
'''
# model.summary()

W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)

# Setting up model for training
model.compile(      
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)

# training model with the normalized and expanded X_train and Y_train
model.fit(
    Xt,Yt,            
    epochs=10,                              # number of steps in gradient descent
)

W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print("W1:\n", W1, "\nb1:", b1)
print("W2:\n", W2, "\nb2:", b2)

# Setting set values as given in the tutorial for comparison
W1 = np.array([
    [-8.94,  0.29, 12.89],
    [-0.17, -7.34, 10.79]] )
b1 = np.array([-9.87, -9.28,  1.01])
W2 = np.array([
    [-31.38],
    [-27.86],
    [-32.79]])
b2 = np.array([15.54])
model.get_layer("layer1").set_weights([W1,b1])
model.get_layer("layer2").set_weights([W2,b2])

# testing 
X_test = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example
X_testn = norm_l(X_test)        # normalizing the values
predictions = model.predict(X_testn)
print("predictions = \n", predictions)

yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decisions = \n{yhat}")