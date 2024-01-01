import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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

pos = Y.flatten() == 1
neg = Y.flatten() == 0

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.scatter(X[pos, 0], X[pos, 1], marker='x', s=100, c='red', label="y=1")
ax.scatter(X[neg, 0], X[neg, 1], marker='o', s=100, c='blue', label="y=0")
ax.set_ylabel('y', fontsize=12)
ax.set_xlabel('x', fontsize=12)
ax.legend(fontsize=12)
plt.show()

print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X)
print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")


def sigmoid(z):
    g = 1/(1+np.exp(-z))
   
    return g

def my_dense(a_in, W, b):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example 
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units  
    Returns
      a_out (ndarray (j,))  : j units|
    """
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):               
        w = W[:,j]                                    
        z = np.dot(w, a_in) + b[j]         
        a_out[j] = sigmoid(z)       

    return(a_out)


def my_dense_v(A_in, W, b, g):
    """
    Computes dense layer
    Args:
      A_in (ndarray (m,n)) : Data, m examples, n features each
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (1,j)) : bias vector, j units  
      g    activation function (e.g. sigmoid, relu..)
    Returns
      A_out (tf.Tensor or ndarray (m,j)) : m examples, j units
    """
    # incase A_in is (n.m) in size, do A_in.T to get transpose
    z = np.matmul(A_in, W) + b
    A_out = g(z)

    return(A_out)
    

def my_sequential(x, W1, b1, W2, b2):
    a1 = my_dense(x,  W1, b1)
    a2 = my_dense(a1, W2, b2)
    return(a2)

def my_predict(X, W1, b1, W2, b2):
    m = X.shape[0]
    p = np.zeros((m,1))
    for i in range(m):
        p[i,0] = my_sequential(X[i], W1, b1, W2, b2)
    return(p)