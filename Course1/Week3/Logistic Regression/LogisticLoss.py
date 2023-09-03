import numpy as np
import matplotlib.pyplot as plt

X_train = np.array([[0.5, 1.5], 
                    [1, 1], 
                    [1.5, 0.5], 
                    [3, 0.5], 
                    [2, 2], 
                    [1, 2.5]])                                                   #(m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])                                           #(m,)

fig,ax = plt.subplots(1,1,figsize=(4,4))
ax.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], marker='x', s=80, c='red', label="y=1")
ax.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], marker='o', s=100, c='blue', label="y=0")

# Set both axes to be from 0-4
ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.show()

def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
         
    """

    g = 1/(1+np.exp(-z))
   
    return g

def compute_cost_logistic(X, y, w, b):
    """
    Computes cost

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """

    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
             
    cost = cost / m
    return cost


w_tmp = np.array([1,1])
b_tmp = -3
print(compute_cost_logistic(X_train, y_train, w_tmp, b_tmp))
