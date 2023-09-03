import numpy as np
import matplotlib as plt

#             lenght  wight  area
X = np.array([[30, 30, 900],
             [20, 30, 600],
             [10, 10, 100]])
Y = np.array([1000, 600, 400])

W = np.array([2, 1, 1])
b = 1

def cost_cal(X, Y, w, b):
    cost = 0
    m = X.shape[0]
    for i in range(m):
        cost += (((np.dot(X[i],w) + b)-Y[i])**2)
    cost = cost/(2*m)
    return cost

cost = cost_cal(X, Y, W, b)
print(cost)
