import numpy as np
import matplotlib.pyplot as plt

  
x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2]) #(size in 1000 square feet)
y_train = np.array([250, 300, 480,  430,   630, 730,]) #(price in 1000s of dollars)



def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b   # (y = w*x + b) LINEAR REGRESSION FORMULA
        
    return f_wb

def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    cost_sum = 0 
    for i in range(m): 
        f_wb = w * x[i] + b   
        cost = (f_wb - y[i]) ** 2  
        cost_sum = cost_sum + cost  
    total_cost = (1 / (2 * m)) * cost_sum  

    return total_cost

w = 150
b = 100
print(f"w: {w}")
print(f"b: {b}")

#plotting of predicted model(Y-hat line)
tmp_f_wb = compute_model_output(x_train, w, b,)
tmp_j_wb = compute_cost(x_train, y_train, w, b)
# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')
# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

print(f"costfunctionvalue = {tmp_j_wb}")

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()