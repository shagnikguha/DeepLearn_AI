import numpy as np
import matplotlib.pyplot as  plt

def estimate_gaussian(X): 
    """
    Calculates mean and variance of all features 
    in the dataset
    
    Args:
        X (ndarray): (m, n) Data matrix
    """

    m, n = X.shape

    mu = np.array([])
    var = np.array([])
    for i in range(n):
        mut = 0
        for j in range(m):
            mut = mut + X[j][i]
        
        mut = mut/m
        mu = np.append(mu, mut)
        
        v = 0
        for j in range(m):
            v = v + ((X[j][i] - mut) ** 2)
        
        v = v/m
        var = np.append(var, v)
            
    return mu, var


# UNQ_C2
# GRADED FUNCTION: select_threshold

def select_threshold(y_val, p_val): 
    """
    Finds the best threshold to use for selecting outliers 
    based on the results from a validation set (p_val) 
    and the ground truth (y_val)
    
    Args:
        y_val (ndarray): Ground truth on validation set
        p_val (ndarray): Results on validation set
        
    Returns:
        epsilon (float): Threshold chosen 
        F1 (float):      F1 score by choosing epsilon as threshold
    """ 

    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    
    step_size = (max(p_val) - min(p_val)) / 1000
    
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
    
        ### START CODE HERE ### 
        predictions = p_val < epsilon
        
        tp = np.sum((y_val==1)&(predictions==1))
        fp = np.sum((y_val==0)&(predictions==1))
        fn = np.sum((y_val==1)&(predictions==0))
        
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
        
        F1 = (2*rec*prec)/(rec+prec)
        
        ### END CODE HERE ### 
        
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
        
    return best_epsilon, best_F1


# X_Train, X_val, y_cal = load_data()   # Loading the data set

'''
The first 5 elements of X_train are:
 [[13.04681517 14.74115241]
 [13.40852019 13.7632696 ]
 [14.19591481 15.85318113]
 [14.91470077 16.17425987]
 [13.57669961 14.04284944]]
The first 5 elements of X_val are
 [[15.79025979 14.9210243 ]
 [13.63961877 15.32995521]
 [14.86589943 16.47386514]
 [13.58467605 13.98930611]
 [13.46404167 15.63533011]]
The first 5 elements of y_val are
 [0 0 0 0 0]
'''