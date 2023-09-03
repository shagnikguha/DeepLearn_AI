import numpy as np
import matplotlib as plt

X_train = np.array([[2104, 5, 1, 45],   #goes like x(j=0,i=0)|x(j=1,i=0).....   where j = jth feature
                    [1416, 3, 2, 40],   #          x(j=0,i=1)|x(j=1,i=1).....   where i = ith training example
                    [852, 2, 1, 35]])
X_features = ['size(sqft)','bedrooms','floors','age']

def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column                   (In a chart, the column represents the i=0-m region)
    
    Args:
      X (ndarray (m,n))     : input data, m examples, n features
      
    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    # find the mean of each column/feature
    mu     = np.mean(X, axis=0)                 # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma  = np.std(X, axis=0)                  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma      

    return (X_norm, mu, sigma)                        
#X was a 2D array. Mu and sigma was the mean and mean-deviation for each feature respectively(the n-columns) When finding X_norm, X is subtracted by mu and sigma. Meaning that eacch column of x was being modified by the other vector of size (0,n)
 
#check our work
#from sklearn.preprocessing import scale
#scale(X_orig, axis=0, with_mean=True, with_std=True, copy=True)
