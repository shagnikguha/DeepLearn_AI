import numpy as np
import tensorflow as tf
from tensorflow import keras

# we will be using 2 matrices of same size. R contains data for checking if movie has been rated or not. Y contains actual rating data
# Column = users, Row = movies

# dataset = https://grouplens.org/datasets/movielens/latest/


# GRADED FUNCTION: cofi_cost_func
# UNQ_C1

def cofi_cost_func(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    nm, nu = Y.shape
    J = 0
    # Compute the prediction
    prediction = X.dot(W.T) + b
    # Compute the error
    error = (prediction - Y) * R
    # Compute the squared error cost
    J_error = 0.5 * np.sum(np.square(error))
    # Compute the regularization terms
    J_regX = 0.5 * lambda_ * np.sum(np.square(X))
    J_regW = 0.5 * lambda_ * np.sum(np.square(W))
    # Total cost
    J = J_error + J_regX + J_regW

    return J