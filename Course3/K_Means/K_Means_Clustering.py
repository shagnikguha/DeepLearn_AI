import numpy as np
import matplotlib as plt

# K-means algorithm is a method to automatically cluster similar data points together.


def find_closest_centroids(X, centroids):
      # Set K
    K = centroids.shape[0]

    idx = np.zeros(X.shape[0], dtype=int)

    for i in range(X.shape[0]):
        min_dist = float('inf')
        for j in range(K):
            dist = np.linalg.norm(X[i] - centroids[j])
            if dist<min_dist:
                min_dist = dist
                idx[i] = j

    return idx

def compute_centroids(X, idx, K):
    # Useful variables
    m, n = X.shape
    
    # You need to return the following variables correctly
    centroids = np.zeros((K, n))
    '''
    # longer method
    for i in range(K):
        summation = 0
        count = 0
        
        for j in range(idx):
            if idx[j] == i:
                count = count + 1
                summation = summation + X[j]
        
        centroids[i] = summation/count
    '''

    for i in range(K):
        points = X[idx == i]

        centroids[i] = np.mean(points, axis=0)
    return centroids

def kMeans_init_centroids(X, K):
    """
        X (ndarray): Data points 
        K (int):     number of centroids/clusters
    """
    
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    
    # Take the first K examples as centroids
    centroids = X[randidx[:K]]
    
    return centroids

def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """
    
    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids    
    idx = np.zeros(m)
    plt.figure(figsize=(8, 6))

    # Run K-Means
    for i in range(max_iters):
        
        #Output progress
        print("K-Means iteration %d/%d" % (i, max_iters-1))
        
        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)
        
        # Optionally plot progress
        # if plot_progress:
        #     plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
        #     previous_centroids = centroids
            
        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)
    plt.show() 
    return centroids, idx

K = 3
max_iters = 10

X = np.array([1.84207953, 4.6075716 ],
             [5.65858312, 4.79996405],
             [6.35257892, 3.2908545 ],
             [2.90401653, 4.61220411],
             [3.23197916, 4.93989405])
# Set initial centroids by picking random examples from the dataset
initial_centroids = kMeans_init_centroids(X, K)

# Run K-Means
centroids, idx = run_kMeans(X, initial_centroids, max_iters, plot_progress=True)
# First five elements of X are:
#  [[1.84207953 4.6075716 ]
#  [5.65858312 4.79996405]
#  [6.35257892 3.2908545 ]
#  [2.90401653 4.61220411]
#  [3.23197916 4.93989405]]
# centroid = [3,3], [6,2], [8,5]]