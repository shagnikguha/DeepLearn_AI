import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

"""Lecture Example"""
X = np.array([[ 99,  -1],
       [ 98,  -1],
       [ 97,  -2],
       [101,   1],
       [102,   1],
       [103,   2]])

plt.plot(X[:,0], X[:,1], 'ro')
plt.show()
# Loading the PCA algorithm
pca_2 = PCA(n_components=2)
print(pca_2)

# Let's fit the data. We do not need to scale it, since sklearn's implementation already handles it.
pca_2.fit(X)
print(pca_2.explained_variance_ratio_)

X_trans_2 = pca_2.transform(X)
print(X_trans_2)

pca_1 = PCA(n_components=1)
print(pca_1)

pca_1.fit(X)
print(pca_1.explained_variance_ratio_)

X_trans_1 = pca_1.transform(X)
print(X_trans_1)
