import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0, 0, 0, 1, 1, 1])
X_train2 = np.array([[0.5, 1.5], 
                     [1, 1], 
                     [1.5, 0.5], 
                     [3, 0.5], 
                     [2, 2], 
                     [1, 2.5]])
y_train2 = np.array([0, 0, 0, 1, 1, 1])

pos = y_train == 1
neg = y_train == 0

fig, ax = plt.subplots(1, 2, figsize=(8, 3))

# Plot 1, single variable
ax[0].scatter(x_train[pos], y_train[pos], marker='x', s=80, c='red', label="y=1")
ax[0].scatter(x_train[neg], y_train[neg], marker='o', s=100, label="y=0", facecolors='none',
              edgecolors='blue', lw=3)

ax[0].set_ylim(-0.08, 1.1)
ax[0].set_ylabel('y', fontsize=12)
ax[0].set_xlabel('x', fontsize=12)
ax[0].set_title('One Variable Plot')
ax[0].legend()

# Plot 2, two variables
ax[1].scatter(X_train2[y_train2 == 1][:, 0], X_train2[y_train2 == 1][:, 1], marker='x', s=80, c='red', label="y=1")
ax[1].scatter(X_train2[y_train2 == 0][:, 0], X_train2[y_train2 == 0][:, 1], marker='o', s=100, label="y=0",
              facecolors='none', edgecolors='blue', lw=3)

ax[1].axis([0, 4, 0, 4])
ax[1].set_ylabel('$x_1$', fontsize=12)
ax[1].set_xlabel('$x_0$', fontsize=12)
ax[1].set_title('Two Variable Plot')
ax[1].legend()

plt.tight_layout()
plt.show()
