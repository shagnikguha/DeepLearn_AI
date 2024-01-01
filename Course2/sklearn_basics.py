import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor                   # Gradient Descent model
from sklearn.preprocessing import StandardScaler                # Feature Normalization package
from sklearn.linear_model import LogisticRegression             # Logistic Regression model
from sklearn.model_selection import train_test_split            # Package to split data into training and validation sets
from sklearn.metrics import mean_squared_error, accuracy_score

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y_linear = 4 + 3 * X + np.random.randn(100, 1)
y_logistic = (X > 1).astype(int).ravel()

# Normalize the data
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train_linear, y_test_linear, y_train_logistic, y_test_logistic = train_test_split(
    X_normalized, y_linear, y_logistic, test_size=0.2, random_state=42
)

# Linear Regression
linear_reg_model = SGDRegressor()
linear_reg_model.fit(X_train, y_train_linear)

# Logistic Regression
logistic_reg_model = LogisticRegression()
logistic_reg_model.fit(X_train, y_train_logistic)

# Make predictions
y_pred_linear = linear_reg_model.predict(X_test)
y_pred_logistic = logistic_reg_model.predict(X_test)

# Evaluate the models
mse = mean_squared_error(y_test_linear, y_pred_linear) / 2
accuracy = accuracy_score(y_test_logistic, y_pred_logistic)

fig, ax = plt.subplots(1, 2, figsize=(10,4))

# Linear Regression
ax[0].scatter(X_test, y_test_linear, color='black', label='Actual')
ax[0].plot(X_test, y_pred_linear, color='blue', label='Linear Regression')
ax[0].set_title(f'Linear Regression\nMSE: {mse:.2%}')
ax[0].set_xlabel('Normalized X')
ax[0].set_ylabel('y')
ax[0].legend()

# Logistic Regression
ax[1].scatter(X_test, y_test_logistic, color='black', s=80, label='Actual')
ax[1].scatter(X_test, y_pred_logistic, color='red', s=50, label='Logistic Regression')
ax[1].set_title(f'Logistic Regression\nAccuracy: {accuracy:.2%}')
ax[1].set_xlabel('Normalized X')
ax[1].set_ylabel('y')
ax[1].legend()

plt.tight_layout()
plt.show()
