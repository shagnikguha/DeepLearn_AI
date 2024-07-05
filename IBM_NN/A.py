import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Reading csv file
data = pd.read_csv('concrete_data.csv')

# Splitting into features and target
features = data.drop(columns=['Strength']).copy()   # Dropping 'Strength' table and copying the rest
target = data[['Strength']].copy()                  # Only copying the last comlumn

# converting pd to np
features = features.to_numpy()
target = target.to_numpy()

mse_list = []

scaler = StandardScaler()
features = scaler.fit_transform(features)

for i in range(50):
    print(f"Iteration: {i}")
    # Splitting the data set
    X_train, X_test, Y_train, Y_test = train_test_split(
        features, target, test_size=0.3, random_state=40
    )

    size = features.shape[1]                            # number of features

    # creating model with hidden layer and output layer
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(units=10, activation='relu', name='layer1'),
            tf.keras.layers.Dense(units=10, activation='relu', name='layer2'),
            tf.keras.layers.Dense(units=10, activation='relu', name='layer3'),
            tf.keras.layers.Dense(units=1, activation='linear', name='layer4')
        ]
    )

    # compiling model with adam optimizer and mean-squared-error loss
    model.compile(
        loss = tf.keras.losses.MeanSquaredError(),
        optimizer = tf.keras.optimizers.Adam(0.001)
    )

    # training model
    model.fit(
        X_train,
        Y_train,
        epochs=100
    )

    Y_pred = model.predict(X_test)                      # getting prediction

    mse = mean_squared_error(Y_test, Y_pred)            # getting mse

    mse_list.append(mse)

# getting mean and standard deviation
mean = np.mean(mse_list)
std_dev = np.std(mse_list)

# printing results
print(f"mse_list: {mse_list}")
print(f"Mean: {mean}, Standard Deviation: {std_dev}")


