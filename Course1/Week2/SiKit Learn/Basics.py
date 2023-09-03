import numpy as np
import matplotlib.pyplot as plt                     #only imorting the pyplot function to plot graphs. You can use "import matplotlib as plt", but will jave to use plt.pyplot.scatter("")
from sklearn.linear_model import SGDRegressor       #Gradient Descnet model
from sklearn.preprocessing import StandardScaler    #Feature Normalization package


X_train = np.array([[2104, 5, 1, 45],   #goes like x(j=0,i=0)|x(j=1,i=0).....   where j = jth feature
                    [1416, 3, 2, 40],   #          x(j=0,i=1)|x(j=1,i=1).....   where i = ith training example
                    [852, 2, 1, 35]])
Y_train = np.array([1000, 600, 400])
X_features = ['size(sqft)','bedrooms','floors','age']


#normalizing the data set(bringing all parameters into a copmarable range so that gradient descent becomes more effictient and gives us a more accurate prediction)
print("Normalizing")
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")


#Using liner Regression model
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, Y_train)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

#getting the values of w and b parameters
b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")



# make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(X_norm)
'''
# make a prediction using w,b. 
y_pred = np.dot(X_norm, w_norm) + b_norm  
print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")
'''
print(f"Prediction on training set:\n{y_pred_sgd}" )
print(f"Target values \n{Y_train}")


# plot predictions and targets vs original features    
fig,ax=plt.subplots(1,4,figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],Y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],y_pred_sgd,color="orange", label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()