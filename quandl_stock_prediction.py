# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#this program predicts stock prices by using machine learning models

#install the dependencies
import quandl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split


# %%
#get the stock data
df = quandl.get("WIKI/MSFT")
#take a look at the data
print(df.head())


# %%
#get the adjusted close price
df= df[['Adj. Close']]
#take a look at the new data
print(df.head())


# %%
#a variable for predicting 'n' days out into the future #12:41 build a stock prediction program youtube video
forecast_out = 30
#create another column (the target or dependent variable) shifted 'n' units up
df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)
#print the new data set
print(df.head())
#print the new data set
print(df.tail())


# %%
#Create the independent data set x 
#convert the dataframe to a numpy array
X = np.array(df.drop(['Prediction'],1))
#remove the last 'n' rows
X = X[:-forecast_out]
print(X)


# %%
#create the dependent data set (y)
#convert the dataframeto a numpy array(All of the values including the nan's)
y = np.array(df['Prediction'])
#get all of the y values except the last 'n' rows
y = y[:-forecast_out]
print(y)


# %%
#split the data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2)


# %%
#create and train the support vector machine (regressor)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(x_train,y_train)


# %%
#testing model: score returns the coefficient of determination r^2 of the prediction
#The best possible score is 1.0
svm_confidence = svr_rbf.score(x_test, y_test)
print("svm confidence: ", svm_confidence)


# %%
#create and train the linear regression model
lr = LinearRegression()
#train the model
lr.fit(x_train, y_train)


# %%
#testing model: score returns the coefficient of determination r^2 of the prediction
#The best possible score is 1.0
lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence)


# %%
#set x_forecast equal to the last 30 rows of the original data set from Adj. Close
x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
print(x_forecast)


# %%
#print the linear regression model prediction for the 'n' days
lr_prediction = lr.predict(x_forecast)
print(lr_prediction)

#print the support regressor model prediction for the 'n' days
svm_prediction = svr_rbf.predict(x_forecast)
print(lr_prediction)


# %%



