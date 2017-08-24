# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]  # the libraries actually do take care of this but this is a reminder

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

print(y_pred)
print(y_test)

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
# need to add column of 1s because statsmodels does not take into account b0
# y = b0x0 + b1x1 + b2x2 + b3x3, x0 = 1
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# step 2
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
# step 3
print(regressor_OLS.summary())
# step 4
X_opt = X[:, [0, 1, 3, 4, 5]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())
X_opt = X[:, [0, 3, 4, 5]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())
X_opt = X[:, [0, 3, 5]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())
X_opt = X[:, [0, 3]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())

# Simple Linear Regression just using R&D
regressor = LinearRegression()
regressor.fit(X_train[:, [2]].reshape(-1, 1), y_train)

# Visualizing the Training set results
plt.scatter(X_train[:, 2], y_train, c='red')
plt.plot(X_train[:, 2], regressor.predict(X_train[:, 2].reshape(-1, 1)), c='blue')
plt.title('Profit vs R&D Spend (Training Set)')
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.show()