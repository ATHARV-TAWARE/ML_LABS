# import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# read file 
df = pd.read_csv("file_name.csv") # file in same folder
print(df.head())

np.random.seed(42)
n_samples = len(df)
X = df["hours"].values
y = df["marks"].values

# split data into training and testing set 


X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1, 1),y,test_size=0.2, random_state=42)
print("training data: ", X_train.size)
print("training data: ", X_test.size)

# train the linear regresssion Model

model = LinearRegression()
model.fit(X_train, y_train)
print("slope: ", model.coef_.round(2))
print("intercept: ", model.intercept_.round(2))


# predict mse, rmse, r2

y_pred = model.predict(X_test)
mse= mean_squared_error(y_test, y_pred)
rmse= np.sqrt(mse)
r2= r2_score(y_test, y_pred)
print("mse: ", mse)
print("rmse: ", rmse)
print("r2: ", r2)

# visualize the regression line

plt.scatter(X_train, y_train, label="training data")
plt.scatter(X_test, y_test, label="testing data")
x_line= np.linspace(X.min(),X.max(), 100).reshape(-1,1)
y_line= model.predict(x_line)
plt.plot(x_line, y_line, color="red", label= " linear regression")
plt.xlabel("hours")
plt.ylabel("marks")
plt.title("linear_regression")
plt.legend()
plt.show()