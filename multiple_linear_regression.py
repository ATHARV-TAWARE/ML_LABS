# import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# read file 
df = pd.read_csv("file_name.csv") # file in same folder
print(df.head())

# Generate dateset

np.random.seed(42)
n_samples = 500

sqft = np.random.normal(2100, 500, n_samples).clip(800, 4500)
bedrooms = np.random.choice([1, 2, 3, 4, 5], n_samples)
dist_transit = np.random.exponential(4.0, n_samples).clip(0.5, 20)
age = np.random.uniform(1, 40, n_samples)
school_rating = np.random.uniform(3.0, 9.8, n_samples)

noise = np.random.normal(0, 25000, n_samples)

price = (
    150 * sqft + 12000 * bedrooms - 8500 * dist_transit
    - 1500 * age + 9000 * school_rating + 50000 + noise
)

df = pd.DataFrame({
    "Square_Footage": sqft,
    "Number_of_Bedrooms": bedrooms,
    "Distance_to_Transit_km": dist_transit,
    "Age_of_Property": age,
    "School_Rating": school_rating,
    "House_Price": price
})

print(df.head())

# split the dataset

x = df[[
    "Square_Footage", "Number_of_Bedrooms", "Distance_to_Transit_km",
     "Age_of_Property", "School_Rating"
]]

y = df["House_Price"]
x_test, x_train, y_test, y_train = train_test_split(x, y , test_size=0.2, random_state=42)
print("training set shpe: ", x_train.shape)
print("testing set shpe: ", x_test.shape)

# train model

model = LinearRegression()
model.fit(x_train, y_train)
print("intercept: ", model.intercept_)
coefficient = pd.DataFrame({
    "features" : x.columns,
    "coefficient": model.coef_
})

print("coefficients: ", coefficient.round(2))

# predict and evaluate performance

y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
n = len(y_test)
p = x_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
print("Adjusted R2 Score:", adj_r2)

# residual and VIF check

residual = y_test - y_pred

vif_data = pd.DataFrame()
vif_data["features"] = x.columns
vif_data["VIF"] = [
    variance_inflation_factor(x.values, i)
    for i in range(x.shape[1])
]
print("VIF values: ")
print(vif_data.round(2))

# detailed OLS summery

x_train_sm =sm.add_constant(x_train)
ols_model = sm.OLS(y_train, x_train_sm).fit()
print(ols_model.summary())

# residual plot

#%% Cell 8
plt.scatter(y_pred, residual)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted House Price")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.show()