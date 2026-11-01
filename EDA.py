# Import libraries and create the dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.DataFrame({
    "customer_id": [101, 102,103, 104, 105, 106, 107], 
    "age": [23, 45, None, 31, 22, 54, 38],
    "annual_income_k": [50, 85, 63, None, 45, 120, 350],
    "city": ["New York", "Paris", "New York", "London", None, "London", "Paris"]
})
print(df)

# Inspect the dataset and fill missing values

df.info()
print(df.isnull().sum())
df["age"]= df["age"].fillna(df["age"].median())
df["annual_income_k"]= df["annual_income_k"].fillna(df["annual_income_k"].median())
df["city"]= df["city"].fillna(df["city"].mode()[0])
print(df)

# Detect and cap outliers using IQR

Q1 = df["annual_income_k"].quantile(0.25)
Q3 = df["annual_income_k"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - (1.5 * IQR)
upper = Q3 + (1.5 * IQR)
df["annual_income_k"] =df["annual_income_k"].clip(lower, upper)
print("lower:", lower, "\nupper:", upper)
print(df["annual_income_k"])

# Encode City and scale numerical features

scaler = StandardScaler()
df[["age", "annual_income_k"]] = scaler.fit_transform(df[["age", "annual_income_k"]])
print("final cleaned data: ")
print(df.round(2))

# generate histogram for age and annual_income_k

df[["age", "annual_income_k"]].hist(figsize=(8, 6))
plt.tight_layout()
plt.show()

# generate correlation heatmap 

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show()