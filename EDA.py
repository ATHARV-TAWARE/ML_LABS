
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load the dataset
df = pd.read_csv("file_name.csv")

print("first five rows:")
print(df.head())

# 2. Display dataset information
print("\ndataset information:")
df.info()

print("\nmissing values:")
print(df.isnull().sum())

# 3. Define numerical and categorical columns
num_cols = ["age", "annual_income_k"]
cat_cols = ["city"]

# 4. Handle missing values in numerical columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# 5. Handle missing values in categorical columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\ndata after handling missing values:")
print(df)

# 6. Detect and handle outliers using IQR
Q1 = df["annual_income_k"].quantile(0.25)
Q3 = df["annual_income_k"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - (1.5 * IQR)
upper = Q3 + (1.5 * IQR)

df["annual_income_k"] = df["annual_income_k"].clip(
    lower=lower, upper=upper
)

print("\nlower limit:", lower)
print("upper limit:", upper)
print("\nincome after outlier capping:")
print(df["annual_income_k"])

# 7. Categorical encoding
encoder = LabelEncoder()
df["city"] = encoder.fit_transform(df["city"])

print("\ndata after categorical encoding:")
print(df)

# 8. Feature scaling
scaler = StandardScaler()

df[num_cols] = scaler.fit_transform(df[num_cols])

print("\ncleaned and scaled data:")
print(df.round(2))

# 9. Plot histograms
df[num_cols].hist(figsize=(8, 6))

plt.tight_layout()
plt.show()

# 10. Plot correlation heatmap
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("correlation heatmap")
plt.show()
