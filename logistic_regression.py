
# import libraires

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay
)

# read csv file 

df = pd.read_csv("heart_disease_data.csv")
print(df.head())
print(df.columns)

# select features and split the Data 

features = [
    "Age", 
    "Resting_Blood_Pressure", 
    "Cholesterol", 
    "Maximum_Heart_Rate",
    "Exercise_Induced_Angina"
]

x = df[features]
y = df["Heart_Disease"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2,random_state=42, stratify=y)
print("Training data: ", x_train.shape)
print("Testing data: ", x_test.shape)

# scale data and train model 
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

model = LogisticRegression(random_state=42)
model.fit(x_train_scaled, y_train)

# make predicyion and cal. matrix
