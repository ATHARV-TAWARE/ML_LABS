
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

df = pd.read_csv("heart.csv")
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
x_test_scaled = scaler.transform(x_test)

model = LogisticRegression(random_state=42)
model.fit(x_train_scaled, y_train)

# make predicyion and cal. matrix

y_pred = model.predict(x_test_scaled)
y_prob = model.predict_proba(x_test_scaled)[:, 1]

print("accuracy: ", accuracy_score(y_test, y_pred))
print("f1- score: ", round(f1_score(y_test, y_pred, zero_division=0), 4))
print("precision: ", precision_score(y_test, y_pred, zero_division=0))
print("recall: ", recall_score(y_test, y_pred, zero_division=0))
print("roc-auc: ", roc_auc_score(y_test,y_prob))

# confusion_matrix heatmap
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# roc curve 

RocCurveDisplay.from_predictions(y_test, y_prob)
plt.show()

# probability distribuation plot 
#%% Cell 8
plt.hist(y_prob[y_test.to_numpy() == 0],  label="no heart disease")
plt.hist(y_prob[y_test.to_numpy() == 1],  label="heart disease")

plt.xlabel("predicted prob")
plt.ylabel("no. of patients")
plt.title("probability distribution")
plt.legend() # identify eac class
plt.show()


