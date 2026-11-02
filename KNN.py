# import libraroies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score
)

# read dataset

# the csv has empty filler columns that don't line up with the header,
# so keep only the non-blank values in each row
raw = pd.read_csv("credit_risk_knn_dataset.csv", header=None, skiprows=1,
                  dtype=str, encoding="utf-8-sig")
columns = ['Debt_to_Income_Ratio', 'Credit_Score', 'Annual_Income',
           'Total_Assets', 'Credit_Risk_Rating']
df = pd.DataFrame(
    [[v for v in row if isinstance(v, str) and v.strip()] for row in raw.values],
    columns=columns
).astype(float)
df['Credit_Risk_Rating'] = df['Credit_Risk_Rating'].astype(int)

print(df.head())
print(df.columns)



# seperate features and split data 

x = df[['Debt_to_Income_Ratio', 'Credit_Score',
        'Annual_Income', 'Total_Assets']]

y = df['Credit_Risk_Rating']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)
print("training data: ", len(x_train))
print("testing data: ", len(x_test))

# scale and feature

scaler = StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled =scaler.transform(x_test)

# find best valu fro k 

k_range = range(1, 26)
cv_scores  = []

for k in k_range:
    model = KNeighborsClassifier(n_neighbors= k)
    scores = cross_val_score(
        model, x_train_scaled, y_train,
        cv=5, scoring='accuracy'
    )
    cv_scores.append(scores.mean())

optimal_k = list(k_range)[np.argmax(cv_scores)]
print("optimal K:", optimal_k)

# plot k vs error rste 

error_rate = [1 - score for score in cv_scores]

plt.plot(list(k_range), error_rate, marker='o')
plt.xlabel("value of k")
plt.ylabel("error rate")
plt.title("k vs error rate")
plt.show()

# train final knn model and evaluate

knn = KNeighborsClassifier(
    n_neighbors=optimal_k,
    weights='distance'
)

knn.fit(x_train_scaled, y_train)

y_pred = knn.predict(x_test_scaled)
y_prob = knn.predict_proba(x_test_scaled)

print("confusion matrix: \n", confusion_matrix(y_test, y_pred))
print("classification report: \n")
print(classification_report(
    y_test, y_pred,
    target_names=['low risk', 'moedium risk', 'high risk'],
    zero_division=0
 ))

auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
print("multi-class roc-auc: ", round(auc,4))

# confusio matrix heatmap

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap= "Purples")
plt.xlabel("predicted")
plt.ylabel("original")
plt.title("knn confusion matrix1" )
plt.show()
