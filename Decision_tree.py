# import librareis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score
)

# read file 

# df = pd.read_csv("file_name.csv") # file in same folder
# print(df.head()) 

# generate dataset

np.random.seed(42)
n_samples = 1000

data = {
    'Debt_to_Income_Ratio': np.random.uniform(0.1, 0.8, n_samples),
    'Credit_Score': np.random.randint(300, 850, n_samples),
    'Annual_Income': np.random.uniform(20000, 150000, n_samples),
    'Missed_Payments_Count': np.random.randint(0, 10, n_samples)
}

df = pd.DataFrame(data)

risk_index = (
    df['Debt_to_Income_Ratio'] * 2.5
    + df['Missed_Payments_Count'] * 0.4
    - df['Credit_Score'] / 300.0
)

df['Credit_Risk_Rating'] = pd.qcut(
    risk_index,
    q=3,
    labels=[0, 1, 2]
).astype(int)

print(df.head())
print("\nDataset shape:", df.shape)
print("\nRisk class counts:")
print(df['Credit_Risk_Rating'].value_counts().sort_index())

# seperate features and split data

x= df[[
    'Debt_to_Income_Ratio',
    'Credit_Score',
    'Annual_Income',
    'Missed_Payments_Count'
]]
y = df['Credit_Risk_Rating']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)
print("training data: ", len(x_train))
print("testing data: ", len(x_test))

# train the model 

dt_model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

dt_model.fit(x_train, y_train)

# cost complexity pruning using cross valuidation

path = dt_model.cost_complexity_pruning_path(x_train, y_train)
alphas = path.ccp_alphas

best_alpha = 0
best_score = -1

for alpha in alphas:
    model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        ccp_alpha=alpha,
        random_state=42
    )

    scores = cross_val_score(model, x_train, y_train, cv=5)
    mean_score = scores.mean()

    if mean_score > best_score:
        best_score = mean_score
        best_alpha = alpha

dt_model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    ccp_alpha=best_alpha,
    random_state=42
)

dt_model.fit(x_train, y_train)

print("best ccp_alpha:", best_alpha)
print("best cross-validation accuracy:", round(best_score, 4 ))

# prediction and evaluation

y_pred = dt_model.predict(x_test)
y_probs = dt_model.predict_proba(x_test)
print("accuarracy: ", accuracy_score(y_test, y_pred))
print("confusion matrix: ", confusion_matrix(y_test, y_pred))
print("classification report: \n")
print(classification_report(
    y_test, y_pred,
    labels=[0, 1, 2],
    target_names=['low risk', 'medium risk', 'high risk'],
    zero_division=0
))
print("roc auc (ovr): ",
      round(roc_auc_score(y_test, y_probs,multi_class='ovr',
      labels=dt_model.classes_), 4))


# feature importanc bar chart
feature_imp = pd.Series(
    dt_model.feature_importances_, 
    index=x.columns
).sort_values(ascending=False)

print(feature_imp)

feature_imp.plot(kind='bar')
plt.title("feature importance")
plt.xlabel("feature")
plt.ylabel("importance")
plt.xticks(rotation=10)
plt.tight_layout()
plt.show()

# extract decision rules

tree_rules = export_text(
    dt_model,
    feature_names=list(x.columns)
)

print("rules: \n", tree_rules)

# confusion matrix heatmap

cm = confusion_matrix(y_test, y_pred, labels=[0,1,2])
sns.heatmap(
    cm, annot=True, fmt='d', cmap="Purples",
    xticklabels=['low resk', 'medium risk', 'high risk'],
    yticklabels=['low resk', 'medium risk', 'high risk']
)
plt.xlabel("predict risk")
plt.ylabel("actual risk")
plt.title("decision tree cm")
plt.tight_layout()
plt.show()

# show decision tree 
plt.figure(figsize=(20, 10))
plot_tree(
    dt_model, 
    feature_names=x.columns,
    class_names=['low risk', 'medium risk', 'high risk'],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("decision tree")
plt.show()
