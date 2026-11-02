
# use this when data in One-hot encoded
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# read file
# CHANGE "exy.csv" if your CSV file has a different name.

df = pd.read_csv("exy.csv")

print("Dataset:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

# convert dataset into transactions
# CHANGE THIS PART according to your dataset format:
# If your CSV contains 0/1 or True/False columns for each item,
# keep the following encoding code.
# If your CSV has TransactionID and Item columns, use the alternative
# code given below this program.

df = df.select_dtypes(include=["number", "bool"])

# CHANGE THIS PART if your dataset uses values other than 0/1.
# Converts non-zero numeric values to True and zero values to False.
df = df.fillna(0).astype(bool)

# find frequent itemsets
# CHANGE min_support according to your dataset.
# Example: 0.2 means an itemset must appear in at least 20% of transactions.

frequent_itemsets = apriori(
    df,
    min_support=0.2,
    use_colnames=True
)

print("\nFrequent Itemsets:")
print(frequent_itemsets)

# generate association rules
# CHANGE min_threshold according to your required confidence.
# Example: 0.6 means minimum confidence is 60%.

if not frequent_itemsets.empty:
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.6
    )

    print("\nAssociation Rules:")
    if not rules.empty:
        print(
            rules[
                ["antecedents", "consequents", "support", "confidence", "lift"]
            ]
        )
    else:
        print("No association rules found. Try lowering min_support or min_threshold.")
else:
    print("No frequent itemsets found. Try lowering min_support.")












# use this when data contain TransactionID and Item columns

# Lab 9: Implementation of Apriori Algorithm for Association Rule Mining


#%% Cell 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# read file
# CHANGE "exy.csv" if your CSV file has a different name.

df = pd.read_csv("exy.csv")

print("Dataset:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

# convert dataset into transactions
# CHANGE "TransactionID" and "Item" to match your actual CSV column names.

transactions = (
    df.groupby("TransactionID")["Item"]
    .apply(list)
    .tolist()
)

# convert transactions into one-hot encoded format

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

print("\nOne-hot encoded dataset:")
print(df_encoded)

# find frequent itemsets
# CHANGE min_support according to your dataset.

frequent_itemsets = apriori(
    df_encoded,
    min_support=0.2,
    use_colnames=True
)

print("\nFrequent Itemsets:")
print(frequent_itemsets)

# generate association rules
# CHANGE min_threshold according to your required confidence.

if not frequent_itemsets.empty:
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.6
    )

    print("\nAssociation Rules:")
    if not rules.empty:
        print(
            rules[
                ["antecedents", "consequents", "support", "confidence", "lift"]
            ]
        )
    else:
        print("No association rules found. Try lowering min_support or min_threshold.")
else:
    print("No frequent itemsets found. Try lowering min_support.")

