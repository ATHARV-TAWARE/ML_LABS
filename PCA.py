#import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("exy.csv")

x = df.select_dtypes(include=np.number).values
feature_names = df.select_dtypes(include=np.number).columns

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

# standardize the feature
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
print("scaled data: ")
print(np.round(x_scaled[:5, :5], 3))

# apply pca and plot explained variance
pca_full = PCA()
pca_full.fit(x_scaled)

variance = pca_full.explained_variance_ratio_
cum_variance = np.cumsum(variance)

plt.bar(range(1, len(variance)+1), variance)
plt.plot(range(1, len(cum_variance)+1), cum_variance, color="red")
plt.axhline(y=0.9, linestyle="--")
plt.title("pca explained variance")
plt.xlabel("principle component")
plt.ylabel("variance ratio")
plt.grid(True)
plt.show()

print("cumulative variance: ")
print(np.round(cum_variance[:10],3))

# reduce data to 2 components
pca =PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

print("original shape: ", x.shape)
print("reduced shape: ", x_pca.shape)
print("pc1 variance", round(pca.explained_variance_ratio_[0], 3))
print("pc2 variance", round(pca.explained_variance_ratio_[1], 3))
print("total variance preserved: ", round(sum(pca.explained_variance_ratio_), 3))

# vizualize the reduced data 

plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y, cmap="coolwarm")
plt.title("2d pca projection")
plt.xlabel("principal component 1")
plt.ylabel("principal component 2")
plt.grid(True)
plt.show()

# display component loading
loading = pd.DataFrame(
    pca.components_, 
    columns=feature_names,
    index=["pc1", "pc2"]
)

print("component loading: ")
print(loading.iloc[:, :5])

