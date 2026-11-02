# import libraries

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv("Mall_Customers.csv")

print(df.head())
print("\nDataset information:")
df.info()

# select features and scale 

x= df[["Annual Income (k$)", "Spending Score (1-100)"]]
scaler = StandardScaler()
x_scaled =scaler.fit_transform(x)
print("scaled data: ")
print(x_scaled[:5])

# find optimal no. clustre

wcss =[]
silhouette_scores = []

for k in range(2, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(x_scaled)

    wcss.append(model.inertia_)
    silhouette_scores.append(
        silhouette_score(x_scaled,model.labels_)
    )

plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.plot(range(2, 11), wcss, marker="o")
plt.title("elbow method")
plt.xlabel("no. of clustre")
plt.ylabel("wcss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores, marker="o")
plt.title("silhouette score")
plt.xlabel("no. of clustre")
plt.ylabel("silhouette score")
plt.grid(True)


plt.tight_layout()
plt.show()

for k, score in zip(range(2, 11), silhouette_scores):
    print(f"k = {k}  silhouette score: {score:.4f}")

# train model 

k = 5
kmeans = KMeans(n_clusters= k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(x_scaled)
print(df[["Annual Income (k$)", "Spending Score (1-100)", "Cluster"]].head())

print("\nfinal silhouette Score:")
print(round(silhouette_score(x_scaled, df["Cluster"]), 4))

# picture customer cluster

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="*",
    s=250,
    edgecolors="black",
    label = "centroids"
)

plt.title("clusters")
plt.xlabel("annual income")
plt.ylabel("spending score")
plt.legend()
plt.grid(True)
plt.show()

# display customer profile

profile =df.groupby("Cluster")[
    ["Annual Income (k$)", "Spending Score (1-100)", "Age"]
].mean()
print("avg. customer profile for eacj customer: ")
print(profile.round(2))