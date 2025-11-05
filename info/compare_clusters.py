import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# load data
base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, "dogs_cluster_results.csv")

df = pd.read_csv(csv_path)
print(f"✅ Loaded dataset with shape: {df.shape}")

# Select numeric columns (for consistent scoring)
X = df.select_dtypes(include=["number"])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# k-means

print("\n🔹 Running K-Means for comparison...")
kmeans = KMeans(n_clusters=4, random_state=42)
labels_kmeans = kmeans.fit_predict(X_scaled)
sil_kmeans = silhouette_score(X_scaled, labels_kmeans)
print(f"K-Means Silhouette Score: {sil_kmeans:.3f}")


# gmm & hierarchial

labels_gmm = df["Cluster_GMM"]
labels_hier = df["Cluster_Hierarchical"]

sil_gmm = silhouette_score(X_scaled, labels_gmm)
sil_hier = silhouette_score(X_scaled, labels_hier)

print(f"GMM Silhouette Score: {sil_gmm:.3f}")
print(f"Hierarchical Silhouette Score: {sil_hier:.3f}")

# visualize the scores
methods = ["K-Means", "GMM", "Hierarchical"]
scores = [sil_kmeans, sil_gmm, sil_hier]

plt.figure(figsize=(7, 5))
plt.bar(methods, scores, color=["#66c2a5", "#fc8d62", "#8da0cb"])
plt.title("Cluster Quality Comparison (Silhouette Scores)")
plt.ylabel("Silhouette Score")
plt.ylim(0, 1)
for i, v in enumerate(scores):
    plt.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.show()

print("\n Comparison complete! Check the bar chart above.")
