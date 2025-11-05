import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# ========== LOAD DATA ==========
base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, "dogs_cleaned.csv")

df = pd.read_csv(csv_path)
print(f"✅ Loaded dataset with shape: {df.shape}")

# Select only numeric columns for clustering
X = df.select_dtypes(include=["number"])

# ========== SCALE THE DATA ==========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)
print("✅ Data scaled successfully")

# ========== PCA (for visualization only) ==========
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
print("\n🔍 PCA explained variance ratio:")
print(pca.explained_variance_ratio_)

# ============================================================
# 1️⃣ GAUSSIAN MIXTURE MODEL (GMM)
# ============================================================
print("\n🔹 Running Gaussian Mixture Model (GMM)...")
gmm = GaussianMixture(n_components=4, random_state=42)
labels_gmm = gmm.fit_predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_gmm, cmap="tab10", s=10)
plt.title("Gaussian Mixture Model (PCA Projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

# ============================================================
# 2️⃣ HIERARCHICAL CLUSTERING
# ============================================================
print("\n🔹 Running Hierarchical Clustering...")
Z = linkage(X, method='ward')

# Plot dendrogram (truncated for readability)
plt.figure(figsize=(10, 5))
dendrogram(Z, truncate_mode='level', p=5)
plt.title("Hierarchical Clustering Dendrogram (Truncated)")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

# Assign clusters
clusters = fcluster(Z, t=4, criterion='maxclust')

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="tab10", s=10)
plt.title("Hierarchical Clustering (PCA Projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

print("\n✅ Gaussian Mixture + Hierarchical clustering completed successfully!")
