# Libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

# Load and scale data
base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, "dogs_cleaned.csv")
df = pd.read_csv(csv_path)

print("Loaded dataset with shape:", df.shape)

# Select numeric columns
numeric_df = df.select_dtypes(include=["number", "bool"])
scaler = StandardScaler()
scaled_arr = scaler.fit_transform(numeric_df)
scaled_df = pd.DataFrame(scaled_arr, columns=numeric_df.columns, index=numeric_df.index)
print("Data scaled successfully")

# PCA for plotting
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(scaled_df)

# Gaussian Mixture Model
print("\n=== Running Gaussian Mixture Model ===")
gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
gmm_labels = gmm.fit_predict(scaled_df)

plt.figure(figsize=(8, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap="coolwarm", s=10)
plt.title("Gaussian Mixture Model (PCA Projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

# Save GMM results
df["Cluster_GMM"] = gmm_labels
gmm_output = os.path.join(base_dir, "dogs_gmm_results.csv")
df.to_csv(gmm_output, index=False)
print(f"GMM results saved to: {gmm_output}")

# ===== Hierarchical Clustering =====
print("\n=== Running Hierarchical Clustering ===")
hierarchical = AgglomerativeClustering(n_clusters=4, linkage="ward")
hier_labels = hierarchical.fit_predict(scaled_df)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=hier_labels, cmap="plasma", s=10)
plt.title("Hierarchical Clustering (PCA Projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

# Save Hierarchical results
# Dendrogram (more detailed)
print("\n=== Generating Dendrogram ===")

# use a smaller sample to reduce noise but large enough to show structure
sample_data = scaled_df.sample(n=500, random_state=42)

# perform hierarchical linkage
linked = linkage(sample_data, method="ward")

plt.figure(figsize=(12, 6))
# increase p to 50 or 100 for more branches
dendrogram(
    linked,
    truncate_mode="lastp",
    p=50,                # increase this to show more clusters before truncation
    leaf_rotation=90.,
    leaf_font_size=8.,
    show_contracted=True
)

plt.title("Hierarchical Clustering Dendrogram (Truncated, p=50)")
plt.xlabel("Cluster Samples")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

print("\n✅ Dendrogram generated successfully!")

print("\n All clustering results completed and saved successfully!")
