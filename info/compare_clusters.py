import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score

# === Load datasets ===
base_dir = os.path.dirname(__file__)
kmeans_path = os.path.join(base_dir, "dogs_kmeans_results.csv")
gmm_path = os.path.join(base_dir, "dogs_gmm_results.csv")
hier_path = os.path.join(base_dir, "dogs_hierarchical_results.csv")

kmeans_df = pd.read_csv(kmeans_path)
gmm_df = pd.read_csv(gmm_path)
hier_df = pd.read_csv(hier_path)

# === Align them (just in case indexes differ) ===
merged = kmeans_df.copy()
merged["Cluster_GMM"] = gmm_df["Cluster_GMM"]
merged["Cluster_Hier"] = hier_df["Cluster_Hier"]

print("✅ Data merged successfully")
print(merged.head())

# === Compare clusters quantitatively ===
def compare_clusters(a, b, label1, label2):
    ari = adjusted_rand_score(a, b)
    homo = homogeneity_score(a, b)
    comp = completeness_score(a, b)
    print(f"\n🔹 Comparison: {label1} vs {label2}")
    print(f"Adjusted Rand Index: {ari:.3f}")
    print(f"Homogeneity: {homo:.3f}")
    print(f"Completeness: {comp:.3f}")

compare_clusters(merged["Cluster_KMeans"], merged["Cluster_GMM"], "KMeans", "GMM")
compare_clusters(merged["Cluster_KMeans"], merged["Cluster_Hier"], "KMeans", "Hierarchical")
compare_clusters(merged["Cluster_GMM"], merged["Cluster_Hier"], "GMM", "Hierarchical")

# === Simple visualization ===
plt.figure(figsize=(6, 4))
plt.hist(merged["Cluster_KMeans"], alpha=0.5, label="KMeans")
plt.hist(merged["Cluster_GMM"], alpha=0.5, label="GMM")
plt.hist(merged["Cluster_Hier"], alpha=0.5, label="Hierarchical")
plt.legend()
plt.title("Cluster Label Distribution Across Methods")
plt.xlabel("Cluster Label")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
