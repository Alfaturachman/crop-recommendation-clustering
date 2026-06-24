import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Helper function to find files in current, parent, or models directories
def find_file(filename):
    search_paths = [
        filename,
        os.path.join('..', filename),
        os.path.join('models', filename),
        os.path.join('..', 'models', filename)
    ]
    for path in search_paths:
        if os.path.exists(path):
            return path
    return filename

# Load dataset and scaler/pca
csv_path = find_file('crop_recommendation.csv')
df = pd.read_csv(csv_path)
X = df.drop('label', axis=1)

scaler = joblib.load(find_file('scaler.joblib'))
pca = joblib.load(find_file('pca_5.joblib'))

X_scaled = scaler.transform(X)
X_pca = pca.transform(X_scaled)

# Load clustering models
kmeans = joblib.load(find_file('kmeans_model.joblib'))
gmm = joblib.load(find_file('gmm_model.joblib'))

# Predict cluster labels
kmeans_labels = kmeans.predict(X_pca)
gmm_labels = gmm.predict(X_pca)

# Determine output directory (root folder)
output_dir = os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.'

# ----------------- Figure 1: PCA Scatter Plot K-Means -----------------
plt.figure(figsize=(10, 8))
# Use first 2 components of PCA for visualization
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='tab20', s=25, alpha=0.8, edgecolor='none')
plt.title('K-Means Cluster Partitioning (PCA 2D Projection)', fontsize=14, weight='bold', pad=15)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
cbar = plt.colorbar(scatter, ticks=range(22))
cbar.set_label('K-Means Cluster ID', fontsize=12)
plt.tight_layout()
kmeans_scatter_out = os.path.join(output_dir, 'kmeans_pca_scatter.png')
plt.savefig(kmeans_scatter_out, dpi=300)
plt.close()
print(f"Figure 1 saved to {kmeans_scatter_out}")

# ----------------- Figure 2: PCA Scatter Plot GMM -----------------
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap='tab20', s=25, alpha=0.8, edgecolor='none')
plt.title('GMM Probabilistic Partitioning (PCA 2D Projection)', fontsize=14, weight='bold', pad=15)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
cbar = plt.colorbar(scatter, ticks=range(22))
cbar.set_label('GMM Cluster ID', fontsize=12)
plt.tight_layout()
gmm_scatter_out = os.path.join(output_dir, 'gmm_pca_scatter.png')
plt.savefig(gmm_scatter_out, dpi=300)
plt.close()
print(f"Figure 2 saved to {gmm_scatter_out}")

# ----------------- Figure 3: Cluster Size Distribution Comparison -----------------
kmeans_counts = pd.Series(kmeans_labels).value_counts().sort_index()
gmm_counts = pd.Series(gmm_labels).value_counts().sort_index()

x = np.arange(22)
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
sns.set_theme(style="whitegrid")

rects1 = ax.bar(x - width/2, kmeans_counts, width, label='K-Means', color='#80b1d3')
rects2 = ax.bar(x + width/2, gmm_counts, width, label='GMM', color='#bc80bd')

ax.set_title('Cluster Size Distribution Comparison (K-Means vs GMM)', fontsize=14, weight='bold', pad=15)
ax.set_xlabel('Cluster ID', fontsize=12)
ax.set_ylabel('Number of Samples', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(x)
ax.legend(fontsize=11)

plt.tight_layout()
dist_out = os.path.join(output_dir, 'cluster_distribution.png')
plt.savefig(dist_out, dpi=300)
plt.close()
print(f"Figure 3 saved to {dist_out}")
