import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
import os

# Determine paths relative to this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

csv_path = os.path.join(project_root, 'crop_recommendation.csv')
output_img = os.path.join(project_root, 'crop_clustering_comparison.png')

# 1. Load data
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset '{csv_path}' tidak ditemukan. Pastikan dataset berada di project root.")

df = pd.read_csv(csv_path)
X = df.drop('label', axis=1)
y_true = df['label']

# 2. Preprocess and Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA (5 components for modeling, 2 components for plotting)
pca_5 = PCA(n_components=5, random_state=42)
X_pca_5 = pca_5.fit_transform(X_scaled)

pca_2 = PCA(n_components=2, random_state=42)
X_pca_2 = pca_2.fit_transform(X_scaled)

# 4. Train Models
kmeans_model = KMeans(n_clusters=22, init='k-means++', random_state=42, n_init=10)
kmeans_labels = kmeans_model.fit_predict(X_pca_5)

gmm_model = GaussianMixture(n_components=22, random_state=42)
gmm_labels = gmm_model.fit_predict(X_pca_5)
gmm_probs = gmm_model.predict_proba(X_pca_5)

# Create a dataframe for 2D visualization
pca_df = pd.DataFrame(data=X_pca_2, columns=['PC1', 'PC2'])
pca_df['KMeans_Cluster'] = kmeans_labels
pca_df['GMM_Cluster'] = gmm_labels

# 5. Define Ellipse Plotting Function
def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    
    vals = np.clip(vals, 1e-10, None)
    width, height = 2 * nstd * np.sqrt(vals)
    
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ax.add_patch(ellip)
    return ellip

# Set up the figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), sharey=True)

# Generate 22 premium distinct colors
palette = sns.color_palette("husl", 22)

# ==================== LEFT PLOT: K-MEANS ====================
ax1.set_title('K-MEANS CLUSTERING\n(Hard, Spherical Assumed)', fontsize=18, fontweight='bold', pad=15)
ax1.set_xlabel('Principal Component 1', fontsize=14)
ax1.set_ylabel('Principal Component 2', fontsize=14)

# Plot scatter points for K-Means
sns.scatterplot(x='PC1', y='PC2', hue='KMeans_Cluster', data=pca_df, palette=palette, 
                ax=ax1, legend=False, alpha=0.6, s=25)

# Calculate and plot centroids and isotropic circles
for c in range(22):
    cluster_points = X_pca_2[kmeans_labels == c]
    if len(cluster_points) == 0:
        continue
    centroid = np.mean(cluster_points, axis=0)
    std_dev = np.sqrt(np.mean(np.var(cluster_points, axis=0)))
    
    # Plot centroid as a large 'X'
    ax1.scatter(centroid[0], centroid[1], color='black', marker='x', s=120, linewidths=3, zorder=10)
    
    # Draw circle of radius 2*std_dev representing spherical assumption
    circle = plt.Circle(centroid, 2 * std_dev, fill=True, color=palette[c], alpha=0.1, edgecolor=palette[c], linewidth=1.5, linestyle='--')
    ax1.add_patch(circle)

# ==================== RIGHT PLOT: GMM ====================
ax2.set_title('GAUSSIAN MIXTURE MODEL (GMM)\n(Soft, Elliptical Flexible)', fontsize=18, fontweight='bold', pad=15)
ax2.set_xlabel('Principal Component 1', fontsize=14)

# GMM Soft membership mapping (alpha proportional to probability of assigned cluster)
gmm_assigned_probs = gmm_probs[np.arange(len(gmm_labels)), gmm_labels]
rgba_colors = np.zeros((len(gmm_labels), 4))
for i in range(len(gmm_labels)):
    color = palette[gmm_labels[i]]
    rgba_colors[i, :3] = color
    rgba_colors[i, 3] = 0.15 + 0.7 * gmm_assigned_probs[i]

# Plot scatter points with soft opacity
ax2.scatter(pca_df['PC1'], pca_df['PC2'], color=rgba_colors, s=25)

# Calculate and plot means and covariance ellipses
for c in range(22):
    cluster_points = X_pca_2[gmm_labels == c]
    if len(cluster_points) < 2:
        continue
    mean = np.mean(cluster_points, axis=0)
    cov = np.cov(cluster_points, rowvar=False)
    
    # Plot mean as a dot
    ax2.scatter(mean[0], mean[1], color='black', marker='o', s=40, zorder=10)
    
    # Draw covariance ellipses at 1 and 2 standard deviations
    plot_cov_ellipse(cov, mean, nstd=1, ax=ax2, fill=True, color=palette[c], alpha=0.15, zorder=1)
    plot_cov_ellipse(cov, mean, nstd=2, ax=ax2, fill=False, color=palette[c], alpha=0.4, linewidth=1.5, zorder=2)

# Final formatting
for ax in [ax1, ax2]:
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal', 'datalim')

# Add descriptive annotations
ax1.text(0.05, 0.05, "2200 points, 22 crop clusters\nProjected via PCA 2D", transform=ax1.transAxes, 
         fontsize=12, verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax2.text(0.05, 0.05, "GMM adapts covariance matrix (Sigma)\nand mean (mu) for elliptical flexibility", transform=ax2.transAxes, 
         fontsize=12, verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(output_img, dpi=150, bbox_inches='tight')
print(f"Plot saved successfully as '{output_img}'")
