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

# Create contingency matrices
df['KMeans_Cluster'] = kmeans_labels
df['GMM_Cluster'] = gmm_labels

kmeans_crosstab = pd.crosstab(df['label'], df['KMeans_Cluster'])
gmm_crosstab = pd.crosstab(df['label'], df['GMM_Cluster'])

# Sort index of crops alphabetically to align both heatmaps
kmeans_crosstab = kmeans_crosstab.sort_index()
gmm_crosstab = gmm_crosstab.sort_index()

# Determine output directory (save figures in the root folder so draft_jurnal.md finds them)
output_dir = os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.'

# Plot K-Means Cluster Composition Heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(kmeans_crosstab, cmap='YlGnBu', annot=True, fmt='d', cbar_kws={'label': 'Sample Count'})
plt.title('K-Means Cluster Composition Heatmap (Crop vs Cluster)', fontsize=14, weight='bold', pad=15)
plt.ylabel('Crop Type (Label)', fontsize=12)
plt.xlabel('K-Means Cluster ID', fontsize=12)
plt.tight_layout()
kmeans_out = os.path.join(output_dir, 'kmeans_heatmap.png')
plt.savefig(kmeans_out, dpi=300)
plt.close()

# Plot GMM Cluster Composition Heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(gmm_crosstab, cmap='Purples', annot=True, fmt='d', cbar_kws={'label': 'Sample Count'})
plt.title('GMM Cluster Composition Heatmap (Crop vs Cluster)', fontsize=14, weight='bold', pad=15)
plt.ylabel('Crop Type (Label)', fontsize=12)
plt.xlabel('GMM Cluster ID', fontsize=12)
plt.tight_layout()
gmm_out = os.path.join(output_dir, 'gmm_heatmap.png')
plt.savefig(gmm_out, dpi=300)
plt.close()

print(f"Heatmaps successfully saved to {kmeans_out} and {gmm_out}.")
