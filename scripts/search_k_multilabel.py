import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os

# Set paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
csv_path = os.path.join(project_root, 'crop_recommendation.csv')
output_img = os.path.join(project_root, 'multilabel_k_search.png')

# 1. Load data
df = pd.read_csv(csv_path)
X = df.drop('label', axis=1)
y_true = df['label']

# 2. Scale and PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=5, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 3. Calculate crop profiles (mean requirements for each crop) to measure compatibility
crop_means = df.groupby('label').mean()

# 4. Search over K
k_values = list(range(4, 21))
results = []

for k in k_values:
    # --- GMM ---
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm_labels = gmm.fit_predict(X_pca)
    
    # Calculate GMM clustering metrics
    gmm_sil = silhouette_score(X_pca, gmm_labels)
    gmm_dbi = davies_bouldin_score(X_pca, gmm_labels)
    gmm_bic = gmm.bic(X_pca)
    
    # Calculate GMM multilabel metrics
    gmm_ct = pd.crosstab(gmm_labels, y_true)
    # A crop is considered recommended for a cluster if it has at least 5% of its instances in this cluster
    # Or if it represents at least 5% of the cluster's population
    gmm_multilabel_counts = []
    gmm_cluster_variance = []
    
    for cluster_id in range(k):
        # Crops in this cluster
        cluster_crops = gmm_ct.loc[cluster_id]
        total_in_cluster = cluster_crops.sum()
        if total_in_cluster > 0:
            # Significant crops (> 5% of cluster population)
            sig_crops = cluster_crops[cluster_crops / total_in_cluster >= 0.05].index.tolist()
            gmm_multilabel_counts.append(len(sig_crops))
            
            # Agronomic compatibility: Variance of rainfall and NPK requirements among these crops
            if len(sig_crops) > 1:
                crop_profiles = crop_means.loc[sig_crops]
                # Normalized variance across features
                norm_var = crop_profiles.var() / crop_means.var()
                gmm_cluster_variance.append(norm_var.mean())
            else:
                gmm_cluster_variance.append(0.0)
        else:
            gmm_multilabel_counts.append(0)
            gmm_cluster_variance.append(0.0)
            
    avg_gmm_crops = np.mean(gmm_multilabel_counts)
    gmm_multilabel_prop = np.mean([1 if c >= 2 else 0 for c in gmm_multilabel_counts])
    avg_gmm_var = np.mean(gmm_cluster_variance)
    
    # --- K-Means ---
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_pca)
    
    # Calculate K-Means clustering metrics
    km_sil = silhouette_score(X_pca, km_labels)
    km_dbi = davies_bouldin_score(X_pca, km_labels)
    km_inertia = km.inertia_
    
    # Calculate K-Means multilabel metrics
    km_ct = pd.crosstab(km_labels, y_true)
    km_multilabel_counts = []
    km_cluster_variance = []
    
    for cluster_id in range(k):
        cluster_crops = km_ct.loc[cluster_id]
        total_in_cluster = cluster_crops.sum()
        if total_in_cluster > 0:
            sig_crops = cluster_crops[cluster_crops / total_in_cluster >= 0.05].index.tolist()
            km_multilabel_counts.append(len(sig_crops))
            
            if len(sig_crops) > 1:
                crop_profiles = crop_means.loc[sig_crops]
                norm_var = crop_profiles.var() / crop_means.var()
                km_cluster_variance.append(norm_var.mean())
            else:
                km_cluster_variance.append(0.0)
        else:
            km_multilabel_counts.append(0)
            km_cluster_variance.append(0.0)
            
    avg_km_crops = np.mean(km_multilabel_counts)
    km_multilabel_prop = np.mean([1 if c >= 2 else 0 for c in km_multilabel_counts])
    avg_km_var = np.mean(km_cluster_variance)
    
    results.append({
        'K': k,
        'GMM_Silhouette': gmm_sil,
        'GMM_DBI': gmm_dbi,
        'GMM_BIC': gmm_bic,
        'GMM_Avg_Crops': avg_gmm_crops,
        'GMM_Multilabel_Prop': gmm_multilabel_prop,
        'GMM_Agronomic_Variance': avg_gmm_var,
        'KM_Silhouette': km_sil,
        'KM_DBI': km_dbi,
        'KM_Inertia': km_inertia,
        'KM_Avg_Crops': avg_km_crops,
        'KM_Multilabel_Prop': km_multilabel_prop,
        'KM_Agronomic_Variance': avg_km_var
    })

results_df = pd.DataFrame(results)

# Print results table
print(results_df.to_string(index=False, formatters={
    'GMM_Silhouette': '{:.3f}'.format,
    'GMM_DBI': '{:.3f}'.format,
    'GMM_BIC': '{:.1f}'.format,
    'GMM_Avg_Crops': '{:.2f}'.format,
    'GMM_Multilabel_Prop': '{:.2%}'.format,
    'GMM_Agronomic_Variance': '{:.3f}'.format,
    'KM_Silhouette': '{:.3f}'.format,
    'KM_DBI': '{:.3f}'.format,
    'KM_Avg_Crops': '{:.2f}'.format,
    'KM_Multilabel_Prop': '{:.2%}'.format,
    'KM_Agronomic_Variance': '{:.3f}'.format,
}))

# 5. Plot the search metrics
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Subplot 1: Silhouette Score (Higher is better)
axes[0, 0].plot(k_values, results_df['GMM_Silhouette'], marker='o', label='GMM', color='blue', linewidth=2)
axes[0, 0].plot(k_values, results_df['KM_Silhouette'], marker='x', label='K-Means', color='orange', linewidth=2)
axes[0, 0].set_title('Silhouette Score (Tinggi = Baik)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Jumlah Klaster (K)', fontsize=10)
axes[0, 0].set_ylabel('Silhouette Score', fontsize=10)
axes[0, 0].set_xticks(k_values)
axes[0, 0].grid(True, linestyle='--', alpha=0.5)
axes[0, 0].legend()

# Subplot 2: Davies-Bouldin Index (Lower is better)
axes[0, 1].plot(k_values, results_df['GMM_DBI'], marker='o', label='GMM', color='blue', linewidth=2)
axes[0, 1].plot(k_values, results_df['KM_DBI'], marker='x', label='K-Means', color='orange', linewidth=2)
axes[0, 1].set_title('Davies-Bouldin Index (Rendah = Baik)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Jumlah Klaster (K)', fontsize=10)
axes[0, 1].set_ylabel('DBI Score', fontsize=10)
axes[0, 1].set_xticks(k_values)
axes[0, 1].grid(True, linestyle='--', alpha=0.5)
axes[0, 1].legend()

# Subplot 3: Average Crop Types per Cluster (Multilabel Diversity)
axes[1, 0].plot(k_values, results_df['GMM_Avg_Crops'], marker='o', label='GMM', color='blue', linewidth=2)
axes[1, 0].plot(k_values, results_df['KM_Avg_Crops'], marker='x', label='K-Means', color='orange', linewidth=2)
axes[1, 0].set_title('Rata-rata Jumlah Komoditas per Klaster (Multilabel Diversity)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Jumlah Klaster (K)', fontsize=10)
axes[1, 0].set_ylabel('Rerata Komoditas', fontsize=10)
axes[1, 0].set_xticks(k_values)
axes[1, 0].grid(True, linestyle='--', alpha=0.5)
axes[1, 0].legend()

# Subplot 4: GMM BIC (Lower is better)
axes[1, 1].plot(k_values, results_df['GMM_BIC'], marker='o', label='GMM BIC', color='purple', linewidth=2)
ax_twin = axes[1, 1].twinx()
ax_twin.plot(k_values, results_df['KM_Inertia'], marker='x', label='KM Inertia', color='red', linestyle='--')
axes[1, 1].set_title('GMM BIC (Ungu) & K-Means Inertia (Merah) (Rendah = Baik)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Jumlah Klaster (K)', fontsize=10)
axes[1, 1].set_ylabel('GMM BIC Score', fontsize=10)
ax_twin.set_ylabel('K-Means Inertia (WCSS)', fontsize=10)
axes[1, 1].set_xticks(k_values)
axes[1, 1].grid(True, linestyle='--', alpha=0.5)
axes[1, 1].legend(loc='upper left')
ax_twin.legend(loc='upper right')

plt.suptitle('ANALISIS PENENTUAN JUMLAH KLASTER (K) OPTIMAL UNTUK REKOMENDASI TANAMAN MULTILABEL', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(output_img, dpi=150, bbox_inches='tight')
print(f"\nVisualisasi pencarian K optimal disimpan sebagai '{output_img}'")
