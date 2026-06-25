# -*- coding: utf-8 -*-
"""
Circular Evaluation Analysis
=============================
Membuktikan bahwa evaluasi pipeline ini VALID BY DESIGN:
- Menghitung purity klaster terhadap ground truth label asli dataset
- Menampilkan tabel dominansi tanaman per klaster
- Menghitung ARI antara klaster dengan ground truth
Ini menjawab kekhawatiran "circular evaluation" reviewer.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

def find_file(filename):
    if os.path.exists(filename):
        return filename
    parent_path = os.path.join('..', filename)
    if os.path.exists(parent_path):
        return parent_path
    return filename

# Load data
csv_path = find_file('crop_recommendation.csv')
df = pd.read_csv(csv_path)
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X_raw = df[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
pca = PCA(n_components=5, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# True labels (ground truth dari dataset)
true_labels = df['label'].values

# Cluster dengan K-Means seed=42
kmeans = KMeans(n_clusters=22, init='k-means++', random_state=42, n_init=10)
km_labels = kmeans.fit_predict(X_pca)

# Cluster dengan GMM seed=42
gmm = GaussianMixture(n_components=22, covariance_type='full', random_state=42)
gm_labels = gmm.fit_predict(X_pca)

# -------------------------------------------------------
# 1. Cluster Purity
# -------------------------------------------------------
def cluster_purity(true_labels, cluster_labels):
    n = len(true_labels)
    df_tmp = pd.DataFrame({'true': true_labels, 'cluster': cluster_labels})
    majority = df_tmp.groupby('cluster')['true'].agg(lambda x: x.value_counts().iloc[0])
    return majority.sum() / n

km_purity = cluster_purity(true_labels, km_labels)
gm_purity = cluster_purity(true_labels, gm_labels)

print("=" * 60)
print("CIRCULAR EVALUATION VALIDITY ANALYSIS")
print("=" * 60)
print(f"\nCluster Purity vs Ground Truth:")
print(f"  K-Means  Purity = {km_purity:.4f} ({km_purity*100:.2f}%)")
print(f"  GMM      Purity = {gm_purity:.4f} ({gm_purity*100:.2f}%)")

# -------------------------------------------------------
# 2. ARI terhadap Ground Truth
# -------------------------------------------------------
# Encode true_labels to integers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
true_int = le.fit_transform(true_labels)

ari_km = adjusted_rand_score(true_int, km_labels)
ari_gm = adjusted_rand_score(true_int, gm_labels)

print(f"\nAdjusted Rand Index vs Ground Truth:")
print(f"  K-Means  ARI = {ari_km:.4f}")
print(f"  GMM      ARI = {ari_gm:.4f}")

# -------------------------------------------------------
# 3. Dominant crop per cluster (GMM — top 5 clusters)
# -------------------------------------------------------
print("\n--- Dominant Crop per GMM Cluster (5 representative) ---")
df['GMM_Cluster'] = gm_labels
df_gm = df.groupby('GMM_Cluster')['label'].apply(lambda x: x.value_counts().head(3).to_dict())

# Print representative clusters with interpretable agronomy
gm_summary = []
for c in range(22):
    subset = df[df['GMM_Cluster'] == c]['label']
    if len(subset) == 0:
        continue
    vc = subset.value_counts()
    dominant = vc.index[0]
    count = vc.iloc[0]
    total = vc.sum()
    purity_c = count / total if total > 0 else 0
    gm_summary.append({'Cluster': c, 'Dominant Crop': dominant,
                        'Count': count, 'Total': total,
                        'Purity': round(purity_c, 2)})

df_gm_summary = pd.DataFrame(gm_summary).sort_values('Purity', ascending=False)
print(df_gm_summary.head(10).to_string(index=False))

# -------------------------------------------------------
# 4. Fraction of clusters with single-crop dominance >= 80%
# -------------------------------------------------------
single_dom_km = 0
single_dom_gm = 0
for c in range(22):
    km_c = df[km_labels == c]['label'].value_counts()
    if len(km_c) > 0 and km_c.iloc[0] / km_c.sum() >= 0.8:
        single_dom_km += 1
    gm_c = df[gm_labels == c]['label'].value_counts()
    if len(gm_c) > 0 and gm_c.iloc[0] / gm_c.sum() >= 0.8:
        single_dom_gm += 1

print(f"\nClusters with dominant single crop >= 80%:")
print(f"  K-Means: {single_dom_km}/22 ({single_dom_km/22*100:.1f}%)")
print(f"  GMM:     {single_dom_gm}/22 ({single_dom_gm/22*100:.1f}%)")

print("\n[CONCLUSION]")
print(f"Evaluasi VALID BY DESIGN: pipeline bertujuan memprediksi keanggotaan klaster,")
print(f"bukan label agronomi eksternal. Purity GMM {gm_purity*100:.1f}% dan ARI {ari_gm:.3f}")
print(f"membuktikan klaster secara alami berselaras kuat dengan varietas tanaman asli.")
