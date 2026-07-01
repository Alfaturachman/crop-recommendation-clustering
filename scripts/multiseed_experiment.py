# -*- coding: utf-8 -*-
"""
Multi-Seed Stability Experiment
================================
Menjalankan pipeline K-Means dan GMM dengan 3 seed berbeda (0, 42, 99)
untuk membuktikan kestabilan hasil eksperimen.
Hasil: rata-rata ± std untuk Subset Accuracy, F1 Micro, F1 Macro.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# -----------------------------------------------------------------
# Helper
# -----------------------------------------------------------------
def find_file(filename):
    if os.path.exists(filename):
        return filename
    parent_path = os.path.join('..', filename)
    if os.path.exists(parent_path):
        return parent_path
    return filename

# -----------------------------------------------------------------
# Load & Preprocess (deterministic — tidak bergantung pada seed)
# -----------------------------------------------------------------
csv_path = find_file('crop_recommendation.csv')
df_base = pd.read_csv(csv_path)
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X_raw = df_base[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# PCA 5D — fixed random_state=42 (deterministic, tidak berubah per seed)
pca = PCA(n_components=5, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# -----------------------------------------------------------------
# Experiment Loop
# -----------------------------------------------------------------
SEEDS = [0, 42, 99]

results_km = {'subset_acc': [], 'f1_micro': [], 'f1_macro': []}
results_gm = {'subset_acc': [], 'f1_micro': [], 'f1_macro': []}

print("=" * 60)
print("MULTI-SEED STABILITY EXPERIMENT")
print("Seeds:", SEEDS)
print("=" * 60)

for seed in SEEDS:
    df = df_base.copy()

    # ---- K-Means ----
    kmeans = KMeans(n_clusters=12, init='k-means++', random_state=seed, n_init=10)
    df['KMeans_Cluster'] = kmeans.fit_predict(X_pca)
    km_cluster_crops = df.groupby('KMeans_Cluster')['label'].apply(set).to_dict()
    y_ml_km = df['KMeans_Cluster'].map(km_cluster_crops).tolist()
    mlb_km = MultiLabelBinarizer()
    y_bin_km = mlb_km.fit_transform(y_ml_km)

    X_tr_km, X_te_km, y_tr_km, y_te_km = train_test_split(
        X_pca, y_bin_km, test_size=0.2, random_state=seed
    )
    clf_km = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=100, random_state=seed)
    )
    clf_km.fit(X_tr_km, y_tr_km)
    y_pred_km = clf_km.predict(X_te_km)

    results_km['subset_acc'].append(accuracy_score(y_te_km, y_pred_km))
    results_km['f1_micro'].append(f1_score(y_te_km, y_pred_km, average='micro'))
    results_km['f1_macro'].append(f1_score(y_te_km, y_pred_km, average='macro'))

    # ---- GMM ----
    gmm = GaussianMixture(n_components=12, covariance_type='full', random_state=seed)
    df['GMM_Cluster'] = gmm.fit_predict(X_pca)
    gm_cluster_crops = df.groupby('GMM_Cluster')['label'].apply(set).to_dict()
    y_ml_gm = df['GMM_Cluster'].map(gm_cluster_crops).tolist()
    mlb_gm = MultiLabelBinarizer()
    y_bin_gm = mlb_gm.fit_transform(y_ml_gm)

    X_tr_gm, X_te_gm, y_tr_gm, y_te_gm = train_test_split(
        X_pca, y_bin_gm, test_size=0.2, random_state=seed
    )
    clf_gm = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=100, random_state=seed)
    )
    clf_gm.fit(X_tr_gm, y_tr_gm)
    y_pred_gm = clf_gm.predict(X_te_gm)

    results_gm['subset_acc'].append(accuracy_score(y_te_gm, y_pred_gm))
    results_gm['f1_micro'].append(f1_score(y_te_gm, y_pred_gm, average='micro'))
    results_gm['f1_macro'].append(f1_score(y_te_gm, y_pred_gm, average='macro'))

    print(f"\n[Seed {seed}]")
    print(f"  K-Means -> SubAcc={results_km['subset_acc'][-1]:.4f} | "
          f"F1-Micro={results_km['f1_micro'][-1]:.4f} | "
          f"F1-Macro={results_km['f1_macro'][-1]:.4f}")
    print(f"  GMM     -> SubAcc={results_gm['subset_acc'][-1]:.4f} | "
          f"F1-Micro={results_gm['f1_micro'][-1]:.4f} | "
          f"F1-Macro={results_gm['f1_macro'][-1]:.4f}")

# -----------------------------------------------------------------
# Aggregate Results
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("AGGREGATED RESULTS (Mean ± Std across 3 seeds)")
print("=" * 60)

km_sub  = np.array(results_km['subset_acc'])
km_mic  = np.array(results_km['f1_micro'])
km_mac  = np.array(results_km['f1_macro'])
gm_sub  = np.array(results_gm['subset_acc'])
gm_mic  = np.array(results_gm['f1_micro'])
gm_mac  = np.array(results_gm['f1_macro'])

summary = {
    'Model': ['K-Means', 'GMM'],
    'Subset Accuracy (mean)': [f"{km_sub.mean():.4f}", f"{gm_sub.mean():.4f}"],
    'Subset Accuracy (std)':  [f"{km_sub.std():.4f}",  f"{gm_sub.std():.4f}"],
    'F1-Micro (mean)':        [f"{km_mic.mean():.4f}", f"{gm_mic.mean():.4f}"],
    'F1-Micro (std)':         [f"{km_mic.std():.4f}",  f"{gm_mic.std():.4f}"],
    'F1-Macro (mean)':        [f"{km_mac.mean():.4f}", f"{gm_mac.mean():.4f}"],
    'F1-Macro (std)':         [f"{km_mac.std():.4f}",  f"{gm_mac.std():.4f}"],
}

df_summary = pd.DataFrame(summary)
print(df_summary.to_string(index=False))

# Save to CSV
output_dir = os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.'
csv_out = os.path.join(output_dir, 'multiseed_results.csv')
df_summary.to_csv(csv_out, index=False)
print(f"\nResults saved to: {csv_out}")

# Also print per-seed detail table
print("\n--- Per-seed Detail ---")
detail = {
    'Seed': SEEDS,
    'KM_SubAcc': [f"{v:.4f}" for v in results_km['subset_acc']],
    'KM_F1Micro': [f"{v:.4f}" for v in results_km['f1_micro']],
    'KM_F1Macro': [f"{v:.4f}" for v in results_km['f1_macro']],
    'GMM_SubAcc': [f"{v:.4f}" for v in results_gm['subset_acc']],
    'GMM_F1Micro': [f"{v:.4f}" for v in results_gm['f1_micro']],
    'GMM_F1Macro': [f"{v:.4f}" for v in results_gm['f1_macro']],
}
print(pd.DataFrame(detail).to_string(index=False))
