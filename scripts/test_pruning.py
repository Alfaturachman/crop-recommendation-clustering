# -*- coding: utf-8 -*-
"""
Random Forest Tree Pruning Experiment
======================================
Menganalisis dampak pemangkasan pohon (tree pruning) dengan membatasi kedalaman
maksimum pohon (max_depth) terhadap akurasi model klasifikasi terawasi multilabel
(Subset Accuracy) untuk target klasterisasi K-Means dan GMM.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def find_file(filename):
    if os.path.exists(filename):
        return filename
    parent_path = os.path.join('..', filename)
    if os.path.exists(parent_path):
        return parent_path
    return filename

# Load & Preprocess
csv_path = find_file('crop_recommendation.csv')
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File {csv_path} tidak ditemukan. Harap pastikan dataset berada di folder proyek.")

df_base = pd.read_csv(csv_path)
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X_raw = df_base[features].values

# StandardScaler & PCA 5D
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
pca = PCA(n_components=5, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 1. K-Means Targets
kmeans = KMeans(n_clusters=12, init='k-means++', random_state=42, n_init=10)
df_base['KMeans_Cluster'] = kmeans.fit_predict(X_pca)
km_cluster_crops = df_base.groupby('KMeans_Cluster')['label'].apply(set).to_dict()
y_ml_km = df_base['KMeans_Cluster'].map(km_cluster_crops).tolist()
mlb_km = MultiLabelBinarizer()
y_bin_km = mlb_km.fit_transform(y_ml_km)

# 2. GMM Targets
gmm = GaussianMixture(n_components=12, covariance_type='full', random_state=42)
df_base['GMM_Cluster'] = gmm.fit_predict(X_pca)
gm_cluster_crops = df_base.groupby('GMM_Cluster')['label'].apply(set).to_dict()
y_ml_gm = df_base['GMM_Cluster'].map(gm_cluster_crops).tolist()
mlb_gm = MultiLabelBinarizer()
y_bin_gm = mlb_gm.fit_transform(y_ml_gm)

depths = [5, 10, 15, 20, 25, None]

def run_experiment(X, y, target_name):
    print(f"\nEksperimen Pruning untuk Target {target_name} (seed 42):")
    print("-" * 65)
    print(f"{'max_depth':<10} | {'Train SubAcc':<12} | {'Test SubAcc':<12} | {'Gap (Train-Test)':<15}")
    print("-" * 65)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    for d in depths:
        clf = MultiOutputClassifier(
            RandomForestClassifier(n_estimators=100, max_depth=d, random_state=42)
        )
        clf.fit(X_train, y_train)
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        d_str = "None" if d is None else str(d)
        print(f"{d_str:<10} | {train_acc*100:10.2f}% | {test_acc*100:10.2f}% | {(train_acc - test_acc)*100:13.2f}%")
        
    print("-" * 65)

if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING RANDOM FOREST TREE PRUNING EXPERIMENT")
    print("=" * 65)
    
    run_experiment(X_pca, y_bin_km, "K-Means (Hard Clustering)")
    run_experiment(X_pca, y_bin_gm, "GMM (Soft Clustering)")
