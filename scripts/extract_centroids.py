# -*- coding: utf-8 -*-
"""
Extract Centroids
==================
Mengekstrak rata-rata parameter agronomi asli (N, P, K, temp, hum, ph, rain)
untuk klaster GMM dan K-Means guna dianalisis dan dimasukkan ke Section 3.3.
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

def find_file(filename):
    if os.path.exists(filename):
        return filename
    parent_path = os.path.join('..', filename)
    if os.path.exists(parent_path):
        return parent_path
    return filename

csv_path = find_file('crop_recommendation.csv')
df = pd.read_csv(csv_path)
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X_raw = df[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
pca = PCA(n_components=5, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# K-Means
kmeans = KMeans(n_clusters=22, init='k-means++', random_state=42, n_init=10)
df['KMeans_Cluster'] = kmeans.fit_predict(X_pca)

# GMM
gmm = GaussianMixture(n_components=22, covariance_type='full', random_state=42)
df['GMM_Cluster'] = gmm.fit_predict(X_pca)

# Menampilkan rata-rata fitur per klaster GMM untuk beberapa klaster penting
# Misalnya klaster dengan P & K sangat tinggi (Apel & Anggur), klaster basah (Padi & Yute), klaster kering (Sorgum & Millet)
print("=== Rerata Karakteristik Tanah per Klaster GMM ===")
gmm_means = df.groupby('GMM_Cluster')[features + ['label']].agg({
    'N': 'mean',
    'P': 'mean',
    'K': 'mean',
    'temperature': 'mean',
    'humidity': 'mean',
    'ph': 'mean',
    'rainfall': 'mean',
    'label': lambda x: '/'.join(x.unique()[:3])
})
print(gmm_means.round(2).to_string())

# K-Means
print("\n=== Rerata Karakteristik Tanah per Klaster K-Means ===")
km_means = df.groupby('KMeans_Cluster')[features + ['label']].agg({
    'N': 'mean',
    'P': 'mean',
    'K': 'mean',
    'temperature': 'mean',
    'humidity': 'mean',
    'ph': 'mean',
    'rainfall': 'mean',
    'label': lambda x: '/'.join(x.unique()[:3])
})
print(km_means.round(2).to_string())
