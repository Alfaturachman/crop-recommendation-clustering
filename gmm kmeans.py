import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load data
df = pd.read_csv('crop_recommendation.csv')
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = df[features]

# Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=5, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 1. K-Means
kmeans = KMeans(n_clusters=22, init='k-means++', random_state=42, n_init=10)
df['KMeans_Cluster'] = kmeans.fit_predict(X_pca)
kmeans_cluster_crops = df.groupby('KMeans_Cluster')['label'].apply(set).to_dict()
y_ml_kmeans = df['KMeans_Cluster'].map(kmeans_cluster_crops).tolist()
mlb_kmeans = MultiLabelBinarizer()
y_bin_kmeans = mlb_kmeans.fit_transform(y_ml_kmeans)

# Train K-Means Multilabel
X_train_km, X_test_km, y_train_km, y_test_km = train_test_split(
    X_pca, y_bin_kmeans, test_size=0.2, random_state=42
)
clf_km = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
clf_km.fit(X_train_km, y_train_km)
y_pred_km = clf_km.predict(X_test_km)

sub_acc_km = accuracy_score(y_test_km, y_pred_km)
f1_micro_km = f1_score(y_test_km, y_pred_km, average='micro')
f1_macro_km = f1_score(y_test_km, y_pred_km, average='macro')

print("K-Means Multilabel Metrics:")
print(f"Subset Accuracy: {sub_acc_km:.4f}")
print(f"F1 Micro: {f1_micro_km:.4f}")
print(f"F1 Macro: {f1_macro_km:.4f}")

# 2. GMM
gmm = GaussianMixture(n_components=22, random_state=42)
df['GMM_Cluster'] = gmm.fit_predict(X_pca)
gmm_cluster_crops = df.groupby('GMM_Cluster')['label'].apply(set).to_dict()
y_ml_gmm = df['GMM_Cluster'].map(gmm_cluster_crops).tolist()
mlb_gmm = MultiLabelBinarizer()
y_bin_gmm = mlb_gmm.fit_transform(y_ml_gmm)

# Train GMM Multilabel
X_train_gm, X_test_gm, y_train_gm, y_test_gm = train_test_split(
    X_pca, y_bin_gmm, test_size=0.2, random_state=42
)
clf_gm = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
clf_gm.fit(X_train_gm, y_train_gm)
y_pred_gm = clf_gm.predict(X_test_gm)

sub_acc_gm = accuracy_score(y_test_gm, y_pred_gm)
f1_micro_gm = f1_score(y_test_gm, y_pred_gm, average='micro')
f1_macro_gm = f1_score(y_test_gm, y_pred_gm, average='macro')

print("\nGMM Multilabel Metrics:")
print(f"Subset Accuracy: {sub_acc_gm:.4f}")
print(f"F1 Micro: {f1_micro_gm:.4f}")
print(f"F1 Macro: {f1_macro_gm:.4f}")

