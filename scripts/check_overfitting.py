# -*- coding: utf-8 -*-
"""
Overfitting Check for Random Forest
====================================
Check train vs test subset accuracy for KMeans and GMM labels
with RandomForestClassifier(max_depth=None).
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

# Labels
kmeans = KMeans(n_clusters=22, init='k-means++', random_state=42, n_init=10)
km_labels = kmeans.fit_predict(X_pca)

gmm = GaussianMixture(n_components=22, covariance_type='full', random_state=42)
gm_labels = gmm.fit_predict(X_pca)

# Function to run RF multilabel classification and print train/test accuracy
def check_overfitting(labels, name):
    # Transform labels to multilabel binary matrix
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels.reshape(-1, 1))
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    
    # Train model
    rf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42))
    rf.fit(X_train, y_train)
    
    # Predict
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print(f"[{name}]")
    print(f"  Train Subset Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Test Subset Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Gap (Train - Test):    {train_acc - test_acc:.4f}\n")

print("Checking Overfitting with max_depth=None:")
check_overfitting(km_labels, "K-Means Labels")
check_overfitting(gm_labels, "GMM Labels")
