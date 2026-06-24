import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix

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

# Load GMM model and MLB
gmm = joblib.load(find_file('gmm_model.joblib'))
mlb_gmm = joblib.load(find_file('mlb_gmm.joblib'))
clf_gm = joblib.load(find_file('supervised_multilabel_gmm.joblib'))

# Recreate the target labels
df['GMM_Cluster'] = gmm.predict(X_pca)
gmm_cluster_crops = df.groupby('GMM_Cluster')['label'].apply(set).to_dict()
y_ml_gmm = df['GMM_Cluster'].map(gmm_cluster_crops).tolist()
y_bin_gmm = mlb_gmm.transform(y_ml_gmm)

# Split test set (same state as training)
X_train_gm, X_test_gm, y_train_gm, y_test_gm = train_test_split(
    X_pca, y_bin_gmm, test_size=0.2, random_state=42
)

# Predict
y_pred_gm = clf_gm.predict(X_test_gm)

# Calculate multilabel confusion matrix
mcm = multilabel_confusion_matrix(y_test_gm, y_pred_gm)
classes = mlb_gmm.classes_

# Base output directory for images (root folder)
output_dir = os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.'

# 1. Save consolidated grid heatmap
fig, axes = plt.subplots(6, 4, figsize=(16, 20))
axes = axes.ravel()
cmap = sns.light_palette("purple", as_cmap=True)

for i in range(22):
    ax = axes[i]
    cm = mcm[i]
    sns.heatmap(cm, annot=True, fmt='d', cbar=False, ax=ax, cmap=cmap,
                xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
    ax.set_title(f'Crop: {classes[i].capitalize()}', fontsize=12, weight='bold')
    if i >= 18:
        ax.set_xlabel('Predicted', fontsize=10)
    if i % 4 == 0:
        ax.set_ylabel('True', fontsize=10)

for i in range(22, 24):
    fig.delaxes(axes[i])

plt.suptitle('Multilabel Confusion Matrices per Crop Type (Proposed GMM Pipeline)', fontsize=16, weight='bold', y=0.98)
plt.tight_layout()
grid_output = os.path.join(output_dir, 'multilabel_confusion_matrix.png')
plt.savefig(grid_output, dpi=300)
plt.close()
print(f"Grid Multilabel Confusion Matrix saved to {grid_output}.")

# 2. Save 22 individual confusion matrix images
matrices_dir = os.path.join(output_dir, 'confusion_matrices')
os.makedirs(matrices_dir, exist_ok=True)

for i, crop_name in enumerate(classes):
    cm = mcm[i]
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cbar=False, cmap=cmap,
                xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
    plt.title(f'Confusion Matrix: {crop_name.capitalize()}', fontsize=12, weight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    # Save individual crop image
    crop_filename = f'confusion_matrix_{crop_name.lower()}.png'
    crop_path = os.path.join(matrices_dir, crop_filename)
    plt.savefig(crop_path, dpi=200)
    plt.close()

print(f"All 22 individual confusion matrices successfully saved inside: {matrices_dir}")
