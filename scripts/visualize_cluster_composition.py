import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import os

# Determine paths relative to this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

csv_path = os.path.join(project_root, 'crop_recommendation.csv')
output_img = os.path.join(project_root, 'cluster_composition_proof.png')

# 1. Load data
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset '{csv_path}' tidak ditemukan. Pastikan dataset berada di project root.")

df = pd.read_csv(csv_path)
X = df.drop('label', axis=1)
y_true = df['label']

# 2. Scale and PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca_5 = PCA(n_components=5, random_state=42)
X_pca_5 = pca_5.fit_transform(X_scaled)

# 3. Fit Models
kmeans_model = KMeans(n_clusters=22, init='k-means++', random_state=42, n_init=10)
kmeans_labels = kmeans_model.fit_predict(X_pca_5)

gmm_model = GaussianMixture(n_components=22, random_state=42)
gmm_labels = gmm_model.fit_predict(X_pca_5)

# 4. Create Crosstabulations (Clusters vs Crops)
kmeans_ct = pd.crosstab(kmeans_labels, y_true)
gmm_ct = pd.crosstab(gmm_labels, y_true)

# ==============================================================================
# DOKUMENTASI & PROFILING AGRONOMI KLASTER (MAPPING ID KLASTER KE KATEGORI LAHAN)
# ==============================================================================
# Masing-masing klaster dipetakan berdasarkan analisis statistik parameter tanah
# (N, P, K) dan cuaca (Kelembaban, Curah Hujan, Temp, pH) di dalam klaster tersebut.

gmm_names = {
    0: "Klaster 0: Lahan Kacang Polong Tropis Kering (Lentil/Blackgram)",
    1: "Klaster 1: Lahan Basah Netral Subur / Neutral Wetland (Jute/Rice)",
    2: "Klaster 2: Lahan Dingin Kalium & Fosfor Tinggi (Apple)",
    3: "Klaster 3: Lahan Alkali Subur Nitrogen (Cotton)",
    4: "Klaster 4: Lahan Kacang Pigeonpea Basah (Pigeonpeas)",
    5: "Klaster 5: Lahan Kacang Mungbean Tropis Humid (Mungbean/Blackgram)",
    6: "Klaster 6: Lahan Tropis Basah Nitrogen Tinggi (Banana)",
    7: "Klaster 7: Lahan Basah Masam Sedang / Acidic Wetland (Jute)",
    8: "Klaster 8: Lahan Kering Ekstrem Dingin/Semi-Arid (Chickpea)",
    9: "Klaster 9: Lahan Basah-Kering Tropis (Muskmelon)",
    10: "Klaster 10: Lahan Kering Ekstrem Hangat/Semi-Arid (Kidneybeans)",
    11: "Klaster 11: Lahan Tropis Suhu Hangat Humid (Papaya)",
    12: "Klaster 12: Lahan Sereal Terbuka Dataran Sedang (Maize)",
    13: "Klaster 13: Lahan Kacang Polong Rendah Hujan (Mothbeans)",
    14: "Klaster 14: Lahan Perkebunan Dataran Sedang (Coffee)",
    15: "Klaster 15: Lahan Hortikultura Tropis Dingin (Orange)",
    16: "Klaster 16: Lahan Pesisir Basah Humid (Coconut)",
    17: "Klaster 17: Lahan Kering Kalium & Fosfor Tinggi (Grapes)",
    18: "Klaster 18: Lahan Hortikultura Tropis Netral (Pomegranate/Orange)",
    19: "Klaster 19: Lahan Mangga Iklim Tropis Kering (Mango)",
    20: "Klaster 20: Lahan Sawah Irigasi Basah Ekstrem / Core Wetland (Rice)",
    21: "Klaster 21: Lahan Humid Kaya Nitrogen (Watermelon/Cotton)"
}

kmeans_names = {
    0: "Klaster 0: Campuran Orange/Pomegranate/Coffee",
    1: "Klaster 1: Campuran Apple/Grapes (Hara Tinggi)",
    2: "Klaster 2: Campuran Banana/Papaya",
    3: "Klaster 3: Campuran Coconut/Papaya",
    4: "Klaster 4: Campuran Kidneybeans/Chickpea",
    5: "Klaster 5: Campuran Mungbean/Blackgram",
    6: "Klaster 6: Campuran Coffee/Jute (Lahan Basah)",
    7: "Klaster 7: Lahan Mothbeans",
    8: "Klaster 8: Campuran Mango/Pigeonpeas",
    9: "Klaster 9: Lahan Rice (Lahan Basah)",
    10: "Klaster 10: Campuran Chickpea/Lentil",
    11: "Klaster 11: Campuran Lentil/Blackgram",
    12: "Klaster 12: Campuran Mango/Mothbeans",
    13: "Klaster 13: Campuran Kidneybeans/Pigeonpeas",
    14: "Klaster 14: Lahan Grapes (Hara Tinggi)",
    15: "Klaster 15: Campuran Maize/Coffee (Lahan Sedang)",
    16: "Klaster 16: Campuran Cotton/Watermelon",
    17: "Klaster 17: Lahan Grapes",
    18: "Klaster 18: Campuran Papaya/Orange",
    19: "Klaster 19: Campuran Muskmelon/Watermelon (Lahan Kering)",
    20: "Klaster 20: Campuran Pomegranate/Orange/Papaya",
    21: "Klaster 21: Campuran Rice/Jute"
}

# Rename index using the agronomic labels
kmeans_ct.index = [kmeans_names[i] for i in kmeans_ct.index]
gmm_ct.index = [gmm_names[i] for i in gmm_ct.index]

# Mask 0 values to keep the heatmaps clean and easy to read
kmeans_ct_masked = kmeans_ct.replace(0, np.nan)
gmm_ct_masked = gmm_ct.replace(0, np.nan)

# 5. Set up plotting figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 14))

# Plot K-Means composition heatmap
sns.heatmap(kmeans_ct_masked, annot=True, fmt='g', cmap='Oranges', cbar=True,
            linewidths=0.5, linecolor='lightgray', ax=ax1, 
            annot_kws={"size": 10, "weight": "bold"}, mask=kmeans_ct_masked.isna())
ax1.set_title("KOMPOSISI & PROFILING KLASTER K-MEANS\n(Bercampur dan Terfragmentasi Secara Geometris)", fontsize=16, fontweight='bold', pad=15)
ax1.set_xlabel("Nama Komoditas Tanaman Asli (Ground Truth)", fontsize=12, labelpad=10)
ax1.set_ylabel("Profiling Zona Agronomi Klaster K-Means", fontsize=12)
ax1.tick_params(axis='x', rotation=45, labelsize=10)
ax1.tick_params(axis='y', labelsize=10)

# Plot GMM composition heatmap
sns.heatmap(gmm_ct_masked, annot=True, fmt='g', cmap='Blues', cbar=True,
            linewidths=0.5, linecolor='lightgray', ax=ax2, 
            annot_kws={"size": 10, "weight": "bold"}, mask=gmm_ct_masked.isna())
ax2.set_title("KOMPOSISI & PROFILING KLASTER GMM (GAUSSIAN MIXTURE)\n(Murni, Koheren, dan Selaras Secara Ekologis)", fontsize=16, fontweight='bold', pad=15)
ax2.set_xlabel("Nama Komoditas Tanaman Asli (Ground Truth)", fontsize=12, labelpad=10)
ax2.set_ylabel("Profiling Zona Agronomi Klaster GMM", fontsize=12)
ax2.tick_params(axis='x', rotation=45, labelsize=10)
ax2.tick_params(axis='y', labelsize=10)

plt.tight_layout()
plt.savefig(output_img, dpi=150, bbox_inches='tight')
print(f"Bukti komposisi klaster ber-dokumentasi berhasil disimpan sebagai '{output_img}'")
