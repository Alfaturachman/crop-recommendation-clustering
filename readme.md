# Klasterisasi Jenis Tanaman Berdasarkan Karakteristik Agronomi

Proyek ini bertujuan untuk mengelompokkan jenis tanaman berdasarkan karakteristik agronomi menggunakan algoritma **K-Means** dan **DBSCAN**. Dengan pendekatan _Unsupervised Learning_, kita dapat menemukan pola alami dari kondisi lahan yang paling sesuai untuk berbagai jenis tanaman.

---

## 1. Ringkasan Proyek

### 1.1 Judul

**Klasterisasi Jenis Tanaman Berdasarkan Karakteristik Agronomi Menggunakan Algoritma K-Means dan DBSCAN**

### 1.2 Latar Belakang

Dalam bidang pertanian modern, pemilihan jenis tanaman yang sesuai dengan kondisi lahan menjadi faktor penting untuk meningkatkan produktivitas. Kondisi agronomi seperti kandungan nitrogen (N), fosfor (P), kalium (K), suhu, kelembaban, pH tanah, dan curah hujan memiliki pengaruh besar terhadap kecocokan suatu tanaman. Machine Learning, khususnya clustering, dapat membantu menemukan pola tersembunyi dalam data tanpa memerlukan label pada proses pelatihan.

### 1.3 Permasalahan

1. Belum adanya sistem otomatis yang mampu mengelompokkan tanaman berdasarkan kemiripan karakteristik agronomi.
2. Sulitnya menentukan pola hubungan antar tanaman secara manual ketika data berukuran besar.
3. Perlunya perbandingan performa algoritma clustering (K-Means vs DBSCAN) pada data pertanian.

### 1.4 Jenis Learning Task

- **Metode**: Unsupervised Learning (Clustering)
- **Algoritma**:
    - K-Means Clustering
    - DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

### 1.5 Fitur Input

| Fitur           | Deskripsi                      |
| --------------- | ------------------------------ |
| **N**           | Kandungan Nitrogen dalam tanah |
| **P**           | Kandungan Fosfor dalam tanah   |
| **K**           | Kandungan Kalium dalam tanah   |
| **Temperature** | Suhu lingkungan rata-rata      |
| **Humidity**    | Kelembaban relatif udara       |
| **pH**          | Tingkat keasaman tanah         |
| **Rainfall**    | Curah hujan tahunan            |

---

## 2. Dataset dan Representasi Data

### 2.1 Informasi Dataset

- **Sumber**: [Crop Recommendation Dataset (Kaggle)](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- **Format**: CSV (.csv)
- **Ukuran**: 2.200 baris × 8 kolom

### 2.2 Deskripsi Fitur

- **N, P, K**: Rasio kandungan unsur hara (mg/kg).
- **Temperature**: Derajat Celsius.
- **Humidity**: Kelembaban relatif (%).
- **pH**: Skala keasaman (0-14).
- **Rainfall**: Curah hujan (mm).
- **Label**: Jenis tanaman (hanya untuk evaluasi ARI, 22 kelas).

### 2.3 Tantangan Dataset

- **Outliers**: Deteksi outlier dilakukan menggunakan Boxplot, namun outlier sengaja **dipertahankan** (tidak dihapus). Beberapa jenis tanaman (seperti apel & anggur) secara alami membutuhkan kadar Kalium (K) yang sangat tinggi, sehingga penghapusan outlier secara global akan melenyapkan data tanaman tersebut dari dataset.
- **Skala Berbeda**: Memerlukan _Feature Scaling_ (StandardScaler).
- **Dimensi Tinggi**: Visualisasi memerlukan reduksi dimensi (PCA).

---

## 3. Pipeline Machine Learning

1. **Data Preprocessing**:
    - Menangani missing values dan duplikat.
    - Analisis outlier (dideteksi menggunakan Boxplot tetapi dipertahankan untuk menjaga keutuhan kelas tanaman).
    - Feature Scaling (StandardScaler).
2. **Feature Engineering**:
    - PCA untuk reduksi dimensi (5 komponen untuk pemodelan dan 2 komponen untuk visualisasi 2D).
3. **Data Splitting**:
    - Seluruh data digunakan untuk clustering (validasi menggunakan metrik internal).
4. **Evaluation**:
    - **Silhouette Score**: Kualitas pemisahan klaster.
    - **Adjusted Rand Index (ARI)**: Kesesuaian dengan label asli.

---

## 4. Modeling & Eksperimen

### 4.1 Strategi Algoritma

- **K-Means**: Menentukan jumlah klaster optimal dengan _Elbow Method_.
- **DBSCAN**: Menentukan parameter `eps` dan `min_samples` untuk menangani noise.

### 4.2 Metrik Evaluasi

| Metrik                    | Fungsi                                            |
| ------------------------- | ------------------------------------------------- |
| Silhouette Score          | Mengukur kualitas pemisahan antar klaster.        |
| Davies-Bouldin Index      | Mengukur kemiripan antar klaster.                 |
| Adjusted Rand Index (ARI) | Mengukur akurasi clustering dibanding label asli. |

### 4.3 Karakteristik Klaster Utama (Hasil K-Means)

Berikut adalah beberapa cluster khas yang berhasil diidentifikasi secara logis oleh model K-Means:

| Cluster | Contoh Tanaman                    | Karakteristik Agronomi Utama (Rata-rata)                               | Klasifikasi Lahan                       |
| :-----: | :-------------------------------- | :--------------------------------------------------------------------- | :-------------------------------------- |
|  **0**  | Mungbean, Orange, Papaya          | Suhu Hangat (~29°C), Kelembaban Tinggi (~88%), Hujan Rendah (~62 mm)   | Lahan Tropis Basah Lembut, Hujan Rendah |
|  **1**  | Coffee                            | N Tinggi (~101 mg/kg), Curah Hujan Tinggi (~158 mm)                    | Perkebunan Basah Dataran Tinggi         |
|  **2**  | Chickpea                          | Kelembaban Kering (~17%), Suhu Sejuk (~19°C), pH (~7.4)                | Lahan Kering Dingin / Sub-Tropis        |
|  **3**  | Grapes, Apple                     | Kalium Tinggi (~200 mg/kg), Fosfor Tinggi (~134 mg/kg)                 | Kebun Buah Khusus Hara Kalium Tinggi    |
|  **4**  | Pomegranate, Orange, Coconut      | Kelembaban Tinggi (~91%), Hara Rendah (N ~19, P ~18, K ~33)            | Lahan Pesisir Basah (Miskin Hara)       |
|  **5**  | Pigeonpeas, Mothbeans, Mango      | Suhu Panas (~31°C), Kelembaban Rendah (~49%)                           | Lahan Tropis Kering Hangat (Semi-Arid)  |
|  **6**  | Watermelon, Muskmelon             | Kelembaban Tinggi (~89%), Curah Hujan Rendah (~38 mm)                  | Lahan Tropis Hangat (Buah Kaya Air)     |
|  **7**  | Pigeonpeas, Mothbeans, Mango      | Suhu Panas (~30°C), Kelembaban Rendah (~50%), pH Asam (~4.9)           | Lahan Kering Hangat (Tanah Asam)        |
|  **8**  | Pigeonpeas, Mothbeans, Lentil     | Kelembaban Sedang (~63%), Curah Hujan Rendah (~62 mm), pH Netral       | Lahan Kering Nutrisi Sedang (Netral)    |
|  **9**  | Rice, Jute, Papaya                | Curah Hujan Sangat Tinggi (~180 mm), Kelembaban Tinggi (~81%)          | Lahan Sawah Basah (Genangan Air)        |
| **10**  | Banana, Cotton                    | N Tinggi (~101 mg/kg), P & K Sedang (~81 & ~49), Hujan (~104 mm)       | Lahan Subur Sedang, Curah Hujan Cukup   |
| **11**  | Pomegranate, Orange               | Suhu Sejuk (~18°C), Kelembaban Tinggi (~91%), Hara Rendah              | Lahan Tropis Basah Sejuk (Hara Rendah)  |
| **12**  | Chickpea, Kidneybeans             | Suhu Sejuk (~20°C), Kelembaban Sangat Kering (~27%), pH Asam (~5.7)    | Lahan Kering Sejuk Asam Sedang          |
| **13**  | Mothbeans                         | pH Basa (~8.7), Curah Hujan Rendah (~54 mm), Suhu Hangat (~28°C)       | Lahan Kering Basa (Alkaloid Tinggi)     |
| **14**  | Rice, Papaya                      | Curah Hujan Ekstrem (~252 mm), Kelembaban Tinggi (~83%)                | Lahan Basah Genangan Tinggi / Rawa      |
| **15**  | Maize, Papaya, Cotton             | Suhu Sejuk (~22°C), Kelembaban Sedang (~67%), Hujan Sedang (~84 mm)    | Lahan Subur Sedang Iklim Sejuk          |
| **16**  | Grapes                            | Suhu Ekstrem Panas (~35°C), Kelembaban Tinggi (~82%), K Tinggi (~200)  | Lahan Kalium Tinggi Tropis Sangat Panas |
| **17**  | Rice, Pigeonpeas, Orange, Coconut | Kelembaban Sangat Tinggi (~94%), Hara Rendah, Hujan Tinggi (~175 mm)   | Lahan Basah Pesisir Humid (Hara Rendah) |
| **18**  | Papaya                            | Suhu Ekstrem Panas (~39°C), Kelembaban Sangat Tinggi (~92%)            | Lahan Basah Tropis Ekstrem Panas        |
| **19**  | Grapes                            | Suhu Ekstrem Dingin (~14°C), Kelembaban Tinggi (~82%), K Tinggi (~200) | Lahan Kalium Tinggi Sub-Tropis Dingin   |
| **20**  | Pigeonpeas, Mothbeans, Lentil     | Kelembaban Sedang (~63%), Curah Hujan Rendah (~52 mm)                  | Lahan Kering Hara Rendah                |
| **21**  | Maize, Watermelon, Cotton         | N Ekstrem Tinggi (~118 mg/kg), Kelembaban Tinggi (~80%), pH (~7.0)     | Lahan Sangat Subur Nitrogen Tinggi      |

---

## 5. Deployment Plan (Inference Design)

Aplikasi akan di-deploy menggunakan **Streamlit**.

### 5.1 Format Input (JSON)

```json
{
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 21.5,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 202.9
}
```

### 5.2 Format Output (JSON)

```json
{
    "cluster": 3,
    "characteristic": "Tanaman tropis lembab",
    "recommended_crops": ["rice", "banana", "papaya"]
}
```

---

## 6. Kesimpulan dan Saran

### Kesimpulan

1. **Pentingnya Skala Data:** Fitur dengan skala bervariasi membutuhkan _Standard Scaling_ agar algoritma berbasis jarak bekerja optimal.
2. **K-Means vs DBSCAN:** **K-Means Clustering** jauh lebih unggul (_Adjusted Rand Index_ stabil dan tinggi) karena data cenderung membentuk kelompok berbasis pusat (_centroid_). Sebaliknya, **DBSCAN** kesulitan menangani klaster tanaman yang saling tumpang tindih (_overlapping_) dan variasi kepadatan data, sehingga mengklasifikasikan banyak titik sebagai _noise_.
3. **Interpretasi Klaster:** Profil rata-rata (mean) dari setiap klaster K-Means berhasil memisahkan kelompok tanaman dengan karakteristik kebutuhan iklim dan hara yang spesifik secara logis.
4. **Kebijakan Outlier:** Penanganan outlier dengan penghapusan menggunakan IQR tidak disarankan secara global karena nilai-nilai ekstrim tersebut representasi alami dari tanaman spesifik (seperti kebutuhan Kalium tinggi pada apel/anggur, serta curah hujan tinggi pada padi).

### Saran Pengembangan Lanjutan

1. **Penerapan Sistem Rekomendasi:** Nilai _Centroid_ dari K-Means dapat diekstrak menjadi basis aturan _(knowledge base)_ untuk mengembangkan aplikasi web Sistem Rekomendasi cerdas berbasis input kondisi tanah _real-time_.
2. **Eksplorasi Algoritma Probabilistik:** Disarankan mencoba algoritma _Gaussian Mixture Models (GMM)_ untuk memodelkan batas klaster yang tumpang tindih dengan lebih fleksibel dibanding K-Means.
3. **Reduksi Fitur Berlebih:** Fitur yang saling berkorelasi sangat kuat dapat direduksi lebih lanjut untuk efisiensi komputasi model.
