# PRESENTASI UAS: KLASTERISASI JENIS TANAMAN

## Berdasarkan Karakteristik Agronomi Menggunakan K-Means & DBSCAN

**Mata Kuliah: Pembelajaran Mesin (A11.4405)**

---

## Slide 1: Pendahuluan & Latar Belakang

### Latar Belakang

- **Tantangan Pertanian Modern:** Pemilihan varietas tanaman yang tidak sesuai dengan karakteristik agronomi lahan menyebabkan penurunan produktivitas hasil panen.
- **Solusi Machine Learning:** Menggunakan pendekatan _Unsupervised Learning_ (Clustering) untuk mengelompokkan lahan berdasarkan kemiripan nutrisi tanah dan kondisi iklim mikro secara otomatis.

### Permasalahan

1. Bagaimana mengelompokkan jenis tanaman berdasarkan kemiripan agronomi tanah tanpa label manual?
2. Bagaimana perbandingan performa algoritma berbasis pusat klaster (**K-Means**) dengan algoritma berbasis kepadatan (**DBSCAN**)?

---

## Slide 2: Alur 1 & 2 - Import Library & Data Loading

### 1. Import Library

- Menggunakan pustaka standar Python:
    - **Manipulasi Data:** `pandas`, `numpy`
    - **Visualisasi:** `matplotlib`, `seaborn`
    - **Machine Learning:** `scikit-learn` (`StandardScaler`, `PCA`, `KMeans`, `DBSCAN`)
    - **Metrik Evaluasi:** `silhouette_score`, `davies_bouldin_score`, `adjusted_rand_score`

### 2. Data Loading

- **Dataset:** _Crop Recommendation Dataset_ (CSV format, 2200 baris × 8 kolom).
- Fitur yang digunakan: Nitrogen (N), Fosfor (P), Kalium (K), Temperatur, Kelembaban, pH tanah, dan Curah Hujan.

---

## Slide 3: Alur 3 - Data Understanding

### Deskripsi Fitur Agronomi

- **N, P, K:** Rasio kandungan unsur hara esensial dalam tanah (mg/kg).
- **Temperature & Humidity:** Suhu lingkungan (°C) dan kelembaban relatif udara (%).
- **pH:** Tingkat keasaman tanah (skala 0-14).
- **Rainfall:** Curah hujan tahunan (mm).
- **Label:** Jenis tanaman asli (22 kelas, masing-masing 100 baris, hanya digunakan sebagai pembanding/ground-truth pada evaluasi ARI).

---

## Slide 4: Alur 4 - Data Cleaning

### Mengapa Outlier Tidak Dihapus?

- **Pengecekan Missing Values & Duplikat:** Tidak ditemukan nilai kosong atau baris duplikat pada dataset (bersih).
- **Pengecekan Outlier (Boxplot):** Beberapa fitur menunjukkan adanya data pencilan (outlier) yang cukup banyak.
- **Kebijakan Analisis:** Outlier sengaja **dipertahankan (tidak dihapus)** karena merupakan karakteristik alami tanaman tertentu.
    - _Contoh:_ Tanaman apel dan anggur secara alami memerlukan kadar Kalium (K) yang sangat tinggi (~200 mg/kg). Jika dibersihkan dengan metode IQR, seluruh data apel dan anggur akan terhapus.

---

## Slide 5: Alur 5 - Exploratory Data Analysis (EDA)

### Analisis Sebaran & Korelasi Fitur

- **Distribusi Fitur:** Sebagian besar fitur tidak terdistribusi normal sempurna (bimodal atau menceng), yang wajar terjadi pada karakteristik iklim yang berbeda-beda.
- **Matriks Korelasi:**
    - Hubungan korelasi terkuat ditemukan antara kandungan **Fosfor (P)** dan **Kalium (K)** dengan nilai korelasi positif yang tinggi (~0.74).
    - Fitur iklim (Suhu, Kelembaban, Curah Hujan) tidak memiliki korelasi linear yang kuat satu sama lain.

---

## Slide 6: Alur 6 - Data Preprocessing

### Pentingnya standardisasi data (_Feature Scaling_)

- **Standardisasi (StandardScaler):** Mengubah nilai setiap fitur sehingga memiliki rata-rata (_mean_) = 0 dan standar deviasi (_std_) = 1.
- **Alasan Krusial:** Algoritma clustering K-Means dan DBSCAN berbasis pada perhitungan **jarak Euclidean**.
    - Tanpa scaling, fitur dengan nilai nominal besar seperti curah hujan (hingga 300 mm) akan mendominasi fitur bernilai nominal kecil seperti pH (rentang 3.5 - 9).

---

## Slide 7: Alur 7 - Feature Engineering (PCA)

### Reduksi Dimensi dengan PCA

- **Tantangan:** Data asli memiliki 7 dimensi hara dan iklim, sehingga sulit untuk dimodelkan secara efisien dan mustahil divisualisasikan dalam grafik 2D secara langsung.
- **Metode:** _Principal Component Analysis (PCA)_ digunakan untuk mereduksi dimensi data:
    - **Pemodelan (Modeling):** Reduksi menjadi **5 komponen utama** untuk mempertahankan ~89% informasi kumulatif.
    - **Visualisasi:** Proyeksi menjadi **2 komponen utama** untuk visualisasi scatter plot 2D.
- **Tujuan:** Mengoptimalkan kualitas modeling klaster pada ruang dimensi yang tepat, sekaligus menyediakan visualisasi 2D dan 5D (pairplot).

---

## Slide 8: Alur 8 - Training Model: K-Means Clustering

### Penentuan Jumlah Klaster (K)

- **Elbow Method & Silhouette Score:** Eksperimen dijalankan pada rentang $K = 2$ hingga $K = 30$.
- **Hasil:** Kurva Silhouette dan grafik WCSS menunjukkan kestabilan yang logis. Mengingat label asli memiliki 22 kelas tanaman, dipilih **$K = 22$** sebagai jumlah klaster optimal agar setiap klaster merepresentasikan kelompok tanaman tertentu secara spesifik.

---

## Slide 9: Alur 8 (Lanjutan) - Training Model: DBSCAN

### Penentuan Parameter DBSCAN

- **Heuristik MinSamples:** Menggunakan parameter `min_samples = 10` (berdasarkan rumus $2 \times D$, di mana dimensi PCA $D=5$).
- **K-Distance Graph:** Menentukan nilai Epsilon ($\epsilon$) optimal dengan mencari titik "tekukan siku" terluar pada data PCA 5D.
- **Hasil Eksperimen:** Dipilih nilai **$\epsilon = 0.6$** dengan `min_samples = 14` untuk mengoptimalkan ARI, menghasilkan 11 klaster valid dengan 636 titik data yang dikategorikan sebagai noise (`label -1`).

---

## Slide 10: Alur 9 - Evaluasi Model

### Perbandingan Performa: K-Means vs DBSCAN

| Metrik Evaluasi               | K-Means (K=22) |   DBSCAN (Eps=0.6)    | Interpretasi                                            |
| :---------------------------- | :------------: | :-------------------: | :------------------------------------------------------ |
| **Silhouette Score**          |   **0.307**    | 0.063 (Bersih: 0.295) | K-Means memisahkan klaster dengan batas lebih tegas     |
| **Davies-Bouldin Index**      |   **1.081**    | 1.818 (Bersih: 0.889) | Klaster K-Means lebih padat dan terpisah baik           |
| **Adjusted Rand Index (ARI)** |   **0.519**    |         0.288         | Pengelompokan K-Means sangat cocok dengan varietas asli |

- **Analisis Performa:** K-Means jauh lebih unggul pada ruang PCA 5D karena sebaran data pertanian ini cenderung terpusat di sekitar centroid tanaman masing-masing (_centroid-based_), sedangkan DBSCAN kesulitan memisahkan kelompok yang saling tumpang tindih.

---

## Slide 11: Alur 10 - Visualisasi & Interpretasi

### Analisis Profil Klaster (K-Means)

- **Profil Rata-rata:** Dengan menghitung nilai rata-rata (_mean_) tiap fitur per klaster, kita dapat mengetahui kecocokan lingkungan:
    - _Klaster dengan curah hujan tinggi (>200 mm):_ Sangat cocok untuk **Padi** dan **Kelapa**.
    - _Klaster dengan Kalium tinggi (>150 mg/kg):_ Sangat cocok untuk **Apel** dan **Anggur**.
    - _Klaster dengan kelembaban rendah (<40%):_ Sangat cocok untuk **Kacang Arab (_chickpea_)** dan **Kacang Gude**.

---

## Slide 12: Alur 11 - Kesimpulan & Saran

### Kesimpulan

1. **K-Means Clustering** terbukti sebagai model terbaik untuk klasterisasi karakteristik agronomi pada data PCA 5D dengan nilai ARI **0.519**.
2. **Pembersihan outlier global tidak direkomendasikan** karena akan menghapus kelas tanaman bernilai ekstrim alami.

### Saran Pengembangan

1. **Integrasi Web Apps:** Memanfaatkan nilai centroid klaster K-Means sebagai aturan inferensi real-time di aplikasi web pertanian cerdas (_AgriAI_).
2. **Model Probabilistik:** Mencoba metode _Gaussian Mixture Models (GMM)_ untuk memodelkan klaster yang saling tumpang tindih secara lebih fleksibel di masa mendatang.
