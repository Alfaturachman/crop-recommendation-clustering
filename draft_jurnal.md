# DRAF PENULISAN JURNAL ILMIAH (TARGET: SINTA 3)

## Klasterisasi Karakteristik Lahan & Rekomendasi Tanaman Menggunakan Unsupervised Learning

Dokumen ini berisi panduan, struktur, dan konten draf jurnal ilmiah yang diadaptasi dari proyek UAS Pembelajaran Mesin Anda.

---

## 1. STRUKTUR JURNAL ILMIAH (OUTLINE)

### 1.1 Rekomendasi Judul (Pilih Salah Satu)

1. **"Analisis Komparatif Algoritma K-Means dan DBSCAN untuk Rekomendasi Komoditas Pertanian Berdasarkan Karakteristik Agronomi"** (Fokus pada komparasi metode).
2. **"Penerapan Unsupervised Learning Berbasis Karakteristik Lahan: Pendekatan Pemeliharaan Outlier untuk Akurasi Rekomendasi Tanaman"** (Fokus pada novelty penanganan outlier).
3. **"Sistem Pendukung Keputusan Pemilihan Tanaman Pangan Menggunakan Klasterisasi K-Means Berdasarkan Fitur Hara dan Iklim"** (Fokus pada aplikasi praktis).

---

### 1.2 Abstrak / Abstract (Template)

**Abstrak**  
Pemilihan jenis tanaman yang tidak sesuai dengan karakteristik agronomi lahan merupakan salah satu penyebab utama rendahnya produktivitas pertanian. Penelitian ini mengusulkan solusi pengelompokan lahan otomatis berbasis _Unsupervised Learning_ menggunakan algoritma K-Means dan DBSCAN pada 2.200 data sampel lahan dengan 7 fitur agronomi (N, P, K, suhu, kelembaban, pH, dan curah hujan). Reduksi dimensi dengan Principal Component Analysis (PCA) 5 komponen digunakan untuk mempertahankan informasi yang tinggi (~89% variansi kumulatif) dan mengoptimalkan modeling klaster. Berbeda dengan pipeline standar yang menghapus pencilan (outlier), penelitian ini mempertahankan data pencilan karena merepresentasikan karakteristik alami spesifik dari komoditas tertentu seperti anggur dan apel yang membutuhkan kadar Kalium (K) tinggi. Hasil pengujian menunjukkan bahwa pada ruang PCA 5D, K-Means memiliki kecocokan yang baik dengan varietas asli lahan dengan _Adjusted Rand Index_ (ARI) sebesar 0,519, _Silhouette Score_ 0,307, dan _Davies-Bouldin Index_ (DBI) 1,081. Sebaliknya, DBSCAN (eps=0.6, min_samples=14) berkinerja kurang optimal dengan ARI sebesar 0,288 disebabkan batas kepadatan data antar tanaman yang saling tumpang tindih (_overlapping_), sehingga mengkategorikan 636 data (28,91%) sebagai _noise_. Nilai centroid hasil klasterisasi K-Means kemudian diekstraksi menjadi basis aturan keputusan untuk aplikasi web pendukung keputusan petani secara _real-time_.  
**Kata Kunci:** K-Means, DBSCAN, Klasterisasi Lahan, Pertanian Cerdas, Pemeliharaan Outlier.

---

## 2. BAB I: PENDAHULUAN (INTRODUCTION)

### 2.1 Latar Belakang & Permasalahan

- **Masalah Sektor Pertanian:** Petani sering kali menentukan jenis tanaman berdasarkan kebiasaan atau perkiraan, bukan berdasarkan data kandungan hara tanah (N, P, K) dan parameter iklim mikro (suhu, kelembaban, pH, curah hujan) yang aktual.
- **Kebutuhan Sistem Otomatis:** Perlu adanya model komputasi yang mampu mengelompokkan karakteristik lahan secara objektif tanpa pengawasan label (_unsupervised_).

### 2.2 Kebaruan Penelitian (Novelty)

Penulis harus menekankan tiga poin _novelty_ berikut di bagian akhir bab pendahuluan:

1. **Kebijakan Pemeliharaan Outlier (_Outlier Preservation_):** Menolak anggapan umum bahwa semua outlier statistik harus dibuang. Dalam domain agronomi, data bernilai ekstrim mencerminkan kebutuhan biologis tanaman khusus.
2. **Analisis Geometris Data Lahan:** Mengidentifikasi bahwa data agronomi memiliki kecenderungan pengelompokan terpusat (_centroid-based_) dibanding spasial kerapatan (_density-based_).
3. **Model Integrasi Inferensi:** Menghubungkan titik centroid klaster K-Means langsung menjadi logika inferensi pada aplikasi web pendukung keputusan.

---

## 3. BAB II: METODE PENELITIAN (METHODOLOGY)

### 2.1 Alur Penelitian (Research Flow)

Metode digambarkan menggunakan diagram alir dengan tahapan:

1. **Pengumpulan Data:** Menggunakan _Crop Recommendation Dataset_ (2.200 baris, 22 kelas).
2. **Preprocessing:**
    - Deteksi duplikasi & nilai hilang (_missing values_).
    - Analisis Outlier menggunakan _Boxplot_ (diputuskan untuk dipertahankan).
    - Standardisasi Fitur menggunakan _StandardScaler_ untuk mengatasi perbedaan skala (misal: curah hujan vs pH).
3. **Reduksi Dimensi (PCA):** Menggunakan `PCA(n_components=5)` untuk pemodelan (mempertahankan ~89% variansi kumulatif) dan `PCA(n_components=2)` hanya untuk visualisasi hasil klasterisasi secara 2D.
4. **Pemodelan Klaster:**
    - **K-Means:** Penentuan $K=22$ berdasarkan analisis Elbow & Silhouette pada data PCA 5D.
    - **DBSCAN:** Penentuan $\epsilon=0.6$ dan $MinPts=10$ (untuk $2 \times D$, $D=5$) menggunakan K-Distance Graph, serta eksperimen perbandingan dengan $\epsilon=0.6$ dan $MinPts=14$.
5. **Evaluasi:** Perbandingan kinerja pada ruang PCA 5D menggunakan Silhouette Score, Davies-Bouldin Index (DBI), dan Adjusted Rand Index (ARI).

---

## 4. BAB III: HASIL DAN PEMBAHASAN (RESULTS & DISCUSSION)

### 4.1 Justifikasi Penahanan Outlier (Penting untuk Novelty)

- Tampilkan grafik Boxplot sebelum pemodelan.
- Berikan penjelasan matematis: _Jika metode IQR ($\pm 1.5 \times IQR$) diterapkan secara global:_
    - Sebanyak 432 data (19.6%) akan terhapus.
    - Tanaman **Anggur** dan **Apel** akan **terhapus 100%** dari dataset karena kebutuhan Kalium (K) yang sangat tinggi secara alami terdeteksi sebagai outlier statistik.
    - Hal ini membuktikan bahwa penahanan outlier mempertahankan keutuhan taksonomi data.

### 4.2 Perbandingan Performa Algoritma

| Algoritma                                | Silhouette Score (Semua Data) | Silhouette Score (Tanpa Noise) | Davies-Bouldin Index | Adjusted Rand Index (ARI) |
| :--------------------------------------- | :---------------------------: | :----------------------------: | :------------------: | :-----------------------: |
| **K-Means (K=22) pada PCA 5D**           |           **0,307**           |               -                |      **1,081**       |         **0,519**         |
| **DBSCAN ($\epsilon=0.6$, $MinPts=14$)** |             0,063             |             0,295              |        1,818         |           0,288           |

- **Pembahasan:** K-Means unggul karena data pertanian membentuk densitas terpusat di sekitar centroid tanaman masing-masing. DBSCAN gagal memisahkan klaster dengan baik karena batas persebaran tanaman saling tumpang tindih (_overlapping_), mengakibatkan 636 titik valid dianggap sebagai data sampah (_noise_).

### 4.3 Interpretasi Centroid Klaster

Tuliskan beberapa contoh hasil klasterisasi K-Means yang terbukti valid secara ilmiah:

- **Klaster Dataran Tinggi Basah (Cluster 1):** Karakteristik N tinggi dan hujan tinggi sangat cocok untuk tanaman **Kopi**.
- **Klaster Hara Ekstrem (Cluster 3):** Kalium (>200 mg/kg) dan Fosfor (>130 mg/kg) sangat cocok untuk tanaman **Apel & Anggur**.
- **Klaster Sawah Basah (Cluster 9):** Curah hujan tinggi (>180 mm) sangat cocok untuk **Padi & Yute**.

---

## 5. BAB IV: KESIMPULAN (CONCLUSION)

### 5.1 Kesimpulan

1. K-Means dengan $K=22$ merupakan model terbaik untuk klasterisasi karakteristik agronomi dengan nilai ARI 0,519 (pada data PCA 5D).
2. Pembersihan pencilan secara statistik global merusak representasi varietas tanaman ekstrim sehingga data outlier harus dipertahankan.
3. Struktur persebaran data agronomi bersifat _centroid-based_, bukan _density-based_.

### 5.2 Saran

1. Eksplorasi algoritma klasterisasi probabilistik seperti _Gaussian Mixture Models (GMM)_ untuk menangani tumpang tindih sebaran secara lebih luwes.
2. Integrasi model yang telah diekspor (`joblib`) ke sistem informasi pertanian di tingkat pedesaan.

---

## 6. REKOMENDASI REFERENSI UNTUK DAFTAR PUSTAKA

Untuk Sinta 3, Anda membutuhkan minimal 15-20 referensi, dengan 80% berasal dari jurnal ilmiah (terutama jurnal lokal terakreditasi Sinta 1-4 atau Scopus).

- **Topik Referensi yang dicari:**
    1. Penerapan K-Means untuk pemetaan lahan pertanian.
    2. Studi perbandingan K-Means vs DBSCAN pada data multivariat.
    3. Penggunaan machine learning untuk sistem rekomendasi tanaman pangan.
    4. Dampak preprocessing dan penanganan outlier pada hasil clustering.
