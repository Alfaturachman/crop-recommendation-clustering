# DRAF PENULISAN JURNAL ILMIAH (TARGET: SINTA 3)

## Klasterisasi Karakteristik Lahan & Rekomendasi Tanaman Menggunakan Unsupervised Learning untuk Solusi Multilabel

Dokumen ini berisi panduan, struktur, dan konten draf jurnal ilmiah yang diadaptasi dari proyek UAS Pembelajaran Mesin Anda yang telah diperbarui dengan formulasi masalah multilabel.

---

## 1. STRUKTUR JURNAL ILMIAH (OUTLINE)

### 1.1 Rekomendasi Judul (Pilih Salah Satu)

1. **"Analisis Komparatif Algoritma Unsupervised Learning Berbasis PCA dalam Menyelesaikan Permasalahan Rekomendasi Tanaman Multilabel"** (Fokus pada komparasi metode dan formulasi multilabel).
2. **"Penerapan Soft dan Hard Clustering pada Karakteristik Lahan Agronomi: Pendekatan Tanpa Label untuk Sistem Rekomendasi Tanaman Multi-Komoditas"** (Fokus pada pendekatan soft clustering vs hard clustering).
3. **"Optimasi Sistem Rekomendasi Tanaman Multilabel Berbasis Gaussian Mixture Model (GMM) Menggunakan Karakteristik Agronomi dan Kebijakan Pemeliharaan Outlier"** (Fokus pada model terbaik dan penanganan pencilan).

---

### 1.2 Abstrak / Abstract (Template)

**Abstrak**  
Pemilihan jenis komoditas yang tidak sesuai dengan karakteristik agronomi lahan merupakan penyebab utama rendahnya produktivitas pertanian presisi. Secara akademis, sistem rekomendasi tanaman konvensional sering kali didekati dengan klasifikasi terawasi (*supervised learning*) berlabel tunggal (*single-label*). Namun, secara ekologis dan kebutuhan bisnis tani, sebidang lahan bersifat multi-kompatibel (cocok untuk beberapa jenis tanaman sekaligus), sehingga permasalahan ini secara esensial merupakan kasus **rekomendasi multilabel**. Penelitian ini mengusulkan solusi rekomendasi tanaman multilabel otomatis berbasis *Unsupervised Learning* berbasis reduksi dimensi *Principal Component Analysis* (PCA) 5-komponen untuk mengatasi ketiadaan dataset berlabel multilabel di publik. Eksperimen dilakukan pada 2.200 sampel data lahan pertanian dengan membandingkan enam algoritma klasterisasi: *Gaussian Mixture Model* (GMM), *Fuzzy C-Means* (FCM), *K-Means*, *K-Medoids*, *Hierarchical Clustering*, dan DBSCAN. Berbeda dengan pipeline standar yang menghapus pencilan (*outlier*), penelitian ini mempertahankan data pencilan karena merepresentasikan karakteristik alami spesifik dari komoditas bernilai tinggi seperti apel/anggur (kebutuhan Kalium tinggi) dan padi (curah hujan tinggi). Hasil eksperimen menunjukkan bahwa GMM berkinerja terbaik dengan *Adjusted Rand Index* (ARI) mencapai 0,816, mengungguli Hierarchical (0,609), K-Means (0,519), FCM (0,511), K-Medoids (0,448), dan DBSCAN (0,288). GMM terbukti unggul karena kemampuannya memodelkan batas kepadatan data yang tumpang tindih (*overlapping*) dan mengekstrak probabilitas posterior keanggotaan klaster sebagai dasar rekomendasi tanaman multilabel berperingkat (*ranked multilabel*).  
**Kata Kunci:** Rekomendasi Tanaman Multilabel, Unsupervised Learning, Gaussian Mixture Model, Fuzzy C-Means, Pertanian Presisi, Pemeliharaan Outlier.

---

## 2. BAB I: PENDAHULUAN (INTRODUCTION)

### 2.1 Latar Belakang & Permasalahan

- **Masalah Sektor Pertanian Modern:** Optimalisasi penanaman komoditas memerlukan analisis komparatif parameter hara tanah (N, P, K) dan variabel cuaca mikro (suhu, kelembaban, pH, curah hujan).
- **Keterbatasan Paradigma Klasifikasi Tunggal:** Pendekatan supervised learning memaksa sistem memprediksi satu jenis tanaman saja ($X \to Y$). Hal ini mengabaikan fakta ekologis bahwa sebidang lahan dapat cocok untuk beberapa tanaman sekaligus. Petani kehilangan opsi tanaman alternatif yang berguna untuk rotasi tanaman, tumpang sari, dan diversifikasi risiko kerugian.
- **Tantangan Kelangkaan Data Multilabel:** Dataset pertanian berlabel multilabel (di mana satu baris parameter lahan langsung mencatat beberapa komoditas yang kompatibel) sangat langka dan mahal untuk diproduksi lewat laboratorium.
- **Justifikasi Unsupervised Clustering:** Dengan melakukan klasterisasi tanpa label pada profil tanah, data dikelompokkan berdasarkan kemiripan hara. Setiap klaster secara alami diisi oleh beberapa komoditas tanaman unik. Dengan memetakan lahan baru ke klaster terdekat, sistem secara otomatis menghasilkan rekomendasi multilabel (seluruh tanaman di klaster tersebut) tanpa memerlukan data pelatihan berlabel multilabel.

### 2.2 Kebaruan Penelitian (Novelty)

Penulis harus menekankan tiga poin _novelty_ berikut di bagian akhir bab pendahuluan:

1. **Penyelesaian Kasus Multilabel via Unsupervised Clustering:** Menggunakan klasterisasi tanpa pengawasan label sebagai jembatan untuk memproduksi luaran rekomendasi multilabel dari dataset berlabel tunggal.
2. **Implementasi Klasterisasi Fuzzy & Probabilistik (Soft Clustering):** Menggunakan GMM dan Fuzzy C-Means untuk mengekstrak derajat keanggotaan/probabilitas posterior lahan, memecahkan masalah rigiditas batas klaster kaku pada perbatasan lahan pertanian.
3. **Kebijakan Pemeliharaan Outlier Agronomi (*Outlier Preservation*):** Membuktikan secara kuantitatif bahwa penghapusan outlier statistik global ($\pm 1.5 \times IQR$) akan melenyapkan komoditas bernilai tinggi (apel, anggur, padi) dari sistem rekomendasi.

---

## 3. BAB II: METODE PENELITIAN (METHODOLOGY)

### 2.1 Alur Penelitian (Research Flow)

Metode digambarkan menggunakan diagram alir dengan tahapan:

1. **Pengumpulan Data:** Menggunakan *Crop Recommendation Dataset* (2.200 baris, 7 fitur agronomi, 22 varietas komoditas asli sebagai pembanding).
2. **Preprocessing & Penanganan Outlier:** 
    - Melakukan deteksi pencilan menggunakan Boxplot dan IQR.
    - Mengambil kebijakan mempertahankan pencilan demi keutuhan data taksonomi tanaman ekstrem.
    - Standardisasi fitur menggunakan `StandardScaler`.
3. **Feature Engineering (Reduksi Dimensi PCA):** Menggunakan `PCA(n_components=5)` untuk mempertahankan ~89% variansi kumulatif informasi data, serta `PCA(n_components=2)` khusus untuk kebutuhan visualisasi spasial 2D.
4. **Pemodelan Klaster Komparatif ($K=22$):**
    - **Model Hard Clustering:** K-Means, K-Medoids (PAM), dan Hierarchical (Agglomerative) Clustering.
    - **Model Soft/Probabilistic Clustering:** Gaussian Mixture Model (GMM) dan Fuzzy C-Means (FCM).
    - **Model Density-Based:** DBSCAN (sebagai pembanding analisis spasial kerapatan).
5. **Evaluasi Multivariat:** Perbandingan kinerja model pada ruang PCA 5D menggunakan Adjusted Rand Index (ARI), Silhouette Score, dan Davies-Bouldin Index (DBI).

---

## 4. BAB III: HASIL DAN PEMBAHASAN (RESULTS & DISCUSSION)

### 4.1 Justifikasi Penahanan Outlier (Penting untuk Novelty)

- **Analisis Kuantitatif IQR:** Penerapan eliminasi outlier IQR ($\pm 1.5 \times IQR$) secara standar pada dataset ini akan menghapus 432 data (19.6% dari total dataset).
- **Dampak Eliminasi:** Tanaman **Anggur** dan **Apel** akan **terhapus 100%** dari database karena secara biologis keduanya memerlukan kadar Kalium (K) yang sangat tinggi (>200 mg/kg), yang terdeteksi sebagai outlier statistik ekstrim. Ini membuktikan kebijakan pemeliharaan outlier sangat krusial bagi sistem rekomendasi pertanian.

### 4.2 Perbandingan Performa Algoritma

| Algoritma | Silhouette Score | Davies-Bouldin Index (DBI) | Adjusted Rand Index (ARI) | Karakteristik Rekomendasi |
| :--- | :---: | :---: | :---: | :--- |
| **Gaussian Mixture Model (GMM)** | 0.230 | 1.716 | **0.816** | **Probabilistik (Soft)** - Rekomendasi multilabel berperingkat |
| **Hierarchical Clustering** | 0.288 | 1.098 | 0.609 | Spasial Rigid (Hard) |
| **K-Means Clustering** | **0.307** | **1.081** | 0.519 | Spasial Rigid (Hard) |
| **Fuzzy C-Means (FCM)** | 0.243 | 1.379 | 0.511 | **Fuzzy (Soft)** - Keanggotaan multilabel kontinu |
| **K-Medoids (PAM)** | 0.285 | 1.221 | 0.448 | Spasial Robust (Hard) |
| **DBSCAN** | 0.063 | 1.818 | 0.288 | Density-Based (Terlalu sensitif: 636 data dianggap noise) |

- **Pembahasan Keunggulan GMM:** GMM unggul mutlak pada metrik ARI (0.816). Data agronomi antar tanaman memiliki persebaran yang saling bertumpukan (*overlapping*). GMM merepresentasikan batas kelompok dengan fungsi kepadatan probabilitas (PDF) Gaussian multivariat yang saling tumpang tindih secara luwes, sangat berbeda dari K-Means yang membagi klaster dengan batas linear Voronoi secara kaku.
- **Pembahasan untuk Multilabel:** Pada GMM, sebidang tanah baru diprediksi memiliki probabilitas keanggotaan (misalnya: 70% di Klaster A, 20% di Klaster B, 10% di Klaster C). Dari klaster-klaster tersebut, sistem menyusun rekomendasi tanaman berperingkat (*ranked multilabel recommendation*) berdasarkan kontribusi probabilitas keanggotaan klaster tersebut, memberikan alternatif tanaman yang jauh lebih aman bagi petani.

### 4.3 Interpretasi Centroid & Kelompok Tanaman Multilabel

Beberapa klaster tanah yang terbukti memuat kelompok rekomendasi multilabel logis secara agronomi:
- **Klaster Nutrisi Ekstrim (Apel & Anggur):** Terbentuk pada klaster dengan rata-rata Kalium >200 mg/kg dan Fosfor >130 mg/kg.
- **Klaster Sawah Tropika Basah (Padi & Yute):** Terbentuk pada kondisi curah hujan >180 mm dan kelembaban >80%.
- **Klaster Lahan Kering Rendah Hara (Sorgum & Millet):** Terbentuk pada curah hujan rendah (<80 mm) dan kandungan nitrogen moderat.

---

## 5. BAB IV: KESIMPULAN (CONCLUSION)

### 5.1 Kesimpulan

1. Sistem rekomendasi komoditas pertanian multilabel berhasil diimplementasikan menggunakan pendekatan *unsupervised clustering* tanpa memerlukan anotasi multilabel manual yang mahal.
2. *Gaussian Mixture Model* (GMM) pada ruang fitur PCA 5D merupakan model terbaik untuk menangani batas agronomi tanah yang tumpang tindih dengan akurasi pengelompokan tertinggi (ARI = 0,816).
3. Kebijakan pemeliharaan outlier terbukti mutlak diperlukan untuk menjaga integritas taksonomi tanaman bernilai tinggi yang memiliki kebutuhan hara ekstrem.

### 5.2 Saran

1. **Ensemble Soft-Clustering:** Penelitian selanjutnya dapat mengombinasikan keanggotaan Fuzzy C-Means dan probabilitas posterior GMM untuk mereduksi ketidakpastian perbatasan iklim mikro.
2. **Integrasi Data Temporal:** Memasukkan parameter waktu/musim tanam secara dinamis untuk melengkapi keluaran sistem rekomendasi multilabel agronomi.

---

## 6. REKOMENDASI REFERENSI UNTUK DAFTAR PUSTAKA

Untuk Sinta 3, Anda membutuhkan minimal 15-20 referensi. Tambahkan referensi yang relevan dengan kata kunci berikut:
1. *Precision agriculture multilabel crop recommendation using machine learning.*
2. *Comparative study of GMM, Fuzzy C-Means, and K-Means for multivariate soil classification.*
3. *The impact of outlier preservation in agricultural datamining.*
4. *Fuzzy decision support systems for crop rotation and intercropping.*
