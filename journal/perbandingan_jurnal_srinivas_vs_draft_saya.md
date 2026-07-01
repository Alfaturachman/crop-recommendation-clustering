# TINJAUAN METODOLOGI DAN PERBANDINGAN JURNAL ACUAN

Dokumen ini mendokumentasikan tinjauan metodologi terhadap penelitian yang telah dipublikasikan oleh Srinivas et al. (2026) serta komparasi objektif dengan draf usulan penelitian hibrida dua tahap (_Two-Stage Hybrid Pipeline_).

---

## I. TINJAUAN METODOLOGI JURNAL ACUAN

- **Judul Jurnal:** _Utilizing Supervised and Unsupervised Machine Learning Techniques for Crop Yield Prediction, Pest Detection, And Precision Farming_
- **Penulis:** Dr. Inumarthi V. Srinivas, Megha Dhotay, Dr. J. Sridevi, Ritika Sanwal, Nikita Jain
- **Tahun Publikasi:** Maret 2026 (J Int Commer Law Technol)
- **Dataset Utama:** Crop Recommendation Dataset (Kaggle) - 2.200 baris, 8 variabel.

### 1. Deskripsi Umum Jurnal (Overview & Core Topic)

Penelitian acuan mengeksplorasi penerapan algoritma pembelajaran mesin terawasi (_supervised_) dan tidak terawasi (_unsupervised_) pada dataset rekomendasi tanaman publik untuk mendukung pertanian presisi.

- **Analisis Terawasi (Supervised)**: Menggunakan _Logistic Regression_ dan _Random Forest_ untuk mengklasifikasikan 22 komoditas tanaman berdasarkan parameter hara tanah (N, P, K) dan parameter iklim (suhu, kelembaban, pH, curah hujan). Evaluasi menunjukkan akurasi sebesar **0.9727** untuk Logistic Regression dan **0.9955** untuk Random Forest.
- **Analisis Tidak Terawasi (Unsupervised)**: Menggunakan _K-Means_ dan _Agglomerative Clustering_ untuk segmentasi lahan tanpa menyertakan label komoditas. Berdasarkan visualisasi grafik Elbow, penulis memilih konfigurasi **$K=4$** klaster yang menghasilkan Silhouette Score sebesar **0.3229** (K-Means) dan **0.3468** (Agglomerative).
- **Deteksi Hama (Pest Detection)**: Menyajikan alur konseptual (diagram alir) terkait arsitektur sistem deteksi hama di sektor pertanian.

### 2. Kontribusi / Kebaruan yang Diajukan Penulis

- **Pendekatan Analisis Ganda**: Memadukan analisis klasifikasi untuk rekomendasi komoditas dan klasterisasi untuk pengelompokan karakteristik wilayah pertanian dalam satu kajian.
- **Visualisasi Spasial Berbasis PCA**: Mengurangi dimensi fitur menggunakan _Principal Component Analysis_ (PCA) menjadi 2 komponen utama untuk menyajikan visualisasi spasial sebaran klaster secara dua dimensi.

### 3. Keterbatasan Penelitian (Research Limitations)

Dalam tinjauan akademis, terdapat beberapa batasan ruang lingkup pada penelitian acuan yang berpotensi untuk dikembangkan lebih lanjut:

- **Batasan Ruang Lingkup Judul vs Implementasi**:
    - Judul mencantumkan kata kunci _"Crop Yield Prediction"_ dan _"Pest Detection"_. Namun, pemodelan data hasil panen (yield) dan klasifikasi hama berbasis gambar belum diimplementasikan di dalam eksperimen utama. Topik deteksi hama disajikan secara konseptual dalam bentuk diagram alir pada Bab 4.
- **Penyederhanaan Jumlah Klaster ($K=4$)**:
    - Dataset asli mencakup 22 komoditas dengan kebutuhan tanah dan iklim yang unik. Keputusan menggunakan $K=4$ klaster membatasi resolusi pengelompokan, sehingga varietas tanaman dengan profil hara yang bertolak belakang terkelompokkan ke dalam zona yang sama.
- **Metrik Evaluasi Klastering**:
    - Evaluasi klastering berfokus pada metrik spasial-geometris (Silhouette Score) tanpa menyertakan metrik validitas ekologis untuk mengukur keselarasan hasil pengelompokan dengan rumpun tanaman yang sesungguhnya (seperti _Adjusted Rand Index_ atau _Cluster Purity_).
- **Aliran Informasi Non-Hibrida (Siloed Approach)**:
    - Modul supervised dan unsupervised dijalankan secara independen tanpa adanya integrasi fungsional di mana luaran satu model memperkuat model lainnya.

---

## II. _RESEARCH GAP_ DAN ARAH PENGEMBANGAN JURNAL SAYA

Keterbatasan pada penelitian acuan membuka peluang penelitian (_research gap_) yang menjadi fokus pengembangan draf jurnal saya melalui pendekatan _Two-Stage Hybrid Pipeline_:

### Gap 1: Perluasan ke Rekomendasi Multilabel (_Multi-Label Extension_)

- **Penelitian Acuan**: Menggunakan pendekatan klasifikasi kelas tunggal (_single-label_) di mana sistem merekomendasikan satu jenis komoditas saja untuk satu sampel tanah.
- **Usulan Draf Saya**: Mengembangkan sistem rekomendasi **multilabel** (_multi-label crop recommendation_) agar sistem dapat memberikan rekomendasi beberapa tanaman sekaligus yang cocok dibudidayakan bersamaan (tumpang sari) atau bergiliran (rotasi tanaman) pada lahan dengan kondisi hara yang sama.

### Gap 2: Penentuan Jumlah Klaster Terarah ($K=22$) sebagai Generator Target

- **Penelitian Acuan**: Membatasi klaster pada $K=4$ untuk pengelompokan spasial kasar.
- **Usulan Draf Saya**: Menggunakan konfigurasi $K=22$ sesuai dengan jumlah kelas komoditas asli. Langkah ini ditujukan untuk menguji kemampuan algoritma mempartisi relung ekologi komoditas secara mandiri, kemudian mentransformasikan label klaster tersebut menjadi target biner multilabel (_multi-hot encoding_) untuk melatih pengklasifikasi Fase 2.

### Gap 3: Penanganan Korelasi Hara Tanah (K-Means vs GMM)

- **Penelitian Acuan**: Hanya menerapkan algoritma berbasis jarak geometris (K-Means dan Agglomerative) yang mengasumsikan klaster berbentuk bulat/sferis. Model ini mengabaikan korelasi hara tanah yang kuat (seperti hubungan linier Fosfor-Kalium dengan koefisien korelasi 0,74).
- **Usulan Draf Saya**: Memperbandingkan kinerja K-Means dengan **Gaussian Mixture Model (GMM)** bermatriks kovariansi penuh. GMM terbukti lebih adaptif terhadap bentuk sebaran data elips alami hara tanah, ditunjukkan dengan nilai Adjusted Rand Index yang lebih representatif (**0,816** vs K-Means **0,519**).

### Gap 4: Analisis Dampak Data Pencilan terhadap Komoditas Khusus

- **Penelitian Acuan**: Tidak menguraikan proses penanganan data pencilan (_outliers_).
- **Usulan Draf Saya**: Menganalisis secara khusus bahwa pembersihan pencilan secara konvensional (misalnya dengan metode IQR) berisiko menghapus komoditas bernilai tinggi seperti **Apel dan Anggur** (100% sampel terhapus) karena tanaman tersebut secara biologis membutuhkan kadar Kalium yang sangat tinggi. Kebijakan preservasi pencilan (_outlier preservation_) diusulkan untuk menjaga validitas agronomi sistem rekomendasi.

---

## III. MATRIKS PERBANDINGAN METODOLOGI & ESTIMASI HASIL

| Parameter Perbandingan         | Jurnal Srinivas et al. (2026)                  | Usulan Draf Jurnal (Two-Stage GMM-RF)                       |
| :----------------------------- | :--------------------------------------------- | :---------------------------------------------------------- |
| **Integrasi Model**            | Terpisah (Supervised & Unsupervised Mandiri)   | **Terintegrasi (Fase 1 menghasilkan label untuk Fase 2)**   |
| **Sifat Rekomendasi**          | _Single-Label_ (Rekomendasi 1 tanaman tunggal) | **Multi-Label** (Rekomendasi $\geq 1$ komoditas)            |
| **Metode Klastering**          | K-Means & Agglomerative ($K=4$)                | **K-Means vs GMM ($K=22$)**                                 |
| **Adjusted Rand Index**        | Tidak diukur                                   | K-Means: `0,5194` vs **GMM: `0,8157`**                      |
| **Kemurnian Klaster (Purity)** | Tidak diukur                                   | K-Means: `69,97%` vs **GMM: `90,17%`**                      |
| **Akurasi Klasifikasi**        | Random Forest _Single-Label_: `99,55%`         | MultiOutput Random Forest (Subset Acc): **`83,03%`**        |
| **Penanganan Pencilan**        | Tidak dirinci                                  | **Preservasi Teranalisis** (Mencegah eliminasi Apel/Anggur) |

---

## IV. BUKTI VISUAL: FRAGMENTASI SPASIAL K-MEANS VS. KEMURNIAN EKOLOGIS GMM

Untuk memperkuat justifikasi pemilihan model klasterisasi kepada dosen pembimbing atau reviewer, grafik sebaran komposisi komoditas pada masing-masing klaster dapat dirujuk pada berkas:  
📂 **`cluster_composition_proof.png`** (terletak pada direktori utama proyek).

### Temuan Utama pada Grafik Perbandingan:

1.  **Fragmentasi Geometris K-Means (Sisi Kiri Grafik)**:
    - K-Means membagi sampel tanah secara merata berdasar jarak Euclidean tanpa memedulikan batas biologis komoditas.
    - Akibatnya, terjadi pencampuran tanaman yang tidak sejenis di dalam satu klaster (misal: _Klaster 0_ mencampur komoditas Orange, Pomegranate, dan Coffee).
    - Hal ini berbahaya bagi rekomendasi pertanian karena kebutuhan nutrisi ketiga tanaman tersebut berbeda secara signifikan.
2.  **Kemurnian Ekologis GMM (Sisi Kanan Grafik)**:
    - GMM menyesuaikan bentuk distribusinya (elipsoid) dengan kontur alami parameter tanah.
    - Hasilnya, hampir seluruh klaster bersifat monospesifik (murni berisi 1 komoditas dengan purity rata-rata **90,17%**).
    - Pada klaster transisi basah (_Klaster 1_ dan _Klaster 7_), GMM secara cerdas hanya mencampurkan komoditas **Jute** dan **Rice** yang memang tumbuh berdampingan secara alami di lahan basah dengan tingkat pH yang sedikit berbeda (gradasi kelembaban).

---

## V. ALIRAN KERJA TEKNIS PIPELINE HIBRIDA DUA TAHAP

Berikut adalah langkah-langkah bagaimana Fase 1 (Unsupervised) berkesinambungan dengan Fase 2 (Supervised):

1.  **Tahap 1 (GMM Clustering)**:
    - Seluruh data parameter tanah (N, P, K, pH, curah hujan, dll.) dimasukkan ke dalam model GMM dengan $K=22$.
    - Setiap baris data mendapatkan probabilitas keanggotaan untuk 22 klaster tersebut.
2.  **Tahap 2 (Ekstraksi Pengetahuan & Binarisasi)**:
    - Untuk setiap klaster yang terbentuk, dicari komoditas tanaman dominan yang tumbuh di sana.
    - Data pemetaan ini digunakan sebagai basis pembentukan target biner multilabel. Setiap sampel tanah dikonversi menjadi representasi vektor biner $y \in \{0, 1\}^{22}$ menggunakan `MultiLabelBinarizer` (angka `1` untuk tanaman cocok, `0` untuk tanaman tidak cocok).
3.  **Tahap 3 (Pelatihan MultiOutput Classifier)**:
    - Model _MultiOutput Random Forest Classifier_ dilatih menggunakan fitur tanah asli sebagai input ($X$), dan vektor multilabel hasil binarisasi Fase 1 sebagai target ($y$).
    - Model terlatih mampu memberikan rekomendasi multilabel instan untuk input tanah baru dengan subset accuracy sebesar **83,03%**.
