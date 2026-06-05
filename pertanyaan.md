# Panduan Tanya Jawab Presentasi UAS Pembelajaran Mesin: AgriAI

Dokumen ini berisi daftar pertanyaan penting yang sering diajukan oleh dosen penguji saat sidang/presentasi proyek machine learning, khususnya untuk kasus **Clustering (K-Means & DBSCAN)** pada rekomendasi komoditas lahan, lengkap dengan draf jawabannya.

---

## Bagian 1: Pemilihan Model & Algoritma

### 1. Mengapa Anda memilih algoritma K-Means dan DBSCAN untuk proyek ini?

- **Jawaban K-Means:** K-Means dipilih sebagai model deployment utama karena efisien untuk data numerik kontinu (N, P, K, pH, suhu, dll.) dan membagi data secara tegas (_centroid-based partitioning_). Hal ini memastikan setiap input karakteristik lahan baru dari petani pasti mendapatkan klaster rekomendasi tanaman yang konkret (tidak ada data yang terbuang).
- **Jawaban DBSCAN:** DBSCAN dipilih sebagai pembanding ilmiah (_density-based_). Tujuannya adalah memvalidasi kerapatan alami titik data dan menyaring pencilan (_outliers/noise_) seperti data cuaca ekstrem atau kegagalan sensor tanah.

### 2. Mengapa hasil metrik evaluasi K-Means jauh lebih bagus daripada DBSCAN pada kasus ini?

- **Jawaban:** K-Means membagi ruang data secara merata ke dalam $K$ klaster bola (_spherical clusters_), yang sangat cocok dengan persebaran dataset rekomendasi tanaman ini yang didesain berkelompok berdasarkan profil tanaman spesifik.
- Sementara **DBSCAN** sangat bergantung pada parameter kerapatan (`eps` dan `min_samples`). Karena kerapatan data di beberapa wilayah sangat bervariasi (_multi-density_), DBSCAN cenderung menyatukan klaster padat menjadi satu klaster besar dan menganggap data berkerapatan rendah lainnya sebagai _noise_ (sehingga Silhouette Score-nya rendah, yaitu ~0.052).

---

## Bagian 2: Preprocessing & Rekayasa Data

### 3. Mengapa Anda melakukan Scaling/Standardisasi (misal MinMaxScaler / StandardScaler) pada data sebelum Clustering?

- **Jawaban:** Algoritma clustering berbasis jarak (seperti K-Means dan DBSCAN) sangat sensitif terhadap perbedaan skala/satuan fitur.
- **Contoh Kasus:** Parameter Nitrogen memiliki rentang nilai `0 - 150`, sedangkan parameter pH tanah hanya berkisar `3 - 10`. Jika tidak di-scale, fitur dengan nilai nominal besar (seperti Nitrogen) akan mendominasi perhitungan jarak _Euclidean_, sehingga parameter pH tidak akan berpengaruh pada hasil klasterisasi. Scaling menyetarakan bobot seluruh fitur agar memiliki kontribusi yang adil.

### 4. Bagaimana Anda menentukan jumlah klaster $K = 22$ pada model K-Means?

- **Jawaban:** Penentuan $K=22$ didasarkan pada dua pertimbangan:
    1.  **Metode Elbow & Silhouette Analysis (Secara Teknis):** Melalui visualisasi kurva inersia (Elbow Method) di Jupyter Notebook untuk mencari titik tekukan optimal, divalidasi dengan nilai Silhouette Score tertinggi.
    2.  **Dataset (Secara Praktis):** Jumlah klaster disesuaikan dengan jumlah label kelas tanaman (komoditas) yang ada di dataset pelatihan (yaitu 22 jenis tanaman, mulai dari padi, jagung, apel, kopi, hingga mangga), sehingga masing-masing klaster dapat merepresentasikan profil ideal dari tiap komoditas tersebut.

---

## Bagian 3: Metrik Evaluasi & Validasi

### 5. Tolong jelaskan arti dari metrik Silhouette Score dan Davies-Bouldin Index!

- **Silhouette Score (Rentang -1 hingga 1):** Mengukur seberapa mirip suatu data dengan klasternya sendiri dibandingkan dengan klaster tetangganya. Skor mendekati `1` berarti klaster terpisah dengan sangat baik. Skor proyek kita (`0.347` pada K-Means) menunjukkan pemisahan yang cukup terdefinisi.
- **Davies-Bouldin Index (Semakin kecil semakin baik):** Mengukur rasio jarak di dalam klaster (_intra-cluster distance_) dengan jarak antar klaster (_inter-cluster distance_). Skor rendah (`1.105` pada K-Means) menunjukkan klaster yang padat di dalam dan terpisah jauh dengan klaster lain.

### 6. Apa itu Adjusted Rand Index (ARI) yang Anda tampilkan di tabel evaluasi?

- **Jawaban:** ARI mengukur tingkat kesamaan (kemiripan struktur) antara hasil pengelompokan klaster tanpa label (unsupervised) dengan label asli tanaman yang ada di dataset (ground truth). Skor ARI berkisar antara `-1` hingga `1` (nilai `1` berarti hasil klasterisasi 100% sama persis dengan pembagian kelas asli).

---

## Bagian 4: Arsitektur Sistem & Web Deployment

### 7. Bagaimana sistem web AgriAI ini memberikan rekomendasi tanaman secara real-time setelah pengguna memasukkan parameter?

- **Jawaban:**
    1.  Pengguna menggeser slider parameter tanah & cuaca di frontend.
    2.  Nilai input tersebut dinormalisasi menggunakan parameter _scaling_ yang sudah dilatih sebelumnya.
    3.  Script JavaScript (`app.js`) menghitung jarak terdekat (_Euclidean Distance_) antara vektor input pengguna dengan **22 koordinat centroid** K-Means yang sudah diekspor dari Jupyter Notebook.
    4.  Sistem mendeteksi klaster terdekat (misalnya Klaster 9), lalu mengambil data rekomendasi tanaman, profil iklim, status nitrogen, dan tindakan agronomi yang berasosiasi dengan klaster tersebut dari kamus data di `app.js`.

### 8. Mengapa Anda melakukan deployment secara Client-Side (di dalam JavaScript) alih-alih menggunakan backend server Flask/FastAPI?

- **Jawaban:** Karena model K-Means yang sudah dilatih hanya memerlukan pencocokan koordinat centroid (vektor 7 dimensi). Menjalankan perhitungan jarak Euclidean untuk 22 centroid sangatlah ringan. Dengan memindahkannya ke sisi klien (_client-side JavaScript_):
    - **Zero Latency:** Inferensi berjalan instan tanpa menunggu request jaringan ke API backend.
    - **Serverless & Cost-Efficient:** Tidak membutuhkan biaya hosting server backend aktif.
    - **Offline Capability:** Aplikasi web dapat berfungsi penuh secara offline setelah halaman pertama dimuat.
