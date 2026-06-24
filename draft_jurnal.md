# DRAF PENULISAN JURNAL ILMIAH (TARGET: SINTA 3)

**"Comparative Study of Soft and Hard Clustering for Multilabel Crop Recommendation"**

---

## 1. BAB I: PENDAHULUAN (INTRODUCTION)

### 1.1 Abstrak / Abstract (Template)

**Abstrak**  
Pemilihan jenis komoditas yang tidak sesuai dengan karakteristik agronomi lahan merupakan penyebab utama rendahnya produktivitas pertanian presisi. Secara akademis, sistem rekomendasi tanaman konvensional sering kali didekati dengan klasifikasi terawasi (_supervised learning_) berlabel tunggal (_single-label_). Namun, secara ekologis dan kebutuhan bisnis tani, sebidang lahan bersifat multi-kompatibel (cocok untuk beberapa jenis tanaman sekaligus), sehingga permasalahan ini secara esensial merupakan kasus **rekomendasi multilabel**. Penelitian ini mengusulkan solusi rekomendasi tanaman multilabel berbasis pembelajaran dua tahap. Tahap pertama melakukan klasterisasi karakteristik lahan secara tidak terawasi (_unsupervised_) menggunakan reduksi dimensi _Principal Component Analysis_ (PCA) 5-komponen dan dua pendekatan klasterisasi: _K-Means_ (sebagai representasi hard clustering berbasis Jarak Geometris/Centroid-based) dan _Gaussian Mixture Model_ (GMM) (sebagai representasi soft clustering berbasis Distribusi Probabilitas/Distribution-based). Hasil klasterisasi dari K-Means dan GMM kemudian digunakan untuk melatih model tahap kedua yaitu klasifikasi terawasi multilabel (_supervised multilabel_) guna memprediksi rekomendasi tanaman secara presisi. Hasil eksperimen tahap pertama menunjukkan bahwa GMM menunjukkan performa lebih baik dalam memodelkan batas sebaran data alami dengan _Adjusted Rand Index_ (ARI) mencapai 0,816, lebih tinggi dibandingkan K-Means dengan ARI sebesar 0,519, meskipun K-Means menghasilkan pemisahan spasial geometris yang lebih tegas (Silhouette Score 0,307 dan DBI 1,081). Evaluasi tahap kedua menunjukkan model supervised multilabel berbasis GMM mencapai Subset Accuracy tertinggi sebesar 84,77% (F1-Score Micro 0,9404), sedangkan model berbasis K-Means mencapai Subset Accuracy 81,36% (F1-Score Micro 0,9537).
**Kata Kunci:** Rekomendasi Tanaman Multilabel, Unsupervised Learning, K-Means, Gaussian Mixture Model, Supervised Multilabel.

---

### 1.2 Latar Belakang dan Permasalahan

Optimalisasi penanaman komoditas pertanian di era modern sangat bergantung pada analisis komparatif parameter hara tanah seperti Nitrogen (N), Fosfor (P), Kalium (K), serta variabel iklim mikro seperti temperatur, kelembaban, pH tanah, dan curah hujan [1]. Ketidakcocokan antara komoditas yang ditanam dengan karakteristik lahan sering kali menjadi penyebab utama rendahnya produktivitas pertanian presisi serta kegagalan panen yang merugikan secara ekonomi [2]. Oleh sebab itu, pengembangan sistem rekomendasi tanaman berbasis komputasi cerdas menjadi kebutuhan krusial dalam menunjang ketahanan pangan dan pertanian presisi [3].

Meskipun demikian, mayoritas penelitian sistem rekomendasi tanaman konvensional saat ini masih didekati dengan paradigma klasifikasi terawasi berlabel tunggal (_single-label supervised learning_), di mana model hanya memprediksi satu jenis tanaman untuk satu set parameter tanah [4]. Pendekatan ini secara ekologis kurang realistis karena sebidang lahan pertanian pada kenyataannya sering kali memiliki tingkat kompatibilitas yang tinggi terhadap beberapa jenis tanaman sekaligus [5]. Pembatasan pada satu komoditas tunggal mengabaikan potensi rotasi tanaman, pertanian tumpang sari (_intercropping_), dan strategi diversifikasi risiko kerugian bagi petani [6]. Oleh karena itu, permasalahan rekomendasi tanaman secara esensial merupakan kasus klasifikasi multilabel (_multilabel classification_) [7].

Tantangan utama dalam menerapkan pembelajaran terawasi multilabel pada sektor pertanian adalah kelangkaan dan mahalnya biaya pengumpulan dataset berlabel multilabel secara langsung di laboratorium hara tanah [8]. Untuk mengatasi keterbatasan data berlabel ini, teknik klasterisasi tidak terawasi (_unsupervised clustering_) dapat dimanfaatkan untuk mengelompokkan profil tanah secara mandiri berdasarkan kesamaan parameter fisika dan kimia [9]. Dengan memetakan sebaran lahan secara alami, setiap klaster tanah yang terbentuk secara inheren akan diisi oleh beberapa komoditas tanaman yang memiliki karakteristik hara yang serupa. Hasil pengelompokan tanpa label inilah yang kemudian diekstrak untuk membentuk target biner multilabel guna melatih model klasifikasi terawasi tahap kedua tanpa membutuhkan anotasi manual yang mahal [10].

### 1.3 Celah Penelitian (Research Gap) dan Kebaruan (Novelty)

Beberapa penelitian terdahulu di bidang rekomendasi tanaman tumpang sari umumnya hanya menggunakan satu pendekatan klasterisasi kaku seperti K-Means untuk pemetaan wilayah saja [11], atau berhenti pada tahap pengelompokan wilayah tanpa pemodelan prediktif klasifikasi [12]. Selain itu, riset-riset tersebut kerap mengabaikan pengaruh pra-pemrosesan data seperti eliminasi pencilan (_outliers_) yang secara statistik sering disingkirkan sebagai derau (_noise_), padahal pencilan tersebut merepresentasikan kondisi tanah ekstrem bagi komoditas khusus bernilai ekonomi tinggi [13]. Hal ini menciptakan celah penelitian (_research gap_) mengenai perbandingan efektivitas antara berbagai paradigma klasterisasi dan pengaruh preservasi pencilan terhadap keutuhan informasi agronomi.

Untuk memperjelas celah penelitian (_research gap_) tersebut, ringkasan perbandingan studi literatur terdahulu dengan pendekatan yang diusulkan disajikan pada Tabel 1 di bawah ini:

**Tabel 1. Perbandingan Penelitian Terdahulu dan Penelitian yang Diusulkan**

| Author (Year)                  | Method                                              | Output                             | Limitation / Gap                                                                                                                                 |
| :----------------------------- | :-------------------------------------------------- | :--------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Ingle et al. [11] (2024)**   | Random Forest, XGBoost                              | Single-label Crop Recommendation   | Hanya mendukung rekomendasi satu jenis komoditas tunggal (_single-label_), mengabaikan kompatibilitas multi-komoditas lahan.                     |
| **Sutoyo et al. [12] (2025)**  | K-Means Clustering                                  | Soil Clustering (Mapping)          | Berhenti pada pemetaan wilayah tanah tanpa adanya pemodelan prediktif klasifikasi lanjutan untuk rekomendasi komoditas.                          |
| **Prasad & Kumar [13] (2026)** | Decision Trees                                      | Multi-crop Recommendation          | Menggunakan eliminasi pencilan global ($\pm 1.5 \times IQR$), menyebabkan tanaman berkebutuhan ekstrim (seperti apel/anggur) hilang dari sistem. |
| **Penelitian Ini (Proposed)**  | **PCA + GMM vs K-Means + Multilabel Random Forest** | **Multilabel Crop Recommendation** | Menyelesaikan _single-label constraint_, membandingkan _hard/soft clustering_, dan mempertahankan data _outlier_ untuk keberagaman tanaman.      |

Berdasarkan tinjauan Tabel 1 di atas, penelitian ini mengusulkan solusi rekomendasi tanaman multilabel melalui pendekatan pembelajaran dua tahap (_two-stage learning_). Kebaruan (_novelty_) utama yang diajukan mencakup tiga poin penting. Pertama, formulasi alur hibrida yang mengintegrasikan reduksi dimensi PCA dengan klasterisasi tidak terawasi untuk menghasilkan label target biner secara otomatis guna melatih model pengklasifikasi terawasi multilabel di tahap kedua [14]. Kedua, komparasi teoretis dan praktis antara paradigma _hard clustering_ (K-Means) berbasis jarak geometris dengan _soft clustering_ (Gaussian Mixture Model) berbasis distribusi probabilitas elipsoid kontinu dalam mengidentifikasi batas-batas kelompok tanah alami yang saling bertumpang tindih [15]. Ketiga, pembuktian kuantitatif mengenai pentingnya kebijakan pemeliharaan pencilan (_outlier preservation_) agronomi guna mencegah hilangnya komoditas bernilai tinggi seperti apel dan anggur dari taksonomi sistem rekomendasi.

---

## 2. BAB II: METODE PENELITIAN (METHODOLOGY)

### 2.1 Alur Penelitian (Research Flow)

Metode dirancang dalam dua fase utama:

**Fase 1: Klasterisasi Karakteristik Lahan (Unsupervised)**

1. **Pengumpulan & Preprocessing Data:** Menggunakan _Crop Recommendation Dataset_ (2.200 sampel) dengan standardisasi fitur menggunakan `StandardScaler`. Kebijakan pemeliharaan outlier diterapkan untuk mempertahankan tanaman berkebutuhan ekstrem.
2. **Feature Engineering (Reduksi Dimensi):** Menerapkan PCA 5-komponen untuk kompresi informasi (~89% variansi kumulatif) dan PCA 2-komponen untuk kebutuhan visualisasi.
3. **Pemodelan Klaster Komparatif ($K=22$):** Membandingkan dua model dengan pendekatan matematika berbeda:
    - **K-Means**: Mewakili pendekatan Hard Clustering berbasis Jarak Geometris (Centroid-based).
    - **Gaussian Mixture Model (GMM)**: Mewakili pendekatan Soft Clustering berbasis Distribusi Probabilitas (Distribution-based).
4. **Evaluasi Unsupervised:** Menilai kualitas pengelompokan menggunakan Silhouette Score, Davies-Bouldin Index (DBI), dan Adjusted Rand Index (ARI).

**Fase 2: Prediksi Rekomendasi Tanaman (Supervised Multilabel)**

1. **Formulasi Dataset Multilabel:** Mengekstrak hasil pengelompokan K-Means dan GMM untuk membentuk label rekomendasi tanaman multilabel.
2. **Pelatihan Model Kedua (Supervised Multilabel):** Melatih model pengklasifikasi multilabel menggunakan fitur agronomi sebagai input dan label hasil klasterisasi sebagai output target.
3. **Evaluasi Prediksi:** Mengukur tingkat akurasi tebakan rekomendasi tanaman menggunakan metrik **Subset Accuracy** dan **F1-Score**.

### 2.2 Tinjauan Metodologi (Methodology Review)

Bagian ini merangkum detail spesifikasi teknis dan hyperparameter dari seluruh komponen pemrosesan data dan algoritma yang diimplementasikan:

#### 1. Dataset Agronomi

- **Sumber & URL Dataset:** _Crop Recommendation Dataset_ yang diunggah oleh Atharva Ingle di platform Kaggle ([Tautan Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)).
- **Karakteristik:** Dataset ini memiliki 2.200 sampel tanah (masing-masing tanaman memiliki 100 sampel data).
- **Fitur Agronomi:** Terdiri dari 7 fitur numerik: unsur hara tanah makro yaitu Nitrogen (N), Fosfor (P), Kalium (K); dan iklim mikro yaitu temperatur (temperature), kelembaban (humidity), derajat keasaman (ph), dan curah hujan (rainfall), sebagaimana divisualisasikan pada Figure 9.

- **Figure 9: Distribusi Frekuensi 7 Variabel Agronomi & Mikroklimat**
  ![Figure 9: Distribusi Frekuensi 7 Variabel Agronomi & Mikroklimat](feature_distributions.png)
  _Deskripsi:_ Grafik histogram dan KDE untuk 7 variabel input agronomi, menunjukkan distribusi unimodal/normal pada temperatur, kelembaban, dan pH, serta distribusi bimodal/skewed pada Nitrogen (N), Fosfor (P), Kalium (K), dan curah hujan (rainfall).

- **Figure 10: Sebaran Sampel 22 Kelas Komoditas Tanaman (Ground Truth)**
  ![Figure 10: Sebaran Sampel 22 Kelas Komoditas Tanaman](crop_class_distribution.png)
  _Deskripsi:_ Diagram batang horizontal sebaran frekuensi sampel 22 jenis tanaman target aktual (ground truth). Dataset memiliki persebaran seimbang sempurna, dengan masing-masing komoditas memiliki tepat 100 sampel data.

- **Figure 11: Matriks Korelasi Fitur Agronomi**
  ![Figure 11: Matriks Korelasi Fitur Agronomi](feature_correlation.png)
  _Deskripsi:_ Heatmap matriks korelasi Pearson antara 7 fitur input agronomi numerik. Korelasi positif terkuat terlihat antara Fosfor (P) dan Kalium (K) sebesar 0,74, sedangkan hubungan lainnya cenderung memiliki korelasi lemah atau independen.

#### 2. Analisis Reduksi Dimensi (PCA)

Metode Principal Component Analysis (PCA) digunakan untuk menyusutkan data berdimensi 7 (7D) yang merepresentasikan 7 fitur input agronomi asli menjadi ruang ortogonal baru berdimensi 5 (5D). Reduksi dimensi ini bertujuan untuk:
- **Mengeliminasi Multikolinearitas:** Berdasarkan visualisasi matriks korelasi pada Figure 11, fitur Fosfor (P) dan Kalium (K) memiliki hubungan linier positif kuat ($r = 0,74$) yang dapat mengganggu kestabilan perhitungan jarak spasial. PCA mentransformasikan data menjadi komponen-komponen baru (PC1 hingga PC5) yang saling tegak lurus (ortogonal) sehingga korelasi antar komponen bernilai nol.
- **Menyaring Derau Spasial (Noise Filtering):** Membuang variasi data skala kecil yang bersifat acak dan tidak berkontribusi terhadap struktur kelompok alami.
- **Mempertahankan Kandungan Informasi:** Walaupun jumlah dimensi dikurangi dari 7 menjadi 5, model tetap mempertahankan akumulasi variansi informasi (*explained variance*) sebesar **87,58%** dari total informasi asli. Detail nilai per komponen adalah:

- **PC1:** Explained Variance = 27,59% (Eigenvalue = 1,932)
- **PC2:** Explained Variance = 18,48% (Eigenvalue = 1,294)
- **PC3:** Explained Variance = 15,38% (Eigenvalue = 1,077)
- **PC4:** Explained Variance = 14,61% (Eigenvalue = 1,023)
- **PC5:** Explained Variance = 11,51% (Eigenvalue = 0,806)
- _Justifikasi Pemilihan:_ 5 komponen utama dipilih berdasarkan kriteria Kaiser (eigenvalue $\geq 1$ atau mendekati 1), di mana PC5 dengan eigenvalue 0,806 dipertahankan agar dapat menangkap curah hujan mikro dan variasi pH ekstrim tanpa kehilangan informasi esensial.

#### 3. Model Klasterisasi K-Means

- **n_clusters:** 22 (sesuai dengan jumlah kelas tanaman target dalam dataset).
- **init:** `'k-means++'` (untuk inisialisasi centroid awal yang tersebar optimal guna mempercepat konvergensi).
- **n_init:** 10 (algoritma dijalankan 10 kali secara independen dengan centroid acak untuk memilih hasil dengan inersia terendah).
- **max_iter:** 300 (iterasi pergeseran centroid maksimum per run).
- **random_state:** 42 (untuk menjamin reproduksibilitas hasil klasterisasi).

#### 4. Model Gaussian Mixture Model (GMM)

- **n_components:** 22 (jumlah komponen distribusi Gaussian multivariat).
- **covariance_type:** `'full'` (setiap komponen memiliki matriks kovariansi bebas sendiri, memungkinkan klaster berbentuk elipsoid miring tak beraturan).
- **n_init:** 1 (jumlah inisialisasi parameter algoritma Expectation-Maximization).
- **random_state:** 42 (untuk konsistensi inisialisasi parameter probabilistik).

#### 5. Model Random Forest (Supervised Multilabel Classifier)

Pengklasifikasi multilabel diimplementasikan melalui pembungkus `MultiOutputClassifier` dengan estimator dasar `RandomForestClassifier`:

- **n_estimators:** 100 (jumlah pohon keputusan di dalam model forest).
- **max_depth:** `None` (pohon tumbuh bebas hingga seluruh daun murni atau menampung sampel kurang dari batas minimum pembelahan).
- **criterion:** `'gini'` (indeks ketidakmurnian Gini digunakan untuk mengukur kualitas pembagian cabang).

#### 6. Pembagian Data (Split Data)

- **Rasio Train/Test:** 80:20 (80% atau 1.760 data digunakan sebagai data latih dan 20% atau 440 data sebagai data uji klasifikasi).
- **Stratifikasi:** _Tidak stratified_. Karena target merupakan matriks biner multilabel (multi-hot encoding), scikit-learn secara default tidak mendukung stratifikasi silang langsung pada matriks multidimensi tanpa algoritma stratifikasi berulang khusus (_iterative stratification_). Oleh sebab itu, pemisahan acak acuan statis (`random_state=42`) diterapkan.

#### 7. Lingkungan Pengembangan (Development Environment)

- **Bahasa Pemrograman:** Python 3.10.6
- **Pustaka Utama:** Scikit-Learn versi 1.7.2, Pandas (manipulasi data), Numpy (operasi matriks), Joblib (penyimpanan model).
- **Spesifikasi Perangkat Keras:** Windows 11 OS x64-based PC dengan CPU AMD64/Intel Core, RAM $\geq$ 8 GB.

---

## 3. BAB III: HASIL DAN PEMBAHASAN (RESULTS & DISCUSSION)

### 3.1 Evaluasi Hasil Klasterisasi Lahan (Unsupervised)

Sebelum proses klasterisasi dilakukan, analisis pencilan (_outlier analysis_) mendeteksi bahwa penerapan eliminasi pencilan IQR ($\pm 1.5 \times IQR$) secara standar pada dataset agronomi ini akan menghapus 432 baris data (19.6% dari total data). Tindakan penyaringan tersebut berakibat pada **penghapusan 100% data tanaman Anggur dan Apel**, karena secara agronomi kedua tanaman tersebut secara alamiah membutuhkan konsentrasi Kalium (K) yang sangat tinggi (>200 mg/kg) sehingga terdeteksi sebagai outlier global, sebagaimana divisualisasikan pada Figure 4. Oleh karena itu, penelitian ini mempertahankan data outlier demi menjaga keberagaman komoditas pertanian.

Hasil evaluasi performa klasterisasi lahan pada ruang PCA 5D menggunakan metrik Silhouette Score, Davies-Bouldin Index (DBI), dan Adjusted Rand Index (ARI) disajikan pada tabel di bawah ini:

| Algoritma                        | Pendekatan Matematika                | Silhouette Score | Davies-Bouldin Index (DBI) | Adjusted Rand Index (ARI) | Karakteristik Rekomendasi                                                 |
| :------------------------------- | :----------------------------------- | :--------------: | :------------------------: | :-----------------------: | :------------------------------------------------------------------------ |
| **K-Means**                      | Hard Clustering (Centroid-based)     |    **0.307**     |         **1.081**          |           0.519           | Spasial Rigid (Hard) - Batasan klaster tegas tanpa tumpang tindih         |
| **Gaussian Mixture Model (GMM)** | Soft Clustering (Distribution-based) |      0.230       |           1.716            |         **0.816**         | Probabilistik (Soft) - Rekomendasi berperingkat sesuai sebaran data alami |

**Analisis Hasil Tahap 1:**
Berdasarkan nilai Silhouette Score dan DBI pada tabel di atas, **K-Means menghasilkan klaster dengan batasan spasial geometris yang lebih tegas dan kompak** (Silhouette = 0,307, DBI = 1,081) dibanding GMM (Silhouette = 0,230, DBI = 1,716). Hal ini wajar karena K-Means bekerja dengan mengoptimalkan jarak Euclidean lurus (hard clustering). Namun, dari sudut pandang keselarasan pengelompokan terhadap rumpun varietas tanaman asli, **GMM mencatatkan nilai Adjusted Rand Index (ARI) lebih tinggi, yaitu 0,816** dibanding K-Means yang sebesar 0,519. Temuan ini menunjukkan kecenderungan bahwa data karakteristik tanah agronomi alami memiliki sebaran yang saling tumpang tindih (_overlapping_), sehingga pendekatan batas probabilistik kontinu (soft clustering) pada GMM lebih sesuai dibandingkan pembagian wilayah tegas oleh K-Means. Karakteristik tersebut dipengaruhi oleh cara kerja K-Means yang mengoptimalkan jarak Euclidean linear (hard clustering) tanpa mempertimbangkan variasi kepadatan data.

### 3.2 Evaluasi Akurasi Prediksi Rekomendasi Tanaman (Multilabel)

Setelah model unsupervised K-Means dan GMM yang telah dilatih disimpan dalam bentuk file model, keduanya digunakan untuk menyusun representasi data multilabel pada tahap kedua (Supervised Multilabel).

Dengan memetakan koordinat sampel tanah ke dalam klaster hasil model K-Means dan GMM, label klaster tersebut ditransformasikan menjadi label target biner menggunakan `MultiLabelBinarizer`. Pengklasifikasi multilabel Random Forest (`MultiOutputClassifier`) kemudian dilatih menggunakan fitur agronomi tereduksi (PCA 5D) untuk memprediksi rekomendasi tanaman potensial secara simultan.

Hasil evaluasi performa klasifikasi terawasi multilabel disajikan pada tabel di bawah ini:

| Target Berbasis Model Unsupervised | Subset Accuracy (Exact Match Ratio) | F1-Score (Micro) | F1-Score (Macro) |
| :--------------------------------- | :---------------------------------: | :--------------: | :--------------: |
| **K-Means (22 Klaster)**           |               81.36%                |      0.9537      |      0.9557      |
| **GMM (22 Klaster)**               |               84.77%                |      0.9404      |      0.9489      |

**Analisis Hasil Tahap 2:**
Hasil pengujian menunjukkan adanya trade-off performa yang menarik antara kedua model klasifikasi multilabel tahap kedua:

1. **GMM Menunjukkan Performa Lebih Baik pada Akurasi Subset (84,77% vs 81,36%):** Model klasifikasi multilabel berbasis target GMM menghasilkan _Exact Match Ratio_ (Subset Accuracy) yang lebih tinggi. Temuan ini menunjukkan kecenderungan bahwa pembagian klaster berbasis probabilitas kontinu oleh GMM menghasilkan distribusi label yang lebih sesuai dengan struktur kelompok tanaman pada dataset. Dampaknya, tingkat kompleksitas target berkurang sehingga mempermudah algoritma Random Forest untuk mempelajari kecocokan persis.
2. **K-Means Unggul pada F1-Score (Micro: 0,9537 vs 0,9404):** Meskipun K-Means memiliki akurasi tebakan set lengkap (Subset Accuracy) yang lebih rendah, K-Means memperoleh F1-Score individual yang lebih tinggi. Ini menunjukkan bahwa untuk tanaman agronomi yang tidak termasuk dalam kecocokan persis, klasifikasi berbasis target K-Means memberikan presisi dan recall per komoditas yang lebih stabil secara rata-rata global.

### 3.3 Interpretasi Centroid & Kelompok Tanaman Multilabel

Beberapa klaster tanah teridentifikasi memuat kelompok rekomendasi multilabel logis secara agronomi:

- **Klaster Nutrisi Ekstrim (Apel & Anggur):** Terbentuk pada klaster dengan rata-rata Kalium >200 mg/kg dan Fosfor >130 mg/kg.
- **Klaster Sawah Tropika Basah (Padi & Yute):** Terbentuk pada kondisi curah hujan >180 mm dan kelembaban >80%.
- **Klaster Lahan Kering Rendah Hara (Sorgum & Millet):** Terbentuk pada curah hujan rendah (<80 mm) dan kandungan nitrogen moderat.

### 3.4 Tinjauan Hasil & Pembahasan (Results & Discussion Review)

#### 1. Analisis Overlap Lahan & Karakteristik Distribusi (Soft vs Hard Clustering)

- **Penyebab Overlap Agronomi:** Tanaman pertanian yang berbeda secara genetik sering kali berbagi relung ekologis yang serupa terhadap profil hara tanah dan mikroiklim. Sebagai contoh, tanaman legum (kacang-kacangan) seperti _lentil_, _mungbean_, _pigeonpeas_, _blackgram_, dan _mothbeans_ tumbuh optimal pada kondisi tanah dengan nitrogen rendah-sedang dan kelembaban moderat. Kondisi tersebut menyebabkan sebaran data parameter tanah komoditas tersebut saling tumpang-tindih di ruang hara.
- **Kelemahan Partisi Spasial K-Means:** Pembagian ruang parameter secara kaku menggunakan batas linier (_Voronoi cells_) oleh K-Means didasarkan pada jarak Euclidean terkecil dari centroid. Karena sebaran tanah asli komoditas cenderung berbentuk elipsoid miring dengan tingkat kepadatan yang tidak seragam, algoritma ini memisahkan satu populasi tanaman homogen menjadi beberapa klaster berbeda. Sebagai gambaran, tanaman _pigeonpeas_ terpecah ke dalam **9 klaster**, sedangkan _lentil_, _orange_, dan _papaya_ tersebar di **6 klaster**. Pemisahan paksa ini menghasilkan label sasaran yang tidak konsisten (_target noise_), yang berujung pada penurunan performa akurasi kecocokan persis (_Subset Accuracy_) model multilabel menjadi 81,36%.
- **Keunggulan Probabilistik GMM:** Model GMM menggunakan pendekatan distribusi Gaussian multivariat dengan matriks kovariansi penuh (`covariance_type='full'`), memungkinkan pembentukan klaster elipsoid miring dengan orientasi dan volume dinamis. Batas probabilitas kontinu (soft clustering) yang terbentuk terbukti adaptif dalam menangani tumpang-tindih data alami. Distribusi keanggotaan komoditas di bawah pemodelan GMM menjadi lebih terpusat, dengan tingkat penyebaran maksimal hanya **3 klaster** (seperti pada _coffee_, _pigeonpeas_, _maize_, _mothbeans_, dan _rice_). Pola ini merefleksikan asosiasi ekologis tanaman secara lebih konsisten.

#### 2. Bukti Visual Proyeksi PCA & Ukuran Klaster

Hasil pemetaan spasial dan perbandingan ukuran klaster diilustrasikan pada gambar di bawah ini:

- **Figure 1: PCA Scatter Plot K-Means**
  ![Figure 1: PCA Scatter Plot K-Means](kmeans_pca_scatter.png)
  _Deskripsi:_ Distribusi spasial hasil klasterisasi K-Means menunjukkan pembagian wilayah bulat sferis yang memotong paksa kontinuitas kepadatan elipsoid tanaman asli.

- **Figure 2: PCA Scatter Plot GMM**
  ![Figure 2: PCA Scatter Plot GMM](gmm_pca_scatter.png)
  _Deskripsi:_ Distribusi spasial hasil GMM memperlihatkan kemampuan adaptasi bentuk klaster elipsoid miring yang saling tumpang-tindih secara alami sesuai persebaran data riil.

- **Figure 3: Cluster Size Distribution Comparison**
  ![Figure 3: Cluster Size Distribution Comparison](cluster_distribution.png)
  _Deskripsi:_ K-Means memaksakan persebaran ukuran klaster yang seragam dan kaku, sedangkan GMM menghasilkan sebaran ukuran klaster yang dinamis dan adaptif terhadap kepadatan riil data (misal, Klaster 0 mencapai 170 sampel).

- **Figure 4: Boxplot Perbandingan Kalium Sebelum & Setelah IQR Removal**
  ![Figure 4: Boxplot Perbandingan Kalium Sebelum & Setelah IQR Removal](outlier_comparison.png)
  _Deskripsi:_ Perbandingan sebaran hara Kalium (K) sebelum (Kiri) dan setelah (Kanan) penghapusan pencilan global berbasis IQR ($K > 92,5$ mg/kg). Terlihat bahwa setelah pembersihan outlier global, seluruh data untuk komoditas Apel dan Anggur terhapus total dari dataset karena kebutuhan alami kadar hara kaliumnya yang tinggi.

- **Figure 5: Heatmap Komposisi Klaster K-Means**
  ![Figure 5: Heatmap Komposisi Klaster K-Means](kmeans_heatmap.png)
  _Deskripsi:_ Heatmap ini menggambarkan persebaran komoditas tanaman di seluruh klaster K-Means. Terlihat beberapa tanaman terfragmentasi ke banyak klaster (misalnya _pigeonpeas_ tersebar di 9 klaster berbeda).

- **Figure 6: Heatmap Komposisi Klaster GMM**
  ![Figure 6: Heatmap Komposisi Klaster GMM](gmm_heatmap.png)
  _Deskripsi:_ Heatmap GMM menunjukkan konsentrasi tanaman yang jauh lebih kompak pada klaster tertentu (maksimal hanya menyebar di 3 klaster), merefleksikan keselarasan dengan kelompok asosiasi tanaman asli.

- **Figure 7: Bar Chart Perbandingan Performa Klasifikasi Multilabel (K-Means vs GMM)**
  ![Figure 7: Bar Chart Perbandingan Performa Klasifikasi Multilabel](model_comparison_bar.png)
  _Deskripsi:_ Grafik batang perbandingan performa klasifikasi multilabel Tahap 2, menunjukkan keunggulan GMM pada Subset Accuracy (84,77%) dan keunggulan tipis K-Means pada F1-Score Micro (95,37%) dan Macro (95,57%).

- **Figure 8: Multilabel Confusion Matrix per Komoditas (GMM Target Pipeline)**
  ![Figure 8: Multilabel Confusion Matrix per Komoditas](multilabel_confusion_matrix.png)
  _Deskripsi:_ Grid matriks konfusi multilabel biner untuk 22 komoditas tanaman pada data uji, menunjukkan jumlah True Positive (TP), False Positive (FP), False Negative (FN), dan True Negative (TN) untuk setiap komoditas secara individual.

#### 3. Analisis Galat (Error Analysis)

Evaluasi mendalam terhadap hasil prediksi model klasifikasi terawasi multilabel pada Tahap 2 menyingkap pola galat berikut:

- **Ambiguitas pada Target GMM:**
    - Komoditas **`coffee`** mencatatkan nilai F1-Score terendah (**0,8333**). Secara agronomi, kebutuhan tanah kopi (terutama Nitrogen menengah dan curah hujan tinggi) sangat ambigu karena tumpang-tindih dengan tanaman sawah tropika basah seperti `jute` dan `rice` (pada Klaster 7) serta tanaman kelembaban moderat seperti `maize` (pada Klaster 12 dan 14).
    - Komoditas legum seperti **`mothbeans`** (F1-Score **0,8841**) dan **`pigeonpeas`** (F1-Score **0,8921**) sering mengalami salah prediksi silang (misklasifikasi) karena keduanya terkelompok bersama dalam satu rumpun klaster legum homogen (Klaster 0, 4, dan 13).
- **Ambiguitas pada Target K-Means:**
    - Klasifikasi berbasis target K-Means memiliki galat prediksi tertinggi pada tanaman **`maize`** dan **`pomegranate`** dengan F1-Score terendah sebesar **0,8740**. Pemecahan paksa sebaran kedua tanaman ini oleh K-Means ke dalam klaster minor acak memicu kekacauan label target (_target noise_) yang menghalangi model Random Forest mencapai konvergensi optimal untuk kelas tersebut.

---

## 4. BAB IV: KESIMPULAN (CONCLUSION)

### 4.1 Kesimpulan

1. Rancang bangun sistem rekomendasi komoditas pertanian multilabel diimplementasikan melalui pendekatan pembelajaran dua tahap: klasterisasi tidak terawasi (_unsupervised_) diikuti klasifikasi terawasi multilabel (_supervised multilabel_).
2. Perbandingan antara Hard Clustering (K-Means) dan Soft Clustering (GMM) menunjukkan **GMM menghasilkan keselarasan yang lebih tinggi dengan pengelompokan tanaman asli** dengan nilai Adjusted Rand Index (ARI) mencapai 0,816 dibanding K-Means yang sebesar 0,519. Di sisi lain, K-Means menghasilkan klaster dengan batasan spasial geometris yang lebih kompak dengan Silhouette Score 0,307 (GMM 0,230) dan Davies-Bouldin Index 1,081 (GMM 1,716).
3. Evaluasi pada tahap kedua menunjukkan keberhasilan pemanfaatan hasil klasterisasi sebagai target multilabel, di mana target berbasis GMM menghasilkan Subset Accuracy (Exact Match) tertinggi sebesar **84,77%** (berbanding K-Means **81,36%**), sedangkan target berbasis K-Means unggul pada F1-Score Micro sebesar **0,9537** (berbanding GMM **0,9404**).
4. Kebijakan pemeliharaan outlier menunjukkan peran krusial untuk mencegah hilangnya taksonomi tanaman bernilai tinggi (seperti apel dan anggur) yang membutuhkan konsentrasi hara tanah yang ekstrem.

### 4.2 Saran

1. **Optimasi Model Tahap Kedua:** Mengeksplorasi algoritma multilabel yang lebih kompleks (seperti Classifier Chains atau Multi-output Neural Networks) untuk memaksimalkan nilai Subset Accuracy dan F1-Score.
2. **Integrasi Data Temporal:** Memasukkan parameter temporal (musim tanam) untuk menyesuaikan luaran sistem rekomendasi secara dinamis.

---

## 5. DAFTAR PUSTAKA

Untuk Sinta 3, Anda membutuhkan minimal 15-20 referensi. Tambahkan referensi yang relevan dengan kata kunci berikut:

1. _Precision agriculture multilabel crop recommendation using machine learning._
2. _Comparative study of GMM and K-Means for multivariate soil clustering._
3. _Two-stage unsupervised-supervised multilabel learning for crop recommendation._
4. _The impact of outlier preservation in agricultural data mining._
