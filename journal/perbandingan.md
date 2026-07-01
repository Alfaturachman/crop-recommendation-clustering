# Tabel Perbandingan Jurnal Terkait: Crop Recommendation dengan Machine Learning

> Disusun sebagai tinjauan literatur pendukung jurnal:  
> **"Comparative Hybrid Clustering-Classification Pipeline Using K-Means and GMM for Multi-Label Crop Recommendation"**

---

## Ringkasan Jurnal yang Dianalisis

| #   | Judul Jurnal                                                                                                                         | Penulis         | Tahun | Sumber/Publisher                                              |
| --- | ------------------------------------------------------------------------------------------------------------------------------------ | --------------- | ----- | ------------------------------------------------------------- |
| J1  | _Utilizing Supervised and Unsupervised Machine Learning Techniques for Crop Yield Prediction, Pest Detection, And Precision Farming_ | Srinivas et al. | 2026  | Journal of International Commerce Law & Technology, vol. 7(1) |
| J2  | _Applying Tree Based Model for Crop Recommendation System Based on Soil Parameters and Weather Conditions_                           | Abdullah et al. | 2025  | JITK – Jurnal Ilmu Pengetahuan dan Teknologi Komputer         |
| J3  | _An Effective Crop Recommendation Method Using Machine Learning Techniques_                                                          | Garg & Alam     | 2023  | (Peer-reviewed open access journal, Jamia Millia Islamia)     |
| J4  | _Comparative Analysis of Machine Learning Approaches for Crop Recommendation in Sustainable Agriculture in India_                    | Yadav et al.    | 2026  | SERB-SURE (M.J.P. Rohilkhand University)                      |

---

## Tabel Perbandingan Komprehensif

| Aspek Perbandingan             | **J1: Srinivas et al. (2026)**                                                                                                              | **J2: Abdullah et al. (2025)**                                                                          | **J3: Garg & Alam (2023)**                                                                                       | **J4: Yadav et al. (2026)**                                                         | **Penelitian Ini (Proposed)**                                                                                              |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **Dataset**                    | Kaggle Crop Recommendation, 2.200 sampel, 8 fitur (N, P, K, Suhu, Kelembaban, pH, Curah Hujan, Label)                                       | Kaggle Crop Recommendation, 2.200 sampel; setelah IQR removal menjadi **1.846** sampel                  | Kaggle Crop Recommendation, 2.200 sampel, 7 fitur (tanpa outlier removal)                                        | Kaggle Crop Recommendation, 2.200 sampel, 8 kolom                                   | Kaggle Crop Recommendation, **2.200 sampel** (dipertahankan penuh dengan outlier preservation)                             |
| **Jumlah Kelas/Label**         | 22 kelas tanaman (single-label)                                                                                                             | 22 kelas tanaman (single-label)                                                                         | 22 kelas tanaman (single-label)                                                                                  | 22 kelas tanaman (single-label)                                                     | 22 kelas tanaman → **multilabel** (12 klaster sebagai target biner)                                                        |
| **Tipe Problem**               | Klasifikasi supervised multi-kelas tunggal + clustering eksplorasi                                                                          | Klasifikasi supervised multi-kelas tunggal                                                              | Klasifikasi supervised multi-kelas tunggal                                                                       | Klasifikasi supervised multi-kelas tunggal                                          | **Hybrid: Unsupervised clustering → Supervised multilabel classification**                                                 |
| **Metode Supervised**          | Logistic Regression, Random Forest                                                                                                          | Decision Tree, Random Forest, XGBoost                                                                   | IBk (KNN), MLP, CDT (C4.5), REP Tree, **PART** (Partial C4.5)                                                    | Decision Tree, Random Forest, Gradient Boosting, Extra Trees, AdaBoost, **XGBoost** | **MultiOutputClassifier (Random Forest)** berbasis target GMM dan K-Means                                                  |
| **Metode Unsupervised**        | K-Means (K=4, Silhouette=0,3229), Agglomerative (Silhouette=0,3468)                                                                         | _(Tidak digunakan)_                                                                                     | _(Tidak digunakan — hanya filter feature selection PCA)_                                                         | _(Tidak digunakan)_                                                                 | **K-Means** (K=12, Silhouette=0,324) vs **GMM full covariance** (K=12, Silhouette=0,286)                                   |
| **Reduksi Dimensi**            | PCA 2-komponen (untuk visualisasi)                                                                                                          | Tidak ada                                                                                               | PCA (filter feature selection)                                                                                   | Tidak ada                                                                           | **PCA 5-komponen** (87,58% variansi kumulatif, sebagai fitur input)                                                        |
| **Penanganan Outlier**         | StandardScaler; tidak ada outlier removal eksplisit                                                                                         | **IQR removal global** (hapus 16,09% data; menjadi 1.846 sampel) — outlier dibuang                      | Tidak ada outlier removal                                                                                        | StandardScaler; tidak ada outlier removal                                           | **Outlier preservation** — pencilan dipertahankan karena K>92,5 mg/kg penting untuk Apple & Grapes                         |
| **Pembagian Data**             | 80:20 (train:test)                                                                                                                          | 70:30 (train:test)                                                                                      | K-fold cross-validation (k=10)                                                                                   | 80:20 (train:test)                                                                  | 80:20 (train:test), multi-seed validation (seed: 0, 42, 99)                                                                |
| **Hyperparameter Tuning**      | Tidak disebutkan secara eksplisit                                                                                                           | Tidak ada tuning formal                                                                                 | Grid Search (GS) hyperparameter optimization                                                                     | Parameter Grid Optimization + cross-validation                                      | n_estimators=100, max_depth=None (Random Forest), n_init=10 (K-Means), BIC-based K selection (GMM)                         |
| **Metrik Evaluasi**            | Accuracy, Precision, Recall, F1-Score (supervised) + Silhouette Score (unsupervised)                                                        | Accuracy, Precision, Recall, F1-Score + Cross-validation (StratifiedKFold, k=5)                         | TP, FP, Precision, Recall, F1-Score, RMSE, MAE, Kappa Statistic                                                  | Accuracy, Precision, Recall, F1-Score                                               | **Subset Accuracy (Exact Match)**, F1-Micro, F1-Macro, Silhouette Score, DBI, ARI                                          |
| **Hasil Terbaik (Supervised)** | Random Forest: **Accuracy 99,55%**                                                                                                          | Random Forest: **Accuracy 99,81%** (F1=0,998)                                                           | PART + Wrapper-Grid: **Accuracy 99,31%** (Kappa=0,9929)                                                          | XGBoost: **Accuracy 99,60%** (F1=99,61%)                                            | GMM-target pipeline: **avg Subset Accuracy 89,70%**, F1-Micro 0,9662, F1-Macro 0,9657                                      |
| **Hasil Unsupervised**         | K-Means Silhouette=0,3229 (K=4); Agglomerative Silhouette=0,3468                                                                            | _(N/A)_                                                                                                 | _(N/A)_                                                                                                          | _(N/A)_                                                                             | K-Means Silhouette=0,324, DBI=1,084, ARI=0,413; GMM Silhouette=0,286, DBI=1,246, **ARI=0,524**                             |
| **Label Target**               | Single-label (1 dari 22 crop)                                                                                                               | Single-label (1 dari 22 crop)                                                                           | Single-label (1 dari 22 crop)                                                                                    | Single-label (1 dari 22 crop)                                                       | **Multilabel biner** (multi-hot vector L=22, dibangkitkan dari klaster)                                                    |
| **Kontribusi Utama**           | Benchmark supervised+unsupervised dalam satu studi dengan dataset Kaggle; menunjukkan K-Means berguna untuk segmentasi lingkungan pertanian | Benchmark tiga tree-based model (DT, RF, XGB) dengan outlier removal IQR; RF terbaik                    | Wrapper feature selection + PART classifier dengan Grid Search hyperparameter tuning; best accuracy 99,31%       | Benchmark enam model ML dengan hyperparameter tuning; XGBoost terbaik 99,60%        | **Pipeline dua-tahap**: unsupervised clustering → multilabel supervised; perbandingan K-Means vs GMM; outlier preservation |
| **Keterbatasan/Gap**           | K=4 klaster tanpa justifikasi optimal; tidak ada evaluasi dampak klasterisasi ke downstream supervised; single-label output                 | IQR removal global menghilangkan 16% data termasuk Apple & Grapes; tidak ada klasterisasi; single-label | Single-label output; tidak menggunakan klasterisasi sebagai label generator; tidak ada analisis outlier sensitif | Single-label output; tidak ada klasterisasi; tidak ada outlier analysis             | Jumlah seed validasi terbatas (n=3); dataset statis satu wilayah; tidak ada validasi lapangan                              |

---

## Analisis Perbandingan Mendalam

### 1. Perbedaan Penanganan Outlier

| Studi                    | Kebijakan Outlier                                | Dampak pada Data                                                            |
| ------------------------ | ------------------------------------------------ | --------------------------------------------------------------------------- |
| Srinivas et al. (J1)     | Tidak ada outlier removal                        | 2.200 sampel utuh                                                           |
| **Abdullah et al. (J2)** | **IQR global removal (Q1−1,5×IQR / Q3+1,5×IQR)** | **16,09% data hilang → 1.846 sampel; Apple & Grapes kemungkinan terdampak** |
| Garg & Alam (J3)         | Tidak ada outlier removal                        | 2.200 sampel utuh                                                           |
| Yadav et al. (J4)        | Tidak ada outlier removal eksplisit              | 2.200 sampel utuh                                                           |
| **Penelitian Ini**       | **Outlier preservation (dipertahankan)**         | **2.200 sampel utuh; Apple & Grapes terlindungi**                           |

> **Catatan Penting:** Jurnal J2 (Abdullah et al.) menggunakan IQR removal global yang secara empiris akan menghilangkan data Apple dan Grapes karena kebutuhan K>200 mg/kg melampaui threshold IQR global. Ini merupakan penelitian yang memverifikasi keterbatasan tersebut dan menjadi dasar justifikasi _outlier preservation_ pada penelitian ini.

---

### 2. Perbandingan Pendekatan Problem

```
J1, J2, J3, J4:
Input (N,P,K,...) → [Supervised Classifier] → Output: SATU crop label (single-label)

Penelitian Ini:
Input (N,P,K,...) → [PCA 5D] → [K-Means / GMM] → [Label Generator] → [Multilabel RF] → Output: BEBERAPA crop yang cocok (multilabel)
```

Seluruh jurnal pembanding menggunakan pendekatan **single-label** (memilih satu crop terbaik). Tidak ada yang menghasilkan rekomendasi **multilabel** (beberapa crop cocok sekaligus) — inilah kebaruan utama penelitian ini.

---

### 3. Perbandingan Metrik dan Interpretabilitas

| Jurnal             | Metrik Utama                                      | Nilai Terbaik                               | Catatan                                                                                        |
| ------------------ | ------------------------------------------------- | ------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| J1 (Srinivas)      | Accuracy                                          | RF: 99,55%                                  | Single-label, tidak ada uji generalisasi multi-seed                                            |
| J2 (Abdullah)      | Accuracy + Cross-Val                              | RF: 99,81%, CV: 99,34%                      | Setelah IQR removal; data berkurang 16%                                                        |
| J3 (Garg & Alam)   | F1-Score + Kappa                                  | PART: 99,31%, κ=0,9929                      | Best feature selection + hyperparameter tuning                                                 |
| J4 (Yadav)         | Accuracy, F1                                      | XGBoost: 99,60%                             | Enam model dibandingkan; ensemble terbaik                                                      |
| **Penelitian Ini** | **Subset Accuracy (Multilabel) + F1-Micro/Macro** | **GMM: 89,70% Subset Acc, F1-Micro 0,9662** | **Multi-seed (n=3); BERBEDA dimensi — multilabel jauh lebih ketat dari single-label accuracy** |

> **Penting:** Angka Subset Accuracy 89,70% pada penelitian ini **tidak bisa dibandingkan langsung** dengan Accuracy 99% jurnal lain, karena:
>
> - Single-label accuracy: benar jika **1 dari 22** label tepat
> - Subset Accuracy (Exact Match): benar hanya jika **SELURUH vektor biner** 22-dimensi identik — jauh lebih ketat

---

### 4. Posisi Penelitian Ini dalam Lanskap Literatur

```
Spektrum Kompleksitas Pendekatan:
─────────────────────────────────────────────────────────────────────────────────────
SIMPLE                                                                        COMPLEX
  │                                                                               │
  │   J1,J2,J3,J4                        Penelitian Ini                          │
  │   (Single-label supervised)          (2-Stage Hybrid Pipeline)               │
  │   • 1 model tahap                    • Unsupervised + Supervised             │
  │   • 1 output label                   • Multilabel output                     │
  │   • Accuracy ~99%                    • Subset Acc ~89,70% (lebih ketat)      │
  │   • Tidak butuh klasterisasi         • K-Means vs GMM comparison             │
  │   • Single paradigma                 • Outlier preservation                  │
─────────────────────────────────────────────────────────────────────────────────────
```

---

## Referensi Jurnal

1. **[J1]** I. V. Srinivas, M. Dhotay, J. Sridevi, R. Sanwal, and N. Jain, "Utilizing Supervised and Unsupervised Machine Learning Techniques for Crop Yield Prediction, Pest Detection, And Precision Farming," _Journal of International Commerce Law and Technology_, vol. 7, no. 1, pp. 660–669, 2026.

2. **[J2]** A. Abdullah, M. Iwan, and S. R. Dani, "Applying Tree Based Model for Crop Recommendation System Based on Soil Parameters and Weather Conditions," _JITK (Jurnal Ilmu Pengetahuan dan Teknologi Komputer)_, 2025.

3. **[J3]** D. Garg and M. Alam, "An effective crop recommendation method using machine learning techniques," _[International Open Access Journal]_, Received Nov. 2022, Accepted May 2023.

4. **[J4]** S. Yadav, H. K. Singh, A. Garg, and M. M. Khan, "Comparative Analysis of Machine Learning Approaches for Crop Recommendation in Sustainable Agriculture in India," M.J.P. Rohilkhand University, doi: SERB-SURE SUR/2022/000940, 2026.
