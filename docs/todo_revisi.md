# TO-DO LIST REVISI — Siap Submit SINTA 3

> Berdasarkan hasil audit di [`docs/audit.md`](./audit.md).
> Tandai `[x]` ketika selesai, `[/]` saat sedang dikerjakan.

---

## 🔴 CRITICAL — Harus Selesai Sebelum Apapun

- [x] **[C1] Ganti referensi fiktif di Tabel 1 (Section 1.3)** — SELESAI
    - `[11]` → Savla et al. (2022) — RF/DT/SVM crop recommendation (Lecture Notes in Networks and Systems)
    - `[12]` → Purba et al. (2022) — K-Means clustering karakter tanah (Jurnal Ilmiah Infotek)
    - `[13]` → Karthikeyan et al. (2023) — Decision Tree & Naive Bayes crop recommendation

- [x] **[C3] Daftar Pustaka lengkap (Section 5)** — SELESAI
    - 20 referensi terverifikasi dalam format IEEE
    - Mencakup: crop recommendation [1-8], RF [9], EM/GMM [10-11], K-Means [12], multilabel [13-15], metrik evaluasi [16-18], PCA [19], scikit-learn [20]

- [x] **[C4] Tambahkan justifikasi untuk potensi circular evaluation** — SELESAI
    - Justifikasi teoretis: pipeline bertujuan memprediksi keanggotaan klaster, bukan label agronomi eksternal
    - Kuantitatif: Menambahkan Tabel 5 (Purity GMM = 87.32%, ARI = 0.816, 16/22 klaster dominasi tunggal >= 80%, dengan 10 klaster memiliki purity 100% seperti Apel dan Anggur) yang membuktikan klaster secara alami merepresentasikan kelompok ekologis tanaman.

---

## 🟠 MAJOR — Berdampak Signifikan pada Skor Reviewer

- [x] **[M1] 5 persamaan matematika kunci (Section 2.2.8)** — SELESAI
    - Persamaan 1: K-Means WCSS $J = \sum_k \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$
    - Persamaan 2: GMM Log-Likelihood EM
    - Persamaan 3: Silhouette Score $s(i)$
    - Persamaan 4: Subset Accuracy (Exact Match Ratio)
    - Persamaan 5: F1-Score Multilabel Micro

- [x] **[M2] Eksperimen 3 seed berbeda (Section 3.2)** — SELESAI
    - Hasil: KM 0.8288 ± 0.0183 | GMM 0.8303 ± 0.0149 (Subset Accuracy)
    - Tabel 3 (per-seed) dan Tabel 4 (agregat mean±std) sudah masuk draft jurnal
    - File hasil tersimpan: `multiseed_results.csv`

- [x] **[M3] Tambahkan tabel nilai centroid per klaster representatif (Section 3.3)** — SELESAI
    - Membuat dan menjalankan `scripts/extract_centroids.py`
    - Menambahkan Tabel 6 yang memuat rerata N, P, K, temp, hum, ph, rain untuk klaster 2 (Apel), 17 (Anggur), 20 (Padi), 8 (Kacang Arab), dan 9 (Melon).
    - Menganalisis keselarasan klaster terhadap kondisi agronomi riil secara kuantitatif.

- [x] **[M4] Tambahkan justifikasi GMM `n_init=1` (Section 2.2.4)** — SELESAI
    - Justifikasi: GMM menggunakan centroid K-Means (`init_params='kmeans'`) yang telah diinisialisasi 10 kali (`n_init=10`). EM dimulai dari titik spasial optimal, bukan random acak, serta dijamin stabil oleh `random_state=42`.

- [x] **[M5] Tambahkan justifikasi RF `max_depth=None` (Section 2.2.5)** — SELESAI
    - Justifikasi: Menjamin bias latih minimum (Train Subset Accuracy 100,00%). Selisih 15,68% dari Test Subset Accuracy (~84,32%) tidak tergolong overfitting destruktif karena mekanisme bootstrapping (_bagging_) dan random feature selection pada RF bertindak sebagai regularisasi alami.

- [x] **[M6] Tambahkan versi lengkap library (Section 2.2.7)** — SELESAI
    - Menambahkan spesifikasi lengkap versi pustaka utama: Numpy 2.2.6, Pandas 2.3.3, Scikit-Learn 1.7.2.

---

## 🟡 MINOR — Mudah Diperbaiki, Tapi Diperhatikan Reviewer

- [x] **[N1] Revisi judul paper** — SELESAI (batasan max 12 kata)
    - Judul baru: _"K-Means vs GMM Clustering for Multilabel Crop Recommendation with Outlier Preservation"_ (11 kata)

- [x] **[N2] Ubah pola novelty dari "Pertama/Kedua/Ketiga" ke paragraf natural** — SELESAI (dikerjakan bersama C2)
    - Novelty paragraph di Section 1.3 sudah direvisi ke narasi akademik yang lebih natural

- [x] **[N3] Hapus kalimat duplikat di akhir Section 3.1** — SELESAI
    - Kalimat duplikat yang mengulang cara kerja K-Means sudah dihapus secara bersih.

- [x] **[N4] Standarisasi terminologi bilingual** — SELESAI
    - Mengganti istilah "outlier" dengan "pencilan", "clustering" dengan "klasterisasi", serta seluruh label/referensi "Figure X" menjadi "Gambar X" secara konsisten dalam seluruh teks Bahasa Indonesia.

- [x] **[N5] Revisi frasa klise AI-like (Section 1.2)** — SELESAI
    - Merevisi frasa klise tentang ketahanan pangan menjadi kalimat yang lebih objektif dan akademik: "diperlukan untuk meminimalisasi risiko kegagalan budidaya melalui pencocokan parameter biofisik tanah secara objektif".

- [x] **[N6] Perluas Section 4.2 (Saran) dari 2 menjadi 4 poin** — SELESAI
    - Menambahkan batasan penelitian geografis/jenis tanah serta ketiadaan sensor IoT real-time sebagai 2 poin saran tambahan untuk arah riset mendatang.

- [x] **[N7] Tambahkan tabel statistik deskriptif dataset (Section 2.2.1)** — SELESAI
    - Menambahkan Tabel 2 yang merinci nilai mean, std, min, median, dan max untuk 7 parameter agronomi guna memperkuat justifikasi prapemrosesan StandardScaler.

---

## 📋 STATUS RINGKASAN

| Grup        | Total Item | Selesai | Sisa  |
| :---------- | :--------: | :-----: | :---: |
| 🔴 Critical |     4      |    4    |   0   |
| 🟠 Major    |     6      |    6    |   0   |
| 🟡 Minor    |     7      |    7    |   0   |
| **Total**   |   **17**   | **17**  | **0** |

---

## 🎯 TARGET SKOR PASCA-REVISI

| Kriteria             | Sebelum | Target |
| :------------------- | :-----: | :----: |
| Scientific Quality   |   62    |   78   |
| Novelty              |   52    |   65   |
| Methodology          |   58    |   80   |
| Experimental Design  |   50    |   68   |
| Results & Discussion |   65    |   80   |
| Academic Writing     |   68    |   82   |
| Reproducibility      |   63    |   82   |
| **Overall**          | **60**  | **76** |

> 76/100 → Masuk rentang **"Siap Submit dengan Revisi Minor"** untuk SINTA 3.
