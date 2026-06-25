# LAPORAN AUDIT REVIEWER JURNAL ILMIAH
### Target: SINTA 3 | Bidang: Computer Science / AI
**Paper:** *"Comparative Study of Soft and Hard Clustering for Multilabel Crop Recommendation"*

---

> [!NOTE]
> Audit ini dilakukan layaknya reviewer sungguhan. Sifatnya objektif dan kritis. Setiap kelemahan yang dicatat bukan berarti paper tidak bisa diterima — ini adalah peta kerja untuk revisi yang dibutuhkan sebelum submit.

---

## A. SCIENTIFIC QUALITY

### Relevansi Topik
**Cukup Relevan.** Topik rekomendasi tanaman berbasis clustering + multilabel classification memiliki nilai praktis tinggi dan sejalan dengan tren *precision agriculture*. Namun, tidak ada referensi ke literatur terbaru (2023–2025) yang membuktikan bahwa riset ini *aware* terhadap perkembangan terkini. Penulisan referensi tabel gap masih sangat hipotetis (Author fiktif: "Prasad & Kumar [13] (2026)" — tahun 2026 bukan tahun valid saat ini).

### Research Gap
**Cukup terdefinisi, namun masih semu.** Tiga gap yang diklaim (single-label constraint, absensi prediksi setelah clustering, outlier deletion) memang nyata. Namun, klaim "penelitian terdahulu mengabaikan outlier preservation" tidak dibuktikan secara empiris dengan sitasi metode spesifik dari paper asli. Reviewer akan menanyakan: *"Apakah benar-benar tidak ada paper yang membahas outlier preservation dalam konteks ini?"*

### Tujuan Penelitian
**Tidak diformulasikan secara eksplisit.** Tidak ada kalimat *"Penelitian ini bertujuan untuk..."* yang berdiri sendiri secara formal. Tujuan tersebar implisit di dalam narasi novelty. Ini **wajib diperbaiki** untuk standar jurnal manapun.

### Kontribusi
Tiga kontribusi diklaim pada paragraf terakhir Section 1.3:
1. Formulasi alur hibrida PCA + unsupervised → supervised multilabel. *(Ini lebih merupakan pipeline design, bukan novelty metodologis murni)*
2. Komparasi K-Means vs GMM. *(Incremental — komparasi jenis ini sangat umum)*
3. Pembuktian pentingnya outlier preservation. *(Ini yang paling unik dan berpotensi jadi kontribusi utama)*

**Masalah kritis:** Tidak ada rumusan masalah (*research question*) yang eksplisit. Reviewer tidak dapat menilai apakah eksperimen menjawab pertanyaan yang diajukan jika pertanyaannya tidak pernah dinyatakan.

### Novelty Assessment
| Dimensi Novelty | Status |
|:---|:---|
| Method novelty | Sangat lemah — K-Means, GMM, dan RF sudah sangat mainstream |
| Dataset novelty | Tidak ada — dataset publik Kaggle yang umum digunakan |
| Experimental novelty | Sedang — kombinasi unsupervised→multilabel target generation cukup jarang |
| Application novelty | Sedang — penerapan di domain agronomi multilabel masih relatif terbatas |

**Kesimpulan:** Novelty bersifat **incremental improvement** dengan elemen *experimental novelty* yang cukup menarik, khususnya pada aspek *outlier preservation justification* dan *two-stage pipeline*.

---

## B. INTRODUCTION REVIEW

### Kekuatan
- Latar belakang Section 1.2 memiliki alur logis yang baik: masalah produktivitas → single-label limitation → kebutuhan multilabel.
- Tabel 1 (gap analysis) adalah pendekatan yang baik untuk SINTA 3.

### Kelemahan Spesifik

**1. Tidak ada rumusan masalah eksplisit.**
Setelah paragraf latar belakang, tidak ada kalimat seperti: *"Berdasarkan gap tersebut, penelitian ini merumuskan masalah sebagai berikut: (1)... (2)..."*. Ini adalah kelemahan struktural serius.

**2. Referensi [11], [12], [13] di Tabel 1 terlihat fiktif.**
- "Ingle et al. [11] (2024)" — Atharva Ingle adalah *dataset creator*, bukan penulis paper klasterisasi.
- "Prasad & Kumar [13] (2026)" — Tahun 2026 belum valid.
Reviewer yang berpengalaman akan langsung mendeteksi ini dan mempertanyakan validitas gap analysis.

**3. Paragraf terakhir Section 1.3 terlalu padat dan generik.**
> *"Kebaruan (novelty) utama yang diajukan mencakup tiga poin penting. Pertama... Kedua... Ketiga..."*

Pola enumerasi novelty seperti ini terlalu menyerupai template AI. Setiap poin perlu disampaikan dengan konteks spesifik *mengapa hal tersebut baru*, bukan hanya *apa yang dilakukan*.

**4. Tidak ada hypothesis atau research question formal.**

---

## C. METHODOLOGY REVIEW

### Yang Sudah Cukup
- Hyperparameter tercatat dengan baik (K-Means: `n_init=10`, `init='k-means++'`; GMM: `covariance_type='full'`).
- Justifikasi PCA 5 komponen dengan nilai eigenvalue per PC sudah cukup baik.
- Outlier preservation dijelaskan dengan bukti kuantitatif (432 baris = 19,6%).

### Informasi yang Masih Kurang (Wajib Ditambahkan)

| Item | Status | Keterangan |
|:---|:---|:---|
| Rumus matematika K-Means objective function | ❌ Tidak ada | Wajib untuk reproducibility |
| Rumus GMM likelihood / EM formulation | ❌ Tidak ada | Minimal 1-2 persamaan kunci |
| Preprocessing: range nilai sebelum/sesudah StandardScaler | ❌ Tidak ada | Minimal tabel statistik deskriptif |
| Cross-validation scheme | ❌ Tidak ada | Single 80:20 split sangat lemah |
| GMM `n_init=1` — justifikasi | ⚠️ Lemah | n_init=1 berisiko konvergensi ke local optimum |
| Random Forest max_depth=None — justifikasi | ⚠️ Lemah | Bisa menyebabkan overfitting |
| Waktu training (training time) | ❌ Tidak ada | Relevan untuk efisiensi komputasi |
| Versi library yang digunakan secara lengkap | ⚠️ Parsial | Numpy dan Pandas versinya tidak disebutkan |

**Masalah paling kritis:** Tidak ada rumus matematika sama sekali dalam keseluruhan draft. Untuk SINTA 3, setidaknya persamaan Silhouette Score, K-Means objective, dan GMM likelihood harus dicantumkan.

---

## D. EXPERIMENTAL DESIGN REVIEW

### Kelemahan Utama

**1. Single train/test split tanpa cross-validation.**
Pembagian 80:20 tanpa stratifikasi dan tanpa k-fold cross-validation menghasilkan evaluasi yang sangat bergantung pada satu state random (`random_state=42`). Ini adalah **kelemahan metodologis mayor** untuk standar SINTA 3. Satu eksperimen dengan satu seed tidak cukup untuk mengklaim generalisasi.

**2. Tidak ada baseline komparatif.**
Paper tidak membandingkan pipeline yang diusulkan dengan metode lain yang dikenal (misalnya: direct supervised multilabel tanpa clustering, DBSCAN sebagai alternatif clustering, atau Label Powerset). Tanpa baseline, klaim superioritas GMM hanya relatif terhadap K-Means dalam konteks yang sama — bukan terhadap state-of-the-art.

**3. Circular evaluation risk.**
Label target multilabel dibangkitkan dari hasil klasterisasi (unsupervised), kemudian classifier dilatih untuk memprediksi label tersebut, dan evaluasinya menggunakan label yang sama. Ini menciptakan potensi *circular evaluation* — model dievaluasi seberapa baik ia belajar dari label yang sudah bias oleh clustering. Reviewer akan menanyakan: *"Apakah Subset Accuracy 84,77% bermakna secara agronomi, atau hanya mengukur seberapa baik model belajar dari noise clustering?"*

**4. Tidak ada ablation study.**
Tidak ada eksperimen untuk menguji: "Apa dampak jika PCA tidak digunakan?", "Bagaimana jika K ≠ 22?", "Apakah n_components PCA yang berbeda mengubah hasil secara signifikan?"

---

## E. RESULTS & DISCUSSION REVIEW

### Yang Sudah Baik
- Tabel perbandingan Tahap 1 dan Tahap 2 ringkas dan informatif.
- Analisis error per komoditas (coffee F1=0,8333, mothbeans F1=0,8841) menunjukkan kedalaman analisis yang baik.
- Narasi trade-off GMM vs K-Means (Subset Accuracy vs F1-Score) dijelaskan dengan baik.
- Interpretasi centroid (Section 3.3) memberikan nilai agronomi yang menarik.

### Yang Perlu Diperdalam

**1. Tidak ada interpretasi mengapa GMM lebih tinggi dari perspektif matematika distribusi.**
Kalimat:
> *"Temuan ini menunjukkan kecenderungan bahwa data karakteristik tanah agronomi alami memiliki sebaran yang saling tumpang tindih..."*

Ini hanya *deskripsi ulang* dari yang sudah diketahui. Yang kurang adalah: *"Berdasarkan matriks kovariansi GMM, komponen ke-X memiliki orientasi elipsoid yang merefleksikan korelasi tinggi antara fitur Y dan Z, yang secara agronomi konsisten dengan..."*

**2. Kalimat terakhir Section 3.1 duplikasi dengan paragraf sebelumnya.**
> *"Karakteristik tersebut dipengaruhi oleh cara kerja K-Means yang mengoptimalkan jarak Euclidean linear (hard clustering) tanpa mempertimbangkan variasi kepadatan data."*

Ini adalah pengulangan dari kalimat sebelumnya dalam paragraf yang sama. Harus dihapus atau digabung.

**3. Section 3.3 terlalu singkat dan tidak didukung data kuantitatif.**
Interpretasi centroid (Klaster Nutrisi Ekstrim, Sawah Tropika, dll.) sangat menarik secara agronomi tetapi tidak ada satu pun tabel nilai centroid rata-rata per klaster yang mendukung klaim tersebut.

**4. Metrik Precision dan Recall per kelas tidak dilaporkan** — hanya F1-Score agregat yang disebutkan dalam error analysis.

---

## F. NOVELTY STRENGTH ASSESSMENT

> **Penilaian: SEDANG (Cenderung Lemah)**

**Alasan:**
- K-Means, GMM, PCA, dan Random Forest adalah metode yang sudah sangat mapan dalam literatur.
- Dataset Kaggle yang digunakan bersifat publik, kecil (2.200 baris), dan sudah banyak digunakan dalam paper sebelumnya.
- Kombinasi unsupervised → multilabel label generation adalah yang paling menarik, namun konsep ini sudah dieksplorasi dalam domain lain (misalnya: text clustering → label generation).

**Saran Konkret untuk Meningkatkan Novelty:**
1. **Tambahkan theoretical justification** mengapa GMM lebih cocok secara matematis untuk data agronomi yang memiliki distribusi elipsoid — ini akan mengangkat novelty dari *empirical comparison* ke *theoretically motivated comparison*.
2. **Jabarkan outlier preservation** bukan hanya sebagai kebijakan, tetapi sebagai metode formal dengan kriteria seleksi berbasis domain knowledge (contoh: konsentrasi K ≥ 150 mg/kg sebagai batas fisiologis tanaman buah).
3. **Tambahkan analisis sensitivitas** jumlah klaster (K=15, 18, 22, 25) untuk membuktikan bahwa K=22 bukan kebetulan dan pilihan tersebut optimal.

---

## G. ACADEMIC WRITING REVIEW

### Masalah Umum
Secara keseluruhan, tulisan memiliki struktur yang baik dan readable. Namun ada beberapa pola yang perlu diperbaiki:

**1. Penggunaan "Hal ini" yang berlebihan (pola AI-like)**
Meskipun sudah lebih baik dari draft awal, masih ada pola kalimat yang terlalu formulaic. Contoh:
> *"Hal ini wajar karena K-Means bekerja dengan mengoptimalkan jarak Euclidean lurus."*

Revisi: *"Kompaknya klaster K-Means dapat dijelaskan secara geometris: algoritma ini meminimalkan inertia (within-cluster sum of squares) yang secara inheren menghasilkan batas Voronoi sferis."*

**2. Tidak konsisten antara istilah Bahasa Indonesia dan Inggris**
- "Hard Clustering" vs "klasterisasi keras" — pilih salah satu secara konsisten.
- "outlier" vs "pencilan" — bergantian tanpa pola.
- "Figure" vs "Gambar" — harus konsisten (pilih Gambar jika bahasa Indonesia, Figure jika bahasa Inggris/format IEEE).

**3. Klaim tanpa bukti kuantitatif di Section 3.3**
> *"Terbentuk pada klaster dengan rata-rata Kalium >200 mg/kg"*

Ini harus didukung tabel nilai centroid, bukan hanya klaim naratif.

**4. Judul paper tidak mencerminkan kontribusi utama**
Judul saat ini: *"Comparative Study of Soft and Hard Clustering for Multilabel Crop Recommendation"*
Judul ini generik. Tidak menyebut PCA, tidak menyebut outlier preservation, tidak menyebut two-stage pipeline. Pertimbangkan:
*"A Two-Stage PCA-Clustering Pipeline for Multilabel Crop Recommendation: Comparing K-Means and GMM with Outlier Preservation"*

---

## H. AI-GENERATED WRITING DETECTION REVIEW

Berikut kalimat-kalimat yang berindikasi kuat sebagai AI-generated berdasarkan pola linguistik:

### Temuan 1 — Pola "Pertama... Kedua... Ketiga..."
> *"Kebaruan (novelty) utama yang diajukan mencakup tiga poin penting. Pertama, formulasi alur hibrida... Kedua, komparasi teoretis dan praktis... Ketiga, pembuktian kuantitatif..."*

**Indikator:** Enumerasi novelty dengan format "Pertama/Kedua/Ketiga" adalah pola ChatGPT yang sangat umum. Penulis manusia umumnya memisahkan setiap kontribusi ke dalam paragraf terpisah dengan konteks yang lebih kaya.

**Revisi yang lebih natural:**
> *"Kontribusi utama penelitian ini terletak pada tiga aspek. Pipeline dua tahap yang diusulkan menggabungkan reduksi dimensi PCA dengan klasterisasi tidak terawasi sebagai generator label otomatis — sebuah pendekatan yang menghilangkan kebutuhan anotasi manual multilabel yang mahal. Lebih lanjut, evaluasi komparatif antara K-Means dan GMM dilakukan tidak hanya pada metrik klasterisasi konvensional, tetapi juga pada dampaknya terhadap kualitas label target di tahap supervisi. Akhirnya, justifikasi empiris terhadap kebijakan outlier preservation disajikan melalui analisis kuantitatif kehilangan data per komoditas."*

---

### Temuan 2 — Frasa klise yang berlebihan
> *"...menjadi kebutuhan krusial dalam menunjang ketahanan pangan dan pertanian presisi"*

**Indikator:** "Kebutuhan krusial", "menunjang ketahanan pangan" adalah frasa yang muncul secara generik dalam ribuan paper AI-generated tentang pertanian. Tidak ada spesifisitas.

**Revisi:** *"...menjadikan sistem pendukung keputusan agronomi berbasis data sebagai komponen penting dalam rantai produksi pertanian modern."*

---

### Temuan 3 — Kalimat terlalu panjang dan multi-klausa
> *"Dengan memetakan sebaran lahan secara alami, setiap klaster tanah yang terbentuk secara inheren akan diisi oleh beberapa komoditas tanaman yang memiliki karakteristik hara yang serupa."*

**Indikator:** Kalimat panjang dengan anak klausa berlapis yang masih bisa dibaca tapi terasa seperti *fluent filler text* — informatif tapi tidak menambah presisi ilmiah.

**Revisi:** *"Setiap klaster yang terbentuk secara alami mengandung beberapa komoditas dengan profil hara serupa, sehingga keanggotaan klaster dapat langsung digunakan sebagai label multilabel tanpa anotasi manual."*

---

### Temuan 4 — Penggunaan berlebihan kata sifat evaluatif
Beberapa frasa seperti "terbukti adaptif", "lebih sesuai", "merefleksikan" muncul berulang di seksi Results tanpa diikuti angka atau sitasi pendukung. Ini adalah pola AI yang mengevaluasi hasil secara subjektif tanpa basis kuantitatif yang kuat.

---

## I. SINTA 3 READINESS ASSESSMENT

| Kriteria | Skor | Catatan |
|:---|:---:|:---|
| Scientific Quality | 62/100 | Topik relevan, namun gap dan kontribusi belum solid |
| Novelty | 52/100 | Incremental; perlu differentiator yang lebih kuat |
| Methodology | 58/100 | Tidak ada rumus matematis; GMM n_init=1 lemah; tidak ada cross-validation |
| Experimental Design | 50/100 | Single split; tidak ada baseline komparatif; circular evaluation risk |
| Results & Discussion | 65/100 | Error analysis baik; centroid analysis kurang data kuantitatif |
| Academic Writing | 68/100 | Readable; beberapa pola AI-like; inkonsistensi terminologi |
| Reproducibility | 63/100 | Hyperparameter ada; rumus tidak ada; versi library parsial |
| **Overall SINTA 3 Readiness** | **60/100** | **Perlu revisi mayor** |

> **Interpretasi: Belum layak submit (60/100) → Perlu revisi mayor sebelum submission.**

---

## J. CRITICAL ISSUES

### 🔴 Critical (Dapat Menyebabkan Penolakan Langsung)

1. **Referensi fiktif di Tabel 1** — "Prasad & Kumar [13] (2026)" dan "Ingle et al. [11] (2024)" dengan klaim yang tidak dapat diverifikasi adalah *academic integrity issue*. Harus diganti dengan sitasi yang benar-benar ada.
2. **Tidak ada rumusan masalah (research question) eksplisit** — Reviewer tidak dapat menilai apakah eksperimen menjawab pertanyaan yang diajukan.
3. **Daftar pustaka kosong** — Section 5 hanya berisi keyword placeholder. Jurnal SINTA 3 membutuhkan minimal 15–20 referensi terverifikasi dalam format standar (IEEE/APA).
4. **Circular evaluation** — Evaluasi model Tahap 2 menggunakan label yang dibangkitkan dari Tahap 1 tanpa validasi eksternal atau ground truth agronomi independen.

### 🟠 Major (Berpengaruh Signifikan terhadap Skor Review)

5. **Tidak ada rumus matematika** — Setidaknya K-Means objective, GMM likelihood, Silhouette Score, dan F1-Score multilabel harus diformulasikan.
6. **Tidak ada cross-validation** — Single 80:20 split dengan satu random seed tidak membuktikan generalisasi.
7. **Tidak ada baseline komparatif eksternal** — Perlu minimal satu metode pembanding di luar K-Means/GMM (misalnya: Direct RF tanpa clustering, atau Spectral Clustering).
8. **Section 3.3 (Centroid Interpretation) tidak didukung tabel kuantitatif** — Klaim nilai centroid harus dibuktikan dengan tabel rata-rata fitur per klaster.

### 🟡 Minor (Dapat Diperbaiki dengan Mudah)

9. Inkonsistensi penggunaan istilah bilingual (outlier/pencilan, clustering/klasterisasi).
10. Judul paper terlalu generik dan tidak mencerminkan kontribusi spesifik.
11. Tujuan penelitian tidak diformulasikan secara eksplisit dalam satu paragraf terpisah.
12. Kalimat duplikasi di akhir Section 3.1 perlu dihapus.
13. Section 4.2 (Saran) terlalu singkat — hanya 2 poin tanpa justifikasi mendalam.

---

## K. REVIEWER RECOMMENDATION

> ### ⚠️ MAJOR REVISION BEFORE SUBMISSION

**Alasan:**

Paper ini memiliki **fondasi ide yang menarik** — pendekatan dua tahap dengan outlier preservation justification adalah aspek yang berpotensi menjadi kontribusi nyata. Namun dalam kondisi saat ini, paper belum memenuhi standar minimum untuk SINTA 3 karena:

1. Referensi gap analysis tidak terverifikasi dan berpotensi merusak kredibilitas akademik.
2. Tidak ada satupun persamaan matematis dalam paper yang bersifat *technical contribution*.
3. Evaluasi bersifat single-split dan berisiko circular.
4. Daftar pustaka belum ada sama sekali.

**Prioritas Revisi Sebelum Submit (urutan dampak):**
1. 🔴 Ganti seluruh referensi Tabel 1 dengan sitasi terverifikasi yang benar-benar ada.
2. 🔴 Buat Daftar Pustaka lengkap (15–20 referensi).
3. 🔴 Tambahkan Research Question/Rumusan Masalah eksplisit.
4. 🟠 Tambahkan minimal 3–5 persamaan matematis kunci.
5. 🟠 Lakukan 5-fold cross-validation atau setidaknya 3 kali pengulangan dengan seed berbeda.
6. 🟠 Tambahkan tabel centroid per klaster untuk mendukung Section 3.3.
7. 🟡 Revisi judul paper.
8. 🟡 Standarisasi terminologi bilingual.
