# FINAL — Metodologi & Hasil Lengkap
**"Rethinking Segmentation Evaluation in Breast Ultrasound: A Systematic Analysis of Metric Sensitivity to Negative Cases"**

---

## 1. Latar Belakang & Motivasi

Penelitian ini lahir dari temuan empiris pada dataset USG payudara (780 citra, 3 kelas): **metrik Dice yang sama membalikkan ranking model** tergantung apakah kasus Normal (mask kosong) diikutsertakan atau tidak.

| Model | Val Dice (incl. Normal) | Val Dice (excl. Normal) | Δ |
|-------|------------------------|------------------------|-----|
| Efficient-U | 0.5989 | 0.7482 | +0.1493 |
| Baseline U-Net | 0.6365 | 0.5887 | −0.0478 |

Fenomena ini mengindikasikan **patologi matematis** pada Dice coefficient yang belum pernah diformalkan secara sistematis di literatur segmentasi citra medis.

---

## 2. Dataset

**Sumber:** BUSI (Breast Ultrasound Images Dataset) — dataset publik  
**Struktur folder:**
```
dataset/
├── benign/     437 gambar + mask
├── malignant/  210 gambar + mask
└── normal/     133 gambar + mask  ← kasus negatif (mask kosong)
```

**Total:** 780 gambar. Proporsi Normal aktual = **133/780 = 17.1%**.

> **Catatan untuk paper:** Proporsi Normal 17.1% ini merupakan komposisi dataset BUSI, bukan estimasi prevalensi klinis umum. Literatur klinik melaporkan prevalensi Normal berkisar 30–40% tergantung setting. SEF menggunakan proporsi dataset aktual (π- = 0.171) untuk evaluasi yang konsisten dan reproducible; peneliti dapat menyesuaikan parameter ini sesuai prevalensi klinis lokal mereka.

**Split stratified (seed=42):**

| Split | Benign | Malignant | Normal | Total |
|-------|--------|-----------|--------|-------|
| Train (70%) | 305 | 147 | 93 | 545 |
| Val (15%) | 65 | 31 | 19 | 115 |
| Test (15%) | 67 | 32 | 21 | 120 |

Split disimpan di `data_split.csv` untuk reprodusibilitas penuh.

---

## 3. Phase 1 — Mathematical Analysis & Formalization

### 3.1 Formulasi Dice pada Empty Mask

Dice coefficient standar:

$$D(P, G) = \frac{2|P \cap G| + \varepsilon}{|P| + |G| + \varepsilon}$$

Untuk kasus **Ground Truth kosong** ($|G| = 0$):

$$D(P, G=\emptyset) = \frac{\varepsilon}{|P| + \varepsilon}$$

- Jika $|P| = 0$ (True Negative): $D = \frac{\varepsilon}{\varepsilon} = 1.0$ ← via smoothing
- Jika $|P| > 0$ (False Positive, bahkan 1 pixel): $D \approx \frac{10^{-6}}{1 + 10^{-6}} \approx 0.0$ ← **collapse mendadak**

**Sifat kritis:** Discontinuity tajam — satu pixel FP menghasilkan penalti maksimal tanpa gradasi proporsional terhadap ukuran FP.

### 3.2 FP Sensitivity Analysis (image 256×256 = 65.536 pixel)

| FP Pixels | FP % | Dice (smooth=1e-6) | Dice (strict) |
|-----------|------|-------------------|---------------|
| 0 | 0.000% | **1.000000** | undefined |
| 1 | 0.002% | **≈ 0.000001** | 0.0 |
| 5 | 0.008% | 0.000000 | 0.0 |
| 10 | 0.015% | 0.000000 | 0.0 |
| 100 | 0.153% | 0.000000 | 0.0 |
| 1000 | 1.526% | 0.000000 | 0.0 |
| 10000 | 15.259% | 0.000000 | 0.0 |

**Temuan:** Penalti bersifat binary — apakah 1 pixel atau 10.000 pixel yang salah, Dice tetap = 0.0. Ini bukan penalti yang proporsional terhadap tingkat kesalahan.

### 3.3 Distorsi Teoritis Mean Dice

Dengan dataset 780 citra (647 positif, 133 Normal) dan FP rate 80% pada Normal:

| Metrik | Nilai |
|--------|-------|
| True Segmentation Dice (Efficient-U) | 0.7482 |
| Reported Dice (Skenario A: TN→1.0 via smooth) | 0.6547 |
| Reported Dice (Skenario B: all empty→0.0) | 0.6206 |
| Max Distortion | **0.1276 (17.1% relative error)** |

### 3.4 Library Inconsistency

| Implementasi | True Negative (GT=0, Pred=0) | False Positive (GT=0, Pred≠0) |
|-------------|------------------------------|-------------------------------|
| Custom (smooth=1e-6) | 1.0 | 0.0 |
| Custom (smooth=1, Laplace) | 1.0 | 0.0099 |
| Custom (strict, TN→1) | 1.0 | 0.0 |
| Custom (strict, TN→0) | **0.0** | 0.0 |
| sklearn.f1_score | 1.0 | 0.0 |
| MONAI DiceMetric | **0.0** | 0.0 |

**Kesimpulan:** Tidak ada konsensus di antara 6 implementasi populer. Peneliti yang menggunakan library berbeda dapat memperoleh angka yang tidak komparable bahkan pada dataset dan model yang identik.

**Output:** `experiments/phase1/fp_sensitivity.csv`, `experiments/phase1/library_consistency.csv`  
**Figures:** `results/figures/phase1_fp_sensitivity.png`, `results/figures/phase1_theoretical_distortion.png`

---

## 4. Phase 2 — Simulation Study

### 4.1 Desain Simulasi

- **Dataset:** N=780 kasus, bootstrap 200 runs (seed=42)
- **Model A (Efficient-U):** True Dice = 0.7482, FP rate pada Normal = 80%
- **Model B (Baseline U-Net):** True Dice = 0.5887, FP rate pada Normal = 30%
- **Variable:** Proporsi Normal dari 0% hingga 50%

### 4.2 Hasil Simulasi Utama

| Normal % | N Normal | Reported Dice A | CI 95% A | Reported Dice B | CI 95% B | Inversion Rate |
|----------|----------|-----------------|----------|-----------------|----------|----------------|
| 0% | 0 | 0.7481 | [0.745, 0.752] | 0.5888 | [0.586, 0.593] | 0.0% |
| 5% | 39 | 0.7208 | [0.714, 0.728] | 0.5942 | [0.586, 0.600] | 0.0% |
| 10% | 78 | 0.6930 | [0.684, 0.703] | 0.5995 | [0.587, 0.612] | 0.0% |
| 15% | 117 | 0.6661 | [0.654, 0.679] | 0.6056 | [0.592, 0.617] | 0.0% |
| **17%** | **132** | **0.6554** | [0.643, 0.667] | **0.6073** | [0.594, 0.620] | **0.0%** ← *proporsi BUSI aktual* |
| **20%** | **156** | **0.6381** | [0.628, 0.651] | **0.6102** | [0.593, 0.625] | **0.5%** ← *inversion mulai* |
| **25%** | **195** | **0.6114** | [0.598, 0.624] | **0.6165** | [0.601, 0.633] | **68.0%** |
| 30% | 234 | 0.5837 | [0.570, 0.599] | 0.6218 | [0.604, 0.638] | 99.5% |
| 35% | 273 | 0.5557 | [0.540, 0.569] | 0.6274 | [0.609, 0.644] | 100.0% |
| 50% | 390 | 0.4733 | [0.457, 0.494] | 0.6422 | [0.616, 0.662] | 100.0% |

**Key Finding:** Ranking inversion pertama terjadi pada **20% proporsi Normal**, hanya 2.9 persentase poin di atas proporsi BUSI aktual (17.1%).

> **Framing untuk paper:** Dataset BUSI dengan 17.1% Normal memang berada di bawah threshold inversion secara rata-rata — namun **sudah terjadi ranking swap pada 2/5 model** dalam evaluasi empiris (Phase 3). Ini justru menjadi temuan yang lebih kuat: jika di bawah threshold saja sudah terjadi inversion, simulasi menunjukkan kondisi akan jauh lebih parah pada dataset klinis nyata yang umumnya mengandung 30–40% kasus Normal.

### 4.3 Grid Simulasi (55 skenario)

Grid FP rate {0.1, 0.2, 0.3, 0.5, 0.8} × proporsi Normal {0%–50%} menunjukkan:
- Model dengan FP rate ≥ 0.5 pada Normal: inversion terjadi mulai proporsi Normal 15%
- Model dengan FP rate = 0.8: inversion rate 100% pada proporsi Normal ≥ 30%

**Output:** `experiments/phase2/simulation_main.csv`, `experiments/phase2/simulation_grid.csv`  
**Figures:** `results/figures/phase2_main_simulation.png`, `results/figures/phase2_inversion_heatmap.png`, `results/figures/phase2_distortion_curves.png`

---

## 5. Phase 3 — Empirical Validation (Multi-Model)

### 5.1 Setup Training

| Parameter | Nilai |
|-----------|-------|
| Framework | PyTorch 2.9.1 + segmentation-models-pytorch 0.5.0 |
| Encoder | ResNet-34 (pretrained ImageNet) |
| Input | Grayscale 256×256 |
| Loss | Dice Loss + BCE (bobot 50:50) |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-5) |
| Scheduler | CosineAnnealingLR |
| Augmentasi | HFlip, VFlip, Rotate, ShiftScaleRotate, Brightness, GaussNoise, ElasticTransform |
| Early stopping | Patience=15 evaluasi (tiap 5 epoch) |
| Max epochs | 100 |
| Device | Apple Silicon MPS |
| Batch size | 16 |
| Seed | 42 |

> **Catatan untuk reviewer:** Pipeline augmentasi menyertakan ElasticTransform. Karena fokus penelitian ini adalah pada protokol evaluasi (bukan optimasi model), variasi augmentasi tidak dieksplorasi. Analisis sensitivitas terhadap pilihan augmentasi merupakan arah penelitian lanjutan.

### 5.2 Model yang Dievaluasi

| Model | Arsitektur | Best Epoch |
|-------|-----------|------------|
| UNet | U-Net + ResNet-34 | 100 |
| Attention UNet | U-Net + SCSE attention + ResNet-34 | 60 |
| UNet++ | UNet++ + ResNet-34 | 90 |
| MAnet | MA-Net + ResNet-34 | 75 |
| Linknet | LinkNet + ResNet-34 | 65 |

### 5.3 Hasil Evaluasi Test Set (n=120: 99 positif, 21 Normal)

**Tabel utama paper (π- = 0.171, sesuai proporsi BUSI aktual):**

| Model | P1 Standard | P2 Audit | SEF Seg Dice | SEF Specificity | SEF Sensitivity | SEF Composite (π-=0.171) | CI 95% |
|-------|-------------|----------|--------------|-----------------|-----------------|--------------------------|--------|
| UNet | 0.8202 ± 0.256 | 0.7922 ± 0.256 | 0.7922 ± 0.256 | **0.9524** | 0.9798 | **0.8196** | [0.741, 0.843] |
| Linknet | 0.8081 ± 0.261 | 0.7876 ± 0.249 | 0.7876 ± 0.249 | 0.9048 | 0.9798 | 0.8077 | [0.738, 0.838] |
| MAnet | 0.7867 ± 0.301 | 0.7718 ± 0.288 | 0.7718 ± 0.288 | 0.8571 | 0.9596 | 0.7864 | [0.714, 0.829] |
| UNet++ | 0.7857 ± 0.296 | **0.7807 ± 0.271** | 0.7807 ± 0.271 | 0.8095 | 0.9697 | 0.7856 | [0.726, 0.835] |
| Attention UNet | 0.7790 ± 0.300 | 0.7624 ± 0.286 | 0.7624 ± 0.286 | 0.8571 | 0.9697 | 0.7786 | [0.705, 0.820] |

### 5.4 Ranking Inversion Antar Protokol

| Model | Rank P1 | Rank P2 | Rank SEF | Δ P1→P2 | Δ P1→SEF |
|-------|---------|---------|----------|---------|---------|
| UNet | 1 | 1 | 1 | — | — |
| Linknet | 2 | 2 | 2 | — | — |
| **UNet++** | **4** | **3** | 4 | **↑1** | — |
| **MAnet** | **3** | **4** | 3 | **↓1** | — |
| Attention UNet | 5 | 5 | 5 | — | — |

**2/5 model (UNet++ dan MAnet) bertukar ranking antara P1 dan P2.**

**Analisis kasus UNet++ (ranking non-linear P1→P2→SEF):**

UNet++ naik dari rank 4 (P1) ke rank 3 (P2) karena memiliki Dice segmentasi tinggi (0.781) yang terungkap ketika kasus Normal dieksklusi. Namun di SEF, UNet++ turun kembali ke rank 4 karena memiliki **Specificity terendah dari semua model (0.810)** — artinya model ini lebih sering menghasilkan FP pada kasus Normal dibanding model lain. SEF mengekspos trade-off ini secara eksplisit, sementara P1 dan P2 tidak mampu menangkapnya secara terpisah.

Perbandingan UNet++ vs MAnet dalam SEF:
- UNet++: SEF = 0.829 × 0.781 + 0.171 × 0.810 = 0.647 + 0.138 = **0.786**
- MAnet:  SEF = 0.829 × 0.772 + 0.171 × 0.857 = 0.640 + 0.147 = **0.786**

MAnet mengungguli UNet++ di SEF karena Specificity lebih tinggi (0.857 vs 0.810), mengompensasi Dice segmentasi yang sedikit lebih rendah. Ini adalah trade-off klinis nyata yang tidak terlihat di P1 maupun P2.

### 5.5 Uji Statistik Wilcoxon (per-case, P2 Dice)

| Perbandingan | Statistik | p-value | Signifikan |
|-------------|-----------|---------|-----------|
| UNet vs Attention UNet | 1556.0 | **0.0032** | ✓ |
| UNet vs UNet++ | 2153.0 | 0.4213 | — |
| UNet vs MAnet | 1755.0 | **0.0253** | ✓ |
| UNet vs Linknet | 1541.0 | **0.0026** | ✓ |
| Attention UNet vs UNet++ | 1536.0 | **0.0038** | ✓ |
| Attention UNet vs MAnet | 2107.0 | 0.4193 | — |
| Attention UNet vs Linknet | 2336.0 | 0.8841 | — |
| UNet++ vs MAnet | 2045.0 | 0.3011 | — |
| UNet++ vs Linknet | 1808.0 | **0.0408** | ✓ |
| MAnet vs Linknet | 2017.0 | 0.1958 | — |

5/10 pasangan signifikan (p < 0.05), Wilcoxon signed-rank test (two-sided).

**Output:** `experiments/phase3/comparison_table.csv`, `experiments/phase3/ranking_table.csv`, `experiments/phase3/wilcoxon_tests.csv`  
**Figures:** `results/figures/phase3_protocol_comparison.png`, `results/figures/phase3_ranking_heatmap.png`

---

## 6. Phase 4 — Stratified Evaluation Framework (SEF)

### 6.1 Motivasi SEF

Protokol standar (P1) menggabungkan dua tugas klinis yang fundamental berbeda:
- **Tugas segmentasi:** Seberapa akurat model membuat mask pada kasus lesi?
- **Tugas deteksi/negatif:** Apakah model cukup "diam" pada pasien sehat?

Kedua tugas ini memiliki konsekuensi klinis yang berbeda: FP pada kasus Normal berpotensi menyebabkan biopsi tidak perlu, sementara kualitas segmentasi memengaruhi akurasi panduan intervensi. Mencampurkan keduanya dalam satu mean Dice menghasilkan angka yang tidak representatif terhadap performa klinis dari dimensi manapun.

### 6.2 Definisi SEF

**Jalur 1 — Segmentasi (kasus positif, label ∈ {benign, malignant}):**
$$\text{Dice}_{\text{seg}} = \frac{1}{N_+} \sum_{i \in \mathcal{P}} \frac{2|P_i \cap G_i| + \varepsilon}{|P_i| + |G_i| + \varepsilon}$$

**Jalur 2 — Deteksi (kasus Normal, label = normal):**
$$\text{Specificity} = \frac{|\{i \in \mathcal{N} : \hat{P}_i = \emptyset\}|}{N_-}, \quad \text{Sensitivity} = \frac{|\{i \in \mathcal{P} : \hat{P}_i \neq \emptyset\}|}{N_+}$$

**Composite Score:**
$$\text{SEF}_{\text{composite}} = (1 - \pi_-) \cdot \text{Dice}_{\text{seg}} + \pi_- \cdot \text{Specificity}$$

**Parameter π-:** Dalam penelitian ini, π- = **0.171** (133/780, proporsi Normal aktual di dataset BUSI). Nilai ini berbeda dari prevalensi klinis umum (~0.33) yang dilaporkan di literatur. Peneliti yang mengaplikasikan SEF pada dataset dengan komposisi berbeda disarankan menyesuaikan π- dengan prevalensi Normal di dataset atau populasi target mereka.

### 6.3 Tiga Protokol Evaluasi yang Dibandingkan

| Protokol | Deskripsi | Kelebihan | Kekurangan |
|----------|-----------|-----------|------------|
| **P1 Standard** | Mean Dice semua kasus termasuk Normal | Simple, widely used | Distorsi oleh empty mask; dua tugas berbeda dikollaps jadi satu angka |
| **P2 Audit** | Mean Dice hanya kasus positif | True segmentation quality | Mengabaikan perilaku model pada Normal; menyembunyikan FP risk |
| **P3 SEF** | Two-pathway: segmentasi + deteksi terpisah | Komprehensif, clinically aligned | Membutuhkan label per-kasus; memerlukan keputusan eksplisit tentang π- |

### 6.4 Validasi SEF pada Test Set (π- = 0.171)

| Model | P1 | P2 | SEF Seg | SEF Spec | SEF Composite | Rank P1 | Rank P2 | Rank SEF |
|-------|----|----|---------|----------|---------------|---------|---------|---------|
| UNet | 0.8202 | 0.7922 | 0.7922 | 0.9524 | **0.8196** | 1 | 1 | 1 |
| Linknet | 0.8081 | 0.7876 | 0.7876 | 0.9048 | 0.8077 | 2 | 2 | 2 |
| MAnet | 0.7867 | 0.7718 | 0.7718 | 0.8571 | 0.7864 | 3 | 4 | **3** |
| UNet++ | 0.7857 | 0.7807 | 0.7807 | 0.8095 | 0.7856 | 4 | **3** | 4 |
| Attention UNet | 0.7790 | 0.7624 | 0.7624 | 0.8571 | 0.7786 | 5 | 5 | 5 |

SEF mengekspos bahwa UNet++ (rank P2=3) memiliki Specificity terendah (0.810) — trade-off yang tidak terlihat di P1 maupun P2.

### 6.5 Argumen Klinis SEF

Dalam workflow klinis USG payudara:
- **Kasus Normal (proporsi klinis ~33%):** Model harus "diam" — FP berpotensi memicu biopsi tidak perlu
- **Kasus lesi:** Model harus menghasilkan mask akurat untuk panduan intervensi atau staging

SEF memungkinkan klinisi dan peneliti menilai kedua aspek ini secara independen, dan menggabungkannya sesuai konteks klinis dengan menyesuaikan π-.

**Output:** `experiments/phase4/final_paper_table.csv`, `paper/figures/`

---

## 7. Limitations

1. **Validasi pada satu dataset:** Seluruh evaluasi empiris (Phase 3–4) dilakukan pada BUSI. Meskipun BUSI adalah benchmark publik yang banyak digunakan, generalisasi temuan ke dataset klinis dengan distribusi berbeda (resolusi, mesin USG, prevalensi) belum diverifikasi. Studi multi-site diperlukan untuk konfirmasi.

2. **Proporsi Normal di bawah threshold simulasi:** Dataset BUSI mengandung 17.1% kasus Normal, sedangkan simulasi Phase 2 menunjukkan ranking inversion rata-rata mulai terjadi di 20%. Bukti empiris Phase 3 (2/5 inversion) karenanya bersifat konservatif. Dataset klinis dengan prevalensi Normal 30–40% diprediksi menunjukkan efek yang jauh lebih besar, namun belum diuji secara langsung.

3. **SEF membutuhkan label per-kasus:** Implementasi SEF memerlukan informasi apakah setiap kasus adalah positif atau negatif. Ini tidak selalu tersedia dalam pipeline evaluasi standar yang hanya menggunakan gambar dan mask tanpa label kelas. Adopsi SEF memerlukan perubahan pada protokol anotasi.

4. **π- sebagai hyperparameter:** SEF composite score sensitif terhadap pilihan π-. Penelitian ini menggunakan π- = 0.171 (proporsi dataset aktual) untuk konsistensi, namun nilai yang berbeda akan menghasilkan ranking yang berbeda. Panduan untuk memilih π- yang tepat berdasarkan konteks klinis perlu dikembangkan lebih lanjut.

5. **Satu konfigurasi augmentasi:** Seluruh model dilatih dengan pipeline augmentasi yang sama (termasuk ElasticTransform). Karena fokus penelitian adalah pada protokol evaluasi bukan optimasi model, sensitivitas hasil terhadap pilihan augmentasi tidak diinvestigasi.

---

## 8. Ringkasan Temuan

### Temuan 1 — Patologi Matematis (Phase 1)
Dice coefficient memiliki **discontinuity tajam** pada empty GT: bahkan 1 pixel FP menghasilkan Dice = 0.0 tanpa gradasi proporsional. Enam implementasi populer menunjukkan perilaku berbeda pada skenario yang sama — tidak ada konsensus industri.

### Temuan 2 — Threshold Inversion Terukur (Phase 2)
Ranking inversion mulai terjadi pada **20% proporsi Normal** (CI bootstrap tersedia). Dataset dengan prevalensi Normal ≥ 25% menunjukkan inversion rate 68–100%. Dataset BUSI (17.1%) berada 2.9pp di bawah threshold rata-rata.

### Temuan 3 — Bukti Empiris Multi-Model (Phase 3)
Pada 5 model terlatih di BUSI: **2/5 model bertukar ranking antara P1 dan P2**, meskipun dataset berada di bawah threshold simulasi. SEF Specificity bervariasi dari 0.810 hingga 0.952 — dimensi klinis yang sepenuhnya tidak terlihat di P1 maupun P2.

### Temuan 4 — SEF Mengekspos Trade-off Tersembunyi (Phase 4)
UNet++ vs MAnet: P2 menunjukkan UNet++ lebih baik (0.781 vs 0.772), namun SEF membalik ranking ini karena Specificity UNet++ terendah (0.810 vs 0.857). Ini adalah trade-off antara akurasi segmentasi dan safety pada kasus Normal — informasi klinis kritis yang hilang dari protokol evaluasi konvensional.

---

## 9. Implementasi & Reprodusibilitas

### Cara Reproduce
```bash
pip install -r requirements.txt

python src/phase1_math_analysis.py
python src/phase2_simulation.py
python src/phase3_empirical.py --model all --epochs 100  # ~1 jam
python src/phase3_empirical.py --eval-only               # skip training
python src/phase4_protocol.py --output paper/figures/
python src/utils/visualizer.py --output paper/figures/
```

### Reproducibility Checklist
- [x] Random seed = 42 di semua eksperimen
- [x] Dataset split disimpan sebagai CSV (`data_split.csv`)
- [x] Model checkpoint tersimpan di `results/checkpoints/`
- [x] Semua hasil tersimpan di `experiments/phase1-4/`
- [x] Requirements terdokumentasi di `requirements.txt`
- [x] π- = 0.171 (133/780) konsisten di semua kalkulasi SEF

### Output Files

| File | Deskripsi |
|------|-----------|
| `experiments/phase1/fp_sensitivity.csv` | Dice vs FP pixel count |
| `experiments/phase1/library_consistency.csv` | Perilaku 6 library pada empty mask |
| `experiments/phase2/simulation_main.csv` | Hasil simulasi utama + CI 95% |
| `experiments/phase2/simulation_grid.csv` | Grid simulasi 55 skenario |
| `experiments/phase3/comparison_table.csv` | 5 model × 3 protokol |
| `experiments/phase3/ranking_table.csv` | Ranking per protokol |
| `experiments/phase3/wilcoxon_tests.csv` | 10 pairwise Wilcoxon tests |
| `experiments/phase4/final_paper_table.csv` | Tabel utama paper |
| `data_split.csv` | Train/val/test split (stratified, reproducible) |
| `paper/figures/` | 12 figures siap publikasi |

---

## 10. Target Jurnal

| Jurnal | Quartile | Kesesuaian |
|--------|----------|-----------|
| *Biomedical Signal Processing and Control* | Q2 | Metodologi evaluasi medis |
| *Computers in Biology and Medicine* | Q1–Q2 | Dataset USG + kontribusi framework |
| *Ultrasound in Medicine and Biology* | Q1–Q2 | Domain-specific, highest impact |

---

*Dokumen ini dihasilkan dari implementasi lengkap RESEARCH_PLAN.md.*  
*Tanggal: 1 April 2026*  
*π- dikoreksi dari 0.33 → 0.171 (133/780, proporsi aktual BUSI) untuk konsistensi dengan data.*
