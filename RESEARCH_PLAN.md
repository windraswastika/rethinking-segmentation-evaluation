# Research Plan — Gap 1: Dice Metric Pathology on Negative Cases
**"Rethinking Segmentation Evaluation in Breast Ultrasound: A Systematic Analysis of Metric Sensitivity to Negative Cases"**

---

## Konteks & Motivasi

Penelitian ini lahir dari temuan empiris berikut pada dataset USG payudara internal (780 citra, 3 kelas):

```
Model             Val Dice (incl. Normal)   Val Dice (excl. Normal)   Δ
─────────────────────────────────────────────────────────────────────
Efficient-U                0.5989                    0.7482          +0.1493
Baseline U-Net             0.6365                    0.5887          -0.0478
```

Temuan kritis: **metrik yang sama membalikkan ranking model** tergantung apakah kasus Normal (mask kosong) diikutsertakan atau tidak. Ini mengindikasikan bahwa Dice coefficient memiliki perilaku patologis pada empty mask yang belum pernah diformalkan di literatur segmentasi citra medis.

---

## Tujuan Penelitian

1. Membuktikan secara matematis dan empiris bahwa Dice coefficient menghasilkan nilai yang misleading pada kasus dengan ground truth mask kosong
2. Mengkuantifikasi efek distorsi sebagai fungsi dari proporsi kasus negatif dalam dataset
3. Menunjukkan bahwa fenomena ini menyebabkan ranking inversion yang merusak reproducibility penelitian
4. Mengusulkan Stratified Evaluation Framework (SEF) sebagai protokol evaluasi baru

---

## Struktur Direktori Proyek

```
research/
├── RESEARCH_PLAN.md              ← file ini
├── data/
│   ├── internal/                 ← dataset USG internal (780 citra)
│   │   ├── images/
│   │   ├── masks/
│   │   └── labels.csv            ← kolom: filename, class (normal/benign/malignant)
│   └── busi/                     ← dataset publik BUSI untuk validasi eksternal
│       ├── images/
│       └── masks/
├── src/
│   ├── phase1_math_analysis.py
│   ├── phase2_simulation.py
│   ├── phase3_empirical.py
│   ├── phase4_protocol.py
│   ├── models/
│   │   ├── unet.py
│   │   ├── attention_unet.py
│   │   ├── efficient_u.py        ← model dari Stage 3 sebelumnya
│   │   └── transunet.py
│   ├── metrics/
│   │   ├── standard_metrics.py   ← Dice, IoU, F1 standar
│   │   └── sef_metrics.py        ← Stratified Evaluation Framework (kontribusi baru)
│   └── utils/
│       ├── dataset.py
│       ├── trainer.py
│       └── visualizer.py
├── experiments/
│   ├── phase1/
│   ├── phase2/
│   ├── phase3/
│   └── phase4/
├── results/
│   ├── figures/
│   ├── tables/
│   └── checkpoints/
└── paper/
    └── figures/                  ← figure siap publikasi
```

---

## Phase 1 — Mathematical Analysis & Formalization
**Output:** Bukti matematis + 1 figure (kurva sensitivitas teoretis)

### 1.1 Formalisasi Patologi Dice pada Empty Mask

Implementasikan `src/phase1_math_analysis.py`:

```python
# Tujuan: buktikan secara analitik dan numerik perilaku Dice pada GT kosong

import numpy as np
import matplotlib.pyplot as plt

def dice_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    """Implementasi Dice standar — TANPA handling khusus untuk empty mask."""
    intersection = (pred * gt).sum()
    return (2 * intersection + smooth) / (pred.sum() + gt.sum() + smooth)

def dice_undefined_behavior(gt_empty: bool, pred_has_fp: bool) -> str:
    """Peta seluruh kombinasi kasus GT kosong."""
    if gt_empty and pred_has_fp:
        return "0.0  ← misleading: model salah tapi tidak proporsional"
    if gt_empty and not pred_has_fp:
        return "1.0 atau 0/0 ← implementation-dependent, tidak konsisten"
    return "normal Dice behavior"

def analyze_fp_sensitivity(image_size: int = 256, fp_pixel_counts: list = None):
    """
    Hitung Dice sebagai fungsi dari jumlah FP pixel pada GT kosong.
    Tunjukkan: bahkan 1 pixel FP menghasilkan Dice = 0.0
    """
    if fp_pixel_counts is None:
        fp_pixel_counts = [0, 1, 5, 10, 50, 100, 500, 1000]

    gt = np.zeros((image_size, image_size))
    results = []

    for n_fp in fp_pixel_counts:
        pred = np.zeros((image_size, image_size))
        if n_fp > 0:
            indices = np.random.choice(image_size * image_size, n_fp, replace=False)
            pred.flat[indices] = 1.0
        score = dice_score(pred, gt)
        results.append({
            "n_fp_pixels": n_fp,
            "fp_percentage": n_fp / (image_size ** 2) * 100,
            "dice": score,
        })

    return results

def theoretical_mean_dice_drop(
    n_positive_cases: int,
    n_negative_cases: int,
    true_positive_dice: float,
    model_fp_rate_on_negatives: float,
) -> dict:
    """
    Hitung secara teoretis berapa expected mean Dice setelah memasukkan kasus Normal.

    Asumsi: setiap kasus Normal dengan FP menghasilkan Dice = 0.0
    """
    expected_dice_on_negatives = (1 - model_fp_rate_on_negatives) * 1.0  # true negative → Dice=1.0 (atau undefined)

    # Skenario A: implementasi Dice=1.0 untuk true negative (GT=0, Pred=0)
    mean_dice_A = (
        (n_positive_cases * true_positive_dice + n_negative_cases * expected_dice_on_negatives)
        / (n_positive_cases + n_negative_cases)
    )

    # Skenario B: implementasi Dice=0.0 untuk semua kasus GT kosong (termasuk true negative)
    mean_dice_B = (
        n_positive_cases * true_positive_dice
        / (n_positive_cases + n_negative_cases)
    )

    return {
        "true_segmentation_dice": true_positive_dice,
        "reported_dice_scenario_A": mean_dice_A,
        "reported_dice_scenario_B": mean_dice_B,
        "max_distortion": true_positive_dice - min(mean_dice_A, mean_dice_B),
    }

if __name__ == "__main__":
    # Reproduksi temuan empiris awal
    print("=== Reproduksi Temuan Empiris ===")
    result = theoretical_mean_dice_drop(
        n_positive_cases=520,       # ~67% dari 780 (benign + malignant)
        n_negative_cases=260,       # ~33% Normal
        true_positive_dice=0.7482,  # Efficient-U AUDIT Dice
        model_fp_rate_on_negatives=0.8,  # asumsi: 80% Normal case ada FP kecil
    )
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")

    # Simpan hasil analisis sensitivitas
    sensitivity_results = analyze_fp_sensitivity()
    # → save ke experiments/phase1/fp_sensitivity.csv
```

### 1.2 Analisis Implementation Inconsistency

Kumpulkan dan dokumentasikan bagaimana 5 library populer menangani kasus 0/0:

```python
# experiments/phase1/library_consistency_check.py

def check_monai_behavior():
    """Test MONAI DiceLoss dan DiceMetric pada empty mask."""
    pass  # implementasikan dengan: from monai.metrics import DiceMetric

def check_segmentation_models_pytorch():
    """Test smp.losses.DiceLoss."""
    pass  # from segmentation_models_pytorch.losses import DiceLoss

def check_sklearn_f1():
    """Test sklearn F1 score sebagai proxy Dice."""
    pass  # from sklearn.metrics import f1_score

def check_torchmetrics():
    """Test torchmetrics Dice."""
    pass  # from torchmetrics import Dice

def check_custom_implementations():
    """3 variasi implementasi manual dari GitHub populer."""
    pass
```

**Target output Phase 1:**
- [ ] Tabel: perilaku 5 library pada empty mask (1.0 / 0.0 / error / skip)
- [ ] Figure: kurva Dice vs jumlah FP pixel pada GT kosong
- [ ] Persamaan matematis yang membuktikan Dice = 0.0 untuk setiap FP > 0

---

## Phase 2 — Simulation Study
**Output:** Kurva sensitivitas empiris + tabel ranking inversion rate

### 2.1 Desain Simulasi

Implementasikan `src/phase2_simulation.py`:

```python
import numpy as np
import pandas as pd
from itertools import product

NORMAL_PROPORTIONS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
MODEL_FP_RATES     = [0.1, 0.2, 0.3, 0.5]   # seberapa sering model salah prediksi di Normal
N_TOTAL_CASES      = 780
N_SIMULATION_RUNS  = 100                       # bootstrap untuk confidence interval
SEED               = 42

def simulate_metric_distortion(
    normal_proportion: float,
    model_a_true_dice: float,   # Dice pada kasus positif
    model_b_true_dice: float,
    model_a_fp_rate: float,
    model_b_fp_rate: float,
    n_total: int = N_TOTAL_CASES,
    n_runs: int = N_SIMULATION_RUNS,
) -> dict:
    """
    Simulasikan efek memasukkan kasus Normal terhadap reported mean Dice.
    Ukur: apakah ranking model A vs B berubah?
    """
    np.random.seed(SEED)
    n_normal   = int(n_total * normal_proportion)
    n_positive = n_total - n_normal

    ranking_inversions = 0
    reported_dices_a, reported_dices_b = [], []

    for _ in range(n_runs):
        # Dice pada kasus positif (sampling dari distribusi)
        dices_pos_a = np.random.normal(model_a_true_dice, 0.05, n_positive).clip(0, 1)
        dices_pos_b = np.random.normal(model_b_true_dice, 0.05, n_positive).clip(0, 1)

        # Dice pada kasus Normal: 0.0 jika ada FP, 1.0 jika benar-benar kosong
        fp_mask_a   = np.random.binomial(1, model_a_fp_rate, n_normal)
        fp_mask_b   = np.random.binomial(1, model_b_fp_rate, n_normal)
        dices_neg_a = np.where(fp_mask_a, 0.0, 1.0)
        dices_neg_b = np.where(fp_mask_b, 0.0, 1.0)

        mean_a = np.mean(np.concatenate([dices_pos_a, dices_neg_a]))
        mean_b = np.mean(np.concatenate([dices_pos_b, dices_neg_b]))

        reported_dices_a.append(mean_a)
        reported_dices_b.append(mean_b)

        # Cek ranking inversion
        true_winner  = "A" if model_a_true_dice > model_b_true_dice else "B"
        report_winner = "A" if mean_a > mean_b else "B"
        if true_winner != report_winner:
            ranking_inversions += 1

    return {
        "normal_proportion": normal_proportion,
        "ranking_inversion_rate": ranking_inversions / n_runs,
        "mean_reported_dice_a": np.mean(reported_dices_a),
        "mean_reported_dice_b": np.mean(reported_dices_b),
        "ci_95_a": np.percentile(reported_dices_a, [2.5, 97.5]).tolist(),
        "ci_95_b": np.percentile(reported_dices_b, [2.5, 97.5]).tolist(),
    }

def run_full_simulation():
    """Jalankan grid simulasi lengkap."""
    # Gunakan nilai empiris dari dataset Anda sebagai anchor
    model_a = {"true_dice": 0.7482, "fp_rate": 0.8}   # Efficient-U
    model_b = {"true_dice": 0.5887, "fp_rate": 0.3}   # Baseline U-Net

    results = []
    for prop in NORMAL_PROPORTIONS:
        result = simulate_metric_distortion(
            normal_proportion=prop,
            model_a_true_dice=model_a["true_dice"],
            model_b_true_dice=model_b["true_dice"],
            model_a_fp_rate=model_a["fp_rate"],
            model_b_fp_rate=model_b["fp_rate"],
        )
        results.append(result)
        print(f"Normal={prop:.0%} | Inversion rate: {result['ranking_inversion_rate']:.2%}")

    df = pd.DataFrame(results)
    df.to_csv("experiments/phase2/simulation_results.csv", index=False)
    return df
```

### 2.2 Validasi Eksternal di BUSI

Replikasi simulasi menggunakan dataset BUSI yang sudah publik — ini kunci untuk meyakinkan reviewer bahwa fenomenanya bukan artefak dataset internal Anda.

```python
def replicate_on_busi(busi_data_dir: str):
    """
    BUSI: 780 gambar, 3 kelas (normal/benign/malignant)
    Ukuran hampir identik dengan dataset internal → natural replication.
    """
    # Load BUSI, pisahkan Normal dari Benign+Malignant
    # Train 4 model standar
    # Laporkan Dice dengan/tanpa Normal
    # Ukur ranking inversion
    pass
```

**Target output Phase 2:**
- [ ] Figure utama: kurva "Reported Dice vs True Dice" sebagai fungsi proporsi Normal
- [ ] Figure: "Ranking Inversion Rate vs Proporsi Normal" — ini killer figure untuk paper
- [ ] Tabel CI 95% untuk setiap konfigurasi

---

## Phase 3 — Empirical Validation (Multi-Model)
**Output:** Tabel perbandingan 5+ model dengan 3 protokol evaluasi berbeda

### 3.1 Training Pipeline

Implementasikan `src/phase3_empirical.py` — latih 5 model pada dataset yang sama:

```python
MODELS_TO_EVALUATE = [
    "unet",           # baseline klasik
    "attention_unet", # baseline + attention
    "efficient_u",    # model dari Stage 3 penelitian ini
    "transunet",      # transformer-based
    "nnunet",         # auto-configuration U-Net
]

EVALUATION_PROTOCOLS = {
    "P1_standard":  "Dice rata-rata semua kelas termasuk Normal",
    "P2_audit":     "Dice hanya pada kasus positif (benign + malignant)",
    "P3_sef":       "Stratified Evaluation Framework (kontribusi baru)",
}
```

### 3.2 Statistical Significance Testing

```python
from scipy import stats

def wilcoxon_test(scores_a: list, scores_b: list) -> dict:
    """
    Wilcoxon signed-rank test untuk perbandingan per-kasus.
    Lebih tepat dari t-test untuk distribusi Dice yang tidak normal.
    """
    stat, p_value = stats.wilcoxon(scores_a, scores_b)
    return {"statistic": stat, "p_value": p_value, "significant": p_value < 0.05}

def report_with_ci(scores: list, confidence: float = 0.95) -> dict:
    """Bootstrap confidence interval untuk mean Dice."""
    import scipy.stats as st
    n  = len(scores)
    ci = st.t.interval(confidence, df=n-1, loc=np.mean(scores), scale=st.sem(scores))
    return {"mean": np.mean(scores), "ci_low": ci[0], "ci_high": ci[1]}
```

**Target output Phase 3:**
- [ ] Tabel utama paper: 5 model × 3 protokol evaluasi + p-value
- [ ] Heatmap ranking: seberapa sering ranking berubah antar protokol
- [ ] Kasus studi kualitatif: 6 contoh gambar yang ilustrasi masalah secara visual

---

## Phase 4 — Proposed Framework (SEF)
**Output:** Implementasi Stratified Evaluation Framework + validasi

### 4.1 Stratified Evaluation Framework

Implementasikan `src/metrics/sef_metrics.py` — ini kontribusi utama paper:

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class SEFResult:
    segmentation_dice: float       # Dice hanya pada kasus positif
    detection_specificity: float   # TNR hanya pada kasus Normal
    detection_sensitivity: float   # TPR — apakah model mendeteksi ada lesi
    composite_score: float         # gabungan berbobot sesuai prevalensi klinis
    n_positive_cases: int
    n_negative_cases: int

def stratified_evaluation_framework(
    predictions: list,             # list of np.ndarray per kasus
    ground_truths: list,           # list of np.ndarray per kasus
    labels: list,                  # list of str: "normal" / "benign" / "malignant"
    prevalence_negative: float = 0.33,   # proporsi Normal di populasi klinis
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> SEFResult:
    """
    Stratified Evaluation Framework (SEF).

    Memisahkan evaluasi menjadi dua jalur:
    - Jalur Segmentasi: evaluasi kualitas mask pada kasus positif
    - Jalur Deteksi: evaluasi kemampuan model untuk diam pada kasus Normal
    """
    positive_labels = {"benign", "malignant"}
    seg_dices, det_tn, det_tp = [], [], []

    for pred, gt, label in zip(predictions, ground_truths, labels):
        pred_bin = (pred > threshold).astype(np.float32)

        if label in positive_labels:
            # Jalur segmentasi: hitung Dice standar
            intersection = (pred_bin * gt).sum()
            dice = (2 * intersection + smooth) / (pred_bin.sum() + gt.sum() + smooth)
            seg_dices.append(dice)
            # Deteksi sensitivity: apakah model menghasilkan prediksi non-kosong?
            det_tp.append(1.0 if pred_bin.sum() > 0 else 0.0)
        else:
            # Jalur deteksi: apakah model benar-benar diam?
            det_tn.append(1.0 if pred_bin.sum() == 0 else 0.0)

    seg_dice   = np.mean(seg_dices) if seg_dices else 0.0
    specificity = np.mean(det_tn)   if det_tn   else 0.0
    sensitivity = np.mean(det_tp)   if det_tp   else 0.0

    # Composite score: F-beta style gabungan segmentasi + deteksi
    # Bobot sesuai prevalensi klinis: kasus Normal ~33%, kasus lesi ~67%
    w_seg = 1 - prevalence_negative
    w_det = prevalence_negative
    composite = w_seg * seg_dice + w_det * specificity

    return SEFResult(
        segmentation_dice=seg_dice,
        detection_specificity=specificity,
        detection_sensitivity=sensitivity,
        composite_score=composite,
        n_positive_cases=len(seg_dices),
        n_negative_cases=len(det_tn),
    )

def compare_protocols(predictions, ground_truths, labels) -> dict:
    """Bandingkan P1, P2, P3 pada set prediksi yang sama."""
    # P1 — Standard (dengan Normal, implementasi smooth=1e-6)
    all_dices_p1 = []
    for pred, gt in zip(predictions, ground_truths):
        pred_bin = (pred > 0.5).astype(np.float32)
        inter = (pred_bin * gt).sum()
        d = (2 * inter + 1e-6) / (pred_bin.sum() + gt.sum() + 1e-6)
        all_dices_p1.append(d)

    # P2 — Audit (hanya kasus positif)
    positive_indices = [i for i, l in enumerate(labels) if l != "normal"]
    all_dices_p2 = []
    for i in positive_indices:
        pred_bin = (predictions[i] > 0.5).astype(np.float32)
        inter = (pred_bin * ground_truths[i]).sum()
        d = (2 * inter + 1e-6) / (pred_bin.sum() + ground_truths[i].sum() + 1e-6)
        all_dices_p2.append(d)

    # P3 — SEF
    sef = stratified_evaluation_framework(predictions, ground_truths, labels)

    return {
        "P1_standard_mean_dice": np.mean(all_dices_p1),
        "P2_audit_mean_dice":    np.mean(all_dices_p2),
        "P3_sef_seg_dice":       sef.segmentation_dice,
        "P3_sef_specificity":    sef.detection_specificity,
        "P3_sef_composite":      sef.composite_score,
    }
```

**Target output Phase 4:**
- [ ] Implementasi SEF yang fully tested
- [ ] Validasi SEF vs protokol standar pada semua model
- [ ] Tabel: bagaimana SEF mengubah (atau mempertahankan) ranking dibanding P1 dan P2
- [ ] Argumen mengapa SEF lebih aligned dengan clinical workflow

---

## Checklist Reprodusibilitas (untuk Reviewer)

```markdown
### Yang wajib disertakan saat submission:
- [ ] Kode lengkap di GitHub (public repo)
- [ ] Script reproduce semua tabel dan figure
- [ ] Random seed didokumentasikan untuk semua eksperimen
- [ ] Dataset split (train/val/test) disimpan sebagai CSV, bukan hardcoded
- [ ] Requirements.txt / environment.yml
- [ ] Instruksi untuk download BUSI dan reproduce hasil Phase 2
```

---

## Target Metrik Capaian untuk Q2

| Komponen | Target |
|----------|--------|
| Ranking inversion rate terukur | ≥ pada 1 proporsi Normal |
| Jumlah model dievaluasi | ≥ 5 |
| Dataset validasi | Internal + BUSI (eksternal) |
| Uji statistik | Wilcoxon, p < 0.05 terdokumentasi |
| Implementasi SEF | Open-source, terdokumentasi |

---

## Target Jurnal & Estimasi Timeline

| Jurnal | Quartile | Alasan |
|--------|----------|--------|
| *Biomedical Signal Processing and Control* | Q2 | Fokus metodologi medis, fit paper evaluasi |
| *Computers in Biology and Medicine* | Q1-Q2 | High visibility, dataset USG relevan |
| *Ultrasound in Medicine and Biology* | Q1-Q2 | Domain-specific, highest impact |

```
Bulan 1   : Phase 1 — Mathematical Analysis
Bulan 2-3 : Phase 2 — Simulation Study + Validasi BUSI
Bulan 4-5 : Phase 3 — Empirical Multi-Model Training
Bulan 5-6 : Phase 4 — SEF Implementation + Finalisasi
Bulan 7   : Writing + Internal Review
Bulan 8   : Submission
```

---

## Cara Menjalankan di Claude Code

```bash
# 1. Setup environment
pip install torch torchvision opencv-python albumentations \
            segmentation-models-pytorch monai torchmetrics \
            scipy scikit-learn pandas matplotlib seaborn

# 2. Jalankan Phase 1 terlebih dulu
python src/phase1_math_analysis.py

# 3. Jalankan simulasi (Phase 2)
python src/phase2_simulation.py

# 4. Training semua model (Phase 3) — paling lama
python src/phase3_empirical.py --models all --epochs 100

# 5. Evaluasi dengan ketiga protokol
python src/phase4_protocol.py --checkpoint results/checkpoints/

# 6. Generate semua figure untuk paper
python src/utils/visualizer.py --output paper/figures/
```
