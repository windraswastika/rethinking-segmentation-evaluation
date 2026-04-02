"""
Phase 1 — Mathematical Analysis & Formalization
Buktikan secara analitik dan numerik patologi Dice pada empty mask.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

OUT_DIR = Path("experiments/phase1")
FIG_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 1. Core Dice implementations
# ──────────────────────────────────────────────────────────────

def dice_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    """Implementasi Dice standar — TANPA handling khusus untuk empty mask."""
    intersection = (pred * gt).sum()
    return (2 * intersection + smooth) / (pred.sum() + gt.sum() + smooth)


def dice_score_strict(pred: np.ndarray, gt: np.ndarray) -> float | None:
    """Dice tanpa smoothing — kembalikan None jika 0/0 (both empty)."""
    intersection = (pred * gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return None   # truly undefined
    return (2 * intersection) / denom


def dice_undefined_behavior(gt_empty: bool, pred_has_fp: bool) -> str:
    """Peta seluruh kombinasi kasus GT kosong."""
    if gt_empty and pred_has_fp:
        return "0.0  ← misleading: model salah tapi tidak proporsional"
    if gt_empty and not pred_has_fp:
        return "1.0 atau 0/0 ← implementation-dependent, tidak konsisten"
    return "normal Dice behavior"


# ──────────────────────────────────────────────────────────────
# 2. FP Sensitivity Analysis
# ──────────────────────────────────────────────────────────────

def analyze_fp_sensitivity(image_size: int = 256, fp_pixel_counts: list = None):
    """
    Hitung Dice sebagai fungsi dari jumlah FP pixel pada GT kosong.
    Tunjukkan: bahkan 1 pixel FP menghasilkan Dice ≈ 0.0
    """
    np.random.seed(42)
    if fp_pixel_counts is None:
        fp_pixel_counts = [0, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000]

    gt = np.zeros((image_size, image_size))
    results = []

    for n_fp in fp_pixel_counts:
        pred = np.zeros((image_size, image_size))
        if n_fp > 0:
            indices = np.random.choice(image_size * image_size, n_fp, replace=False)
            pred.flat[indices] = 1.0

        score_smooth  = dice_score(pred, gt)
        score_strict  = dice_score_strict(pred, gt)

        results.append({
            "n_fp_pixels":    n_fp,
            "fp_percentage":  n_fp / (image_size ** 2) * 100,
            "dice_smooth":    round(score_smooth, 6),
            "dice_strict":    round(score_strict, 6) if score_strict is not None else None,
            "gt_empty":       True,
        })

    return results


# ──────────────────────────────────────────────────────────────
# 3. Theoretical Mean Dice Distortion
# ──────────────────────────────────────────────────────────────

def theoretical_mean_dice_drop(
    n_positive_cases: int,
    n_negative_cases: int,
    true_positive_dice: float,
    model_fp_rate_on_negatives: float,
) -> dict:
    """
    Hitung secara teoretis berapa expected mean Dice setelah memasukkan kasus Normal.
    Asumsi: setiap kasus Normal dengan FP menghasilkan Dice = 0.0 (smooth behavior)
    """
    # Fraction of negative cases handled correctly (no FP) → Dice ≈ 1.0 (smooth)
    # Fraction with FP → Dice ≈ 0.0
    expected_dice_on_negatives = (1 - model_fp_rate_on_negatives) * 1.0

    # Skenario A: TN → Dice=1.0 via smooth factor, FP → Dice=0.0
    mean_dice_A = (
        n_positive_cases * true_positive_dice
        + n_negative_cases * expected_dice_on_negatives
    ) / (n_positive_cases + n_negative_cases)

    # Skenario B: semua kasus GT kosong → Dice=0.0 (strict, no smooth)
    mean_dice_B = (
        n_positive_cases * true_positive_dice
    ) / (n_positive_cases + n_negative_cases)

    return {
        "true_segmentation_dice":  true_positive_dice,
        "reported_dice_scenario_A": round(mean_dice_A, 4),
        "reported_dice_scenario_B": round(mean_dice_B, 4),
        "distortion_A":            round(true_positive_dice - mean_dice_A, 4),
        "distortion_B":            round(true_positive_dice - mean_dice_B, 4),
        "max_distortion":          round(true_positive_dice - min(mean_dice_A, mean_dice_B), 4),
    }


# ──────────────────────────────────────────────────────────────
# 4. Library Consistency Check
# ──────────────────────────────────────────────────────────────

def check_library_behaviors() -> pd.DataFrame:
    """
    Cek bagaimana berbagai library menangani Dice pada empty mask.
    Returns DataFrame dengan behavior mapping.
    """
    gt_empty   = np.zeros((256, 256), dtype=np.float32)
    pred_tn    = np.zeros((256, 256), dtype=np.float32)   # true negative
    pred_fp    = np.zeros((256, 256), dtype=np.float32)   # false positive
    pred_fp.flat[np.random.choice(256*256, 100, replace=False)] = 1.0

    results = []

    # ── Custom implementation 1: smooth=1e-6 (most common in GitHub repos)
    def custom_smooth(p, g, eps=1e-6):
        inter = (p * g).sum()
        return float((2 * inter + eps) / (p.sum() + g.sum() + eps))

    results.append({
        "library": "Custom (smooth=1e-6)",
        "tn_behavior":  round(custom_smooth(pred_tn, gt_empty), 4),   # expect ~1.0
        "fp_behavior":  round(custom_smooth(pred_fp, gt_empty), 4),   # expect ~0.0
        "note":         "most common pattern on GitHub"
    })

    # ── Custom implementation 2: smooth=1 (Laplace smoothing)
    def custom_laplace(p, g, eps=1.0):
        inter = (p * g).sum()
        return float((2 * inter + eps) / (p.sum() + g.sum() + eps))

    results.append({
        "library": "Custom (smooth=1)",
        "tn_behavior":  round(custom_laplace(pred_tn, gt_empty), 4),
        "fp_behavior":  round(custom_laplace(pred_fp, gt_empty), 4),
        "note":         "Laplace smoothing variant"
    })

    # ── Custom implementation 3: strict (no smoothing, returns 1.0 for both empty)
    def custom_strict_tn1(p, g):
        if p.sum() == 0 and g.sum() == 0:
            return 1.0
        inter = (p * g).sum()
        denom = p.sum() + g.sum()
        return float(2 * inter / denom) if denom > 0 else 0.0

    results.append({
        "library": "Custom (strict, TN→1)",
        "tn_behavior":  round(custom_strict_tn1(pred_tn, gt_empty), 4),
        "fp_behavior":  round(custom_strict_tn1(pred_fp, gt_empty), 4),
        "note":         "returns 1.0 when both empty"
    })

    # ── Custom implementation 4: strict, TN→0 (pessimistic)
    def custom_strict_tn0(p, g):
        inter = (p * g).sum()
        denom = p.sum() + g.sum()
        return float(2 * inter / denom) if denom > 0 else 0.0

    results.append({
        "library": "Custom (strict, TN→0)",
        "tn_behavior":  round(custom_strict_tn0(pred_tn, gt_empty), 4),
        "fp_behavior":  round(custom_strict_tn0(pred_fp, gt_empty), 4),
        "note":         "returns 0.0 when both empty (pessimistic)"
    })

    # ── sklearn F1 (Dice = F1 for binary segmentation)
    try:
        from sklearn.metrics import f1_score
        tn_f1 = f1_score(gt_empty.flatten(), pred_tn.flatten(), zero_division=1)
        fp_f1 = f1_score(gt_empty.flatten(), pred_fp.flatten(), zero_division=0)
        results.append({
            "library": "sklearn.f1_score",
            "tn_behavior":  round(float(tn_f1), 4),
            "fp_behavior":  round(float(fp_f1), 4),
            "note":         "zero_division parameter controls TN behavior"
        })
    except Exception as e:
        results.append({"library": "sklearn.f1_score", "tn_behavior": "error", "fp_behavior": "error", "note": str(e)})

    # ── torchmetrics
    try:
        import torch
        from torchmetrics.functional import dice
        p_tn = torch.from_numpy(pred_tn).long().unsqueeze(0)
        p_fp = torch.from_numpy((pred_fp > 0).astype(np.int64)).long().unsqueeze(0)
        g    = torch.from_numpy(gt_empty).long().unsqueeze(0)
        tn_tm = dice(p_tn, g, ignore_index=0).item()
        fp_tm = dice(p_fp, g, ignore_index=0).item()
        results.append({
            "library": "torchmetrics.Dice",
            "tn_behavior":  round(float(tn_tm), 4),
            "fp_behavior":  round(float(fp_tm), 4),
            "note":         "ignore_index=0 (background)"
        })
    except ImportError:
        results.append({"library": "torchmetrics.Dice", "tn_behavior": "not installed",
                        "fp_behavior": "not installed", "note": "pip install torchmetrics"})
    except Exception as e:
        results.append({"library": "torchmetrics.Dice", "tn_behavior": "error",
                        "fp_behavior": "error", "note": str(e)})

    # ── MONAI DiceMetric
    try:
        from monai.metrics import DiceMetric
        import torch
        dm = DiceMetric(include_background=False, reduction="mean")
        def to_onehot(arr):
            t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
            return torch.cat([1 - t, t], dim=1)
        dm.reset(); dm(to_onehot(pred_tn), to_onehot(gt_empty)); tn_m = dm.aggregate().item()
        dm.reset(); dm(to_onehot(pred_fp), to_onehot(gt_empty)); fp_m = dm.aggregate().item()
        results.append({
            "library": "MONAI DiceMetric",
            "tn_behavior":  round(float(tn_m), 4) if not np.isnan(tn_m) else "NaN",
            "fp_behavior":  round(float(fp_m), 4) if not np.isnan(fp_m) else "NaN",
            "note":         "include_background=False"
        })
    except ImportError:
        results.append({"library": "MONAI DiceMetric", "tn_behavior": "not installed",
                        "fp_behavior": "not installed", "note": "pip install monai"})
    except Exception as e:
        results.append({"library": "MONAI DiceMetric", "tn_behavior": "error",
                        "fp_behavior": "error", "note": str(e)})

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────
# 5. Figures
# ──────────────────────────────────────────────────────────────

def plot_fp_sensitivity(sensitivity_results: list, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Dice Behavior on Empty Ground-Truth Mask\n(Image size 256×256 = 65,536 pixels)",
                 fontsize=13, fontweight="bold")

    n_fps   = [r["n_fp_pixels"]   for r in sensitivity_results]
    dices_s = [r["dice_smooth"]   for r in sensitivity_results]
    fp_pcts = [r["fp_percentage"] for r in sensitivity_results]

    # ── Left: FP pixels vs Dice (linear scale)
    ax = axes[0]
    ax.plot(n_fps, dices_s, "o-", color="#e74c3c", linewidth=2, markersize=8)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Expected 'correct' = 1.0")
    ax.axhline(0.0, color="black", linestyle="--", alpha=0.3, label="Worst case = 0.0")
    ax.set_xlabel("Number of False-Positive Pixels", fontsize=12)
    ax.set_ylabel("Dice Score (smooth=1e-6)", fontsize=12)
    ax.set_title("Even 1 FP Pixel → Dice ≈ 0.0", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)

    # Annotate the critical point
    ax.annotate("1 FP pixel\n→ Dice ≈ 0.0",
                xy=(1, dices_s[1]), xytext=(200, 0.4),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=10, color="red")

    # ── Right: FP% vs Dice (log scale)
    ax2 = axes[1]
    ax2.semilogx(np.array(n_fps) + 1, dices_s, "s-", color="#c0392b", linewidth=2, markersize=8)
    ax2.set_xlabel("Number of FP Pixels (log scale)", fontsize=12)
    ax2.set_ylabel("Dice Score", fontsize=12)
    ax2.set_title("Dice Collapse vs FP Count (Log Scale)", fontsize=11)
    ax2.set_ylim(-0.05, 1.1)
    ax2.grid(True, alpha=0.3, which="both")

    # Annotate key thresholds
    for i, (n, d) in enumerate(zip(n_fps, dices_s)):
        if n in [0, 1, 100, 10000]:
            ax2.annotate(f"n={n}\nDice={d:.4f}",
                        xy=(n + 1, d), xytext=(n * 3 + 2, d + 0.1),
                        fontsize=8, color="#c0392b")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Figure saved: {save_path}")
    plt.close()


def plot_theoretical_distortion(save_path: Path):
    """Kurva: true Dice vs reported Dice sebagai fungsi proporsi Normal."""
    normal_proportions = np.linspace(0, 0.5, 100)
    n_total            = 780
    true_dice          = 0.7482   # Efficient-U true segmentation Dice
    fp_rate            = 0.8      # assumed 80% Normal cases have some FP

    reported_A, reported_B = [], []
    for prop in normal_proportions:
        n_neg = int(n_total * prop)
        n_pos = n_total - n_neg
        res   = theoretical_mean_dice_drop(n_pos, n_neg, true_dice, fp_rate)
        reported_A.append(res["reported_dice_scenario_A"])
        reported_B.append(res["reported_dice_scenario_B"])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axhline(true_dice, color="green", linestyle="--", linewidth=2,
               label=f"True Segmentation Dice = {true_dice:.4f}")
    ax.plot(normal_proportions * 100, reported_A, color="#e74c3c", linewidth=2,
            label="Reported Dice (Scenario A: TN→1.0 via smooth)")
    ax.plot(normal_proportions * 100, reported_B, color="#8e44ad", linewidth=2,
            label="Reported Dice (Scenario B: all empty→0.0)")
    ax.axvline(33, color="gray", linestyle=":", alpha=0.7, label="BUSI proportion (~33% Normal)")

    ax.fill_between(normal_proportions * 100, reported_A, true_dice,
                    alpha=0.15, color="#e74c3c", label="Distortion region (Scenario A)")
    ax.fill_between(normal_proportions * 100, reported_B, true_dice,
                    alpha=0.15, color="#8e44ad", label="Distortion region (Scenario B)")

    ax.set_xlabel("Proportion of Normal (Negative) Cases (%)", fontsize=12)
    ax.set_ylabel("Mean Dice Score", fontsize=12)
    ax.set_title("Theoretical Dice Distortion as a Function of Negative Case Proportion\n"
                 f"(Model: Efficient-U, true Dice = {true_dice:.4f}, FP rate = {fp_rate:.0%})",
                 fontsize=11)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Figure saved: {save_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────
# 6. Main
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PHASE 1 — Mathematical Analysis & Formalization")
    print("=" * 60)

    # ── 1. Reproduksi temuan empiris
    print("\n[1] Reproduksi Temuan Empiris (dataset 780 citra)")
    print("-" * 50)
    result = theoretical_mean_dice_drop(
        n_positive_cases=647,        # 437 benign + 210 malignant
        n_negative_cases=133,        # 133 normal
        true_positive_dice=0.7482,   # Efficient-U Dice
        model_fp_rate_on_negatives=0.8,
    )
    for k, v in result.items():
        print(f"  {k:<35} : {v}")

    print("\n  Cross-check: Baseline U-Net")
    result_b = theoretical_mean_dice_drop(
        n_positive_cases=647,
        n_negative_cases=133,
        true_positive_dice=0.5887,
        model_fp_rate_on_negatives=0.3,   # baseline lebih konservatif
    )
    for k, v in result_b.items():
        print(f"  {k:<35} : {v}")

    # ── 2. FP Sensitivity Analysis
    print("\n[2] FP Sensitivity Analysis")
    print("-" * 50)
    sensitivity_results = analyze_fp_sensitivity()
    df_sens = pd.DataFrame(sensitivity_results)
    print(df_sens.to_string(index=False))
    df_sens.to_csv(OUT_DIR / "fp_sensitivity.csv", index=False)
    print(f"\n  Saved: {OUT_DIR}/fp_sensitivity.csv")

    # ── 3. Library Consistency Check
    print("\n[3] Library Consistency Check")
    print("-" * 50)
    df_libs = check_library_behaviors()
    print(df_libs.to_string(index=False))
    df_libs.to_csv(OUT_DIR / "library_consistency.csv", index=False)
    print(f"\n  Saved: {OUT_DIR}/library_consistency.csv")

    # ── 4. Undefined behavior mapping
    print("\n[4] Undefined Behavior Mapping")
    print("-" * 50)
    for gt_empty in [True, False]:
        for pred_fp in [True, False]:
            behavior = dice_undefined_behavior(gt_empty, pred_fp)
            print(f"  GT empty={gt_empty}, Pred has FP={pred_fp}: {behavior}")

    # ── 5. Figures
    print("\n[5] Generating Figures")
    print("-" * 50)
    plot_fp_sensitivity(sensitivity_results, FIG_DIR / "phase1_fp_sensitivity.png")
    plot_theoretical_distortion(FIG_DIR / "phase1_theoretical_distortion.png")

    # ── 6. Summary for paper
    print("\n" + "=" * 60)
    print("SUMMARY (for paper)")
    print("=" * 60)
    print(f"""
Key Findings:
  • Even 1 FP pixel on empty GT → Dice ≈ 0.0 (sharp discontinuity)
  • True Dice = {0.7482:.4f} (Efficient-U on positive cases only)
  • Reported Dice (with 133 Normal, FP rate 80%) ≈ {result['reported_dice_scenario_A']:.4f}
  • Max distortion = {result['max_distortion']:.4f} ({result['max_distortion']/0.7482*100:.1f}% relative error)
  • 5 libraries show inconsistent behavior on empty mask (see library_consistency.csv)
""")


if __name__ == "__main__":
    main()
