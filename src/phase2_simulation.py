"""
Phase 2 — Simulation Study
Kuantifikasi efek distorsi Dice sebagai fungsi proporsi kasus negatif.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from itertools import product

OUT_DIR = Path("experiments/phase2")
FIG_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

NORMAL_PROPORTIONS = [0.0, 0.05, 0.10, 0.15, 0.17, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
MODEL_FP_RATES     = [0.1, 0.2, 0.3, 0.5, 0.8]
N_TOTAL_CASES      = 780
N_SIMULATION_RUNS  = 200
SEED               = 42


# ──────────────────────────────────────────────────────────────
# Core simulation
# ──────────────────────────────────────────────────────────────

def simulate_metric_distortion(
    normal_proportion: float,
    model_a_true_dice: float,
    model_b_true_dice: float,
    model_a_fp_rate: float,
    model_b_fp_rate: float,
    n_total: int = N_TOTAL_CASES,
    n_runs: int = N_SIMULATION_RUNS,
    seed: int = SEED,
) -> dict:
    """
    Simulasikan efek memasukkan kasus Normal terhadap reported mean Dice.
    Ukur: apakah ranking model A vs B berubah?
    """
    rng = np.random.default_rng(seed)
    n_normal   = int(n_total * normal_proportion)
    n_positive = n_total - n_normal

    ranking_inversions = 0
    reported_dices_a, reported_dices_b = [], []

    for _ in range(n_runs):
        # Dice pada kasus positif — sample dari distribusi normal, clip ke [0,1]
        dices_pos_a = rng.normal(model_a_true_dice, 0.05, n_positive).clip(0, 1)
        dices_pos_b = rng.normal(model_b_true_dice, 0.05, n_positive).clip(0, 1)

        # Dice pada kasus Normal: 0.0 jika ada FP, 1.0 jika benar-benar clean
        fp_mask_a   = rng.binomial(1, model_a_fp_rate, n_normal)
        fp_mask_b   = rng.binomial(1, model_b_fp_rate, n_normal)
        dices_neg_a = np.where(fp_mask_a, 0.0, 1.0)
        dices_neg_b = np.where(fp_mask_b, 0.0, 1.0)

        if n_normal > 0:
            mean_a = np.mean(np.concatenate([dices_pos_a, dices_neg_a]))
            mean_b = np.mean(np.concatenate([dices_pos_b, dices_neg_b]))
        else:
            mean_a = np.mean(dices_pos_a)
            mean_b = np.mean(dices_pos_b)

        reported_dices_a.append(mean_a)
        reported_dices_b.append(mean_b)

        # Cek ranking inversion
        true_winner   = "A" if model_a_true_dice > model_b_true_dice else "B"
        report_winner = "A" if mean_a > mean_b else "B"
        if true_winner != report_winner:
            ranking_inversions += 1

    return {
        "normal_proportion":       normal_proportion,
        "n_normal":                n_normal,
        "n_positive":              n_positive,
        "ranking_inversion_rate":  ranking_inversions / n_runs,
        "mean_reported_dice_a":    float(np.mean(reported_dices_a)),
        "mean_reported_dice_b":    float(np.mean(reported_dices_b)),
        "ci_95_a_low":             float(np.percentile(reported_dices_a, 2.5)),
        "ci_95_a_high":            float(np.percentile(reported_dices_a, 97.5)),
        "ci_95_b_low":             float(np.percentile(reported_dices_b, 2.5)),
        "ci_95_b_high":            float(np.percentile(reported_dices_b, 97.5)),
        "model_a_true_dice":       model_a_true_dice,
        "model_b_true_dice":       model_b_true_dice,
        "model_a_fp_rate":         model_a_fp_rate,
        "model_b_fp_rate":         model_b_fp_rate,
    }


# ──────────────────────────────────────────────────────────────
# Full grid simulation
# ──────────────────────────────────────────────────────────────

def run_full_simulation() -> pd.DataFrame:
    """Jalankan simulasi utama (Efficient-U vs U-Net, nilai empiris)."""
    print("[2.1] Main Simulation: Efficient-U (A) vs Baseline U-Net (B)")
    print("-" * 60)

    # Anchor values dari dataset internal
    model_a = {"true_dice": 0.7482, "fp_rate": 0.8}   # Efficient-U (worse on Normal)
    model_b = {"true_dice": 0.5887, "fp_rate": 0.3}   # Baseline U-Net (better on Normal)

    results = []
    for prop in NORMAL_PROPORTIONS:
        res = simulate_metric_distortion(
            normal_proportion   = prop,
            model_a_true_dice   = model_a["true_dice"],
            model_b_true_dice   = model_b["true_dice"],
            model_a_fp_rate     = model_a["fp_rate"],
            model_b_fp_rate     = model_b["fp_rate"],
        )
        results.append(res)
        inversion_flag = " ← INVERSION!" if res["ranking_inversion_rate"] > 0 else ""
        print(
            f"  Normal={prop:.0%} | "
            f"A={res['mean_reported_dice_a']:.4f} vs B={res['mean_reported_dice_b']:.4f} | "
            f"Inversion rate: {res['ranking_inversion_rate']:.1%}{inversion_flag}"
        )

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "simulation_main.csv", index=False)
    print(f"\n  Saved: {OUT_DIR}/simulation_main.csv")
    return df


def run_fp_rate_grid() -> pd.DataFrame:
    """Grid simulasi: berbagai FP rate × proporsi Normal (untuk Figure 2)."""
    print("\n[2.2] FP Rate Grid Simulation")
    print("-" * 60)

    results = []
    # Model A selalu lebih baik secara segmentasi; variasikan FP rate A
    model_b = {"true_dice": 0.5887, "fp_rate": 0.2}

    for fp_a, prop in product(MODEL_FP_RATES, NORMAL_PROPORTIONS):
        res = simulate_metric_distortion(
            normal_proportion   = prop,
            model_a_true_dice   = 0.7482,
            model_b_true_dice   = model_b["true_dice"],
            model_a_fp_rate     = fp_a,
            model_b_fp_rate     = model_b["fp_rate"],
        )
        results.append(res)

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "simulation_grid.csv", index=False)
    print(f"  Grid size: {len(df)} scenarios | Saved: {OUT_DIR}/simulation_grid.csv")
    return df


# ──────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────

def plot_main_simulation(df: pd.DataFrame, save_path: Path):
    """
    Figure utama (killer figure):
    Reported Dice vs True Dice sebagai fungsi proporsi Normal
    + Ranking Inversion Rate
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Dice Metric Distortion as a Function of Negative Case Proportion\n"
        "(Efficient-U vs Baseline U-Net, BUSI-scale dataset, n=780)",
        fontsize=13, fontweight="bold"
    )

    props_pct = df["normal_proportion"] * 100

    # ── Left: Reported Dice trajectories
    ax = axes[0]
    ax.axhline(0.7482, color="#27ae60", linestyle="--", linewidth=2, label="True Dice A (Efficient-U)")
    ax.axhline(0.5887, color="#2980b9", linestyle="--", linewidth=2, label="True Dice B (U-Net)")

    ax.plot(props_pct, df["mean_reported_dice_a"], "o-", color="#e74c3c", linewidth=2,
            markersize=7, label="Reported Dice A (incl. Normal)")
    ax.plot(props_pct, df["mean_reported_dice_b"], "s-", color="#3498db", linewidth=2,
            markersize=7, label="Reported Dice B (incl. Normal)")

    # CI shading
    ax.fill_between(props_pct, df["ci_95_a_low"], df["ci_95_a_high"],
                    alpha=0.15, color="#e74c3c")
    ax.fill_between(props_pct, df["ci_95_b_low"], df["ci_95_b_high"],
                    alpha=0.15, color="#3498db")

    ax.axvline(17, color="gray", linestyle=":", alpha=0.7,
               label="BUSI Normal proportion (~17%)")

    ax.set_xlabel("Proportion of Normal (Negative) Cases (%)", fontsize=12)
    ax.set_ylabel("Mean Dice Score", fontsize=12)
    ax.set_title("Reported Dice Diverges from True Dice\nas Normal Proportion Increases", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    # ── Right: Ranking Inversion Rate
    ax2 = axes[1]
    colors = ["#27ae60" if r == 0 else "#e74c3c" for r in df["ranking_inversion_rate"]]
    bars = ax2.bar(props_pct, df["ranking_inversion_rate"] * 100, color=colors, width=2.5,
                   edgecolor="white", linewidth=0.5)

    ax2.axhline(50, color="gray", linestyle="--", alpha=0.5, label="50% inversion (random)")
    ax2.axvline(17, color="gray", linestyle=":", alpha=0.7,
                label="BUSI Normal proportion (~17%)")

    # Annotate bars with values
    for bar, val in zip(bars, df["ranking_inversion_rate"] * 100):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 1,
                     f"{val:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2.set_xlabel("Proportion of Normal (Negative) Cases (%)", fontsize=12)
    ax2.set_ylabel("Ranking Inversion Rate (%)", fontsize=12)
    ax2.set_title("Model Ranking Inversion Rate\nvs Proportion of Normal Cases", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 110)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Figure saved: {save_path}")
    plt.close()


def plot_fp_rate_heatmap(df_grid: pd.DataFrame, save_path: Path):
    """Heatmap: Ranking Inversion Rate sebagai fungsi FP rate × proporsi Normal."""
    # Pivot table: rows=fp_rate_A, cols=normal_proportion
    pivot = df_grid.pivot_table(
        values="ranking_inversion_rate",
        index="model_a_fp_rate",
        columns="normal_proportion",
        aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    import matplotlib.cm as cm
    im = ax.imshow(pivot.values * 100, aspect="auto", cmap="RdYlGn_r",
                   vmin=0, vmax=100)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.0%}" for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r:.0%}" for r in pivot.index])
    ax.set_xlabel("Proportion of Normal Cases", fontsize=12)
    ax.set_ylabel("FP Rate of Model A (Efficient-U) on Normal Cases", fontsize=12)
    ax.set_title("Ranking Inversion Rate (%)\n"
                 "(Model A = Efficient-U, True Dice=0.748; Model B = U-Net, True Dice=0.589)",
                 fontsize=11)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j] * 100
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if val > 50 else "black")

    plt.colorbar(im, ax=ax, label="Inversion Rate (%)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Figure saved: {save_path}")
    plt.close()


def plot_distortion_curves_by_fp(df_grid: pd.DataFrame, save_path: Path):
    """Figure: Kurva distorsi Dice untuk berbagai FP rates."""
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(MODEL_FP_RATES)))
    for color, fp_rate in zip(colors, MODEL_FP_RATES):
        subset = df_grid[df_grid["model_a_fp_rate"] == fp_rate]
        props  = subset["normal_proportion"] * 100
        dices  = subset["mean_reported_dice_a"]
        ax.plot(props, dices, "o-", color=color, linewidth=2, markersize=6,
                label=f"FP rate = {fp_rate:.0%}")

    ax.axhline(0.7482, color="green", linestyle="--", linewidth=2,
               label="True Dice (Efficient-U) = 0.748")
    ax.axhline(0.5887, color="blue", linestyle=":", linewidth=2,
               label="True Dice (U-Net) = 0.589")
    ax.axvline(17, color="gray", linestyle=":", alpha=0.7,
               label="BUSI Normal proportion (~17%)")

    ax.set_xlabel("Proportion of Normal (Negative) Cases (%)", fontsize=12)
    ax.set_ylabel("Reported Mean Dice", fontsize=12)
    ax.set_title("Dice Distortion for Model A (Efficient-U)\nby False-Positive Rate on Normal Cases",
                 fontsize=11)
    ax.legend(fontsize=10, loc="lower left")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Figure saved: {save_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────

def print_summary_table(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("SIMULATION SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Normal%':>8} {'N_neg':>6} {'Dice_A':>8} {'Dice_B':>8} "
          f"{'Winner':>8} {'Inversion%':>12} {'Note'}")
    print("-" * 80)

    true_winner = "A" if 0.7482 > 0.5887 else "B"

    for _, row in df.iterrows():
        reported_winner = "A" if row["mean_reported_dice_a"] > row["mean_reported_dice_b"] else "B"
        inversion = reported_winner != true_winner
        note = "← INVERSION" if inversion else ""
        print(
            f"  {row['normal_proportion']:>5.0%}   "
            f"{row['n_normal']:>5.0f}   "
            f"{row['mean_reported_dice_a']:>7.4f}  "
            f"{row['mean_reported_dice_b']:>7.4f}  "
            f"{'Model '+reported_winner:>8}  "
            f"{row['ranking_inversion_rate']:>10.1%}  "
            f"{note}"
        )
    print()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PHASE 2 — Simulation Study")
    print("=" * 60)

    # ── Main simulation
    df_main = run_full_simulation()
    print_summary_table(df_main)

    # ── Grid simulation
    df_grid = run_fp_rate_grid()

    # ── Figures
    print("\n[2.3] Generating Figures")
    print("-" * 50)
    plot_main_simulation(df_main, FIG_DIR / "phase2_main_simulation.png")
    plot_fp_rate_heatmap(df_grid, FIG_DIR / "phase2_inversion_heatmap.png")
    plot_distortion_curves_by_fp(df_grid, FIG_DIR / "phase2_distortion_curves.png")

    # ── Key finding
    inversion_threshold = df_main[df_main["ranking_inversion_rate"] > 0]["normal_proportion"].min()
    print(f"\n[KEY FINDING] Ranking inversion first occurs at {inversion_threshold:.0%} Normal proportion")
    print(f"  Dataset BUSI has ~{133/780:.0%} Normal → check if inversion occurs")


if __name__ == "__main__":
    main()
