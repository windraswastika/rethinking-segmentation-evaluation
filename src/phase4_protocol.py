"""
Phase 4 — SEF Validation & Final Protocol Comparison
Evaluasi dan visualisasi SEF vs P1/P2 pada semua model terlatih.

Usage:
    python src/phase4_protocol.py --checkpoint results/checkpoints/
    python src/phase4_protocol.py --output paper/figures/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.dataset import BUSIDataset, scan_busi_dataset, create_train_val_test_split
from src.metrics.sef_metrics import (
    compare_protocols,
    stratified_evaluation_framework,
    standard_dice_p1,
    audit_dice_p2,
    compute_full_metrics,
)

import segmentation_models_pytorch as smp

DATASET_ROOT = Path("dataset")
CKPT_DIR     = Path("results/checkpoints")
OUT_DIR      = Path("experiments/phase4")
FIG_DIR      = Path("results/figures")
PAPER_FIG    = Path("paper/figures")
SPLIT_CSV    = Path("data_split.csv")

for d in [OUT_DIR, FIG_DIR, PAPER_FIG]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE     = torch.device("mps" if torch.backends.mps.is_available()
                           else "cuda" if torch.cuda.is_available()
                           else "cpu")
IMAGE_SIZE = 256
BATCH_SIZE = 16

MODELS_REGISTRY = {
    "unet":           dict(arch="Unet",        encoder="resnet34"),
    "attention_unet": dict(arch="Unet",        encoder="resnet34",  decoder_attention_type="scse"),
    "unetpp":         dict(arch="UnetPlusPlus", encoder="resnet34"),
    "manet":          dict(arch="MAnet",        encoder="resnet34"),
    "linknet":        dict(arch="Linknet",      encoder="resnet34"),
}

PROTOCOL_LABELS = {
    "P1_Standard":  "P1: Standard\n(incl. Normal)",
    "P2_Audit":     "P2: Audit\n(excl. Normal)",
    "SEF_Seg":      "P3 SEF:\nSeg Dice",
    "SEF_Specific": "P3 SEF:\nSpecificity",
    "SEF_Composite":"P3 SEF:\nComposite",
}

plt.rcParams.update({"font.size": 11, "figure.dpi": 100})


# ──────────────────────────────────────────────────────────────
# Load and evaluate
# ──────────────────────────────────────────────────────────────

def build_model(name: str):
    cfg    = MODELS_REGISTRY[name]
    kwargs = dict(encoder_name=cfg["encoder"], encoder_weights=None,
                  in_channels=1, classes=1, activation=None)
    if "decoder_attention_type" in cfg:
        kwargs["decoder_attention_type"] = cfg["decoder_attention_type"]
    arch = cfg["arch"]
    return getattr(smp, arch)(**kwargs).to(DEVICE)


@torch.no_grad()
def load_and_eval(model_name: str, test_loader: DataLoader) -> dict | None:
    ckpt_path = CKPT_DIR / f"{model_name}_best.pt"
    if not ckpt_path.exists():
        return None

    model = build_model(model_name)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    all_preds, all_gts, all_labels = [], [], []
    for batch in test_loader:
        imgs   = batch["image"].to(DEVICE)
        masks  = batch["mask"].numpy()
        labels = batch["label_str"]
        preds  = torch.sigmoid(model(imgs)).cpu().numpy()
        for p, g, l in zip(preds, masks, labels):
            all_preds.append(p[0])
            all_gts.append(g[0])
            all_labels.append(l)

    metrics = compare_protocols(all_preds, all_gts, all_labels)
    sef     = stratified_evaluation_framework(all_preds, all_gts, all_labels)

    per_case = []
    for p, g, l in zip(all_preds, all_gts, all_labels):
        m = compute_full_metrics(p, g)
        m["label"] = l
        per_case.append(m)

    return {
        "model":     model_name,
        "metrics":   metrics,
        "sef":       sef,
        "per_case":  per_case,
    }


# ──────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────

def plot_sef_breakdown(all_results: list, save_path: Path):
    """
    Figure: SEF two-pathway decomposition for each model.
    Shows seg dice AND specificity side by side.
    """
    models = [r["model"].replace("_", " ").title() for r in all_results]
    seg    = [r["sef"].segmentation_dice     for r in all_results]
    spec   = [r["sef"].detection_specificity for r in all_results]
    comp   = [r["sef"].composite_score       for r in all_results]

    x = np.arange(len(models))
    w = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w,   seg,  w, label="Segmentation Dice (positive cases)", color="#3498db", alpha=0.85)
    ax.bar(x,       spec, w, label="Detection Specificity (Normal cases)", color="#e67e22", alpha=0.85)
    ax.bar(x + w,   comp, w, label="SEF Composite Score", color="#9b59b6", alpha=0.85)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("SEF Two-Pathway Decomposition\n"
                 "Segmentation Quality vs Detection Specificity",
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate
    for xi, (s, sp, c) in enumerate(zip(seg, spec, comp)):
        ax.text(xi - w, s + 0.01, f"{s:.3f}", ha="center", fontsize=8)
        ax.text(xi,    sp + 0.01, f"{sp:.3f}", ha="center", fontsize=8)
        ax.text(xi + w, c + 0.01, f"{c:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Figure: {save_path}")
    plt.close()


def plot_protocol_scatter(all_results: list, save_path: Path):
    """
    Scatter plot: P1 Dice vs P2 Dice per model.
    Diagonal = no difference; off-diagonal = protocol effect.
    """
    p1 = [r["metrics"]["P1_standard_mean_dice"] for r in all_results]
    p2 = [r["metrics"]["P2_audit_mean_dice"]    for r in all_results]
    names = [r["model"].replace("_", " ").title() for r in all_results]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="P1 = P2 line")

    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    for i, (x, y, name) in enumerate(zip(p1, p2, names)):
        ax.scatter(x, y, color=colors[i], s=120, zorder=5)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(8, 4), fontsize=9)

    ax.set_xlabel("P1 Standard (incl. Normal)", fontsize=12)
    ax.set_ylabel("P2 Audit (excl. Normal)", fontsize=12)
    ax.set_title("P1 vs P2 Dice Score per Model\n"
                 "(Points below diagonal = P1 underestimates true seg quality)",
                 fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Figure: {save_path}")
    plt.close()


def plot_per_case_dice_distribution(all_results: list, save_path: Path):
    """
    Violin plot: distribution of per-case Dice per model,
    split by positive vs negative cases.
    """
    records = []
    for res in all_results:
        for r in res["per_case"]:
            records.append({
                "Model":   res["model"].replace("_", " ").title(),
                "Dice":    r["dice"],
                "Type":    "Normal" if r["label"] == "normal" else "Positive",
                "label":   r["label"],
            })

    df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Per-Case Dice Distribution by Model and Case Type",
                 fontsize=12, fontweight="bold")

    for ax, case_type in zip(axes, ["Positive", "Normal"]):
        subset = df[df["Type"] == case_type]
        if subset.empty:
            ax.text(0.5, 0.5, f"No {case_type} cases", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        sns.violinplot(data=subset, x="Model", y="Dice", ax=ax,
                       palette="Set2", inner="quart")
        ax.set_title(f"{case_type} Cases (n={len(subset)})", fontsize=11)
        ax.set_xlabel("Model"); ax.set_ylabel("Dice Score")
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, alpha=0.3, axis="y")

        if case_type == "Normal":
            ax.axhline(0, color="red", linestyle="--", alpha=0.5,
                       label="Dice=0 (FP on empty mask)")
            ax.axhline(1, color="green", linestyle="--", alpha=0.5,
                       label="Dice=1 (correct silence)")
            ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Figure: {save_path}")
    plt.close()


def plot_sef_clinical_argument(save_path: Path):
    """
    Conceptual figure: show how P1 conflates two different clinical tasks.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Why Stratified Evaluation Aligns Better with Clinical Workflow",
                 fontsize=12, fontweight="bold")

    # ── Panel 1: Standard P1 collapse
    ax = axes[0]
    ax.set_title("P1: Standard Protocol\n(Single mean Dice)", fontsize=10)
    ax.barh(["Benign", "Malignant", "Normal"], [0.80, 0.72, 0.0],
            color=["#27ae60", "#2980b9", "#e74c3c"])
    ax.axvline(np.mean([0.80, 0.72, 0.0]), color="black", linestyle="--",
               label=f"Mean = {np.mean([0.80, 0.72, 0.0]):.2f}")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Dice Score")
    ax.legend(fontsize=9)
    ax.text(0.5, -0.15, "PROBLEM: Normal case penalizes good segmenter",
            transform=ax.transAxes, ha="center", color="red", fontsize=9)

    # ── Panel 2: P2 Audit
    ax2 = axes[1]
    ax2.set_title("P2: Audit Protocol\n(Exclude Normal)", fontsize=10)
    ax2.barh(["Benign", "Malignant"], [0.80, 0.72],
             color=["#27ae60", "#2980b9"])
    ax2.axvline(np.mean([0.80, 0.72]), color="black", linestyle="--",
                label=f"Mean = {np.mean([0.80, 0.72]):.2f}")
    ax2.set_xlim(0, 1.05)
    ax2.set_xlabel("Dice Score")
    ax2.legend(fontsize=9)
    ax2.text(0.5, -0.15, "WARNING: Ignores Normal — misses FP risk",
             transform=ax2.transAxes, ha="center", color="orange", fontsize=9)

    # ── Panel 3: SEF decomposed
    ax3 = axes[2]
    ax3.set_title("P3: SEF Protocol\n(Two-pathway evaluation)", fontsize=10)
    categories = ["Seg Dice\n(Benign)", "Seg Dice\n(Malignant)", "Specificity\n(Normal)"]
    values     = [0.80, 0.72, 0.85]
    colors     = ["#27ae60", "#2980b9", "#9b59b6"]
    ax3.barh(categories, values, color=colors)
    composite = 0.67 * np.mean([0.80, 0.72]) + 0.33 * 0.85
    ax3.axvline(composite, color="black", linestyle="--",
                label=f"Composite = {composite:.2f}")
    ax3.set_xlim(0, 1.05)
    ax3.set_xlabel("Score")
    ax3.legend(fontsize=9)
    ax3.text(0.5, -0.15, "SEF: Separates segmentation from detection",
             transform=ax3.transAxes, ha="center", color="green", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Figure: {save_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────
# Build final paper table
# ──────────────────────────────────────────────────────────────

def build_final_paper_table(all_results: list) -> pd.DataFrame:
    """
    Table siap untuk paper: Model × [P1, P2, SEF_Seg, SEF_Spec, SEF_Comp]
    dengan ranking di setiap kolom.
    """
    rows = []
    for res in all_results:
        m   = res["metrics"]
        sef = res["sef"]
        rows.append({
            "Model":            res["model"].replace("_", " ").title(),
            "P1_Mean_Dice":     round(m["P1_standard_mean_dice"], 4),
            "P2_Audit_Dice":    round(m["P2_audit_mean_dice"],    4),
            "SEF_Seg_Dice":     round(sef.segmentation_dice,      4),
            "SEF_Specificity":  round(sef.detection_specificity,  4),
            "SEF_Sensitivity":  round(sef.detection_sensitivity,  4),
            "SEF_Composite":    round(sef.composite_score,        4),
            "N_Positive":       sef.n_positive_cases,
            "N_Negative":       sef.n_negative_cases,
        })

    df = pd.DataFrame(rows)

    # Add rankings
    for col in ["P1_Mean_Dice", "P2_Audit_Dice", "SEF_Composite"]:
        df[f"Rank_{col}"] = df[col].rank(ascending=False).astype(int)

    return df.sort_values("P2_Audit_Dice", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main(args):
    print("=" * 60)
    print("PHASE 4 — SEF Validation & Protocol Analysis")
    print("=" * 60)

    # ── Dataset
    if SPLIT_CSV.exists():
        df = pd.read_csv(SPLIT_CSV)
    else:
        df = scan_busi_dataset(DATASET_ROOT)
        df = create_train_val_test_split(df, save_path=SPLIT_CSV)

    test_ds     = BUSIDataset(df, split="test", image_size=IMAGE_SIZE, augment=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  Test set: {len(test_ds)} samples")

    # ── Load and evaluate all models
    all_results = []
    for name in MODELS_REGISTRY:
        res = load_and_eval(name, test_loader)
        if res:
            all_results.append(res)
            print(f"  {name}: P1={res['metrics']['P1_standard_mean_dice']:.4f} | "
                  f"P2={res['metrics']['P2_audit_mean_dice']:.4f} | "
                  f"SEF_seg={res['sef'].segmentation_dice:.4f} | "
                  f"SEF_spec={res['sef'].detection_specificity:.4f}")

    if not all_results:
        print("\n  No checkpoints found. Run Phase 3 training first:")
        print("  python src/phase3_empirical.py --epochs 100")
        return

    # ── Final table
    paper_table = build_final_paper_table(all_results)
    print("\n── Final Paper Table ──")
    print(paper_table.to_string(index=False))
    paper_table.to_csv(OUT_DIR / "final_paper_table.csv", index=False)
    paper_table.to_csv(PAPER_FIG / "table1_main_results.csv", index=False)

    # ── Figures
    print("\n── Generating Paper Figures ──")
    out = Path(args.output)
    plot_sef_breakdown(all_results,             out / "fig_sef_breakdown.png")
    plot_protocol_scatter(all_results,          out / "fig_protocol_scatter.png")
    plot_per_case_dice_distribution(all_results, out / "fig_dice_distribution.png")
    plot_sef_clinical_argument(                  out / "fig_sef_clinical_argument.png")

    # ── Key findings
    if len(all_results) >= 2:
        print("\n── Ranking Inversion Analysis ──")
        models_sorted_p1 = sorted(all_results, key=lambda r: r["metrics"]["P1_standard_mean_dice"], reverse=True)
        models_sorted_p2 = sorted(all_results, key=lambda r: r["metrics"]["P2_audit_mean_dice"],    reverse=True)

        inversions = sum(
            models_sorted_p1[i]["model"] != models_sorted_p2[i]["model"]
            for i in range(len(all_results))
        )
        print(f"  {inversions}/{len(all_results)} models change rank between P1 and P2")
        print(f"  P1 top model: {models_sorted_p1[0]['model']}")
        print(f"  P2 top model: {models_sorted_p2[0]['model']}")

    print(f"\nAll Phase 4 outputs saved to {OUT_DIR}/ and {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/checkpoints/",
                        help="Checkpoint directory")
    parser.add_argument("--output",     default="paper/figures/",
                        help="Output directory for paper figures")
    args = parser.parse_args()
    main(args)
