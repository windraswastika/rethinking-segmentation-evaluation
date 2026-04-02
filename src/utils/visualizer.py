"""
Visualizer — Generate paper-ready figures.

Usage:
    python src/utils/visualizer.py --output paper/figures/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.dataset import BUSIDataset
from src.metrics.sef_metrics import compute_full_metrics

DATASET_ROOT = Path("dataset")
CKPT_DIR     = Path("results/checkpoints")
SPLIT_CSV    = Path("data_split.csv")
FIG_DIR      = Path("results/figures")
PAPER_FIG    = Path("paper/figures")


# ──────────────────────────────────────────────────────────────
# Case study panel: 6 examples illustrating the problem
# ──────────────────────────────────────────────────────────────

def plot_qualitative_cases(
    predictions: list,
    images: list,
    ground_truths: list,
    labels: list,
    filenames: list,
    save_path: Path,
    n_cases: int = 6,
):
    """
    6-panel case study figure:
    Row 1: 2 benign + 1 malignant (positive, correct)
    Row 2: 1 normal (TN, Dice=1.0), 1 normal (FP, Dice=0.0), 1 positive (FP)
    Each cell: image | GT mask | pred | Dice value
    """
    # Select representative cases
    pos_indices = [i for i, l in enumerate(labels) if l != "normal"]
    neg_indices = [i for i, l in enumerate(labels) if l == "normal"]

    # Sort by interesting metrics
    from src.metrics.sef_metrics import compute_full_metrics

    pos_cases = []
    for i in pos_indices:
        m = compute_full_metrics(predictions[i], ground_truths[i])
        pos_cases.append((i, m["dice"]))
    pos_cases.sort(key=lambda x: x[1], reverse=True)

    neg_cases = []
    for i in neg_indices:
        pred_sum = (predictions[i] > 0.5).sum()
        neg_cases.append((i, pred_sum))

    # Select: 3 good positive, 1 normal-TN, 1 normal-FP
    selected = []
    if pos_cases:
        # High dice positive
        selected.extend([pos_cases[0][0], pos_cases[len(pos_cases)//2][0]])
    if len(pos_cases) > 1:
        selected.append(pos_cases[-1][0])  # lowest dice positive

    tn_cases = [i for i, s in neg_cases if s == 0]   # true negatives
    fp_cases = [i for i, s in neg_cases if s > 0]    # false positives

    if tn_cases: selected.append(tn_cases[0])
    if fp_cases: selected.append(fp_cases[0])

    # Limit to n_cases
    selected = selected[:n_cases]
    if not selected:
        print("  [visualizer] No cases to plot")
        return

    n_cols = len(selected)
    fig = plt.figure(figsize=(4 * n_cols, 12))
    fig.suptitle("Qualitative Case Study\n"
                 "Illustrating Dice Behavior on Positive and Normal (Negative) Cases",
                 fontsize=12, fontweight="bold")

    for col, idx in enumerate(selected):
        img   = images[idx]
        gt    = ground_truths[idx]
        pred  = predictions[idx]
        label = labels[idx]
        fname = filenames[idx] if filenames else str(idx)
        pred_bin = (pred > 0.5).astype(np.float32)

        metrics = compute_full_metrics(pred, gt)
        dice    = metrics["dice"]
        fp_pixels = int(metrics["fp"])

        # Row 1: Image
        ax_img = fig.add_subplot(3, n_cols, col + 1)
        ax_img.imshow(img, cmap="gray")
        ax_img.set_title(f"{label.title()}\n{Path(fname).stem}", fontsize=8)
        ax_img.axis("off")

        # Row 2: GT mask overlay
        ax_gt = fig.add_subplot(3, n_cols, n_cols + col + 1)
        overlay = np.stack([img / img.max()] * 3, axis=-1) if img.max() > 0 else np.stack([img] * 3, axis=-1)
        overlay = (overlay * 255).astype(np.uint8)
        if gt.sum() > 0:
            overlay[gt > 0.5] = [0, 255, 0]   # green = GT
        ax_gt.imshow(overlay)
        ax_gt.set_title("Ground Truth", fontsize=8)
        ax_gt.axis("off")

        # Row 3: Prediction overlay + Dice
        ax_pred = fig.add_subplot(3, n_cols, 2 * n_cols + col + 1)
        overlay2 = np.stack([img / img.max()] * 3, axis=-1) if img.max() > 0 else np.stack([img] * 3, axis=-1)
        overlay2 = (overlay2 * 255).astype(np.uint8)
        if pred_bin.sum() > 0:
            overlay2[pred_bin > 0.5] = [255, 100, 100]   # red = prediction

        dice_color = "green" if dice > 0.7 else ("orange" if dice > 0.3 else "red")
        note = ""
        if label == "normal" and fp_pixels > 0:
            note = f"\n⚠ FP: {fp_pixels}px → Dice=0"
        elif label == "normal" and fp_pixels == 0:
            note = f"\nTN: Model correct"

        ax_pred.imshow(overlay2)
        ax_pred.set_title(f"Pred  Dice={dice:.3f}{note}", fontsize=8, color=dice_color)
        ax_pred.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Figure: {save_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────
# Training curves
# ──────────────────────────────────────────────────────────────

def plot_training_curves(model_name: str, save_path: Path = None):
    """Load training history CSV and plot loss + Dice curves."""
    history_path = Path("experiments/phase3") / f"{model_name}_history.csv"
    if not history_path.exists():
        print(f"  [skip] No history for {model_name}")
        return

    df = pd.read_csv(history_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Training History: {model_name.replace('_',' ').title()}", fontsize=12)

    # Loss
    axes[0].plot(df["epoch"], df["train_loss"], "b-o", markersize=4, label="Train Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Dice protocols
    for col, label, color in [
        ("P1_standard_mean_dice", "P1 Standard", "#e74c3c"),
        ("P2_audit_mean_dice",    "P2 Audit",    "#27ae60"),
        ("P3_sef_composite",      "P3 SEF",      "#3498db"),
    ]:
        if col in df.columns:
            axes[1].plot(df["epoch"], df[col], "o-", markersize=4, color=color, label=label)

    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score")
    axes[1].set_title("Validation: 3 Protocols"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure: {save_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────
# All figures entrypoint
# ──────────────────────────────────────────────────────────────

def generate_all_figures(output_dir: Path = Path("paper/figures")):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1 + 2 figures already generated by their scripts
    import shutil
    for src_fig in Path("results/figures").glob("*.png"):
        shutil.copy(src_fig, output_dir / src_fig.name)
        print(f"  Copied: {src_fig.name} → {output_dir}/")

    # Training curves per model
    for model_name in ["unet", "attention_unet", "unetpp", "manet", "linknet"]:
        plot_training_curves(model_name, output_dir / f"training_{model_name}.png")

    print(f"\nAll figures exported to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="paper/figures/", help="Output directory")
    args = parser.parse_args()
    generate_all_figures(Path(args.output))
