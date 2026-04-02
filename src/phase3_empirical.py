"""
Phase 3 — Empirical Validation (Multi-Model)
Latih 5 model, evaluasi dengan 3 protokol, uji statistik.

Usage:
    python src/phase3_empirical.py                # train + eval all models
    python src/phase3_empirical.py --eval-only    # skip training, load checkpoints
    python src/phase3_empirical.py --model unet   # single model
    python src/phase3_empirical.py --epochs 50    # custom epoch count
"""

import argparse
import json
import time
import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy import stats

import segmentation_models_pytorch as smp

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.dataset import scan_busi_dataset, create_train_val_test_split, BUSIDataset
from src.metrics.sef_metrics import compare_protocols, compute_full_metrics

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

DATASET_ROOT  = Path("dataset")
CKPT_DIR      = Path("results/checkpoints")
OUT_DIR       = Path("experiments/phase3")
FIG_DIR       = Path("results/figures")
SPLIT_CSV     = Path("data_split.csv")

for d in [CKPT_DIR, OUT_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE      = torch.device("mps" if torch.backends.mps.is_available()
                            else "cuda" if torch.cuda.is_available()
                            else "cpu")
IMAGE_SIZE  = 256
BATCH_SIZE  = 16
LR          = 1e-4
WEIGHT_DECAY= 1e-5
SEED        = 42

MODELS_REGISTRY = {
    "unet":           dict(arch="Unet",         encoder="resnet34"),
    "attention_unet": dict(arch="Unet",         encoder="resnet34",   decoder_attention_type="scse"),
    "unetpp":         dict(arch="UnetPlusPlus",  encoder="resnet34"),
    "manet":          dict(arch="MAnet",         encoder="resnet34"),
    "linknet":        dict(arch="Linknet",       encoder="resnet34"),
}

print(f"Device: {DEVICE}")


# ──────────────────────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────────────────────

def build_model(name: str) -> nn.Module:
    cfg = MODELS_REGISTRY[name]
    kwargs = dict(
        encoder_name     = cfg["encoder"],
        encoder_weights  = "imagenet",
        in_channels      = 1,
        classes          = 1,
        activation       = None,
    )
    if "decoder_attention_type" in cfg:
        kwargs["decoder_attention_type"] = cfg["decoder_attention_type"]

    arch = cfg["arch"]
    model_cls = getattr(smp, arch)
    model = model_cls(**kwargs)
    return model.to(DEVICE)


# ──────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """Dice + BCE (standard for segmentation)."""
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight  = bce_weight
        self.dice_loss   = smp.losses.DiceLoss(mode="binary", smooth=1.0)
        self.bce_loss    = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        return (self.dice_weight * self.dice_loss(pred, target)
                + self.bce_weight * self.bce_loss(pred, target))


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler=None):
    model.train()
    total_loss = 0.0
    for batch in loader:
        imgs  = batch["image"].to(DEVICE)
        masks = batch["mask"].to(DEVICE)

        optimizer.zero_grad()
        if scaler:
            with torch.amp.autocast("cuda"):
                preds = model(imgs)
                loss  = criterion(preds, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(imgs)
            loss  = criterion(preds, masks)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader) -> dict:
    """Evaluate and return per-case predictions, GTs, labels."""
    model.eval()
    all_preds, all_gts, all_labels = [], [], []

    for batch in loader:
        imgs   = batch["image"].to(DEVICE)
        masks  = batch["mask"].numpy()
        labels = batch["label_str"]

        preds = torch.sigmoid(model(imgs)).cpu().numpy()

        for pred, gt, label in zip(preds, masks, labels):
            all_preds.append(pred[0])   # (H, W)
            all_gts.append(gt[0])       # (H, W)
            all_labels.append(label)

    return all_preds, all_gts, all_labels


def train_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 100,
    patience: int = 15,
) -> dict:
    """Train model, return best val metrics."""
    print(f"\n{'='*60}")
    print(f"Training: {model_name.upper()}")
    print(f"{'='*60}")

    model     = build_model(model_name)
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_dice  = -1
    patience_count = 0
    history        = []

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()

        # Val evaluation every 5 epochs
        if epoch % 5 == 0 or epoch == n_epochs:
            preds, gts, labels = evaluate(model, val_loader)
            metrics = compare_protocols(preds, gts, labels)
            val_dice = metrics["P2_audit_mean_dice"]   # use P2 (audit) as training signal

            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}/{n_epochs} | loss={train_loss:.4f} | "
                  f"P2_dice={val_dice:.4f} | P1_dice={metrics['P1_standard_mean_dice']:.4f} | "
                  f"SEF={metrics['P3_sef_composite']:.4f} | {elapsed:.1f}s")

            history.append({"epoch": epoch, "train_loss": train_loss, **metrics})

            if val_dice > best_val_dice:
                best_val_dice  = val_dice
                patience_count = 0
                ckpt_path = CKPT_DIR / f"{model_name}_best.pt"
                torch.save({
                    "epoch":       epoch,
                    "state_dict":  model.state_dict(),
                    "val_metrics": metrics,
                }, ckpt_path)
                print(f"    ✓ Checkpoint saved (val_dice={val_dice:.4f})")
            else:
                patience_count += 1
                if patience_count >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

    # Save training history
    pd.DataFrame(history).to_csv(OUT_DIR / f"{model_name}_history.csv", index=False)
    return history


# ──────────────────────────────────────────────────────────────
# Evaluation (load best checkpoint)
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model_all_protocols(
    model_name: str,
    test_loader: DataLoader,
) -> dict | None:
    """Load best checkpoint, evaluate with all 3 protocols."""
    ckpt_path = CKPT_DIR / f"{model_name}_best.pt"
    if not ckpt_path.exists():
        print(f"  [SKIP] {model_name}: checkpoint not found at {ckpt_path}")
        return None

    model = build_model(model_name)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    preds, gts, labels = evaluate(model, test_loader)
    metrics = compare_protocols(preds, gts, labels)

    # Also save per-case results for statistical testing
    per_case = []
    for pred, gt, label in zip(preds, gts, labels):
        m = compute_full_metrics(pred, gt)
        m["label"] = label
        m["model"] = model_name
        per_case.append(m)

    pd.DataFrame(per_case).to_csv(OUT_DIR / f"{model_name}_per_case.csv", index=False)
    print(f"  {model_name}: P1={metrics['P1_standard_mean_dice']:.4f} | "
          f"P2={metrics['P2_audit_mean_dice']:.4f} | SEF={metrics['P3_sef_composite']:.4f}")

    return {"model": model_name, "metrics": metrics, "per_case": per_case}


# ──────────────────────────────────────────────────────────────
# Statistical testing
# ──────────────────────────────────────────────────────────────

def wilcoxon_test(scores_a: list, scores_b: list) -> dict:
    """Wilcoxon signed-rank test per kasus."""
    if len(scores_a) != len(scores_b):
        return {"statistic": None, "p_value": None, "significant": None}
    stat, p = stats.wilcoxon(scores_a, scores_b, alternative="two-sided")
    return {"statistic": float(stat), "p_value": float(p), "significant": bool(p < 0.05)}


def report_with_ci(scores: list, confidence: float = 0.95) -> dict:
    """Bootstrap CI untuk mean Dice."""
    scores_arr = np.array(scores)
    n  = len(scores_arr)
    ci = stats.t.interval(confidence, df=n-1,
                          loc=np.mean(scores_arr),
                          scale=stats.sem(scores_arr))
    return {
        "mean":    float(np.mean(scores_arr)),
        "std":     float(np.std(scores_arr)),
        "ci_low":  float(ci[0]),
        "ci_high": float(ci[1]),
    }


def build_comparison_table(all_results: list) -> pd.DataFrame:
    """Build final comparison table: models × protocols."""
    rows = []
    for res in all_results:
        m  = res["metrics"]
        pc = res["per_case"]
        pos_dices = [r["dice"] for r in pc if r["label"] != "normal"]
        ci = report_with_ci(pos_dices)

        rows.append({
            "Model":           res["model"].replace("_", " ").title(),
            "P1_Standard":     f"{m['P1_standard_mean_dice']:.4f} ± {m['P1_standard_std']:.4f}",
            "P2_Audit":        f"{m['P2_audit_mean_dice']:.4f} ± {m['P2_audit_std']:.4f}",
            "SEF_Seg_Dice":    f"{m['P3_sef_seg_dice']:.4f} ± {m['P3_sef_seg_dice_std']:.4f}",
            "SEF_Specificity": f"{m['P3_sef_specificity']:.4f}",
            "SEF_Sensitivity": f"{m['P3_sef_sensitivity']:.4f}",
            "SEF_Composite":   f"{m['P3_sef_composite']:.4f}",
            "CI_95_low":       f"{ci['ci_low']:.4f}",
            "CI_95_high":      f"{ci['ci_high']:.4f}",
            "n_positive":      m["n_positive"],
            "n_negative":      m["n_negative"],
        })

    return pd.DataFrame(rows)


def compute_ranking_table(all_results: list) -> pd.DataFrame:
    """Compute model rankings under each protocol."""
    data = {r["model"]: r["metrics"] for r in all_results}

    protocols = {
        "P1_Standard":  lambda m: m["P1_standard_mean_dice"],
        "P2_Audit":     lambda m: m["P2_audit_mean_dice"],
        "SEF_Composite":lambda m: m["P3_sef_composite"],
    }

    rows = []
    for model_name, metrics in data.items():
        row = {"model": model_name}
        for proto_name, fn in protocols.items():
            row[proto_name] = fn(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    ranks = pd.DataFrame()
    ranks["model"] = df["model"]
    for proto_name in protocols:
        ranks[f"rank_{proto_name}"] = df[proto_name].rank(ascending=False).astype(int)

    # Count inversions
    ranks["rank_change_P1_vs_P2"]  = abs(ranks["rank_P1_Standard"] - ranks["rank_P2_Audit"])
    ranks["rank_change_P1_vs_SEF"] = abs(ranks["rank_P1_Standard"] - ranks["rank_SEF_Composite"])

    return ranks


# ──────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────

def plot_protocol_comparison(all_results: list, save_path: Path):
    """Grouped bar chart: all models × 3 protocols."""
    import matplotlib.pyplot as plt

    models  = [r["model"].replace("_", " ").title() for r in all_results]
    p1_vals = [r["metrics"]["P1_standard_mean_dice"] for r in all_results]
    p2_vals = [r["metrics"]["P2_audit_mean_dice"]    for r in all_results]
    sef_vals= [r["metrics"]["P3_sef_composite"]      for r in all_results]

    x     = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - width, p1_vals, width, label="P1 Standard (incl. Normal)",  color="#e74c3c", alpha=0.85)
    b2 = ax.bar(x,          p2_vals, width, label="P2 Audit (excl. Normal)",     color="#27ae60", alpha=0.85)
    b3 = ax.bar(x + width,  sef_vals,width, label="P3 SEF Composite",            color="#3498db", alpha=0.85)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Under 3 Evaluation Protocols\n"
                 "Illustrating Metric-Dependent Ranking Shifts", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate bars
    for bars in [b1, b2, b3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Figure saved: {save_path}")
    plt.close()


def plot_ranking_heatmap(ranks_df: pd.DataFrame, save_path: Path):
    """Heatmap of model rankings under different protocols."""
    import matplotlib.pyplot as plt

    rank_cols = [c for c in ranks_df.columns if c.startswith("rank_")]
    data = ranks_df.set_index("model")[rank_cols].T

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(data.values, cmap="RdYlGn_r", vmin=1, vmax=len(ranks_df))

    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels([c.replace("_", " ").title() for c in data.columns],
                       rotation=25, ha="right")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels([c.replace("rank_", "").replace("_", " ") for c in data.index])
    ax.set_title("Model Rankings by Protocol\n(Lower rank = better)", fontsize=11)

    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            ax.text(j, i, str(data.values[i, j]), ha="center", va="center",
                    fontsize=12, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Rank (1=best)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Figure saved: {save_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main(args):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("PHASE 3 — Empirical Validation (Multi-Model)")
    print(f"Device: {DEVICE} | Image size: {IMAGE_SIZE} | Batch: {BATCH_SIZE}")
    print("=" * 60)

    # ── Dataset
    if SPLIT_CSV.exists():
        df = pd.read_csv(SPLIT_CSV)
        print(f"  Loaded existing split from {SPLIT_CSV}")
    else:
        df = scan_busi_dataset(DATASET_ROOT)
        df = create_train_val_test_split(df, save_path=SPLIT_CSV, seed=SEED)

    train_ds = BUSIDataset(df, split="train", image_size=IMAGE_SIZE, augment=True)
    val_ds   = BUSIDataset(df, split="val",   image_size=IMAGE_SIZE, augment=False)
    test_ds  = BUSIDataset(df, split="test",  image_size=IMAGE_SIZE, augment=False)

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, generator=g)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Select models
    if args.model == "all":
        model_names = list(MODELS_REGISTRY.keys())
    elif args.model in MODELS_REGISTRY:
        model_names = [args.model]
    else:
        print(f"Unknown model: {args.model}. Options: {list(MODELS_REGISTRY.keys())} or 'all'")
        return

    # ── Training
    if not args.eval_only:
        for name in model_names:
            train_model(name, train_loader, val_loader, n_epochs=args.epochs)

    # ── Evaluation
    print("\n" + "=" * 60)
    print("EVALUATION — Test Set (3 Protocols)")
    print("=" * 60)

    all_results = []
    for name in model_names:
        res = evaluate_model_all_protocols(name, test_loader)
        if res:
            all_results.append(res)

    if not all_results:
        print("No results found. Run training first.")
        return

    # ── Summary tables
    comparison_table = build_comparison_table(all_results)
    ranks_table      = compute_ranking_table(all_results)

    print("\n── Comparison Table ──")
    print(comparison_table.to_string(index=False))
    comparison_table.to_csv(OUT_DIR / "comparison_table.csv", index=False)

    print("\n── Ranking Table ──")
    print(ranks_table.to_string(index=False))
    ranks_table.to_csv(OUT_DIR / "ranking_table.csv", index=False)

    # ── Statistical testing (pairwise, all models)
    print("\n── Wilcoxon Tests (P2 Dice, per-case) ──")
    stat_rows = []
    for i in range(len(all_results)):
        for j in range(i + 1, len(all_results)):
            a_name = all_results[i]["model"]
            b_name = all_results[j]["model"]
            a_pos  = [r["dice"] for r in all_results[i]["per_case"] if r["label"] != "normal"]
            b_pos  = [r["dice"] for r in all_results[j]["per_case"] if r["label"] != "normal"]
            test   = wilcoxon_test(a_pos[:min(len(a_pos), len(b_pos))],
                                   b_pos[:min(len(a_pos), len(b_pos))])
            row = {"model_a": a_name, "model_b": b_name, **test}
            stat_rows.append(row)
            sig = "✓" if test["significant"] else " "
            print(f"  {a_name} vs {b_name}: p={test['p_value']:.4f} {sig}")

    pd.DataFrame(stat_rows).to_csv(OUT_DIR / "wilcoxon_tests.csv", index=False)

    # ── Figures
    print("\n── Generating Figures ──")
    plot_protocol_comparison(all_results, FIG_DIR / "phase3_protocol_comparison.png")
    plot_ranking_heatmap(ranks_table, FIG_DIR / "phase3_ranking_heatmap.png")

    print(f"\nAll results saved to {OUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="all",
                        help="Model name or 'all'")
    parser.add_argument("--epochs",    type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, load existing checkpoints")
    args = parser.parse_args()
    main(args)
