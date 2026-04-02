"""
Stratified Evaluation Framework (SEF) — Kontribusi utama paper.
Memisahkan evaluasi segmentasi dan deteksi.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from typing import Optional


@dataclass
class SEFResult:
    segmentation_dice: float         # Dice hanya pada kasus positif
    detection_specificity: float     # TNR hanya pada kasus Normal (model "diam")
    detection_sensitivity: float     # TPR — model mendeteksi ada lesi
    composite_score: float           # gabungan berbobot
    n_positive_cases: int
    n_negative_cases: int
    segmentation_dice_std: float = 0.0
    detection_specificity_std: float = 0.0

    def __str__(self) -> str:
        return (
            f"SEFResult(\n"
            f"  segmentation_dice    = {self.segmentation_dice:.4f} ± {self.segmentation_dice_std:.4f}  (n={self.n_positive_cases})\n"
            f"  detection_specificity= {self.detection_specificity:.4f} ± {self.detection_specificity_std:.4f}  (n={self.n_negative_cases})\n"
            f"  detection_sensitivity= {self.detection_sensitivity:.4f}\n"
            f"  composite_score      = {self.composite_score:.4f}\n"
            f")"
        )

    def to_dict(self) -> dict:
        return {
            "seg_dice":            self.segmentation_dice,
            "seg_dice_std":        self.segmentation_dice_std,
            "det_specificity":     self.detection_specificity,
            "det_specificity_std": self.detection_specificity_std,
            "det_sensitivity":     self.detection_sensitivity,
            "composite":           self.composite_score,
            "n_positive":          self.n_positive_cases,
            "n_negative":          self.n_negative_cases,
        }


def stratified_evaluation_framework(
    predictions: list,
    ground_truths: list,
    labels: list,
    prevalence_negative: float = 0.171,  # 133/780 = proporsi Normal aktual di BUSI
                                          # Sesuaikan ke prevalensi klinis lokal jika berbeda
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> SEFResult:
    """
    Stratified Evaluation Framework (SEF).

    Memisahkan evaluasi menjadi dua jalur:
    - Jalur Segmentasi: evaluasi kualitas mask pada kasus positif (benign/malignant)
    - Jalur Deteksi: evaluasi kemampuan model untuk diam pada kasus Normal

    Args:
        predictions:          list of np.ndarray (raw logits or probabilities)
        ground_truths:        list of np.ndarray (binary masks)
        labels:               list of str: "normal" / "benign" / "malignant"
        prevalence_negative:  bobot Normal dalam composite score
        threshold:            binarization threshold
        smooth:               smoothing factor untuk Dice numerator/denominator

    Returns:
        SEFResult
    """
    positive_labels = {"benign", "malignant"}
    seg_dices, det_tn, det_tp = [], [], []

    for pred, gt, label in zip(predictions, ground_truths, labels):
        pred_bin = (pred > threshold).astype(np.float32)

        if label in positive_labels:
            # ── Jalur Segmentasi: Dice standar
            intersection = (pred_bin * gt).sum()
            dice = (2 * intersection + smooth) / (pred_bin.sum() + gt.sum() + smooth)
            seg_dices.append(float(dice))
            # Deteksi sensitivity: apakah model menghasilkan prediksi non-kosong?
            det_tp.append(1.0 if pred_bin.sum() > 0 else 0.0)

        else:
            # ── Jalur Deteksi: apakah model benar-benar diam? (TNR)
            det_tn.append(1.0 if pred_bin.sum() == 0 else 0.0)

    seg_dice    = float(np.mean(seg_dices)) if seg_dices else 0.0
    seg_dice_std = float(np.std(seg_dices)) if len(seg_dices) > 1 else 0.0
    specificity  = float(np.mean(det_tn))  if det_tn  else 1.0
    spec_std     = float(np.std(det_tn))   if len(det_tn) > 1 else 0.0
    sensitivity  = float(np.mean(det_tp))  if det_tp  else 0.0

    # Composite score: w_seg × Dice_seg + w_det × Specificity
    w_seg     = 1 - prevalence_negative
    composite = w_seg * seg_dice + prevalence_negative * specificity

    return SEFResult(
        segmentation_dice      = seg_dice,
        segmentation_dice_std  = seg_dice_std,
        detection_specificity  = specificity,
        detection_specificity_std = spec_std,
        detection_sensitivity  = sensitivity,
        composite_score        = composite,
        n_positive_cases       = len(seg_dices),
        n_negative_cases       = len(det_tn),
    )


def standard_dice_p1(predictions, ground_truths, smooth=1e-6) -> dict:
    """
    P1 — Standard protocol: mean Dice semua kasus (termasuk Normal).
    Ini adalah protokol yang paling umum digunakan.
    """
    dices = []
    for pred, gt in zip(predictions, ground_truths):
        pred_bin = (pred > 0.5).astype(np.float32)
        inter = (pred_bin * gt).sum()
        d = float((2 * inter + smooth) / (pred_bin.sum() + gt.sum() + smooth))
        dices.append(d)
    return {
        "mean_dice":  float(np.mean(dices)),
        "std_dice":   float(np.std(dices)),
        "n_cases":    len(dices),
    }


def audit_dice_p2(predictions, ground_truths, labels, smooth=1e-6) -> dict:
    """
    P2 — Audit protocol: Dice hanya pada kasus positif (benign + malignant).
    Mengeksklusi Normal dari perhitungan.
    """
    positive_labels = {"benign", "malignant"}
    dices = []
    for pred, gt, label in zip(predictions, ground_truths, labels):
        if label not in positive_labels:
            continue
        pred_bin = (pred > 0.5).astype(np.float32)
        inter    = (pred_bin * gt).sum()
        d        = float((2 * inter + smooth) / (pred_bin.sum() + gt.sum() + smooth))
        dices.append(d)
    return {
        "mean_dice": float(np.mean(dices)) if dices else 0.0,
        "std_dice":  float(np.std(dices))  if dices else 0.0,
        "n_cases":   len(dices),
    }


def compare_protocols(predictions, ground_truths, labels) -> dict:
    """
    Bandingkan P1 (Standard), P2 (Audit), P3 (SEF) pada set prediksi yang sama.
    Returns dict dengan semua metrik.
    """
    p1 = standard_dice_p1(predictions, ground_truths)
    p2 = audit_dice_p2(predictions, ground_truths, labels)
    p3 = stratified_evaluation_framework(predictions, ground_truths, labels)

    return {
        "P1_standard_mean_dice":    p1["mean_dice"],
        "P1_standard_std":          p1["std_dice"],
        "P2_audit_mean_dice":       p2["mean_dice"],
        "P2_audit_std":             p2["std_dice"],
        "P3_sef_seg_dice":          p3.segmentation_dice,
        "P3_sef_seg_dice_std":      p3.segmentation_dice_std,
        "P3_sef_specificity":       p3.detection_specificity,
        "P3_sef_sensitivity":       p3.detection_sensitivity,
        "P3_sef_composite":         p3.composite_score,
        "n_positive":               p2["n_cases"],
        "n_negative":               p3.n_negative_cases,
    }


# ──────────────────────────────────────────────────────────────
# Additional metrics: IoU, Precision, Recall, Hausdorff
# ──────────────────────────────────────────────────────────────

def iou_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    pred_bin  = (pred > 0.5).astype(np.float32)
    inter     = (pred_bin * gt).sum()
    union     = pred_bin.sum() + gt.sum() - inter
    return float((inter + smooth) / (union + smooth))


def precision_recall(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6):
    pred_bin = (pred > 0.5).astype(np.float32)
    tp = (pred_bin * gt).sum()
    fp = pred_bin.sum() - tp
    fn = gt.sum() - tp
    precision = float((tp + smooth) / (tp + fp + smooth))
    recall    = float((tp + smooth) / (tp + fn + smooth))
    return precision, recall


def compute_full_metrics(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> dict:
    """Hitung semua metrik standar sekaligus."""
    pred_bin  = (pred > 0.5).astype(np.float32)
    inter     = (pred_bin * gt).sum()
    tp        = inter
    fp        = pred_bin.sum() - tp
    fn        = gt.sum() - tp
    tn        = (1 - pred_bin) * (1 - gt)
    tn_sum    = tn.sum()

    dice      = float((2 * tp + smooth) / (pred_bin.sum() + gt.sum() + smooth))
    iou       = float((tp + smooth) / (tp + fp + fn + smooth))
    precision = float((tp + smooth) / (tp + fp + smooth))
    recall    = float((tp + smooth) / (tp + fn + smooth))
    specificity = float((tn_sum + smooth) / (tn_sum + fp + smooth))

    return {
        "dice":        dice,
        "iou":         iou,
        "precision":   precision,
        "recall":      recall,
        "specificity": specificity,
        "tp":          float(tp),
        "fp":          float(fp),
        "fn":          float(fn),
        "tn":          float(tn_sum),
    }
