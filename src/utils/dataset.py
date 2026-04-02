"""
Dataset utilities — disesuaikan untuk struktur BUSI:
    dataset/
    ├── benign/    ← image + mask  (format: "benign (N).png" / "benign (N)_mask.png")
    ├── normal/    ← image + mask
    └── malignant/ ← image + mask
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ──────────────────────────────────────────────────────────────
# Dataset scanning
# ──────────────────────────────────────────────────────────────

def scan_busi_dataset(root_dir: str | Path) -> pd.DataFrame:
    """
    Scan folder BUSI dan kembalikan DataFrame dengan kolom:
    filename, mask_filename, class, split_class_dir
    """
    root = Path(root_dir)
    records = []

    for cls in ["benign", "malignant", "normal"]:
        cls_dir = root / cls
        if not cls_dir.exists():
            print(f"  [WARNING] Folder {cls_dir} tidak ditemukan, skip.")
            continue

        # Cari semua file image (bukan mask)
        for img_path in sorted(cls_dir.glob("*.png")):
            if "_mask" in img_path.name:
                continue

            # Cari mask yang sesuai (bisa ada >1 mask per gambar di BUSI)
            stem = img_path.stem   # e.g. "benign (1)"
            masks = sorted(cls_dir.glob(f"{stem}_mask*.png"))

            if not masks:
                print(f"  [WARNING] Mask tidak ditemukan untuk {img_path.name}, skip.")
                continue

            # Jika ada multiple masks (BUSI malignant kadang punya 2), gabung
            records.append({
                "filename":       img_path.name,
                "filepath":       str(img_path),
                "mask_filename":  masks[0].name,
                "mask_filepath":  str(masks[0]),
                "extra_masks":    [str(m) for m in masks[1:]],
                "class":          cls,
                "is_negative":    cls == "normal",
            })

    df = pd.DataFrame(records)
    print(f"Dataset scan result: {len(df)} images")
    print(df["class"].value_counts().to_string())
    return df


def create_labels_csv(root_dir: str | Path, out_path: str | Path = None) -> pd.DataFrame:
    """Buat labels.csv dari scan dataset. Simpan jika out_path diberikan."""
    df = scan_busi_dataset(root_dir)
    if out_path:
        df.to_csv(out_path, index=False)
        print(f"  Saved labels.csv → {out_path}")
    return df


def create_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    test_ratio:  float = 0.15,
    seed: int = 42,
    stratify: bool = True,
    save_path: str | Path = None,
) -> pd.DataFrame:
    """
    Buat split stratified per kelas.
    Returns df dengan kolom 'split' (train/val/test).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    rng = np.random.default_rng(seed)
    splits = []

    classes = df["class"].unique() if stratify else ["all"]
    for cls in classes:
        subset = df[df["class"] == cls] if stratify else df
        idx    = subset.index.tolist()
        rng.shuffle(idx)

        n      = len(idx)
        n_tr   = int(n * train_ratio)
        n_val  = int(n * val_ratio)

        for i in idx[:n_tr]:           splits.append((i, "train"))
        for i in idx[n_tr:n_tr+n_val]: splits.append((i, "val"))
        for i in idx[n_tr+n_val:]:     splits.append((i, "test"))

    split_df = pd.DataFrame(splits, columns=["index", "split"]).set_index("index")
    df = df.copy()
    df["split"] = split_df["split"]

    print("\nSplit distribution:")
    print(df.groupby(["split", "class"]).size().unstack(fill_value=0).to_string())

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"\n  Saved split CSV → {save_path}")

    return df


# ──────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────

class BUSIDataset(Dataset):
    """
    PyTorch Dataset untuk BUSI (Breast Ultrasound Images).
    Supports albumentations augmentations.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        split: str = "train",
        image_size: int = 256,
        augment: bool = True,
    ):
        self.df    = df[df["split"] == split].reset_index(drop=True)
        self.split = split
        self.image_size = image_size
        self.transform = self._build_transform(augment and split == "train")

        # Label encoding
        self.label_map = {"normal": 0, "benign": 1, "malignant": 2}

    def _build_transform(self, augment: bool) -> A.Compose:
        base = [
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=(0.485,), std=(0.229,)),
            ToTensorV2(),
        ]
        if augment:
            aug = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                                   rotate_limit=15, p=0.4),
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                           contrast_limit=0.2, p=0.4),
                A.GaussNoise(p=0.2),
                A.ElasticTransform(alpha=30, sigma=5, p=0.2),
            ]
            return A.Compose(aug + base)
        return A.Compose(base)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # Load grayscale USG image
        img  = cv2.imread(row["filepath"], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(row["mask_filepath"], cv2.IMREAD_GRAYSCALE)

        # Binarize mask
        mask = (mask > 127).astype(np.uint8)

        # Parse extra_masks (may be string repr when loaded from CSV)
        extra_masks = row.get("extra_masks", [])
        if isinstance(extra_masks, str):
            import ast
            try:
                extra_masks = ast.literal_eval(extra_masks)
            except Exception:
                extra_masks = []

        # Handle multiple masks (merge)
        for extra_mask_path in extra_masks:
            if extra_mask_path:
                em = cv2.imread(extra_mask_path, cv2.IMREAD_GRAYSCALE)
                if em is not None:
                    mask = np.logical_or(mask, em > 127).astype(np.uint8)

        augmented = self.transform(image=img, mask=mask)
        image     = augmented["image"]   # (1, H, W) after ToTensorV2
        mask_t    = augmented["mask"].float().unsqueeze(0)  # (1, H, W)

        return {
            "image":      image,
            "mask":       mask_t,
            "label":      self.label_map[row["class"]],
            "label_str":  row["class"],
            "filename":   row["filename"],
            "is_negative": row["is_negative"],
        }


def build_dataloaders(
    root_dir: str | Path,
    image_size: int = 256,
    batch_size: int = 16,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame]:
    """
    Scan dataset, buat split, return (train_loader, val_loader, test_loader, df_split).
    """
    root = Path(root_dir)
    split_csv = root.parent / "data_split.csv"

    if split_csv.exists():
        df = pd.read_csv(split_csv)
        print(f"  Loaded existing split from {split_csv}")
    else:
        df = scan_busi_dataset(root)
        df = create_train_val_test_split(df, save_path=split_csv, seed=seed)

    train_ds = BUSIDataset(df, split="train", image_size=image_size, augment=True)
    val_ds   = BUSIDataset(df, split="val",   image_size=image_size, augment=False)
    test_ds  = BUSIDataset(df, split="test",  image_size=image_size, augment=False)

    print(f"\nDataset sizes — train:{len(train_ds)} val:{len(val_ds)} test:{len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, df
