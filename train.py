"""
EcoSort AI - Training Pipeline
================================
Transfer-learning classifier for 12-class waste image dataset.

Usage:
    python train.py                     # full training run
    python train.py --smoke-test        # dry-run: imports + split counts only
    python train.py --epochs 5          # quick test with fewer epochs

Outputs (written to models/):
    waste_classifier.pt   - best checkpoint by validation accuracy
    class_names.txt        - newline-delimited ordered class list
    training_metrics.json  - per-epoch loss/accuracy history
    evaluation_report.json - test-set metrics (accuracy, F1, per-class stats)
    confusion_matrix.png   - visual confusion matrix
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

# Force UTF-8 stdout/stderr on Windows so unicode chars in print() don't crash
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
import matplotlib
matplotlib.use("Agg")          # non-interactive backend -- safe for training scripts
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_ROOT = Path("data/garbage_classification")
MODELS_DIR   = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

MODEL_SAVE_PATH    = MODELS_DIR / "waste_classifier.pt"
CLASS_NAMES_PATH   = MODELS_DIR / "class_names.txt"
METRICS_PATH       = MODELS_DIR / "training_metrics.json"
EVAL_REPORT_PATH   = MODELS_DIR / "evaluation_report.json"
CONFUSION_MAT_PATH = MODELS_DIR / "confusion_matrix.png"

# ---------------------------------------------------------------------------
# Hyper-parameters (defaults -- override via CLI flags)
# ---------------------------------------------------------------------------
SEED         = 42
IMG_SIZE     = 224
BATCH_SIZE   = 32
NUM_WORKERS  = 4          # reduce to 0 if multiprocessing causes issues on Windows
LEARNING_RATE= 1e-4
NUM_EPOCHS   = 30
PATIENCE     = 5          # early-stopping patience (epochs without val improvement)
TRAIN_FRAC   = 0.70
VAL_FRAC     = 0.15
# TEST_FRAC  = 0.15       # implicit: 1 - TRAIN_FRAC - VAL_FRAC

# ImageNet normalisation -- required for EfficientNet pretrained weights
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random; random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# 1. Discover dataset -- build flat lists of (path, label_index)
# ---------------------------------------------------------------------------
def discover_dataset(dataset_root: Path) -> tuple[list[str], list[int], list[str]]:
    """
    Walk dataset_root, collect every image file, and assign integer labels.

    Returns
    -------
    image_paths : list[str]
    labels      : list[int]   parallel to image_paths
    class_names : list[str]   sorted class name list; index == label integer
    """
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_root}")

    class_names = sorted(
        d.name for d in dataset_root.iterdir() if d.is_dir()
    )
    if not class_names:
        raise ValueError(f"No class subdirectories found in {dataset_root}")

    image_paths: list[str] = []
    labels: list[int]      = []

    for idx, cls in enumerate(class_names):
        cls_dir = dataset_root / cls
        for fpath in sorted(cls_dir.iterdir()):
            if fpath.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                image_paths.append(str(fpath))
                labels.append(idx)

    print(f"[Dataset] {len(class_names)} classes | {len(image_paths)} images total")
    counts = Counter(labels)
    for idx, cls in enumerate(class_names):
        print(f"  {cls:<15} {counts[idx]:>5}")

    return image_paths, labels, class_names


# ---------------------------------------------------------------------------
# 2. Stratified 70 / 15 / 15 split -- no data leakage
# ---------------------------------------------------------------------------
def stratified_split(
    image_paths: list[str],
    labels: list[int],
    train_frac: float = TRAIN_FRAC,
    val_frac: float   = VAL_FRAC,
    seed: int         = SEED,
) -> tuple[list, list, list, list, list, list]:
    """
    Split indices into train / val / test with stratification.
    No file is duplicated or omitted.

    Returns six lists: train_paths, train_labels,
                       val_paths,   val_labels,
                       test_paths,  test_labels
    """
    test_frac = 1.0 - train_frac - val_frac

    # First split: train vs (val + test)
    (train_paths, temp_paths,
     train_labels, temp_labels) = train_test_split(
        image_paths, labels,
        test_size=(1.0 - train_frac),
        stratify=labels,
        random_state=seed,
    )

    # Second split: val vs test from the temporary pool
    relative_val = val_frac / (val_frac + test_frac)
    (val_paths, test_paths,
     val_labels, test_labels) = train_test_split(
        temp_paths, temp_labels,
        test_size=(1.0 - relative_val),
        stratify=temp_labels,
        random_state=seed,
    )

    return (train_paths, train_labels,
            val_paths,   val_labels,
            test_paths,  test_labels)


# ---------------------------------------------------------------------------
# 3. PyTorch Dataset
# ---------------------------------------------------------------------------
class WasteDataset(Dataset):
    """
    Minimal image dataset that loads JPEGs on-the-fly.
    The original files are never modified.
    """

    def __init__(self, image_paths: list[str], labels: list[int], transform=None):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ---------------------------------------------------------------------------
# 4. Transforms
# ---------------------------------------------------------------------------
def get_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# 5. WeightedRandomSampler -- handles class imbalance in training
# ---------------------------------------------------------------------------
def make_weighted_sampler(train_labels: list[int], num_classes: int) -> WeightedRandomSampler:
    counts = Counter(train_labels)
    class_weights = {cls: 1.0 / counts[cls] for cls in range(num_classes)}
    sample_weights = [class_weights[lbl] for lbl in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


# ---------------------------------------------------------------------------
# 6. Model -- EfficientNet-B0 with custom head
# ---------------------------------------------------------------------------
def build_model(num_classes: int, device: torch.device) -> nn.Module:
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Freeze all feature-extraction layers initially (fine-tune head first)
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier: (dropout + linear -> num_classes)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    # Only the new head is trainable at the start
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model.to(device)


# ---------------------------------------------------------------------------
# 7. Training & validation loops
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
) -> tuple[float, float]:
    model.train()
    running_loss   = 0.0
    correct        = 0
    total          = 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds         = outputs.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += images.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds         = outputs.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += images.size(0)

    return running_loss / total, correct / total


# ---------------------------------------------------------------------------
# 8. Test-set evaluation (called once with the best checkpoint)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_test_set(
    model: nn.Module,
    loader: DataLoader,
    class_names: list[str],
    device: torch.device,
) -> dict:
    model.eval()
    all_preds  = []
    all_labels = []

    for images, labels in tqdm(loader, desc="  Test eval", leave=False):
        images = images.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="weighted")

    report_dict = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
    )
    report_str = classification_report(
        all_labels, all_preds,
        target_names=class_names,
    )

    cm = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy":      float(acc),
        "weighted_f1":   float(f1),
        "report_dict":   report_dict,
        "report_str":    report_str,
        "confusion_matrix": cm.tolist(),
        "class_names":   class_names,
    }


# ---------------------------------------------------------------------------
# 9. Save confusion-matrix plot
# ---------------------------------------------------------------------------
def save_confusion_matrix(cm: list, class_names: list[str], save_path: Path) -> None:
    cm_arr = np.array(cm)
    # Normalise rows to show recall per class
    cm_norm = cm_arr.astype(float) / (cm_arr.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Normalised Confusion Matrix (row = recall)", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"[Eval] Confusion matrix saved -> {save_path}")


# ---------------------------------------------------------------------------
# 10. Unfreeze backbone for fine-tuning (called mid-training)
# ---------------------------------------------------------------------------
def unfreeze_backbone(model: nn.Module, optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Unfreeze all backbone layers and add them to the optimizer."""
    for param in model.parameters():
        param.requires_grad = True
    # Add backbone params with a lower learning rate
    optimizer.add_param_group({"params": [p for p in model.features.parameters()], "lr": lr * 0.1})
    print("[Train] Backbone unfrozen -- fine-tuning all layers.")


# ---------------------------------------------------------------------------
# 11. Main training orchestration
# ---------------------------------------------------------------------------
def run_training(
    epochs: int   = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float     = LEARNING_RATE,
    patience: int = PATIENCE,
    smoke_test: bool = False,
    num_workers: int = NUM_WORKERS,
) -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")
    if device.type == "cuda":
        print(f"         GPU: {torch.cuda.get_device_name(0)}")

    # -- 1. Discover dataset ----------------------------------------------
    image_paths, labels, class_names = discover_dataset(DATASET_ROOT)
    num_classes = len(class_names)

    # -- 2. Stratified split ----------------------------------------------
    (train_paths, train_labels,
     val_paths,   val_labels,
     test_paths,  test_labels) = stratified_split(image_paths, labels)

    print(f"\n[Split] Seed={SEED} | "
          f"Train={len(train_paths)} | "
          f"Val={len(val_paths)} | "
          f"Test={len(test_paths)}")

    # Verify no leakage
    train_set = set(train_paths)
    val_set   = set(val_paths)
    test_set  = set(test_paths)
    assert len(train_set & val_set)  == 0, "DATA LEAKAGE: train/val overlap!"
    assert len(train_set & test_set) == 0, "DATA LEAKAGE: train/test overlap!"
    assert len(val_set   & test_set) == 0, "DATA LEAKAGE: val/test overlap!"
    print("[Split] Leakage check: PASSED -- no overlap between splits.")

    if smoke_test:
        print("\n[Smoke Test] PASSED -- dataset, splits, and imports are all valid.")
        print("[Smoke Test] No training performed. Pass --smoke-test=false to train.")
        return

    # -- 3. Datasets & DataLoaders ----------------------------------------
    train_ds = WasteDataset(train_paths, train_labels, transform=get_train_transform())
    val_ds   = WasteDataset(val_paths,   val_labels,   transform=get_eval_transform())
    test_ds  = WasteDataset(test_paths,  test_labels,  transform=get_eval_transform())

    sampler  = make_weighted_sampler(train_labels, num_classes)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=(device.type=="cuda"),
                              persistent_workers=(num_workers > 0))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=(device.type=="cuda"),
                              persistent_workers=(num_workers > 0))
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=(device.type=="cuda"),
                              persistent_workers=(num_workers > 0))

    # -- 4. Model ---------------------------------------------------------
    model     = build_model(num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # -- 5. Training loop -------------------------------------------------
    best_val_acc   = 0.0
    epochs_no_improve = 0
    history: list[dict] = []
    unfreeze_epoch = max(3, epochs // 5)   # unfreeze backbone after first few epochs

    print(f"\n[Train] Starting {epochs} epochs  |  patience={patience}  |  batch={batch_size}")
    print(f"        Head-only epochs: 1-{unfreeze_epoch}  |  Full fine-tune: {unfreeze_epoch+1}-{epochs}")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Unfreeze backbone partway through
        if epoch == unfreeze_epoch + 1:
            unfreeze_backbone(model, optimizer, lr)
            # Reset scheduler over remaining epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=(epochs - unfreeze_epoch)
            )

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{epochs}  "
            f"| Train loss {train_loss:.4f}  acc {train_acc:.4f}  "
            f"| Val loss {val_loss:.4f}  acc {val_acc:.4f}  "
            f"| {elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  6),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,    6),
        })

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "class_names": class_names,
                "val_acc":     val_acc,
            }, MODEL_SAVE_PATH)
            print(f"  >> Best model saved  (val_acc={val_acc:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[EarlyStopping] No improvement for {patience} epochs -- stopping.")
                break

    # -- 6. Save class names ----------------------------------------------
    CLASS_NAMES_PATH.write_text("\n".join(class_names))
    print(f"\n[Save] Class names -> {CLASS_NAMES_PATH}")

    # -- 7. Save training metrics -----------------------------------------
    METRICS_PATH.write_text(json.dumps({"history": history}, indent=2))
    print(f"[Save] Training metrics -> {METRICS_PATH}")

    # -- 8. Evaluate on held-out test set ---------------------------------
    print("\n[Eval] Loading best checkpoint for test evaluation ...")
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    eval_results = evaluate_test_set(model, test_loader, class_names, device)

    print(f"\n{'='*70}")
    print(f"  TEST ACCURACY : {eval_results['accuracy']:.4f}")
    print(f"  WEIGHTED F1   : {eval_results['weighted_f1']:.4f}")
    print(f"{'='*70}")
    print(eval_results["report_str"])

    # Save confusion matrix plot
    save_confusion_matrix(eval_results["confusion_matrix"], class_names, CONFUSION_MAT_PATH)

    # Save evaluation report (strip non-serialisable items already converted above)
    eval_save = {
        "test_accuracy":   eval_results["accuracy"],
        "weighted_f1":     eval_results["weighted_f1"],
        "per_class_report": eval_results["report_dict"],
        "confusion_matrix": eval_results["confusion_matrix"],
        "class_names":      class_names,
        "best_val_acc":     best_val_acc,
        "epochs_trained":   len(history),
    }
    EVAL_REPORT_PATH.write_text(json.dumps(eval_save, indent=2))
    print(f"[Save] Evaluation report -> {EVAL_REPORT_PATH}")
    print("\n[Done] Training pipeline complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EcoSort AI - Training Pipeline")
    parser.add_argument("--smoke-test",  action="store_true",
                        help="Run dry-run only (no training): verify imports, split counts, leakage check.")
    parser.add_argument("--epochs",      type=int,   default=NUM_EPOCHS,    help=f"Training epochs (default {NUM_EPOCHS})")
    parser.add_argument("--batch-size",  type=int,   default=BATCH_SIZE,    help=f"Batch size (default {BATCH_SIZE})")
    parser.add_argument("--lr",          type=float, default=LEARNING_RATE, help=f"Initial learning rate (default {LEARNING_RATE})")
    parser.add_argument("--patience",    type=int,   default=PATIENCE,      help=f"Early stopping patience (default {PATIENCE})")
    parser.add_argument("--num-workers", type=int,   default=NUM_WORKERS,   help=f"DataLoader workers (default {NUM_WORKERS})")
    args = parser.parse_args()

    run_training(
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        lr          = args.lr,
        patience    = args.patience,
        smoke_test  = args.smoke_test,
        num_workers = args.num_workers,
    )
