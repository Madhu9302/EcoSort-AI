"""
Dataset utilities
-----------------
Helper functions for inspecting and loading the garbage-classification dataset.
"""

from __future__ import annotations
import os
from pathlib import Path

DATASET_ROOT = Path("data/garbage_classification")

WASTE_CLASSES = [
    "battery",
    "biological",
    "brown-glass",
    "cardboard",
    "clothes",
    "green-glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
    "white-glass",
]


def get_class_counts(dataset_root: Path = DATASET_ROOT) -> dict[str, int]:
    """Return a dict mapping each class name to its image count."""
    counts: dict[str, int] = {}
    for cls in sorted(os.listdir(dataset_root)):
        cls_path = dataset_root / cls
        if cls_path.is_dir():
            counts[cls] = len(list(cls_path.glob("*")))
    return counts


def get_total_images(dataset_root: Path = DATASET_ROOT) -> int:
    """Return the total number of images across all classes."""
    return sum(get_class_counts(dataset_root).values())


def get_dataset_summary(dataset_root: Path = DATASET_ROOT) -> dict:
    """Return a full summary dict suitable for the analytics page."""
    counts = get_class_counts(dataset_root)
    total = sum(counts.values())
    return {
        "dataset_root": str(dataset_root),
        "num_classes": len(counts),
        "total_images": total,
        "class_counts": counts,
    }
