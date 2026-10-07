"""
Analytics service
-----------------
Computes and caches waste classification statistics for the
Waste Analytics page.
"""

from __future__ import annotations
from pathlib import Path
from utils.dataset_utils import get_dataset_summary, DATASET_ROOT


def get_dataset_analytics() -> dict:
    """
    Return dataset-level analytics (class distribution, total images).
    Uses the on-disk dataset – does not require a trained model.
    """
    return get_dataset_summary(DATASET_ROOT)


def get_session_analytics(classification_history: list[dict]) -> dict:
    """
    Compute per-session classification statistics.

    Parameters
    ----------
    classification_history : list[dict]
        Each entry has keys: timestamp, predicted_class, confidence.

    Returns
    -------
    dict with keys: total_scans, class_counts, avg_confidence
    """
    if not classification_history:
        return {"total_scans": 0, "class_counts": {}, "avg_confidence": 0.0}

    class_counts: dict[str, int] = {}
    total_confidence = 0.0

    for entry in classification_history:
        cls = entry.get("predicted_class", "unknown")
        class_counts[cls] = class_counts.get(cls, 0) + 1
        total_confidence += entry.get("confidence", 0.0)

    return {
        "total_scans": len(classification_history),
        "class_counts": class_counts,
        "avg_confidence": total_confidence / len(classification_history),
    }
