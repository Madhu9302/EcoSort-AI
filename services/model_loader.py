"""
Model loader service
---------------------
Singleton that loads waste_classifier.pt exactly once and caches it.
Recreates the same EfficientNet-B0 + 12-class head used during training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

MODEL_PATH       = Path("models/waste_classifier.pt")
CLASS_NAMES_PATH = Path("models/class_names.txt")
NUM_CLASSES      = 12


def _build_efficientnet_b0(num_classes: int) -> nn.Module:
    """Recreate the exact architecture used during training (no pretrained weights)."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


class ModelLoader:
    """
    Thread-safe singleton that loads the trained checkpoint once and
    serves it for every subsequent inference call.

    Usage
    -----
        loader = ModelLoader()
        loader.load()                 # idempotent -- safe to call multiple times
        model       = loader.get_model()
        class_names = loader.get_class_names()
        device      = loader.device
    """

    _instance: Optional["ModelLoader"] = None

    def __new__(cls) -> "ModelLoader":
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._model: Optional[nn.Module] = None
            inst._class_names: Optional[list[str]] = None
            inst._device: Optional[torch.device] = None
            inst._load_error: Optional[str] = None
            cls._instance = inst
        return cls._instance

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load the trained checkpoint from disk.  Idempotent -- calling this
        multiple times has no effect if the model is already loaded.

        Raises
        ------
        FileNotFoundError  -- checkpoint or class-names file is missing
        RuntimeError       -- checkpoint is structurally incompatible
        """
        if self.is_loaded:
            return

        # -- Validate files exist ----------------------------------------
        if not MODEL_PATH.exists():
            msg = (
                f"Trained model not found at '{MODEL_PATH}'. "
                "Please ensure model training completed successfully."
            )
            self._load_error = msg
            raise FileNotFoundError(msg)

        if not CLASS_NAMES_PATH.exists():
            msg = f"Class names file not found at '{CLASS_NAMES_PATH}'."
            self._load_error = msg
            raise FileNotFoundError(msg)

        # -- Load class names --------------------------------------------
        class_names = [
            line.strip()
            for line in CLASS_NAMES_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        num_classes = len(class_names)

        # -- Rebuild architecture ----------------------------------------
        model = _build_efficientnet_b0(num_classes)

        # -- Load checkpoint ---------------------------------------------
        checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)

        # Checkpoint may be a plain state_dict or our training dict
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        self._model       = model
        self._class_names = class_names
        self._load_error  = None

    # ------------------------------------------------------------------
    # Accessors (auto-load on first call)
    # ------------------------------------------------------------------

    def get_model(self) -> nn.Module:
        if not self.is_loaded:
            self.load()
        return self._model  # type: ignore[return-value]

    def get_class_names(self) -> list[str]:
        if not self.is_loaded:
            self.load()
        return self._class_names  # type: ignore[return-value]
