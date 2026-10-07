"""
Waste Classification Agent
--------------------------
Accepts a PIL image, preprocesses it identically to training,
runs the trained EfficientNet-B0 model, and returns predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

# Preprocessing constants -- must match train.py exactly
IMG_SIZE      = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


class ModelNotReadyError(RuntimeError):
    """Raised when inference is attempted before the model is loaded."""


class InvalidImageError(ValueError):
    """Raised when the supplied image cannot be decoded or is unsupported."""


class WasteClassificationAgent:
    """
    Classifies a waste image into one of the 12 dataset categories.

    The ModelLoader singleton is used so the model is loaded only once
    per application lifetime (Streamlit reruns included).

    Example
    -------
        agent  = WasteClassificationAgent()
        result = agent.classify(pil_image)
        # result["predicted_class"]  -> "plastic"
        # result["confidence"]       -> 0.9341
        # result["top3"]             -> [("plastic", 0.93), ("metal", 0.04), ...]
    """

    def __init__(self) -> None:
        from services.model_loader import ModelLoader
        self._loader = ModelLoader()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """Return True if the model checkpoint is loaded and ready."""
        return self._loader.is_loaded

    def ensure_loaded(self) -> None:
        """Load the model if not already loaded. Propagates any load errors."""
        self._loader.load()

    def classify(self, image: Image.Image) -> dict:
        """
        Run inference on a PIL image.

        Parameters
        ----------
        image : PIL.Image.Image
            Any size / mode; will be converted to RGB and resized internally.

        Returns
        -------
        dict with keys:
            predicted_class : str          -- top-1 class name
            confidence      : float        -- top-1 probability  (0.0 - 1.0)
            top3            : list[tuple]  -- [(class_name, prob), ...] top 3
            all_scores      : dict[str, float]  -- full 12-class probability map
            device          : str          -- "cuda" or "cpu"
        """
        # -- Validate and normalise image --------------------------------
        image = self._validate_image(image)

        # -- Ensure model is ready ---------------------------------------
        if not self.is_ready():
            try:
                self._loader.load()
            except (FileNotFoundError, RuntimeError) as exc:
                raise ModelNotReadyError(str(exc)) from exc

        model       = self._loader.get_model()
        class_names = self._loader.get_class_names()
        device      = self._loader.device

        # -- Preprocess --------------------------------------------------
        tensor = _EVAL_TRANSFORM(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]

        # -- Inference ---------------------------------------------------
        with torch.no_grad():
            logits = model(tensor)                        # [1, 12]
            probs  = F.softmax(logits, dim=1)[0]          # [12]

        # -- Build result ------------------------------------------------
        probs_list  = probs.cpu().tolist()
        top3_idx    = sorted(range(len(probs_list)), key=lambda i: probs_list[i], reverse=True)[:3]
        top1_idx    = top3_idx[0]

        return {
            "predicted_class": class_names[top1_idx],
            "confidence":      probs_list[top1_idx],
            "top3":            [(class_names[i], probs_list[i]) for i in top3_idx],
            "all_scores":      {class_names[i]: probs_list[i] for i in range(len(class_names))},
            "device":          str(device),
        }

    def classify_from_path(self, path: str | Path) -> dict:
        """Convenience wrapper: load image from a file path, then classify."""
        path = Path(path)
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise InvalidImageError(
                f"Unsupported file extension '{path.suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        try:
            image = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError(f"Cannot read image '{path}': {exc}") from exc
        return self.classify(image)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_image(image: object) -> Image.Image:
        """
        Accept PIL.Image, file-like objects, or paths.  Always returns an
        RGB PIL.Image.  Raises InvalidImageError on failure.
        """
        if isinstance(image, Image.Image):
            try:
                return image.convert("RGB")
            except Exception as exc:
                raise InvalidImageError(f"Could not convert image to RGB: {exc}") from exc

        # File-like object (e.g. Streamlit UploadedFile)
        if hasattr(image, "read"):
            try:
                return Image.open(image).convert("RGB")
            except (UnidentifiedImageError, OSError) as exc:
                raise InvalidImageError(f"Cannot decode uploaded image: {exc}") from exc

        # Path-like
        if isinstance(image, (str, Path)):
            try:
                return Image.open(image).convert("RGB")
            except (UnidentifiedImageError, OSError) as exc:
                raise InvalidImageError(f"Cannot open image at '{image}': {exc}") from exc

        raise InvalidImageError(
            f"Unsupported image input type: {type(image).__name__}. "
            "Expected PIL.Image, file-like object, or path string."
        )
