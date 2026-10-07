"""
Image utilities
---------------
Helpers for loading and pre-processing images before inference.
"""

from __future__ import annotations
from pathlib import Path
from PIL import Image


def load_image(source) -> Image.Image:
    """
    Load an image from a file path or a file-like object.

    Parameters
    ----------
    source : str | Path | file-like object

    Returns
    -------
    PIL.Image.Image in RGB mode
    """
    if isinstance(source, (str, Path)):
        img = Image.open(source)
    else:
        img = Image.open(source)
    return img.convert("RGB")


def resize_image(image: Image.Image, size: tuple[int, int] = (224, 224)) -> Image.Image:
    """Resize an image to the target size using LANCZOS resampling."""
    return image.resize(size, Image.LANCZOS)
