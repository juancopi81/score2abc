"""Shared image-processing helpers."""

from __future__ import annotations

from PIL import Image, ImageStat


def estimate_ink_threshold(gray: Image.Image) -> int:
    """Pick a per-page ink/paper luma cutoff from the image's mean brightness."""
    mean_luma = ImageStat.Stat(gray).mean[0]
    return max(150, min(220, int(mean_luma - 28)))
