from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image

from score2abc.chord_ocr.base import ChordDetection
from score2abc.utils.imaging import estimate_ink_threshold


def detect_barlines(
    image_path: Path,
    *,
    leading_margin_fraction: float = 0.05,
    trailing_margin_fraction: float = 0.02,
    min_column_density: float = 0.4,
    min_gap_fraction: float = 0.03,
) -> list[float]:
    """Return barline x-fractions (in [0, 1]) detected in a system crop.

    A barline shows up as a near-vertical column of ink spanning most of the
    staff's height. We approximate that with a threshold on per-column
    dark-pixel density, filter out the clef/time-sig area on the left and a
    narrow trailing strip on the right, and suppress duplicates within a
    minimum horizontal gap.
    """
    with Image.open(image_path) as image:
        gray = image.convert("L")
        return _detect_barlines_in_image(
            gray,
            leading_margin_fraction=leading_margin_fraction,
            trailing_margin_fraction=trailing_margin_fraction,
            min_column_density=min_column_density,
            min_gap_fraction=min_gap_fraction,
        )


def assign_measures(
    detections: Sequence[ChordDetection],
    barlines: Sequence[float],
) -> list[int]:
    """Return the 1-based system-local measure index for each detection."""
    sorted_barlines = sorted(barlines)
    return [_measure_for_x(detection.x_fraction, sorted_barlines) for detection in detections]


def measures_in_system(barlines: Sequence[float]) -> int:
    """Count measures in a system: N barlines are fences between N+1 measures."""
    return len(barlines) + 1


def _measure_for_x(x_fraction: float, sorted_barlines: Sequence[float]) -> int:
    for index, boundary in enumerate(sorted_barlines):
        if x_fraction < boundary:
            return index + 1
    return len(sorted_barlines) + 1


def _detect_barlines_in_image(
    gray: Image.Image,
    *,
    leading_margin_fraction: float,
    trailing_margin_fraction: float,
    min_column_density: float,
    min_gap_fraction: float,
) -> list[float]:
    width, height = gray.size
    if width <= 2 or height <= 2:
        return []

    threshold = estimate_ink_threshold(gray)
    pixels = gray.load()

    column_density = [0.0] * width
    for x in range(width):
        dark = 0
        for y in range(height):
            if pixels[x, y] < threshold:
                dark += 1
        column_density[x] = dark / height

    leading_margin = int(width * leading_margin_fraction)
    trailing_margin = int(width * trailing_margin_fraction)
    scan_left = max(0, leading_margin)
    scan_right = max(scan_left + 1, width - trailing_margin)
    min_gap = max(1, int(width * min_gap_fraction))

    peaks: list[int] = []
    last_peak = -min_gap - 1
    for x in range(scan_left, scan_right):
        if column_density[x] < min_column_density:
            continue
        if x - last_peak < min_gap:
            continue
        window_start = max(0, x - 2)
        window_end = min(width, x + 3)
        neighborhood = column_density[window_start:window_end]
        if column_density[x] < max(neighborhood):
            continue
        peaks.append(x)
        last_peak = x

    return [peak / width for peak in peaks]
