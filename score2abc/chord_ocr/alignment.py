from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image

from score2abc.chord_ocr.base import ChordDetection
from score2abc.utils.imaging import estimate_ink_threshold


def detect_barlines(
    image_path: Path,
    *,
    leading_margin_fraction: float = 0.02,
    trailing_margin_fraction: float = 0.02,
    min_run_fraction: float = 0.75,
    min_gap_fraction: float = 0.03,
    staff_pad_px: int = 4,
    oblique_tolerance_px: int = 1,
) -> list[float]:
    """Return barline x-fractions (in [0, 1]) detected in a system crop.

    A barline is a vertical stroke that spans the full 5-line staff. The
    detector first locates the staff vertical band via horizontal projection
    (rows where dark pixels cover ~100% of the width are the staff lines),
    then for each column measures the longest run of consecutive dark pixels
    inside that band. A small horizontal dilation absorbs slight oblique drift
    in handwritten or old-engraved barlines: at row y, column x counts as
    dark if any pixel in [x - tol, x + tol] is below the ink threshold.
    """
    with Image.open(image_path) as image:
        gray = image.convert("L")
        return _detect_barlines_in_image(
            gray,
            leading_margin_fraction=leading_margin_fraction,
            trailing_margin_fraction=trailing_margin_fraction,
            min_run_fraction=min_run_fraction,
            min_gap_fraction=min_gap_fraction,
            staff_pad_px=staff_pad_px,
            oblique_tolerance_px=oblique_tolerance_px,
        )


def assign_measures(
    detections: Sequence[ChordDetection],
    barlines: Sequence[float],
) -> list[int]:
    """Return the 1-based system-local measure index for each detection."""
    sorted_barlines = sorted(barlines)
    return [_measure_for_x(detection.x_fraction, sorted_barlines) for detection in detections]


def measures_in_system(
    barlines: Sequence[float],
    *,
    leading_barline_fraction: float = 0.05,
    trailing_barline_fraction: float = 0.97,
) -> int:
    """Count measures, accounting for leading and terminal barlines.

    The naive count is N+1 (N barlines fence N+1 measures). But when a system
    *starts* with a barline (continuation systems often do), the leftmost
    barline is the left edge of measure 1, not a fence between measures, so
    we subtract one. Likewise the rightmost barline is typically the terminal
    barline that closes the final measure rather than a fence opening another,
    so we subtract one again. Defaults treat barlines in the leftmost 5% /
    rightmost 3% of the crop as leading / terminal.
    """
    count = len(barlines) + 1
    sorted_barlines = sorted(barlines)
    if sorted_barlines and sorted_barlines[0] <= leading_barline_fraction:
        count -= 1
    if sorted_barlines and sorted_barlines[-1] >= trailing_barline_fraction:
        count -= 1
    return max(count, 0)


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
    min_run_fraction: float,
    min_gap_fraction: float,
    staff_pad_px: int,
    oblique_tolerance_px: int,
) -> list[float]:
    width, height = gray.size
    if width <= 2 or height <= 2:
        return []

    threshold = estimate_ink_threshold(gray)
    pixels = gray.load()

    staff_top, staff_bot = _staff_band(pixels, width, height, threshold, staff_pad_px)
    staff_height = staff_bot - staff_top + 1
    if staff_height <= 4:
        return []

    tol = max(0, oblique_tolerance_px)
    column_runs = [0] * width
    for x in range(width):
        cur = 0
        best = 0
        x_lo = max(0, x - tol)
        x_hi = min(width - 1, x + tol)
        for y in range(staff_top, staff_bot + 1):
            is_dark = False
            for xx in range(x_lo, x_hi + 1):
                if pixels[xx, y] < threshold:
                    is_dark = True
                    break
            if is_dark:
                cur += 1
                if cur > best:
                    best = cur
            else:
                cur = 0
        column_runs[x] = best

    run_threshold = min_run_fraction * staff_height
    leading_margin = int(width * leading_margin_fraction)
    trailing_margin = int(width * trailing_margin_fraction)
    scan_left = max(0, leading_margin)
    scan_right = max(scan_left + 1, width - trailing_margin)
    min_gap = max(1, int(width * min_gap_fraction))

    peaks: list[int] = []
    last_peak = -min_gap - 1
    for x in range(scan_left, scan_right):
        if column_runs[x] < run_threshold:
            continue
        if x - last_peak < min_gap:
            if peaks and column_runs[x] > column_runs[last_peak]:
                peaks[-1] = x
                last_peak = x
            continue
        window_start = max(0, x - 2)
        window_end = min(width, x + 3)
        if column_runs[x] < max(column_runs[window_start:window_end]):
            continue
        peaks.append(x)
        last_peak = x

    return [peak / width for peak in peaks]


def _staff_band(
    pixels,
    width: int,
    height: int,
    threshold: int,
    pad: int,
) -> tuple[int, int]:
    """Return (top, bottom) Y indices bracketing the 5 staff lines.

    Staff lines span ~100% of the crop width and are very dark. 0.7 * width
    isolates them cleanly from chord-text and dynamics rows that touch
    ~0.5 * width. Falls back to the top-5 densest rows when the staff is
    faint enough that none clear the threshold.
    """
    row_density = [0] * height
    for y in range(height):
        dark = 0
        for x in range(width):
            if pixels[x, y] < threshold:
                dark += 1
        row_density[y] = dark

    high = [y for y, d in enumerate(row_density) if d >= 0.7 * width]
    if len(high) < 5:
        ranked = sorted(range(height), key=lambda y: row_density[y], reverse=True)
        high = sorted(ranked[:5])

    top = max(0, min(high) - pad)
    bot = min(height - 1, max(high) + pad)
    return top, bot
