from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image

from score2abc.chord_ocr.base import ChordDetection
from score2abc.utils.imaging import estimate_ink_threshold

LEADING_BARLINE_FRACTION = 0.08
TRAILING_BARLINE_FRACTION = 0.92


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
    boundaries = measure_boundaries(barlines)
    return [_measure_for_x(detection.x_fraction, boundaries) for detection in detections]


def measure_boundaries(
    barlines: Sequence[float],
    *,
    leading_barline_fraction: float = LEADING_BARLINE_FRACTION,
    trailing_barline_fraction: float = TRAILING_BARLINE_FRACTION,
) -> list[float]:
    """Return normalized measure-edge x-fractions for one system.

    Barlines near the crop's left edge are treated as the left edge of measure
    1, not as an interior fence. Barlines near the crop's right edge are
    treated as the closing edge of the final measure. Interior barlines remain
    normal measure fences.
    """
    sorted_barlines = sorted(float(b) for b in barlines if 0.0 <= float(b) <= 1.0)
    leading = [b for b in sorted_barlines if b <= leading_barline_fraction]
    trailing = [b for b in sorted_barlines if b >= trailing_barline_fraction]
    middle = [
        b for b in sorted_barlines if leading_barline_fraction < b < trailing_barline_fraction
    ]

    left = max(leading) if leading else 0.0
    right = min(trailing) if trailing else 1.0
    if right <= left:
        return [0.0, 1.0]
    return _dedupe_boundaries([left, *middle, right])


def measures_in_system(
    barlines: Sequence[float],
    *,
    leading_barline_fraction: float = LEADING_BARLINE_FRACTION,
    trailing_barline_fraction: float = TRAILING_BARLINE_FRACTION,
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
    return max(
        0,
        len(
            measure_boundaries(
                barlines,
                leading_barline_fraction=leading_barline_fraction,
                trailing_barline_fraction=trailing_barline_fraction,
            )
        )
        - 1,
    )


def _measure_for_x(x_fraction: float, boundaries: Sequence[float]) -> int:
    for index, right_boundary in enumerate(boundaries[1:], start=1):
        if x_fraction < right_boundary:
            return index
    return max(1, len(boundaries) - 1)


def _dedupe_boundaries(boundaries: Sequence[float]) -> list[float]:
    deduped: list[float] = []
    for boundary in boundaries:
        if not deduped or abs(boundary - deduped[-1]) > 1e-6:
            deduped.append(boundary)
    return deduped


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
    staff_line_rows = _staff_line_rows(pixels, width, staff_top, staff_bot, threshold)

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
    peak_scores: list[float] = []
    last_peak = -min_gap - 1
    for x in range(scan_left, scan_right):
        if column_runs[x] < run_threshold:
            continue
        score = _barline_candidate_score(
            pixels,
            width=width,
            staff_top=staff_top,
            staff_bot=staff_bot,
            staff_height=staff_height,
            threshold=threshold,
            staff_line_rows=staff_line_rows,
            x=x,
            run=column_runs[x],
        )
        if x - last_peak < min_gap:
            if peaks and score > peak_scores[-1]:
                peaks[-1] = x
                peak_scores[-1] = score
                last_peak = x
            continue
        window_start = max(0, x - 2)
        window_end = min(width, x + 3)
        if column_runs[x] < max(column_runs[window_start:window_end]):
            continue
        peaks.append(x)
        peak_scores.append(score)
        last_peak = x

    peaks.extend(
        _recover_edge_barlines(
            pixels,
            width=width,
            staff_top=staff_top,
            staff_bot=staff_bot,
            staff_height=staff_height,
            threshold=threshold,
            staff_line_rows=staff_line_rows,
            column_runs=column_runs,
            existing_peaks=peaks,
        )
    )
    peaks = sorted(set(peaks))
    return [peak / width for peak in peaks]


def _barline_candidate_score(
    pixels,
    *,
    width: int,
    staff_top: int,
    staff_bot: int,
    staff_height: int,
    threshold: int,
    staff_line_rows: Sequence[int],
    x: int,
    run: int,
) -> float:
    side_ink = _side_ink_density_between_staff_lines(
        pixels,
        width=width,
        staff_top=staff_top,
        staff_bot=staff_bot,
        threshold=threshold,
        staff_line_rows=staff_line_rows,
        x=x,
    )
    return (run / staff_height) - (0.45 * side_ink)


def _recover_edge_barlines(
    pixels,
    *,
    width: int,
    staff_top: int,
    staff_bot: int,
    staff_height: int,
    threshold: int,
    staff_line_rows: Sequence[int],
    column_runs: Sequence[int],
    existing_peaks: Sequence[int],
) -> list[int]:
    recovered: list[int] = []
    edge_min_run = 0.35 * staff_height
    edge_min_dark_rows = 0.5

    for left, right, is_left_edge in (
        (int(width * 0.015), int(width * 0.04), True),
        (int(width * 0.96), width - 1, False),
    ):
        if right <= left:
            continue
        if is_left_edge and any(peak <= right for peak in existing_peaks):
            continue
        if not is_left_edge and any(peak >= left for peak in existing_peaks):
            continue

        best_x: int | None = None
        best_score = float("-inf")
        for x in range(left, right + 1):
            dark_row_fraction = _dark_row_fraction(
                pixels,
                width=width,
                staff_top=staff_top,
                staff_bot=staff_bot,
                threshold=threshold,
                x=x,
            )
            if column_runs[x] < edge_min_run or dark_row_fraction < edge_min_dark_rows:
                continue
            score = _barline_candidate_score(
                pixels,
                width=width,
                staff_top=staff_top,
                staff_bot=staff_bot,
                staff_height=staff_height,
                threshold=threshold,
                staff_line_rows=staff_line_rows,
                x=x,
                run=column_runs[x],
            )
            if score > best_score:
                best_x = x
                best_score = score
        if best_x is not None:
            recovered.append(best_x)

    return recovered


def _dark_row_fraction(
    pixels,
    *,
    width: int,
    staff_top: int,
    staff_bot: int,
    threshold: int,
    x: int,
) -> float:
    dark_rows = 0
    for y in range(staff_top, staff_bot + 1):
        if any(pixels[xx, y] < threshold for xx in range(max(0, x - 1), min(width - 1, x + 1) + 1)):
            dark_rows += 1
    return dark_rows / (staff_bot - staff_top + 1)


def _side_ink_density_between_staff_lines(
    pixels,
    *,
    width: int,
    staff_top: int,
    staff_bot: int,
    threshold: int,
    staff_line_rows: Sequence[int],
    x: int,
) -> float:
    total = 0
    dark = 0
    for xx in range(max(0, x - 24), min(width - 1, x + 24) + 1):
        if abs(xx - x) <= 2:
            continue
        for y in range(staff_top, staff_bot + 1):
            if _near_staff_line(y, staff_line_rows):
                continue
            total += 1
            if pixels[xx, y] < threshold:
                dark += 1
    return dark / total if total else 0.0


def _near_staff_line(y: int, staff_line_rows: Sequence[int]) -> bool:
    return any(abs(y - line_y) <= 3 for line_y in staff_line_rows)


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


def _staff_line_rows(
    pixels,
    width: int,
    staff_top: int,
    staff_bot: int,
    threshold: int,
) -> list[int]:
    rows = []
    for y in range(staff_top, staff_bot + 1):
        dark = 0
        for x in range(width):
            if pixels[x, y] < threshold:
                dark += 1
        rows.append((dark, y))

    lines: list[int] = []
    for _, y in sorted(rows, reverse=True):
        if all(abs(y - line_y) > 8 for line_y in lines):
            lines.append(y)
        if len(lines) == 5:
            break
    return sorted(lines)
