from __future__ import annotations

from pathlib import Path
from statistics import median
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

    A barline is a vertical stroke that spans nearly the full 5-line staff. The
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
    return assign_measures_to_boundaries(detections, boundaries)


def assign_measures_to_boundaries(
    detections: Sequence[ChordDetection],
    boundaries: Sequence[float],
) -> list[int]:
    """Return the 1-based system-local measure index using explicit boundaries."""
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


def measure_boundaries_for_system(
    image_path: Path,
    barlines: Sequence[float],
) -> list[float]:
    """Return measure boundaries after image-aware cleanup."""
    with Image.open(image_path) as image:
        gray = image.convert("L")
        cleaned_barlines = _merge_accidental_slices(
            gray,
            _dedupe_boundaries(sorted(float(b) for b in barlines if 0.0 <= float(b) <= 1.0)),
        )
        cleaned_barlines = _reject_leading_note_stem(gray, cleaned_barlines)
        cleaned_barlines = _reject_note_stem_before_terminal_barline(gray, cleaned_barlines)
        boundaries = measure_boundaries(cleaned_barlines)
        boundaries = _trim_blank_tail(gray, boundaries)
        boundaries = _merge_accidental_slices(gray, boundaries)
        boundaries = _merge_note_stem_slices(gray, boundaries)
        boundaries = _merge_blank_trailing_slice(gray, boundaries)
        return _merge_note_stem_slices(gray, boundaries)


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
        if column_runs[x] < run_threshold and (
            column_runs[x] < max(0.0, min_run_fraction - 0.005) * staff_height
            or not _spans_staff_edges(
                pixels,
                width=width,
                staff_top=staff_top,
                staff_bot=staff_bot,
                threshold=threshold,
                x=x,
            )
        ):
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


def _spans_staff_edges(
    pixels,
    *,
    width: int,
    staff_top: int,
    staff_bot: int,
    threshold: int,
    x: int,
) -> bool:
    staff_height = staff_bot - staff_top + 1
    edge_height = max(5, round(0.12 * staff_height))
    top_fraction = _dark_fraction_in_rows(
        pixels,
        width=width,
        threshold=threshold,
        x=x,
        y0=staff_top,
        y1=min(staff_bot, staff_top + edge_height - 1),
    )
    bottom_fraction = _dark_fraction_in_rows(
        pixels,
        width=width,
        threshold=threshold,
        x=x,
        y0=max(staff_top, staff_bot - edge_height + 1),
        y1=staff_bot,
    )
    return top_fraction >= 0.75 and bottom_fraction >= 0.75


def _dark_fraction_in_rows(
    pixels,
    *,
    width: int,
    threshold: int,
    x: int,
    y0: int,
    y1: int,
) -> float:
    rows = y1 - y0 + 1
    if rows <= 0:
        return 0.0
    dark_rows = 0
    for y in range(y0, y1 + 1):
        if any(pixels[xx, y] < threshold for xx in range(max(0, x - 1), min(width - 1, x + 1) + 1)):
            dark_rows += 1
    return dark_rows / rows


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


def _near_staff_line(
    y: int,
    staff_line_rows: Sequence[int],
    *,
    pad_px: int = 3,
) -> bool:
    return any(abs(y - line_y) <= pad_px for line_y in staff_line_rows)


def _reject_leading_note_stem(
    gray: Image.Image,
    barlines: Sequence[float],
) -> list[float]:
    if not barlines or barlines[0] <= 0.05 or barlines[0] > LEADING_BARLINE_FRACTION:
        return list(barlines)

    width, height = gray.size
    threshold = estimate_ink_threshold(gray)
    pixels = gray.load()
    staff_top, staff_bot = _staff_band(pixels, width, height, threshold, pad=4)
    staff_line_rows = _staff_line_rows(pixels, width, staff_top, staff_bot, threshold)
    if len(staff_line_rows) < 5:
        return list(barlines)

    staff_spacings = [
        right - left for left, right in zip(staff_line_rows, staff_line_rows[1:], strict=False)
    ]
    candidate_x = round(barlines[0] * width)
    top_extension = _contiguous_vertical_extension(
        pixels,
        width=width,
        height=height,
        threshold=threshold,
        x=candidate_x,
        start_y=staff_top - 1,
        step=-1,
    )
    if top_extension < 0.9 * median(staff_spacings):
        return list(barlines)

    left_ink_density = _nonstaff_ink_density(
        pixels,
        width=width,
        staff_top=staff_top,
        staff_bot=staff_bot,
        threshold=threshold,
        staff_line_rows=staff_line_rows,
        x0=0,
        x1=max(0, candidate_x - 12),
    )
    if left_ink_density < 0.02:
        return list(barlines)

    return list(barlines[1:])


def _reject_note_stem_before_terminal_barline(
    gray: Image.Image,
    barlines: Sequence[float],
) -> list[float]:
    if len(barlines) < 2:
        return list(barlines)

    terminal = barlines[-1]
    candidate = barlines[-2]
    if terminal < TRAILING_BARLINE_FRACTION or candidate >= TRAILING_BARLINE_FRACTION:
        return list(barlines)

    width, height = gray.size
    threshold = estimate_ink_threshold(gray)
    pixels = gray.load()
    staff_top, staff_bot = _staff_band(pixels, width, height, threshold, pad=4)
    staff_line_rows = _staff_line_rows(pixels, width, staff_top, staff_bot, threshold)
    if len(staff_line_rows) < 5:
        return list(barlines)

    staff_spacings = [
        right - left for left, right in zip(staff_line_rows, staff_line_rows[1:], strict=False)
    ]
    gap_px = round((terminal - candidate) * width)
    if gap_px > 4.0 * median(staff_spacings):
        return list(barlines)

    candidate_x = round(candidate * width)
    terminal_x = round(terminal * width)
    candidate_cluster = _vertical_cluster_width(
        pixels,
        width=width,
        staff_top=staff_top,
        staff_bot=staff_bot,
        threshold=threshold,
        x=candidate_x,
    )
    terminal_cluster = _vertical_cluster_width(
        pixels,
        width=width,
        staff_top=staff_top,
        staff_bot=staff_bot,
        threshold=threshold,
        x=terminal_x,
    )
    if candidate_cluster >= 8 or terminal_cluster < 8:
        return list(barlines)

    top_extension = _contiguous_vertical_extension(
        pixels,
        width=width,
        height=height,
        threshold=threshold,
        x=candidate_x,
        start_y=staff_top - 1,
        step=-1,
    )
    if top_extension < 0.7 * median(staff_spacings):
        return list(barlines)

    candidate_side_ink = _side_ink_density_between_staff_lines(
        pixels,
        width=width,
        staff_top=staff_top,
        staff_bot=staff_bot,
        threshold=threshold,
        staff_line_rows=staff_line_rows,
        x=candidate_x,
    )
    terminal_side_ink = _side_ink_density_between_staff_lines(
        pixels,
        width=width,
        staff_top=staff_top,
        staff_bot=staff_bot,
        threshold=threshold,
        staff_line_rows=staff_line_rows,
        x=terminal_x,
    )
    if candidate_side_ink < terminal_side_ink + 0.02:
        return list(barlines)

    return [*barlines[:-2], terminal]


def _contiguous_vertical_extension(
    pixels,
    *,
    width: int,
    height: int,
    threshold: int,
    x: int,
    start_y: int,
    step: int,
) -> int:
    extension = 0
    y = start_y
    while 0 <= y < height:
        if not any(
            pixels[xx, y] < threshold for xx in range(max(0, x - 1), min(width - 1, x + 1) + 1)
        ):
            break
        extension += 1
        y += step
    return extension


def _trim_blank_tail(gray: Image.Image, boundaries: Sequence[float]) -> list[float]:
    if len(boundaries) < 3 or abs(boundaries[-1] - 1.0) > 1e-6:
        return list(boundaries)

    width, height = gray.size
    threshold = estimate_ink_threshold(gray)
    pixels = gray.load()
    staff_top, staff_bot = _staff_band(pixels, width, height, threshold, pad=4)
    staff_line_rows = _staff_line_rows(pixels, width, staff_top, staff_bot, threshold)

    terminal_candidate = boundaries[-2]
    tail_width_fraction = 1.0 - terminal_candidate
    if tail_width_fraction < 0.12:
        return list(boundaries)

    terminal_x = round(terminal_candidate * width)
    if (
        _vertical_cluster_width(
            pixels,
            width=width,
            staff_top=staff_top,
            staff_bot=staff_bot,
            threshold=threshold,
            x=terminal_x,
        )
        < 8
    ):
        return list(boundaries)

    x0 = min(width - 1, round(terminal_candidate * width) + 12)
    density = _nonstaff_ink_density(
        pixels,
        width=width,
        staff_top=staff_top,
        staff_bot=staff_bot,
        threshold=threshold,
        staff_line_rows=staff_line_rows,
        x0=x0,
        x1=width - 1,
    )
    if density <= 0.02:
        return list(boundaries[:-1])
    return list(boundaries)


def _merge_accidental_slices(gray: Image.Image, boundaries: Sequence[float]) -> list[float]:
    if len(boundaries) < 4:
        return list(boundaries)

    width, height = gray.size
    threshold = estimate_ink_threshold(gray)
    pixels = gray.load()
    staff_top, staff_bot = _staff_band(pixels, width, height, threshold, pad=4)
    staff_line_rows = _staff_line_rows(pixels, width, staff_top, staff_bot, threshold)

    cleaned = list(boundaries)
    while len(cleaned) >= 4:
        widths = _boundary_widths_px(cleaned, width)
        typical_width = median(widths)
        removed_index: int | None = None

        for slice_index, slice_width in enumerate(widths):
            if slice_width >= min(160.0, 0.62 * typical_width):
                continue
            left_index = slice_index
            right_index = slice_index + 1
            candidates = []
            if left_index > 0:
                candidates.append(left_index)
            if right_index < len(cleaned) - 1:
                candidates.append(right_index)
            removed_index = _most_accidental_like_boundary_index(
                pixels,
                boundaries=cleaned,
                candidate_indices=candidates,
                width=width,
                staff_top=staff_top,
                staff_bot=staff_bot,
                threshold=threshold,
                staff_line_rows=staff_line_rows,
            )
            if removed_index is not None:
                break

        if removed_index is None:
            for boundary_index in range(1, len(cleaned) - 1):
                left_width = widths[boundary_index - 1]
                right_width = widths[boundary_index]
                if left_width >= 0.8 * typical_width or right_width >= 0.8 * typical_width:
                    continue
                if left_width + right_width < 1.15 * typical_width:
                    continue
                if (
                    _accidental_boundary_score(
                        pixels,
                        boundary=cleaned[boundary_index],
                        width=width,
                        staff_top=staff_top,
                        staff_bot=staff_bot,
                        threshold=threshold,
                        staff_line_rows=staff_line_rows,
                    )
                    >= 0.11
                ):
                    removed_index = boundary_index
                    break

        if removed_index is None:
            return cleaned
        del cleaned[removed_index]

    return cleaned


def _merge_note_stem_slices(gray: Image.Image, boundaries: Sequence[float]) -> list[float]:
    """Remove severe note-stem boundaries only when they create a narrow slice."""
    if len(boundaries) < 4:
        return list(boundaries)

    width, height = gray.size
    threshold = estimate_ink_threshold(gray)
    pixels = gray.load()
    staff_top, staff_bot = _staff_band(pixels, width, height, threshold, pad=4)
    staff_line_rows = _staff_line_rows(pixels, width, staff_top, staff_bot, threshold)

    cleaned = list(boundaries)
    while len(cleaned) >= 4:
        widths = _boundary_widths_px(cleaned, width)
        typical_width = median(widths)
        removed_index: int | None = None

        for slice_index, slice_width in enumerate(widths):
            if slice_width >= min(160.0, 0.62 * typical_width):
                continue

            candidate_indices = []
            if slice_index > 0:
                candidate_indices.append(slice_index)
            if slice_index + 1 < len(cleaned) - 1:
                candidate_indices.append(slice_index + 1)

            scored_candidates = [
                (
                    _note_stem_boundary_score(
                        pixels,
                        boundary=cleaned[index],
                        width=width,
                        staff_top=staff_top,
                        staff_bot=staff_bot,
                        threshold=threshold,
                        staff_line_rows=staff_line_rows,
                    ),
                    index,
                )
                for index in candidate_indices
            ]
            if scored_candidates and max(scored_candidates)[0] > 0.0:
                removed_index = max(scored_candidates)[1]
                break

        if removed_index is None:
            return cleaned
        del cleaned[removed_index]

    return cleaned


def _note_stem_boundary_score(
    pixels,
    *,
    boundary: float,
    width: int,
    staff_top: int,
    staff_bot: int,
    threshold: int,
    staff_line_rows: Sequence[int],
) -> float:
    x = round(boundary * width)
    staff_height = staff_bot - staff_top + 1
    edge_height = max(5, round(0.12 * staff_height))
    top_fraction = _dark_fraction_in_rows(
        pixels,
        width=width,
        threshold=threshold,
        x=x,
        y0=staff_top,
        y1=min(staff_bot, staff_top + edge_height - 1),
    )
    bottom_fraction = _dark_fraction_in_rows(
        pixels,
        width=width,
        threshold=threshold,
        x=x,
        y0=max(staff_top, staff_bot - edge_height + 1),
        y1=staff_bot,
    )
    side_ink = _side_ink_density_between_staff_lines(
        pixels,
        width=width,
        staff_top=staff_top,
        staff_bot=staff_bot,
        threshold=threshold,
        staff_line_rows=staff_line_rows,
        x=x,
    )
    if side_ink < 0.08:
        return 0.0

    cluster_width = _vertical_cluster_width(
        pixels,
        width=width,
        staff_top=staff_top,
        staff_bot=staff_bot,
        threshold=threshold,
        x=x,
    )
    misses_both_edges = top_fraction < 0.75 and bottom_fraction < 0.75
    severely_misses_one_edge = min(top_fraction, bottom_fraction) < 0.5 and cluster_width >= 12
    if not (misses_both_edges or severely_misses_one_edge):
        return 0.0

    return (0.75 - min(top_fraction, bottom_fraction)) + side_ink + (0.01 * cluster_width)


def _merge_blank_trailing_slice(
    gray: Image.Image,
    boundaries: Sequence[float],
) -> list[float]:
    """Merge a wide final staff-only slice without clipping preceding music."""
    if len(boundaries) < 4:
        return list(boundaries)

    width, height = gray.size
    penultimate_x = round(boundaries[-2] * width)
    final_x = round(boundaries[-1] * width)
    if final_x - penultimate_x < 0.12 * width:
        return list(boundaries)

    threshold = estimate_ink_threshold(gray)
    pixels = gray.load()
    staff_top, staff_bot = _staff_band(pixels, width, height, threshold, pad=4)
    staff_line_rows = _staff_line_rows(pixels, width, staff_top, staff_bot, threshold)
    if len(staff_line_rows) < 5:
        return list(boundaries)

    staff_spacings = [
        right - left for left, right in zip(staff_line_rows, staff_line_rows[1:], strict=False)
    ]
    staff_spacing = median(staff_spacings)
    boundary_fraction = boundaries[-2]
    stem_score = _note_stem_boundary_score(
        pixels,
        boundary=boundary_fraction,
        width=width,
        staff_top=staff_top,
        staff_bot=staff_bot,
        threshold=threshold,
        staff_line_rows=staff_line_rows,
    )
    cluster_width = _vertical_cluster_width(
        pixels,
        width=width,
        staff_top=staff_top,
        staff_bot=staff_bot,
        threshold=threshold,
        x=penultimate_x,
    )
    side_ink = _side_ink_density_between_staff_lines(
        pixels,
        width=width,
        staff_top=staff_top,
        staff_bot=staff_bot,
        threshold=threshold,
        staff_line_rows=staff_line_rows,
        x=penultimate_x,
    )
    extension = max(
        _contiguous_vertical_extension(
            pixels,
            width=width,
            height=height,
            threshold=threshold,
            x=penultimate_x,
            start_y=staff_top - 1,
            step=-1,
        ),
        _contiguous_vertical_extension(
            pixels,
            width=width,
            height=height,
            threshold=threshold,
            x=penultimate_x,
            start_y=staff_bot + 1,
            step=1,
        ),
    )
    attached_stem = cluster_width < 12 and side_ink >= 0.08 and extension >= 0.4 * staff_spacing
    if stem_score <= 0.0 and not attached_stem:
        return list(boundaries)

    line_pad_px = max(3, round(0.24 * staff_spacing))
    x0 = penultimate_x + 12
    x1 = final_x - 12
    if x1 <= x0:
        return list(boundaries)

    density = _nonstaff_ink_density(
        pixels,
        width=width,
        staff_top=staff_top,
        staff_bot=staff_bot,
        threshold=threshold,
        staff_line_rows=staff_line_rows,
        x0=x0,
        x1=x1,
        staff_line_pad_px=line_pad_px,
    )
    if density <= 0.03:
        return [*boundaries[:-2], boundaries[-1]]
    return list(boundaries)


def _boundary_widths_px(boundaries: Sequence[float], width: int) -> list[int]:
    return [
        max(0, round(right * width) - round(left * width))
        for left, right in zip(boundaries, boundaries[1:], strict=False)
    ]


def _most_accidental_like_boundary_index(
    pixels,
    *,
    boundaries: Sequence[float],
    candidate_indices: Sequence[int],
    width: int,
    staff_top: int,
    staff_bot: int,
    threshold: int,
    staff_line_rows: Sequence[int],
) -> int | None:
    best_index: int | None = None
    best_score = 0.0
    for index in candidate_indices:
        score = _accidental_boundary_score(
            pixels,
            boundary=boundaries[index],
            width=width,
            staff_top=staff_top,
            staff_bot=staff_bot,
            threshold=threshold,
            staff_line_rows=staff_line_rows,
        )
        if score > best_score:
            best_index = index
            best_score = score
    return best_index if best_score >= 0.11 else None


def _accidental_boundary_score(
    pixels,
    *,
    boundary: float,
    width: int,
    staff_top: int,
    staff_bot: int,
    threshold: int,
    staff_line_rows: Sequence[int],
) -> float:
    x = round(boundary * width)
    if (
        _vertical_cluster_width(
            pixels,
            width=width,
            staff_top=staff_top,
            staff_bot=staff_bot,
            threshold=threshold,
            x=x,
        )
        < 12
    ):
        return 0.0
    return _side_ink_density_between_staff_lines(
        pixels,
        width=width,
        staff_top=staff_top,
        staff_bot=staff_bot,
        threshold=threshold,
        staff_line_rows=staff_line_rows,
        x=x,
    )


def _vertical_cluster_width(
    pixels,
    *,
    width: int,
    staff_top: int,
    staff_bot: int,
    threshold: int,
    x: int,
) -> int:
    center = max(0, min(width - 1, x))
    dark_threshold = 0.45

    left = center
    empty_gap = 0
    for xx in range(center, max(-1, center - 32), -1):
        if (
            _dark_row_fraction(
                pixels,
                width=width,
                staff_top=staff_top,
                staff_bot=staff_bot,
                threshold=threshold,
                x=xx,
            )
            >= dark_threshold
        ):
            left = xx
            empty_gap = 0
        else:
            empty_gap += 1
            if empty_gap > 8:
                break

    right = center
    empty_gap = 0
    for xx in range(center, min(width, center + 33)):
        if (
            _dark_row_fraction(
                pixels,
                width=width,
                staff_top=staff_top,
                staff_bot=staff_bot,
                threshold=threshold,
                x=xx,
            )
            >= dark_threshold
        ):
            right = xx
            empty_gap = 0
        else:
            empty_gap += 1
            if empty_gap > 8:
                break

    return right - left + 1


def _nonstaff_ink_density(
    pixels,
    *,
    width: int,
    staff_top: int,
    staff_bot: int,
    threshold: int,
    staff_line_rows: Sequence[int],
    x0: int,
    x1: int,
    staff_line_pad_px: int = 3,
) -> float:
    total = 0
    dark = 0
    for x in range(max(0, x0), min(width - 1, x1) + 1):
        for y in range(staff_top, staff_bot + 1):
            if _near_staff_line(y, staff_line_rows, pad_px=staff_line_pad_px):
                continue
            total += 1
            if pixels[x, y] < threshold:
                dark += 1
    return dark / total if total else 0.0


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
