"""Detect initial and changed key signatures in consumed handwritten score images.

This spike extends the earlier one-sharp detector with two deliberately bounded
capabilities: flat glyphs and multi-glyph signatures.  Prediction uses only the
image, staff geometry, and structural boundary mode.  Consumed expectations are
opened only after all predictions have been materialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from score2abc.chord_ocr.alignment import detect_barlines  # noqa: E402
from scripts.experiments import spike_consumed_key_state_detector as legacy  # noqa: E402

SCHEMA_VERSION = 1
OUTPUT_VERSION = "consumed_key_signature_detector_v3"
MODE_INITIAL = "initial"
MODE_CHANGE = "change"
FAMILY_SHARP = "sharp"
FAMILY_FLAT = "flat"
PREDICTION_UNKNOWN = "unknown"

SHARP_POSITIONS = (0.0, 1.5, -0.5, 1.0, 2.5, 0.5, 2.0)
FLAT_POSITIONS = (2.0, 0.5, 2.5, 1.0, 3.0, 1.5, 3.5)
POSITION_TOLERANCE_SPACES = 0.55
MIN_GLYPH_SCORE = 0.34
MAX_CHANGE_PREFIX_SPACES = 1.65
MIN_FRAGMENTED_CLEF_WIDTH_SPACES = 0.55
MIN_BROAD_SHARP_WIDTH_SPACES = 0.65
MIN_BROAD_SHARP_HEIGHT_SPACES = 4.0
MIN_INDEPENDENT_SHAPE_SCORE = 0.70
MIN_INDEPENDENT_SHAPE_MARGIN = 0.18
MIN_STRONG_SHIFTED_SHARP_SCORE = 0.90
MIN_STRONG_SHIFTED_SHARP_MARGIN = 0.70
INTERNAL_BOUNDARY_SEARCH_SPACES = 1.25
INTERNAL_SCAN_EDGE_FRACTION = 0.08
MAX_HINTED_SIGNATURE_START_SPACES = 3.6
MIN_HINTED_FLAT_SCORE = 0.55
MIN_HINTED_FLAT_MARGIN = 0.12
MIN_HINTED_WEAK_FLAT_SCORE = 0.64
MIN_HINTED_WEAK_FLAT_RIGHT_INK_RATIO = 0.30
MIN_HINTED_STRONG_FLAT_SCORE = 0.70
MIN_HINTED_STRONG_FLAT_MARGIN = 0.18
MIN_HINTED_STRONG_FLAT_RIGHT_INK_RATIO = 0.35
MIN_HINTED_TWO_FLAT_RIVALRY_MARGIN = 0.12
MIN_HINTED_KEY_CHANGE_BAR_GAP_SPACES = 0.50


@dataclass(frozen=True)
class EventInput:
    image: Path
    mode: str
    start_measure: int
    name: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def _work_slug(path: Path) -> str | None:
    """Return the work directory immediately below an ``out`` path segment."""
    parts = path.expanduser().resolve().parts
    for index, part in enumerate(parts[:-1]):
        if part == "out" and index + 1 < len(parts):
            return parts[index + 1]
    return None


def _staff_geometry(image: Image.Image) -> tuple[list[list[bool]], list[int], float, int]:
    mask, threshold = legacy._mask_from_image(image)
    row_counts = [sum(row) for row in mask]
    lines = legacy._staff_lines(row_counts, image.width, image.height)
    if len(lines) < 5:
        raise ValueError("image has no stable five-line staff")
    lines = lines[:5]
    spacing = statistics.mean(right - left for left, right in zip(lines, lines[1:], strict=False))
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("image has invalid staff spacing")
    return mask, lines, spacing, threshold


def _residual_mask(
    mask: Sequence[Sequence[bool]], staff_lines: Sequence[int], spacing: float
) -> list[list[bool]]:
    residual = [list(row) for row in mask]
    radius = max(1, round(spacing * 0.07))
    height = len(residual)
    width = len(residual[0]) if height else 0
    for line in staff_lines:
        for y in range(max(0, line - radius), min(len(residual), line + radius + 1)):
            for x in range(width):
                if not mask[y][x]:
                    continue
                above = any(
                    mask[probe_y][x]
                    for probe_y in range(max(0, y - radius - 2), max(0, y - radius))
                )
                below = any(
                    mask[probe_y][x]
                    for probe_y in range(min(height, y + radius + 1), min(height, y + radius + 3))
                )
                if not (above and below):
                    residual[y][x] = False
    return residual


def _clusters(values: Sequence[int], max_gap: int) -> list[tuple[int, int]]:
    if not values:
        return []
    groups: list[tuple[int, int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value - previous > max_gap:
            groups.append((start, previous))
            start = value
        previous = value
    groups.append((start, previous))
    return groups


def _ink_bbox(
    residual: Sequence[Sequence[bool]], left: int, right: int, top: int, bottom: int
) -> tuple[int, int, int, int] | None:
    points = [
        (x, y) for y in range(top, bottom + 1) for x in range(left, right + 1) if residual[y][x]
    ]
    if not points:
        return None
    return (
        min(x for x, _ in points),
        min(y for _, y in points),
        max(x for x, _ in points),
        max(y for _, y in points),
    )


def _glyph_regions(
    residual: Sequence[Sequence[bool]],
    *,
    x_start: int,
    x_end: int,
    top: int,
    bottom: int,
    spacing: float,
) -> list[tuple[int, int, int, int]]:
    width = len(residual[0]) if residual else 0
    x_start = max(0, min(width - 1, x_start))
    x_end = max(x_start, min(width - 1, x_end))
    minimum_column_ink = max(2, round(spacing * 0.10))
    active = [
        x
        for x in range(x_start, x_end + 1)
        if sum(residual[y][x] for y in range(top, bottom + 1)) >= minimum_column_ink
    ]
    regions = []
    # A handwritten flat's vertical stem and bulb can be separated by staff
    # suppression, while adjacent key glyphs remain farther apart.
    for left, right in _clusters(active, max_gap=max(1, round(spacing * 0.25))):
        bbox = _ink_bbox(
            residual,
            max(x_start, left - 1),
            min(x_end, right + 1),
            top,
            bottom,
        )
        if bbox is None:
            continue
        width_spaces = (bbox[2] - bbox[0] + 1) / spacing
        height_spaces = (bbox[3] - bbox[1] + 1) / spacing
        if width_spaces < 0.12 or height_spaces < 0.55:
            continue
        regions.append(bbox)
    return regions


def _candidate_regions(
    mask: list[list[bool]],
    residual: Sequence[Sequence[bool]],
    *,
    x_start: int,
    x_end: int,
    top: int,
    bottom: int,
    staff_lines: Sequence[int],
    spacing: float,
) -> list[tuple[int, int, int, int]]:
    """Build accidental-sized windows without requiring disconnected glyphs.

    Staff lines connect handwritten accidentals to one another and to the first
    note.  Vertical-ink tracks remain useful after that connection, so propose
    both individual tracks and short adjacent-track groups, then let the
    family/position sequence scorer decide which windows form a signature.
    """
    height = len(mask)
    staff_rows = legacy._line_rows(staff_lines, spacing, height)
    tracks = legacy._vertical_clusters(
        mask,
        x_start=x_start,
        x_end=x_end,
        y_start=top,
        y_end=bottom,
        staff_rows=staff_rows,
        spacing=spacing,
    )
    windows: set[tuple[int, int]] = set()
    for start in range(len(tracks)):
        for end in range(start, min(len(tracks), start + 3)):
            left = tracks[start][0]
            right = tracks[end][1]
            width_spaces = (right - left + 1) / spacing
            if width_spaces > 1.35:
                break
            if width_spaces < 0.08:
                continue
            if end > start:
                gap = tracks[end][0] - tracks[end - 1][1] - 1
                if gap > 0.55 * spacing:
                    break
            windows.add((left, right))

    regions = []
    margin = max(1, round(spacing * 0.08))
    for left, right in sorted(windows):
        # The vertical proposal for a flat sees its stem but not necessarily
        # the right-hand bulb. Keep a narrow view and a right-expanded view.
        for right_margin in (margin, round(spacing * 0.65)):
            bbox = _ink_bbox(
                residual,
                max(x_start, left - margin),
                min(x_end, right + right_margin),
                top,
                bottom,
            )
            if bbox is None:
                continue
            width_spaces = (bbox[2] - bbox[0] + 1) / spacing
            height_spaces = (bbox[3] - bbox[1] + 1) / spacing
            if 0.08 <= width_spaces <= 1.5 and height_spaces >= 0.55:
                regions.append(bbox)
    return sorted(set(regions), key=lambda bbox: (bbox[0], bbox[2], bbox[1], bbox[3]))


def _vertical_tracks(
    residual: Sequence[Sequence[bool]], bbox: tuple[int, int, int, int], spacing: float
) -> list[dict[str, float]]:
    left, top, right, bottom = bbox
    counts = [sum(residual[y][x] for y in range(top, bottom + 1)) for x in range(left, right + 1)]
    maximum = max(counts, default=0)
    if maximum <= 0:
        return []
    active = [left + index for index, value in enumerate(counts) if value >= max(3, maximum * 0.42)]
    tracks = []
    for track_left, track_right in _clusters(active, max_gap=1):
        rows = [
            y
            for y in range(top, bottom + 1)
            if any(residual[y][x] for x in range(track_left, track_right + 1))
        ]
        if not rows:
            continue
        tracks.append(
            {
                "left": float(track_left),
                "right": float(track_right),
                "center": (track_left + track_right) / 2,
                "top": float(min(rows)),
                "bottom": float(max(rows)),
                "support": max(counts[track_left - left : track_right - left + 1])
                / max(1, bottom - top + 1),
            }
        )
    return tracks


def _cross_rows(
    residual: Sequence[Sequence[bool]], bbox: tuple[int, int, int, int], spacing: float
) -> list[int]:
    left, top, right, bottom = bbox
    required = max(3, round((right - left + 1) * 0.48))
    rows = []
    for y in range(top, bottom + 1):
        longest = 0
        run = 0
        for x in range(left, right + 1):
            if residual[y][x]:
                run += 1
                longest = max(longest, run)
            else:
                run = 0
        if longest >= required:
            rows.append(y)
    return [
        round((start + end) / 2)
        for start, end in _clusters(rows, max_gap=max(1, round(spacing * 0.08)))
    ]


def _vertical_overlap(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    overlap = max(
        0.0, min(first["bottom"], second["bottom"]) - max(first["top"], second["top"]) + 1
    )
    union = max(first["bottom"], second["bottom"]) - min(first["top"], second["top"]) + 1
    return overlap / max(1.0, union)


def _glyph_features(
    residual: Sequence[Sequence[bool]],
    bbox: tuple[int, int, int, int],
    staff_lines: Sequence[int],
    spacing: float,
    glyph_id: str,
) -> dict[str, Any]:
    left, top, right, bottom = bbox
    width = right - left + 1
    height = bottom - top + 1
    tracks = _vertical_tracks(residual, bbox, spacing)
    crosses = _cross_rows(residual, bbox, spacing)
    ink = [(x, y) for y in range(top, bottom + 1) for x in range(left, right + 1) if residual[y][x]]
    strongest = max(tracks, key=lambda item: item["support"], default=None)
    right_ink = []
    if strongest is not None:
        right_ink = [(x, y) for x, y in ink if x > strongest["center"] + 0.08 * spacing]
    track_position = (
        (strongest["center"] - left) / max(1.0, width) if strongest is not None else 1.0
    )
    right_ink_ratio = len(right_ink) / max(1, len(ink))
    right_y = statistics.median([y for _, y in right_ink]) if right_ink else (top + bottom) / 2
    pair = sorted(tracks, key=lambda item: item["support"], reverse=True)[:2]
    pair = sorted(pair, key=lambda item: item["center"])
    overlap = _vertical_overlap(pair[0], pair[1]) if len(pair) == 2 else 0.0
    pair_gap = (pair[1]["center"] - pair[0]["center"]) / spacing if len(pair) == 2 else 0.0
    sharp_score = (
        0.28 * min(1.0, len(crosses) / 2)
        + 0.25 * min(1.0, len(tracks) / 2)
        + 0.27 * overlap
        + 0.20 * math.exp(-abs(pair_gap - 0.55))
    )
    flat_score = (
        0.34 * (strongest["support"] if strongest is not None else 0.0)
        + 0.30 * min(1.0, right_ink_ratio / 0.32)
        + 0.22 * max(0.0, 1.0 - track_position / 0.55)
        + 0.14 * math.exp(-abs(width / spacing - 0.55))
    )
    if len(tracks) >= 2 and overlap >= 0.65 and len(crosses) >= 2:
        flat_score *= 0.55
    if len(pair) == 2 and min(track["support"] for track in pair) >= 0.35 * max(
        track["support"] for track in pair
    ):
        flat_score *= 0.45
    sharp_anchor = statistics.median(crosses) if crosses else statistics.median([y for _, y in ink])
    flat_anchor = right_y
    return {
        "glyph_id": glyph_id,
        "bbox": {"left": left, "top": top, "right": right, "bottom": bottom},
        "width_staff_spaces": round(width / spacing, 6),
        "height_staff_spaces": round(height / spacing, 6),
        "tracks": [{key: round(value, 6) for key, value in track.items()} for track in tracks],
        "cross_rows_y_px": crosses,
        "vertical_overlap": round(overlap, 6),
        "right_ink_ratio": round(right_ink_ratio, 6),
        "strongest_track_x_fraction": round(track_position, 6),
        "family_scores": {FAMILY_SHARP: round(sharp_score, 6), FAMILY_FLAT: round(flat_score, 6)},
        "anchor_positions": {
            FAMILY_SHARP: round((sharp_anchor - staff_lines[0]) / spacing, 6),
            FAMILY_FLAT: round((flat_anchor - staff_lines[0]) / spacing, 6),
        },
    }


def _clef_right(glyphs: Sequence[Mapping[str, Any]], boundary: int, spacing: float) -> int | None:
    candidates = [
        glyph
        for glyph in glyphs
        if int(glyph["bbox"]["left"]) <= boundary + 1.1 * spacing
        and float(glyph["height_staff_spaces"]) >= 2.6
        and float(glyph["width_staff_spaces"]) >= 0.55
    ]
    if not candidates:
        return None
    return max(int(glyph["bbox"]["right"]) for glyph in candidates)


def _initial_clef_right(
    mask: list[list[bool]],
    staff_lines: Sequence[int],
    spacing: float,
    boundary: int,
) -> int | None:
    """Locate the first wide, tall symbol after the opening staff bar."""
    height = len(mask)
    width = len(mask[0]) if height else 0
    top = max(0, round(staff_lines[0] - spacing))
    bottom = min(height - 1, round(staff_lines[-1] + spacing))
    tracks = legacy._vertical_clusters(
        mask,
        x_start=min(width - 1, boundary + 1),
        x_end=min(width - 1, boundary + round(spacing * 3.5)),
        y_start=top,
        y_end=bottom,
        staff_rows=legacy._line_rows(staff_lines, spacing, height),
        spacing=spacing,
    )
    for left, right, support in tracks:
        width_spaces = (right - left + 1) / spacing
        if MIN_FRAGMENTED_CLEF_WIDTH_SPACES <= width_spaces <= 3.2 and support >= 0.22:
            return right
    return None


def _sharp_shape_eligible(glyph: Mapping[str, Any]) -> bool:
    if len(glyph["tracks"]) >= 2:
        return True
    return (
        len(glyph["cross_rows_y_px"]) >= 2
        and float(glyph["width_staff_spaces"]) >= MIN_BROAD_SHARP_WIDTH_SPACES
        and float(glyph["height_staff_spaces"]) >= MIN_BROAD_SHARP_HEIGHT_SPACES
    )


def _independent_shape_representatives(
    glyphs: Sequence[Mapping[str, Any]],
    family: str,
    *,
    minimum_score: float = MIN_INDEPENDENT_SHAPE_SCORE,
    minimum_margin: float = MIN_INDEPENDENT_SHAPE_MARGIN,
) -> list[Mapping[str, Any]]:
    """Collapse overlapping hypotheses into independently supported glyphs."""
    other_family = FAMILY_FLAT if family == FAMILY_SHARP else FAMILY_SHARP
    eligible = sorted(
        [
            glyph
            for glyph in glyphs
            if float(glyph["family_scores"][family]) >= minimum_score
            and float(glyph["family_scores"][family]) - float(glyph["family_scores"][other_family])
            >= minimum_margin
            and (family != FAMILY_SHARP or _sharp_shape_eligible(glyph))
        ],
        key=lambda glyph: (int(glyph["bbox"]["left"]), int(glyph["bbox"]["right"])),
    )
    groups: list[list[Mapping[str, Any]]] = []
    for glyph in eligible:
        left = int(glyph["bbox"]["left"])
        if groups and left <= max(int(item["bbox"]["right"]) for item in groups[-1]):
            groups[-1].append(glyph)
        else:
            groups.append([glyph])
    return [max(group, key=lambda glyph: float(glyph["family_scores"][family])) for group in groups]


def _shape_support(glyphs: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    support = {}
    for family in (FAMILY_SHARP, FAMILY_FLAT):
        representatives = _independent_shape_representatives(glyphs, family)
        support[family] = {
            "count": len(representatives),
            "glyph_ids": [str(glyph["glyph_id"]) for glyph in representatives],
            "minimum_score": MIN_INDEPENDENT_SHAPE_SCORE,
            "minimum_family_margin": MIN_INDEPENDENT_SHAPE_MARGIN,
        }
    return support


def _strong_shifted_two_sharps(
    glyphs: Sequence[Mapping[str, Any]], boundary: int, spacing: float
) -> dict[str, Any] | None:
    """Recognize two unmistakable sharps when handwritten placement is nonstandard."""
    representatives = _independent_shape_representatives(
        glyphs,
        FAMILY_SHARP,
        minimum_score=MIN_STRONG_SHIFTED_SHARP_SCORE,
        minimum_margin=MIN_STRONG_SHIFTED_SHARP_MARGIN,
    )
    if len(representatives) != 2:
        return None
    first_left = int(representatives[0]["bbox"]["left"])
    if (first_left - boundary) / spacing > MAX_CHANGE_PREFIX_SPACES:
        return None
    return {
        "family": FAMILY_SHARP,
        "count": 2,
        "fifths": 2,
        "score": round(
            sum(float(glyph["family_scores"][FAMILY_SHARP]) for glyph in representatives),
            6,
        ),
        "glyph_ids": [str(glyph["glyph_id"]) for glyph in representatives],
    }


def _boundary_near(
    mask: list[list[bool]],
    staff_lines: Sequence[int],
    spacing: float,
    hint_x: int,
) -> tuple[int, dict[str, Any]]:
    """Resolve a structural bar near a full-system x-coordinate hint."""
    height = len(mask)
    width = len(mask[0]) if height else 0
    if width <= 0 or not 0 <= hint_x < width:
        raise ValueError(f"boundary hint must be within image width: {hint_x}")
    top = max(0, round(staff_lines[0] - spacing * 0.4))
    bottom = min(height - 1, round(staff_lines[-1] + spacing * 0.4))
    staff_rows = legacy._line_rows(staff_lines, spacing, height)
    radius = max(4, round(spacing * INTERNAL_BOUNDARY_SEARCH_SPACES))
    left = max(0, hint_x - radius)
    right = min(width - 1, hint_x + radius)
    usable_rows = [y for y in range(top, bottom + 1) if y not in staff_rows]
    support = [
        sum(mask[y][x] for y in usable_rows) / max(1, len(usable_rows))
        for x in range(left, right + 1)
    ]
    strong = [
        left + offset
        for offset, value in enumerate(support)
        if value >= legacy.BOUNDARY_TRACK_SUPPORT
    ]
    tracks = [
        {
            "left": start,
            "right": end,
            "center": round((start + end) / 2, 3),
            "support": round(max(support[start - left : end - left + 1]), 6),
        }
        for start, end in legacy._cluster_indices(strong)
    ]
    pairs = []
    for first, second in zip(tracks, tracks[1:], strict=False):
        gap = (float(second["center"]) - float(first["center"])) / spacing
        if legacy.MIN_DOUBLE_BAR_GAP_SPACES <= gap <= legacy.MAX_DOUBLE_BAR_GAP_SPACES:
            pairs.append(
                (
                    abs((float(first["center"]) + float(second["center"])) / 2 - hint_x),
                    first,
                    second,
                    gap,
                )
            )
    if pairs:
        _, first, second, gap = min(pairs, key=lambda item: item[0])
        boundary = int(second["right"])
        return boundary, {
            "method": "hinted_full_system_vertical_tracks",
            "style": "double_bar",
            "tracks": tracks,
            "bbox": {"left": int(first["left"]), "right": boundary},
            "support": float(first["support"]),
            "double_bar_gap_staff_spaces": round(gap, 6),
            "hint_x_px": hint_x,
        }
    if tracks:
        nearest = min(tracks, key=lambda item: abs(float(item["center"]) - hint_x))
        boundary = int(nearest["right"])
        return boundary, {
            "method": "hinted_full_system_vertical_tracks",
            "style": "single_bar",
            "tracks": tracks,
            "bbox": {"left": int(nearest["left"]), "right": boundary},
            "support": float(nearest["support"]),
            "double_bar_gap_staff_spaces": None,
            "hint_x_px": hint_x,
        }
    return hint_x, {
        "method": "hinted_full_system_fallback",
        "style": "unknown",
        "tracks": [],
        "bbox": {"left": hint_x, "right": hint_x},
        "support": 0.0,
        "double_bar_gap_staff_spaces": None,
        "hint_x_px": hint_x,
    }


def _left_groups(
    glyphs: Sequence[Mapping[str, Any]], spacing: float
) -> list[list[Mapping[str, Any]]]:
    groups: list[list[Mapping[str, Any]]] = []
    for glyph in sorted(glyphs, key=lambda item: int(item["bbox"]["left"])):
        left = int(glyph["bbox"]["left"])
        if groups and left - int(groups[-1][0]["bbox"]["left"]) <= 0.35 * spacing:
            groups[-1].append(glyph)
        else:
            groups.append([glyph])
    return groups


def _hinted_two_flat_signature(
    glyphs: Sequence[Mapping[str, Any]], boundary: int, spacing: float
) -> dict[str, Any] | None:
    """Recover two left-stem flat shapes after an internal double bar."""
    groups = [
        group
        for group in _left_groups(glyphs, spacing)
        if (int(group[0]["bbox"]["left"]) - boundary) / spacing <= MAX_HINTED_SIGNATURE_START_SPACES
    ]
    selected = []
    for group in groups:
        glyph = max(group, key=lambda item: float(item["family_scores"][FAMILY_FLAT]))
        flat_score = float(glyph["family_scores"][FAMILY_FLAT])
        sharp_score = float(glyph["family_scores"][FAMILY_SHARP])
        if flat_score < MIN_HINTED_FLAT_SCORE or flat_score - sharp_score < MIN_HINTED_FLAT_MARGIN:
            continue
        if float(glyph["strongest_track_x_fraction"]) > 0.35:
            continue
        if float(glyph["right_ink_ratio"]) < 0.35:
            continue
        selected.append(glyph)
    if len(selected) != 2:
        return None
    first_left = int(selected[0]["bbox"]["left"])
    second_left = int(selected[1]["bbox"]["left"])
    gap = (second_left - first_left) / spacing
    anchor_drop = float(selected[0]["anchor_positions"][FAMILY_FLAT]) - float(
        selected[1]["anchor_positions"][FAMILY_FLAT]
    )
    if not 0.8 <= gap <= 2.5 or not 0.8 <= anchor_drop <= 3.2:
        return None
    return {
        "family": FAMILY_FLAT,
        "count": 2,
        "fifths": -2,
        "score": round(
            sum(float(glyph["family_scores"][FAMILY_FLAT]) for glyph in selected),
            6,
        ),
        "glyph_ids": [str(glyph["glyph_id"]) for glyph in selected],
    }


def _hinted_one_strong_two_flat_signature(
    signatures: Sequence[Mapping[str, Any]],
    glyphs: Sequence[Mapping[str, Any]],
    boundary: int,
    spacing: float,
) -> dict[str, Any] | None:
    """Recover a two-flat sequence whose second glyph is partially merged.

    This consumed-data fallback is intentionally stricter than the generic
    hinted-flat path: the positional flat family must beat a same-count sharp
    rival, both glyphs must retain weak flat morphology, and at least one must
    retain strong independent flat morphology.
    """
    flat_candidates = [
        signature
        for signature in signatures
        if signature["family"] == FAMILY_FLAT and int(signature["count"]) == 2
    ]
    rival_candidates = [
        signature
        for signature in signatures
        if signature["family"] != FAMILY_FLAT and int(signature["count"]) == 2
    ]
    if not flat_candidates or not rival_candidates:
        return None
    selected = max(flat_candidates, key=lambda item: float(item["score"]))
    rival = max(rival_candidates, key=lambda item: float(item["score"]))
    if float(selected["score"]) - float(rival["score"]) < MIN_HINTED_TWO_FLAT_RIVALRY_MARGIN:
        return None

    glyph_by_id = {str(glyph["glyph_id"]): glyph for glyph in glyphs}
    selected_glyphs = [glyph_by_id[str(glyph_id)] for glyph_id in selected["glyph_ids"]]
    first_left = int(selected_glyphs[0]["bbox"]["left"])
    if (first_left - boundary) / spacing > MAX_HINTED_SIGNATURE_START_SPACES:
        return None

    weak_support = []
    strong_support = []
    for glyph in selected_glyphs:
        flat_score = float(glyph["family_scores"][FAMILY_FLAT])
        sharp_score = float(glyph["family_scores"][FAMILY_SHARP])
        margin = flat_score - sharp_score
        track_fraction = float(glyph["strongest_track_x_fraction"])
        right_ink_ratio = float(glyph["right_ink_ratio"])
        weak_support.append(
            flat_score >= MIN_HINTED_WEAK_FLAT_SCORE
            and margin >= MIN_HINTED_FLAT_MARGIN
            and track_fraction <= 0.35
            and right_ink_ratio >= MIN_HINTED_WEAK_FLAT_RIGHT_INK_RATIO
        )
        strong_support.append(
            flat_score >= MIN_HINTED_STRONG_FLAT_SCORE
            and margin >= MIN_HINTED_STRONG_FLAT_MARGIN
            and track_fraction <= 0.35
            and right_ink_ratio >= MIN_HINTED_STRONG_FLAT_RIGHT_INK_RATIO
        )
    if not all(weak_support) or not any(strong_support):
        return None

    second_left = int(selected_glyphs[1]["bbox"]["left"])
    gap = (second_left - first_left) / spacing
    anchor_drop = float(selected_glyphs[0]["anchor_positions"][FAMILY_FLAT]) - float(
        selected_glyphs[1]["anchor_positions"][FAMILY_FLAT]
    )
    if not 0.8 <= gap <= 2.5 or not 0.8 <= anchor_drop <= 3.2:
        return None
    return dict(selected)


def _hinted_cancellation_two_sharps(
    signatures: Sequence[Mapping[str, Any]],
    glyphs: Sequence[Mapping[str, Any]],
    boundary: int,
    spacing: float,
    shape_support: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Accept a positional two-sharp sequence after a separate cancellation glyph."""
    candidates = [
        signature
        for signature in signatures
        if signature["family"] == FAMILY_SHARP
        and int(signature["count"]) == 2
        and float(signature["score"]) >= 0.85
    ]
    if not candidates or int(shape_support[FAMILY_SHARP]["count"]) < 1:
        return None
    selected = max(candidates, key=lambda item: float(item["score"]))
    selected_glyphs = [
        next(glyph for glyph in glyphs if glyph["glyph_id"] == glyph_id)
        for glyph_id in selected["glyph_ids"]
    ]
    if int(selected_glyphs[0]["bbox"]["right"]) >= int(selected_glyphs[1]["bbox"]["left"]):
        return None
    anchor_rise = float(selected_glyphs[1]["anchor_positions"][FAMILY_SHARP]) - float(
        selected_glyphs[0]["anchor_positions"][FAMILY_SHARP]
    )
    if not 0.5 <= anchor_rise <= 2.2:
        return None
    first_left = int(selected_glyphs[0]["bbox"]["left"])
    prefix_limit = first_left - round(spacing * 0.75)
    prefix_groups = [
        group
        for group in _left_groups(glyphs, spacing)
        if boundary < int(group[0]["bbox"]["left"]) <= prefix_limit
    ]
    if len(prefix_groups) != 1:
        return None
    prefix = max(
        prefix_groups[0],
        key=lambda glyph: max(float(value) for value in glyph["family_scores"].values()),
    )
    return dict(selected), {
        "kind": "natural_cancellation_prefix",
        "glyph_ids": [str(prefix["glyph_id"])],
        "bbox": dict(prefix["bbox"]),
    }


def _best_signature(
    glyphs: Sequence[Mapping[str, Any]], family: str, spacing: float
) -> dict[str, Any] | None:
    expected = SHARP_POSITIONS if family == FAMILY_SHARP else FLAT_POSITIONS
    max_width = 1.2 if family == FAMILY_SHARP else 1.45
    eligible = sorted(
        [
            glyph
            for glyph in glyphs
            if float(glyph["family_scores"][family]) >= MIN_GLYPH_SCORE
            and float(glyph["width_staff_spaces"]) <= max_width
            and (family != FAMILY_SHARP or _sharp_shape_eligible(glyph))
        ],
        key=lambda glyph: (int(glyph["bbox"]["left"]), int(glyph["bbox"]["right"])),
    )

    candidates: list[tuple[int, float, list[Mapping[str, Any]]]] = []

    def visit(
        selected: list[Mapping[str, Any]],
        score: float,
        expected_index: int,
        eligible_index: int,
        previous_right: int | None,
    ) -> None:
        if selected:
            candidates.append((len(selected), score, list(selected)))
        if expected_index >= len(expected):
            return
        for index in range(eligible_index, len(eligible)):
            glyph = eligible[index]
            left = int(glyph["bbox"]["left"])
            right = int(glyph["bbox"]["right"])
            if previous_right is not None:
                if left <= previous_right:
                    continue
                if left - previous_right > 1.35 * spacing:
                    continue
            error = abs(float(glyph["anchor_positions"][family]) - expected[expected_index])
            if error > POSITION_TOLERANCE_SPACES:
                continue
            visit(
                [*selected, glyph],
                score + float(glyph["family_scores"][family]) - 0.25 * error,
                expected_index + 1,
                index + 1,
                right,
            )

    visit([], 0.0, 0, 0, None)
    if not candidates:
        return None
    count, score, selected = max(candidates, key=lambda item: (item[0], item[1]))
    return {
        "family": family,
        "count": count,
        "fifths": count if family == FAMILY_SHARP else -count,
        "score": round(score, 6),
        "glyph_ids": [str(glyph["glyph_id"]) for glyph in selected],
    }


def detect_signature(
    image_path: Path,
    *,
    mode: str,
    boundary_hint_x: int | None = None,
) -> dict[str, Any]:
    if mode not in {MODE_INITIAL, MODE_CHANGE}:
        raise ValueError(f"unsupported key-signature mode: {mode}")
    if boundary_hint_x is not None and mode != MODE_CHANGE:
        raise ValueError("boundary hints are supported only in change mode")
    image_path = image_path.expanduser().resolve()
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    mask, staff_lines, spacing, threshold = _staff_geometry(image)
    residual = _residual_mask(mask, staff_lines, spacing)
    boundary, boundary_features = (
        _boundary_near(mask, staff_lines, spacing, boundary_hint_x)
        if boundary_hint_x is not None
        else legacy._find_boundary(mask, staff_lines, spacing)
    )
    top = max(0, round(staff_lines[0] - spacing * 1.0))
    bottom = min(image.height - 1, round(staff_lines[-1] + spacing * 1.0))
    clef_right = (
        _initial_clef_right(mask, staff_lines, spacing, boundary) if mode == MODE_INITIAL else None
    )
    gate_passed = (
        boundary_features["style"] == "double_bar"
        if mode == MODE_CHANGE
        else clef_right is not None
    )
    search_start = (
        clef_right + max(1, round(spacing * 0.04))
        if clef_right is not None
        else boundary + max(1, round(spacing * 0.12))
    )
    search_end = min(image.width - 1, search_start + round(spacing * 4.8))
    regions = _candidate_regions(
        mask,
        residual,
        x_start=search_start,
        x_end=search_end,
        top=top,
        bottom=bottom,
        staff_lines=staff_lines,
        spacing=spacing,
    )
    glyphs = [
        _glyph_features(residual, bbox, staff_lines, spacing, f"g{index:03d}")
        for index, bbox in enumerate(regions, start=1)
    ]
    signatures = [
        candidate
        for family in (FAMILY_SHARP, FAMILY_FLAT)
        if (candidate := _best_signature(glyphs, family, spacing))
    ]
    signatures.sort(key=lambda item: (int(item["count"]), float(item["score"])), reverse=True)
    selected = signatures[0] if gate_passed and signatures else None
    selection_method = "standard_position_sequence" if selected is not None else None
    rejected_standard_signature = None
    suppressed_internal_default_signature = None
    cancellation_prefix = None
    if len(signatures) >= 2 and signatures[0]["count"] == signatures[1]["count"]:
        if float(signatures[0]["score"]) - float(signatures[1]["score"]) < 0.18:
            selected = None
            selection_method = None
    if selected is not None and mode == MODE_CHANGE:
        first_selected = next(
            glyph for glyph in glyphs if glyph["glyph_id"] == selected["glyph_ids"][0]
        )
        prefix_spaces = (int(first_selected["bbox"]["left"]) - boundary) / spacing
        if prefix_spaces > MAX_CHANGE_PREFIX_SPACES:
            selected = None
            selection_method = None
    shape_support = _shape_support(glyphs)
    if selected is not None and mode == MODE_CHANGE:
        selected_family = str(selected["family"])
        if int(shape_support[selected_family]["count"]) < int(selected["count"]):
            rejected_standard_signature = {
                **selected,
                "reason": "insufficient_independent_shape_support",
                "independent_shape_count": int(shape_support[selected_family]["count"]),
            }
            selected = None
            selection_method = None
        else:
            selection_method = "standard_position_sequence_shape_validated"
    if selected is None and gate_passed and mode == MODE_CHANGE:
        shifted_sharps = _strong_shifted_two_sharps(glyphs, boundary, spacing)
        if shifted_sharps is not None:
            selected = shifted_sharps
            selection_method = "strong_shifted_two_sharp_shape"
    if selected is not None and boundary_hint_x is not None:
        suppressed_internal_default_signature = {
            **selected,
            "selection_method": selection_method,
            "reason": "internal_scan_requires_validated_hinted_pattern",
        }
        selected = None
        selection_method = None
    hinted_change_gate = (
        boundary_hint_x is not None
        and float(boundary_features.get("double_bar_gap_staff_spaces") or 0.0)
        >= MIN_HINTED_KEY_CHANGE_BAR_GAP_SPACES
    )
    if selected is None and gate_passed and hinted_change_gate:
        cancellation_sharps = _hinted_cancellation_two_sharps(
            signatures,
            glyphs,
            boundary,
            spacing,
            shape_support,
        )
        if cancellation_sharps is not None:
            selected, cancellation_prefix = cancellation_sharps
            selection_method = "hinted_cancellation_prefix_two_sharp_sequence"
    if selected is None and gate_passed and hinted_change_gate:
        hinted_flats = _hinted_two_flat_signature(glyphs, boundary, spacing)
        if hinted_flats is not None:
            selected = hinted_flats
            selection_method = "hinted_two_flat_shape_sequence"
    if selected is None and gate_passed and boundary_hint_x is not None:
        hinted_flats = _hinted_one_strong_two_flat_signature(
            signatures,
            glyphs,
            boundary,
            spacing,
        )
        if hinted_flats is not None:
            selected = hinted_flats
            selection_method = "hinted_one_strong_two_flat_shape_sequence"
    return {
        "schema_version": SCHEMA_VERSION,
        "input": {"path": str(image_path), "sha256": _sha256(image_path)},
        "mode": mode,
        "threshold": threshold,
        "staff_lines_y_px": staff_lines,
        "staff_spacing_px": round(spacing, 6),
        "structural_boundary": {"x_px": boundary, **boundary_features},
        "clef_right_px": clef_right,
        "search_region": {"left_px": search_start, "right_px": search_end},
        "glyphs": glyphs,
        "signature_candidates": signatures,
        "independent_shape_support": shape_support,
        "rejected_standard_signature": rejected_standard_signature,
        "suppressed_internal_default_signature": suppressed_internal_default_signature,
        "gate_passed": gate_passed,
        "predicted_signature_family": selected["family"] if selected else None,
        "accidental_count": int(selected["count"]) if selected else 0,
        "fifths": int(selected["fifths"]) if selected else None,
        "predicted_change": selected["family"] if selected else PREDICTION_UNKNOWN,
        "selected_glyph_ids": list(selected["glyph_ids"]) if selected else [],
        "selection_method": selection_method,
        "cancellation_prefix": cancellation_prefix,
        "truth_used_for_prediction": False,
    }


def scan_internal_change_signatures(image_path: Path) -> dict[str, Any]:
    """Evaluate every detected interior bar while preserving full-system context."""
    image_path = image_path.expanduser().resolve()
    with Image.open(image_path) as opened:
        width = opened.width
    barline_fractions = detect_barlines(image_path)
    predictions = []
    seen_boundaries: set[int] = set()
    for fraction in barline_fractions:
        if fraction <= INTERNAL_SCAN_EDGE_FRACTION or fraction >= 1 - INTERNAL_SCAN_EDGE_FRACTION:
            continue
        hint_x = round(fraction * width)
        prediction = detect_signature(
            image_path,
            mode=MODE_CHANGE,
            boundary_hint_x=hint_x,
        )
        boundary = int(prediction["structural_boundary"]["x_px"])
        if boundary in seen_boundaries:
            continue
        seen_boundaries.add(boundary)
        predictions.append(prediction)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "full_system_internal_key_change_scan",
        "input": {"path": str(image_path), "sha256": _sha256(image_path)},
        "detected_barlines_x_fraction": [round(value, 12) for value in barline_fractions],
        "prediction_count": len(predictions),
        "double_bar_count": sum(
            prediction["structural_boundary"]["style"] == "double_bar" for prediction in predictions
        ),
        "hit_count": sum(prediction["fifths"] is not None for prediction in predictions),
        "predictions": predictions,
        "hits": [prediction for prediction in predictions if prediction["fifths"] is not None],
        "truth_used_for_prediction": False,
    }


def _draw_overlay(prediction: Mapping[str, Any], path: Path) -> None:
    with Image.open(str(prediction["input"]["path"])) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    boundary = int(prediction["structural_boundary"]["x_px"])
    draw.line((boundary, 0, boundary, image.height - 1), fill=(40, 100, 220), width=2)
    selected = set(prediction["selected_glyph_ids"])
    for glyph in prediction["glyphs"]:
        bbox = glyph["bbox"]
        color = (20, 160, 60) if glyph["glyph_id"] in selected else (230, 150, 30)
        draw.rectangle(
            (bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]), outline=color, width=2
        )
        draw.text((bbox["left"], max(0, bbox["top"] - 11)), glyph["glyph_id"], fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def analyze_events(
    events: Sequence[EventInput],
    out_dir: Path,
    *,
    expected_fifths: Mapping[str, int | None] | None = None,
) -> dict[str, Any]:
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = []
    for event in events:
        prediction = detect_signature(event.image, mode=event.mode)
        prediction["name"] = event.name
        prediction["start_measure"] = event.start_measure
        prediction["slug"] = _work_slug(event.image)
        overlay = out_dir / f"{event.name}_key_signature_overlay.png"
        _draw_overlay(prediction, overlay)
        prediction["overlay_path"] = str(overlay)
        predictions.append(prediction)

    context_hints = {
        "schema_version": SCHEMA_VERSION,
        "source": "automatic_visual_key_signature_detector_v3",
        "truth_used": False,
        "events": [
            {
                "start_measure": int(prediction["start_measure"]),
                "key_hint": {"fifths": int(prediction["fifths"])},
                "source": {
                    "kind": "automatic_visual_key_signature",
                    "slug": prediction["slug"],
                    "image": prediction["input"]["path"],
                    "sha256": prediction["input"]["sha256"],
                    "mode": prediction["mode"],
                    "glyph_ids": prediction["selected_glyph_ids"],
                },
            }
            for prediction in predictions
            if prediction["fifths"] is not None
        ],
    }
    context_path = out_dir / "context_hints.json"
    context_path.write_text(
        json.dumps(context_hints, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Evaluation is intentionally separated from the completed prediction list.
    evaluation = None
    if expected_fifths is not None:
        rows = [
            {
                "name": prediction["name"],
                "predicted_fifths": prediction["fifths"],
                "expected_fifths": expected_fifths.get(str(prediction["name"])),
                "match": prediction["fifths"] == expected_fifths.get(str(prediction["name"])),
            }
            for prediction in predictions
            if str(prediction["name"]) in expected_fifths
        ]
        evaluation = {
            "predictions_materialized_before_expectations": True,
            "compared_count": len(rows),
            "matches": sum(bool(row["match"]) for row in rows),
            "accuracy": sum(bool(row["match"]) for row in rows) / len(rows) if rows else None,
            "rows": rows,
        }
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "consumed_key_signature_detector_spike",
        "output_version": OUTPUT_VERSION,
        "truth_used_for_prediction": False,
        "protocol": {
            "supported_families": [FAMILY_SHARP, FAMILY_FLAT],
            "supported_counts": [1, 2, 3, 4, 5, 6, 7],
            "boundary_modes": [MODE_INITIAL, MODE_CHANGE],
            "position_tolerance_staff_spaces": POSITION_TOLERANCE_SPACES,
            "max_change_prefix_staff_spaces": MAX_CHANGE_PREFIX_SPACES,
            "consumed_evaluation_only": expected_fifths is not None,
        },
        "predictions": predictions,
        "context_hints": context_hints,
        "evaluation": evaluation,
        "artifacts": {
            "report_json": str(out_dir / "report.json"),
            "context_hints": str(context_path),
        },
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def scan_change_inputs(image_paths: Sequence[Path]) -> dict[str, Any]:
    predictions = []
    for path in sorted({item.expanduser().resolve() for item in image_paths}):
        prediction = detect_signature(path, mode=MODE_CHANGE)
        predictions.append(
            {
                "input": prediction["input"],
                "boundary_style": prediction["structural_boundary"]["style"],
                "fifths": prediction["fifths"],
                "selected_glyph_ids": prediction["selected_glyph_ids"],
                "truth_used_for_prediction": False,
            }
        )
    return {
        "input_count": len(predictions),
        "double_bar_count": sum(row["boundary_style"] == "double_bar" for row in predictions),
        "hit_count": sum(row["fifths"] is not None for row in predictions),
        "hits": [row for row in predictions if row["fifths"] is not None],
        "truth_used_for_prediction": False,
    }


def _scan_paths(directories: Sequence[Path]) -> list[Path]:
    paths = []
    for raw_directory in directories:
        directory = raw_directory.expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        staff_paths = sorted(directory.rglob("measure_*_staff.png"))
        paths.extend(staff_paths or sorted(directory.rglob("measure_*.png")))
    if directories and not paths:
        raise ValueError("change scan directories contain no measure images")
    return sorted(set(paths))


def _event(value: str) -> EventInput:
    parts = value.rsplit("|", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("event must be PATH|MODE|START_MEASURE|NAME")
    path, mode, measure, name = parts
    if mode not in {MODE_INITIAL, MODE_CHANGE}:
        raise argparse.ArgumentTypeError("event mode must be initial or change")
    try:
        start_measure = int(measure)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("event start measure must be an integer") from exc
    return EventInput(Path(path), mode, start_measure, name)


def _expected_fifths(value: str) -> tuple[str, int | None]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected fifths must be NAME=INTEGER or NAME=none")
    name, raw_fifths = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("expected fifths name cannot be empty")
    if raw_fifths.lower() == "none":
        return name, None
    try:
        return name, int(raw_fifths)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected fifths must be NAME=INTEGER or NAME=none"
        ) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", action="append", type=_event, required=True)
    parser.add_argument("--expected-fifths", action="append", type=_expected_fifths, default=[])
    parser.add_argument("--change-scan-dir", action="append", type=Path, default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        expected = dict(args.expected_fifths)
        if len(expected) != len(args.expected_fifths):
            raise ValueError("expected fifths names must be unique")
        report = analyze_events(
            args.event,
            args.out_dir,
            expected_fifths=expected or None,
        )
        scan_paths = _scan_paths(args.change_scan_dir)
        if scan_paths:
            sweep = scan_change_inputs(scan_paths)
            sweep_path = args.out_dir.expanduser().resolve() / "change_control_sweep.json"
            sweep_path.write_text(
                json.dumps(sweep, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report["change_control_sweep"] = sweep
            report["artifacts"]["change_control_sweep"] = str(sweep_path)
            Path(report["artifacts"]["report_json"]).write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(report["artifacts"]["report_json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
