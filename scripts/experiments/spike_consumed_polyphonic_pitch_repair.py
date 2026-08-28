"""Run a consumed-evidence, polyphonic pitch-repair postprocessor spike.

The spike replays the frozen candidate scores and selector configuration without
touching the sealed inference or evaluation artifacts.  It evaluates an
x-only NMS baseline and a small, explicit 2D NMS sweep.  Truth is opened only
after natural-context and optional context-hint predictions are materialized.

This is postmortem evidence: the automatic lane replays any key hint frozen in
the inference rows, the context lane requires an explicitly supplied
human/external hint file, and the oracle lane is diagnostic only. No MusicXML
is read here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from score2abc.utils.imaging import estimate_ink_threshold  # noqa: E402

SCHEMA_VERSION = 1
DEFAULT_Y_SEPARATION_GRID = (0.5, 1.0, 1.5, 2.0)
DEFAULT_RECOVERY_MIN_Y_GAP_GRID = (1.0, 1.5)
DEFAULT_RECOVERY_SCORE_RATIO_GRID = (0.5, 0.75)
DEFAULT_RECOVERY_MAX_Y_GAP_STAFF_SPACES = 3.0
DEFAULT_STEM_SCORE_GRID = (0.8, 0.9, 1.0)
EDGE_SAFE_STEM_DYAD_CONFIG_ID = "edge_safe_stem_dyad__y_1_3__ratio_0.5__stem_0.55__leading_x_1"
EDGE_SAFE_STEM_DYAD_PARAMETERS = {
    "minimum_y_gap_staff_spaces": 1.0,
    "maximum_y_gap_staff_spaces": 3.0,
    "minimum_score_ratio": 0.5,
    "minimum_stem_score": 0.55,
    "minimum_group_x_staff_spaces": 1.0,
}
EDGE_SAFE_STEM_MULTIHEAD_CONFIG_ID = (
    "edge_safe_stem_multihead__y_1_3__ratio_0.5__stem_0.55__leading_x_1__cap_2"
)
EDGE_SAFE_STEM_MULTIHEAD_PARAMETERS = {
    "minimum_y_gap_staff_spaces": 1.0,
    "maximum_y_gap_staff_spaces": 3.0,
    "minimum_score_ratio": 0.5,
    "minimum_stem_score": 0.55,
    "minimum_group_x_staff_spaces": 1.0,
    "maximum_recovered_heads_per_group": 2,
}
STEM_SEARCH_SIDE_PIXELS = 5
STEM_SEARCH_EXTRA_SPACES = 4.0
STEM_MAX_GAP_PIXELS = 3
STEM_STAFF_SUPPRESSION_RADIUS_PIXELS = 1
STEM_PATH_HALF_WIDTH_PIXELS = 1
LANE_AUTOMATIC = "automatic_natural_context"
LANE_CONTEXT = "context_hint"
LANE_ORACLE = "diagnostic_oracle"

_NATURAL_MIDI = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_DIATONIC_LETTERS = ("C", "D", "E", "F", "G", "A", "B")
_SHARP_ORDER = ("F", "C", "G", "D", "A", "E", "B")
_FLAT_ORDER = ("B", "E", "A", "D", "G", "C", "F")
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pin(path: Path, data: bytes) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_bytes(data),
        "bytes": len(data),
    }


def _read_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.expanduser().resolve()
    data = path.read_bytes()
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {label} JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value, _pin(path, data)


def _read_jsonl(path: Path, *, label: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = path.expanduser().resolve()
    data = path.read_bytes()
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(data.decode("utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid {label} JSONL at line {line_number}: {path}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label} row {line_number} must be an object: {path}")
        rows.append(value)
    if not rows:
        raise ValueError(f"{label} JSONL is empty: {path}")
    return rows, _pin(path, data)


def selector_config_from_model(model: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the frozen selector knobs needed to replay candidate selection."""
    replay = model.get("replay")
    selector = replay.get("selector") if isinstance(replay, Mapping) else None
    if not isinstance(selector, Mapping):
        selector = model.get("selector")
    if not isinstance(selector, Mapping):
        raise ValueError("Frozen model has no replay.selector configuration")
    required = ("threshold", "nms_x_spaces", "minimum_selected_count", "maximum_selected_count")
    missing = [key for key in required if key not in selector]
    if missing:
        raise ValueError(f"Frozen selector is missing: {', '.join(missing)}")
    config = {
        "threshold": float(selector["threshold"]),
        "nms_x_spaces": float(selector["nms_x_spaces"]),
        "minimum_selected_count": int(selector["minimum_selected_count"]),
        "maximum_selected_count": int(selector["maximum_selected_count"]),
    }
    if config["threshold"] != config["threshold"] or config["nms_x_spaces"] < 0:
        raise ValueError("Frozen selector contains invalid numeric settings")
    if config["minimum_selected_count"] < 0 or config["maximum_selected_count"] < 0:
        raise ValueError("Frozen selector counts must be non-negative")
    if config["minimum_selected_count"] > config["maximum_selected_count"]:
        raise ValueError("Frozen selector minimum exceeds maximum")
    return config


def _staff_spacing(row: Mapping[str, Any]) -> float:
    geometry = row.get("staff_geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError("Inference row has no staff_geometry")
    raw_lines = geometry.get("raw_staff_lines_y_px")
    if isinstance(raw_lines, Sequence) and not isinstance(raw_lines, (str, bytes)):
        lines = [float(value) for value in raw_lines]
        if len(lines) < 2:
            raise ValueError("Staff geometry needs at least two staff lines")
        spacing = sum(abs(right - left) for left, right in zip(lines, lines[1:], strict=False)) / (
            len(lines) - 1
        )
    elif geometry.get("staff_spacing_px") is not None:
        spacing = float(geometry["staff_spacing_px"])
    else:
        raise ValueError("Staff geometry has no usable staff spacing")
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("Staff spacing must be finite and positive")
    return spacing


def _candidate_score(candidate: Mapping[str, Any]) -> float:
    value = candidate.get("score", candidate.get("probability"))
    if value is None:
        raise ValueError(f"Candidate has no score: {candidate.get('candidate_id')!r}")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError("Candidate score must be finite")
    return score


def _candidate_id(candidate: Mapping[str, Any]) -> str:
    value = candidate.get("candidate_id", candidate.get("id"))
    if value is None:
        raise ValueError("Candidate has no candidate_id")
    return str(value)


def _candidate_center(candidate: Mapping[str, Any]) -> tuple[float, float]:
    center = candidate.get("center")
    if not isinstance(center, Mapping) or center.get("x") is None or center.get("y") is None:
        raise ValueError(f"Candidate has no usable center: {_candidate_id(candidate)}")
    return float(center["x"]), float(center["y"])


def select_candidates(
    inference_row: Mapping[str, Any],
    selector: Mapping[str, Any],
    *,
    y_separation_staff_spaces: float | None = None,
) -> list[dict[str, Any]]:
    """Replay threshold, deterministic ranking, NMS, and frozen count limits.

    ``None`` is the legacy x-only mode.  Otherwise two candidates conflict only
    when both their x distance is inside the frozen x radius and their y
    distance is below the supplied staff-space radius.
    """
    candidates = inference_row.get("candidate_predictions")
    if not isinstance(candidates, list):
        raise ValueError("Inference row has no candidate_predictions list")
    spacing = _staff_spacing(inference_row)
    threshold = float(selector["threshold"])
    x_radius = spacing * float(selector["nms_x_spaces"])
    minimum = int(selector["minimum_selected_count"])
    maximum = int(selector["maximum_selected_count"])
    if y_separation_staff_spaces is not None and y_separation_staff_spaces < 0:
        raise ValueError("y separation must be non-negative")

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("candidate_predictions entries must be objects")
        candidate_id = _candidate_id(candidate)
        if candidate_id in seen_ids:
            raise ValueError(f"Duplicate candidate_id: {candidate_id}")
        seen_ids.add(candidate_id)
        x, y = _candidate_center(candidate)
        normalized.append(
            {
                "candidate_id": candidate_id,
                "center": {"x": x, "y": y},
                "score": _candidate_score(candidate),
                "detector_rank": int(candidate.get("detector_rank", 10**9)),
                "source": candidate,
            }
        )
    ranked = sorted(
        normalized,
        key=lambda candidate: (
            -candidate["score"],
            candidate["detector_rank"],
            candidate["candidate_id"],
        ),
    )

    def conflicts(candidate: Mapping[str, Any], selected: Mapping[str, Any]) -> bool:
        dx = abs(float(candidate["center"]["x"]) - float(selected["center"]["x"]))
        if dx >= x_radius:
            return False
        if y_separation_staff_spaces is None:
            return True
        dy_spaces = abs(float(candidate["center"]["y"]) - float(selected["center"]["y"])) / spacing
        return dy_spaces < y_separation_staff_spaces

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for candidate in (item for item in ranked if item["score"] >= threshold):
        if any(conflicts(candidate, other) for other in selected):
            continue
        if y_separation_staff_spaces is not None:
            candidate_adds_group = not any(
                abs(float(candidate["center"]["x"]) - float(other["center"]["x"])) < x_radius
                for other in selected
            )
            if candidate_adds_group and _onset_group_count(selected, x_radius) >= maximum:
                continue
        elif len(selected) >= maximum:
            break
        selected.append(candidate)
        selected_ids.add(candidate["candidate_id"])
        if y_separation_staff_spaces is None and len(selected) >= maximum:
            break
    if len(selected) < minimum:
        for candidate in ranked:
            if candidate["candidate_id"] in selected_ids:
                continue
            if any(conflicts(candidate, other) for other in selected):
                continue
            if y_separation_staff_spaces is not None:
                candidate_adds_group = not any(
                    abs(float(candidate["center"]["x"]) - float(other["center"]["x"])) < x_radius
                    for other in selected
                )
                if candidate_adds_group and _onset_group_count(selected, x_radius) >= maximum:
                    continue
            selected.append(candidate)
            selected_ids.add(candidate["candidate_id"])
            if len(selected) >= min(minimum, maximum):
                break
    return selected


def _onset_group_count(selected: Sequence[Mapping[str, Any]], x_radius_px: float) -> int:
    """Count horizontal onset groups using the frozen x-NMS radius."""
    if not selected:
        return 0
    ordered = sorted(float(item["center"]["x"]) for item in selected)
    count = 1
    previous = ordered[0]
    for x in ordered[1:]:
        if x - previous >= x_radius_px:
            count += 1
        previous = x
    return count


def _horizontal_candidate_groups(
    selected: Sequence[Mapping[str, Any]], x_radius_px: float
) -> list[list[Mapping[str, Any]]]:
    groups: list[list[Mapping[str, Any]]] = []
    for candidate in sorted(
        selected,
        key=lambda item: (
            float(item["center"]["x"]),
            float(item["center"]["y"]),
            str(item["candidate_id"]),
        ),
    ):
        if not groups:
            groups.append([candidate])
            continue
        previous_x = float(groups[-1][-1]["center"]["x"])
        if float(candidate["center"]["x"]) - previous_x < x_radius_px:
            groups[-1].append(candidate)
        else:
            groups.append([candidate])
    return groups


def _ranked_candidate_predictions(inference_row: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = inference_row.get("candidate_predictions")
    if not isinstance(candidates, list):
        raise ValueError("Inference row has no candidate_predictions list")
    normalized = []
    seen_ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("candidate_predictions entries must be objects")
        candidate_id = _candidate_id(candidate)
        if candidate_id in seen_ids:
            raise ValueError(f"Duplicate candidate_id: {candidate_id}")
        seen_ids.add(candidate_id)
        x, y = _candidate_center(candidate)
        normalized.append(
            {
                "candidate_id": candidate_id,
                "center": {"x": x, "y": y},
                "score": _candidate_score(candidate),
                "detector_rank": int(candidate.get("detector_rank", 10**9)),
                "source": candidate,
            }
        )
    return sorted(
        normalized,
        key=lambda candidate: (
            -candidate["score"],
            candidate["detector_rank"],
            candidate["candidate_id"],
        ),
    )


def recover_chord_candidates(
    inference_row: Mapping[str, Any],
    selector: Mapping[str, Any],
    baseline_selected: Sequence[Mapping[str, Any]],
    *,
    minimum_y_gap_staff_spaces: float,
    maximum_y_gap_staff_spaces: float,
    minimum_score_ratio: float,
) -> list[dict[str, Any]]:
    """Recover at most one extra head inside each x-only onset group."""
    if minimum_y_gap_staff_spaces < 0:
        raise ValueError("Recovery minimum y gap must be non-negative")
    if maximum_y_gap_staff_spaces < minimum_y_gap_staff_spaces:
        raise ValueError("Recovery maximum y gap must be at least the minimum")
    if not 0 <= minimum_score_ratio <= 1:
        raise ValueError("Recovery score ratio must be between zero and one")
    spacing = _staff_spacing(inference_row)
    x_radius = spacing * float(selector["nms_x_spaces"])
    threshold = float(selector["threshold"])
    baseline = list(baseline_selected)
    baseline_group_count = _onset_group_count(baseline, x_radius)
    groups = _horizontal_candidate_groups(baseline, x_radius)
    selected_ids = {str(candidate["candidate_id"]) for candidate in baseline}
    ranked_candidates = _ranked_candidate_predictions(inference_row)
    recovered: list[dict[str, Any]] = []

    for group_index, group in enumerate(groups, start=1):
        best_selected = min(
            group,
            key=lambda candidate: (
                -float(candidate["score"]),
                int(candidate["detector_rank"]),
                str(candidate["candidate_id"]),
            ),
        )
        best_score = float(best_selected["score"])
        eligible = []
        for candidate in ranked_candidates:
            candidate_id = str(candidate["candidate_id"])
            if candidate_id in selected_ids or any(
                candidate_id == str(item["candidate_id"]) for item in recovered
            ):
                continue
            candidate_x = float(candidate["center"]["x"])
            candidate_y = float(candidate["center"]["y"])
            if not any(
                abs(candidate_x - float(anchor["center"]["x"])) < x_radius for anchor in group
            ):
                continue
            nearest_y_gap = min(
                abs(candidate_y - float(anchor["center"]["y"])) / spacing for anchor in group
            )
            if not minimum_y_gap_staff_spaces <= nearest_y_gap <= maximum_y_gap_staff_spaces:
                continue
            if float(candidate["score"]) < threshold:
                continue
            if float(candidate["score"]) < best_score * minimum_score_ratio:
                continue
            if _onset_group_count([*baseline, *recovered, candidate], x_radius) != (
                baseline_group_count
            ):
                continue
            eligible.append(candidate)
        if eligible:
            chosen = dict(eligible[0])
            chosen["recovery_group_index"] = group_index
            chosen["recovery_y_gap_staff_spaces"] = min(
                abs(float(chosen["center"]["y"]) - float(anchor["center"]["y"])) / spacing
                for anchor in group
            )
            chosen["recovery_score_ratio"] = (
                float(chosen["score"]) / best_score if best_score > 0 else None
            )
            recovered.append(chosen)
    return recovered


def _source_image_path_and_pin(inference_row: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Resolve and hash-pin the source crop before any pixel access."""
    source = inference_row.get("source")
    if not isinstance(source, Mapping):
        source_image = inference_row.get("source_image")
        source_sha256 = inference_row.get("source_sha256")
        if source_image is None or source_sha256 is None:
            raise ValueError("Inference row has no hash-pinned source image")
        source = {"image": source_image, "sha256": source_sha256}
    raw_path = source.get("image", source.get("path"))
    expected_sha256 = source.get("sha256", source.get("image_sha256"))
    if raw_path is None or expected_sha256 is None:
        raise ValueError("Inference row source image needs image/path and sha256")
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    data = path.read_bytes()
    actual_sha256 = _sha256_bytes(data)
    if actual_sha256 != str(expected_sha256):
        raise ValueError(f"Source image hash mismatch: {path}")
    return path, {"path": str(path), "sha256": actual_sha256, "bytes": len(data)}


def _staff_suppressed_ink_mask(
    gray: Image.Image,
    staff_lines: Sequence[Any],
    *,
    spacing: float,
    threshold: int,
) -> tuple[list[list[bool]], set[int]]:
    """Build a dark-ink mask with staff-line rows removed."""
    width, height = gray.size
    pixels = gray.load()
    suppressed_rows: set[int] = set()
    for line in staff_lines:
        center = int(round(float(line)))
        for offset in range(
            -STEM_STAFF_SUPPRESSION_RADIUS_PIXELS,
            STEM_STAFF_SUPPRESSION_RADIUS_PIXELS + 1,
        ):
            row = center + offset
            if 0 <= row < height:
                suppressed_rows.add(row)
    mask = []
    for y in range(height):
        if y in suppressed_rows:
            mask.append([False] * width)
        else:
            mask.append([pixels[x, y] < threshold for x in range(width)])
    return mask, suppressed_rows


def _run_occupancy(mask: Sequence[Sequence[bool]], x: int, y: int) -> bool:
    if y < 0 or y >= len(mask):
        return False
    row = mask[y]
    return any(row[max(0, x - STEM_PATH_HALF_WIDTH_PIXELS) : x + STEM_PATH_HALF_WIDTH_PIXELS + 1])


def _candidate_bbox(candidate: Mapping[str, Any]) -> tuple[int, int, int, int]:
    bbox = candidate.get("bbox")
    if not isinstance(bbox, Mapping):
        raise ValueError(f"Candidate has no bbox: {_candidate_id(candidate)}")
    try:
        left = int(round(float(bbox["left"])))
        top = int(round(float(bbox["top"])))
        right = int(round(float(bbox["right"])))
        bottom = int(round(float(bbox["bottom"])))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Candidate has invalid bbox: {_candidate_id(candidate)}") from exc
    if right < left or bottom < top:
        raise ValueError(f"Candidate bbox is inverted: {_candidate_id(candidate)}")
    return left, top, right, bottom


def _directional_stem_run(
    mask: Sequence[Sequence[bool]],
    suppressed_rows: set[int],
    *,
    bbox: tuple[int, int, int, int],
    x: int,
    direction: str,
    spacing: float,
) -> dict[str, Any] | None:
    """Score a narrow run extending up or down from a candidate bbox."""
    _, top, _, bottom = bbox
    height = len(mask)
    if direction == "up":
        attach_start = max(0, top - 3)
        attach_end = min(height - 1, top + 3)
        outward_start = top - 1
        outward_end = max(0, top - int(round(spacing * STEM_SEARCH_EXTRA_SPACES)))
        step = -1
    elif direction == "down":
        attach_start = max(0, bottom - 3)
        attach_end = min(height - 1, bottom + 3)
        outward_start = bottom + 1
        outward_end = min(height - 1, bottom + int(round(spacing * STEM_SEARCH_EXTRA_SPACES)))
        step = 1
    else:
        raise ValueError(f"Unknown stem direction: {direction}")
    if outward_start < 0 or outward_start >= height:
        return None
    if not any(_run_occupancy(mask, x, y) for y in range(attach_start, attach_end + 1)):
        return None
    rows = list(range(outward_start, outward_end + step, step))
    first_ink_index = next(
        (index for index, y in enumerate(rows) if _run_occupancy(mask, x, y)), None
    )
    if first_ink_index is None:
        return None
    if first_ink_index > STEM_MAX_GAP_PIXELS:
        return None
    ink_rows = 0
    nonstaff_rows = 0
    last_index = first_ink_index
    gap = 0
    for index in range(first_ink_index, len(rows)):
        y = rows[index]
        occupied = _run_occupancy(mask, x, y)
        if y not in suppressed_rows:
            nonstaff_rows += 1
            if occupied:
                ink_rows += 1
        if occupied:
            gap = 0
            last_index = index
        else:
            gap += 1
            if gap > STEM_MAX_GAP_PIXELS:
                break
    if nonstaff_rows <= 0 or ink_rows <= 0:
        return None
    run_fraction = min(1.0, ink_rows / nonstaff_rows)
    run_length_px = abs(rows[last_index] - rows[first_ink_index]) + 1
    length_score = min(1.0, run_length_px / max(1.0, 1.5 * spacing))
    score = run_fraction * length_score
    return {
        "direction": direction,
        "x": x,
        "run_fraction": round(run_fraction, 6),
        "length_score": round(length_score, 6),
        "score": round(score, 6),
        "run_length_px": run_length_px,
        "run_length_staff_spacing": round(run_length_px / spacing, 6),
        "ink_rows": ink_rows,
        "nonstaff_rows": nonstaff_rows,
        "staff_rows_suppressed": len(suppressed_rows),
    }


def candidate_local_stem_features(
    inference_row: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Compute deterministic candidate-local stem scores from the source crop."""
    source_path, source_pin = _source_image_path_and_pin(inference_row)
    staff_lines = _row_staff_lines(inference_row)
    spacing = _staff_spacing(inference_row)
    with Image.open(source_path) as opened:
        gray = opened.convert("L")
        threshold = estimate_ink_threshold(gray)
        mask, suppressed_rows = _staff_suppressed_ink_mask(
            gray, staff_lines, spacing=spacing, threshold=threshold
        )
        features: dict[str, dict[str, Any]] = {}
        candidates = inference_row.get("candidate_predictions")
        if not isinstance(candidates, list):
            raise ValueError("Inference row has no candidate_predictions list")
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise ValueError("candidate_predictions entries must be objects")
            candidate_id = _candidate_id(candidate)
            left, top, right, bottom = _candidate_bbox(candidate)
            best: dict[str, Any] | None = None
            for x in range(
                max(0, left - STEM_SEARCH_SIDE_PIXELS),
                min(gray.width - 1, right + STEM_SEARCH_SIDE_PIXELS) + 1,
            ):
                for direction in ("up", "down"):
                    result = _directional_stem_run(
                        mask,
                        suppressed_rows,
                        bbox=(left, top, right, bottom),
                        x=x,
                        direction=direction,
                        spacing=spacing,
                    )
                    if result is None:
                        continue
                    result["side"] = "left" if x < left else "right" if x > right else "inside"
                    result["candidate_bbox"] = {
                        "left": left,
                        "top": top,
                        "right": right,
                        "bottom": bottom,
                    }
                    if best is None or (
                        float(result["score"]),
                        -abs(x - (left + right) / 2.0),
                        direction,
                    ) > (
                        float(best["score"]),
                        -abs(int(best["x"]) - (left + right) / 2.0),
                        str(best["direction"]),
                    ):
                        best = result
            features[candidate_id] = best or {
                "score": 0.0,
                "direction": None,
                "x": None,
                "run_fraction": 0.0,
                "length_score": 0.0,
                "run_length_px": 0,
                "run_length_staff_spacing": 0.0,
                "ink_rows": 0,
                "nonstaff_rows": 0,
                "staff_rows_suppressed": len(suppressed_rows),
                "candidate_bbox": {
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                },
            }
    return features, {
        "source_image": source_pin,
        "ink_threshold": threshold,
        "staff_lines": [float(value) for value in staff_lines],
        "staff_spacing_px": spacing,
        "staff_suppression_radius_px": STEM_STAFF_SUPPRESSION_RADIUS_PIXELS,
        "path_half_width_px": STEM_PATH_HALF_WIDTH_PIXELS,
    }


def recover_stem_aware_chord_candidates(
    inference_row: Mapping[str, Any],
    selector: Mapping[str, Any],
    baseline_selected: Sequence[Mapping[str, Any]],
    *,
    minimum_y_gap_staff_spaces: float,
    maximum_y_gap_staff_spaces: float,
    minimum_score_ratio: float,
    minimum_stem_score: float,
    stem_features: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Apply the fixed stem threshold to the existing bounded recovery."""
    recovered = recover_chord_candidates(
        inference_row,
        selector,
        baseline_selected,
        minimum_y_gap_staff_spaces=minimum_y_gap_staff_spaces,
        maximum_y_gap_staff_spaces=maximum_y_gap_staff_spaces,
        minimum_score_ratio=minimum_score_ratio,
    )
    filtered = []
    for candidate in recovered:
        feature = stem_features.get(str(candidate["candidate_id"]))
        if feature is None:
            raise ValueError(f"Missing stem feature for candidate {_candidate_id(candidate)}")
        if float(feature["score"]) < minimum_stem_score:
            continue
        chosen = dict(candidate)
        chosen["stem_attachment_score"] = float(feature["score"])
        filtered.append(chosen)
    return filtered


def recover_edge_safe_stem_aware_chord_candidates(
    inference_row: Mapping[str, Any],
    selector: Mapping[str, Any],
    baseline_selected: Sequence[Mapping[str, Any]],
    *,
    minimum_y_gap_staff_spaces: float,
    maximum_y_gap_staff_spaces: float,
    minimum_score_ratio: float,
    minimum_stem_score: float,
    minimum_group_x_staff_spaces: float,
    stem_features: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Recover stem-supported companions away from ambiguous leading-edge ink.

    The leading staff-space often contains a crop barline, clef fragment, or
    key-signature ink.  Recovery is intentionally fail-closed there while the
    original x-only selection remains untouched.
    """
    if minimum_group_x_staff_spaces < 0:
        raise ValueError("Recovery minimum group x must be non-negative")
    recovered = recover_stem_aware_chord_candidates(
        inference_row,
        selector,
        baseline_selected,
        minimum_y_gap_staff_spaces=minimum_y_gap_staff_spaces,
        maximum_y_gap_staff_spaces=maximum_y_gap_staff_spaces,
        minimum_score_ratio=minimum_score_ratio,
        minimum_stem_score=minimum_stem_score,
        stem_features=stem_features,
    )
    spacing = _staff_spacing(inference_row)
    minimum_x = minimum_group_x_staff_spaces * spacing
    filtered = []
    for candidate in recovered:
        candidate_x = float(candidate["center"]["x"])
        if candidate_x < minimum_x:
            continue
        chosen = dict(candidate)
        chosen["leading_edge_distance_staff_spaces"] = candidate_x / spacing
        filtered.append(chosen)
    return filtered


def recover_edge_safe_stem_aware_multihead_candidates(
    inference_row: Mapping[str, Any],
    selector: Mapping[str, Any],
    baseline_selected: Sequence[Mapping[str, Any]],
    *,
    minimum_y_gap_staff_spaces: float,
    maximum_y_gap_staff_spaces: float,
    minimum_score_ratio: float,
    minimum_stem_score: float,
    minimum_group_x_staff_spaces: float,
    maximum_recovered_heads_per_group: int,
    stem_features: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Recover a bounded vertical chain inside each existing onset group.

    Candidates remain tied to a baseline x-group and are accepted in ranked
    order. Each accepted companion becomes a vertical anchor for the next one,
    allowing triads or four-note chords without inventing a new onset.
    """
    if minimum_y_gap_staff_spaces < 0:
        raise ValueError("Recovery minimum y gap must be non-negative")
    if maximum_y_gap_staff_spaces < minimum_y_gap_staff_spaces:
        raise ValueError("Recovery maximum y gap must be at least the minimum")
    if not 0 <= minimum_score_ratio <= 1:
        raise ValueError("Recovery score ratio must be between zero and one")
    if minimum_group_x_staff_spaces < 0:
        raise ValueError("Recovery minimum group x must be non-negative")
    if maximum_recovered_heads_per_group < 1:
        raise ValueError("Recovery per-group cap must be positive")

    spacing = _staff_spacing(inference_row)
    x_radius = spacing * float(selector["nms_x_spaces"])
    threshold = float(selector["threshold"])
    minimum_x = minimum_group_x_staff_spaces * spacing
    baseline = list(baseline_selected)
    groups = _horizontal_candidate_groups(baseline, x_radius)
    ranked = _ranked_candidate_predictions(inference_row)
    selected_ids = {str(candidate["candidate_id"]) for candidate in baseline}
    recovered: list[dict[str, Any]] = []

    for group_index, group in enumerate(groups, start=1):
        best_selected = min(
            group,
            key=lambda candidate: (
                -float(candidate["score"]),
                int(candidate["detector_rank"]),
                str(candidate["candidate_id"]),
            ),
        )
        best_score = float(best_selected["score"])
        anchors = list(group)
        group_recovered: list[dict[str, Any]] = []
        while len(group_recovered) < maximum_recovered_heads_per_group:
            eligible = []
            for candidate in ranked:
                candidate_id = str(candidate["candidate_id"])
                if candidate_id in selected_ids or any(
                    candidate_id == str(item["candidate_id"]) for item in recovered
                ):
                    continue
                candidate_x = float(candidate["center"]["x"])
                candidate_y = float(candidate["center"]["y"])
                if candidate_x < minimum_x:
                    continue
                if not any(
                    abs(candidate_x - float(anchor["center"]["x"])) < x_radius for anchor in group
                ):
                    continue
                nearest_y_gap = min(
                    abs(candidate_y - float(anchor["center"]["y"])) / spacing for anchor in anchors
                )
                if not minimum_y_gap_staff_spaces <= nearest_y_gap <= (maximum_y_gap_staff_spaces):
                    continue
                if float(candidate["score"]) < threshold:
                    continue
                if float(candidate["score"]) < best_score * minimum_score_ratio:
                    continue
                stem = stem_features.get(candidate_id)
                if stem is None:
                    raise ValueError(f"Missing stem feature for candidate {candidate_id}")
                if float(stem["score"]) < minimum_stem_score:
                    continue
                eligible.append((candidate, nearest_y_gap, float(stem["score"])))
            if not eligible:
                break
            candidate, nearest_y_gap, stem_score = eligible[0]
            chosen = dict(candidate)
            chosen["recovery_group_index"] = group_index
            chosen["recovery_y_gap_staff_spaces"] = nearest_y_gap
            chosen["recovery_score_ratio"] = (
                float(chosen["score"]) / best_score if best_score > 0 else None
            )
            chosen["stem_attachment_score"] = stem_score
            chosen["leading_edge_distance_staff_spaces"] = float(chosen["center"]["x"]) / spacing
            group_recovered.append(chosen)
            recovered.append(chosen)
            anchors.append(chosen)

    if _onset_group_count([*baseline, *recovered], x_radius) != _onset_group_count(
        baseline, x_radius
    ):
        raise ValueError("Multihead recovery created a new onset group")
    return recovered


def _number(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    return _NUMBER_WORDS.get(value.lower())


def _fifths_accidentals(fifths: int) -> dict[str, int]:
    accidentals: dict[str, int] = {}
    if fifths > 0:
        for letter in _SHARP_ORDER[:fifths]:
            accidentals[letter] = 1
    elif fifths < 0:
        for letter in _FLAT_ORDER[:-fifths]:
            accidentals[letter] = -1
    return accidentals


def _key_accidentals(key_hint: Any) -> dict[str, int]:
    if key_hint is None:
        return {}
    if isinstance(key_hint, Mapping):
        for key in ("key_fifths", "fifths"):
            if key in key_hint:
                return _fifths_accidentals(int(key_hint[key]))
        raw_accidentals = key_hint.get("accidentals")
        if isinstance(raw_accidentals, Mapping):
            result = {}
            for letter, value in raw_accidentals.items():
                result[str(letter).upper()[0]] = _alteration_value(value)
            return result
        key_hint = key_hint.get("key_hint", key_hint.get("description"))
    if not isinstance(key_hint, str):
        raise ValueError(f"Unsupported key hint: {key_hint!r}")
    text = key_hint.lower().replace("-", " ")
    match = re.search(r"(\d+|one|two|three|four|five|six|seven)\s+(flat|flats|sharp|sharps)", text)
    if match:
        count = _number(match.group(1))
        if count is None:
            raise ValueError(f"Unsupported key hint count: {key_hint!r}")
        direction = -1 if match.group(2).startswith("flat") else 1
        return _fifths_accidentals(direction * count)
    match = re.search(r"key\s*fifths?\s*[:=]\s*(-?\d+)", text)
    if match:
        return _fifths_accidentals(int(match.group(1)))
    raise ValueError(f"Could not parse key hint: {key_hint!r}")


def _alteration_value(value: Any) -> int:
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"flat", "b", "-1"}:
            return -1
        if lowered in {"sharp", "#", "1"}:
            return 1
        if lowered in {"natural", "0"}:
            return 0
    return int(value)


def _staff_pitch(
    y: float, staff_lines: Sequence[Any], alterations: Mapping[str, int]
) -> dict[str, Any]:
    lines = [float(value) for value in staff_lines]
    spacing = sum(abs(right - left) for left, right in zip(lines, lines[1:], strict=False)) / (
        len(lines) - 1
    )
    position = math.floor((lines[-1] - y) / (spacing / 2.0) + 0.5)
    letter_index = 2 + position  # bottom treble line is E4
    letter = _DIATONIC_LETTERS[letter_index % 7]
    octave = 4 + math.floor(letter_index / 7)
    alteration = int(alterations.get(letter, 0))
    midi = 12 * (octave + 1) + _NATURAL_MIDI[letter] + alteration
    accidental = "#" if alteration == 1 else "b" if alteration == -1 else ""
    return {
        "pitch_midi": midi,
        "pitch": f"{letter}{accidental}{octave}",
        "staff_position": position,
    }


def _row_staff_lines(row: Mapping[str, Any]) -> Sequence[Any]:
    geometry = row.get("staff_geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError("Inference row has no staff geometry")
    lines = geometry.get("raw_staff_lines_y_px")
    if not isinstance(lines, Sequence) or isinstance(lines, (str, bytes)) or len(lines) < 2:
        raise ValueError("Inference row has no raw staff lines")
    return lines


def _identity_measure(row: Mapping[str, Any]) -> int:
    identity = row.get("identity")
    if not isinstance(identity, Mapping) or identity.get("automatic_measure_index") is None:
        raise ValueError("Inference row has no automatic_measure_index")
    return int(identity["automatic_measure_index"])


def _row_key_hint(row: Mapping[str, Any]) -> Any:
    context = row.get("allowed_context")
    return context.get("key_hint") if isinstance(context, Mapping) else None


def _automatic_context_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    hints = {
        str(_identity_measure(row)): _row_key_hint(row)
        for row in rows
        if _row_key_hint(row) is not None
    }
    if not hints:
        return {"key_hint": None, "source": "natural_staff_context"}
    return {
        "key_hints_by_measure": hints,
        "source": "frozen_inference_allowed_context",
    }


def _materialize_predictions(
    rows: Sequence[Mapping[str, Any]],
    selector: Mapping[str, Any],
    y_grid: Sequence[float | None],
    *,
    lane: str,
    recovery_grid: Sequence[tuple[float, float, float]] = (),
    stem_recovery_grid: Sequence[tuple[float, float, float, float]] = (),
    context_hints: Mapping[int, Any] | None = None,
    truth_alterations: Mapping[tuple[str, int], int] | None = None,
) -> dict[str, dict[int, dict[str, Any]]]:
    predictions: dict[str, dict[int, dict[str, Any]]] = {}
    for y_separation in y_grid:
        config_id = _config_id(y_separation)
        by_measure: dict[int, dict[str, Any]] = {}
        for row in rows:
            if row.get("truth_used") is True:
                raise ValueError("Inference row is marked truth_used; refusing automatic replay")
            measure = _identity_measure(row)
            hint = (
                context_hints.get(measure, _row_key_hint(row))
                if context_hints is not None
                else _row_key_hint(row)
            )
            alterations = _key_accidentals(hint)
            selected = select_candidates(
                row,
                selector,
                y_separation_staff_spaces=y_separation,
            )
            by_measure[measure] = _materialize_selected_prediction(
                row,
                selector,
                selected,
                config_id=config_id,
                config_family=(
                    "x_only_baseline" if y_separation is None else "two_dimensional_nms"
                ),
                lane=lane,
                alterations=alterations,
                truth_alterations=truth_alterations,
                y_separation_staff_spaces=y_separation,
            )
        predictions[config_id] = by_measure
    for minimum_y_gap, maximum_y_gap, minimum_score_ratio in recovery_grid:
        config_id = _recovery_config_id(
            minimum_y_gap_staff_spaces=minimum_y_gap,
            maximum_y_gap_staff_spaces=maximum_y_gap,
            minimum_score_ratio=minimum_score_ratio,
        )
        by_measure = {}
        for row in rows:
            if row.get("truth_used") is True:
                raise ValueError("Inference row is marked truth_used; refusing automatic replay")
            measure = _identity_measure(row)
            hint = (
                context_hints.get(measure, _row_key_hint(row))
                if context_hints is not None
                else _row_key_hint(row)
            )
            alterations = _key_accidentals(hint)
            baseline_selected = select_candidates(row, selector)
            recovered = recover_chord_candidates(
                row,
                selector,
                baseline_selected,
                minimum_y_gap_staff_spaces=minimum_y_gap,
                maximum_y_gap_staff_spaces=maximum_y_gap,
                minimum_score_ratio=minimum_score_ratio,
            )
            selected = [*baseline_selected, *recovered]
            baseline_group_count = _onset_group_count(
                baseline_selected, _staff_spacing(row) * float(selector["nms_x_spaces"])
            )
            by_measure[measure] = _materialize_selected_prediction(
                row,
                selector,
                selected,
                config_id=config_id,
                config_family="chord_recovery",
                lane=lane,
                alterations=alterations,
                truth_alterations=truth_alterations,
                recovered=recovered,
                baseline_group_count=baseline_group_count,
                recovery_parameters={
                    "minimum_y_gap_staff_spaces": minimum_y_gap,
                    "maximum_y_gap_staff_spaces": maximum_y_gap,
                    "minimum_score_ratio": minimum_score_ratio,
                },
            )
        predictions[config_id] = by_measure
    for minimum_y_gap, maximum_y_gap, minimum_score_ratio, minimum_stem_score in stem_recovery_grid:
        config_id = _stem_recovery_config_id(
            minimum_y_gap_staff_spaces=minimum_y_gap,
            maximum_y_gap_staff_spaces=maximum_y_gap,
            minimum_score_ratio=minimum_score_ratio,
            minimum_stem_score=minimum_stem_score,
        )
        by_measure = {}
        for row in rows:
            if row.get("truth_used") is True:
                raise ValueError("Inference row is marked truth_used; refusing automatic replay")
            measure = _identity_measure(row)
            hint = (
                context_hints.get(measure, _row_key_hint(row))
                if context_hints is not None
                else _row_key_hint(row)
            )
            alterations = _key_accidentals(hint)
            stem_features, stem_metadata = candidate_local_stem_features(row)
            baseline_selected = select_candidates(row, selector)
            recovered = recover_stem_aware_chord_candidates(
                row,
                selector,
                baseline_selected,
                minimum_y_gap_staff_spaces=minimum_y_gap,
                maximum_y_gap_staff_spaces=maximum_y_gap,
                minimum_score_ratio=minimum_score_ratio,
                minimum_stem_score=minimum_stem_score,
                stem_features=stem_features,
            )
            selected = [*baseline_selected, *recovered]
            baseline_group_count = _onset_group_count(
                baseline_selected, _staff_spacing(row) * float(selector["nms_x_spaces"])
            )
            by_measure[measure] = _materialize_selected_prediction(
                row,
                selector,
                selected,
                config_id=config_id,
                config_family="chord_recovery_stem",
                lane=lane,
                alterations=alterations,
                truth_alterations=truth_alterations,
                recovered=recovered,
                baseline_group_count=baseline_group_count,
                recovery_parameters={
                    "minimum_y_gap_staff_spaces": minimum_y_gap,
                    "maximum_y_gap_staff_spaces": maximum_y_gap,
                    "minimum_score_ratio": minimum_score_ratio,
                    "minimum_stem_score": minimum_stem_score,
                },
                stem_features=stem_features,
                stem_metadata=stem_metadata,
            )
        predictions[config_id] = by_measure
    return predictions


def _materialize_selected_prediction(
    row: Mapping[str, Any],
    selector: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    *,
    config_id: str,
    config_family: str,
    lane: str,
    alterations: Mapping[str, int],
    truth_alterations: Mapping[tuple[str, int], int] | None,
    y_separation_staff_spaces: float | None = None,
    recovered: Sequence[Mapping[str, Any]] = (),
    baseline_group_count: int | None = None,
    recovery_parameters: Mapping[str, float] | None = None,
    stem_features: Mapping[str, Mapping[str, Any]] | None = None,
    stem_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    staff_lines = _row_staff_lines(row)
    recovered_ids = {str(candidate["candidate_id"]) for candidate in recovered}
    notes: list[dict[str, Any]] = []
    for candidate in sorted(
        selected,
        key=lambda item: (
            float(item["center"]["x"]),
            float(item["center"]["y"]),
            str(item["candidate_id"]),
        ),
    ):
        x = float(candidate["center"]["x"])
        y = float(candidate["center"]["y"])
        base = _staff_pitch(y, staff_lines, {})
        if truth_alterations is not None:
            oracle_key = (base["pitch"][0], int(re.search(r"\d+", base["pitch"])[0]))
            oracle_alteration = truth_alterations.get(oracle_key)
            if oracle_alteration is not None:
                base = _staff_pitch(y, staff_lines, {base["pitch"][0]: oracle_alteration})
        else:
            base = _staff_pitch(y, staff_lines, alterations)
        note = {
            "candidate_id": candidate["candidate_id"],
            "x": x,
            "y": y,
            "score": candidate["score"],
            "pitch": base["pitch"],
            "pitch_midi": base["pitch_midi"],
            "recovered": str(candidate["candidate_id"]) in recovered_ids,
        }
        if stem_features is not None:
            note["stem_attachment"] = dict(stem_features[str(candidate["candidate_id"])])
        notes.append(note)
    spacing = _staff_spacing(row)
    onset_group_count = _onset_group_count(selected, spacing * float(selector["nms_x_spaces"]))
    onset_group_count_unchanged = None
    if baseline_group_count is not None:
        onset_group_count_unchanged = onset_group_count == baseline_group_count
        if not onset_group_count_unchanged:
            raise ValueError(
                f"Chord recovery changed onset-group count for measure {_identity_measure(row)}"
            )
    result = {
        "automatic_measure_index": _identity_measure(row),
        "config_id": config_id,
        "config_family": config_family,
        "y_separation_staff_spaces": y_separation_staff_spaces,
        "recovery_parameters": dict(recovery_parameters) if recovery_parameters else None,
        "staff_spacing_px": spacing,
        "selected_candidate_ids": [note["candidate_id"] for note in notes],
        "recovered_candidate_ids": [note["candidate_id"] for note in notes if note["recovered"]],
        "recovered_head_count": len(recovered_ids),
        "onset_group_count": onset_group_count,
        "baseline_onset_group_count": baseline_group_count,
        "onset_group_count_unchanged_from_x_only": onset_group_count_unchanged,
        "total_head_count": len(notes),
        "notes": notes,
        "lane": lane,
    }
    if stem_features is not None:
        result["stem_feature_diagnostics"] = {
            candidate_id: dict(stem_features[candidate_id])
            for candidate_id in sorted(stem_features)
        }
        result["stem_feature_metadata"] = dict(stem_metadata or {})
    return result


def _config_id(y_separation_staff_spaces: float | None) -> str:
    if y_separation_staff_spaces is None:
        return "x_only"
    return f"x_and_y_lt_{y_separation_staff_spaces:g}_staff_spaces"


def _recovery_config_id(
    *,
    minimum_y_gap_staff_spaces: float,
    maximum_y_gap_staff_spaces: float,
    minimum_score_ratio: float,
) -> str:
    return (
        f"chord_recovery__min_y_{minimum_y_gap_staff_spaces:g}"
        f"__max_y_{maximum_y_gap_staff_spaces:g}"
        f"__score_ratio_{minimum_score_ratio:g}"
    )


def _stem_recovery_config_id(
    *,
    minimum_y_gap_staff_spaces: float,
    maximum_y_gap_staff_spaces: float,
    minimum_score_ratio: float,
    minimum_stem_score: float,
) -> str:
    return (
        f"chord_recovery_stem__min_y_{minimum_y_gap_staff_spaces:g}"
        f"__max_y_{maximum_y_gap_staff_spaces:g}"
        f"__score_ratio_{minimum_score_ratio:g}"
        f"__stem_score_{minimum_stem_score:g}"
    )


def _truth_alterations(truth_rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], int]:
    observed: dict[tuple[str, int], int | None] = {}
    seen: set[tuple[str, int]] = set()
    pitch_pattern = re.compile(r"^([A-Ga-g])(?:[#b])?(\d+)$")
    for row in truth_rows:
        notes = row.get("notes", [])
        if not isinstance(notes, list):
            continue
        for note in notes:
            if not isinstance(note, Mapping):
                continue
            match = pitch_pattern.match(str(note.get("pitch", "")))
            if not match:
                continue
            key = (match.group(1).upper(), int(match.group(2)))
            alteration = int(note.get("sounding_alter", 0))
            if key in seen and observed.get(key) != alteration:
                observed[key] = None
            elif key not in seen:
                observed[key] = alteration
            seen.add(key)
    return {key: int(value) for key, value in observed.items() if value is not None}


def _load_context_hints(
    path: Path | None,
    *,
    measure_indices: Sequence[int],
) -> tuple[dict[int, Any], dict[str, Any] | None]:
    if path is None:
        return {}, None
    payload, pin = _read_json_object(path, label="context hint")
    if "events" in payload:
        events = payload["events"]
        if not isinstance(events, list) or not events:
            raise ValueError("Context hint events must be a non-empty list")
        normalized_events: list[tuple[int, Any]] = []
        for event in events:
            if not isinstance(event, Mapping) or event.get("start_measure") is None:
                raise ValueError("Context hint events need start_measure")
            start_measure = int(event["start_measure"])
            hint = event.get("key_hint", event.get("hint"))
            if start_measure <= 0 or hint is None:
                raise ValueError("Context hint events need a positive start_measure and key_hint")
            _key_accidentals(hint)
            normalized_events.append((start_measure, hint))
        starts = [start for start, _ in normalized_events]
        if len(starts) != len(set(starts)):
            raise ValueError("Context hint events have duplicate start_measure values")
        normalized_events.sort()
        values = {}
        for measure in sorted(set(int(value) for value in measure_indices)):
            active = [hint for start, hint in normalized_events if start <= measure]
            if active:
                values[measure] = active[-1]
    else:
        entries: Any = payload.get("measures", payload.get("hints", payload))
        if isinstance(entries, list):
            values = {}
            for entry in entries:
                if not isinstance(entry, Mapping) or entry.get("automatic_crop_index") is None:
                    raise ValueError("Context hint list entries need automatic_crop_index")
                values[int(entry["automatic_crop_index"])] = entry.get("key_hint", entry)
        elif isinstance(entries, Mapping):
            values = {}
            for key, entry in entries.items():
                if key in {"source", "provenance", "schema_version"}:
                    continue
                if isinstance(entry, Mapping):
                    entry = entry.get("key_hint", entry.get("hint", entry))
                values[int(key)] = entry
        else:
            raise ValueError("Context hints must be a measure map, list, or event sequence")
    if not values:
        raise ValueError("Context hint file contains no measure-scoped hints")
    for measure, hint in values.items():
        _key_accidentals(hint)
        if measure <= 0:
            raise ValueError("Context hint measure indices must be positive")
    return values, pin


def _verify_x_only_replay_parity(
    inference_rows: Sequence[Mapping[str, Any]],
    automatic_predictions: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Require the experiment baseline to reproduce frozen natural-context output."""
    x_only = automatic_predictions.get("x_only")
    if x_only is None:
        raise ValueError("Automatic predictions have no x-only baseline")
    per_measure = []
    for row in inference_rows:
        measure = _identity_measure(row)
        canonical = row.get("canonical_prediction")
        if not isinstance(canonical, Mapping) or not isinstance(canonical.get("notes"), list):
            raise ValueError(f"Inference row {measure} has no frozen canonical_prediction notes")
        frozen_notes = canonical["notes"]
        if all(isinstance(note, Mapping) and note.get("candidate_id") for note in frozen_notes):
            frozen_ids = [str(note["candidate_id"]) for note in frozen_notes]
        else:
            anchors = row.get("automatic_anchors")
            if not isinstance(anchors, list) or len(anchors) != len(frozen_notes):
                raise ValueError(
                    f"Inference row {measure} cannot bind canonical notes to automatic anchors"
                )
            frozen_ids = [
                str(anchor["source"]["candidate_id"])
                for anchor in anchors
                if isinstance(anchor, Mapping) and isinstance(anchor.get("source"), Mapping)
            ]
            if len(frozen_ids) != len(frozen_notes):
                raise ValueError(
                    f"Inference row {measure} has incomplete automatic-anchor identities"
                )
        frozen_pitches = [int(note["pitch_midi"]) for note in frozen_notes]
        replay_notes = x_only[measure]["notes"]
        replay_ids = [str(note["candidate_id"]) for note in replay_notes]
        replay_pitches = [int(note["pitch_midi"]) for note in replay_notes]
        matches = frozen_ids == replay_ids and frozen_pitches == replay_pitches
        per_measure.append(
            {
                "automatic_measure_index": measure,
                "matches": matches,
                "frozen_candidate_ids": frozen_ids,
                "replay_candidate_ids": replay_ids,
                "frozen_pitch_midi": frozen_pitches,
                "replay_pitch_midi": replay_pitches,
            }
        )
    if not all(row["matches"] for row in per_measure):
        raise ValueError("X-only replay does not reproduce the frozen baseline")
    return {
        "status": "exact_frozen_baseline_replay",
        "measure_count": len(per_measure),
        "all_candidate_ids_and_pitches_match": True,
        "per_measure": per_measure,
    }


def _pitch_value(note: Mapping[str, Any]) -> int:
    if note.get("pitch_midi") is not None:
        return int(note["pitch_midi"])
    if isinstance(note.get("pitch"), (int, float)):
        return int(note["pitch"])
    raise ValueError("Prediction note has no pitch_midi")


def _truth_groups(notes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for note in notes:
        if note.get("onset_divisions") is None:
            raise ValueError("Truth note has no onset_divisions")
        grouped.setdefault(int(note["onset_divisions"]), []).append(note)
    result = []
    for onset in sorted(grouped):
        members = sorted(
            grouped[onset],
            key=lambda note: (
                int(note.get("xml_order", 10**9)),
                int(note.get("physical_note_index", 10**9)),
            ),
        )
        result.append(
            {
                "onset_divisions": onset,
                "pitch_set": sorted(_pitch_value(note) for note in members),
            }
        )
    return result


def _prediction_groups(
    notes: Sequence[Mapping[str, Any]],
    *,
    x_group_radius_px: float | None,
) -> list[dict[str, Any]]:
    if not notes:
        return []
    if all(note.get("onset_divisions") is not None for note in notes):
        grouped: dict[int, list[Mapping[str, Any]]] = {}
        for note in notes:
            grouped.setdefault(int(note["onset_divisions"]), []).append(note)
        return [
            {
                "onset_divisions": onset,
                "pitch_set": sorted(_pitch_value(note) for note in grouped[onset]),
            }
            for onset in sorted(grouped)
        ]
    ordered = sorted(
        notes,
        key=lambda note: (
            float(note.get("x", 0.0)),
            float(note.get("y", 0.0)),
            str(note.get("candidate_id", "")),
        ),
    )
    groups: list[list[Mapping[str, Any]]] = []
    for note in ordered:
        if not groups or x_group_radius_px is None:
            groups.append([note])
            continue
        previous_x = float(groups[-1][-1].get("x", 0.0))
        if float(note.get("x", 0.0)) - previous_x <= x_group_radius_px:
            groups[-1].append(note)
        else:
            groups.append([note])
    return [
        {
            "onset_divisions": None,
            "x_center": sum(float(note.get("x", 0.0)) for note in group) / len(group),
            "pitch_set": sorted(_pitch_value(note) for note in group),
        }
        for group in groups
    ]


def _align_sequences(predicted: Sequence[Any], truth: Sequence[Any]) -> dict[str, Any]:
    rows = len(predicted) + 1
    columns = len(truth) + 1
    distance = [[0] * columns for _ in range(rows)]
    for index in range(1, rows):
        distance[index][0] = index
    for index in range(1, columns):
        distance[0][index] = index
    for index in range(1, rows):
        for other in range(1, columns):
            distance[index][other] = min(
                distance[index - 1][other] + 1,
                distance[index][other - 1] + 1,
                distance[index - 1][other - 1] + (predicted[index - 1] != truth[other - 1]),
            )
    operations: list[dict[str, Any]] = []
    index = len(predicted)
    other = len(truth)
    while index or other:
        if index and other and predicted[index - 1] == truth[other - 1]:
            operations.append(
                {"operation": "exact", "predicted": predicted[index - 1], "truth": truth[other - 1]}
            )
            index -= 1
            other -= 1
        elif index and other and distance[index][other] == distance[index - 1][other - 1] + 1:
            operations.append(
                {
                    "operation": "substitution",
                    "predicted": predicted[index - 1],
                    "truth": truth[other - 1],
                }
            )
            index -= 1
            other -= 1
        elif index and distance[index][other] == distance[index - 1][other] + 1:
            operations.append(
                {"operation": "insertion", "predicted": predicted[index - 1], "truth": None}
            )
            index -= 1
        else:
            operations.append(
                {"operation": "deletion", "predicted": None, "truth": truth[other - 1]}
            )
            other -= 1
    operations.reverse()
    counts = {
        name: sum(operation["operation"] == name for operation in operations)
        for name in ("exact", "substitution", "insertion", "deletion")
    }
    return {"edit_distance": distance[-1][-1], "operations": operations, **counts}


def score_pitch_groups(
    predicted_notes: Sequence[Mapping[str, Any]],
    truth_notes: Sequence[Mapping[str, Any]],
    *,
    x_group_radius_px: float | None = None,
) -> dict[str, Any]:
    """Score unordered pitch sets per onset group and legacy ordered pitches."""
    predicted_groups = _prediction_groups(predicted_notes, x_group_radius_px=x_group_radius_px)
    truth_groups = _truth_groups(truth_notes)
    predicted_group_tokens = [tuple(group["pitch_set"]) for group in predicted_groups]
    truth_group_tokens = [tuple(group["pitch_set"]) for group in truth_groups]
    group_alignment = _align_sequences(predicted_group_tokens, truth_group_tokens)
    predicted_ordered = [
        _pitch_value(note)
        for note in sorted(
            predicted_notes,
            key=lambda note: (
                float(note.get("x", 0.0)),
                float(note.get("y", 0.0)),
                str(note.get("candidate_id", "")),
            ),
        )
    ]
    truth_ordered = [
        _pitch_value(note)
        for note in sorted(
            truth_notes,
            key=lambda note: (
                int(note["onset_divisions"]),
                int(note.get("xml_order", 10**9)),
                int(note.get("physical_note_index", 10**9)),
            ),
        )
    ]
    legacy_alignment = _align_sequences(predicted_ordered, truth_ordered)
    return {
        "pitch_groups": {
            "predicted_groups": predicted_groups,
            "truth_groups": truth_groups,
            "alignment": group_alignment,
            "predicted_group_count": len(predicted_groups),
            "truth_group_count": len(truth_groups),
            "exact_group_count": group_alignment["exact"],
            "sequence_exact": group_alignment["edit_distance"] == 0,
        },
        "legacy_ordered_pitch": {
            "predicted_ordered_pitches": predicted_ordered,
            "truth_ordered_pitches": truth_ordered,
            "alignment": legacy_alignment,
        },
        "note_count": {
            "predicted_note_count": len(predicted_notes),
            "truth_note_count": len(truth_notes),
            "absolute_count_error": abs(len(predicted_notes) - len(truth_notes)),
            "exact_count": len(predicted_notes) == len(truth_notes),
        },
    }


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _aggregate_scores(per_measure: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    group = [row["pitch_groups"] for row in per_measure]
    legacy = [row["legacy_ordered_pitch"] for row in per_measure]
    counts = [row["note_count"] for row in per_measure]
    truth_groups = sum(item["truth_group_count"] for item in group)
    predicted_groups = sum(item["predicted_group_count"] for item in group)
    exact_groups = sum(item["exact_group_count"] for item in group)
    exact_sequences = sum(bool(item["sequence_exact"]) for item in group)
    group_edit = sum(item["alignment"]["edit_distance"] for item in group)
    truth_notes = sum(item["truth_note_count"] for item in counts)
    predicted_notes = sum(item["predicted_note_count"] for item in counts)
    absolute_errors = sum(item["absolute_count_error"] for item in counts)
    predicted_onset_groups = sum(int(row["onset_group_count"]) for row in per_measure)
    predicted_heads = sum(int(row["total_head_count"]) for row in per_measure)
    truth_onset_groups = sum(int(row["truth_onset_group_count"]) for row in per_measure)
    truth_heads = sum(int(row["truth_total_head_count"]) for row in per_measure)
    recovered_heads = sum(int(row["recovered_head_count"]) for row in per_measure)
    recovery_invariants = [
        bool(row["onset_group_count_unchanged_from_x_only"])
        for row in per_measure
        if row["onset_group_count_unchanged_from_x_only"] is not None
    ]
    legacy_edit = sum(item["alignment"]["edit_distance"] for item in legacy)
    return {
        "pitch_group_metrics": {
            "truth_group_count": truth_groups,
            "predicted_group_count": predicted_groups,
            "exact_group_count": exact_groups,
            "exact_group_precision": _ratio(exact_groups, predicted_groups),
            "exact_group_recall": _ratio(exact_groups, truth_groups),
            "sequence_exact_count": exact_sequences,
            "sequence_exact_rate": _ratio(exact_sequences, len(group)),
            "group_edit_distance": group_edit,
            "substitutions": sum(item["alignment"]["substitution"] for item in group),
            "insertions": sum(item["alignment"]["insertion"] for item in group),
            "deletions": sum(item["alignment"]["deletion"] for item in group),
        },
        "legacy_ordered_pitch_metrics": {
            "ordered_pitch_edit_distance": legacy_edit,
            "exact_pitch_matches": sum(item["alignment"]["exact"] for item in legacy),
            "substitutions": sum(item["alignment"]["substitution"] for item in legacy),
            "insertions": sum(item["alignment"]["insertion"] for item in legacy),
            "deletions": sum(item["alignment"]["deletion"] for item in legacy),
        },
        "note_count_metrics": {
            "predicted_note_count": predicted_notes,
            "truth_note_count": truth_notes,
            "absolute_count_error": absolute_errors,
            "mean_absolute_count_error": _ratio(absolute_errors, len(counts)),
            "exact_count": sum(bool(item["exact_count"]) for item in counts),
            "exact_count_rate": _ratio(
                sum(bool(item["exact_count"]) for item in counts), len(counts)
            ),
        },
        "selection_count_metrics": {
            "predicted_onset_group_count": predicted_onset_groups,
            "truth_onset_group_count": truth_onset_groups,
            "predicted_total_head_count": predicted_heads,
            "truth_total_head_count": truth_heads,
            "recovered_head_count": recovered_heads,
            "onset_group_count_unchanged_from_x_only": (
                all(recovery_invariants) if recovery_invariants else None
            ),
        },
    }


def _score_lane(
    predictions: Mapping[str, Mapping[int, Mapping[str, Any]]],
    truth_by_measure: Mapping[int, Mapping[str, Any]],
    *,
    selector: Mapping[str, Any],
) -> list[dict[str, Any]]:
    results = []
    for config_id, by_measure in predictions.items():
        per_measure = []
        for measure in sorted(by_measure):
            if measure not in truth_by_measure:
                raise ValueError(f"No consumed truth row for automatic measure {measure}")
            prediction = by_measure[measure]
            truth = truth_by_measure[measure]
            truth_notes = truth.get("notes")
            if not isinstance(truth_notes, list):
                raise ValueError(f"Truth row has no notes list for measure {measure}")
            score = score_pitch_groups(
                prediction["notes"],
                truth_notes,
                x_group_radius_px=prediction["staff_spacing_px"] * float(selector["nms_x_spaces"]),
            )
            per_measure.append(
                {
                    "automatic_measure_index": measure,
                    "selected_candidate_ids": prediction["selected_candidate_ids"],
                    "recovered_candidate_ids": prediction["recovered_candidate_ids"],
                    "recovered_head_count": prediction["recovered_head_count"],
                    "onset_group_count": prediction["onset_group_count"],
                    "baseline_onset_group_count": prediction["baseline_onset_group_count"],
                    "onset_group_count_unchanged_from_x_only": prediction[
                        "onset_group_count_unchanged_from_x_only"
                    ],
                    "total_head_count": prediction["total_head_count"],
                    "truth_onset_group_count": score["pitch_groups"]["truth_group_count"],
                    "truth_total_head_count": len(truth_notes),
                    **score,
                    **(
                        {
                            "stem_feature_diagnostics": prediction["stem_feature_diagnostics"],
                            "stem_feature_metadata": prediction["stem_feature_metadata"],
                        }
                        if "stem_feature_diagnostics" in prediction
                        else {}
                    ),
                }
            )
        results.append(
            {
                "config_id": config_id,
                "config_family": by_measure[next(iter(sorted(by_measure)))]["config_family"],
                "y_separation_staff_spaces": by_measure[next(iter(sorted(by_measure)))][
                    "y_separation_staff_spaces"
                ],
                "recovery_parameters": by_measure[next(iter(sorted(by_measure)))][
                    "recovery_parameters"
                ],
                "metrics": _aggregate_scores(per_measure),
                "per_measure": per_measure,
            }
        )
    return results


def _lane_payload(
    label: str,
    *,
    sweep: list[dict[str, Any]],
    context: Mapping[str, Any],
    truth_used: bool,
) -> dict[str, Any]:
    return {
        "label": label,
        "status": "evaluated" if sweep else "not_run_no_context_hint",
        "context": dict(context),
        "eligibility": {
            "automatic_claim": label == LANE_AUTOMATIC,
            "automatic_context": label == LANE_AUTOMATIC,
            "runtime_adoption_eligible": False,
            "postmortem_consumed_evidence": True,
            "oracle_or_external_context_excluded_from_automatic_claims": label != LANE_AUTOMATIC,
        },
        "truth_access": {
            "used_for_selection": False,
            "used_for_scoring": True,
            "truth_derived_context": truth_used,
            "predictions_materialized_before_truth_read": True,
            "musicxml_read": False,
        },
        "sweep": sweep,
    }


def _write_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Consumed Polyphonic Pitch Repair",
        "",
        (
            "This is a create-once postmortem over consumed evaluation evidence. "
            "Frozen inputs were not modified."
        ),
        "",
        f"- Automatic claim lane: `{report['automatic_claims']['source_lane']}`",
        "- MusicXML opened: `false`",
        "- Context hints are external and excluded from automatic claims.",
        "- Diagnostic oracle is not eligible for runtime adoption.",
        "",
    ]
    for label, lane in report["lanes"].items():
        lines.extend([f"## {label}", ""])
        lines.append(f"Status: `{lane['status']}`.")
        if not lane["sweep"]:
            lines.append("")
            continue
        lines.extend(
            [
                "",
                "| Configuration | Family | Parameters | Group edit | Legacy pitch edit | "
                "Onset groups | Heads | Recovered | Groups unchanged |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for result in lane["sweep"]:
            metrics = result["metrics"]
            if result["config_family"] in {"chord_recovery", "chord_recovery_stem"}:
                recovery = result["recovery_parameters"]
                parameters = (
                    f"min y {recovery['minimum_y_gap_staff_spaces']:g}; "
                    f"max y {recovery['maximum_y_gap_staff_spaces']:g}; "
                    f"ratio {recovery['minimum_score_ratio']:g}"
                )
                if result["config_family"] == "chord_recovery_stem":
                    parameters += f"; stem >= {recovery['minimum_stem_score']:g}"
            elif result["config_family"] == "two_dimensional_nms":
                parameters = f"y < {result['y_separation_staff_spaces']:g}"
            else:
                parameters = "x-only"
            selection = metrics["selection_count_metrics"]
            groups_unchanged = selection["onset_group_count_unchanged_from_x_only"]
            invariant = "n/a" if groups_unchanged is None else str(groups_unchanged).lower()
            lines.append(
                f"| `{result['config_id']}` | `{result['config_family']}` | {parameters} | "
                f"{metrics['pitch_group_metrics']['group_edit_distance']} | "
                f"{metrics['legacy_ordered_pitch_metrics']['ordered_pitch_edit_distance']} | "
                f"{selection['predicted_onset_group_count']} | "
                f"{selection['predicted_total_head_count']} | "
                f"{selection['recovered_head_count']} | {invariant} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _write_create_once(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite create-once output: {output_dir}")
    temporary = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"Temporary create-once output already exists: {temporary}")
    temporary.mkdir()
    try:
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "report.md").write_text(_write_markdown(report), encoding="utf-8")
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite create-once output: {output_dir}")
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run_experiment(
    inference_path: Path,
    truth_path: Path,
    evaluation_report_path: Path,
    model_path: Path,
    *,
    output_dir: Path | None = None,
    context_hint_path: Path | None = None,
    y_separation_grid: Sequence[float] = DEFAULT_Y_SEPARATION_GRID,
    recovery_min_y_gap_grid: Sequence[float] = DEFAULT_RECOVERY_MIN_Y_GAP_GRID,
    recovery_score_ratio_grid: Sequence[float] = DEFAULT_RECOVERY_SCORE_RATIO_GRID,
    recovery_max_y_gap_staff_spaces: float = DEFAULT_RECOVERY_MAX_Y_GAP_STAFF_SPACES,
    stem_score_grid: Sequence[float] = DEFAULT_STEM_SCORE_GRID,
) -> dict[str, Any]:
    """Run the contained experiment and create report.json/report.md once."""
    inference_path = inference_path.expanduser().resolve()
    truth_path = truth_path.expanduser().resolve()
    evaluation_report_path = evaluation_report_path.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    output_dir = (
        (output_dir or evaluation_report_path.parent.parent / "postmortem_polyphonic_v1")
        .expanduser()
        .resolve()
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite create-once output: {output_dir}")
    y_grid: list[float | None] = [None]
    for value in y_separation_grid:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError("y separation grid values must be finite and non-negative")
        if numeric not in y_grid:
            y_grid.append(numeric)
    recovery_maximum = float(recovery_max_y_gap_staff_spaces)
    if not math.isfinite(recovery_maximum) or recovery_maximum <= 0:
        raise ValueError("Recovery maximum y gap must be finite and positive")
    recovery_minimums = []
    for value in recovery_min_y_gap_grid:
        numeric = float(value)
        if not math.isfinite(numeric) or not 0 <= numeric <= recovery_maximum:
            raise ValueError("Recovery minimum y gaps must be between zero and the maximum")
        if numeric not in recovery_minimums:
            recovery_minimums.append(numeric)
    recovery_ratios = []
    for value in recovery_score_ratio_grid:
        numeric = float(value)
        if not math.isfinite(numeric) or not 0 <= numeric <= 1:
            raise ValueError("Recovery score ratios must be between zero and one")
        if numeric not in recovery_ratios:
            recovery_ratios.append(numeric)
    recovery_grid = [
        (minimum, recovery_maximum, ratio)
        for minimum in recovery_minimums
        for ratio in recovery_ratios
    ]
    stem_scores = []
    for value in stem_score_grid:
        numeric = float(value)
        if not math.isfinite(numeric) or not 0 <= numeric <= 1:
            raise ValueError("Stem score grid values must be finite and between zero and one")
        if numeric not in stem_scores:
            stem_scores.append(numeric)
    if 1.0 not in recovery_minimums or 0.5 not in recovery_ratios:
        raise ValueError("Stem-aware grid requires minimum y gap 1.0 and score ratio 0.5")
    stem_recovery_grid = [(1.0, recovery_maximum, 0.5, stem_score) for stem_score in stem_scores]

    # Read only non-truth inputs before materializing every automatic/external
    # prediction.  The consumed truth JSONL is deliberately opened below.
    inference_rows, inference_pin = _read_jsonl(inference_path, label="inference")
    model, model_pin = _read_json_object(model_path, label="model")
    context_hints, context_pin = _load_context_hints(
        context_hint_path,
        measure_indices=[_identity_measure(row) for row in inference_rows],
    )
    automatic_context = _automatic_context_summary(inference_rows)
    automatic_uses_key_hint = automatic_context["source"] == "frozen_inference_allowed_context"
    selector = selector_config_from_model(model)
    automatic_predictions = _materialize_predictions(
        inference_rows,
        selector,
        y_grid,
        lane=LANE_AUTOMATIC,
        recovery_grid=recovery_grid,
        stem_recovery_grid=stem_recovery_grid,
    )
    context_predictions = (
        _materialize_predictions(
            inference_rows,
            selector,
            y_grid,
            lane=LANE_CONTEXT,
            recovery_grid=recovery_grid,
            stem_recovery_grid=stem_recovery_grid,
            context_hints=context_hints,
        )
        if context_hints
        else {}
    )
    baseline_parity = _verify_x_only_replay_parity(inference_rows, automatic_predictions)

    # These are the first consumed-truth accesses. At this point natural-context
    # and optional external-context predictions are already in memory. The prior
    # evaluation report is also truth-derived, so it is deliberately pinned here.
    truth_rows, truth_pin = _read_jsonl(truth_path, label="truth")
    _, evaluation_pin = _read_json_object(evaluation_report_path, label="evaluation report")
    truth_by_measure = {
        int(row["automatic_crop_index"]): row
        for row in truth_rows
        if row.get("automatic_crop_index") is not None
    }
    if len(truth_by_measure) != len(truth_rows):
        raise ValueError("Truth rows need unique automatic_crop_index values")
    oracle_alterations = _truth_alterations(truth_rows)
    oracle_predictions = _materialize_predictions(
        inference_rows,
        selector,
        y_grid,
        lane=LANE_ORACLE,
        recovery_grid=recovery_grid,
        stem_recovery_grid=stem_recovery_grid,
        truth_alterations=oracle_alterations,
    )

    automatic_sweep = _score_lane(automatic_predictions, truth_by_measure, selector=selector)
    context_sweep = _score_lane(context_predictions, truth_by_measure, selector=selector)
    oracle_sweep = _score_lane(oracle_predictions, truth_by_measure, selector=selector)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "consumed_polyphonic_pitch_repair_spike",
        "status": "evaluated_consumed_evidence_create_once",
        "protocol": {
            "candidate_selection": (
                "replay candidate_predictions scores with frozen threshold, deterministic ties, "
                "x-NMS, and frozen min/max limits"
            ),
            "nms_variant": (
                "2D conflict requires x distance inside frozen radius and y distance below "
                "configured staff-space separation"
            ),
            "count_limit_semantics": {
                "x_only": (
                    "frozen maximum_selected_count caps total selected raw heads; "
                    "baseline behavior is unchanged"
                ),
                "two_dimensional": (
                    "frozen maximum_selected_count caps horizontal/onset groups; "
                    "vertically separated heads inside an admitted x group do not consume "
                    "another group slot"
                ),
                "minimum_selected_count": (
                    "frozen minimum_selected_count remains a minimum selected-head fallback "
                    "in both lanes"
                ),
                "chord_recovery": (
                    "starts from exact x-only selection; adds at most one head to each existing "
                    "horizontal group and requires the onset-group count to remain unchanged"
                ),
            },
            "y_separation_grid_staff_spaces": y_grid,
            "chord_recovery_grid": [
                {
                    "minimum_y_gap_staff_spaces": minimum,
                    "maximum_y_gap_staff_spaces": maximum,
                    "minimum_score_ratio": ratio,
                }
                for minimum, maximum, ratio in recovery_grid
            ],
            "chord_recovery_stem_grid": [
                {
                    "minimum_y_gap_staff_spaces": minimum,
                    "maximum_y_gap_staff_spaces": maximum,
                    "minimum_score_ratio": ratio,
                    "minimum_stem_score": stem_score,
                }
                for minimum, maximum, ratio, stem_score in stem_recovery_grid
            ],
            "configuration_selection": (
                "baseline, 2D NMS, and every declared chord-recovery configuration are reported; "
                "truth does not select a winner"
            ),
            "truth_grouping": (
                "truth notes grouped by onset_divisions; pitches within each group compared "
                "unordered; group order remains onset order"
            ),
            "legacy_comparability": (
                "ordered pitch edit alignment and note-count metrics are reported alongside "
                "group metrics"
            ),
            "musicxml_read": False,
            "truth_access": (
                "truth JSONL and the prior truth-derived evaluation report opened only after "
                "natural and context-hint predictions were materialized"
            ),
            "context_hint_semantics": (
                "measure maps apply to explicit measures; ordered start_measure events carry "
                "forward until the next event"
            ),
        },
        "selector_reconstruction": selector,
        "baseline_parity": baseline_parity,
        "automatic_claims": {
            "source_lane": LANE_AUTOMATIC,
            "key_context": automatic_context,
            "truth_used_for_selection": False,
            "context_hint_and_oracle_metrics_excluded": True,
            "runtime_adoption_note": (
                "The lane is automatic in context, but this report is consumed-evidence "
                "postmortem output and is not a new sealed validation claim."
            ),
        },
        "lanes": {
            LANE_AUTOMATIC: _lane_payload(
                LANE_AUTOMATIC,
                sweep=automatic_sweep,
                context=automatic_context,
                truth_used=False,
            ),
            LANE_CONTEXT: _lane_payload(
                LANE_CONTEXT,
                sweep=context_sweep,
                context={
                    "key_hint_source": "human_or_externally_supplied_json",
                    "context_hint_path": context_pin["path"] if context_pin else None,
                },
                truth_used=False,
            ),
            LANE_ORACLE: _lane_payload(
                LANE_ORACLE,
                sweep=oracle_sweep,
                context={
                    "key_hint_source": "truth_derived_diagnostic_only",
                    "truth_derived_alteration_count": len(oracle_alterations),
                },
                truth_used=True,
            ),
        },
        "provenance": {
            "inputs": {
                "inference_jsonl": inference_pin,
                "truth_jsonl": truth_pin,
                "evaluation_report_json": evaluation_pin,
                "frozen_model_json": model_pin,
                "context_hint_json": context_pin,
            },
            "immutability": {
                "sealed_inputs_written": False,
                "original_inference_predictions_rewritten": False,
                "original_evaluation_truth_rewritten": False,
                "output_is_new_sibling": output_dir.parent == evaluation_report_path.parent.parent,
            },
            "access_audit": {
                "predictions_materialized_before_truth_read": True,
                "predictions_materialized_before_truth_derived_evaluation_report_read": True,
                "automatic_lane_used_key_hint": automatic_uses_key_hint,
                "automatic_lane_read_musicxml": False,
                "truth_used_for_automatic_selection": False,
            },
        },
        "artifacts": {
            "output_dir": str(output_dir),
            "report_json": str(output_dir / "report.json"),
            "report_markdown": str(output_dir / "report.md"),
        },
    }
    _write_create_once(output_dir, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inference_jsonl", type=Path)
    parser.add_argument("truth_jsonl", type=Path)
    parser.add_argument("evaluation_report_json", type=Path)
    parser.add_argument("frozen_model_json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--context-hints", type=Path)
    parser.add_argument(
        "--y-separation-staff-spaces",
        nargs="+",
        type=float,
        default=list(DEFAULT_Y_SEPARATION_GRID),
    )
    parser.add_argument(
        "--recovery-min-y-gap-staff-spaces",
        nargs="+",
        type=float,
        default=list(DEFAULT_RECOVERY_MIN_Y_GAP_GRID),
    )
    parser.add_argument(
        "--recovery-score-ratio",
        nargs="+",
        type=float,
        default=list(DEFAULT_RECOVERY_SCORE_RATIO_GRID),
    )
    parser.add_argument(
        "--recovery-max-y-gap-staff-spaces",
        type=float,
        default=DEFAULT_RECOVERY_MAX_Y_GAP_STAFF_SPACES,
    )
    args = parser.parse_args(argv)
    try:
        report = run_experiment(
            args.inference_jsonl,
            args.truth_jsonl,
            args.evaluation_report_json,
            args.frozen_model_json,
            output_dir=args.output_dir,
            context_hint_path=args.context_hints,
            y_separation_grid=args.y_separation_staff_spaces,
            recovery_min_y_gap_grid=args.recovery_min_y_gap_staff_spaces,
            recovery_score_ratio_grid=args.recovery_score_ratio,
            recovery_max_y_gap_staff_spaces=args.recovery_max_y_gap_staff_spaces,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(report["artifacts"]["report_json"])
    print(report["artifacts"]["report_markdown"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
