"""Detect sharp-like key-state changes in consumed measure crops.

This is a spike-only visual experiment. It deliberately identifies only a
sharp-like glyph near the structural left edge of a measure crop. It does not
map that glyph to a musical key and it does not read human labels while
making the prediction.

Example:
    .venv/bin/python scripts/experiments/spike_consumed_key_state_detector.py \
        --input-dir out/jaime-llanos_64_la-chata_pasillo_luis-a-calvo/\
vlm_melody_third_score_heldout/v2/system_007/crops \
        --out-dir /tmp/key-state-detector \
        --labels out/jaime-llanos_64_la-chata_pasillo_luis-a-calvo/\
vlm_melody_third_score_heldout/v2/system_007/context_hints_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw

SCHEMA_VERSION = 1
PREDICTION_SHARP = "sharp-like"
PREDICTION_UNKNOWN = "unknown"
STATE_EXPLICIT_CHANGE = "explicit_change"
STATE_INHERITED = "inherited"
STATE_UNKNOWN_INITIAL = "unknown_initial"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")
DEFAULT_CONSUMED_CROP_DIR = (
    Path(__file__).resolve().parents[2] / "out/jaime-llanos_64_la-chata_pasillo_luis-a-calvo/"
    "vlm_melody_third_score_heldout/v2/system_007/crops"
)
SHARP_DECISION_THRESHOLD = 0.33
MIN_SHARP_TRACK_GAP_SPACES = 0.35
BOUNDARY_TRACK_SUPPORT = 0.55
MIN_DOUBLE_BAR_GAP_SPACES = 0.15
MAX_DOUBLE_BAR_GAP_SPACES = 0.60


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: Sequence[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def _otsu_threshold(pixels: Iterable[int]) -> int:
    histogram = [0] * 256
    count = 0
    for value in pixels:
        histogram[max(0, min(255, int(value)))] += 1
        count += 1
    if count == 0:
        return 128
    total = sum(index * amount for index, amount in enumerate(histogram))
    background_count = 0
    background_sum = 0
    best_between = -1.0
    best_threshold = 128
    for threshold, amount in enumerate(histogram):
        background_count += amount
        if background_count == 0:
            continue
        foreground_count = count - background_count
        if foreground_count == 0:
            break
        background_sum += threshold * amount
        background_mean = background_sum / background_count
        foreground_mean = (total - background_sum) / foreground_count
        between = background_count * foreground_count * (background_mean - foreground_mean) ** 2
        if between > best_between:
            best_between = between
            best_threshold = threshold
    return max(48, min(220, best_threshold))


def _connected_components(mask: list[list[bool]]) -> list[tuple[int, int, int, int, int]]:
    height = len(mask)
    width = len(mask[0]) if height else 0
    visited = [[False] * width for _ in range(height)]
    components: list[tuple[int, int, int, int, int]] = []
    for y in range(height):
        for x in range(width):
            if not mask[y][x] or visited[y][x]:
                continue
            visited[y][x] = True
            queue = deque([(x, y)])
            left = right = x
            top = bottom = y
            area = 0
            while queue:
                current_x, current_y = queue.popleft()
                area += 1
                left = min(left, current_x)
                right = max(right, current_x)
                top = min(top, current_y)
                bottom = max(bottom, current_y)
                for delta_y in (-1, 0, 1):
                    for delta_x in (-1, 0, 1):
                        if delta_x == 0 and delta_y == 0:
                            continue
                        next_x = current_x + delta_x
                        next_y = current_y + delta_y
                        if (
                            0 <= next_x < width
                            and 0 <= next_y < height
                            and mask[next_y][next_x]
                            and not visited[next_y][next_x]
                        ):
                            visited[next_y][next_x] = True
                            queue.append((next_x, next_y))
            components.append((left, top, right, bottom, area))
    return components


def _cluster_indices(indices: Sequence[int], max_gap: int = 1) -> list[tuple[int, int]]:
    if not indices:
        return []
    clusters: list[tuple[int, int]] = []
    start = previous = indices[0]
    for value in indices[1:]:
        if value - previous > max_gap:
            clusters.append((start, previous))
            start = value
        previous = value
    clusters.append((start, previous))
    return clusters


def _staff_lines(row_counts: Sequence[int], width: int, height: int) -> list[int]:
    threshold = max(4, int(width * 0.35), int(_percentile(row_counts, 0.82)))
    indices = [index for index, count in enumerate(row_counts) if count >= threshold]
    clusters = _cluster_indices(indices)
    centers = [round((start + end) / 2) for start, end in clusters]
    if len(centers) < 5:
        return []
    best: tuple[float, list[int]] | None = None
    for start in range(len(centers) - 4):
        group = centers[start : start + 5]
        gaps = [right - left for left, right in zip(group, group[1:], strict=False)]
        if min(gaps) <= 0:
            continue
        spacing = statistics.mean(gaps)
        spread = statistics.pvariance(gaps) / max(1.0, spacing * spacing)
        if spacing < max(3.0, height * 0.025) or spacing > height * 0.35:
            continue
        score = spread + (abs(spacing - statistics.median(gaps)) / max(1.0, spacing))
        if best is None or score < best[0]:
            best = (score, group)
    return best[1] if best else centers[:5]


def _mask_from_image(image: Image.Image) -> tuple[list[list[bool]], int]:
    gray = image.convert("L")
    flattened_data = getattr(gray, "get_flattened_data", None)
    pixels_data = flattened_data() if callable(flattened_data) else gray.getdata()
    threshold = _otsu_threshold(list(pixels_data))
    pixels = gray.load()
    mask = [[pixels[x, y] <= threshold for x in range(gray.width)] for y in range(gray.height)]
    return mask, threshold


def _find_boundary(
    mask: list[list[bool]], staff_lines: Sequence[int], spacing: float
) -> tuple[int, dict[str, Any]]:
    height = len(mask)
    width = len(mask[0]) if height else 0
    top = max(0, round(staff_lines[0] - spacing * 0.4)) if staff_lines else 0
    bottom = min(height - 1, round(staff_lines[-1] + spacing * 0.4)) if staff_lines else height - 1
    staff_row_set = {
        row
        for line in staff_lines
        for row in range(max(0, line - max(1, round(spacing * 0.08))), min(height, line + 2))
    }
    scores: list[float] = []
    for x in range(width):
        usable = [y for y in range(top, bottom + 1) if y not in staff_row_set]
        scores.append(sum(mask[y][x] for y in usable) / max(1, len(usable)))
    search_limit = min(width, max(8, round(width * 0.22)))
    strong = [x for x in range(search_limit) if scores[x] >= BOUNDARY_TRACK_SUPPORT]
    tracks = []
    for start, end in _cluster_indices(strong):
        tracks.append(
            {
                "left": start,
                "right": end,
                "center": round((start + end) / 2, 3),
                "support": round(max(scores[start : end + 1]), 6),
            }
        )
    if not tracks:
        return 0, {
            "method": "crop_edge_fallback",
            "style": "unknown",
            "tracks": [],
            "bbox": {"left": 0, "right": 0},
            "support": 0.0,
        }
    first = tracks[0]
    style = "single_bar"
    boundary = first["right"]
    double_bar_gap = None
    if len(tracks) >= 2:
        second = tracks[1]
        double_bar_gap = (second["center"] - first["center"]) / max(1.0, spacing)
        if MIN_DOUBLE_BAR_GAP_SPACES <= double_bar_gap <= MAX_DOUBLE_BAR_GAP_SPACES:
            style = "double_bar"
            boundary = second["right"]
    return boundary, {
        "method": "staff_height_vertical_tracks",
        "style": style,
        "tracks": tracks,
        "bbox": {"left": first["left"], "right": boundary},
        "support": first["support"],
        "double_bar_gap_staff_spaces": (
            round(double_bar_gap, 6) if double_bar_gap is not None else None
        ),
    }


def _line_rows(staff_lines: Sequence[int], spacing: float, height: int) -> set[int]:
    radius = max(1, round(spacing * 0.08))
    return {
        row
        for line in staff_lines
        for row in range(max(0, line - radius), min(height, line + radius + 1))
    }


def _vertical_clusters(
    mask: list[list[bool]],
    *,
    x_start: int,
    x_end: int,
    y_start: int,
    y_end: int,
    staff_rows: set[int],
    spacing: float,
) -> list[tuple[int, int, float]]:
    support: list[float] = []
    usable_rows = [y for y in range(y_start, y_end + 1) if y not in staff_rows]
    for x in range(x_start, x_end + 1):
        support.append(sum(mask[y][x] for y in usable_rows) / max(1, len(usable_rows)))
    threshold = max(0.12, min(0.30, 5.0 / max(1.0, len(usable_rows))))
    indices = [x_start + offset for offset, value in enumerate(support) if value >= threshold]
    result = []
    for left, right in _cluster_indices(indices):
        values = support[left - x_start : right - x_start + 1]
        result.append((left, right, max(values)))
    return result


def _local_features(
    mask: list[list[bool]],
    left: int,
    right: int,
    top: int,
    bottom: int,
    staff_rows: set[int],
    spacing: float,
) -> dict[str, float]:
    width = right - left + 1
    height = bottom - top + 1
    usable_rows = [y for y in range(top, bottom + 1) if y not in staff_rows]
    dark = [(x, y) for y in range(top, bottom + 1) for x in range(left, right + 1) if mask[y][x]]
    density = len(dark) / max(1, width * height)
    vertical_support = sum(
        sum(mask[y][x] for y in usable_rows) / max(1, len(usable_rows))
        for x in range(left, right + 1)
    ) / max(1, width)
    row_run_values: list[int] = []
    for y in usable_rows:
        run = 0
        best_run = 0
        for x in range(left, right + 1):
            if mask[y][x]:
                run += 1
                best_run = max(best_run, run)
            else:
                run = 0
        row_run_values.append(best_run)
    horizontal_support = sum(
        value >= max(2, round(width * 0.35)) for value in row_run_values
    ) / max(1, len(row_run_values))
    return {
        "bbox_width_px": float(width),
        "bbox_height_px": float(height),
        "bbox_width_staff_spaces": width / max(1.0, spacing),
        "bbox_height_staff_spaces": height / max(1.0, spacing),
        "ink_density": density,
        "vertical_support": vertical_support,
        "horizontal_cross_support": horizontal_support,
        "usable_row_count": float(len(usable_rows)),
    }


def _candidate_rows(
    mask: list[list[bool]],
    *,
    boundary: int,
    staff_lines: Sequence[int],
    spacing: float,
) -> list[dict[str, Any]]:
    height = len(mask)
    width = len(mask[0]) if height else 0
    staff_rows = _line_rows(staff_lines, spacing, height)
    top = max(0, round(staff_lines[0] - spacing * 0.45)) if staff_lines else 0
    bottom = min(height - 1, round(staff_lines[-1] + spacing * 0.45)) if staff_lines else height - 1
    x_start = min(width - 1, boundary + max(1, round(spacing * 0.15)))
    x_end = min(width - 1, max(x_start, boundary + round(spacing * 4.5)))
    clusters = _vertical_clusters(
        mask,
        x_start=x_start,
        x_end=x_end,
        y_start=top,
        y_end=bottom,
        staff_rows=staff_rows,
        spacing=spacing,
    )
    candidates: list[dict[str, Any]] = []
    for index, first in enumerate(clusters):
        for second in clusters[index + 1 :]:
            first_center = (first[0] + first[1]) / 2
            second_center = (second[0] + second[1]) / 2
            gap = second_center - first_center
            if gap < spacing * 0.18 or gap > spacing * 1.8:
                continue
            candidate_left = first[0]
            candidate_right = second[1]
            features = _local_features(
                mask, candidate_left, candidate_right, top, bottom, staff_rows, spacing
            )
            pair_support = min(first[2], second[2])
            separation_prior = math.exp(-abs(gap / spacing - 0.75))
            one_track_penalty = abs(first[2] - second[2])
            score = (
                0.42 * pair_support
                + 0.34 * features["horizontal_cross_support"]
                + 0.18 * separation_prior
                - 0.12 * one_track_penalty
            )
            features.update(
                {
                    "left_track_support": first[2],
                    "right_track_support": second[2],
                    "track_gap_staff_spaces": gap / max(1.0, spacing),
                    "separation_prior": separation_prior,
                    "one_track_penalty": one_track_penalty,
                    "score": score,
                }
            )
            candidates.append(
                {
                    "candidate_id": f"c{len(candidates) + 1:03d}",
                    "bbox": {
                        "left": candidate_left,
                        "top": top,
                        "right": candidate_right,
                        "bottom": bottom,
                    },
                    "features": {key: round(value, 6) for key, value in features.items()},
                }
            )
    return sorted(candidates, key=lambda row: (-row["features"]["score"], row["candidate_id"]))


def _collapse_accepted_glyph_groups(accepted: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Collapse overlapping pair windows so one glyph is counted once."""
    ordered = sorted(
        accepted,
        key=lambda candidate: (
            int(candidate["bbox"]["left"]),
            int(candidate["bbox"]["right"]),
            str(candidate["candidate_id"]),
        ),
    )
    groups: list[dict[str, Any]] = []
    for candidate in ordered:
        bbox = candidate["bbox"]
        left = int(bbox["left"])
        right = int(bbox["right"])
        if not groups or left > groups[-1]["bbox"]["right"]:
            groups.append(
                {
                    "group_id": f"g{len(groups) + 1:03d}",
                    "candidate_ids": [str(candidate["candidate_id"])],
                    "bbox": {
                        "left": left,
                        "top": int(bbox["top"]),
                        "right": right,
                        "bottom": int(bbox["bottom"]),
                    },
                }
            )
            continue
        group = groups[-1]
        group["candidate_ids"].append(str(candidate["candidate_id"]))
        group["bbox"]["left"] = min(group["bbox"]["left"], left)
        group["bbox"]["top"] = min(group["bbox"]["top"], int(bbox["top"]))
        group["bbox"]["right"] = max(group["bbox"]["right"], right)
        group["bbox"]["bottom"] = max(group["bbox"]["bottom"], int(bbox["bottom"]))
    return groups


def _conservative_sharp_fifths(glyph_groups: Sequence[Mapping[str, Any]]) -> int | None:
    """Return +1 only for one unambiguous sharp group; do not infer flats."""
    return 1 if len(glyph_groups) == 1 else None


def detect_key_state(image_path: Path) -> dict[str, Any]:
    """Return a deterministic, truth-blind visual prediction for one crop."""
    image_path = image_path.expanduser().resolve()
    with Image.open(image_path) as loaded:
        image = loaded.convert("RGB")
    mask, threshold = _mask_from_image(image)
    height = len(mask)
    width = len(mask[0]) if height else 0
    row_counts = [sum(row) for row in mask]
    staff_lines = _staff_lines(row_counts, width, height)
    if len(staff_lines) >= 2:
        spacing = statistics.mean(
            right - left for left, right in zip(staff_lines, staff_lines[1:], strict=False)
        )
    else:
        spacing = max(4.0, height / 8.0)
        staff_lines = [round(height * fraction) for fraction in (0.25, 0.375, 0.5, 0.625, 0.75)]
    boundary, boundary_features = _find_boundary(mask, staff_lines, spacing)
    candidates = _candidate_rows(mask, boundary=boundary, staff_lines=staff_lines, spacing=spacing)
    accepted = (
        [
            candidate
            for candidate in candidates
            if candidate["features"]["score"] >= SHARP_DECISION_THRESHOLD
        ]
        if boundary_features["style"] == "double_bar"
        else []
    )
    # Keep the decision conservative: a sharp-like glyph needs cross-stroke
    # support and two reasonably balanced vertical tracks.
    accepted = [
        candidate
        for candidate in accepted
        if candidate["features"]["horizontal_cross_support"] >= 0.08
        and candidate["features"]["left_track_support"] >= 0.12
        and candidate["features"]["right_track_support"] >= 0.12
        and candidate["features"]["track_gap_staff_spaces"] >= MIN_SHARP_TRACK_GAP_SPACES
    ]
    glyph_groups = _collapse_accepted_glyph_groups(accepted)
    fifths = _conservative_sharp_fifths(glyph_groups) if accepted else None
    prediction = PREDICTION_SHARP if accepted else PREDICTION_UNKNOWN
    top_candidate = accepted[0] if accepted else (candidates[0] if candidates else None)
    return {
        "schema_version": SCHEMA_VERSION,
        "input": {"path": str(image_path), "sha256": _sha256(image_path)},
        "image": {"width_px": width, "height_px": height},
        "threshold": threshold,
        "staff_lines_y_px": staff_lines,
        "staff_spacing_px": round(spacing, 6),
        "structural_boundary": {"x_px": boundary, **boundary_features},
        "search_region": {
            "left_px": min(width, boundary + max(1, round(spacing * 0.15))),
            "right_px": min(width, boundary + round(spacing * 4.5)),
        },
        "candidates": candidates,
        "accepted_candidate_ids": [candidate["candidate_id"] for candidate in accepted],
        "glyph_groups": glyph_groups,
        "observed_change": prediction,
        "predicted_change": prediction,
        "predicted_signature_family": "sharp" if accepted else None,
        "fifths": fifths,
        "fifths_status": "confirmed_explicit" if fifths is not None else "unknown",
        "top_candidate_id": top_candidate["candidate_id"] if top_candidate else None,
        "decision_threshold": SHARP_DECISION_THRESHOLD,
        "boundary_gate_passed": boundary_features["style"] == "double_bar",
        "truth_used_for_prediction": False,
    }


def propagate_states(predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Propagate only explicit detector changes through an ordered crop sequence.

    A crop with no observed change is unknown until an explicit change has been
    detected earlier in the sequence. It then inherits only that confirmed
    visual family; it never becomes an implied C-major/default state.
    """
    inherited: dict[str, Any] | None = None
    states: list[dict[str, Any]] = []
    for sequence_index, prediction in enumerate(predictions):
        row = dict(prediction)
        observed_change = prediction.get("observed_change", prediction.get("predicted_change"))
        if observed_change == PREDICTION_SHARP:
            source = {
                "input": prediction["input"]["path"],
                "candidate_id": prediction.get("top_candidate_id"),
            }
            state = {
                "kind": STATE_EXPLICIT_CHANGE,
                "signature_family": prediction.get("predicted_signature_family"),
                "fifths": prediction.get("fifths"),
                "pitch_mapping_ready": prediction.get("fifths") is not None,
                "change": observed_change,
                "source": source,
            }
            inherited = {
                "signature_family": state["signature_family"],
                "fifths": state["fifths"],
                "source": source,
            }
        elif inherited is not None:
            state = {
                "kind": STATE_INHERITED,
                "signature_family": inherited["signature_family"],
                "fifths": inherited["fifths"],
                "pitch_mapping_ready": inherited["fifths"] is not None,
                "change": None,
                "source": inherited["source"],
            }
        else:
            state = {
                "kind": STATE_UNKNOWN_INITIAL,
                "signature_family": None,
                "fifths": None,
                "pitch_mapping_ready": False,
                "change": None,
                "source": None,
            }
        row["sequence_index"] = sequence_index
        row["state"] = state
        states.append(row)
    return states


def _measure_index(input_path: Path) -> int:
    stem = input_path.stem
    if not stem.startswith("measure_"):
        raise ValueError(f"Cannot derive measure index from input name: {input_path.name}")
    suffix = stem.removeprefix("measure_").split("_", 1)[0]
    try:
        measure = int(suffix)
    except ValueError as exc:
        raise ValueError(f"Cannot derive measure index from input name: {input_path.name}") from exc
    if measure <= 0:
        raise ValueError(f"Measure index must be positive: {input_path.name}")
    return measure


def context_hints_from_states(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Convert confirmed visual changes into the existing stateful key-hint contract."""
    events = []
    for prediction in predictions:
        state = prediction.get("state")
        if not isinstance(state, Mapping) or state.get("kind") != STATE_EXPLICIT_CHANGE:
            continue
        fifths = state.get("fifths")
        if fifths is None:
            continue
        input_path = Path(str(prediction["input"]["path"]))
        events.append(
            {
                "start_measure": _measure_index(input_path),
                "key_hint": {"fifths": int(fifths)},
                "source": {
                    "kind": "automatic_visual_key_change",
                    "image": str(input_path),
                    "sha256": str(prediction["input"]["sha256"]),
                    "candidate_id": prediction.get("top_candidate_id"),
                },
            }
        )
    events.sort(key=lambda event: int(event["start_measure"]))
    if len({event["start_measure"] for event in events}) != len(events):
        raise ValueError("Visual key-state output has duplicate start measures")
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "automatic_visual_key_state_detector",
        "truth_used": False,
        "events": events,
    }


def _label_for_input(labels: Mapping[str, Any], input_path: Path) -> str | None:
    rows = labels.get("labels", labels.get("events", labels))
    if isinstance(rows, Mapping):
        rows = [rows]
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return None
    name = input_path.name
    stem = input_path.stem
    measure_index = None
    if stem.startswith("measure_"):
        try:
            measure_index = int(stem.removeprefix("measure_"))
        except ValueError:
            measure_index = None
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        target = row.get("input", row.get("image", row.get("filename")))
        if target is not None and target not in (
            name,
            stem,
            str(input_path),
            str(input_path.resolve()),
        ):
            continue
        if target is None and measure_index != row.get("start_measure"):
            continue
        value = row.get("expected_change", row.get("label", row.get("predicted_change")))
        if value is None and row.get("key_hint") is not None:
            value = (
                PREDICTION_SHARP if "sharp" in str(row["key_hint"]).lower() else PREDICTION_UNKNOWN
            )
        if isinstance(value, Mapping):
            value = value.get("kind", value.get("family"))
        return str(value) if value is not None else None
    return None


def evaluate_predictions(
    predictions: Sequence[Mapping[str, Any]], labels: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    if labels is None:
        return None
    compared = []
    for prediction in predictions:
        label = _label_for_input(labels, Path(str(prediction["input"]["path"])))
        if label is None:
            continue
        predicted = str(prediction["predicted_change"])
        compared.append(
            {
                "input": prediction["input"]["path"],
                "predicted": predicted,
                "expected": label,
                "match": predicted == label,
            }
        )
    return {
        "compared_count": len(compared),
        "matches": sum(row["match"] for row in compared),
        "accuracy": (
            round(sum(row["match"] for row in compared) / len(compared), 6) if compared else None
        ),
        "rows": compared,
        "truth_used_for_prediction": False,
    }


def _load_labels(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("labels JSON must be an object")
    return value


def _input_paths(input_path: Path | None, input_dir: Path | None) -> list[Path]:
    if input_path is not None and input_dir is not None:
        raise ValueError("pass either an input path or --input-dir, not both")
    if input_path is not None:
        paths = [input_path]
    else:
        directory = input_dir or DEFAULT_CONSUMED_CROP_DIR
        paths = sorted(
            path
            for path in directory.expanduser().resolve().iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        )
    if not paths:
        raise ValueError("no input images found")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return [path.expanduser().resolve() for path in paths]


def _draw_overlay(prediction: Mapping[str, Any], output_path: Path) -> None:
    source_path = Path(str(prediction["input"]["path"]))
    with Image.open(source_path) as loaded:
        image = loaded.convert("RGB")
    draw = ImageDraw.Draw(image)
    boundary = int(prediction["structural_boundary"]["x_px"])
    draw.line((boundary, 0, boundary, image.height - 1), fill=(40, 100, 220), width=1)
    accepted = set(prediction["accepted_candidate_ids"])
    for candidate in prediction["candidates"]:
        bbox = candidate["bbox"]
        color = (30, 170, 70) if candidate["candidate_id"] in accepted else (230, 150, 30)
        draw.rectangle(
            (bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]), outline=color, width=2
        )
        draw.text(
            (bbox["left"] + 2, max(0, bbox["top"] - 10)), candidate["candidate_id"], fill=color
        )
    for group in prediction.get("glyph_groups", []):
        bbox = group["bbox"]
        draw.rectangle(
            (bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]),
            outline=(0, 120, 40),
            width=3,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def analyze_inputs(
    input_paths: Sequence[Path], out_dir: Path, labels: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = [detect_key_state(path) for path in input_paths]
    predictions_with_state = propagate_states(predictions)
    artifacts = []
    for prediction in predictions_with_state:
        source = Path(str(prediction["input"]["path"]))
        overlay_path = out_dir / f"{source.stem}_key_state_overlay.png"
        _draw_overlay(prediction, overlay_path)
        prediction_with_artifact = dict(prediction)
        prediction_with_artifact["overlay_path"] = str(overlay_path)
        artifacts.append(prediction_with_artifact)
    context_hints = context_hints_from_states(artifacts)
    context_hints_path = out_dir / "context_hints.json"
    context_hints_path.write_text(
        json.dumps(context_hints, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "spike_consumed_key_state_detector",
        "truth_used_for_prediction": False,
        "detector_config": {
            "sharp_decision_threshold": SHARP_DECISION_THRESHOLD,
            "min_sharp_track_gap_staff_spaces": MIN_SHARP_TRACK_GAP_SPACES,
            "boundary_track_support": BOUNDARY_TRACK_SUPPORT,
            "min_double_bar_gap_staff_spaces": MIN_DOUBLE_BAR_GAP_SPACES,
            "max_double_bar_gap_staff_spaces": MAX_DOUBLE_BAR_GAP_SPACES,
            "requires_boundary_style": "double_bar",
            "fifths_policy": "one_unambiguous_sharp_group_only",
            "flat_support": False,
            "decision_output": "sharp-like_or_unknown",
            "truth_used_for_prediction": False,
        },
        "state_model": {
            "explicit_change": STATE_EXPLICIT_CHANGE,
            "inherited_after_explicit_change": STATE_INHERITED,
            "unknown_before_first_explicit_change": STATE_UNKNOWN_INITIAL,
            "default_key_assumption": None,
            "pitch_mapping_requires_exact_fifths": True,
            "flat_support": False,
        },
        "measures": artifacts,
        "context_hints": context_hints,
        "evaluation": evaluate_predictions(artifacts, labels),
        "artifacts": {
            "context_hints": str(context_hints_path),
            "report": str(out_dir / "report.json"),
        },
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None, help="One measure crop to analyze.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory of measure crops; defaults to the consumed La Chata crop directory.",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="Directory for JSON and overlays."
    )
    parser.add_argument(
        "--labels", type=Path, default=None, help="Optional labels JSON used only for evaluation."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        paths = _input_paths(args.input, args.input_dir)
        labels = _load_labels(args.labels)
        report = analyze_inputs(paths, args.out_dir, labels)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 1
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir.resolve()),
                "measure_count": len(report["measures"]),
                "evaluation": report["evaluation"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
