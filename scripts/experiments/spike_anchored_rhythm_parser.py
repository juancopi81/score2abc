"""Bounded HITL upper-bound experiment for visual rhythm and rest parsing.

Human-reviewed notehead centers and pitches are oracle anchors. They are not an
automatic-recognition result. Inference uses only those anchors, the raw measure
image, staff geometry, and the request's allowed meter context. Benchmark truth
is loaded only after every arm's predictions and overlays have been generated.

Example:
    uv run python scripts/experiments/spike_anchored_rhythm_parser.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "out"
DEFAULT_REVIEWS_DIR = REPO_ROOT / "tests/fixtures/vlm_melody/notehead_reviews"
DEFAULT_SLUG = "jaime-llanos_12_aviador_pasillo_fulgencio-garcia"
SYSTEM_INDEX = 1
MEASURES = (1, 2, 3, 4)

# Fixed before benchmark evaluation. Values are normalized by median staff spacing.
INK_THRESHOLD = 200
STEM_MAX_REACH_SPACING = 2.8
STEM_X_REACH_SPACING = 0.6
STEM_MAX_GAP_PX = 2
STAFF_RUN_MIN_SPACING = 0.6
FLAG_DENSITY_THRESHOLD = 0.055
DOT_AREA_THRESHOLD = 0.05
SIMULTANEOUS_X_TOLERANCE_SPACING = 0.35
REST_HEIGHT_RANGE_SPACING = (1.5, 2.55)
REST_WIDTH_RANGE_SPACING = (0.5, 1.4)
REST_AREA_THRESHOLD = 0.20
METER_EDIT_PENALTY = 0.75


@dataclass(frozen=True)
class StemRun:
    direction: str
    side: str
    x: int
    near_y: int
    tip_y: int
    length_px: int


@dataclass
class PreparedMeasure:
    request: dict[str, Any]
    review_path: Path
    image_path: Path
    image: Image.Image
    staff_lines: list[int]
    staff_spacing: float
    groups: list[dict[str, Any]]
    anchor_features: list[dict[str, Any]]
    rest_features: list[dict[str, Any]]
    arms: dict[str, dict[str, Any]]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = run_experiment(
            args.out_dir,
            slug=args.slug,
            reviews_dir=args.reviews_dir,
            output_dir=args.output_dir,
        )
    except (FileNotFoundError, KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(report["artifacts"]["report_json"])
    print(report["artifacts"]["report_markdown"])
    print(f"gate: {report['decision']['status']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--reviews-dir", type=Path, default=DEFAULT_REVIEWS_DIR)
    parser.add_argument("--output-dir", type=Path)
    return parser


def run_experiment(
    out_dir: Path,
    *,
    slug: str = DEFAULT_SLUG,
    reviews_dir: Path = DEFAULT_REVIEWS_DIR,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    benchmark_dir = out_dir / slug / "vlm_melody_event_benchmark"
    output_dir = output_dir or benchmark_dir / "anchored_rhythm_parser"
    output_dir.mkdir(parents=True, exist_ok=True)

    requests = _read_jsonl(benchmark_dir / "development/requests.jsonl")
    requests = [
        row
        for row in requests
        if int(row["identity"]["system_index"]) == SYSTEM_INDEX
        and int(row["identity"]["system_measure_index"]) in MEASURES
    ]
    if len(requests) != len(MEASURES):
        raise ValueError(f"Expected {len(MEASURES)} development requests, got {len(requests)}")

    # Inference boundary: no benchmark truth path or canonical event object is accepted here.
    prepared = _prepare_inference_rows(
        requests,
        out_dir=out_dir,
        slug=slug,
        reviews_dir=reviews_dir,
        output_dir=output_dir,
    )
    prediction_paths = _write_prediction_arms(prepared, output_dir)

    # Evaluation boundary: this is the first benchmark-truth read in the experiment.
    truth_rows = _read_jsonl(benchmark_dir / "development/truth.jsonl")
    truth_by_measure = {
        int(row["identity"]["system_measure_index"]): row
        for row in truth_rows
        if int(row["identity"]["system_index"]) == SYSTEM_INDEX
        and int(row["identity"]["system_measure_index"]) in MEASURES
    }
    if set(truth_by_measure) != set(MEASURES):
        raise ValueError("Benchmark truth does not contain all S1M1-4 measures")

    arm_metrics: dict[str, dict[str, Any]] = {}
    per_measure: list[dict[str, Any]] = []
    for item in prepared:
        measure = int(item.request["identity"]["system_measure_index"])
        truth = truth_by_measure[measure]
        evaluations = {
            arm_name: evaluate_hypothesis(arm, truth) for arm_name, arm in item.arms.items()
        }
        per_measure.append(
            {
                "measure": measure,
                "identity": item.request["identity"],
                "image": _repo_relative(item.image_path),
                "oracle_anchor_count": sum(len(group["anchors"]) for group in item.groups),
                "simultaneous_group_count": len(item.groups),
                "anchor_features": item.anchor_features,
                "residual_rest_features": item.rest_features,
                "predictions": item.arms,
                "evaluation": evaluations,
                "overlay": _repo_relative(output_dir / f"measure_{measure:03d}_overlay.png"),
            }
        )

    for arm_name in prepared[0].arms:
        arm_metrics[arm_name] = aggregate_metrics(
            [row["evaluation"][arm_name] for row in per_measure]
        )

    decoded = arm_metrics["meter_decoded"]
    gate_a = decoded["exact_measure_count"] >= 3
    gate_b = _passes_joint_metric_gate(decoded)
    gate_passed = gate_a or gate_b
    report_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    report = {
        "schema_version": 2,
        "kind": "anchored_rhythm_parser_spike",
        "slug": slug,
        "system_index": SYSTEM_INDEX,
        "measures": list(MEASURES),
        "experiment_semantics": {
            "recognition_arm": "explicit_hitl_upper_bound",
            "oracle_inputs": "promoted human-reviewed notehead centers and corrected pitches",
            "automatic_notehead_recognition_claimed": False,
            "inference_duration_semantics": "visual glyph duration in quarter-note beats",
            "evaluation_duration_semantics": (
                "benchmark canonical sounding-event duration in quarter-note beats"
            ),
            "duration_accuracy_denominator": (
                "per measure max(predicted note groups, truth note groups); unmatched "
                "groups are wrong"
            ),
            "source_musicxml_read_directly": False,
            "semantic_caveat": (
                "Ties, ornaments, or other notation-to-sound transformations can make visual "
                "glyph duration differ from canonical sounding duration; this bounded slice "
                "scores the visual hypotheses against the benchmark's sounding-event contract."
            ),
        },
        "leakage_audit": {
            "ground_truth_available_during_inference": False,
            "order": (
                "load requests/review anchors/images; generate features and all predictions; "
                "write prediction JSONL and overlays; then load development truth"
            ),
            "truth_used_for_threshold_selection_at_runtime": False,
            "production_code_changed": False,
        },
        "fixed_parameters": {
            "ink_threshold": INK_THRESHOLD,
            "flag_density_threshold": FLAG_DENSITY_THRESHOLD,
            "dot_area_threshold_staff_squared": DOT_AREA_THRESHOLD,
            "simultaneous_x_tolerance_staff_spacing": SIMULTANEOUS_X_TOLERANCE_SPACING,
            "rest_height_range_staff_spacing": list(REST_HEIGHT_RANGE_SPACING),
            "rest_width_range_staff_spacing": list(REST_WIDTH_RANGE_SPACING),
            "rest_area_threshold_staff_squared": REST_AREA_THRESHOLD,
            "meter_edit_penalty": METER_EDIT_PENALTY,
        },
        "arms": {
            "layout_meter_control": {
                "uses_pixels": False,
                "uses_oracle_anchor_layout": True,
                "uses_meter": True,
                "description": "Equal-duration anchor groups; no inferred rests or glyph features.",
            },
            "visual_only": {
                "uses_pixels": True,
                "uses_meter_decoder": False,
                "description": "Stem/flag, dot, and residual-rest hypotheses in x order.",
            },
            "meter_decoded": {
                "uses_pixels": True,
                "uses_meter_decoder": True,
                "description": "Visual hypotheses with conservative exact-3/4 repair.",
            },
        },
        "per_measure": per_measure,
        "aggregate": arm_metrics,
        "decision": {
            "status": "pass" if gate_passed else "fail",
            "material_gate": (
                ">=3/4 exact measures OR duration accuracy >=0.85 and rest F1 >=0.8 "
                "with no predicted note-group overproduction"
            ),
            "exact_measure_gate": {
                "observed": decoded["exact_measure_count"],
                "required": 3,
                "passed": gate_a,
            },
            "joint_metric_gate": {
                "duration_accuracy_observed": decoded["duration_accuracy"],
                "duration_accuracy_required": 0.85,
                "rest_f1_observed": decoded["rest_f1"],
                "rest_f1_required": 0.8,
                "predicted_note_group_count": decoded["predicted_note_group_count"],
                "truth_note_group_count": decoded["truth_note_group_count"],
                "overproduced_measure_count": decoded["overproduced_measure_count"],
                "requires_no_note_group_overproduction": True,
                "passed": gate_b,
            },
        },
        "artifacts": {
            "report_json": _repo_relative(report_path),
            "report_markdown": _repo_relative(markdown_path),
            "prediction_jsonl": {
                name: _repo_relative(path) for name, path in prediction_paths.items()
            },
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown_report(report, markdown_path)
    return report


def _prepare_inference_rows(
    requests: Sequence[dict[str, Any]],
    *,
    out_dir: Path,
    slug: str,
    reviews_dir: Path,
    output_dir: Path,
) -> list[PreparedMeasure]:
    prepared: list[PreparedMeasure] = []
    for request in sorted(requests, key=lambda row: int(row["identity"]["system_measure_index"])):
        identity = request["identity"]
        measure = int(identity["system_measure_index"])
        review_path = reviews_dir / (f"{slug}_system_{SYSTEM_INDEX:03d}_measure_{measure:03d}.json")
        review = _load_json(review_path)
        _validate_review_identity(review, identity)
        image_path = _resolve_input_path(review["source"]["image_path"], out_dir=out_dir)
        _validate_sha256(image_path, str(review["source"]["image_sha256"]))
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")

        staff_lines = [int(value) for value in request["staff_geometry"]["raw_staff_lines_y_px"]]
        spacing = staff_spacing(staff_lines)
        anchors = [dict(item) for item in review["final_noteheads"]]
        groups = group_simultaneous_heads(anchors, spacing)
        features = extract_anchor_features(image, anchors, staff_lines)
        rest_features = extract_residual_rest_features(image, groups, staff_lines)
        expected_beats = float(request["allowed_context"]["expected_measure_beats"])
        allow_pickup = bool(request["allowed_context"].get("allow_pickup", False))

        visual_symbols = build_visual_symbols(groups, features, rest_features)
        visual = symbols_to_hypothesis(
            visual_symbols,
            identity=identity,
            decoder_status="not_applied",
        )
        decoded_symbols, decoder_status = decode_meter(
            visual_symbols,
            expected_beats=expected_beats,
            allow_pickup=allow_pickup,
        )
        decoded = symbols_to_hypothesis(
            decoded_symbols,
            identity=identity,
            decoder_status=decoder_status,
        )
        control_symbols = build_layout_meter_control(
            groups,
            expected_beats=expected_beats,
            allow_pickup=allow_pickup,
        )
        control = symbols_to_hypothesis(
            control_symbols,
            identity=identity,
            decoder_status="layout_meter_only",
        )
        arms = {
            "layout_meter_control": control,
            "visual_only": visual,
            "meter_decoded": decoded,
        }
        item = PreparedMeasure(
            request=request,
            review_path=review_path,
            image_path=image_path,
            image=image,
            staff_lines=staff_lines,
            staff_spacing=spacing,
            groups=groups,
            anchor_features=features,
            rest_features=rest_features,
            arms=arms,
        )
        _write_overlay(item, output_dir / f"measure_{measure:03d}_overlay.png")
        prepared.append(item)
    return prepared


def staff_spacing(staff_lines: Sequence[int]) -> float:
    if len(staff_lines) != 5:
        raise ValueError(f"Expected five staff lines, got {staff_lines!r}")
    gaps = [second - first for first, second in zip(staff_lines, staff_lines[1:], strict=False)]
    spacing = float(statistics.median(gaps))
    if spacing <= 0:
        raise ValueError(f"Invalid staff-line geometry: {staff_lines!r}")
    return spacing


def group_simultaneous_heads(
    anchors: Sequence[Mapping[str, Any]],
    spacing: float,
    *,
    tolerance_spacing: float = SIMULTANEOUS_X_TOLERANCE_SPACING,
) -> list[dict[str, Any]]:
    """Group oracle heads that share an x position into one rhythmic event."""
    tolerance = tolerance_spacing * spacing
    ordered = sorted(
        (deepcopy(dict(anchor)) for anchor in anchors),
        key=lambda item: (float(item["center"]["x"]), int(item.get("order", 0))),
    )
    groups: list[dict[str, Any]] = []
    for anchor in ordered:
        x = float(anchor["center"]["x"])
        if groups and abs(x - float(groups[-1]["center_x"])) <= tolerance:
            groups[-1]["anchors"].append(anchor)
            xs = [float(item["center"]["x"]) for item in groups[-1]["anchors"]]
            groups[-1]["center_x"] = round(statistics.mean(xs), 3)
        else:
            groups.append({"center_x": round(x, 3), "anchors": [anchor]})
    for index, group in enumerate(groups, start=1):
        group["group_id"] = f"g{index:03d}"
        group["pitches"] = [str(anchor["pitch"]) for anchor in group["anchors"]]
    return groups


def extract_anchor_features(
    image: Image.Image,
    anchors: Sequence[Mapping[str, Any]],
    staff_lines: Sequence[int],
) -> list[dict[str, Any]]:
    """Measure staff-normalized local stroke evidence around oracle anchors."""
    spacing = staff_spacing(staff_lines)
    raw_ink = _ink_matrix(image)
    residual_ink = _suppress_staff_runs(raw_ink, staff_lines, spacing)
    features = []
    for anchor in anchors:
        center = anchor["center"]
        x = float(center["x"])
        y = float(center["y"])
        stem, directional = _extract_stem_run(raw_ink, x, y, spacing)
        flag = _extract_flag_connectivity(residual_ink, stem, spacing)
        dot = _extract_dot_evidence(residual_ink, x, y, spacing)
        features.append(
            {
                "order": int(anchor.get("order", len(features) + 1)),
                "pitch": str(anchor["pitch"]),
                "center": {"x": round(x, 3), "y": round(y, 3)},
                "stem": {
                    "direction": stem.direction,
                    "side": stem.side,
                    "x": stem.x,
                    "near_y": stem.near_y,
                    "tip_y": stem.tip_y,
                    "length_px": stem.length_px,
                    "length_staff_spacing": round(stem.length_px / spacing, 6),
                    "up_run_staff_spacing": round(directional["up"] / spacing, 6),
                    "down_run_staff_spacing": round(directional["down"] / spacing, 6),
                },
                "beam_flag": flag,
                "dot": dot,
            }
        )
    return features


def _extract_stem_run(
    ink: Sequence[Sequence[bool]],
    center_x: float,
    center_y: float,
    spacing: float,
) -> tuple[StemRun, dict[str, int]]:
    height = len(ink)
    width = len(ink[0]) if height else 0
    x_min = max(0, round(center_x - STEM_X_REACH_SPACING * spacing))
    x_max = min(width - 1, round(center_x + STEM_X_REACH_SPACING * spacing))
    candidates: list[tuple[int, float, int, str, int, int]] = []
    directional = {"up": 0, "down": 0}
    for x in range(x_min, x_max + 1):
        for direction in ("up", "down"):
            if direction == "up":
                ys = list(
                    range(
                        max(0, round(center_y - STEM_MAX_REACH_SPACING * spacing)),
                        min(height, round(center_y + 0.15 * spacing) + 1),
                    )
                )[::-1]
            else:
                ys = list(
                    range(
                        max(0, round(center_y - 0.15 * spacing)),
                        min(height, round(center_y + STEM_MAX_REACH_SPACING * spacing) + 1),
                    )
                )
            values = [
                (
                    y,
                    any(ink[y][probe_x] for probe_x in range(max(0, x - 1), min(width, x + 2))),
                )
                for y in ys
            ]
            length, start, end = _longest_gapped_run(values, max_gap=STEM_MAX_GAP_PX)
            if start is None or end is None:
                continue
            near_distance = min(abs(start - center_y), abs(end - center_y))
            if near_distance > 0.65 * spacing:
                continue
            directional[direction] = max(directional[direction], length)
            candidates.append((length, -abs(x - center_x), x, direction, start, end))
    if not candidates:
        fallback = StemRun(
            "unknown", "center", round(center_x), round(center_y), round(center_y), 0
        )
        return fallback, directional
    length, _, x, direction, near_y, tip_y = max(candidates)
    side = "left" if x < center_x else "right" if x > center_x else "center"
    return StemRun(direction, side, x, near_y, tip_y, length), directional


def _longest_gapped_run(
    values: Sequence[tuple[int, bool]], *, max_gap: int
) -> tuple[int, int | None, int | None]:
    best = (0, None, None)
    start: int | None = None
    last: int | None = None
    gap = 0
    for coordinate, present in values:
        if present:
            if start is None:
                start = coordinate
            last = coordinate
            gap = 0
        elif start is not None:
            gap += 1
            if gap > max_gap:
                assert last is not None
                candidate = (abs(last - start) + 1, start, last)
                if candidate[0] > best[0]:
                    best = candidate
                start = None
                last = None
                gap = 0
    if start is not None and last is not None:
        candidate = (abs(last - start) + 1, start, last)
        if candidate[0] > best[0]:
            best = candidate
    return best


def _extract_flag_connectivity(
    residual_ink: Sequence[Sequence[bool]], stem: StemRun, spacing: float
) -> dict[str, Any]:
    height = len(residual_ink)
    width = len(residual_ink[0]) if height else 0
    if stem.direction == "up":
        y_min = round(stem.tip_y - 0.15 * spacing)
        y_max = round(stem.tip_y + 1.25 * spacing)
    else:
        y_min = round(stem.tip_y - 1.25 * spacing)
        y_max = round(stem.tip_y + 0.15 * spacing)
    x_min = round(stem.x + 0.12 * spacing)
    x_max = round(stem.x + 1.5 * spacing)
    x_min, x_max, y_min, y_max = _clip_box(x_min, x_max, y_min, y_max, width=width, height=height)
    area = max(1, (x_max - x_min + 1) * (y_max - y_min + 1))
    ink_count = sum(
        residual_ink[y][x] for y in range(y_min, y_max + 1) for x in range(x_min, x_max + 1)
    )
    row_peak = max(
        (sum(residual_ink[y][x] for x in range(x_min, x_max + 1)) for y in range(y_min, y_max + 1)),
        default=0,
    )
    density = ink_count / area
    return {
        "right_tip_box": {
            "left": x_min,
            "top": y_min,
            "right": x_max,
            "bottom": y_max,
        },
        "ink_count": ink_count,
        "density": round(density, 6),
        "row_peak_staff_spacing": round(row_peak / spacing, 6),
        "present": density >= FLAG_DENSITY_THRESHOLD,
    }


def _extract_dot_evidence(
    residual_ink: Sequence[Sequence[bool]], center_x: float, center_y: float, spacing: float
) -> dict[str, Any]:
    height = len(residual_ink)
    width = len(residual_ink[0]) if height else 0
    x_min, x_max, y_min, y_max = _clip_box(
        round(center_x + 0.55 * spacing),
        round(center_x + 1.15 * spacing),
        round(center_y - 0.65 * spacing),
        round(center_y + 0.65 * spacing),
        width=width,
        height=height,
    )
    components = _components_in_box(residual_ink, (x_min, y_min, x_max, y_max))
    eligible = [
        component
        for component in components
        if component["width_px"] <= 0.65 * spacing
        and component["height_px"] <= 0.7 * spacing
        and component["width_px"] >= 2
        and component["height_px"] >= 2
    ]
    best = max(eligible, key=lambda item: (item["area_px"], -item["bbox"]["left"]), default=None)
    area_normalized = 0.0 if best is None else float(best["area_px"]) / (spacing * spacing)
    return {
        "search_box": {"left": x_min, "top": y_min, "right": x_max, "bottom": y_max},
        "component": best,
        "area_staff_squared": round(area_normalized, 6),
        "present": area_normalized >= DOT_AREA_THRESHOLD,
    }


def extract_residual_rest_features(
    image: Image.Image,
    groups: Sequence[Mapping[str, Any]],
    staff_lines: Sequence[int],
) -> list[dict[str, Any]]:
    """Find unanchored rest-like components in leading and internal anchor gaps."""
    if not groups:
        return []
    spacing = staff_spacing(staff_lines)
    residual_ink = _suppress_staff_runs(_ink_matrix(image), staff_lines, spacing)
    height = len(residual_ink)
    width = len(residual_ink[0])
    group_xs = [float(group["center_x"]) for group in groups]
    windows: list[tuple[str, int, int]] = [
        (
            "leading",
            max(0, round(group_xs[0] - 1.9 * spacing)),
            max(0, round(group_xs[0] - 0.6 * spacing)),
        )
    ]
    for index, (left_x, right_x) in enumerate(zip(group_xs, group_xs[1:], strict=False), start=1):
        windows.append(
            (
                f"internal_after_g{index:03d}",
                round(left_x + 0.75 * spacing),
                round(right_x - 0.6 * spacing),
            )
        )
    y_min = max(0, round(staff_lines[0] - 0.4 * spacing))
    y_max = min(height - 1, round(staff_lines[-1] + 0.4 * spacing))
    detections: list[dict[str, Any]] = []
    for role, raw_x_min, raw_x_max in windows:
        x_min = max(0, min(width - 1, raw_x_min))
        x_max = max(0, min(width - 1, raw_x_max))
        if x_max - x_min < max(3, round(0.25 * spacing)):
            continue
        window_width = x_max - x_min + 1
        candidates = []
        components = _components_in_box(residual_ink, (x_min, y_min, x_max, y_max))
        reconnected = _merge_staff_split_components(components, spacing)
        for component in [*components, *reconnected]:
            width_spacing = component["width_px"] / spacing
            height_spacing = component["height_px"] / spacing
            area_spacing = component["area_px"] / (spacing * spacing)
            if not REST_WIDTH_RANGE_SPACING[0] <= width_spacing <= REST_WIDTH_RANGE_SPACING[1]:
                continue
            if not REST_HEIGHT_RANGE_SPACING[0] <= height_spacing <= REST_HEIGHT_RANGE_SPACING[1]:
                continue
            if area_spacing < REST_AREA_THRESHOLD:
                continue
            if component["width_px"] >= 0.95 * window_width:
                continue
            candidates.append((area_spacing, component))
        if not candidates:
            continue
        area_spacing, component = max(candidates, key=lambda item: item[0])
        bbox = component["bbox"]
        detections.append(
            {
                "role": role,
                "center_x": round((bbox["left"] + bbox["right"]) / 2, 3),
                "duration_beats": 0.5,
                "bbox": bbox,
                "area_staff_squared": round(area_spacing, 6),
                "height_staff_spacing": round(component["height_px"] / spacing, 6),
                "width_staff_spacing": round(component["width_px"] / spacing, 6),
                "evidence": "residual_tall_curved_component",
            }
        )
    return sorted(detections, key=lambda item: float(item["center_x"]))


def _merge_staff_split_components(
    components: Sequence[Mapping[str, Any]], spacing: float
) -> list[dict[str, Any]]:
    """Reconnect a vertical glyph split only by erased staff-line pixels."""
    merged = [deepcopy(dict(component)) for component in components]
    changed = True
    while changed:
        changed = False
        for first_index, first in enumerate(merged):
            first_box = first["bbox"]
            for second_index in range(first_index + 1, len(merged)):
                second = merged[second_index]
                second_box = second["bbox"]
                vertical_gap = max(
                    0,
                    max(first_box["top"], second_box["top"])
                    - min(first_box["bottom"], second_box["bottom"])
                    - 1,
                )
                horizontal_gap = max(
                    0,
                    max(first_box["left"], second_box["left"])
                    - min(first_box["right"], second_box["right"])
                    - 1,
                )
                if vertical_gap > 0.3 * spacing or horizontal_gap > 0.25 * spacing:
                    continue
                left = min(first_box["left"], second_box["left"])
                right = max(first_box["right"], second_box["right"])
                top = min(first_box["top"], second_box["top"])
                bottom = max(first_box["bottom"], second_box["bottom"])
                area = int(first["area_px"]) + int(second["area_px"])
                merged[first_index] = {
                    "area_px": area,
                    "width_px": right - left + 1,
                    "height_px": bottom - top + 1,
                    "bbox": {"left": left, "top": top, "right": right, "bottom": bottom},
                    "fill_ratio": round(area / ((right - left + 1) * (bottom - top + 1)), 6),
                }
                del merged[second_index]
                changed = True
                break
            if changed:
                break
    return merged


def _ink_matrix(image: Image.Image) -> list[list[bool]]:
    gray = image.convert("L")
    pixels = gray.load()
    return [[pixels[x, y] < INK_THRESHOLD for x in range(gray.width)] for y in range(gray.height)]


def _suppress_staff_runs(
    raw_ink: Sequence[Sequence[bool]], staff_lines: Sequence[int], spacing: float
) -> list[list[bool]]:
    ink = [list(row) for row in raw_ink]
    if not ink:
        return ink
    width = len(ink[0])
    minimum_run = max(2, round(STAFF_RUN_MIN_SPACING * spacing))
    half_band = max(1, round(0.2 * spacing))
    for line_y in staff_lines:
        for y in range(max(0, line_y - half_band), min(len(ink), line_y + half_band + 1)):
            x = 0
            while x < width:
                if not raw_ink[y][x]:
                    x += 1
                    continue
                start = x
                while x < width and raw_ink[y][x]:
                    x += 1
                if x - start >= minimum_run:
                    for erase_x in range(start, x):
                        ink[y][erase_x] = False
    return ink


def _components_in_box(
    ink: Sequence[Sequence[bool]], box: tuple[int, int, int, int]
) -> list[dict[str, Any]]:
    x_min, y_min, x_max, y_max = box
    points = {(x, y) for y in range(y_min, y_max + 1) for x in range(x_min, x_max + 1) if ink[y][x]}
    components = []
    while points:
        pending = [points.pop()]
        component_points = []
        while pending:
            x, y = pending.pop()
            component_points.append((x, y))
            for neighbor_x in range(x - 1, x + 2):
                for neighbor_y in range(y - 1, y + 2):
                    neighbor = (neighbor_x, neighbor_y)
                    if neighbor in points:
                        points.remove(neighbor)
                        pending.append(neighbor)
        xs = [point[0] for point in component_points]
        ys = [point[1] for point in component_points]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        components.append(
            {
                "area_px": len(component_points),
                "width_px": right - left + 1,
                "height_px": bottom - top + 1,
                "bbox": {"left": left, "top": top, "right": right, "bottom": bottom},
                "fill_ratio": round(
                    len(component_points) / ((right - left + 1) * (bottom - top + 1)), 6
                ),
            }
        )
    return components


def _clip_box(
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    return (
        max(0, min(width - 1, x_min)),
        max(0, min(width - 1, x_max)),
        max(0, min(height - 1, y_min)),
        max(0, min(height - 1, y_max)),
    )


def build_visual_symbols(
    groups: Sequence[Mapping[str, Any]],
    anchor_features: Sequence[Mapping[str, Any]],
    rest_features: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_order = {int(feature["order"]): feature for feature in anchor_features}
    symbols: list[dict[str, Any]] = []
    for group in groups:
        features = [by_order[int(anchor["order"])] for anchor in group["anchors"]]
        flag_density = max(float(feature["beam_flag"]["density"]) for feature in features)
        dot_area = max(float(feature["dot"]["area_staff_squared"]) for feature in features)
        if flag_density >= FLAG_DENSITY_THRESHOLD:
            duration = 0.5
            evidence = "beam_or_flag"
        elif dot_area >= DOT_AREA_THRESHOLD:
            duration = 1.5
            evidence = "augmentation_dot"
        else:
            duration = 1.0
            evidence = "unflagged_undotted_stem"
        symbols.append(
            {
                "kind": "note",
                "x": float(group["center_x"]),
                "duration_beats": duration,
                "pitches": list(group["pitches"]),
                "group_id": group["group_id"],
                "evidence": evidence,
                "visual_scores": {
                    "beam_flag_density": round(flag_density, 6),
                    "dot_area_staff_squared": round(dot_area, 6),
                },
                "duration_costs": _visual_duration_costs(flag_density, dot_area),
            }
        )
    for index, rest in enumerate(rest_features, start=1):
        symbols.append(
            {
                "kind": "rest",
                "x": float(rest["center_x"]),
                "duration_beats": float(rest["duration_beats"]),
                "rest_id": f"r{index:03d}",
                "evidence": rest["evidence"],
                "duration_costs": {"0.5": 0.0, "1.0": 0.8, "1.5": 1.1},
            }
        )
    return _with_onsets(sorted(symbols, key=lambda item: (float(item["x"]), item["kind"])))


def _visual_duration_costs(flag_density: float, dot_area: float) -> dict[str, float]:
    flag_strength = min(1.0, flag_density / FLAG_DENSITY_THRESHOLD)
    dot_strength = min(1.0, dot_area / DOT_AREA_THRESHOLD)
    plain_strength = max(0.0, 1.0 - max(flag_strength, dot_strength))
    return {
        "0.5": round(1.0 - flag_strength, 6),
        "1.0": round(1.0 - plain_strength, 6),
        "1.5": round(1.0 - dot_strength, 6),
        "2.0": 1.3,
    }


def build_layout_meter_control(
    groups: Sequence[Mapping[str, Any]],
    *,
    expected_beats: float,
    allow_pickup: bool,
) -> list[dict[str, Any]]:
    """Build a no-pixel control using only anchor groups and meter."""
    if not groups:
        return []
    average = expected_beats / len(groups)
    duration = min((0.5, 1.0, 1.5, 2.0), key=lambda value: (abs(value - average), value))
    if not allow_pickup and not math.isclose(duration * len(groups), expected_beats):
        duration = 1.0
    symbols = [
        {
            "kind": "note",
            "x": float(group["center_x"]),
            "duration_beats": duration,
            "pitches": list(group["pitches"]),
            "group_id": group["group_id"],
            "evidence": "layout_meter_equal_duration_control",
        }
        for group in groups
    ]
    return _with_onsets(symbols)


def decode_meter(
    symbols: Sequence[Mapping[str, Any]],
    *,
    expected_beats: float,
    allow_pickup: bool,
) -> tuple[list[dict[str, Any]], str]:
    """Conservatively fit visual symbols to 3/4 on a half-beat grid."""
    copied = [deepcopy(dict(symbol)) for symbol in symbols]
    observed = sum(float(symbol["duration_beats"]) for symbol in copied)
    if math.isclose(observed, expected_beats):
        return _with_onsets(copied), "already_meter_complete"
    if allow_pickup and observed <= expected_beats:
        return _with_onsets(copied), "pickup_preserved"
    target_units = round(expected_beats * 2)
    states: dict[int, tuple[float, tuple[float, ...]]] = {0: (0.0, ())}
    for symbol in copied:
        choices = (0.5, 1.0, 1.5) if symbol["kind"] == "rest" else (0.5, 1.0, 1.5, 2.0)
        visual_duration = float(symbol["duration_beats"])
        costs = {str(key): float(value) for key, value in symbol.get("duration_costs", {}).items()}
        next_states: dict[int, tuple[float, tuple[float, ...]]] = {}
        for total_units, (base_cost, durations) in states.items():
            for duration in choices:
                units = round(duration * 2)
                new_total = total_units + units
                if new_total > target_units:
                    continue
                evidence_cost = costs.get(str(duration), 1.0)
                edit_cost = 0.0 if math.isclose(duration, visual_duration) else METER_EDIT_PENALTY
                candidate = (base_cost + evidence_cost + edit_cost, durations + (duration,))
                previous = next_states.get(new_total)
                if previous is None or candidate < previous:
                    next_states[new_total] = candidate
        states = next_states
    if target_units not in states:
        return _with_onsets(copied), "unresolved_visual_total"
    _, selected = states[target_units]
    changed = False
    for symbol, duration in zip(copied, selected, strict=True):
        changed = changed or not math.isclose(float(symbol["duration_beats"]), duration)
        symbol["duration_beats"] = duration
    return _with_onsets(copied), "meter_repaired" if changed else "meter_confirmed"


def _with_onsets(symbols: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    onset = 0.0
    output = []
    for raw_symbol in symbols:
        symbol = deepcopy(dict(raw_symbol))
        symbol["onset_beats"] = _number(onset)
        duration = float(symbol["duration_beats"])
        symbol["duration_beats"] = _number(duration)
        output.append(symbol)
        onset += duration
    return output


def symbols_to_hypothesis(
    symbols: Sequence[Mapping[str, Any]],
    *,
    identity: Mapping[str, Any],
    decoder_status: str,
) -> dict[str, Any]:
    notes = []
    rests = []
    rhythm_tokens = []
    for symbol in symbols:
        onset = _number(float(symbol["onset_beats"]))
        duration = _number(float(symbol["duration_beats"]))
        if symbol["kind"] == "note":
            pitches = [pitch_to_midi(str(pitch)) for pitch in symbol["pitches"]]
            for pitch in pitches:
                notes.append(
                    {"onset_beats": onset, "duration_beats": duration, "pitch_midi": pitch}
                )
            rhythm_tokens.append(
                {
                    "kind": "note",
                    "onset_beats": onset,
                    "duration_beats": duration,
                    "note_count": len(pitches),
                }
            )
        else:
            rests.append({"onset_beats": onset, "duration_beats": duration})
            rhythm_tokens.append({"kind": "rest", "onset_beats": onset, "duration_beats": duration})
    extent = max(
        (float(token["onset_beats"]) + float(token["duration_beats"]) for token in rhythm_tokens),
        default=0.0,
    )
    return {
        "identity": dict(identity),
        "notes": notes,
        "rests": rests,
        "rhythm_tokens": rhythm_tokens,
        "measure_extent_beats": _number(extent),
        "decoder_status": decoder_status,
    }


def pitch_to_midi(pitch: str) -> int:
    if len(pitch) < 2 or pitch[0].upper() not in "ABCDEFG":
        raise ValueError(f"Unsupported oracle pitch: {pitch!r}")
    letter = pitch[0].upper()
    index = 1
    accidental = 0
    if index < len(pitch) and pitch[index] in "#b":
        accidental = 1 if pitch[index] == "#" else -1
        index += 1
    octave = int(pitch[index:])
    semitone = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[letter]
    return 12 * (octave + 1) + semitone + accidental


def evaluate_hypothesis(hypothesis: Mapping[str, Any], truth: Mapping[str, Any]) -> dict[str, Any]:
    predicted_tokens = list(hypothesis["rhythm_tokens"])
    truth_tokens = _truth_rhythm_tokens(truth)
    predicted_note_tokens = [token for token in predicted_tokens if token["kind"] == "note"]
    truth_note_tokens = [token for token in truth_tokens if token["kind"] == "note"]
    duration_correct = sum(
        _same_number(predicted["duration_beats"], expected["duration_beats"])
        for predicted, expected in zip(predicted_note_tokens, truth_note_tokens, strict=False)
    )
    predicted_note_group_count = len(predicted_note_tokens)
    truth_note_group_count = len(truth_note_tokens)
    duration_total = max(predicted_note_group_count, truth_note_group_count)
    predicted_rests = {
        (_fraction_key(item["onset_beats"]), _fraction_key(item["duration_beats"]))
        for item in hypothesis["rests"]
    }
    truth_rests = {
        (_fraction_key(item["onset_beats"]), _fraction_key(item["duration_beats"]))
        for item in truth["rests"]
    }
    rest_tp = len(predicted_rests & truth_rests)
    rest_fp = len(predicted_rests - truth_rests)
    rest_fn = len(truth_rests - predicted_rests)
    rest_precision = _ratio(rest_tp, rest_tp + rest_fp, empty=1.0 if not truth_rests else 0.0)
    rest_recall = _ratio(rest_tp, rest_tp + rest_fn, empty=1.0 if not truth_rests else 0.0)
    rest_f1 = (
        0.0
        if rest_precision + rest_recall == 0
        else 2 * rest_precision * rest_recall / (rest_precision + rest_recall)
    )
    exact = _normalized_tokens(predicted_tokens) == _normalized_tokens(truth_tokens)
    return {
        "duration_correct": duration_correct,
        "duration_total": duration_total,
        "duration_accuracy": _ratio(duration_correct, duration_total),
        "predicted_note_group_count": predicted_note_group_count,
        "truth_note_group_count": truth_note_group_count,
        "has_note_group_overproduction": predicted_note_group_count > truth_note_group_count,
        "rest_tp": rest_tp,
        "rest_fp": rest_fp,
        "rest_fn": rest_fn,
        "rest_precision": round(rest_precision, 6),
        "rest_recall": round(rest_recall, 6),
        "rest_f1": round(rest_f1, 6),
        "exact_ordered_rhythm": exact,
        "predicted_tokens": predicted_tokens,
        "truth_tokens": truth_tokens,
    }


def _truth_rhythm_tokens(truth: Mapping[str, Any]) -> list[dict[str, Any]]:
    grouped_notes: dict[tuple[str, str], int] = {}
    for note in truth["notes"]:
        key = (_fraction_key(note["onset_beats"]), _fraction_key(note["duration_beats"]))
        grouped_notes[key] = grouped_notes.get(key, 0) + 1
    tokens = [
        {
            "kind": "note",
            "onset_beats": _number(float(Fraction(onset))),
            "duration_beats": _number(float(Fraction(duration))),
            "note_count": count,
        }
        for (onset, duration), count in grouped_notes.items()
    ]
    tokens.extend(
        {
            "kind": "rest",
            "onset_beats": _number(float(rest["onset_beats"])),
            "duration_beats": _number(float(rest["duration_beats"])),
        }
        for rest in truth["rests"]
    )
    return sorted(tokens, key=lambda item: (float(item["onset_beats"]), item["kind"]))


def _normalized_tokens(tokens: Sequence[Mapping[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (
            token["kind"],
            _fraction_key(token["onset_beats"]),
            _fraction_key(token["duration_beats"]),
            int(token.get("note_count", 0)),
        )
        for token in tokens
    ]


def aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    duration_correct = sum(int(row["duration_correct"]) for row in rows)
    duration_total = sum(int(row["duration_total"]) for row in rows)
    rest_tp = sum(int(row["rest_tp"]) for row in rows)
    rest_fp = sum(int(row["rest_fp"]) for row in rows)
    rest_fn = sum(int(row["rest_fn"]) for row in rows)
    rest_precision = _ratio(rest_tp, rest_tp + rest_fp, empty=1.0 if rest_fn == 0 else 0.0)
    rest_recall = _ratio(rest_tp, rest_tp + rest_fn, empty=1.0 if rest_fp == 0 else 0.0)
    rest_f1 = (
        0.0
        if rest_precision + rest_recall == 0
        else 2 * rest_precision * rest_recall / (rest_precision + rest_recall)
    )
    predicted_note_group_count = sum(int(row["predicted_note_group_count"]) for row in rows)
    truth_note_group_count = sum(int(row["truth_note_group_count"]) for row in rows)
    overproduced_measure_count = sum(bool(row["has_note_group_overproduction"]) for row in rows)
    return {
        "measure_count": len(rows),
        "duration_correct": duration_correct,
        "duration_total": duration_total,
        "duration_accuracy": round(_ratio(duration_correct, duration_total), 6),
        "predicted_note_group_count": predicted_note_group_count,
        "truth_note_group_count": truth_note_group_count,
        "overproduced_measure_count": overproduced_measure_count,
        "has_note_group_overproduction": overproduced_measure_count > 0,
        "rest_tp": rest_tp,
        "rest_fp": rest_fp,
        "rest_fn": rest_fn,
        "rest_precision": round(rest_precision, 6),
        "rest_recall": round(rest_recall, 6),
        "rest_f1": round(rest_f1, 6),
        "exact_measure_count": sum(bool(row["exact_ordered_rhythm"]) for row in rows),
    }


def _passes_joint_metric_gate(metrics: Mapping[str, Any]) -> bool:
    return (
        float(metrics["duration_accuracy"]) >= 0.85
        and float(metrics["rest_f1"]) >= 0.8
        and int(metrics["overproduced_measure_count"]) == 0
    )


def _write_prediction_arms(
    prepared: Sequence[PreparedMeasure], output_dir: Path
) -> dict[str, Path]:
    paths = {}
    for arm_name in prepared[0].arms:
        path = output_dir / f"predictions_{arm_name}.jsonl"
        _write_jsonl(path, [item.arms[arm_name] for item in prepared])
        paths[arm_name] = path
    return paths


def _write_overlay(item: PreparedMeasure, path: Path) -> None:
    overlay = item.image.copy()
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    feature_by_order = {int(feature["order"]): feature for feature in item.anchor_features}
    # Hypothesis tokens intentionally omit internal group ids; recover durations in x order.
    note_tokens = [
        token for token in item.arms["visual_only"]["rhythm_tokens"] if token["kind"] == "note"
    ]
    for group, token in zip(item.groups, note_tokens, strict=True):
        duration = float(token["duration_beats"])
        for anchor in group["anchors"]:
            order = int(anchor["order"])
            feature = feature_by_order[order]
            x = float(anchor["center"]["x"])
            y = float(anchor["center"]["y"])
            color = (
                (220, 45, 35)
                if duration == 0.5
                else (190, 80, 0) if duration == 1.5 else (20, 115, 210)
            )
            radius = max(4, round(0.3 * item.staff_spacing))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
            stem = feature["stem"]
            draw.line(
                (stem["x"], stem["near_y"], stem["x"], stem["tip_y"]), fill=(20, 160, 70), width=2
            )
            label = f"{order}:{anchor['pitch']} {_duration_text(duration)}"
            draw.text((x + radius + 2, max(1, y - radius - 2)), label, fill=color, font=font)
            dot_component = feature["dot"]["component"]
            if dot_component is not None and feature["dot"]["present"]:
                bbox = dot_component["bbox"]
                draw.rectangle(
                    (bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]),
                    outline=(235, 145, 0),
                    width=2,
                )
    for rest in item.rest_features:
        bbox = rest["bbox"]
        draw.rectangle(
            (bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]),
            outline=(145, 35, 180),
            width=2,
        )
        draw.text(
            (bbox["left"], max(1, bbox["top"] - 11)),
            f"R {_duration_text(float(rest['duration_beats']))}",
            fill=(145, 35, 180),
            font=font,
        )
    draw.rectangle((0, 0, min(330, overlay.width - 1), 14), fill="white")
    draw.text((3, 2), "oracle anchors; visual durations/rests (no GT)", fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(path)


def _write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Anchored Rhythm Parser Spike",
        "",
        f"Decision: **{str(report['decision']['status']).upper()}**",
        "",
        "This is an explicit HITL upper-bound arm: reviewed centers and pitches are "
        "oracle anchors, "
        "not automatic notehead recognition.",
        "",
        "All visual features and predictions were written before benchmark truth was loaded. "
        "Inference predicts glyph durations; evaluation uses canonical sounding-event durations. "
        "The experiment did not read source MusicXML directly.",
        "",
        "## Aggregate",
        "",
        "| Arm | Duration | Rest P/R/F1 | Exact measures |",
        "| --- | ---: | ---: | ---: |",
    ]
    for arm_name in ("layout_meter_control", "visual_only", "meter_decoded"):
        metrics = report["aggregate"][arm_name]
        lines.append(
            f"| `{arm_name}` | {metrics['duration_correct']}/{metrics['duration_total']} "
            f"({metrics['duration_accuracy']:.3f}) | {metrics['rest_precision']:.3f}/"
            f"{metrics['rest_recall']:.3f}/{metrics['rest_f1']:.3f} | "
            f"{metrics['exact_measure_count']}/{metrics['measure_count']} |"
        )
    lines.extend(
        [
            "",
            "## Measures",
            "",
            "| Measure | Control | Visual | Meter decoded | Rest detections |",
            "| ---: | :---: | :---: | :---: | ---: |",
        ]
    )
    for row in report["per_measure"]:
        evaluations = row["evaluation"]
        control_exact = evaluations["layout_meter_control"]["exact_ordered_rhythm"]
        lines.append(
            f"| {row['measure']} | {_yes(control_exact)} "
            f"| {_yes(evaluations['visual_only']['exact_ordered_rhythm'])} "
            f"| {_yes(evaluations['meter_decoded']['exact_ordered_rhythm'])} "
            f"| {len(row['residual_rest_features'])} |"
        )
    gate = report["decision"]
    lines.extend(
        [
            "",
            "## Predeclared Material Gate",
            "",
            f"- Rule: `{gate['material_gate']}`",
            f"- Exact measures: `{gate['exact_measure_gate']['observed']}/4`",
            f"- Duration accuracy: `{gate['joint_metric_gate']['duration_accuracy_observed']:.3f}`",
            f"- Rest F1: `{gate['joint_metric_gate']['rest_f1_observed']:.3f}`",
            (
                "- Note groups (predicted/truth): "
                f"`{gate['joint_metric_gate']['predicted_note_group_count']}/"
                f"{gate['joint_metric_gate']['truth_note_group_count']}`"
            ),
            (
                "- Measures with note-group overproduction: "
                f"`{gate['joint_metric_gate']['overproduced_measure_count']}`"
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _validate_review_identity(review: Mapping[str, Any], identity: Mapping[str, Any]) -> None:
    review_identity = review["identity"]
    keys = ("slug", "system_index", "system_measure_index", "global_measure_index")
    for key in keys:
        if review_identity[key] != identity[key]:
            raise ValueError(f"Review/request identity mismatch for {key}")
    if review.get("provenance", {}).get("review_type") != "human":
        raise ValueError("Anchored upper-bound arm requires a human review fixture")


def _resolve_input_path(raw_path: str, *, out_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidates = (REPO_ROOT / path, out_dir / path, out_dir / path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(raw_path)


def _validate_sha256(path: Path, expected: str) -> None:
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise ValueError(f"Image hash drift for {path}: {observed} != {expected}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _fraction_key(value: Any) -> str:
    return str(Fraction(str(value)).limit_denominator(16))


def _number(value: float) -> int | float:
    rounded = round(value, 6)
    return int(rounded) if rounded.is_integer() else rounded


def _same_number(left: Any, right: Any) -> bool:
    return Fraction(str(left)).limit_denominator(16) == Fraction(str(right)).limit_denominator(16)


def _ratio(numerator: int, denominator: int, *, empty: float = 0.0) -> float:
    return empty if denominator == 0 else numerator / denominator


def _duration_text(duration: float) -> str:
    return {0.5: "1/2", 1.0: "1", 1.5: "3/2", 2.0: "2"}.get(duration, str(duration))


def _yes(value: bool) -> str:
    return "yes" if value else "no"


if __name__ == "__main__":
    raise SystemExit(main())
