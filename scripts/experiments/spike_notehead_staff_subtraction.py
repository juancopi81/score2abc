"""Bounded deterministic notehead-localization spike for Aviador S1 M1-M4.

This experiment challenges the existing staff-grid-density proposal ranking with
staff-line subtraction, vertical-stroke subtraction, stem-end evidence, and
half-space pitch-row quantization. Detection is blind: every candidate list is
generated before any human coordinate fixture is read. A four-fold
leave-one-measure-out evaluation then tunes a small declared parameter grid on
three measures and evaluates the held-out fourth measure.

The human blue-circle annotations are regions rather than exact point centers,
so the primary localization metric uses the authoritative annotation ellipses
expanded by 1.15x. Natural-pitch correctness is reported separately.

Example:
    ./.venv/bin/python scripts/experiments/spike_notehead_staff_subtraction.py
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from score2abc.utils.imaging import estimate_ink_threshold  # noqa: E402
from scripts import build_vlm_notehead_candidates as baseline_detector  # noqa: E402

DEFAULT_OUT_DIR = REPO_ROOT / "out"
DEFAULT_SLUG = "jaime-llanos_12_aviador_pasillo_fulgencio-garcia"
SYSTEM_INDEX = 1
MEASURES = (1, 2, 3, 4)
TOP_K_VALUES = (4, 8, 12, 24)
BASELINE_CAPS = (4, 5, 6, 8, 12, 16, 24)
MAX_CANDIDATES = max(TOP_K_VALUES)
ANNOTATION_REGION_MARGIN = 1.15
GT_DIR = REPO_ROOT / "tests/fixtures/vlm_melody/notehead_ground_truth"
OUTPUT_DIR = DEFAULT_OUT_DIR / "experiments/notehead_staff_subtraction"


@dataclass(frozen=True, order=True)
class DetectorConfig:
    """A deliberately small morphology/ranking parameterization."""

    line_support: float
    vertical_preserve: float
    profile: str
    nms_x: float
    cap: int

    @property
    def key(self) -> str:
        return (
            f"line{self.line_support:.2f}-preserve{self.vertical_preserve:.2f}-"
            f"{self.profile}-nms{self.nms_x:.2f}-cap{self.cap}"
        )


@dataclass(frozen=True)
class Candidate:
    x: float
    y: float
    score: float
    features: dict[str, float]

    def to_point(self, index: int) -> dict[str, Any]:
        return {
            "id": f"c{index:03d}",
            "center": {"x": round(self.x, 3), "y": round(self.y, 3)},
            "score": round(self.score, 6),
            "features": {key: round(value, 6) for key, value in self.features.items()},
        }


@dataclass
class PreparedMeasure:
    measure: int
    source_path: Path
    context_path: Path
    image: Image.Image
    staff_lines: list[int]
    spacing: float
    threshold: int
    ink: list[list[bool]]
    morphology: dict[tuple[float, float], dict[str, list[list[bool]]]]
    proposed_by_config: dict[DetectorConfig, list[Candidate]]
    baseline_ranked: list[Candidate]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Detection for every parameterization is complete before GT is loaded.
        prepared = {
            measure: _prepare_measure(args.out_dir, args.slug, measure) for measure in MEASURES
        }
        ground_truth = {measure: _load_ground_truth(args.slug, measure) for measure in MEASURES}
        report = _evaluate_leave_one_measure_out(
            prepared,
            ground_truth,
            slug=args.slug,
        )
        _write_diagnostics(prepared, ground_truth, report, output_dir)
        _write_report(report, output_dir)
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(output_dir / "report.json")
    print(output_dir / "report.md")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Pipeline output root containing vlm_melody_inputs_manifest.jsonl.",
    )
    parser.add_argument("--slug", default=DEFAULT_SLUG, help="Work slug; defaults to Aviador.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Ignored experiment-artifact directory.",
    )
    return parser


def _parameter_grid() -> list[DetectorConfig]:
    return [
        DetectorConfig(line_support, vertical_preserve, profile, nms_x, cap)
        for line_support in (0.48, 0.60)
        for vertical_preserve in (0.45, 0.60)
        for profile in ("residual", "stem-balanced", "shape")
        for nms_x in (0.62, 0.82)
        for cap in (4, 5, 6, 8, 12, 24)
    ]


def _prepare_measure(out_dir: Path, slug: str, measure: int) -> PreparedMeasure:
    record = _measure_record(out_dir, slug, measure)
    context_path = _resolve_path(out_dir, record["paths"]["context"])
    source_path = _resolve_path(out_dir, record["paths"]["measure_raw"])
    context = json.loads(context_path.read_text(encoding="utf-8"))
    staff_lines = [int(value) for value in context["staff_lines_y_px_in_system"]]
    if len(staff_lines) != 5:
        raise ValueError(f"Expected five staff lines for measure {measure}: {staff_lines!r}")
    spacing = statistics.mean(b - a for a, b in zip(staff_lines, staff_lines[1:], strict=False))
    if spacing <= 0:
        raise ValueError(f"Invalid staff spacing for measure {measure}: {spacing}")

    image = Image.open(source_path).convert("RGB")
    gray = ImageOps.grayscale(image)
    threshold = estimate_ink_threshold(gray)
    width, height = gray.size
    pixels = gray.load()
    ink = [[pixels[x, y] < threshold for x in range(width)] for y in range(height)]

    configs = _parameter_grid()
    morphology: dict[tuple[float, float], dict[str, list[list[bool]]]] = {}
    feature_rows_cache: dict[
        tuple[float, float], dict[float, list[tuple[float, dict[str, float]]]]
    ] = {}
    ranked_cache: dict[tuple[float, float, str, float], list[Candidate]] = {}
    proposed_by_config: dict[DetectorConfig, list[Candidate]] = {}
    for config in configs:
        morphology_key = (config.line_support, config.vertical_preserve)
        if morphology_key not in morphology:
            morphology[morphology_key] = _build_morphology(
                ink,
                staff_lines=staff_lines,
                spacing=spacing,
                line_support=config.line_support,
                vertical_preserve=config.vertical_preserve,
            )
            feature_rows_cache[morphology_key] = _scan_feature_rows(
                ink,
                morphology[morphology_key],
                staff_lines=staff_lines,
                spacing=spacing,
            )
        ranked_key = (*morphology_key, config.profile, config.nms_x)
        if ranked_key not in ranked_cache:
            ranked_cache[ranked_key] = _rank_candidates(
                feature_rows_cache[morphology_key],
                spacing=spacing,
                profile=config.profile,
                nms_x_factor=config.nms_x,
                max_candidates=MAX_CANDIDATES,
            )
        proposed_by_config[config] = ranked_cache[ranked_key][: config.cap]

    baseline = baseline_detector.detect_staff_grid_density_candidates(
        image,
        staff_lines=staff_lines,
        max_candidates=MAX_CANDIDATES,
    )
    baseline_ranked = [
        Candidate(
            x=float(item.center[0]),
            y=float(item.center[1]),
            score=float(item.score),
            features={key: float(value) for key, value in item.features.items()},
        )
        for item in baseline
    ]
    return PreparedMeasure(
        measure=measure,
        source_path=source_path,
        context_path=context_path,
        image=image,
        staff_lines=staff_lines,
        spacing=spacing,
        threshold=threshold,
        ink=ink,
        morphology=morphology,
        proposed_by_config=proposed_by_config,
        baseline_ranked=baseline_ranked,
    )


def _build_morphology(
    ink: list[list[bool]],
    *,
    staff_lines: Sequence[int],
    spacing: float,
    line_support: float,
    vertical_preserve: float,
) -> dict[str, list[list[bool]]]:
    height = len(ink)
    width = len(ink[0]) if ink else 0
    line_window = max(17, round(spacing * 2.1))
    line_band = max(2, round(spacing * 0.12))
    vertical_window = max(11, round(spacing * 0.9))
    row_prefix = [_prefix_sum(row) for row in ink]
    column_prefix = [_prefix_sum([ink[y][x] for y in range(height)]) for x in range(width)]

    staff_mask = _empty_mask(width, height)
    half_line = line_window // 2
    half_vertical = vertical_window // 2
    for staff_y in staff_lines:
        for y in range(max(0, staff_y - line_band), min(height, staff_y + line_band + 1)):
            for x in range(width):
                if not ink[y][x]:
                    continue
                horizontal = _range_sum(row_prefix[y], x - half_line, x + half_line + 1)
                horizontal_fraction = horizontal / min(line_window, width)
                if horizontal_fraction < line_support:
                    continue
                vertical_rows = 0
                for sample_x in range(max(0, x - 1), min(width, x + 2)):
                    vertical_rows = max(
                        vertical_rows,
                        _range_sum(
                            column_prefix[sample_x],
                            y - half_vertical,
                            y + half_vertical + 1,
                        ),
                    )
                if vertical_rows / min(vertical_window, height) < vertical_preserve:
                    staff_mask[y][x] = True

    staff_suppressed = [
        [ink[y][x] and not staff_mask[y][x] for x in range(width)] for y in range(height)
    ]
    vertical_mask = _vertical_stroke_mask(staff_suppressed, spacing=spacing)
    structure = [
        [staff_suppressed[y][x] and not vertical_mask[y][x] for x in range(width)]
        for y in range(height)
    ]
    structure = _binary_close(structure)
    return {
        "staff_mask": staff_mask,
        "staff_suppressed": staff_suppressed,
        "vertical_mask": vertical_mask,
        "structure": structure,
    }


def _vertical_stroke_mask(mask: list[list[bool]], *, spacing: float) -> list[list[bool]]:
    height = len(mask)
    width = len(mask[0]) if mask else 0
    result = _empty_mask(width, height)
    window = max(13, round(spacing * 1.05))
    half = window // 2
    columns = [[mask[y][x] for y in range(height)] for x in range(width)]
    prefixes = [_prefix_sum(column) for column in columns]
    for y in range(height):
        for x in range(width):
            if not mask[y][x]:
                continue
            support = max(
                _range_sum(prefixes[sample_x], y - half, y + half + 1)
                for sample_x in range(max(0, x - 1), min(width, x + 2))
            )
            if support / min(window, height) >= 0.58:
                result[y][x] = True
    return result


def _binary_close(mask: list[list[bool]]) -> list[list[bool]]:
    if not mask:
        return []
    height = len(mask)
    width = len(mask[0])
    dilated = _empty_mask(width, height)
    for y in range(height):
        for x in range(width):
            dilated[y][x] = any(
                mask[ny][nx]
                for ny in range(max(0, y - 1), min(height, y + 2))
                for nx in range(max(0, x - 1), min(width, x + 2))
            )
    eroded = _empty_mask(width, height)
    for y in range(height):
        for x in range(width):
            eroded[y][x] = all(
                dilated[ny][nx]
                for ny in range(max(0, y - 1), min(height, y + 2))
                for nx in range(max(0, x - 1), min(width, x + 2))
            )
    return eroded


def _rank_candidates(
    feature_rows: dict[float, list[tuple[float, dict[str, float]]]],
    *,
    spacing: float,
    profile: str,
    nms_x_factor: float,
    max_candidates: int,
) -> list[Candidate]:
    raw_candidates: list[Candidate] = []
    scores_by_row = {
        pitch_y: [
            Candidate(
                x=x,
                y=float(pitch_y),
                score=_feature_score(features, profile=profile),
                features=features,
            )
            for x, features in row_features
        ]
        for pitch_y, row_features in feature_rows.items()
    }

    local_radius = max(3, round(spacing * 0.33))
    for row_candidates in scores_by_row.values():
        for index, candidate in enumerate(row_candidates):
            neighborhood = row_candidates[
                max(0, index - local_radius) : min(len(row_candidates), index + local_radius + 1)
            ]
            if candidate.score < 0.08:
                continue
            if candidate.score != max(item.score for item in neighborhood):
                continue
            if any(item.score == candidate.score and item.x < candidate.x for item in neighborhood):
                continue
            raw_candidates.append(candidate)

    ordered = sorted(raw_candidates, key=lambda item: (-item.score, item.x, item.y))
    selected: list[Candidate] = []
    x_distance = spacing * nms_x_factor
    y_distance = spacing * 0.72
    for candidate in ordered:
        if any(
            abs(candidate.x - current.x) < x_distance and abs(candidate.y - current.y) < y_distance
            for current in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= max_candidates:
            break
    return selected


def _scan_feature_rows(
    ink: list[list[bool]],
    morphology: dict[str, list[list[bool]]],
    *,
    staff_lines: Sequence[int],
    spacing: float,
) -> dict[float, list[tuple[float, dict[str, float]]]]:
    height = len(ink)
    width = len(ink[0]) if ink else 0
    residual = morphology["staff_suppressed"]
    structure = morphology["structure"]
    pitch_rows = _pitch_rows(staff_lines, spacing=spacing, height=height)
    x_margin = max(2, round(spacing * 0.25))
    feature_rows: dict[float, list[tuple[float, dict[str, float]]]] = {}
    for pitch_y in pitch_rows:
        row_features = []
        for x in range(x_margin, max(x_margin, width - x_margin)):
            features = _window_features(
                ink,
                residual,
                structure,
                center_x=x,
                center_y=pitch_y,
                spacing=spacing,
            )
            row_features.append((float(x), features))
        feature_rows[pitch_y] = row_features
    return feature_rows


def _window_features(
    ink: list[list[bool]],
    residual: list[list[bool]],
    structure: list[list[bool]],
    *,
    center_x: int,
    center_y: float,
    spacing: float,
) -> dict[str, float]:
    height = len(ink)
    width = len(ink[0]) if ink else 0
    radius_x = max(4.0, spacing * 0.39)
    radius_y = max(3.0, spacing * 0.32)
    left = max(0, math.floor(center_x - radius_x))
    right = min(width, math.ceil(center_x + radius_x + 1))
    top = max(0, math.floor(center_y - radius_y))
    bottom = min(height, math.ceil(center_y + radius_y + 1))
    ellipse_pixels: list[tuple[int, int]] = []
    for y in range(top, bottom):
        for x in range(left, right):
            if ((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2 <= 1.0:
                ellipse_pixels.append((x, y))
    ellipse_area = max(1, len(ellipse_pixels))
    original_density = sum(ink[y][x] for x, y in ellipse_pixels) / ellipse_area
    residual_density = sum(residual[y][x] for x, y in ellipse_pixels) / ellipse_area
    structure_density = sum(structure[y][x] for x, y in ellipse_pixels) / ellipse_area

    row_counts = [sum(residual[y][x] for x in range(left, right)) for y in range(top, bottom)]
    column_counts = [sum(residual[y][x] for y in range(top, bottom)) for x in range(left, right)]
    window_width = max(1, right - left)
    window_height = max(1, bottom - top)
    multirow = sum(count >= max(2, round(window_width * 0.18)) for count in row_counts)
    multicolumn = sum(count >= max(2, round(window_height * 0.20)) for count in column_counts)
    row_spread = multirow / window_height
    column_spread = multicolumn / window_width

    up = _stem_side_evidence(
        residual,
        center_x=center_x,
        center_y=center_y,
        spacing=spacing,
        direction=-1,
    )
    down = _stem_side_evidence(
        residual,
        center_x=center_x,
        center_y=center_y,
        spacing=spacing,
        direction=1,
    )
    endpoint = max(up, down) * (1.0 - min(up, down))
    through_stroke = min(up, down)
    removed_fraction = max(0.0, original_density - residual_density)
    return {
        "original_density": original_density,
        "residual_density": residual_density,
        "structure_density": structure_density,
        "row_spread": row_spread,
        "column_spread": column_spread,
        "stem_up": up,
        "stem_down": down,
        "stem_endpoint": endpoint,
        "through_stroke": through_stroke,
        "removed_fraction": removed_fraction,
    }


def _stem_side_evidence(
    mask: list[list[bool]],
    *,
    center_x: int,
    center_y: float,
    spacing: float,
    direction: int,
) -> float:
    height = len(mask)
    width = len(mask[0]) if mask else 0
    near = max(2, round(spacing * 0.12))
    far = max(near + 1, round(spacing * 1.05))
    if direction < 0:
        y0 = max(0, round(center_y) - far)
        y1 = max(0, round(center_y) - near)
    else:
        y0 = min(height, round(center_y) + near)
        y1 = min(height, round(center_y) + far)
    if y1 <= y0:
        return 0.0
    best = 0.0
    search_radius = max(3, round(spacing * 0.55))
    for x in range(max(0, center_x - search_radius), min(width, center_x + search_radius + 1)):
        rows_with_ink = sum(
            any(mask[y][sample_x] for sample_x in range(max(0, x - 1), min(width, x + 2)))
            for y in range(y0, y1)
        )
        best = max(best, rows_with_ink / (y1 - y0))
    return min(1.0, best)


def _feature_score(features: dict[str, float], *, profile: str) -> float:
    weights = {
        "residual": {
            "original_density": 0.10,
            "residual_density": 0.38,
            "structure_density": 0.18,
            "row_spread": 0.12,
            "column_spread": 0.08,
            "stem_endpoint": 0.18,
            "through_stroke": -0.30,
            "removed_fraction": -0.10,
        },
        "stem-balanced": {
            "original_density": 0.08,
            "residual_density": 0.30,
            "structure_density": 0.15,
            "row_spread": 0.10,
            "column_spread": 0.07,
            "stem_endpoint": 0.32,
            "through_stroke": -0.38,
            "removed_fraction": -0.08,
        },
        "shape": {
            "original_density": 0.08,
            "residual_density": 0.25,
            "structure_density": 0.32,
            "row_spread": 0.16,
            "column_spread": 0.12,
            "stem_endpoint": 0.13,
            "through_stroke": -0.26,
            "removed_fraction": -0.08,
        },
    }
    if profile not in weights:
        raise ValueError(f"Unknown score profile: {profile}")
    return sum(features[name] * weight for name, weight in weights[profile].items())


def _pitch_rows(staff_lines: Sequence[int], *, spacing: float, height: int) -> list[float]:
    half_space = spacing / 2.0
    top = float(staff_lines[0])
    minimum = max(0.0, top - spacing * 1.5)
    maximum = min(height - 1.0, float(staff_lines[-1]) + spacing * 2.5)
    first_step = math.ceil((minimum - top) / half_space)
    last_step = math.floor((maximum - top) / half_space)
    return [round(top + step * half_space, 3) for step in range(first_step, last_step + 1)]


def _evaluate_leave_one_measure_out(
    prepared: dict[int, PreparedMeasure],
    ground_truth: dict[int, list[dict[str, Any]]],
    *,
    slug: str,
) -> dict[str, Any]:
    proposed_configs = _parameter_grid()
    folds = []
    for held_out in MEASURES:
        training = [measure for measure in MEASURES if measure != held_out]
        selected_config = max(
            proposed_configs,
            key=lambda config: _selection_key(
                _aggregate_metrics(
                    [
                        _match_region(
                            _candidate_points(prepared[measure].proposed_by_config[config]),
                            ground_truth[measure],
                            prepared[measure].staff_lines,
                        )
                        for measure in training
                    ]
                ),
                cap=config.cap,
                stable_key=config.key,
            ),
        )
        selected_baseline_cap = max(
            BASELINE_CAPS,
            key=lambda cap: _selection_key(
                _aggregate_metrics(
                    [
                        _match_region(
                            _candidate_points(prepared[measure].baseline_ranked[:cap]),
                            ground_truth[measure],
                            prepared[measure].staff_lines,
                        )
                        for measure in training
                    ]
                ),
                cap=cap,
                stable_key=f"cap{cap}",
            ),
        )
        proposed_points = _candidate_points(prepared[held_out].proposed_by_config[selected_config])
        baseline_points = _candidate_points(
            prepared[held_out].baseline_ranked[:selected_baseline_cap]
        )
        folds.append(
            {
                "held_out_measure": held_out,
                "training_measures": training,
                "proposed": {
                    "selected_config": asdict(selected_config),
                    "selected_config_key": selected_config.key,
                    "metrics": _match_region(
                        proposed_points,
                        ground_truth[held_out],
                        prepared[held_out].staff_lines,
                    ),
                    "top_k_recall": _top_k_recall(
                        prepared[held_out].proposed_by_config[
                            DetectorConfig(
                                selected_config.line_support,
                                selected_config.vertical_preserve,
                                selected_config.profile,
                                selected_config.nms_x,
                                24,
                            )
                        ],
                        ground_truth[held_out],
                        prepared[held_out].staff_lines,
                    ),
                    "candidates": proposed_points,
                },
                "baseline": {
                    "selected_cap": selected_baseline_cap,
                    "metrics": _match_region(
                        baseline_points,
                        ground_truth[held_out],
                        prepared[held_out].staff_lines,
                    ),
                    "top_k_recall": _top_k_recall(
                        prepared[held_out].baseline_ranked,
                        ground_truth[held_out],
                        prepared[held_out].staff_lines,
                    ),
                    "candidates": baseline_points,
                },
            }
        )

    proposed_aggregate = _aggregate_metrics([fold["proposed"]["metrics"] for fold in folds])
    baseline_aggregate = _aggregate_metrics([fold["baseline"]["metrics"] for fold in folds])
    proposed_top_k = _aggregate_top_k(folds, "proposed")
    baseline_top_k = _aggregate_top_k(folds, "baseline")
    return {
        "schema_version": 1,
        "kind": "notehead_staff_subtraction_leave_one_measure_out",
        "slug": slug,
        "system_index": SYSTEM_INDEX,
        "measures": list(MEASURES),
        "detection_blindness": (
            "all morphology and ranked candidates were generated before coordinate GT reads"
        ),
        "primary_metric": {
            "name": "annotation-region",
            "ellipse_margin": ANNOTATION_REGION_MARGIN,
            "matching": "global one-to-one minimum normalized ellipse distance",
        },
        "pitch_metric": {
            "name": "pitch-safe accuracy",
            "definition": "correct natural staff pitch / annotation-region true positives",
            "pitch_safe_recall_definition": "correct natural staff pitch / GT noteheads",
        },
        "parameter_grid": {
            "proposed_config_count": len(proposed_configs),
            "line_support": [0.48, 0.60],
            "vertical_preserve": [0.45, 0.60],
            "profiles": ["residual", "stem-balanced", "shape"],
            "nms_x": [0.62, 0.82],
            "caps": [4, 5, 6, 8, 12, 24],
            "selection": (
                "training micro-F1, then pitch-safe recall, recall, precision, smaller cap, "
                "stable config key"
            ),
        },
        "folds": folds,
        "aggregate": {
            "proposed": {**proposed_aggregate, "top_k_recall": proposed_top_k},
            "baseline_staff_grid_density_v2": {
                **baseline_aggregate,
                "top_k_recall": baseline_top_k,
            },
            "delta": {
                "precision": round(
                    proposed_aggregate["precision"] - baseline_aggregate["precision"], 6
                ),
                "recall": round(proposed_aggregate["recall"] - baseline_aggregate["recall"], 6),
                "f1": round(proposed_aggregate["f1"] - baseline_aggregate["f1"], 6),
                "pitch_safe_accuracy": round(
                    proposed_aggregate["pitch_safe_accuracy"]
                    - baseline_aggregate["pitch_safe_accuracy"],
                    6,
                ),
                "pitch_safe_recall": round(
                    proposed_aggregate["pitch_safe_recall"]
                    - baseline_aggregate["pitch_safe_recall"],
                    6,
                ),
            },
        },
        "scope_note": (
            "Four annotated development measures are enough for a bounded structural spike, "
            "not for a generalization claim. Each reported fold is held out from threshold "
            "selection."
        ),
    }


def _selection_key(metrics: dict[str, Any], *, cap: int, stable_key: str) -> tuple[Any, ...]:
    # max() is used; invert cap and stable key only where practical.
    return (
        metrics["f1"],
        metrics["pitch_safe_recall"],
        metrics["recall"],
        metrics["precision"],
        -cap,
        tuple(-ord(character) for character in stable_key),
    )


def _top_k_recall(
    candidates: Sequence[Candidate],
    ground_truth: Sequence[dict[str, Any]],
    staff_lines: Sequence[int],
) -> dict[str, Any]:
    return {
        str(k): _match_region(_candidate_points(candidates[:k]), ground_truth, staff_lines)[
            "recall"
        ]
        for k in TOP_K_VALUES
    }


def _aggregate_top_k(folds: Sequence[dict[str, Any]], detector_key: str) -> dict[str, float]:
    result = {}
    total_gt = sum(fold[detector_key]["metrics"]["gt_count"] for fold in folds)
    for k in TOP_K_VALUES:
        tp = 0
        for fold in folds:
            recall = float(fold[detector_key]["top_k_recall"][str(k)])
            tp += round(recall * fold[detector_key]["metrics"]["gt_count"])
        result[str(k)] = _ratio(tp, total_gt)
    return result


def _match_region(
    candidates: Sequence[dict[str, Any]],
    ground_truth: Sequence[dict[str, Any]],
    staff_lines: Sequence[int],
) -> dict[str, Any]:
    possible = []
    for candidate_index, candidate in enumerate(candidates):
        cx = float(candidate["center"]["x"])
        cy = float(candidate["center"]["y"])
        for gt_index, item in enumerate(ground_truth):
            center_x, center_y, radius_x, radius_y = _annotation_ellipse(item)
            radius_x *= ANNOTATION_REGION_MARGIN
            radius_y *= ANNOTATION_REGION_MARGIN
            dx = cx - center_x
            dy = cy - center_y
            normalized = math.sqrt((dx / radius_x) ** 2 + (dy / radius_y) ** 2)
            if normalized <= 1.0:
                possible.append((normalized, candidate_index, gt_index, dx, dy))

    used_candidates: set[int] = set()
    used_gt: set[int] = set()
    assignments = []
    for normalized, candidate_index, gt_index, dx, dy in sorted(possible):
        if candidate_index in used_candidates or gt_index in used_gt:
            continue
        used_candidates.add(candidate_index)
        used_gt.add(gt_index)
        predicted_pitch = _natural_pitch_from_y(
            float(candidates[candidate_index]["center"]["y"]), staff_lines
        )
        gt_pitch = _natural_pitch_name(str(ground_truth[gt_index]["pitch"]))
        assignments.append(
            {
                "candidate_id": candidates[candidate_index]["id"],
                "ground_truth_id": ground_truth[gt_index]["id"],
                "normalized_ellipse_distance": round(normalized, 6),
                "dx_px": round(dx, 3),
                "dy_px": round(dy, 3),
                "predicted_natural_pitch": predicted_pitch,
                "ground_truth_natural_pitch": gt_pitch,
                "pitch_correct": predicted_pitch == gt_pitch,
            }
        )
    tp = len(assignments)
    fp = len(candidates) - tp
    fn = len(ground_truth) - tp
    pitch_correct = sum(item["pitch_correct"] for item in assignments)
    return {
        "candidate_count": len(candidates),
        "gt_count": len(ground_truth),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _f1(tp, fp, fn),
        "pitch_correct": pitch_correct,
        "pitch_safe_accuracy": _ratio(pitch_correct, tp),
        "pitch_safe_recall": _ratio(pitch_correct, len(ground_truth)),
        "assignments": assignments,
    }


def _aggregate_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    pitch_correct = sum(row["pitch_correct"] for row in rows)
    gt_count = sum(row["gt_count"] for row in rows)
    return {
        "candidate_count": sum(row["candidate_count"] for row in rows),
        "gt_count": gt_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _f1(tp, fp, fn),
        "pitch_correct": pitch_correct,
        "pitch_safe_accuracy": _ratio(pitch_correct, tp),
        "pitch_safe_recall": _ratio(pitch_correct, gt_count),
    }


def _candidate_points(candidates: Sequence[Candidate]) -> list[dict[str, Any]]:
    return [candidate.to_point(index) for index, candidate in enumerate(candidates, start=1)]


def _write_diagnostics(
    prepared: dict[int, PreparedMeasure],
    ground_truth: dict[int, list[dict[str, Any]]],
    report: dict[str, Any],
    output_dir: Path,
) -> None:
    fold_by_measure = {fold["held_out_measure"]: fold for fold in report["folds"]}
    for measure, item in prepared.items():
        fold = fold_by_measure[measure]
        config_payload = fold["proposed"]["selected_config"]
        config = DetectorConfig(**config_payload)
        morphology = item.morphology[(config.line_support, config.vertical_preserve)]
        _write_mask(morphology["staff_mask"], output_dir / f"measure_{measure:03d}_staff_mask.png")
        _write_mask(
            morphology["staff_suppressed"],
            output_dir / f"measure_{measure:03d}_staff_suppressed.png",
        )
        _write_mask(
            morphology["structure"],
            output_dir / f"measure_{measure:03d}_staff_and_stem_suppressed.png",
        )
        _write_overlay(
            item.image,
            fold["proposed"]["candidates"],
            ground_truth[measure],
            item.staff_lines,
            output_dir / f"measure_{measure:03d}_proposed_overlay.png",
        )
        _write_overlay(
            item.image,
            fold["baseline"]["candidates"],
            ground_truth[measure],
            item.staff_lines,
            output_dir / f"measure_{measure:03d}_baseline_overlay.png",
        )


def _write_overlay(
    image: Image.Image,
    candidates: Sequence[dict[str, Any]],
    ground_truth: Sequence[dict[str, Any]],
    staff_lines: Sequence[int],
    path: Path,
) -> None:
    output = image.convert("RGB")
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    metrics = _match_region(candidates, ground_truth, staff_lines)
    matched_candidates = {assignment["candidate_id"] for assignment in metrics["assignments"]}
    matched_gt = {assignment["ground_truth_id"] for assignment in metrics["assignments"]}
    for item in ground_truth:
        cx, cy, rx, ry = _annotation_ellipse(item)
        color = (25, 150, 40) if item["id"] in matched_gt else (150, 25, 180)
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=color, width=2)
    for candidate in candidates:
        cx = float(candidate["center"]["x"])
        cy = float(candidate["center"]["y"])
        color = (25, 150, 40) if candidate["id"] in matched_candidates else (220, 35, 25)
        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), outline=color, width=2)
        draw.text((cx + 5, cy - 6), candidate["id"], fill=color, font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.save(path)


def _write_mask(mask: list[list[bool]], path: Path) -> None:
    height = len(mask)
    width = len(mask[0]) if mask else 0
    image = Image.new("L", (width, height), 255)
    pixels = image.load()
    for y, row in enumerate(mask):
        for x, value in enumerate(row):
            if value:
                pixels[x, y] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _write_report(report: dict[str, Any], output_dir: Path) -> None:
    json_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    aggregate = report["aggregate"]
    proposed = aggregate["proposed"]
    baseline = aggregate["baseline_staff_grid_density_v2"]
    lines = [
        "# Notehead Staff-Subtraction Spike",
        "",
        "All candidate lists were generated before human coordinate GT was read. "
        "Threshold/config selection is leave-one-measure-out.",
        "",
        "## Aggregate Held-Out Metrics",
        "",
        "| Detector | TP/FP/FN | Precision | Recall | F1 | Pitch-safe accuracy | "
        "Pitch-safe recall | R@4 | R@8 | R@12 | R@24 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _markdown_metric_row("staff subtraction", proposed),
        _markdown_metric_row("grid density v2", baseline),
        "",
        "## Per Fold",
        "",
        "| Held out | Detector | Selected | TP/FP/FN | P/R/F1 | Pitch-safe acc. | "
        "Pitch-safe recall | R@4/8/12/24 |",
        "| ---: | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for fold in report["folds"]:
        for detector_key, label in (("proposed", "staff subtraction"), ("baseline", "grid v2")):
            row = fold[detector_key]
            metrics = row["metrics"]
            selected = row.get("selected_config_key")
            if selected is None:
                selected = f"cap={row['selected_cap']}"
            top_k = row["top_k_recall"]
            lines.append(
                f"| {fold['held_out_measure']} | {label} | `{selected}` | "
                f"{metrics['tp']}/{metrics['fp']}/{metrics['fn']} | "
                f"{metrics['precision']:.3f}/{metrics['recall']:.3f}/{metrics['f1']:.3f} | "
                f"{metrics['pitch_safe_accuracy']:.3f} | {metrics['pitch_safe_recall']:.3f} | "
                f"{top_k['4']:.3f}/{top_k['8']:.3f}/{top_k['12']:.3f}/{top_k['24']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Held-out F1 delta vs grid density: `{aggregate['delta']['f1']:+.6f}`.",
            "- Pitch-safe accuracy is conditional on annotation-region matches; pitch-safe "
            "recall divides correct-pitch matches by all GT noteheads.",
            "- With four development measures, this is evidence for or against this morphology "
            "family only, not a production/generalization estimate.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_metric_row(label: str, metrics: dict[str, Any]) -> str:
    top_k = metrics["top_k_recall"]
    return (
        f"| {label} | {metrics['tp']}/{metrics['fp']}/{metrics['fn']} | "
        f"{metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} | "
        f"{metrics['pitch_safe_accuracy']:.3f} | {metrics['pitch_safe_recall']:.3f} | "
        f"{top_k['4']:.3f} | {top_k['8']:.3f} | {top_k['12']:.3f} | {top_k['24']:.3f} |"
    )


def _measure_record(out_dir: Path, slug: str, measure: int) -> dict[str, Any]:
    manifest_path = out_dir / "vlm_melody_inputs_manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing VLM melody input manifest: {manifest_path}")
    matches = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if (
            record.get("slug") == slug
            and int(record["system_index"]) == SYSTEM_INDEX
            and int(record["system_measure_index"]) == measure
        ):
            matches.append(record)
    if len(matches) != 1:
        raise ValueError(
            f"Expected one manifest record for {slug} S{SYSTEM_INDEX} M{measure}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _resolve_path(out_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = out_dir / path
    if candidate.exists():
        return candidate
    repo_candidate = REPO_ROOT / path
    if repo_candidate.exists():
        return repo_candidate
    raise FileNotFoundError(f"Referenced artifact does not exist: {value}")


def _load_ground_truth(slug: str, measure: int) -> list[dict[str, Any]]:
    path = GT_DIR / f"{slug}_system_{SYSTEM_INDEX:03d}_measure_{measure:03d}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    noteheads = payload.get("noteheads")
    if not isinstance(noteheads, list) or not noteheads:
        raise ValueError(f"Invalid notehead GT fixture: {path}")
    return noteheads


def _annotation_ellipse(item: dict[str, Any]) -> tuple[float, float, float, float]:
    geometry = item["annotation_geometry"]
    bbox = geometry["bbox_px"]
    return (
        (float(bbox["left"]) + float(bbox["right"])) / 2.0,
        (float(bbox["top"]) + float(bbox["bottom"])) / 2.0,
        float(geometry["radius_x_px"]),
        float(geometry["radius_y_px"]),
    )


def _natural_pitch_name(pitch: str) -> str:
    letters = "ABCDEFG"
    stripped = pitch.strip()
    if not stripped or stripped[0].upper() not in letters:
        raise ValueError(f"Unsupported pitch: {pitch!r}")
    letter = stripped[0].upper()
    octave = "".join(
        character for character in stripped[1:] if character.isdigit() or character == "-"
    )
    if not octave:
        raise ValueError(f"Unsupported pitch: {pitch!r}")
    return f"{letter}{octave}"


def _natural_pitch_from_y(y: float, staff_lines: Sequence[int]) -> str:
    spacing = statistics.mean(b - a for a, b in zip(staff_lines, staff_lines[1:], strict=False))
    step = math.floor((y - float(staff_lines[0])) / (spacing / 2.0) + 0.5)
    letters = ("C", "D", "E", "F", "G", "A", "B")
    top_letter_index = letters.index("F")
    letter_index = (top_letter_index - step) % len(letters)
    octave = 5 + (top_letter_index - step) // len(letters)
    return f"{letters[letter_index]}{octave}"


def _prefix_sum(values: Sequence[bool]) -> list[int]:
    prefix = [0]
    for value in values:
        prefix.append(prefix[-1] + int(value))
    return prefix


def _range_sum(prefix: Sequence[int], start: int, end: int) -> int:
    bounded_start = max(0, min(len(prefix) - 1, start))
    bounded_end = max(bounded_start, min(len(prefix) - 1, end))
    return prefix[bounded_end] - prefix[bounded_start]


def _empty_mask(width: int, height: int) -> list[list[bool]]:
    return [[False for _ in range(width)] for _ in range(height)]


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return round(2 * precision * recall / (precision + recall), 6) if precision + recall else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
