"""Rank blind notehead proposals with small classical models.

This is a bounded development spike. It uses only Aviador system 1 measures
1-4, labels cap-24 proposals with the authoritative 1.15x human annotation
ellipses, and evaluates every learned scorer with leave-one-measure-out folds.
No pitch labels are loaded or used.

Example:
    python scripts/experiments/spike_notehead_candidate_classifier.py out
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_vlm_notehead_candidates as detector  # noqa: E402

DEFAULT_SLUG = "jaime-llanos_12_aviador_pasillo_fulgencio-garcia"
DEFAULT_MEASURES = (1, 2, 3, 4)
DEFAULT_MAX_CANDIDATES = 24
ANNOTATION_REGION_MARGIN = 1.15
GT_DIR = REPO_ROOT / "tests/fixtures/vlm_melody/notehead_ground_truth"
OUTPUT_SUBDIR = Path("experiments/notehead_candidate_classifier")
TOP_K_VALUES = (1, 3, 4, 5, 8, 12, 16, 24)

DETECTOR_FEATURES = (
    "detector_score",
    "ink_density",
    "core_density",
    "vertical_support",
    "horizontal_support",
    "row_peak_density",
    "column_peak_density",
    "line_dominance",
    "stem_evidence",
)
PATCH_FEATURES = (
    "patch_density",
    "patch_centroid_x",
    "patch_centroid_y",
    "patch_variance_x",
    "patch_variance_y",
    "patch_covariance_abs",
    "patch_lr_balance_abs",
    "patch_tb_balance_abs",
    "patch_horizontal_symmetry",
    "patch_vertical_symmetry",
    "suppressed_density",
    "suppressed_centroid_x",
    "suppressed_centroid_y",
    "suppressed_largest_component_fraction",
    "suppressed_largest_component_aspect",
    "suppressed_largest_component_fill",
    "blob_raw_density",
    "blob_raw_contrast",
    "blob_suppressed_density",
    "blob_suppressed_contrast",
    "blob_offset_x_abs",
    "blob_offset_y_abs",
)
GEOMETRY_FEATURES = (
    "rank_fraction",
    "x_fraction",
    "x_edge_distance_fraction",
    "y_fraction",
    "staff_band_distance",
    "staff_midline_distance",
    "bbox_width_in_staff_spaces",
    "bbox_height_in_staff_spaces",
)

FEATURE_SETS = {
    "detector": DETECTOR_FEATURES,
    "patch": PATCH_FEATURES,
    "detector_patch": DETECTOR_FEATURES + PATCH_FEATURES,
    "detector_patch_geometry": DETECTOR_FEATURES + PATCH_FEATURES + GEOMETRY_FEATURES,
}
MODEL_KINDS = ("pairwise_logistic", "logistic_l2", "diagonal_lda", "gaussian_nb")


@dataclass(frozen=True)
class Ellipse:
    id: str
    center_x: float
    center_y: float
    radius_x: float
    radius_y: float


@dataclass(frozen=True)
class CandidateRow:
    measure: int
    id: str
    rank: int
    center_x: float
    center_y: float
    label: int
    matched_ellipse_id: str | None
    features: dict[str, float]


@dataclass(frozen=True)
class MeasureData:
    measure: int
    source_image: Path
    ground_truth_path: Path
    staff_lines: tuple[int, ...]
    rows: tuple[CandidateRow, ...]
    ellipses: tuple[Ellipse, ...]


@dataclass(frozen=True)
class FittedScorer:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    score_standardized: Callable[[Sequence[float]], float]

    def score(self, row: CandidateRow) -> float:
        values = tuple(row.features[name] for name in self.feature_names)
        standardized = tuple(
            (value - mean) / scale
            for value, mean, scale in zip(values, self.means, self.scales, strict=True)
        )
        return float(self.score_standardized(standardized))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        measures = _positive_unique(args.measure or DEFAULT_MEASURES, "measure")
        data = [
            _load_measure_data(
                args.out_dir,
                slug=args.slug,
                system_index=args.system,
                measure=measure,
                max_candidates=args.max_candidates,
            )
            for measure in measures
        ]
        report = evaluate_leave_one_measure_out(
            data,
            slug=args.slug,
            system_index=args.system,
            max_candidates=args.max_candidates,
        )
        output_dir = args.out_dir / OUTPUT_SUBDIR
        json_path, markdown_path = write_report(report, output_dir)
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json_path)
    print(markdown_path)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", type=Path, default=Path("out"))
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--system", type=int, default=1)
    parser.add_argument("--measure", action="append", type=int)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    return parser


def _load_measure_data(
    out_dir: Path,
    *,
    slug: str,
    system_index: int,
    measure: int,
    max_candidates: int,
) -> MeasureData:
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    base_record = detector._load_or_build_base_record(
        out_dir,
        slug=slug,
        system_index=system_index,
        measure_index=measure,
    )
    context_path = detector._resolve_path(out_dir, base_record["paths"]["context"])
    context = json.loads(context_path.read_text(encoding="utf-8"))
    source_path = detector._source_path(out_dir, base_record, "raw")
    staff_lines = tuple(detector._staff_lines_for_source(context, "raw"))

    with Image.open(source_path) as opened:
        image = opened.convert("RGB")
    stages = detector._detect_staff_grid_stages(
        image,
        staff_lines=list(staff_lines),
        max_candidates=max_candidates,
    )

    # Candidate generation and feature extraction are complete before annotations are read.
    feature_rows = [
        _candidate_feature_row(
            image,
            stages.threshold_mask,
            candidate,
            rank=rank,
            max_candidates=max_candidates,
            staff_lines=staff_lines,
            staff_spacing=stages.staff_spacing,
        )
        for rank, candidate in enumerate(stages.candidates, start=1)
    ]

    ground_truth_path = GT_DIR / f"{slug}_system_{system_index:03d}_measure_{measure:03d}.json"
    ellipses = tuple(_load_annotation_ellipses(ground_truth_path))
    rows = []
    for rank, (candidate, features) in enumerate(
        zip(stages.candidates, feature_rows, strict=True), start=1
    ):
        matched = _ellipse_containing_point(candidate.center, ellipses)
        rows.append(
            CandidateRow(
                measure=measure,
                id=f"c{rank:03d}",
                rank=rank,
                center_x=float(candidate.center[0]),
                center_y=float(candidate.center[1]),
                label=int(matched is not None),
                matched_ellipse_id=matched.id if matched else None,
                features=features,
            )
        )
    return MeasureData(
        measure=measure,
        source_image=source_path,
        ground_truth_path=ground_truth_path,
        staff_lines=staff_lines,
        rows=tuple(rows),
        ellipses=ellipses,
    )


def _candidate_feature_row(
    image: Image.Image,
    mask: Sequence[Sequence[bool]],
    candidate: Any,
    *,
    rank: int,
    max_candidates: int,
    staff_lines: Sequence[int],
    staff_spacing: float,
) -> dict[str, float]:
    width, height = image.size
    center_x, center_y = candidate.center
    features = {"detector_score": float(candidate.score)}
    features.update({name: float(value) for name, value in candidate.features.items()})
    features.update(
        _patch_features(
            mask,
            center_x=center_x,
            center_y=center_y,
            bbox=candidate.bbox,
            staff_lines=staff_lines,
            staff_spacing=staff_spacing,
        )
    )
    top_staff = float(staff_lines[0])
    bottom_staff = float(staff_lines[-1])
    outside_distance = max(top_staff - center_y, center_y - bottom_staff, 0.0)
    staff_midline = (top_staff + bottom_staff) / 2.0
    features.update(
        {
            "rank_fraction": rank / max_candidates,
            "x_fraction": center_x / max(1.0, width - 1.0),
            "x_edge_distance_fraction": min(center_x, width - 1.0 - center_x)
            / max(1.0, width / 2.0),
            "y_fraction": center_y / max(1.0, height - 1.0),
            "staff_band_distance": outside_distance / staff_spacing,
            "staff_midline_distance": abs(center_y - staff_midline) / staff_spacing,
            "bbox_width_in_staff_spaces": candidate.width / staff_spacing,
            "bbox_height_in_staff_spaces": candidate.height / staff_spacing,
        }
    )
    return features


def _patch_features(
    mask: Sequence[Sequence[bool]],
    *,
    center_x: float,
    center_y: float,
    bbox: Sequence[int],
    staff_lines: Sequence[int],
    staff_spacing: float,
) -> dict[str, float]:
    height = len(mask)
    width = len(mask[0]) if mask else 0
    left, top, right, bottom = (int(value) for value in bbox)
    raw_patch = _extract_patch(mask, left=left, top=top, right=right, bottom=bottom)
    suppressed_patch = [row[:] for row in raw_patch]
    for patch_y, source_y in enumerate(range(top, bottom + 1)):
        if any(abs(source_y - line_y) <= 1 for line_y in staff_lines):
            suppressed_patch[patch_y] = [False] * len(suppressed_patch[patch_y])

    raw_projection = _projection_descriptors(raw_patch, "patch")
    suppressed_projection = _projection_descriptors(suppressed_patch, "suppressed")
    component = _largest_component_descriptors(suppressed_patch)
    raw_blob = _best_blob(
        mask,
        center_x=center_x,
        center_y=center_y,
        staff_lines=staff_lines,
        staff_spacing=staff_spacing,
        suppress_staff=False,
        image_width=width,
        image_height=height,
    )
    suppressed_blob = _best_blob(
        mask,
        center_x=center_x,
        center_y=center_y,
        staff_lines=staff_lines,
        staff_spacing=staff_spacing,
        suppress_staff=True,
        image_width=width,
        image_height=height,
    )
    return {
        **raw_projection,
        **suppressed_projection,
        **component,
        "blob_raw_density": raw_blob[0],
        "blob_raw_contrast": raw_blob[1],
        "blob_suppressed_density": suppressed_blob[0],
        "blob_suppressed_contrast": suppressed_blob[1],
        "blob_offset_x_abs": abs(suppressed_blob[2]),
        "blob_offset_y_abs": abs(suppressed_blob[3]),
    }


def _extract_patch(
    mask: Sequence[Sequence[bool]], *, left: int, top: int, right: int, bottom: int
) -> list[list[bool]]:
    image_height = len(mask)
    image_width = len(mask[0]) if mask else 0
    return [
        [
            0 <= source_x < image_width
            and 0 <= source_y < image_height
            and bool(mask[source_y][source_x])
            for source_x in range(left, right + 1)
        ]
        for source_y in range(top, bottom + 1)
    ]


def _projection_descriptors(patch: Sequence[Sequence[bool]], prefix: str) -> dict[str, float]:
    height = len(patch)
    width = len(patch[0]) if patch else 0
    points = [(x, y) for y, row in enumerate(patch) for x, value in enumerate(row) if value]
    area = max(1, width * height)
    density = len(points) / area
    if not points:
        return {
            f"{prefix}_density": 0.0,
            f"{prefix}_centroid_x": 0.0,
            f"{prefix}_centroid_y": 0.0,
            f"{prefix}_variance_x": 0.0,
            f"{prefix}_variance_y": 0.0,
            f"{prefix}_covariance_abs": 0.0,
            f"{prefix}_lr_balance_abs": 0.0,
            f"{prefix}_tb_balance_abs": 0.0,
            f"{prefix}_horizontal_symmetry": 0.0,
            f"{prefix}_vertical_symmetry": 0.0,
        }

    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    x_scale = max(1.0, width - 1.0)
    y_scale = max(1.0, height - 1.0)
    variance_x = sum(((x - mean_x) / x_scale) ** 2 for x, _ in points) / len(points)
    variance_y = sum(((y - mean_y) / y_scale) ** 2 for _, y in points) / len(points)
    covariance = sum(((x - mean_x) / x_scale) * ((y - mean_y) / y_scale) for x, y in points) / len(
        points
    )
    left_count = sum(x < width / 2.0 for x, _ in points)
    top_count = sum(y < height / 2.0 for _, y in points)
    horizontal_symmetry = _reflection_similarity(patch, horizontal=True)
    vertical_symmetry = _reflection_similarity(patch, horizontal=False)
    return {
        f"{prefix}_density": density,
        f"{prefix}_centroid_x": mean_x / x_scale - 0.5,
        f"{prefix}_centroid_y": mean_y / y_scale - 0.5,
        f"{prefix}_variance_x": variance_x,
        f"{prefix}_variance_y": variance_y,
        f"{prefix}_covariance_abs": abs(covariance),
        f"{prefix}_lr_balance_abs": abs(2.0 * left_count / len(points) - 1.0),
        f"{prefix}_tb_balance_abs": abs(2.0 * top_count / len(points) - 1.0),
        f"{prefix}_horizontal_symmetry": horizontal_symmetry,
        f"{prefix}_vertical_symmetry": vertical_symmetry,
    }


def _reflection_similarity(patch: Sequence[Sequence[bool]], *, horizontal: bool) -> float:
    height = len(patch)
    width = len(patch[0]) if patch else 0
    compared = 0
    equal = 0
    for y in range(height):
        for x in range(width):
            reflected_x = width - 1 - x if horizontal else x
            reflected_y = y if horizontal else height - 1 - y
            if (horizontal and x >= reflected_x) or (not horizontal and y >= reflected_y):
                continue
            compared += 1
            equal += patch[y][x] == patch[reflected_y][reflected_x]
    return equal / compared if compared else 0.0


def _largest_component_descriptors(patch: Sequence[Sequence[bool]]) -> dict[str, float]:
    height = len(patch)
    width = len(patch[0]) if patch else 0
    seen: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []
    for y in range(height):
        for x in range(width):
            if not patch[y][x] or (x, y) in seen:
                continue
            stack = [(x, y)]
            seen.add((x, y))
            component = []
            while stack:
                current_x, current_y = stack.pop()
                component.append((current_x, current_y))
                for next_y in range(max(0, current_y - 1), min(height, current_y + 2)):
                    for next_x in range(max(0, current_x - 1), min(width, current_x + 2)):
                        point = (next_x, next_y)
                        if patch[next_y][next_x] and point not in seen:
                            seen.add(point)
                            stack.append(point)
            components.append(component)
    if not components:
        return {
            "suppressed_largest_component_fraction": 0.0,
            "suppressed_largest_component_aspect": 0.0,
            "suppressed_largest_component_fill": 0.0,
        }
    largest = max(components, key=lambda item: (len(item), -min(x for x, _ in item)))
    min_x = min(x for x, _ in largest)
    max_x = max(x for x, _ in largest)
    min_y = min(y for _, y in largest)
    max_y = max(y for _, y in largest)
    component_width = max_x - min_x + 1
    component_height = max_y - min_y + 1
    return {
        "suppressed_largest_component_fraction": len(largest) / max(1, width * height),
        "suppressed_largest_component_aspect": component_width / max(1, component_height),
        "suppressed_largest_component_fill": len(largest)
        / max(1, component_width * component_height),
    }


def _best_blob(
    mask: Sequence[Sequence[bool]],
    *,
    center_x: float,
    center_y: float,
    staff_lines: Sequence[int],
    staff_spacing: float,
    suppress_staff: bool,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    radius_x = max(3.0, staff_spacing * 0.34)
    radius_y = max(2.5, staff_spacing * 0.24)
    search_radius = max(2, round(staff_spacing * 0.25))
    best = (-1.0, -1.0, 0.0, 0.0)
    for offset_y in range(-search_radius, search_radius + 1):
        for offset_x in range(-search_radius, search_radius + 1):
            trial_x = center_x + offset_x
            trial_y = center_y + offset_y
            inside_count = 0
            inside_ink = 0
            ring_count = 0
            ring_ink = 0
            left = max(0, math.floor(trial_x - radius_x * 1.55))
            right = min(image_width - 1, math.ceil(trial_x + radius_x * 1.55))
            top = max(0, math.floor(trial_y - radius_y * 1.55))
            bottom = min(image_height - 1, math.ceil(trial_y + radius_y * 1.55))
            for y in range(top, bottom + 1):
                for x in range(left, right + 1):
                    normalized = ((x - trial_x) / radius_x) ** 2 + ((y - trial_y) / radius_y) ** 2
                    if normalized > 1.55**2:
                        continue
                    value = bool(mask[y][x])
                    if suppress_staff and any(abs(y - line_y) <= 1 for line_y in staff_lines):
                        value = False
                    if normalized <= 1.0:
                        inside_count += 1
                        inside_ink += value
                    else:
                        ring_count += 1
                        ring_ink += value
            density = inside_ink / max(1, inside_count)
            contrast = density - ring_ink / max(1, ring_count)
            candidate = (density, contrast, offset_x / staff_spacing, offset_y / staff_spacing)
            if (contrast, density, -abs(offset_x) - abs(offset_y)) > (
                best[1],
                best[0],
                -abs(best[2] * staff_spacing) - abs(best[3] * staff_spacing),
            ):
                best = candidate
    return best


def _load_annotation_ellipses(path: Path) -> list[Ellipse]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    noteheads = payload.get("noteheads") if isinstance(payload, dict) else payload
    if not isinstance(noteheads, list):
        raise ValueError(f"Ground-truth JSON has no noteheads list: {path}")
    ellipses = []
    for index, item in enumerate(noteheads, start=1):
        geometry = item.get("annotation_geometry")
        if not isinstance(geometry, dict):
            raise ValueError(f"Missing annotation geometry in {path} item {index}")
        bbox = geometry.get("bbox_px")
        if not isinstance(bbox, dict):
            raise ValueError(f"Missing annotation bbox in {path} item {index}")
        radius_x = float(geometry["radius_x_px"]) * ANNOTATION_REGION_MARGIN
        radius_y = float(geometry["radius_y_px"]) * ANNOTATION_REGION_MARGIN
        ellipses.append(
            Ellipse(
                id=str(item.get("id", f"n{index:03d}")),
                center_x=(float(bbox["left"]) + float(bbox["right"])) / 2.0,
                center_y=(float(bbox["top"]) + float(bbox["bottom"])) / 2.0,
                radius_x=radius_x,
                radius_y=radius_y,
            )
        )
    return ellipses


def _ellipse_containing_point(
    point: tuple[float, float], ellipses: Sequence[Ellipse]
) -> Ellipse | None:
    x, y = point
    eligible = []
    for ellipse in ellipses:
        normalized = math.sqrt(
            ((x - ellipse.center_x) / ellipse.radius_x) ** 2
            + ((y - ellipse.center_y) / ellipse.radius_y) ** 2
        )
        if normalized <= 1.0:
            eligible.append((normalized, ellipse.id, ellipse))
    return min(eligible)[2] if eligible else None


def evaluate_leave_one_measure_out(
    measures: Sequence[MeasureData], *, slug: str, system_index: int, max_candidates: int
) -> dict[str, Any]:
    if len(measures) < 3:
        raise ValueError("Leave-one-measure-out evaluation requires at least three measures")
    expected_features = set(DETECTOR_FEATURES + PATCH_FEATURES + GEOMETRY_FEATURES)
    for measure in measures:
        for row in measure.rows:
            missing = expected_features - row.features.keys()
            if missing:
                raise ValueError(f"Candidate {measure.measure}/{row.id} lacks features: {missing}")

    baselines = _evaluate_baselines(measures)
    methods = []
    for feature_set_name, feature_names in FEATURE_SETS.items():
        for model_kind in MODEL_KINDS:
            methods.append(
                _evaluate_method(
                    measures,
                    feature_set_name=feature_set_name,
                    feature_names=feature_names,
                    model_kind=model_kind,
                )
            )
    methods.sort(
        key=lambda method: (
            -method["aggregate"]["threshold_selection"]["f1"],
            -method["aggregate"]["top_k"]["4"]["recall"],
            method["id"],
        )
    )
    best_threshold_method = methods[0]
    best_top4_method = max(
        methods,
        key=lambda method: (
            method["aggregate"]["top_k"]["4"]["f1"],
            method["aggregate"]["top_k"]["4"]["recall"],
            method["id"],
        ),
    )
    baseline_cap4 = baselines["detector_cap4"]
    best_threshold = best_threshold_method["aggregate"]["threshold_selection"]
    best_top4 = best_top4_method["aggregate"]["top_k"]["4"]
    threshold_is_material = (
        best_threshold["f1"] >= baseline_cap4["f1"] + 0.05
        and best_threshold["recall"] >= baseline_cap4["recall"]
    )
    rank_is_material = (
        best_top4["f1"] >= baseline_cap4["f1"] + 0.05 and best_top4["tp"] >= baseline_cap4["tp"] + 2
    )
    material = threshold_is_material or rank_is_material
    return {
        "schema_version": 1,
        "kind": "classical_notehead_candidate_ranking_spike",
        "scope": {
            "slug": slug,
            "system_index": system_index,
            "measures": [measure.measure for measure in measures],
            "outer_evaluation": "leave-one-measure-out",
            "development_only": True,
            "pitch_labels_accessed_or_used": False,
            "candidate_generation_uses_ground_truth": False,
            "label_rule": "candidate center inside authoritative annotation ellipse expanded 1.15x",
            "annotation_ellipse_margin": ANNOTATION_REGION_MARGIN,
            "candidate_cap": max_candidates,
        },
        "dataset": [_measure_summary(measure) for measure in measures],
        "feature_sets": {name: list(features) for name, features in FEATURE_SETS.items()},
        "model_kinds": list(MODEL_KINDS),
        "baselines": baselines,
        "methods": methods,
        "selection": {
            "verdict": "material_improvement" if material else "marginal_not_material",
            "best_threshold_method_id": best_threshold_method["id"],
            "best_top4_ranker_method_id": best_top4_method["id"],
            "threshold_selection_metric": ("aggregate leave-one-measure-out learned-threshold F1"),
            "ranking_selection_metric": "aggregate leave-one-measure-out top-4 F1",
            "best_threshold_metrics": best_threshold,
            "best_top4_metrics": best_top4,
            "delta_f1_vs_detector_cap4": round(best_threshold["f1"] - baseline_cap4["f1"], 6),
            "delta_top4_recall_vs_detector_cap4": round(
                best_top4["recall"] - baseline_cap4["recall"], 6
            ),
            "delta_top4_true_positives_vs_detector_cap4": (best_top4["tp"] - baseline_cap4["tp"]),
            "material_improvement_rule": {
                "threshold": ("F1 improves by at least 0.05 without reducing recall"),
                "ranking": ("top-4 F1 improves by at least 0.05 and finds at least two more heads"),
            },
            "material_improvement_by_spike_rule": material,
        },
        "caveats": [
            "Only four development measures and fourteen annotated noteheads are available.",
            (
                "Outer folds are honest, but choosing the best method on these same folds is "
                "model-selection bias."
            ),
            "Oracle-count top-k uses held-out GT count and is diagnostic, not deployable.",
            (
                "Threshold prediction count is learned from training folds only and is "
                "deployable in principle."
            ),
            "A separate work or system with coordinate GT is required before production use.",
        ],
    }


def _evaluate_baselines(measures: Sequence[MeasureData]) -> dict[str, Any]:
    return {
        "detector_cap4": _aggregate_fixed_rank(measures, 4),
        "detector_cap24": _aggregate_fixed_rank(measures, 24),
        "detector_oracle_count": _aggregate_oracle_count(measures, _detector_scores),
    }


def _evaluate_method(
    measures: Sequence[MeasureData],
    *,
    feature_set_name: str,
    feature_names: Sequence[str],
    model_kind: str,
) -> dict[str, Any]:
    fold_reports = []
    for held_out in measures:
        training = [measure for measure in measures if measure.measure != held_out.measure]
        training_rows = [row for measure in training for row in measure.rows]
        scorer = _fit_scorer(model_kind, training_rows, tuple(feature_names))
        training_scores = {id(row): scorer.score(row) for row in training_rows}
        threshold, training_threshold_metrics = _select_training_threshold(
            training, training_scores
        )
        held_out_scores = {id(row): scorer.score(row) for row in held_out.rows}
        threshold_selected = [row for row in held_out.rows if held_out_scores[id(row)] >= threshold]
        threshold_metrics = _selection_metrics(threshold_selected, held_out)
        ranked = sorted(
            held_out.rows,
            key=lambda row: (-held_out_scores[id(row)], row.rank, row.id),
        )
        top_k = {
            str(k): _selection_metrics(ranked[: min(k, len(ranked))], held_out)
            for k in TOP_K_VALUES
        }
        oracle_count = _selection_metrics(ranked[: len(held_out.ellipses)], held_out)
        fold_reports.append(
            {
                "held_out_measure": held_out.measure,
                "training_measures": [measure.measure for measure in training],
                "training_candidate_count": len(training_rows),
                "training_positive_candidate_count": sum(row.label for row in training_rows),
                "learned_threshold": round(threshold, 9),
                "training_threshold_metrics": training_threshold_metrics,
                "threshold_selection": threshold_metrics,
                "top_k": top_k,
                "oracle_count": oracle_count,
                "ranked_candidates": [
                    {
                        "candidate_id": row.id,
                        "score": round(held_out_scores[id(row)], 9),
                        "label": row.label,
                        "matched_ellipse_id": row.matched_ellipse_id,
                        "selected_by_threshold": held_out_scores[id(row)] >= threshold,
                    }
                    for row in ranked
                ],
            }
        )
    return {
        "id": f"{model_kind}__{feature_set_name}",
        "model_kind": model_kind,
        "feature_set": feature_set_name,
        "feature_count": len(feature_names),
        "folds": fold_reports,
        "aggregate": _aggregate_folds(fold_reports),
    }


def _fit_scorer(
    model_kind: str, rows: Sequence[CandidateRow], feature_names: tuple[str, ...]
) -> FittedScorer:
    if not rows:
        raise ValueError("Cannot fit a scorer without candidates")
    labels = [row.label for row in rows]
    if not any(labels) or all(labels):
        raise ValueError("Training fold must contain positive and negative candidates")
    matrix = [[row.features[name] for name in feature_names] for row in rows]
    means, scales, standardized = _standardize(matrix)
    if model_kind == "pairwise_logistic":
        score_function = _fit_pairwise_logistic(
            standardized,
            labels,
            [row.measure for row in rows],
        )
    elif model_kind == "logistic_l2":
        score_function = _fit_logistic_l2(standardized, labels)
    elif model_kind == "diagonal_lda":
        score_function = _fit_diagonal_lda(standardized, labels)
    elif model_kind == "gaussian_nb":
        score_function = _fit_gaussian_nb(standardized, labels)
    else:
        raise ValueError(f"Unsupported model kind: {model_kind}")
    return FittedScorer(feature_names, tuple(means), tuple(scales), score_function)


def _standardize(
    matrix: Sequence[Sequence[float]],
) -> tuple[list[float], list[float], list[list[float]]]:
    feature_count = len(matrix[0])
    means = [sum(row[index] for row in matrix) / len(matrix) for index in range(feature_count)]
    scales = []
    for index, mean in enumerate(means):
        variance = sum((row[index] - mean) ** 2 for row in matrix) / len(matrix)
        scales.append(max(math.sqrt(variance), 1e-6))
    standardized = [
        [(value - means[index]) / scales[index] for index, value in enumerate(row)]
        for row in matrix
    ]
    return means, scales, standardized


def _fit_logistic_l2(
    matrix: Sequence[Sequence[float]], labels: Sequence[int]
) -> Callable[[Sequence[float]], float]:
    feature_count = len(matrix[0])
    weights = [0.0] * feature_count
    intercept = 0.0
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    positive_weight = len(labels) / (2.0 * positive_count)
    negative_weight = len(labels) / (2.0 * negative_count)
    weight_total = sum(positive_weight if label else negative_weight for label in labels)
    l2 = 1.0
    for iteration in range(1200):
        gradient = [0.0] * feature_count
        intercept_gradient = 0.0
        for row, label in zip(matrix, labels, strict=True):
            probability = _sigmoid(intercept + _dot(weights, row))
            class_weight = positive_weight if label else negative_weight
            error = (probability - label) * class_weight
            intercept_gradient += error
            for index, value in enumerate(row):
                gradient[index] += error * value
        learning_rate = 0.08 / (1.0 + iteration / 500.0)
        intercept -= learning_rate * intercept_gradient / weight_total
        for index in range(feature_count):
            regularized = gradient[index] / weight_total + l2 * weights[index] / len(labels)
            weights[index] -= learning_rate * regularized

    return lambda row: _sigmoid(intercept + _dot(weights, row))


def _fit_pairwise_logistic(
    matrix: Sequence[Sequence[float]],
    labels: Sequence[int],
    measure_ids: Sequence[int],
) -> Callable[[Sequence[float]], float]:
    """Fit a linear ranker from positive-negative pairs within each measure."""
    pairs = []
    for measure_id in sorted(set(measure_ids)):
        positives = [
            row
            for row, label, row_measure in zip(matrix, labels, measure_ids, strict=True)
            if row_measure == measure_id and label
        ]
        negatives = [
            row
            for row, label, row_measure in zip(matrix, labels, measure_ids, strict=True)
            if row_measure == measure_id and not label
        ]
        pairs.extend(
            [positive[index] - negative[index] for index in range(len(positive))]
            for positive in positives
            for negative in negatives
        )
    if not pairs:
        raise ValueError("Pairwise ranker requires positive-negative pairs")

    feature_count = len(matrix[0])
    weights = [0.0] * feature_count
    l2 = 1.0
    for iteration in range(1000):
        gradient = [0.0] * feature_count
        for difference in pairs:
            error = -_sigmoid(-_dot(weights, difference))
            for index, value in enumerate(difference):
                gradient[index] += error * value
        learning_rate = 0.06 / (1.0 + iteration / 400.0)
        for index in range(feature_count):
            regularized = gradient[index] / len(pairs) + l2 * weights[index] / len(pairs)
            weights[index] -= learning_rate * regularized

    return lambda row: _dot(weights, row)


def _fit_diagonal_lda(
    matrix: Sequence[Sequence[float]], labels: Sequence[int]
) -> Callable[[Sequence[float]], float]:
    positive = [row for row, label in zip(matrix, labels, strict=True) if label]
    negative = [row for row, label in zip(matrix, labels, strict=True) if not label]
    positive_mean = _column_means(positive)
    negative_mean = _column_means(negative)
    variances = []
    for index in range(len(matrix[0])):
        numerator = sum((row[index] - positive_mean[index]) ** 2 for row in positive)
        numerator += sum((row[index] - negative_mean[index]) ** 2 for row in negative)
        variances.append(numerator / max(1, len(matrix) - 2) + 0.5)
    weights = [
        (positive_mean[index] - negative_mean[index]) / variances[index]
        for index in range(len(variances))
    ]
    midpoint = [
        (positive_mean[index] + negative_mean[index]) / 2.0 for index in range(len(variances))
    ]
    return lambda row: _dot(
        weights,
        [value - mid for value, mid in zip(row, midpoint, strict=True)],
    )


def _fit_gaussian_nb(
    matrix: Sequence[Sequence[float]], labels: Sequence[int]
) -> Callable[[Sequence[float]], float]:
    positive = [row for row, label in zip(matrix, labels, strict=True) if label]
    negative = [row for row, label in zip(matrix, labels, strict=True) if not label]
    positive_mean = _column_means(positive)
    negative_mean = _column_means(negative)
    positive_variance = _column_variances(positive, positive_mean, floor=0.35)
    negative_variance = _column_variances(negative, negative_mean, floor=0.35)

    def score(row: Sequence[float]) -> float:
        positive_log_likelihood = _diagonal_gaussian_log_likelihood(
            row, positive_mean, positive_variance
        )
        negative_log_likelihood = _diagonal_gaussian_log_likelihood(
            row, negative_mean, negative_variance
        )
        return positive_log_likelihood - negative_log_likelihood

    return score


def _select_training_threshold(
    training: Sequence[MeasureData], scores: dict[int, float]
) -> tuple[float, dict[str, Any]]:
    score_values = sorted({scores[id(row)] for measure in training for row in measure.rows})
    epsilon = 1e-9
    thresholds = [score_values[-1] + epsilon, *reversed(score_values)]
    best_threshold = thresholds[0]
    best_metrics: dict[str, Any] | None = None
    for threshold in thresholds:
        per_measure = []
        for measure in training:
            selected = [row for row in measure.rows if scores[id(row)] >= threshold]
            per_measure.append(_selection_metrics(selected, measure))
        metrics = _aggregate_metric_rows(per_measure)
        ranking = (
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
            -metrics["selected_count"],
            threshold,
        )
        if best_metrics is None or ranking > best_metrics["_ranking"]:
            best_threshold = threshold
            best_metrics = {**metrics, "_ranking": ranking}
    assert best_metrics is not None
    best_metrics.pop("_ranking")
    return best_threshold, best_metrics


def _selection_metrics(selected: Sequence[CandidateRow], measure: MeasureData) -> dict[str, Any]:
    possible_pairs = []
    for candidate_index, row in enumerate(selected):
        for ellipse_index, ellipse in enumerate(measure.ellipses):
            normalized = math.sqrt(
                ((row.center_x - ellipse.center_x) / ellipse.radius_x) ** 2
                + ((row.center_y - ellipse.center_y) / ellipse.radius_y) ** 2
            )
            if normalized <= 1.0:
                possible_pairs.append(
                    (normalized, candidate_index, ellipse_index, row.id, ellipse.id)
                )
    used_candidates: set[int] = set()
    used_ellipses: set[int] = set()
    assignments = []
    for distance, candidate_index, ellipse_index, candidate_id, ellipse_id in sorted(
        possible_pairs
    ):
        if candidate_index in used_candidates or ellipse_index in used_ellipses:
            continue
        used_candidates.add(candidate_index)
        used_ellipses.add(ellipse_index)
        assignments.append(
            {
                "candidate_id": candidate_id,
                "ellipse_id": ellipse_id,
                "normalized_ellipse_distance": round(distance, 6),
            }
        )
    tp = len(assignments)
    fp = len(selected) - tp
    fn = len(measure.ellipses) - tp
    return {
        "selected_count": len(selected),
        "gt_count": len(measure.ellipses),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _f1(tp, fp, fn),
        "exact_note_count": len(selected) == len(measure.ellipses),
        "exact_set": fp == 0 and fn == 0,
        "assignments": assignments,
    }


def _aggregate_folds(folds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "threshold_selection": _aggregate_metric_rows(
            [fold["threshold_selection"] for fold in folds]
        ),
        "top_k": {
            str(k): _aggregate_metric_rows([fold["top_k"][str(k)] for fold in folds])
            for k in TOP_K_VALUES
        },
        "oracle_count": _aggregate_metric_rows([fold["oracle_count"] for fold in folds]),
    }


def _aggregate_fixed_rank(measures: Sequence[MeasureData], cap: int) -> dict[str, Any]:
    rows = []
    for measure in measures:
        ranked = sorted(measure.rows, key=lambda row: row.rank)
        rows.append(_selection_metrics(ranked[: min(cap, len(ranked))], measure))
    return _aggregate_metric_rows(rows)


def _aggregate_oracle_count(
    measures: Sequence[MeasureData], score_function: Callable[[CandidateRow], float]
) -> dict[str, Any]:
    rows = []
    for measure in measures:
        ranked = sorted(measure.rows, key=lambda row: (-score_function(row), row.rank))
        rows.append(_selection_metrics(ranked[: len(measure.ellipses)], measure))
    return _aggregate_metric_rows(rows)


def _aggregate_metric_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    return {
        "selected_count": sum(row["selected_count"] for row in rows),
        "gt_count": sum(row["gt_count"] for row in rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _f1(tp, fp, fn),
        "exact_note_count_fold_count": sum(row["exact_note_count"] for row in rows),
        "exact_set_fold_count": sum(row["exact_set"] for row in rows),
        "fold_count": len(rows),
    }


def _measure_summary(measure: MeasureData) -> dict[str, Any]:
    return {
        "measure": measure.measure,
        "source_image": str(measure.source_image),
        "source_image_sha256": _sha256(measure.source_image),
        "ground_truth_path": str(measure.ground_truth_path),
        "ground_truth_sha256": _sha256(measure.ground_truth_path),
        "staff_lines_y_px": list(measure.staff_lines),
        "candidate_count": len(measure.rows),
        "positive_candidate_count": sum(row.label for row in measure.rows),
        "ground_truth_notehead_count": len(measure.ellipses),
        "positive_candidates": [
            {
                "candidate_id": row.id,
                "rank": row.rank,
                "ellipse_id": row.matched_ellipse_id,
                "center": {"x": row.center_x, "y": row.center_y},
            }
            for row in measure.rows
            if row.label
        ],
    }


def write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path


def _markdown_report(report: dict[str, Any]) -> str:
    baseline4 = report["baselines"]["detector_cap4"]
    baseline24 = report["baselines"]["detector_cap24"]
    selection = report["selection"]
    lines = [
        "# Classical Notehead Candidate Ranking Spike",
        "",
        "Bounded development result on Aviador system 1 measures 1-4. Every outer fold trains "
        "on three measures and evaluates on the fourth. Pitch fields are never read.",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{selection['verdict']}`",
        f"- Best learned-threshold method: `{selection['best_threshold_method_id']}`",
        f"- Best top-4 ranker: `{selection['best_top4_ranker_method_id']}`",
        f"- Learned-threshold P/R/F1: "
        f"`{selection['best_threshold_metrics']['precision']:.3f}` / "
        f"`{selection['best_threshold_metrics']['recall']:.3f}` / "
        f"`{selection['best_threshold_metrics']['f1']:.3f}`",
        f"- Detector cap-4 P/R/F1: `{baseline4['precision']:.3f}` / "
        f"`{baseline4['recall']:.3f}` / `{baseline4['f1']:.3f}`",
        f"- Detector cap-24 recall/F1: `{baseline24['recall']:.3f}` / " f"`{baseline24['f1']:.3f}`",
        f"- Top-4 gain: `{selection['delta_top4_true_positives_vs_detector_cap4']}` "
        "additional matched notehead",
        f"- Conservative material-improvement rule passed: "
        f"`{selection['material_improvement_by_spike_rule']}`",
        "",
        "The threshold and resulting candidate count for each held-out measure are learned only "
        "from its three training measures. Oracle-count metrics below use held-out GT count and "
        "are diagnostics only.",
        "",
        "## Baselines",
        "",
        "| Strategy | Selected | TP/FP/FN | Precision | Recall | F1 | Exact count folds |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in report["baselines"].items():
        lines.append(
            f"| `{name}` | {metrics['selected_count']} | "
            f"{metrics['tp']}/{metrics['fp']}/{metrics['fn']} | "
            f"{metrics['precision']:.3f} | {metrics['recall']:.3f} | "
            f"{metrics['f1']:.3f} | {metrics['exact_note_count_fold_count']}/"
            f"{metrics['fold_count']} |"
        )
    lines.extend(
        [
            "",
            "## Leave-One-Measure-Out Methods",
            "",
            "| Method | Features | Threshold P/R/F1 | Exact count | Top-4 R/F1 | "
            "Oracle-count R/F1 |",
            "| --- | ---: | --- | ---: | --- | --- |",
        ]
    )
    for method in report["methods"]:
        threshold = method["aggregate"]["threshold_selection"]
        top4 = method["aggregate"]["top_k"]["4"]
        oracle = method["aggregate"]["oracle_count"]
        lines.append(
            f"| `{method['id']}` | {method['feature_count']} | "
            f"{threshold['precision']:.3f}/{threshold['recall']:.3f}/"
            f"{threshold['f1']:.3f} | {threshold['exact_note_count_fold_count']}/"
            f"{threshold['fold_count']} | {top4['recall']:.3f}/{top4['f1']:.3f} | "
            f"{oracle['recall']:.3f}/{oracle['f1']:.3f} |"
        )
    best = next(
        method
        for method in report["methods"]
        if method["id"] == selection["best_threshold_method_id"]
    )
    lines.extend(
        [
            "",
            "## Best Method by Fold",
            "",
            "| Held out | Learned threshold | Selected | TP/FP/FN | P/R/F1 | Exact count | "
            "Top-4 recall |",
            "| ---: | ---: | ---: | --- | --- | :---: | ---: |",
        ]
    )
    for fold in best["folds"]:
        metrics = fold["threshold_selection"]
        lines.append(
            f"| {fold['held_out_measure']} | {fold['learned_threshold']:.6f} | "
            f"{metrics['selected_count']} | {metrics['tp']}/{metrics['fp']}/{metrics['fn']} | "
            f"{metrics['precision']:.3f}/{metrics['recall']:.3f}/{metrics['f1']:.3f} | "
            f"{'yes' if metrics['exact_note_count'] else 'no'} | "
            f"{fold['top_k']['4']['recall']:.3f} |"
        )
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {caveat}" for caveat in report["caveats"])
    lines.append("")
    return "\n".join(lines)


def _column_means(matrix: Sequence[Sequence[float]]) -> list[float]:
    return [sum(row[index] for row in matrix) / len(matrix) for index in range(len(matrix[0]))]


def _column_variances(
    matrix: Sequence[Sequence[float]], means: Sequence[float], *, floor: float
) -> list[float]:
    return [
        sum((row[index] - means[index]) ** 2 for row in matrix) / max(1, len(matrix) - 1) + floor
        for index in range(len(means))
    ]


def _diagonal_gaussian_log_likelihood(
    row: Sequence[float], means: Sequence[float], variances: Sequence[float]
) -> float:
    return -0.5 * sum(
        math.log(variance) + (value - mean) ** 2 / variance
        for value, mean, variance in zip(row, means, variances, strict=True)
    )


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponential = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(max(value, -700.0))
    return exponential / (1.0 + exponential)


def _detector_scores(row: CandidateRow) -> float:
    return row.features["detector_score"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_unique(values: Iterable[int], label: str) -> list[int]:
    result = sorted(set(values))
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"{label} values must be positive")
    return result


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    return _ratio(2 * tp, 2 * tp + fp + fn)


if __name__ == "__main__":
    raise SystemExit(main())
