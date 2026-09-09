"""Evaluate raw patch/template notehead selectors with strict measure LOOCV.

This bounded spike regenerates cap-24 candidates for Aviador system 1 measures
1-4, extracts every image patch before loading human-review labels, and then
evaluates small shift-tolerant template and k-nearest-neighbor scorers. It never
reads coordinate ground truth, pitches, or canonical melody events.

Example:
    uv run python scripts/experiments/spike_notehead_patch_templates.py out
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_vlm_notehead_candidates as detector  # noqa: E402

DEFAULT_SLUG = "jaime-llanos_12_aviador_pasillo_fulgencio-garcia"
DEFAULT_MEASURES = (1, 2, 3, 4)
DEFAULT_MAX_CANDIDATES = 24
REVIEW_DIR = REPO_ROOT / "tests/fixtures/vlm_melody/notehead_reviews"
OUTPUT_SUBDIR = Path("experiments/notehead_patch_templates")
PATCH_WIDTH = 17
PATCH_HEIGHT = 13
TOP_K_VALUES = (3, 4, 5, 8)
REFERENCE_CAP4_F1 = 0.533
REFERENCE_LEARNED_TOP4_F1 = 0.600
MEANINGFUL_AUTO_F1_GAIN = 0.05

PatchKind = Literal["grayscale", "binary"]
ScorerKind = Literal["class_template", "class_knn3"]


@dataclass(frozen=True)
class PatchSpec:
    id: str
    kind: PatchKind
    suppress_staff_lines: bool


PATCH_SPECS = (
    PatchSpec("grayscale_raw", "grayscale", False),
    PatchSpec("grayscale_staff_suppressed", "grayscale", True),
    PatchSpec("binary_raw", "binary", False),
    PatchSpec("binary_staff_suppressed", "binary", True),
)
SCORER_KINDS: tuple[ScorerKind, ...] = ("class_template", "class_knn3")


@dataclass(frozen=True)
class CandidatePatch:
    measure: int
    id: str
    rank: int
    center_x: float
    center_y: float
    bbox: tuple[int, int, int, int]
    detector_score: float
    patches: dict[str, tuple[float, ...]]


@dataclass(frozen=True)
class UnlabeledMeasure:
    measure: int
    source_image: Path
    source_sha256: str
    staff_lines: tuple[int, ...]
    staff_spacing: float
    candidates: tuple[CandidatePatch, ...]


@dataclass(frozen=True)
class LabeledCandidate:
    candidate: CandidatePatch
    label: int

    @property
    def id(self) -> str:
        return self.candidate.id

    @property
    def rank(self) -> int:
        return self.candidate.rank


@dataclass(frozen=True)
class MeasureData:
    measure: int
    source_image: Path
    source_sha256: str
    review_path: Path
    review_sha256: str
    staff_lines: tuple[int, ...]
    staff_spacing: float
    rows: tuple[LabeledCandidate, ...]

    @property
    def positive_count(self) -> int:
        return sum(row.label for row in self.rows)


@dataclass(frozen=True)
class PatchScorer:
    patch_id: str
    scorer_kind: ScorerKind
    positive_vectors: tuple[tuple[float, ...], ...]
    negative_vectors: tuple[tuple[float, ...], ...]

    def score(self, candidate: CandidatePatch) -> float:
        vector = candidate.patches[self.patch_id]
        if self.scorer_kind == "class_template":
            positive_distance = _shift_tolerant_distance(
                vector, _mean_vector(self.positive_vectors), PATCH_WIDTH, PATCH_HEIGHT
            )
            negative_distance = _shift_tolerant_distance(
                vector, _mean_vector(self.negative_vectors), PATCH_WIDTH, PATCH_HEIGHT
            )
        else:
            positive_distance = _mean_nearest_distance(vector, self.positive_vectors, k=3)
            negative_distance = _mean_nearest_distance(vector, self.negative_vectors, k=3)
        return negative_distance - positive_distance


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        measures = _positive_unique(args.measure or DEFAULT_MEASURES, "measure")

        # This phase has no review/GT path and completes for every measure first.
        unlabeled = [
            _load_unlabeled_measure(
                args.out_dir,
                slug=args.slug,
                system_index=args.system,
                measure=measure,
                max_candidates=args.max_candidates,
            )
            for measure in measures
        ]

        # Human candidate decisions are loaded only after all inference inputs exist.
        labeled = [
            _attach_review_labels(measure, slug=args.slug, system_index=args.system)
            for measure in unlabeled
        ]
        report = evaluate_leave_one_measure_out(
            labeled,
            slug=args.slug,
            system_index=args.system,
            max_candidates=args.max_candidates,
        )
        json_path, markdown_path = write_report(report, labeled, args.out_dir / OUTPUT_SUBDIR)
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


def _load_unlabeled_measure(
    out_dir: Path,
    *,
    slug: str,
    system_index: int,
    measure: int,
    max_candidates: int,
) -> UnlabeledMeasure:
    if max_candidates != DEFAULT_MAX_CANDIDATES:
        raise ValueError("This bounded experiment requires the promoted cap-24 candidate set")
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
        grayscale = opened.convert("L")
    stages = detector._detect_staff_grid_stages(
        grayscale.convert("RGB"),
        staff_lines=list(staff_lines),
        max_candidates=max_candidates,
    )
    suppressed = _suppress_staff_lines(
        grayscale, staff_lines=staff_lines, staff_spacing=stages.staff_spacing
    )
    candidates = []
    for rank, candidate in enumerate(stages.candidates, start=1):
        patches = {}
        for spec in PATCH_SPECS:
            patch_source = suppressed if spec.suppress_staff_lines else grayscale
            patches[spec.id] = _extract_normalized_patch(
                patch_source,
                center=candidate.center,
                staff_spacing=stages.staff_spacing,
                kind=spec.kind,
                threshold=stages.threshold,
            )
        candidates.append(
            CandidatePatch(
                measure=measure,
                id=f"c{rank:03d}",
                rank=rank,
                center_x=float(candidate.center[0]),
                center_y=float(candidate.center[1]),
                bbox=(
                    int(candidate.bbox[0]),
                    int(candidate.bbox[1]),
                    int(candidate.bbox[2]) + 1,
                    int(candidate.bbox[3]) + 1,
                ),
                detector_score=float(candidate.score),
                patches=patches,
            )
        )
    if len(candidates) != max_candidates:
        raise ValueError(
            f"Measure {measure} generated {len(candidates)} candidates; expected {max_candidates}"
        )
    return UnlabeledMeasure(
        measure=measure,
        source_image=source_path,
        source_sha256=_sha256(source_path),
        staff_lines=staff_lines,
        staff_spacing=float(stages.staff_spacing),
        candidates=tuple(candidates),
    )


def _suppress_staff_lines(
    image: Image.Image, *, staff_lines: Sequence[int], staff_spacing: float
) -> Image.Image:
    suppressed = image.copy()
    pixels = suppressed.load()
    width, height = suppressed.size
    radius = max(1, round(staff_spacing * 0.045))
    for line_y in staff_lines:
        for y in range(max(0, line_y - radius), min(height, line_y + radius + 1)):
            for x in range(width):
                pixels[x, y] = 255
    return suppressed


def _extract_normalized_patch(
    image: Image.Image,
    *,
    center: tuple[float, float],
    staff_spacing: float,
    kind: PatchKind,
    threshold: int,
) -> tuple[float, ...]:
    """Extract a fixed patch whose physical extent is defined in staff spaces."""
    if staff_spacing <= 0:
        raise ValueError("staff_spacing must be positive")
    if kind not in ("grayscale", "binary"):
        raise ValueError(f"Unsupported patch kind: {kind}")
    crop_width = max(3, round(staff_spacing * 1.10))
    crop_height = max(3, round(staff_spacing * 0.85))
    center_x, center_y = center
    left = round(center_x - (crop_width - 1) / 2)
    top = round(center_y - (crop_height - 1) / 2)
    patch = image.crop((left, top, left + crop_width, top + crop_height))
    resampling = Image.Resampling.BILINEAR if kind == "grayscale" else Image.Resampling.NEAREST
    normalized = patch.resize((PATCH_WIDTH, PATCH_HEIGHT), resample=resampling)
    values = tuple(float(value) for value in normalized.get_flattened_data())
    if kind == "binary":
        return tuple(1.0 if value <= threshold else 0.0 for value in values)
    return tuple((255.0 - value) / 255.0 for value in values)


def _attach_review_labels(
    measure: UnlabeledMeasure, *, slug: str, system_index: int
) -> MeasureData:
    review_path = REVIEW_DIR / (
        f"{slug}_system_{system_index:03d}_measure_{measure.measure:03d}.json"
    )
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    identity = payload.get("identity", {})
    if (
        int(identity.get("system_index", -1)) != system_index
        or int(identity.get("system_measure_index", -1)) != measure.measure
    ):
        raise ValueError(f"Review identity does not match requested fold: {review_path}")
    source = payload.get("source", {})
    if source.get("image_sha256") != measure.source_sha256:
        raise ValueError(f"Review image hash does not match regenerated input: {review_path}")
    if int(source.get("candidate_cap", -1)) != len(measure.candidates):
        raise ValueError(
            f"Review candidate cap does not match regenerated candidates: {review_path}"
        )
    if payload.get("manual_noteheads"):
        raise ValueError(
            f"Manual noteheads are outside this candidate-selector spike: {review_path}"
        )

    decisions = payload.get("candidates")
    if not isinstance(decisions, list):
        raise ValueError(f"Review fixture has no candidates list: {review_path}")
    by_id = {str(item["id"]): item for item in decisions}
    if len(by_id) != len(decisions):
        raise ValueError(f"Review fixture contains duplicate candidate IDs: {review_path}")
    expected_ids = {candidate.id for candidate in measure.candidates}
    if set(by_id) != expected_ids:
        raise ValueError(f"Review candidate IDs do not match regenerated cap-24 set: {review_path}")

    rows = []
    for candidate in measure.candidates:
        decision = by_id[candidate.id]
        label = decision.get("label")
        if label not in ("accepted", "rejected"):
            raise ValueError(f"Candidate {candidate.id} has invalid review label: {label!r}")
        # Candidate geometry is provenance, not GT. It is checked only for stale-fixture safety.
        if _review_bbox(decision.get("bbox")) != candidate.bbox:
            raise ValueError(f"Candidate {candidate.id} bbox differs from promoted review fixture")
        rows.append(LabeledCandidate(candidate=candidate, label=int(label == "accepted")))
    return MeasureData(
        measure=measure.measure,
        source_image=measure.source_image,
        source_sha256=measure.source_sha256,
        review_path=review_path,
        review_sha256=_sha256(review_path),
        staff_lines=measure.staff_lines,
        staff_spacing=measure.staff_spacing,
        rows=tuple(rows),
    )


def _review_bbox(value: Any) -> tuple[int, int, int, int]:
    if not isinstance(value, dict):
        raise ValueError("Candidate review bbox must be an object")
    return tuple(int(value[name]) for name in ("left", "top", "right", "bottom"))


def _fit_patch_scorer(
    rows: Sequence[LabeledCandidate], *, patch_id: str, scorer_kind: ScorerKind
) -> PatchScorer:
    positives = tuple(row.candidate.patches[patch_id] for row in rows if row.label)
    negatives = tuple(row.candidate.patches[patch_id] for row in rows if not row.label)
    if not positives or not negatives:
        raise ValueError("Patch scorer requires positive and negative training candidates")
    if scorer_kind not in SCORER_KINDS:
        raise ValueError(f"Unsupported scorer kind: {scorer_kind}")
    return PatchScorer(patch_id, scorer_kind, positives, negatives)


def _mean_vector(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    return tuple(
        sum(vector[index] for vector in vectors) / len(vectors) for index in range(len(vectors[0]))
    )


def _mean_nearest_distance(
    vector: Sequence[float], examples: Sequence[Sequence[float]], *, k: int
) -> float:
    distances = sorted(
        _shift_tolerant_distance(vector, example, PATCH_WIDTH, PATCH_HEIGHT) for example in examples
    )
    neighbors = distances[: min(k, len(distances))]
    return sum(neighbors) / len(neighbors)


def _shift_tolerant_distance(
    first: Sequence[float],
    second: Sequence[float],
    width: int,
    height: int,
    *,
    max_shift: int = 1,
) -> float:
    if len(first) != width * height or len(second) != width * height:
        raise ValueError("Patch dimensions do not match vector lengths")
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative")
    best = float("inf")
    for shift_y in range(-max_shift, max_shift + 1):
        for shift_x in range(-max_shift, max_shift + 1):
            squared_error = 0.0
            for y in range(height):
                source_y = y - shift_y
                for x in range(width):
                    source_x = x - shift_x
                    shifted = (
                        float(first[source_y * width + source_x])
                        if 0 <= source_x < width and 0 <= source_y < height
                        else 0.0
                    )
                    difference = shifted - float(second[y * width + x])
                    squared_error += difference * difference
            best = min(best, squared_error / (width * height))
    return best


def evaluate_leave_one_measure_out(
    measures: Sequence[MeasureData], *, slug: str, system_index: int, max_candidates: int
) -> dict[str, Any]:
    if tuple(measure.measure for measure in measures) != DEFAULT_MEASURES:
        raise ValueError("This bounded experiment requires S1 M1-4 in order")
    methods = [
        _evaluate_method(measures, patch_spec=patch_spec, scorer_kind=scorer_kind)
        for patch_spec in PATCH_SPECS
        for scorer_kind in SCORER_KINDS
    ]
    methods.sort(key=lambda method: method["id"])
    baselines = {
        "detector_cap4": _aggregate_metric_rows(
            [_selection_metrics(measure.rows[:4], measure) for measure in measures]
        ),
        "detector_cap24": _aggregate_metric_rows(
            [_selection_metrics(measure.rows, measure) for measure in measures]
        ),
    }
    best_threshold_method = max(
        methods,
        key=lambda method: _method_metric_key(method, "threshold_selection"),
    )
    best_top4_method = max(
        methods,
        key=lambda method: _method_metric_key(method, "top_k", top_k="4"),
    )
    automatic_options = [
        (method, mode, method["aggregate"][mode])
        for method in methods
        for mode in ("threshold_selection", "learned_count_selection")
    ]
    best_auto_method, best_auto_mode, best_auto_metrics = max(
        automatic_options,
        key=lambda item: (
            item[2]["f1"],
            item[2]["recall"],
            item[2]["precision"],
            -item[2]["selected_count"],
            item[0]["id"],
            item[1],
        ),
    )
    best_top4 = best_top4_method["aggregate"]["top_k"]["4"]
    cap4 = baselines["detector_cap4"]
    top4_material = best_top4["f1"] > REFERENCE_LEARNED_TOP4_F1
    automatic_material = (
        best_auto_metrics["f1"] >= REFERENCE_LEARNED_TOP4_F1 + MEANINGFUL_AUTO_F1_GAIN
        and best_auto_metrics["recall"] >= cap4["recall"]
    )
    material = top4_material or automatic_material
    return {
        "schema_version": 1,
        "kind": "notehead_patch_template_selector_spike",
        "scope": {
            "slug": slug,
            "system_index": system_index,
            "measures": [measure.measure for measure in measures],
            "candidate_cap": max_candidates,
            "outer_evaluation": "leave-one-measure-out",
            "feature_extraction_completed_before_review_labels_loaded": True,
            "candidate_generation_uses_review_or_ground_truth": False,
            "inference_uses_review_labels": False,
            "review_fixture_fields_used_for_labels": ["candidates[].id", "candidates[].label"],
            "coordinate_ground_truth_read": False,
            "canonical_event_ground_truth_read": False,
            "pitch_fields_read": False,
        },
        "patch_family": {
            "canonical_size": {"width": PATCH_WIDTH, "height": PATCH_HEIGHT},
            "physical_extent_staff_spaces": {"width": 1.10, "height": 0.85},
            "shift_tolerance_canonical_pixels": 1,
            "representations": [
                {
                    "id": spec.id,
                    "kind": spec.kind,
                    "staff_line_suppression": spec.suppress_staff_lines,
                }
                for spec in PATCH_SPECS
            ],
            "scorers": list(SCORER_KINDS),
        },
        "dataset": [_measure_summary(measure) for measure in measures],
        "references": {
            "existing_cap4_f1": REFERENCE_CAP4_F1,
            "existing_learned_top4_f1": REFERENCE_LEARNED_TOP4_F1,
            "measured_cap4_f1": baselines["detector_cap4"]["f1"],
        },
        "baselines": baselines,
        "methods": methods,
        "selection": {
            "verdict": "material_win" if material else "no_material_win",
            "best_candidate_threshold_method_id": best_threshold_method["id"],
            "best_candidate_threshold_metrics": best_threshold_method["aggregate"][
                "threshold_selection"
            ],
            "best_top4_method_id": best_top4_method["id"],
            "best_top4_metrics": best_top4,
            "best_automatic_method_id": best_auto_method["id"],
            "best_automatic_mode": best_auto_mode,
            "best_automatic_metrics": best_auto_metrics,
            "delta_top4_f1_vs_existing_cap4": round(best_top4["f1"] - REFERENCE_CAP4_F1, 6),
            "delta_top4_f1_vs_existing_learned_top4": round(
                best_top4["f1"] - REFERENCE_LEARNED_TOP4_F1, 6
            ),
            "material_win_rule": {
                "top4": "LOOCV top-4 F1 must be strictly greater than 0.600",
                "automatic_count": (
                    "F1 must be at least 0.650 and recall must not be below measured cap-4 recall"
                ),
            },
            "top4_material_win": top4_material,
            "automatic_count_material_win": automatic_material,
            "material_win": material,
        },
        "artifacts": {"overlays": []},
        "caveats": [
            "Only four development measures and fourteen accepted candidate regions are available.",
            (
                "Outer folds are label-isolated, but choosing the best method on the four outer "
                "folds creates model-selection bias."
            ),
            (
                "Human reviews label only the cap-24 candidate set; this experiment cannot recover "
                "a notehead absent from candidate generation."
            ),
            "A separate score/system is required before any production claim.",
        ],
    }


def _evaluate_method(
    measures: Sequence[MeasureData], *, patch_spec: PatchSpec, scorer_kind: ScorerKind
) -> dict[str, Any]:
    folds = []
    for held_out in measures:
        training = [measure for measure in measures if measure.measure != held_out.measure]
        training_rows = [row for measure in training for row in measure.rows]

        # Threshold calibration uses inner leave-one-measure-out scores, never fit scores.
        calibration_scores: dict[tuple[int, str], float] = {}
        for calibration_measure in training:
            inner_training_rows = [
                row
                for measure in training
                if measure.measure != calibration_measure.measure
                for row in measure.rows
            ]
            inner_scorer = _fit_patch_scorer(
                inner_training_rows, patch_id=patch_spec.id, scorer_kind=scorer_kind
            )
            for row in calibration_measure.rows:
                calibration_scores[(calibration_measure.measure, row.id)] = inner_scorer.score(
                    row.candidate
                )
        threshold, training_threshold_metrics = _select_training_threshold(
            training, calibration_scores
        )

        scorer = _fit_patch_scorer(training_rows, patch_id=patch_spec.id, scorer_kind=scorer_kind)
        held_out_scores = {row.id: scorer.score(row.candidate) for row in held_out.rows}
        ranked = sorted(
            held_out.rows,
            key=lambda row: (-held_out_scores[row.id], row.rank, row.id),
        )
        threshold_selected = [row for row in ranked if held_out_scores[row.id] >= threshold]
        learned_count = _learned_training_count(training)
        count_selected = ranked[: min(learned_count, len(ranked))]
        folds.append(
            {
                "held_out_measure": held_out.measure,
                "training_measures": [measure.measure for measure in training],
                "training_candidate_count": len(training_rows),
                "training_positive_count": sum(row.label for row in training_rows),
                "learned_threshold": round(threshold, 9),
                "learned_count": learned_count,
                "training_threshold_metrics": training_threshold_metrics,
                "threshold_selection": _selection_metrics(threshold_selected, held_out),
                "learned_count_selection": _selection_metrics(count_selected, held_out),
                "top_k": {
                    str(k): _selection_metrics(ranked[: min(k, len(ranked))], held_out)
                    for k in TOP_K_VALUES
                },
                "ranked_candidates": [
                    {
                        "candidate_id": row.id,
                        "rank": rank,
                        "detector_rank": row.rank,
                        "score": round(held_out_scores[row.id], 9),
                        "review_label": "accepted" if row.label else "rejected",
                        "selected_by_threshold": held_out_scores[row.id] >= threshold,
                        "selected_by_learned_count": rank <= learned_count,
                    }
                    for rank, row in enumerate(ranked, start=1)
                ],
            }
        )
    return {
        "id": f"{scorer_kind}__{patch_spec.id}",
        "scorer_kind": scorer_kind,
        "patch_id": patch_spec.id,
        "folds": folds,
        "aggregate": _aggregate_folds(folds),
    }


def _select_training_threshold(
    training: Sequence[MeasureData], scores: dict[tuple[int, str], float]
) -> tuple[float, dict[str, Any]]:
    score_values = sorted(
        {scores[(measure.measure, row.id)] for measure in training for row in measure.rows}
    )
    thresholds = [score_values[-1] + 1e-9, *reversed(score_values)]
    best_threshold = thresholds[0]
    best_metrics: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    for threshold in thresholds:
        metrics = _aggregate_metric_rows(
            [
                _selection_metrics(
                    [row for row in measure.rows if scores[(measure.measure, row.id)] >= threshold],
                    measure,
                )
                for measure in training
            ]
        )
        key = (
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
            -metrics["selected_count"],
            threshold,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
            best_metrics = metrics
    assert best_metrics is not None
    return best_threshold, best_metrics


def _learned_training_count(training: Sequence[MeasureData]) -> int:
    return int(statistics.median(measure.positive_count for measure in training))


def _selection_metrics(
    selected: Sequence[LabeledCandidate], measure: MeasureData
) -> dict[str, Any]:
    tp = sum(row.label for row in selected)
    fp = len(selected) - tp
    fn = measure.positive_count - tp
    return {
        "selected_count": len(selected),
        "positive_region_count": measure.positive_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _f1(tp, fp, fn),
        "exact_count": len(selected) == measure.positive_count,
        "exact_set": fp == 0 and fn == 0,
        "selected_candidate_ids": [row.id for row in selected],
    }


def _aggregate_folds(folds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "threshold_selection": _aggregate_metric_rows(
            [fold["threshold_selection"] for fold in folds]
        ),
        "learned_count_selection": _aggregate_metric_rows(
            [fold["learned_count_selection"] for fold in folds]
        ),
        "top_k": {
            str(k): _aggregate_metric_rows([fold["top_k"][str(k)] for fold in folds])
            for k in TOP_K_VALUES
        },
    }


def _aggregate_metric_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    return {
        "selected_count": sum(row["selected_count"] for row in rows),
        "positive_region_count": sum(row["positive_region_count"] for row in rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _f1(tp, fp, fn),
        "exact_count_fold_count": sum(row["exact_count"] for row in rows),
        "exact_set_fold_count": sum(row["exact_set"] for row in rows),
        "fold_count": len(rows),
    }


def _method_metric_key(
    method: dict[str, Any], metric: str, *, top_k: str | None = None
) -> tuple[Any, ...]:
    metrics = method["aggregate"][metric]
    if top_k is not None:
        metrics = metrics[top_k]
    return (
        metrics["f1"],
        metrics["recall"],
        metrics["precision"],
        -metrics["selected_count"],
        method["id"],
    )


def _measure_summary(measure: MeasureData) -> dict[str, Any]:
    return {
        "measure": measure.measure,
        "source_image": str(measure.source_image),
        "source_image_sha256": measure.source_sha256,
        "review_path": str(measure.review_path),
        "review_sha256": measure.review_sha256,
        "staff_lines_y_px": list(measure.staff_lines),
        "staff_spacing_px": round(measure.staff_spacing, 6),
        "candidate_count": len(measure.rows),
        "accepted_candidate_count": measure.positive_count,
        "accepted_candidate_ids": [row.id for row in measure.rows if row.label],
    }


def write_report(
    report: dict[str, Any], measures: Sequence[MeasureData], output_dir: Path
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report["artifacts"]["overlays"] = _write_selected_overlays(report, measures, output_dir)
    json_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path


def _write_selected_overlays(
    report: dict[str, Any], measures: Sequence[MeasureData], output_dir: Path
) -> list[dict[str, Any]]:
    method_by_id = {method["id"]: method for method in report["methods"]}
    selections = [
        (
            "best_top4",
            method_by_id[report["selection"]["best_top4_method_id"]],
            "top_k",
        ),
        (
            "best_automatic",
            method_by_id[report["selection"]["best_automatic_method_id"]],
            report["selection"]["best_automatic_mode"],
        ),
    ]
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    measures_by_id = {measure.measure: measure for measure in measures}
    for label, method, mode in selections:
        for fold in method["folds"]:
            measure = measures_by_id[fold["held_out_measure"]]
            if mode == "top_k":
                selected_ids = set(fold["top_k"]["4"]["selected_candidate_ids"])
            else:
                selected_ids = set(fold[mode]["selected_candidate_ids"])
            path = overlay_dir / f"{label}_measure_{measure.measure:03d}.png"
            _write_overlay(
                measure,
                path,
                selected_ids=selected_ids,
                title=f"{label}: {method['id']}",
            )
            artifacts.append(
                {
                    "kind": label,
                    "measure": measure.measure,
                    "method_id": method["id"],
                    "selection_mode": mode,
                    "path": str(path),
                }
            )
    return artifacts


def _write_overlay(measure: MeasureData, path: Path, *, selected_ids: set[str], title: str) -> None:
    with Image.open(measure.source_image) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, image.width, 17), fill=(255, 255, 255))
    draw.text((3, 3), title, fill=(0, 0, 0), font=font)
    for row in measure.rows:
        selected = row.id in selected_ids
        if selected and row.label:
            color = (0, 150, 45)
            width = 3
        elif selected:
            color = (210, 25, 25)
            width = 3
        elif row.label:
            color = (230, 140, 0)
            width = 2
        else:
            color = (110, 110, 110)
            width = 1
        x = row.candidate.center_x
        y = row.candidate.center_y
        radius = max(5, round(measure.staff_spacing * 0.32))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=width)
        draw.text((x + radius + 1, y - radius), row.id[1:], fill=color, font=font)
    image.save(path)


def _markdown_report(report: dict[str, Any]) -> str:
    selection = report["selection"]
    cap4 = report["baselines"]["detector_cap4"]
    cap24 = report["baselines"]["detector_cap24"]
    best_top4 = selection["best_top4_metrics"]
    best_auto = selection["best_automatic_metrics"]
    lines = [
        "# Notehead Patch Template Selector Spike",
        "",
        "Bounded result on Aviador system 1 measures 1-4. Cap-24 candidate generation and all "
        "raw image-patch extraction completed before promoted human-review labels were loaded. "
        "Every reported prediction is from a leave-one-measure-out fold.",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{selection['verdict']}`",
        f"- Best top-4 method: `{selection['best_top4_method_id']}`",
        f"- Best top-4 P/R/F1: `{best_top4['precision']:.3f}` / "
        f"`{best_top4['recall']:.3f}` / `{best_top4['f1']:.3f}`",
        f"- Best automatic selection: `{selection['best_automatic_method_id']}` / "
        f"`{selection['best_automatic_mode']}`",
        f"- Best automatic P/R/F1: `{best_auto['precision']:.3f}` / "
        f"`{best_auto['recall']:.3f}` / `{best_auto['f1']:.3f}`",
        f"- Exact automatic counts: `{best_auto['exact_count_fold_count']}/"
        f"{best_auto['fold_count']}` folds",
        f"- Existing comparisons: cap-4 F1 `{REFERENCE_CAP4_F1:.3f}`, learned top-4 F1 "
        f"`{REFERENCE_LEARNED_TOP4_F1:.3f}`",
        f"- Material-win rule passed: `{selection['material_win']}`",
        "",
        "A top-4 material win requires F1 strictly above 0.600. An automatic-count material "
        "win requires F1 at least 0.650 with recall no lower than the measured cap-4 recall.",
        "",
        "## Baselines",
        "",
        "| Strategy | Selected | TP/FP/FN | Precision | Recall | F1 |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
        f"| Detector cap-4 | {cap4['selected_count']} | {cap4['tp']}/{cap4['fp']}/"
        f"{cap4['fn']} | {cap4['precision']:.3f} | {cap4['recall']:.3f} | "
        f"{cap4['f1']:.3f} |",
        f"| Detector cap-24 | {cap24['selected_count']} | {cap24['tp']}/{cap24['fp']}/"
        f"{cap24['fn']} | {cap24['precision']:.3f} | {cap24['recall']:.3f} | "
        f"{cap24['f1']:.3f} |",
        "",
        "## LOOCV Methods",
        "",
        "Candidate metrics use the inner-LOOCV-trained threshold. Top-k metrics evaluate "
        "accepted candidate regions.",
        "",
        "| Method | Candidate P/R/F1 | Auto-count P/R/F1 | Exact count | Top-3 R/F1 | "
        "Top-4 R/F1 | Top-5 R/F1 | Top-8 R/F1 |",
        "| --- | --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for method in report["methods"]:
        threshold = method["aggregate"]["threshold_selection"]
        automatic = method["aggregate"]["learned_count_selection"]
        top_k = method["aggregate"]["top_k"]
        lines.append(
            f"| `{method['id']}` | {threshold['precision']:.3f}/"
            f"{threshold['recall']:.3f}/{threshold['f1']:.3f} | "
            f"{automatic['precision']:.3f}/{automatic['recall']:.3f}/"
            f"{automatic['f1']:.3f} | {automatic['exact_count_fold_count']}/4 | "
            f"{top_k['3']['recall']:.3f}/{top_k['3']['f1']:.3f} | "
            f"{top_k['4']['recall']:.3f}/{top_k['4']['f1']:.3f} | "
            f"{top_k['5']['recall']:.3f}/{top_k['5']['f1']:.3f} | "
            f"{top_k['8']['recall']:.3f}/{top_k['8']['f1']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Best Top-4 by Measure",
            "",
            "| Held out | Training | Top-4 TP/FP/FN | Precision | Recall | F1 |",
            "| ---: | --- | --- | ---: | ---: | ---: |",
        ]
    )
    best_method = next(
        method for method in report["methods"] if method["id"] == selection["best_top4_method_id"]
    )
    for fold in best_method["folds"]:
        metrics = fold["top_k"]["4"]
        lines.append(
            f"| {fold['held_out_measure']} | {','.join(map(str, fold['training_measures']))} | "
            f"{metrics['tp']}/{metrics['fp']}/{metrics['fn']} | "
            f"{metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Best Automatic Selection by Measure",
            "",
            "| Held out | Learned threshold | Learned count | Selected | TP/FP/FN | P/R/F1 | "
            "Exact count |",
            "| ---: | ---: | ---: | ---: | --- | --- | :---: |",
        ]
    )
    auto_method = next(
        method
        for method in report["methods"]
        if method["id"] == selection["best_automatic_method_id"]
    )
    auto_mode = selection["best_automatic_mode"]
    for fold in auto_method["folds"]:
        metrics = fold[auto_mode]
        lines.append(
            f"| {fold['held_out_measure']} | {fold['learned_threshold']:.6f} | "
            f"{fold['learned_count']} | {metrics['selected_count']} | "
            f"{metrics['tp']}/{metrics['fp']}/{metrics['fn']} | "
            f"{metrics['precision']:.3f}/{metrics['recall']:.3f}/{metrics['f1']:.3f} | "
            f"{'yes' if metrics['exact_count'] else 'no'} |"
        )
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {caveat}" for caveat in report["caveats"])
    lines.append("")
    return "\n".join(lines)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    return _ratio(2 * tp, 2 * tp + fp + fn)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_unique(values: Sequence[int], name: str) -> tuple[int, ...]:
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} values must be positive")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} values must be unique")
    return tuple(values)


if __name__ == "__main__":
    raise SystemExit(main())
