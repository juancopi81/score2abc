"""Evaluate a sealed hybrid grid/stem notehead selector.

The bounded experiment builds cap-24 grid proposals, stem-endpoint proposals,
their deterministic union, and all image features before opening development
coordinate/pitch labels. Method selection uses strict S1M1-4 leave-one-measure-
out predictions. Validation systems 7+8 and, conditionally, heldout system 3
are sealed before their truth files are opened.

Example:
    uv run python scripts/experiments/spike_hybrid_notehead_selector.py out
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import spike_notehead_patch_templates as patch_spike  # noqa: E402
from scripts.experiments import spike_stem_endpoint_detector as stem_spike  # noqa: E402

DEFAULT_SLUG = stem_spike.DEFAULT_SLUG
DEVELOPMENT_TARGETS = stem_spike.DEVELOPMENT_TARGETS
VALIDATION_SYSTEMS = stem_spike.VALIDATION_SYSTEMS
HELDOUT_SYSTEM = stem_spike.HELDOUT_SYSTEM
COORDINATE_DIR = REPO_ROOT / "tests/fixtures/vlm_melody/notehead_ground_truth"
REVIEW_DIR = REPO_ROOT / "tests/fixtures/vlm_melody/notehead_reviews"
OUTPUT_SUBDIR = "hybrid_notehead_selector"
MAX_CANDIDATES = 24
ELLIPSE_MARGIN = 1.15
UNION_X_TOLERANCE_SPACES = 0.62
UNION_Y_TOLERANCE_SPACES = 0.50
PATCH_ID = "binary_raw"
METHOD_IDS = ("grid_only", "stem_only", "intersection_agreement", "hybrid_logistic")
HYBRID_FEATURES = (
    "patch_knn_score",
    "has_grid",
    "has_stem",
    "agreement",
    "grid_score",
    "grid_rank_quality",
    "stem_score",
    "agreement_distance",
)

# Preregistered development advancement rule. The historical automatic-count
# result was 11 TP, 1 FP, 3 FN with three exact-count folds.
REFERENCE_AUTOMATIC_F1 = 22 / 26
REFERENCE_AUTOMATIC_RECALL = 11 / 14
REFERENCE_EXACT_COUNT_FOLDS = 3

# Preregistered before validation truth is opened. "Materially above" means an
# absolute ordered-pitch gain of at least 0.15 over the full-system VLM result.
FULL_SYSTEM_VLM_ORDERED_PITCH = 0.130435
VALIDATION_GATE = {
    "minimum_ordered_natural_pitch_accuracy": FULL_SYSTEM_VLM_ORDERED_PITCH + 0.15,
    "minimum_pitch_only_note_f1": 0.60,
    "minimum_exact_note_count_rate": 0.20,
    "minimum_predicted_to_truth_count_ratio": 0.75,
    "maximum_predicted_to_truth_count_ratio": 1.25,
}

PredictionTruthGate = stem_spike.SealedTruthGate


@dataclass(frozen=True)
class CoordinateEllipse:
    id: str
    order: int
    center_x: float
    center_y: float
    radius_x: float
    radius_y: float
    natural_pitch_midi: int


@dataclass(frozen=True)
class HybridCandidate:
    id: str
    center_x: float
    center_y: float
    bbox: tuple[int, int, int, int]
    patch: tuple[float, ...]
    grid_ids: tuple[str, ...] = ()
    stem_ids: tuple[str, ...] = ()
    grid_rank: int | None = None
    grid_score: float = 0.0
    stem_score: float = 0.0
    agreement_distance: float = 1.0

    @property
    def has_grid(self) -> bool:
        return bool(self.grid_ids)

    @property
    def has_stem(self) -> bool:
        return bool(self.stem_ids)

    @property
    def agreement(self) -> bool:
        return self.has_grid and self.has_stem


@dataclass(frozen=True)
class PreparedTarget:
    request: dict[str, Any]
    prepared: stem_spike.PreparedRequest
    grid: patch_spike.UnlabeledMeasure
    variants: dict[str, tuple[HybridCandidate, ...]]

    @property
    def measure(self) -> int:
        return int(self.request["identity"]["system_measure_index"])


@dataclass(frozen=True)
class DevelopmentTruth:
    coordinate_path: Path
    coordinate_sha256: str
    pitch_review_path: Path
    pitch_review_sha256: str
    ellipses: tuple[CoordinateEllipse, ...]


@dataclass(frozen=True)
class LabeledCandidate:
    candidate: HybridCandidate
    matched_ellipse_id: str | None

    @property
    def label(self) -> int:
        return int(self.matched_ellipse_id is not None)


@dataclass(frozen=True)
class LabeledVariant:
    target: PreparedTarget
    truth: DevelopmentTruth
    config_key: str
    rows: tuple[LabeledCandidate, ...]


@dataclass(frozen=True)
class LogisticModel:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    weights: tuple[float, ...]
    intercept: float

    def score(self, features: Mapping[str, float]) -> float:
        values = [
            (float(features[name]) - mean) / scale
            for name, mean, scale in zip(self.feature_names, self.means, self.scales, strict=True)
        ]
        value = self.intercept + sum(
            weight * feature for weight, feature in zip(self.weights, values, strict=True)
        )
        return _sigmoid(value)


@dataclass(frozen=True)
class PitchCalibrator:
    unanimous_position_map: dict[int, int]

    def predict(self, candidate: HybridCandidate, prepared: stem_spike.PreparedRequest) -> int:
        position = _staff_position(candidate.center_y, prepared)
        return self.unanimous_position_map.get(
            position, stem_spike.natural_midi_for_staff_position(position)
        )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    benchmark_dir = args.out_dir / args.slug / "vlm_melody_event_benchmark"
    output_dir = benchmark_dir / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    leakage_events: list[dict[str, Any]] = []

    try:
        configs = stem_spike.parameter_grid()
        development_requests_path = benchmark_dir / "development/requests.jsonl"
        development_requests = stem_spike._selected_requests(
            stem_spike._read_jsonl(development_requests_path), DEVELOPMENT_TARGETS
        )
        prepared_development = [
            prepare_target_all_configs(
                request,
                out_dir=args.out_dir,
                slug=args.slug,
                configs=configs,
            )
            for request in development_requests
        ]
        leakage_events.append(
            {
                "phase": 1,
                "event": "development_candidates_and_features_complete",
                "before_label_access": True,
                "request_path": str(development_requests_path),
                "request_sha256": _sha256(development_requests_path),
                "stem_config_count": len(configs),
                "targets": len(prepared_development),
                "source_images": [
                    {
                        "path": str(target.prepared.image_path),
                        "sha256": _sha256(target.prepared.image_path),
                    }
                    for target in prepared_development
                ],
            }
        )

        development_truth = load_development_truth(args.slug, prepared_development)
        leakage_events.append(
            {
                "phase": 2,
                "event": "development_coordinate_and_promoted_pitch_labels_opened",
                "after_event": "development_candidates_and_features_complete",
                "coordinate_hashes": [truth.coordinate_sha256 for truth in development_truth],
                "pitch_review_hashes": [truth.pitch_review_sha256 for truth in development_truth],
            }
        )
        labeled = attach_all_labels(prepared_development, development_truth)
        development = evaluate_development(labeled, configs)
        winner_id = str(development["selection"]["winner_method_id"])
        development_predictions = development["selection"].pop("winner_predictions")
        _write_jsonl(output_dir / "development_predictions.jsonl", development_predictions)
        _write_overlays(
            prepared_development,
            development_predictions,
            output_dir / "development_overlays",
        )

        fitted = fit_winner(winner_id, labeled, configs)
        pitch_calibrator = fit_pitch_calibrator(prepared_development, development_truth)
        leakage_events.append(
            {
                "phase": 3,
                "event": "winner_fit_on_all_development_labels",
                "winner_method_id": winner_id,
                "selected_stem_config_key": fitted["config"].key,
                "validation_truth_accessed": False,
            }
        )

        truth_gate = PredictionTruthGate()
        validation_block = run_sealed_split(
            "validation",
            benchmark_dir / "validation/requests.jsonl",
            benchmark_dir / "validation/truth.jsonl",
            output_dir,
            out_dir=args.out_dir,
            expected_systems=VALIDATION_SYSTEMS,
            fitted=fitted,
            pitch_calibrator=pitch_calibrator,
            truth_gate=truth_gate,
            leakage_events=leakage_events,
        )
        validation_gate = apply_validation_gate(validation_block["metrics"])
        validation_block["gate"] = validation_gate
        validation_block["development_gate_passed"] = development["selection"]["development_gate"][
            "passed"
        ]
        truth_gate.record_validation_result(validation_gate)
        heldout_block: dict[str, Any]
        if validation_gate["passed"]:
            heldout_block = run_sealed_split(
                "heldout",
                benchmark_dir / "heldout/requests.jsonl",
                benchmark_dir / "heldout/truth.jsonl",
                output_dir,
                out_dir=args.out_dir,
                expected_systems=(HELDOUT_SYSTEM,),
                fitted=fitted,
                pitch_calibrator=pitch_calibrator,
                truth_gate=truth_gate,
                leakage_events=leakage_events,
            )
            heldout_block["status"] = "evaluated_once"
        else:
            heldout_block = {
                "status": "skipped_validation_gate_failed",
                "prediction_seal": None,
                "metrics": None,
            }

        leakage_log_path = output_dir / "leakage_log.json"
        _write_json(
            leakage_log_path,
            {
                "schema_version": 1,
                "events": leakage_events,
                "truth_access_log": truth_gate.access_log,
            },
        )
        report = {
            "schema_version": 1,
            "kind": "hybrid_notehead_proposal_selector_experiment",
            "scope": {
                "development": "system 1 measures 1-4",
                "validation": "systems 7 and 8",
                "heldout": "system 3, only after validation gate",
                "candidate_label": "center overlap with independent coordinate ellipse",
                "pitch_labels": "promoted review pitches, naturalized for evaluation",
                "rhythm": "out_of_scope",
                "durations": "out_of_scope",
                "network": "not_used",
                "production_integration": False,
            },
            "development": development,
            "fitted_winner": fitted_report(fitted, pitch_calibrator),
            "validation": validation_block,
            "heldout": heldout_block,
            "leakage": {
                "log_path": str(leakage_log_path),
                "log_sha256": _sha256(leakage_log_path),
                "events": leakage_events,
                "truth_access_log": truth_gate.access_log,
            },
            "artifacts": {
                "development_predictions": str(output_dir / "development_predictions.jsonl"),
                "leakage_log": str(leakage_log_path),
            },
        }
        _write_json(output_dir / "report.json", report)
        (output_dir / "report.md").write_text(_markdown_report(report), encoding="utf-8")
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(output_dir / "report.json")
    print(output_dir / "report.md")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", type=Path, default=Path("out"))
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    return parser


def loocv_splits(measures: Sequence[int]) -> tuple[tuple[tuple[int, ...], int], ...]:
    ordered = tuple(measures)
    if len(ordered) < 2 or len(set(ordered)) != len(ordered):
        raise ValueError("LOOCV requires at least two unique measures")
    splits = tuple((tuple(value for value in ordered if value != held), held) for held in ordered)
    if any(held in training or len(training) != len(ordered) - 1 for training, held in splits):
        raise RuntimeError("Invalid LOOCV split")
    return splits


def prepare_target_all_configs(
    request: Mapping[str, Any],
    *,
    out_dir: Path,
    slug: str,
    configs: Sequence[stem_spike.DetectorConfig],
) -> PreparedTarget:
    identity = request["identity"]
    system = int(identity["system_index"])
    measure = int(identity["system_measure_index"])
    prepared = stem_spike.prepare_request(request, out_dir=out_dir)
    grid = patch_spike._load_unlabeled_measure(
        out_dir,
        slug=slug,
        system_index=system,
        measure=measure,
        max_candidates=MAX_CANDIDATES,
    )
    if grid.source_sha256 != request["images"]["raw"]["sha256"]:
        raise ValueError(f"Grid/request image hash mismatch for S{system}M{measure}")
    suppressed = patch_spike._suppress_staff_lines(
        prepared.image,
        staff_lines=tuple(round(line) for line in prepared.staff_lines),
        staff_spacing=prepared.spacing,
    )
    patch_cache: dict[tuple[float, float], tuple[float, ...]] = {}
    variants = {}
    for config in configs:
        stems = stem_spike.detect(prepared, config)
        variants[config.key] = build_candidate_union(
            grid.candidates,
            stems,
            image=prepared.image,
            suppressed_image=suppressed,
            staff_spacing=prepared.spacing,
            threshold=prepared.threshold,
            patch_cache=patch_cache,
        )
    return PreparedTarget(dict(request), prepared, grid, variants)


def prepare_target_for_config(
    request: Mapping[str, Any],
    *,
    out_dir: Path,
    slug: str,
    config: stem_spike.DetectorConfig,
) -> PreparedTarget:
    return prepare_target_all_configs(request, out_dir=out_dir, slug=slug, configs=(config,))


def build_candidate_union(
    grid_candidates: Sequence[patch_spike.CandidatePatch],
    stem_candidates: Sequence[stem_spike.EndpointCandidate],
    *,
    image: Image.Image,
    suppressed_image: Image.Image,
    staff_spacing: float,
    threshold: int,
    patch_cache: dict[tuple[float, float], tuple[float, ...]] | None = None,
) -> tuple[HybridCandidate, ...]:
    """Merge grid/stem duplicates while retaining deterministic provenance."""
    if staff_spacing <= 0:
        raise ValueError("staff_spacing must be positive")
    cache = patch_cache if patch_cache is not None else {}
    rows = [
        HybridCandidate(
            id="",
            center_x=candidate.center_x,
            center_y=candidate.center_y,
            bbox=candidate.bbox,
            patch=candidate.patches[PATCH_ID],
            grid_ids=(candidate.id,),
            grid_rank=candidate.rank,
            grid_score=candidate.detector_score,
        )
        for candidate in grid_candidates
    ]
    for index, stem in enumerate(
        sorted(stem_candidates, key=lambda row: (row.x, row.y, -row.score, row.endpoint)), start=1
    ):
        matches = [
            (candidate_index, candidate)
            for candidate_index, candidate in enumerate(rows)
            if abs(candidate.center_x - stem.x) <= staff_spacing * UNION_X_TOLERANCE_SPACES
            and abs(candidate.center_y - stem.y) <= staff_spacing * UNION_Y_TOLERANCE_SPACES
        ]
        stem_id = f"stem{index:03d}"
        if matches:
            candidate_index, current = min(
                matches,
                key=lambda item: (
                    _normalized_distance(item[1], stem, staff_spacing),
                    not item[1].has_grid,
                    item[1].center_x,
                    item[1].center_y,
                ),
            )
            distance = _normalized_distance(current, stem, staff_spacing)
            if current.has_grid:
                center_x, center_y, bbox, vector = (
                    current.center_x,
                    current.center_y,
                    current.bbox,
                    current.patch,
                )
            else:
                center_x, center_y = (
                    (current.center_x + stem.x) / 2,
                    (current.center_y + stem.y) / 2,
                )
                bbox = _candidate_bbox(center_x, center_y, staff_spacing, image.size)
                vector = _patch_at(
                    suppressed_image,
                    center_x,
                    center_y,
                    staff_spacing,
                    threshold,
                    cache,
                )
            rows[candidate_index] = replace(
                current,
                center_x=center_x,
                center_y=center_y,
                bbox=bbox,
                patch=vector,
                stem_ids=tuple(sorted((*current.stem_ids, stem_id))),
                stem_score=max(current.stem_score, float(stem.score)),
                agreement_distance=(
                    min(current.agreement_distance, distance) if current.has_grid else 1.0
                ),
            )
            continue
        bbox = _candidate_bbox(stem.x, stem.y, staff_spacing, image.size)
        rows.append(
            HybridCandidate(
                id="",
                center_x=float(stem.x),
                center_y=float(stem.y),
                bbox=bbox,
                patch=_patch_at(
                    suppressed_image,
                    stem.x,
                    stem.y,
                    staff_spacing,
                    threshold,
                    cache,
                ),
                stem_ids=(stem_id,),
                stem_score=float(stem.score),
            )
        )
    ordered = sorted(
        rows,
        key=lambda row: (
            row.center_x,
            row.center_y,
            not row.has_grid,
            row.grid_rank if row.grid_rank is not None else MAX_CANDIDATES + 1,
            row.stem_ids,
        ),
    )
    return tuple(replace(row, id=f"u{index:03d}") for index, row in enumerate(ordered, start=1))


def _normalized_distance(
    candidate: HybridCandidate,
    stem: stem_spike.EndpointCandidate,
    spacing: float,
) -> float:
    return math.hypot(
        (candidate.center_x - stem.x) / spacing,
        (candidate.center_y - stem.y) / spacing,
    )


def _candidate_bbox(
    center_x: float, center_y: float, spacing: float, image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    width, height = image_size
    radius_x = max(2, round(spacing * 0.45))
    radius_y = max(2, round(spacing * 0.35))
    return (
        max(0, round(center_x) - radius_x),
        max(0, round(center_y) - radius_y),
        min(width, round(center_x) + radius_x + 1),
        min(height, round(center_y) + radius_y + 1),
    )


def _patch_at(
    image: Image.Image,
    center_x: float,
    center_y: float,
    spacing: float,
    threshold: int,
    cache: dict[tuple[float, float], tuple[float, ...]],
) -> tuple[float, ...]:
    key = (round(center_x, 3), round(center_y, 3))
    if key not in cache:
        cache[key] = patch_spike._extract_normalized_patch(
            image,
            center=(center_x, center_y),
            staff_spacing=spacing,
            kind="binary",
            threshold=threshold,
        )
    return cache[key]


def load_development_truth(
    slug: str, targets: Sequence[PreparedTarget]
) -> tuple[DevelopmentTruth, ...]:
    truths = []
    for target in targets:
        system = int(target.request["identity"]["system_index"])
        measure = target.measure
        coordinate_path = COORDINATE_DIR / f"{slug}_system_{system:03d}_measure_{measure:03d}.json"
        review_path = REVIEW_DIR / f"{slug}_system_{system:03d}_measure_{measure:03d}.json"
        coordinate_payload = json.loads(coordinate_path.read_text(encoding="utf-8"))
        review_payload = json.loads(review_path.read_text(encoding="utf-8"))
        coordinates = sorted(coordinate_payload["noteheads"], key=lambda row: int(row["order"]))
        pitches = sorted(review_payload["final_noteheads"], key=lambda row: int(row["order"]))
        if len(coordinates) != len(pitches):
            raise ValueError(f"Coordinate/pitch row count mismatch: {coordinate_path}")
        ellipses = []
        for coordinate, pitch in zip(coordinates, pitches, strict=True):
            if int(coordinate["order"]) != int(pitch["order"]):
                raise ValueError(f"Coordinate/pitch order mismatch: {coordinate_path}")
            geometry = coordinate["annotation_geometry"]
            ellipses.append(
                CoordinateEllipse(
                    id=str(coordinate["id"]),
                    order=int(coordinate["order"]),
                    center_x=float(coordinate["center"]["x"]),
                    center_y=float(coordinate["center"]["y"]),
                    radius_x=float(geometry["radius_x_px"]) * ELLIPSE_MARGIN,
                    radius_y=float(geometry["radius_y_px"]) * ELLIPSE_MARGIN,
                    natural_pitch_midi=_natural_pitch_midi(str(pitch["pitch"])),
                )
            )
        truths.append(
            DevelopmentTruth(
                coordinate_path,
                _sha256(coordinate_path),
                review_path,
                _sha256(review_path),
                tuple(ellipses),
            )
        )
    return tuple(truths)


def attach_all_labels(
    targets: Sequence[PreparedTarget], truths: Sequence[DevelopmentTruth]
) -> dict[str, dict[int, LabeledVariant]]:
    if len(targets) != len(truths):
        raise ValueError("Development target/truth count mismatch")
    labeled: dict[str, dict[int, LabeledVariant]] = defaultdict(dict)
    for target, truth in zip(targets, truths, strict=True):
        for config_key, candidates in target.variants.items():
            rows = tuple(
                LabeledCandidate(candidate, _ellipse_match(candidate, truth.ellipses))
                for candidate in candidates
            )
            labeled[config_key][target.measure] = LabeledVariant(target, truth, config_key, rows)
    return dict(labeled)


def _ellipse_match(candidate: HybridCandidate, ellipses: Sequence[CoordinateEllipse]) -> str | None:
    matches = []
    for ellipse in ellipses:
        distance = ((candidate.center_x - ellipse.center_x) / ellipse.radius_x) ** 2 + (
            (candidate.center_y - ellipse.center_y) / ellipse.radius_y
        ) ** 2
        if distance <= 1.0:
            matches.append((distance, ellipse.order, ellipse.id))
    return min(matches)[2] if matches else None


def evaluate_development(
    labeled: Mapping[str, Mapping[int, LabeledVariant]],
    configs: Sequence[stem_spike.DetectorConfig],
) -> dict[str, Any]:
    measures = tuple(measure for _, measure in DEVELOPMENT_TARGETS)
    first_config = configs[0].key
    method_folds: dict[str, list[dict[str, Any]]] = {method: [] for method in METHOD_IDS}
    winner_predictions_by_method: dict[str, list[dict[str, Any]]] = {
        method: [] for method in METHOD_IDS
    }
    for training, held in loocv_splits(measures):
        grid_training = [labeled[first_config][measure] for measure in training]
        grid_held = labeled[first_config][held]
        grid_scorer = _fit_patch_scorer(grid_training)
        grid_rows = [row for row in grid_held.rows if row.candidate.has_grid]
        grid_scores = {
            row.candidate.id: grid_scorer.score(_as_patch_candidate(row.candidate, held))
            for row in grid_rows
        }
        grid_count = int(
            statistics.median(_truth_count(labeled[first_config][measure]) for measure in training)
        )
        grid_selected = sorted(
            grid_rows,
            key=lambda row: (
                -grid_scores[row.candidate.id],
                row.candidate.grid_rank or MAX_CANDIDATES + 1,
                row.candidate.id,
            ),
        )[:grid_count]
        _append_fold(
            method_folds["grid_only"],
            winner_predictions_by_method["grid_only"],
            "grid_only",
            grid_held,
            training,
            grid_selected,
            grid_scores,
            {"learned_count": grid_count, "stem_config_key": None},
        )

        selected_config = select_stem_config(training, labeled, configs)
        held_variant = labeled[selected_config.key][held]
        stem_rows = [row for row in held_variant.rows if row.candidate.has_stem]
        stem_scores = {row.candidate.id: row.candidate.stem_score for row in stem_rows}
        _append_fold(
            method_folds["stem_only"],
            winner_predictions_by_method["stem_only"],
            "stem_only",
            held_variant,
            training,
            stem_rows,
            stem_scores,
            {"stem_config_key": selected_config.key},
        )

        agreement_rows = [row for row in held_variant.rows if row.candidate.agreement]
        agreement_scores = {row.candidate.id: row.candidate.stem_score for row in agreement_rows}
        _append_fold(
            method_folds["intersection_agreement"],
            winner_predictions_by_method["intersection_agreement"],
            "intersection_agreement",
            held_variant,
            training,
            agreement_rows,
            agreement_scores,
            {"stem_config_key": selected_config.key},
        )

        threshold, calibration = calibrate_hybrid_threshold(training, labeled, configs)
        model, patch_scorer = fit_hybrid_model(
            [labeled[selected_config.key][measure] for measure in training]
        )
        hybrid_scores = score_hybrid_rows(held_variant.rows, model, patch_scorer, held)
        hybrid_selected = [
            row for row in held_variant.rows if hybrid_scores[row.candidate.id] >= threshold
        ]
        _append_fold(
            method_folds["hybrid_logistic"],
            winner_predictions_by_method["hybrid_logistic"],
            "hybrid_logistic",
            held_variant,
            training,
            hybrid_selected,
            hybrid_scores,
            {
                "stem_config_key": selected_config.key,
                "threshold": threshold,
                "inner_calibration": calibration,
            },
        )

    methods = []
    for method_id in METHOD_IDS:
        aggregate = _aggregate_metrics([fold["metrics"] for fold in method_folds[method_id]])
        methods.append(
            {
                "id": method_id,
                "folds": method_folds[method_id],
                "aggregate": aggregate,
                "development_gate": apply_development_gate(aggregate),
            }
        )
    winner = max(methods, key=_development_method_key)
    return {
        "protocol": {
            "outer_split": "strict leave-one-measure-out",
            "inner_threshold_calibration": "leave-one-training-measure-out",
            "stem_config_selection": "training measures only",
            "candidate_generation_before_coordinate_or_pitch_labels": True,
            "coordinate_label_source": "independent coordinate ellipses",
            "review_candidate_labels_used": False,
            "promoted_review_fields_used": ["final_noteheads[].order", "final_noteheads[].pitch"],
        },
        "gate_reference": {
            "automatic_count_f1": REFERENCE_AUTOMATIC_F1,
            "recall": REFERENCE_AUTOMATIC_RECALL,
            "exact_count_fold_count": REFERENCE_EXACT_COUNT_FOLDS,
            "rule": (
                "F1 must preserve 0.846153846 and either recall must improve above "
                "0.785714286 or exact-count folds must improve above 3/4"
            ),
        },
        "methods": methods,
        "selection": {
            "winner_method_id": winner["id"],
            "winner_metrics": winner["aggregate"],
            "development_gate": winner["development_gate"],
            "winner_predictions": winner_predictions_by_method[winner["id"]],
        },
    }


def apply_development_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    preserves_f1 = float(metrics["f1"]) + 1e-12 >= REFERENCE_AUTOMATIC_F1
    improves_recall = float(metrics["recall"]) > REFERENCE_AUTOMATIC_RECALL + 1e-12
    improves_count = int(metrics["exact_count_fold_count"]) > REFERENCE_EXACT_COUNT_FOLDS
    return {
        "minimum_f1": REFERENCE_AUTOMATIC_F1,
        "reference_recall": REFERENCE_AUTOMATIC_RECALL,
        "reference_exact_count_fold_count": REFERENCE_EXACT_COUNT_FOLDS,
        "preserves_f1": preserves_f1,
        "improves_recall": improves_recall,
        "improves_exact_count": improves_count,
        "passed": preserves_f1 and (improves_recall or improves_count),
    }


def select_stem_config(
    training_measures: Sequence[int],
    labeled: Mapping[str, Mapping[int, LabeledVariant]],
    configs: Sequence[stem_spike.DetectorConfig],
) -> stem_spike.DetectorConfig:
    rows = []
    for config in configs:
        metrics = _aggregate_metrics(
            [
                _selection_metrics(
                    [row for row in labeled[config.key][measure].rows if row.candidate.has_stem],
                    labeled[config.key][measure],
                )
                for measure in training_measures
            ]
        )
        rows.append((config, metrics))
    return max(
        rows,
        key=lambda item: (
            item[1]["f1"],
            item[1]["exact_count_fold_count"],
            item[1]["recall"],
            item[1]["precision"],
            -item[1]["selected_count"],
            item[0].key,
        ),
    )[0]


def calibrate_hybrid_threshold(
    training_measures: Sequence[int],
    labeled: Mapping[str, Mapping[int, LabeledVariant]],
    configs: Sequence[stem_spike.DetectorConfig],
) -> tuple[float, dict[str, Any]]:
    scored_by_measure: dict[int, tuple[LabeledVariant, dict[str, float]]] = {}
    for inner_training, calibration_measure in loocv_splits(training_measures):
        config = select_stem_config(inner_training, labeled, configs)
        model, scorer = fit_hybrid_model(
            [labeled[config.key][measure] for measure in inner_training]
        )
        calibration_variant = labeled[config.key][calibration_measure]
        scored_by_measure[calibration_measure] = (
            calibration_variant,
            score_hybrid_rows(calibration_variant.rows, model, scorer, calibration_measure),
        )
    threshold, metrics = _select_threshold(scored_by_measure)
    return threshold, {"threshold": threshold, "metrics": metrics}


def fit_hybrid_model(
    training: Sequence[LabeledVariant],
) -> tuple[LogisticModel, patch_spike.PatchScorer]:
    if len(training) < 2:
        raise ValueError("Hybrid fitting requires at least two training measures")
    oof_patch_scores: dict[tuple[int, str], float] = {}
    for held in training:
        inner = [row for row in training if row.target.measure != held.target.measure]
        scorer = _fit_patch_scorer(inner)
        for row in held.rows:
            oof_patch_scores[(held.target.measure, row.candidate.id)] = scorer.score(
                _as_patch_candidate(row.candidate, held.target.measure)
            )
    feature_rows = []
    labels = []
    for variant in training:
        for row in variant.rows:
            feature_rows.append(
                hybrid_features(
                    row.candidate,
                    oof_patch_scores[(variant.target.measure, row.candidate.id)],
                )
            )
            labels.append(row.label)
    model = fit_logistic(feature_rows, labels)
    return model, _fit_patch_scorer(training)


def hybrid_features(candidate: HybridCandidate, patch_knn_score: float) -> dict[str, float]:
    return {
        "patch_knn_score": float(patch_knn_score),
        "has_grid": float(candidate.has_grid),
        "has_stem": float(candidate.has_stem),
        "agreement": float(candidate.agreement),
        "grid_score": candidate.grid_score,
        "grid_rank_quality": (
            1.0 - (candidate.grid_rank - 1) / max(1, MAX_CANDIDATES - 1)
            if candidate.grid_rank is not None
            else 0.0
        ),
        "stem_score": candidate.stem_score,
        "agreement_distance": candidate.agreement_distance if candidate.agreement else 1.0,
    }


def fit_logistic(rows: Sequence[Mapping[str, float]], labels: Sequence[int]) -> LogisticModel:
    if not rows or len(rows) != len(labels) or set(labels) != {0, 1}:
        raise ValueError("Logistic fit requires aligned positive and negative rows")
    means = tuple(statistics.mean(float(row[name]) for row in rows) for name in HYBRID_FEATURES)
    scales = tuple(
        max(
            1e-6,
            math.sqrt(statistics.mean((float(row[name]) - mean) ** 2 for row in rows)),
        )
        for name, mean in zip(HYBRID_FEATURES, means, strict=True)
    )
    matrix = [
        tuple(
            (float(row[name]) - mean) / scale
            for name, mean, scale in zip(HYBRID_FEATURES, means, scales, strict=True)
        )
        for row in rows
    ]
    positives = sum(labels)
    negatives = len(labels) - positives
    positive_weight = len(labels) / (2 * positives)
    negative_weight = len(labels) / (2 * negatives)
    weights = [0.0] * len(HYBRID_FEATURES)
    intercept = math.log(positives / negatives)
    learning_rate = 0.08
    l2 = 0.12
    for _ in range(600):
        gradients = [0.0] * len(weights)
        intercept_gradient = 0.0
        for values, label in zip(matrix, labels, strict=True):
            probability = _sigmoid(
                intercept + sum(w * value for w, value in zip(weights, values, strict=True))
            )
            sample_weight = positive_weight if label else negative_weight
            error = (probability - label) * sample_weight
            intercept_gradient += error
            for index, value in enumerate(values):
                gradients[index] += error * value
        count = len(labels)
        intercept -= learning_rate * intercept_gradient / count
        for index in range(len(weights)):
            gradient = gradients[index] / count + l2 * weights[index]
            weights[index] -= learning_rate * gradient
    return LogisticModel(
        HYBRID_FEATURES,
        means,
        scales,
        tuple(round(value, 12) for value in weights),
        round(intercept, 12),
    )


def score_hybrid_rows(
    rows: Sequence[LabeledCandidate],
    model: LogisticModel,
    patch_scorer: patch_spike.PatchScorer,
    measure: int,
) -> dict[str, float]:
    return {
        row.candidate.id: model.score(
            hybrid_features(
                row.candidate,
                patch_scorer.score(_as_patch_candidate(row.candidate, measure)),
            )
        )
        for row in rows
    }


def _fit_patch_scorer(training: Sequence[LabeledVariant]) -> patch_spike.PatchScorer:
    rows = []
    for variant in training:
        rows.extend(
            patch_spike.LabeledCandidate(
                _as_patch_candidate(row.candidate, variant.target.measure), row.label
            )
            for row in variant.rows
            if row.candidate.has_grid
        )
    return patch_spike._fit_patch_scorer(rows, patch_id=PATCH_ID, scorer_kind="class_knn3")


def _as_patch_candidate(candidate: HybridCandidate, measure: int) -> patch_spike.CandidatePatch:
    return patch_spike.CandidatePatch(
        measure=measure,
        id=candidate.id,
        rank=candidate.grid_rank or MAX_CANDIDATES + 1,
        center_x=candidate.center_x,
        center_y=candidate.center_y,
        bbox=candidate.bbox,
        detector_score=candidate.grid_score,
        patches={PATCH_ID: candidate.patch},
    )


def _select_threshold(
    scored_by_measure: Mapping[int, tuple[LabeledVariant, Mapping[str, float]]],
) -> tuple[float, dict[str, Any]]:
    values = sorted(
        {
            float(scores[row.candidate.id])
            for variant, scores in scored_by_measure.values()
            for row in variant.rows
        }
    )
    thresholds = [values[-1] + 1e-9, *reversed(values)]
    best_threshold = thresholds[0]
    best_metrics: dict[str, Any] | None = None
    best_key: tuple[Any, ...] | None = None
    for threshold in thresholds:
        metrics = _aggregate_metrics(
            [
                _selection_metrics(
                    [row for row in variant.rows if scores[row.candidate.id] >= threshold],
                    variant,
                )
                for variant, scores in scored_by_measure.values()
            ]
        )
        key = (
            metrics["f1"],
            metrics["exact_count_fold_count"],
            metrics["recall"],
            metrics["precision"],
            -metrics["selected_count"],
            threshold,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
            best_metrics = metrics
    assert best_metrics is not None
    return best_threshold, best_metrics


def _append_fold(
    folds: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    method_id: str,
    variant: LabeledVariant,
    training: Sequence[int],
    selected: Sequence[LabeledCandidate],
    scores: Mapping[str, float],
    fitting: Mapping[str, Any],
) -> None:
    metrics = _selection_metrics(selected, variant)
    folds.append(
        {
            "held_out_measure": variant.target.measure,
            "training_measures": list(training),
            "metrics": metrics,
            "fitting": dict(fitting),
        }
    )
    predictions.append(
        prediction_payload(
            variant.target,
            [row.candidate for row in selected],
            method_id=method_id,
            scores=scores,
            pitch_calibrator=None,
            include_development_labels={
                row.candidate.id: row.matched_ellipse_id for row in variant.rows
            },
        )
    )


def _selection_metrics(
    selected: Sequence[LabeledCandidate], variant: LabeledVariant
) -> dict[str, Any]:
    matched = set()
    tp = 0
    for row in sorted(selected, key=lambda value: value.candidate.id):
        if row.matched_ellipse_id is not None and row.matched_ellipse_id not in matched:
            matched.add(row.matched_ellipse_id)
            tp += 1
    fp = len(selected) - tp
    truth_count = _truth_count(variant)
    fn = truth_count - tp
    return {
        "selected_count": len(selected),
        "truth_ellipse_count": truth_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _f1(tp, fp, fn),
        "exact_count": len(selected) == truth_count,
        "selected_candidate_ids": [row.candidate.id for row in selected],
    }


def _truth_count(variant: LabeledVariant) -> int:
    return len(variant.truth.ellipses)


def _aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    for row in rows:
        totals.update({name: int(row[name]) for name in ("tp", "fp", "fn", "selected_count")})
    return {
        "selected_count": totals["selected_count"],
        "truth_ellipse_count": totals["tp"] + totals["fn"],
        "tp": totals["tp"],
        "fp": totals["fp"],
        "fn": totals["fn"],
        "precision": _ratio(totals["tp"], totals["tp"] + totals["fp"]),
        "recall": _ratio(totals["tp"], totals["tp"] + totals["fn"]),
        "f1": _f1(totals["tp"], totals["fp"], totals["fn"]),
        "exact_count_fold_count": sum(bool(row["exact_count"]) for row in rows),
        "fold_count": len(rows),
    }


def _development_method_key(method: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = method["aggregate"]
    return (
        bool(method["development_gate"]["passed"]),
        metrics["f1"],
        metrics["recall"],
        metrics["exact_count_fold_count"],
        metrics["precision"],
        -metrics["selected_count"],
        method["id"] == "hybrid_logistic",
        method["id"],
    )


def fit_winner(
    method_id: str,
    labeled: Mapping[str, Mapping[int, LabeledVariant]],
    configs: Sequence[stem_spike.DetectorConfig],
) -> dict[str, Any]:
    measures = tuple(measure for _, measure in DEVELOPMENT_TARGETS)
    config = select_stem_config(measures, labeled, configs)
    variants = [labeled[config.key][measure] for measure in measures]
    fitted: dict[str, Any] = {
        "method_id": method_id,
        "config": config,
        "threshold": None,
        "model": None,
        "patch_scorer": None,
        "learned_count": None,
    }
    if method_id == "hybrid_logistic":
        threshold, calibration = calibrate_hybrid_threshold(measures, labeled, configs)
        model, scorer = fit_hybrid_model(variants)
        fitted.update(
            {
                "threshold": threshold,
                "calibration": calibration,
                "model": model,
                "patch_scorer": scorer,
            }
        )
    elif method_id == "grid_only":
        fitted["patch_scorer"] = _fit_patch_scorer(variants)
        fitted["learned_count"] = int(
            statistics.median(_truth_count(variant) for variant in variants)
        )
    return fitted


def fitted_report(fitted: Mapping[str, Any], calibrator: PitchCalibrator) -> dict[str, Any]:
    model = fitted.get("model")
    return {
        "method_id": fitted["method_id"],
        "stem_config": asdict(fitted["config"]),
        "stem_config_key": fitted["config"].key,
        "threshold": fitted.get("threshold"),
        "learned_count": fitted.get("learned_count"),
        "hybrid_features": list(HYBRID_FEATURES) if model else None,
        "logistic_model": (
            {
                "means": list(model.means),
                "scales": list(model.scales),
                "weights": list(model.weights),
                "intercept": model.intercept,
                "iterations": 600,
                "learning_rate": 0.08,
                "l2": 0.12,
            }
            if model
            else None
        ),
        "pitch_calibration": {
            "kind": "unanimous development staff-position correction",
            "position_to_natural_midi": {
                str(key): value for key, value in sorted(calibrator.unanimous_position_map.items())
            },
        },
    }


def fit_pitch_calibrator(
    targets: Sequence[PreparedTarget], truths: Sequence[DevelopmentTruth]
) -> PitchCalibrator:
    by_position: dict[int, list[int]] = defaultdict(list)
    for target, truth in zip(targets, truths, strict=True):
        for ellipse in truth.ellipses:
            position = _staff_position(ellipse.center_y, target.prepared)
            by_position[position].append(ellipse.natural_pitch_midi)
    unanimous = {
        position: values[0]
        for position, values in by_position.items()
        if len(set(values)) == 1
        and values[0] != stem_spike.natural_midi_for_staff_position(position)
    }
    return PitchCalibrator(unanimous)


def run_sealed_split(
    split: str,
    requests_path: Path,
    truth_path: Path,
    output_dir: Path,
    *,
    out_dir: Path,
    expected_systems: Sequence[int],
    fitted: Mapping[str, Any],
    pitch_calibrator: PitchCalibrator,
    truth_gate: PredictionTruthGate,
    leakage_events: list[dict[str, Any]],
) -> dict[str, Any]:
    requests = stem_spike._read_jsonl(requests_path)
    stem_spike._require_systems(requests, expected_systems, split)
    targets = [
        prepare_target_for_config(
            request,
            out_dir=out_dir,
            slug=str(request["identity"]["slug"]),
            config=fitted["config"],
        )
        for request in requests
    ]
    predictions = [predict_target(target, fitted, pitch_calibrator) for target in targets]
    prediction_path = output_dir / f"{split}_predictions.sealed.jsonl"
    _write_jsonl(prediction_path, predictions)
    _write_overlays(targets, predictions, output_dir / f"{split}_overlays")
    seal = truth_gate.seal_predictions(split, prediction_path)
    leakage_events.append(
        {
            "phase": 4 if split == "validation" else 6,
            "event": f"{split}_predictions_and_features_sealed",
            "request_path": str(requests_path),
            "request_sha256": _sha256(requests_path),
            "prediction_path": str(prediction_path),
            "prediction_sha256": seal["sha256"],
            "truth_accessed": False,
        }
    )
    truth = truth_gate.read_truth(split, truth_path)
    leakage_events.append(
        {
            "phase": 5 if split == "validation" else 7,
            "event": f"{split}_truth_opened_after_prediction_seal",
            "truth_path": str(truth_path),
            "truth_sha256": _sha256(truth_path),
            "after_prediction_sha256": seal["sha256"],
        }
    )
    return {
        "status": "evaluated_after_seal",
        "prediction_seal": seal,
        "metrics": stem_spike.evaluate_pitch_only(truth, predictions),
    }


def predict_target(
    target: PreparedTarget,
    fitted: Mapping[str, Any],
    pitch_calibrator: PitchCalibrator,
) -> dict[str, Any]:
    candidates = target.variants[fitted["config"].key]
    method_id = str(fitted["method_id"])
    scores: dict[str, float]
    if method_id == "hybrid_logistic":
        rows = tuple(LabeledCandidate(candidate, None) for candidate in candidates)
        scores = score_hybrid_rows(
            rows,
            fitted["model"],
            fitted["patch_scorer"],
            target.measure,
        )
        selected = [
            candidate for candidate in candidates if scores[candidate.id] >= fitted["threshold"]
        ]
    elif method_id == "grid_only":
        grid = [candidate for candidate in candidates if candidate.has_grid]
        scores = {
            candidate.id: fitted["patch_scorer"].score(
                _as_patch_candidate(candidate, target.measure)
            )
            for candidate in grid
        }
        selected = sorted(
            grid,
            key=lambda candidate: (
                -scores[candidate.id],
                candidate.grid_rank or MAX_CANDIDATES + 1,
                candidate.id,
            ),
        )[: fitted["learned_count"]]
    elif method_id == "stem_only":
        selected = [candidate for candidate in candidates if candidate.has_stem]
        scores = {candidate.id: candidate.stem_score for candidate in selected}
    elif method_id == "intersection_agreement":
        selected = [candidate for candidate in candidates if candidate.agreement]
        scores = {candidate.id: candidate.stem_score for candidate in selected}
    else:
        raise ValueError(f"Unsupported fitted method: {method_id}")
    return prediction_payload(
        target,
        selected,
        method_id=method_id,
        scores=scores,
        pitch_calibrator=pitch_calibrator,
    )


def prediction_payload(
    target: PreparedTarget,
    candidates: Sequence[HybridCandidate],
    *,
    method_id: str,
    scores: Mapping[str, float],
    pitch_calibrator: PitchCalibrator | None,
    include_development_labels: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    ordered = sorted(candidates, key=lambda row: (row.center_x, row.center_y, row.id))
    pitches = (
        [pitch_calibrator.predict(candidate, target.prepared) for candidate in ordered]
        if pitch_calibrator is not None
        else []
    )
    rows = []
    for index, candidate in enumerate(ordered, start=1):
        row = {
            "id": f"note{index:03d}",
            "union_candidate_id": candidate.id,
            "center": {"x": round(candidate.center_x, 3), "y": round(candidate.center_y, 3)},
            "score": round(float(scores.get(candidate.id, 0.0)), 9),
            "provenance": {
                "grid_ids": list(candidate.grid_ids),
                "stem_ids": list(candidate.stem_ids),
                "agreement": candidate.agreement,
            },
        }
        if pitch_calibrator is not None:
            row["pitch_midi"] = pitches[index - 1]
            row["staff_position"] = _staff_position(candidate.center_y, target.prepared)
        if include_development_labels is not None:
            row["development_matched_ellipse_id"] = include_development_labels.get(candidate.id)
        rows.append(row)
    return {
        "schema_version": 1,
        "identity": dict(target.request["identity"]),
        "method": method_id,
        "natural_pitch_only": True,
        "predicted_note_count": len(ordered),
        "ordered_pitches": pitches,
        "candidates": rows,
        "rhythm": "out_of_scope",
        "durations": "out_of_scope",
    }


def apply_validation_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    summary = metrics["summary"]
    truth_count = int(summary["truth_note_count"])
    ratio = _ratio(int(summary["predicted_note_count"]), truth_count)
    observed = {
        "ordered_natural_pitch_accuracy": float(summary["ordered_natural_pitch_accuracy"]),
        "pitch_only_note_f1": float(summary["pitch_only_note_f1"]),
        "exact_note_count_rate": float(summary["exact_note_count_rate"]),
        "predicted_to_truth_count_ratio": ratio,
    }
    passed = (
        observed["ordered_natural_pitch_accuracy"]
        >= VALIDATION_GATE["minimum_ordered_natural_pitch_accuracy"]
        and observed["pitch_only_note_f1"] >= VALIDATION_GATE["minimum_pitch_only_note_f1"]
        and observed["exact_note_count_rate"] >= VALIDATION_GATE["minimum_exact_note_count_rate"]
        and VALIDATION_GATE["minimum_predicted_to_truth_count_ratio"]
        <= ratio
        <= VALIDATION_GATE["maximum_predicted_to_truth_count_ratio"]
    )
    return {
        **VALIDATION_GATE,
        "full_system_vlm_ordered_pitch_reference": FULL_SYSTEM_VLM_ORDERED_PITCH,
        "observed": observed,
        "passed": passed,
        "failure_action": None if passed else "skip_heldout_without_reading_heldout_truth",
    }


def _staff_position(y: float, prepared: stem_spike.PreparedRequest) -> int:
    return round((prepared.staff_lines[-1] - y) / (prepared.spacing / 2))


def _natural_pitch_midi(pitch: str) -> int:
    if len(pitch) < 2 or pitch[0].upper() not in "ABCDEFG":
        raise ValueError(f"Unsupported promoted pitch: {pitch!r}")
    letter = pitch[0].upper()
    octave_text = pitch[-1]
    if not octave_text.isdigit():
        raise ValueError(f"Unsupported promoted pitch octave: {pitch!r}")
    pitch_class = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[letter]
    return (int(octave_text) + 1) * 12 + pitch_class


def _write_overlays(
    targets: Sequence[PreparedTarget],
    predictions: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for target, prediction in zip(targets, predictions, strict=True):
        image = target.prepared.image.convert("RGB")
        draw = ImageDraw.Draw(image)
        for line in target.prepared.staff_lines:
            draw.line((0, line, image.width - 1, line), fill="#2d73d5", width=1)
        for candidate in prediction["candidates"]:
            x = float(candidate["center"]["x"])
            y = float(candidate["center"]["y"])
            radius = max(3, round(target.prepared.spacing * 0.22))
            agreement = bool(candidate["provenance"]["agreement"])
            color = "#16a34a" if agreement else "#d62728"
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
            label = candidate["id"]
            if "pitch_midi" in candidate:
                label += f":{candidate['pitch_midi']}"
            draw.text((x + radius + 1, y - radius), label, fill=color)
        identity = target.request["identity"]
        path = output_dir / (
            f"system_{int(identity['system_index']):03d}_"
            f"measure_{int(identity['system_measure_index']):03d}.png"
        )
        image.save(path)


def _markdown_report(report: Mapping[str, Any]) -> str:
    development = report["development"]
    selection = development["selection"]
    lines = [
        "# Hybrid Notehead Selector Spike",
        "",
        "## Development",
        "",
        "All cap-24, stem, union, provenance, and patch inputs were built before independent "
        "coordinate ellipses or promoted pitches were opened. Every selection metric below is "
        "from a strict held-out S1 measure.",
        "",
        "| Method | Selected | TP/FP/FN | Precision | Recall | F1 | Exact count | Gate |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for method in development["methods"]:
        metrics = method["aggregate"]
        gate_status = "pass" if method["development_gate"]["passed"] else "fail"
        lines.append(
            f"| `{method['id']}` | {metrics['selected_count']} | "
            f"{metrics['tp']}/{metrics['fp']}/{metrics['fn']} | "
            f"{metrics['precision']:.3f} | {metrics['recall']:.3f} | "
            f"{metrics['f1']:.3f} | {metrics['exact_count_fold_count']}/"
            f"{metrics['fold_count']} | {gate_status} |"
        )
    lines.extend(
        [
            "",
            f"Winner: `{selection['winner_method_id']}`. Development gate passed: "
            f"`{selection['development_gate']['passed']}`.",
            "",
            "The gate preserves automatic-count F1 0.846153846 and additionally requires "
            "recall above 0.785714286 or more than three exact-count folds.",
            "",
            "## Validation",
            "",
            f"Status: `{report['validation']['status']}`.",
        ]
    )
    if report["validation"]["metrics"] is not None:
        summary = report["validation"]["metrics"]["summary"]
        gate = report["validation"]["gate"]
        lines.extend(
            [
                f"- Prediction SHA256: `{report['validation']['prediction_seal']['sha256']}`",
                f"- Predicted/truth count: `{summary['predicted_note_count']}/"
                f"{summary['truth_note_count']}`",
                f"- Exact count rate: `{summary['exact_note_count_rate']:.3f}`",
                f"- Ordered natural-pitch accuracy: "
                f"`{summary['ordered_natural_pitch_accuracy']:.3f}`",
                f"- Multiset natural-pitch P/R/F1: "
                f"`{summary['pitch_only_note_precision']:.3f}/"
                f"{summary['pitch_only_note_recall']:.3f}/"
                f"{summary['pitch_only_note_f1']:.3f}`",
                f"- Validation gate passed: `{gate['passed']}`",
            ]
        )
    lines.extend(["", "## Heldout", "", f"Status: `{report['heldout']['status']}`."])
    if report["heldout"]["metrics"] is not None:
        summary = report["heldout"]["metrics"]["summary"]
        lines.extend(
            [
                f"- Prediction SHA256: `{report['heldout']['prediction_seal']['sha256']}`",
                f"- Exact count rate: `{summary['exact_note_count_rate']:.3f}`",
                f"- Ordered natural-pitch accuracy: "
                f"`{summary['ordered_natural_pitch_accuracy']:.3f}`",
                f"- Multiset natural-pitch F1: `{summary['pitch_only_note_f1']:.3f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Leakage Audit",
            "",
            f"Deterministic phase and hash log: `{report['leakage']['log_path']}` "
            f"(`{report['leakage']['log_sha256']}`).",
            "",
            "No network, rhythm, duration, dependency, or production integration is part of "
            "this experiment.",
        ]
    )
    return "\n".join(lines) + "\n"


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-min(value, 60.0)))
    exponential = math.exp(max(value, -60.0))
    return exponential / (1.0 + exponential)


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    return _ratio(2 * tp, 2 * tp + fp + fn)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
