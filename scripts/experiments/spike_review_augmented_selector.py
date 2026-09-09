"""Train a dense notehead selector from promoted S1+S7 human reviews.

This spike turns the newly reviewed system-7 coordinates and pitches into
training data. It selects all model and threshold choices from systems 1 and 7,
freezes system-8 predictions, and only then opens system-8 benchmark truth.
System 3 is never accessed because its one-shot heldout gate was already used.

Example:
    uv run python scripts/experiments/spike_review_augmented_selector.py out
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_vlm_melody_event_benchmark as benchmark  # noqa: E402
from scripts import build_vlm_notehead_candidates as detector  # noqa: E402
from scripts.experiments import spike_anchored_rhythm_parser as rhythm  # noqa: E402
from scripts.experiments import spike_composed_melody_chain as composed  # noqa: E402
from scripts.experiments import spike_notehead_patch_templates as patches  # noqa: E402
from scripts.experiments import strict_initial_key_context as visual_key_context  # noqa: E402

DEFAULT_OUT_DIR = REPO_ROOT / "out"
DEFAULT_SLUG = patches.DEFAULT_SLUG
DEFAULT_REVIEWS_DIR = REPO_ROOT / "tests/fixtures/vlm_melody/notehead_reviews"
OUTPUT_SUBDIR = "review_augmented_selector"
TRAINING_TARGETS = tuple((1, measure) for measure in range(1, 5)) + tuple(
    (7, measure) for measure in range(1, 8)
)
VALIDATION_TARGETS = tuple((8, measure) for measure in range(1, 8))
METHOD_ID = "dense_staff_grid_logistic_s1_s7"
SELECTION_MODE = composed.THRESHOLD_SELECTOR
CURRENT_COMPOSED_NOTE_F1 = 0.264463
DENSE_MAX_PROPOSALS = 160
LOCAL_MAX_RADIUS_SPACES = 0.30
MIN_GRID_SCORE = 0.04
TARGET_X_TOLERANCE_SPACES = 0.72
NMS_X_SPACES_GRID = (0.45, 0.55, 0.68, 0.85, 1.0)
MAX_SELECTED_COUNT_GRID = (4, 5, 6)
MIN_SELECTED_COUNT_GRID = (0, 2, 3, 4)
MINIMUM_RHYTHM_UNIT_BEATS = 0.5
EXPECTED_MEASURE_BEATS = 3.0
MAX_RHYTHM_EVENTS_PER_MEASURE = round(EXPECTED_MEASURE_BEATS / MINIMUM_RHYTHM_UNIT_BEATS)
LOGISTIC_ITERATIONS = 220
MINIMUM_ACCIDENTAL_OOF_GAIN = 0.05
ACCIDENTAL_PATCH_WIDTH = 15
ACCIDENTAL_PATCH_HEIGHT = 25

DENSE_FEATURES = (
    "grid_score",
    "ink_density",
    "core_density",
    "vertical_support",
    "horizontal_support",
    "row_peak_density",
    "column_peak_density",
    "line_dominance",
    "stem_evidence",
    "patch_ink_density",
    "patch_center_density",
    "patch_horizontal_balance",
    "patch_vertical_balance",
)

_PITCH_PATTERN = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")


@dataclass(frozen=True)
class ReviewNote:
    id: str
    order: int
    x: float
    annotated_y: float
    pitch: str
    pitch_row_y: float
    source_kind: str


@dataclass(frozen=True)
class TrainingExample:
    key: str
    request: dict[str, Any]
    measure: patches.MeasureData
    notes: tuple[ReviewNote, ...]
    matched_candidate_ids: frozenset[str]
    unmatched_note_ids: tuple[str, ...]
    pitch_vectors: tuple[tuple[float, ...], ...]

    @property
    def true_note_count(self) -> int:
        return len(self.notes)


@dataclass(frozen=True)
class LogisticScorer:
    means: tuple[float, ...]
    scales: tuple[float, ...]
    weights: tuple[float, ...]
    intercept: float

    def score(self, candidate: patches.CandidatePatch) -> float:
        vector = candidate.patches["dense_features"]
        normalized = [
            (value - mean) / scale
            for value, mean, scale in zip(vector, self.means, self.scales, strict=True)
        ]
        return _sigmoid(
            self.intercept
            + sum(weight * value for weight, value in zip(self.weights, normalized, strict=True))
        )


@dataclass(frozen=True)
class DenseSelectorModel:
    scorer: LogisticScorer
    learned_threshold: float
    threshold_training_metrics: dict[str, Any]
    training_keys: tuple[str, ...]
    training_positive_count: int
    learned_count: int
    nms_x_spaces: float
    minimum_selected_count: int
    maximum_selected_count: int

    @property
    def training_measures(self) -> tuple[int, ...]:
        return tuple(range(1, len(self.training_keys) + 1))

    def probability(self, score: float) -> float:
        return score

    def rank(
        self,
        measure: patches.UnlabeledMeasure,
        *,
        selection_mode: str = SELECTION_MODE,
    ) -> tuple[list[dict[str, Any]], list[patches.CandidatePatch]]:
        if selection_mode != SELECTION_MODE:
            raise ValueError(f"Dense selector supports only {SELECTION_MODE!r}")
        scores = {candidate.id: self.scorer.score(candidate) for candidate in measure.candidates}
        selected = _select_candidates(
            measure,
            scores,
            threshold=self.learned_threshold,
            nms_x_spaces=self.nms_x_spaces,
            minimum_selected_count=self.minimum_selected_count,
            maximum_selected_count=self.maximum_selected_count,
        )
        selected_ids = {candidate.id for candidate in selected}
        ranked = sorted(
            measure.candidates,
            key=lambda candidate: (-scores[candidate.id], candidate.rank, candidate.id),
        )
        rows = []
        for rank, candidate in enumerate(ranked, start=1):
            score = scores[candidate.id]
            rows.append(
                {
                    "candidate_id": candidate.id,
                    "detector_rank": candidate.rank,
                    "selection_rank": rank,
                    "score": round(score, 9),
                    "probability": round(score, 9),
                    "selected": candidate.id in selected_ids,
                    "selection_mode": selection_mode,
                    "selected_by_learned_count": rank <= self.learned_count,
                    "selected_by_learned_threshold": score >= self.learned_threshold,
                    "center": {
                        "x": round(candidate.center_x, 3),
                        "y": round(candidate.center_y, 3),
                    },
                    "bbox": {
                        "left": candidate.bbox[0],
                        "top": candidate.bbox[1],
                        "right": candidate.bbox[2],
                        "bottom": candidate.bbox[3],
                    },
                }
            )
        return rows, selected


@dataclass(frozen=True)
class PitchSample:
    key: str
    pitch: str
    base_pitch: str
    delta: int
    vector: tuple[float, ...]


@dataclass(frozen=True)
class AccidentalKNN:
    samples: tuple[PitchSample, ...]

    def predict_delta(self, vector: Sequence[float]) -> int:
        by_class: dict[int, list[float]] = {}
        for sample in self.samples:
            distance = _mean_squared_distance(vector, sample.vector)
            by_class.setdefault(sample.delta, []).append(distance)
        if not by_class:
            return 0
        class_scores = {
            delta: statistics.mean(sorted(distances)[: min(3, len(distances))])
            for delta, distances in by_class.items()
        }
        return min(class_scores, key=lambda delta: (class_scores[delta], abs(delta), delta))


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
    print(f"system-8 gate: {report['validation_gate']['status']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", type=Path, default=DEFAULT_OUT_DIR)
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
    output_dir = output_dir or benchmark_dir / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    development_requests_path = benchmark_dir / "development/requests.jsonl"
    validation_requests_path = benchmark_dir / "validation/requests.jsonl"
    development_requests = _read_jsonl(development_requests_path)
    validation_requests = _read_jsonl(validation_requests_path)
    training_requests = _select_requests(
        [*development_requests, *validation_requests], TRAINING_TARGETS
    )
    system8_requests = _select_requests(validation_requests, VALIDATION_TARGETS)

    # Every proposal and image feature is materialized before any review fixture is opened.
    training_unlabeled = [
        _prepare_dense_measure(request, out_dir=out_dir) for request in training_requests
    ]
    validation_unlabeled = [
        _prepare_dense_measure(request, out_dir=out_dir) for request in system8_requests
    ]
    training = [
        _attach_review(
            request,
            measure,
            reviews_dir=reviews_dir,
            slug=slug,
        )
        for request, measure in zip(training_requests, training_unlabeled, strict=True)
    ]

    oof_scores = _out_of_fold_scores(training)
    training_selection = _select_training_configuration(training, oof_scores)
    learned_threshold = float(training_selection["threshold"])
    training_metrics = dict(training_selection["metrics"])
    full_scorer = _fit_logistic_scorer(training)
    selector_model = DenseSelectorModel(
        scorer=full_scorer,
        learned_threshold=learned_threshold,
        threshold_training_metrics=training_metrics,
        training_keys=tuple(example.key for example in training),
        training_positive_count=sum(example.true_note_count for example in training),
        learned_count=int(statistics.median(example.true_note_count for example in training)),
        nms_x_spaces=float(training_selection["nms_x_spaces"]),
        minimum_selected_count=int(training_selection["minimum_selected_count"]),
        maximum_selected_count=int(training_selection["maximum_selected_count"]),
    )

    pitch_cv = _evaluate_pitch_methods(training)
    pitch_method = str(pitch_cv["selected_method"])
    accidental_model = _fit_accidental_model(training) if pitch_method == "accidental_knn" else None
    pitch_predictor = _build_pitch_predictor(accidental_model)

    training_snapshot_path = output_dir / "training_selection.json"
    training_snapshot = {
        "status": "selected_before_validation_prediction",
        "training_targets": [example.key for example in training],
        "review_hashes": [example.measure.review_sha256 for example in training],
        "dense_proposal_recall": _proposal_recall(training),
        "selector_threshold": learned_threshold,
        "selector_oof_metrics": training_metrics,
        "selector_configuration": training_selection,
        "pitch_oof": pitch_cv,
        "validation_request_sha256": _sha256(validation_requests_path),
        "validation_targets": [_identity_key(request["identity"]) for request in system8_requests],
    }
    training_snapshot_path.write_text(
        json.dumps(training_snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    validation_composed = [
        _compose_with_meter_fallback(
            request,
            measure,
            selector_model,  # type: ignore[arg-type]
            out_dir=out_dir,
            selection_mode=SELECTION_MODE,
            selector_method_id=METHOD_ID,
            pitch_predictor=pitch_predictor,
        )
        for request, measure in zip(system8_requests, validation_unlabeled, strict=True)
    ]
    artifacts = composed.freeze_split_predictions(
        split="validation",
        composed=validation_composed,
        output_dir=output_dir,
        requests_path=validation_requests_path,
        training=[example.measure for example in training],
        selector_mode="s1_s7_review_oof_threshold_dense_grid",
        selector_method_id=METHOD_ID,
    )

    # This is the first benchmark-truth access in this experiment run.
    validation_metrics = composed.evaluate_frozen_split(
        benchmark_dir,
        artifacts=artifacts,
    )
    historical = _historical_system8_metrics(benchmark_dir)
    validation_gate = _validation_gate(validation_metrics["summary"], historical)

    report_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    report = {
        "schema_version": 1,
        "kind": "review_augmented_dense_notehead_selector_spike",
        "slug": slug,
        "protocol": {
            "training": "promoted S1M1-4 and S7M1-7 reviews only",
            "model_selection": "measure-level out-of-fold candidate and pitch predictions",
            "validation": "freeze S8M1-7 predictions before opening validation truth",
            "heldout": "not accessed; system 3 was consumed by an earlier one-shot arm",
        },
        "training": {
            "targets": [example.key for example in training],
            "review_hashes": [
                {
                    "key": example.key,
                    "path": _display_path(example.measure.review_path),
                    "sha256": example.measure.review_sha256,
                }
                for example in training
            ],
            "noteheads": sum(example.true_note_count for example in training),
            "dense_proposal_recall": _proposal_recall(training),
            "unmatched_training_noteheads": [
                {"key": example.key, "note_ids": list(example.unmatched_note_ids)}
                for example in training
                if example.unmatched_note_ids
            ],
            "selector_oof": {
                "threshold": round(learned_threshold, 9),
                "metrics": training_metrics,
                "configuration_search": training_selection,
                "proposal_count": sum(len(example.measure.rows) for example in training),
            },
            "pitch_oof": pitch_cv,
        },
        "selector": {
            "method_id": METHOD_ID,
            "features": list(DENSE_FEATURES),
            "maximum_proposals_per_measure": DENSE_MAX_PROPOSALS,
            "minimum_grid_score": MIN_GRID_SCORE,
            "nms_x_spaces": selector_model.nms_x_spaces,
            "minimum_selected_count": selector_model.minimum_selected_count,
            "maximum_selected_count": selector_model.maximum_selected_count,
            "learned_threshold": round(learned_threshold, 9),
            "pitch_method": pitch_method,
        },
        "leakage_audit": {
            "training_review_fields": [
                "candidates[].label",
                "final_noteheads[].center",
                "final_noteheads[].pitch",
            ],
            "validation_expected_counts_used": False,
            "validation_pitch_sequences_used": False,
            "validation_truth_accessed_after_freeze": True,
            "prediction_freeze_sha256": artifacts.freeze_sha256,
            "network_used": False,
            "system3_accessed": False,
            "validation_disclosure": (
                "A first frozen arm reached the truth evaluator but was rejected before metrics "
                "because unresolved predictions exceeded the request's 3-beat extent. This "
                "follow-up adds only request-derived monophony and meter-validity constraints; "
                "the invalid first freeze is preserved under initial_invalid_meter_freeze."
            ),
        },
        "validation": {
            "targets": [_identity_key(request["identity"]) for request in system8_requests],
            "metrics": validation_metrics,
            "historical_threshold_selector_same_targets": historical,
            "per_measure_counts": composed._per_measure_counts(validation_composed),
            "artifacts": composed._artifact_summary(artifacts),
        },
        "validation_gate": validation_gate,
        "artifacts": {
            "report_json": _display_path(report_path),
            "report_markdown": _display_path(markdown_path),
            "training_selection": _display_path(training_snapshot_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report, markdown_path)
    return report


def _compose_with_meter_fallback(
    request: Mapping[str, Any],
    measure: patches.UnlabeledMeasure,
    model: DenseSelectorModel,
    *,
    out_dir: Path,
    selection_mode: str,
    selector_method_id: str,
    pitch_predictor: composed.PitchPredictor,
) -> composed.ComposedMeasure:
    item = composed.compose_measure(
        request,
        measure,
        model,  # type: ignore[arg-type]
        out_dir=out_dir,
        selection_mode=selection_mode,
        selector_method_id=selector_method_id,
        pitch_predictor=pitch_predictor,
    )
    expected_beats = float(request["allowed_context"]["expected_measure_beats"])
    allow_pickup = bool(request["allowed_context"].get("allow_pickup", False))
    observed_beats = float(item.prediction["measure_extent_beats"])
    if math.isclose(observed_beats, expected_beats) or (
        allow_pickup and observed_beats <= expected_beats
    ):
        return item

    symbols = [dict(symbol) for symbol in item.visual_symbols]
    target_units = round(expected_beats * 2)
    symbols = symbols[:MAX_RHYTHM_EVENTS_PER_MEASURE]
    maximum_units = sum(3 if symbol["kind"] == "rest" else 4 for symbol in symbols)
    next_x = float(symbols[-1]["x"] + item.staff_spacing) if symbols else 0.0
    while maximum_units < target_units:
        symbols.append(
            {
                "kind": "rest",
                "x": next_x,
                "duration_beats": 0.5,
                "duration_costs": {"0.5": 1.0, "1.0": 0.8, "1.5": 0.9},
                "evidence": "request_meter_fallback",
            }
        )
        next_x += item.staff_spacing
        maximum_units += 3
    decoded, status = rhythm.decode_meter(
        symbols,
        expected_beats=expected_beats,
        allow_pickup=allow_pickup,
    )
    repaired_extent = sum(float(symbol["duration_beats"]) for symbol in decoded)
    if not math.isclose(repaired_extent, expected_beats):
        raise ValueError(
            f"Request-only meter fallback failed for {request['identity']}: "
            f"{repaired_extent} != {expected_beats}"
        )
    prediction = rhythm.symbols_to_hypothesis(
        decoded,
        identity=request["identity"],
        decoder_status=f"request_meter_fallback:{status}",
    )
    provenance = dict(item.prediction["inference_provenance"])
    provenance["meter_fallback"] = {
        "applied": True,
        "expected_beats": expected_beats,
        "original_extent_beats": observed_beats,
        "truth_used": False,
    }
    prediction["inference_provenance"] = provenance
    item.visual_symbols = symbols
    item.decoded_symbols = decoded
    item.prediction = prediction
    return item


def _prepare_dense_measure(
    request: Mapping[str, Any],
    *,
    out_dir: Path,
) -> patches.UnlabeledMeasure:
    identity = request["identity"]
    measure_index = int(identity["system_measure_index"])
    source_path = composed._resolve_request_image(request, out_dir)
    if _sha256(source_path) != str(request["images"]["raw"]["sha256"]):
        raise ValueError(f"Request image hash mismatch: {source_path}")
    staff_lines = tuple(int(value) for value in request["staff_geometry"]["raw_staff_lines_y_px"])
    with Image.open(source_path) as opened:
        grayscale = opened.convert("L")
    stages = detector._detect_staff_grid_stages(
        grayscale.convert("RGB"),
        staff_lines=list(staff_lines),
        max_candidates=1,
    )
    suppressed = patches._suppress_staff_lines(
        grayscale,
        staff_lines=staff_lines,
        staff_spacing=stages.staff_spacing,
    )
    peaks = _staff_grid_local_maxima(stages)
    candidates_with_score: list[
        tuple[float, float, float, dict[str, float], dict[str, tuple[float, ...]]]
    ] = []
    for center_x, center_y, grid_score in peaks:
        _, features = detector._score_grid_window(
            stages.threshold_mask,
            center_x=round(center_x),
            center_y=round(center_y),
            window_width=stages.window_width,
            window_height=stages.window_height,
            staff_spacing=stages.staff_spacing,
        )
        patch_vectors = {
            spec.id: patches._extract_normalized_patch(
                suppressed if spec.suppress_staff_lines else grayscale,
                center=(center_x, center_y),
                staff_spacing=stages.staff_spacing,
                kind=spec.kind,
                threshold=stages.threshold,
            )
            for spec in patches.PATCH_SPECS
        }
        binary_patch = patch_vectors["binary_staff_suppressed"]
        shape = _binary_patch_features(binary_patch)
        dense_features = {"grid_score": grid_score, **features, **shape}
        candidates_with_score.append(
            (grid_score, center_x, center_y, dense_features, patch_vectors)
        )
    candidates_with_score.sort(key=lambda item: (-item[0], item[1], item[2]))
    candidates_with_score = candidates_with_score[:DENSE_MAX_PROPOSALS]
    half_width = stages.window_width // 2
    half_height = stages.window_height // 2
    candidates = []
    for rank, (score, center_x, center_y, features, patch_vectors) in enumerate(
        candidates_with_score, start=1
    ):
        candidates.append(
            patches.CandidatePatch(
                measure=measure_index,
                id=f"d{rank:03d}",
                rank=rank,
                center_x=center_x,
                center_y=center_y,
                bbox=(
                    round(center_x) - half_width,
                    round(center_y) - half_height,
                    round(center_x) - half_width + stages.window_width,
                    round(center_y) - half_height + stages.window_height,
                ),
                detector_score=score,
                patches={
                    **patch_vectors,
                    "dense_features": tuple(float(features[name]) for name in DENSE_FEATURES),
                },
            )
        )
    if not candidates:
        raise ValueError(f"Dense proposal generation produced no candidates: {source_path}")
    return patches.UnlabeledMeasure(
        measure=measure_index,
        source_image=source_path,
        source_sha256=_sha256(source_path),
        staff_lines=staff_lines,
        staff_spacing=float(stages.staff_spacing),
        candidates=tuple(candidates),
    )


def _staff_grid_local_maxima(
    stages: detector.StaffGridStages,
) -> list[tuple[float, float, float]]:
    radius = max(1, round(stages.staff_spacing * LOCAL_MAX_RADIUS_SPACES))
    by_row: dict[float, list[tuple[float, float]]] = {}
    for center_x, center_y, score in stages.scored_windows:
        by_row.setdefault(center_y, []).append((center_x, score))
    peaks = []
    for center_y, rows in sorted(by_row.items()):
        rows.sort()
        for center_x, score in rows:
            if score < MIN_GRID_SCORE:
                continue
            neighbors = [
                (other_x, other_score)
                for other_x, other_score in rows
                if abs(other_x - center_x) <= radius
            ]
            best_score = max(other_score for _, other_score in neighbors)
            if score < best_score:
                continue
            if any(
                other_score == score and other_x < center_x for other_x, other_score in neighbors
            ):
                continue
            peaks.append((center_x, center_y, score))
    return peaks


def _binary_patch_features(vector: Sequence[float]) -> dict[str, float]:
    width = patches.PATCH_WIDTH
    height = patches.PATCH_HEIGHT
    if len(vector) != width * height:
        raise ValueError("Unexpected normalized patch size")

    def density(left: int, top: int, right: int, bottom: int) -> float:
        values = [vector[y * width + x] for y in range(top, bottom) for x in range(left, right)]
        return sum(values) / max(1, len(values))

    total = density(0, 0, width, height)
    center = density(width // 4, height // 4, width - width // 4, height - height // 4)
    left = density(0, 0, width // 2, height)
    right = density(width - width // 2, 0, width, height)
    top = density(0, 0, width, height // 2)
    bottom = density(0, height - height // 2, width, height)
    return {
        "patch_ink_density": total,
        "patch_center_density": center,
        "patch_horizontal_balance": abs(left - right),
        "patch_vertical_balance": abs(top - bottom),
    }


def _attach_review(
    request: Mapping[str, Any],
    measure: patches.UnlabeledMeasure,
    *,
    reviews_dir: Path,
    slug: str,
) -> TrainingExample:
    identity = request["identity"]
    system_index = int(identity["system_index"])
    measure_index = int(identity["system_measure_index"])
    key = _target_key(system_index, measure_index)
    review_path = reviews_dir / (
        f"{slug}_system_{system_index:03d}_measure_{measure_index:03d}.json"
    )
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    review_identity = payload.get("identity", {})
    if (
        int(review_identity.get("system_index", -1)) != system_index
        or int(review_identity.get("system_measure_index", -1)) != measure_index
    ):
        raise ValueError(f"Review identity mismatch: {review_path}")
    if payload.get("source", {}).get("image_sha256") != measure.source_sha256:
        raise ValueError(f"Review image hash mismatch: {review_path}")

    note_payloads = payload.get("final_noteheads")
    if not isinstance(note_payloads, list) or not note_payloads:
        raise ValueError(f"Review has no final noteheads: {review_path}")
    notes = []
    for index, item in enumerate(note_payloads, start=1):
        pitch = str(item["pitch"])
        notes.append(
            ReviewNote(
                id=f"n{index:03d}",
                order=int(item.get("order", index)),
                x=float(item["center"]["x"]),
                annotated_y=float(item["center"]["y"]),
                pitch=pitch,
                pitch_row_y=_pitch_row_y(pitch, measure.staff_lines),
                source_kind=str(item.get("source", {}).get("kind", "unknown")),
            )
        )
    notes.sort(key=lambda note: note.order)
    matched = _match_dense_candidates(notes, measure)
    labels = {
        candidate.id: int(candidate.id in matched.values()) for candidate in measure.candidates
    }
    rows = tuple(
        patches.LabeledCandidate(candidate=candidate, label=labels[candidate.id])
        for candidate in measure.candidates
    )
    labeled_measure = patches.MeasureData(
        measure=measure.measure,
        source_image=measure.source_image,
        source_sha256=measure.source_sha256,
        review_path=review_path,
        review_sha256=_sha256(review_path),
        staff_lines=measure.staff_lines,
        staff_spacing=measure.staff_spacing,
        rows=rows,
    )
    pitch_vectors = _pitch_vectors(measure.source_image, measure.staff_lines, notes)
    return TrainingExample(
        key=key,
        request=dict(request),
        measure=labeled_measure,
        notes=tuple(notes),
        matched_candidate_ids=frozenset(matched.values()),
        unmatched_note_ids=tuple(note.id for note in notes if note.id not in matched),
        pitch_vectors=pitch_vectors,
    )


def _match_dense_candidates(
    notes: Sequence[ReviewNote],
    measure: patches.UnlabeledMeasure,
) -> dict[str, str]:
    used: set[str] = set()
    matches: dict[str, str] = {}
    x_tolerance = measure.staff_spacing * TARGET_X_TOLERANCE_SPACES
    y_tolerance = max(1.0, measure.staff_spacing * 0.12)
    for note in notes:
        candidates = [
            candidate
            for candidate in measure.candidates
            if candidate.id not in used
            and abs(candidate.center_x - note.x) <= x_tolerance
            and abs(candidate.center_y - note.pitch_row_y) <= y_tolerance
        ]
        if not candidates:
            continue
        candidate = min(
            candidates,
            key=lambda item: (
                abs(item.center_x - note.x),
                -item.detector_score,
                item.rank,
            ),
        )
        matches[note.id] = candidate.id
        used.add(candidate.id)
    return matches


def _out_of_fold_scores(
    examples: Sequence[TrainingExample],
) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    for heldout in examples:
        training = [example for example in examples if example.key != heldout.key]
        scorer = _fit_logistic_scorer(training)
        for row in heldout.measure.rows:
            scores[(heldout.key, row.id)] = scorer.score(row.candidate)
    return scores


def _fit_logistic_scorer(examples: Sequence[TrainingExample]) -> LogisticScorer:
    rows = [row for example in examples for row in example.measure.rows]
    labels = [row.label for row in rows]
    if not rows or set(labels) != {0, 1}:
        raise ValueError("Dense logistic fit requires positive and negative proposals")
    matrix = [row.candidate.patches["dense_features"] for row in rows]
    means = tuple(
        statistics.mean(vector[index] for vector in matrix) for index in range(len(DENSE_FEATURES))
    )
    scales = tuple(
        max(
            1e-6,
            math.sqrt(statistics.mean((vector[index] - means[index]) ** 2 for vector in matrix)),
        )
        for index in range(len(DENSE_FEATURES))
    )
    normalized = [
        tuple(
            (value - mean) / scale for value, mean, scale in zip(vector, means, scales, strict=True)
        )
        for vector in matrix
    ]
    positives = sum(labels)
    negatives = len(labels) - positives
    positive_weight = len(labels) / (2 * positives)
    negative_weight = len(labels) / (2 * negatives)
    weights = [0.0] * len(DENSE_FEATURES)
    intercept = math.log(positives / negatives)
    learning_rate = 0.08
    l2 = 0.08
    for _ in range(LOGISTIC_ITERATIONS):
        gradients = [0.0] * len(weights)
        intercept_gradient = 0.0
        for vector, label in zip(normalized, labels, strict=True):
            probability = _sigmoid(
                intercept
                + sum(weight * value for weight, value in zip(weights, vector, strict=True))
            )
            sample_weight = positive_weight if label else negative_weight
            error = (probability - label) * sample_weight
            intercept_gradient += error
            for index, value in enumerate(vector):
                gradients[index] += error * value
        count = len(rows)
        intercept -= learning_rate * intercept_gradient / count
        for index in range(len(weights)):
            weights[index] -= learning_rate * (gradients[index] / count + l2 * weights[index])
    return LogisticScorer(
        means=means,
        scales=scales,
        weights=tuple(weights),
        intercept=intercept,
    )


def _select_training_threshold(
    examples: Sequence[TrainingExample],
    scores: Mapping[tuple[str, str], float],
    *,
    nms_x_spaces: float,
    minimum_selected_count: int,
    maximum_selected_count: int,
) -> tuple[float, dict[str, Any]]:
    values = sorted({float(value) for value in scores.values()})
    if len(values) > 320:
        values = sorted({values[round(index * (len(values) - 1) / 319)] for index in range(320)})
    thresholds = [values[-1] + 1e-9, *reversed(values)]
    best_threshold = thresholds[0]
    best_metrics: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    for threshold in thresholds:
        metrics = _aggregate_selection_metrics(
            [
                _selection_metrics(
                    example,
                    _select_candidates(
                        example.measure,
                        {row.id: scores[(example.key, row.id)] for row in example.measure.rows},
                        threshold=threshold,
                        nms_x_spaces=nms_x_spaces,
                        minimum_selected_count=minimum_selected_count,
                        maximum_selected_count=maximum_selected_count,
                    ),
                )
                for example in examples
            ]
        )
        key = (
            metrics["f1"],
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


def _select_training_configuration(
    examples: Sequence[TrainingExample],
    scores: Mapping[tuple[str, str], float],
) -> dict[str, Any]:
    rows = []
    for nms_x_spaces in NMS_X_SPACES_GRID:
        for minimum_selected_count in MIN_SELECTED_COUNT_GRID:
            for maximum_selected_count in MAX_SELECTED_COUNT_GRID:
                if minimum_selected_count > maximum_selected_count:
                    continue
                threshold, metrics = _select_training_threshold(
                    examples,
                    scores,
                    nms_x_spaces=nms_x_spaces,
                    minimum_selected_count=minimum_selected_count,
                    maximum_selected_count=maximum_selected_count,
                )
                rows.append(
                    {
                        "nms_x_spaces": nms_x_spaces,
                        "minimum_selected_count": minimum_selected_count,
                        "maximum_selected_count": maximum_selected_count,
                        "threshold": threshold,
                        "metrics": metrics,
                    }
                )
    winner = max(
        rows,
        key=lambda row: (
            float(row["metrics"]["f1"]),
            float(row["metrics"]["recall"]),
            float(row["metrics"]["precision"]),
            int(row["metrics"]["exact_measures"]),
            int(row["minimum_selected_count"]),
            -int(row["maximum_selected_count"]),
            -float(row["nms_x_spaces"]),
        ),
    )
    return {
        **winner,
        "selection_basis": "training-review out-of-fold notehead F1 only",
        "searched": [
            {
                "nms_x_spaces": row["nms_x_spaces"],
                "minimum_selected_count": row["minimum_selected_count"],
                "maximum_selected_count": row["maximum_selected_count"],
                "threshold": round(float(row["threshold"]), 9),
                "f1": row["metrics"]["f1"],
                "precision": row["metrics"]["precision"],
                "recall": row["metrics"]["recall"],
                "exact_measures": row["metrics"]["exact_measures"],
            }
            for row in rows
        ],
    }


def _select_candidates(
    measure: patches.UnlabeledMeasure | patches.MeasureData,
    scores: Mapping[str, float],
    *,
    threshold: float,
    nms_x_spaces: float,
    minimum_selected_count: int,
    maximum_selected_count: int,
) -> list[patches.CandidatePatch]:
    candidates = [
        row.candidate if isinstance(row, patches.LabeledCandidate) else row
        for row in (
            measure.rows if isinstance(measure, patches.MeasureData) else measure.candidates
        )
    ]
    ranked = sorted(
        candidates,
        key=lambda candidate: (-scores[candidate.id], candidate.rank, candidate.id),
    )
    selected = []
    spacing = float(measure.staff_spacing)
    for candidate in (item for item in ranked if scores[item.id] >= threshold):
        if any(
            abs(candidate.center_x - other.center_x) < spacing * nms_x_spaces for other in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= maximum_selected_count:
            break
    selected_ids = {candidate.id for candidate in selected}
    if len(selected) < minimum_selected_count:
        for candidate in ranked:
            if candidate.id in selected_ids:
                continue
            if any(
                abs(candidate.center_x - other.center_x) < spacing * nms_x_spaces
                for other in selected
            ):
                continue
            selected.append(candidate)
            selected_ids.add(candidate.id)
            if len(selected) >= min(minimum_selected_count, maximum_selected_count):
                break
    return selected


def _selection_metrics(
    example: TrainingExample,
    selected: Sequence[patches.CandidatePatch],
) -> dict[str, Any]:
    selected_ids = {candidate.id for candidate in selected}
    tp = len(selected_ids & example.matched_candidate_ids)
    fp = len(selected_ids) - tp
    fn = example.true_note_count - tp
    return {
        "key": example.key,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "selected_count": len(selected_ids),
        "truth_count": example.true_note_count,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _f1(tp, fp, fn),
        "exact_set": fp == 0 and fn == 0,
    }


def _aggregate_selection_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "selected_count": tp + fp,
        "truth_count": tp + fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _f1(tp, fp, fn),
        "exact_measures": sum(bool(row["exact_set"]) for row in rows),
        "measure_count": len(rows),
        "per_measure": list(rows),
    }


def _proposal_recall(examples: Sequence[TrainingExample]) -> dict[str, Any]:
    matched = sum(len(example.matched_candidate_ids) for example in examples)
    truth = sum(example.true_note_count for example in examples)
    return {"matched": matched, "truth": truth, "recall": _ratio(matched, truth)}


def _pitch_vectors(
    image_path: Path,
    staff_lines: Sequence[int],
    notes: Sequence[ReviewNote],
) -> tuple[tuple[float, ...], ...]:
    with Image.open(image_path) as opened:
        grayscale = opened.convert("L")
    spacing = rhythm.staff_spacing(staff_lines)
    suppressed = patches._suppress_staff_lines(
        grayscale,
        staff_lines=staff_lines,
        staff_spacing=spacing,
    )
    threshold = detector.estimate_ink_threshold(suppressed)
    return tuple(
        _accidental_patch(
            suppressed,
            x=note.x,
            y=note.pitch_row_y,
            staff_spacing=spacing,
            threshold=threshold,
        )
        for note in notes
    )


def _accidental_patch(
    image: Image.Image,
    *,
    x: float,
    y: float,
    staff_spacing: float,
    threshold: int,
) -> tuple[float, ...]:
    left = round(x - staff_spacing * 1.35)
    right = round(x - staff_spacing * 0.10)
    top = round(y - staff_spacing * 1.15)
    bottom = round(y + staff_spacing * 1.15)
    width = max(1, right - left)
    height = max(1, bottom - top)
    canvas = Image.new("L", (width, height), 255)
    source_box = (
        max(0, left),
        max(0, top),
        min(image.width, right),
        min(image.height, bottom),
    )
    if source_box[2] > source_box[0] and source_box[3] > source_box[1]:
        crop = image.crop(source_box)
        canvas.paste(crop, (source_box[0] - left, source_box[1] - top))
    resized = canvas.resize(
        (ACCIDENTAL_PATCH_WIDTH, ACCIDENTAL_PATCH_HEIGHT),
        resample=Image.Resampling.NEAREST,
    )
    return tuple(1.0 if value <= threshold else 0.0 for value in resized.get_flattened_data())


def _evaluate_pitch_methods(examples: Sequence[TrainingExample]) -> dict[str, Any]:
    key_only_correct = 0
    knn_correct = 0
    total = 0
    per_measure = []
    for heldout in examples:
        training = [example for example in examples if example.key != heldout.key]
        model = _fit_accidental_model(training)
        heldout_key_correct = 0
        heldout_knn_correct = 0
        for note, vector in zip(heldout.notes, heldout.pitch_vectors, strict=True):
            key_hint = heldout.request.get("allowed_context", {}).get("key_hint")
            base = composed._pitch_for_y(
                note.pitch_row_y,
                heldout.measure.staff_lines,
                key_hint=key_hint,
            )
            key_match = rhythm.pitch_to_midi(base) == rhythm.pitch_to_midi(note.pitch)
            predicted = _pitch_with_delta(base, model.predict_delta(vector))
            knn_match = rhythm.pitch_to_midi(predicted) == rhythm.pitch_to_midi(note.pitch)
            heldout_key_correct += int(key_match)
            heldout_knn_correct += int(knn_match)
        total += heldout.true_note_count
        key_only_correct += heldout_key_correct
        knn_correct += heldout_knn_correct
        per_measure.append(
            {
                "key": heldout.key,
                "notes": heldout.true_note_count,
                "key_only_correct": heldout_key_correct,
                "accidental_knn_correct": heldout_knn_correct,
            }
        )
    key_accuracy = _ratio(key_only_correct, total)
    knn_accuracy = _ratio(knn_correct, total)
    knn_gain = knn_accuracy - key_accuracy
    selected = "accidental_knn" if knn_gain >= MINIMUM_ACCIDENTAL_OOF_GAIN else "key_signature_only"
    return {
        "selected_method": selected,
        "selection_rule": (
            "Use accidental_knn only when measure-level OOF exact-pitch accuracy improves "
            f"by at least {MINIMUM_ACCIDENTAL_OOF_GAIN:.3f}."
        ),
        "accidental_knn_gain": knn_gain,
        "key_signature_only": {
            "correct": key_only_correct,
            "total": total,
            "accuracy": key_accuracy,
        },
        "accidental_knn": {"correct": knn_correct, "total": total, "accuracy": knn_accuracy},
        "per_measure": per_measure,
    }


def _fit_accidental_model(examples: Sequence[TrainingExample]) -> AccidentalKNN:
    samples = []
    for example in examples:
        key_hint = example.request.get("allowed_context", {}).get("key_hint")
        for note, vector in zip(example.notes, example.pitch_vectors, strict=True):
            base = composed._pitch_for_y(
                note.pitch_row_y,
                example.measure.staff_lines,
                key_hint=key_hint,
            )
            delta = rhythm.pitch_to_midi(note.pitch) - rhythm.pitch_to_midi(base)
            if delta not in (-1, 0, 1):
                raise ValueError(f"Unsupported accidental delta {delta}: {note.pitch} vs {base}")
            samples.append(
                PitchSample(
                    key=example.key,
                    pitch=note.pitch,
                    base_pitch=base,
                    delta=delta,
                    vector=vector,
                )
            )
    return AccidentalKNN(tuple(samples))


def _build_pitch_predictor(
    model: AccidentalKNN | None,
) -> composed.PitchPredictor:
    def predict(
        candidate: patches.CandidatePatch,
        request: Mapping[str, Any],
        image: Image.Image,
    ) -> str:
        staff_lines = [int(value) for value in request["staff_geometry"]["raw_staff_lines_y_px"]]
        key_hint = visual_key_context.key_hint_for_candidate(
            request,
            candidate_x_px=candidate.center_x,
        )
        base = composed._pitch_for_y(candidate.center_y, staff_lines, key_hint=key_hint)
        if model is None:
            return base
        spacing = rhythm.staff_spacing(staff_lines)
        suppressed = patches._suppress_staff_lines(
            image.convert("L"),
            staff_lines=staff_lines,
            staff_spacing=spacing,
        )
        threshold = detector.estimate_ink_threshold(suppressed)
        vector = _accidental_patch(
            suppressed,
            x=candidate.center_x,
            y=candidate.center_y,
            staff_spacing=spacing,
            threshold=threshold,
        )
        return _pitch_with_delta(base, model.predict_delta(vector))

    return predict


def _pitch_with_delta(base_pitch: str, delta: int) -> str:
    letter, accidental, octave = _parse_pitch(base_pitch)
    accidental_value = {"b": -1, "": 0, "#": 1}[accidental] + delta
    if accidental_value not in (-1, 0, 1):
        return base_pitch
    suffix = {-1: "b", 0: "", 1: "#"}[accidental_value]
    return f"{letter}{suffix}{octave}"


def _pitch_row_y(pitch: str, staff_lines: Sequence[int]) -> float:
    letter, _, octave = _parse_pitch(pitch)
    letters = "CDEFGAB"
    top_number = 5 * 7 + letters.index("F")
    pitch_number = octave * 7 + letters.index(letter)
    spacing = rhythm.staff_spacing(staff_lines)
    return float(staff_lines[0]) + (top_number - pitch_number) * (spacing / 2.0)


def _parse_pitch(pitch: str) -> tuple[str, str, int]:
    match = _PITCH_PATTERN.fullmatch(pitch)
    if match is None:
        raise ValueError(f"Unsupported pitch: {pitch!r}")
    return match.group(1).upper(), match.group(2), int(match.group(3))


def _historical_system8_metrics(benchmark_dir: Path) -> dict[str, Any] | None:
    historical_dir = benchmark_dir / "composed_melody_chain/threshold_selector/validation"
    prediction_path = historical_dir / "predictions.jsonl"
    freeze_path = historical_dir / "freeze.json"
    if not prediction_path.exists() or not freeze_path.exists():
        return None
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    artifact = next(
        (
            item
            for item in freeze.get("artifacts", [])
            if Path(str(item.get("path", ""))).name == prediction_path.name
        ),
        None,
    )
    if artifact is None or str(artifact.get("sha256")) != _sha256(prediction_path):
        raise ValueError("Historical system-8 baseline prediction hash is not frozen")
    predictions = [
        row for row in _read_jsonl(prediction_path) if int(row["identity"]["system_index"]) == 8
    ]
    truth = [
        row
        for row in _read_jsonl(benchmark_dir / "validation/truth.jsonl")
        if int(row["identity"]["system_index"]) == 8
    ]
    return benchmark.evaluate_predictions(truth, predictions)


def _validation_gate(
    summary: Mapping[str, Any],
    historical: Mapping[str, Any] | None,
) -> dict[str, Any]:
    note_f1 = float(summary["note_f1"])
    historical_f1 = float(historical["summary"]["note_f1"]) if historical is not None else None
    absolute_pass = note_f1 > CURRENT_COMPOSED_NOTE_F1
    same_target_pass = historical_f1 is None or note_f1 > historical_f1
    return {
        "status": "pass" if absolute_pass and same_target_pass else "fail",
        "passed": absolute_pass and same_target_pass,
        "rule": (
            "system-8 strict note F1 must exceed 0.264463 and the frozen historical "
            "threshold-selector score on the same targets"
        ),
        "observed_note_f1": note_f1,
        "absolute_baseline": CURRENT_COMPOSED_NOTE_F1,
        "historical_same_target_note_f1": historical_f1,
        "absolute_pass": absolute_pass,
        "same_target_pass": same_target_pass,
    }


def _select_requests(
    requests: Sequence[dict[str, Any]],
    targets: Sequence[tuple[int, int]],
) -> list[dict[str, Any]]:
    by_key = {
        (
            int(request["identity"]["system_index"]),
            int(request["identity"]["system_measure_index"]),
        ): request
        for request in requests
    }
    missing = [target for target in targets if target not in by_key]
    if missing:
        raise ValueError(f"Missing benchmark requests: {missing}")
    return [by_key[target] for target in targets]


def _identity_key(identity: Mapping[str, Any]) -> str:
    return _target_key(
        int(identity["system_index"]),
        int(identity["system_measure_index"]),
    )


def _target_key(system_index: int, measure_index: int) -> str:
    return f"S{system_index:02d}M{measure_index:02d}"


def _write_markdown(report: Mapping[str, Any], path: Path) -> None:
    training = report["training"]
    validation = report["validation"]
    summary = validation["metrics"]["summary"]
    historical = validation["historical_threshold_selector_same_targets"]
    historical_f1 = historical["summary"]["note_f1"] if historical else "unavailable"
    lines = [
        "# Review-Augmented Dense Selector",
        "",
        "## Protocol",
        "",
        "- Train and select thresholds only from promoted S1 and S7 reviews.",
        "- Freeze S8 predictions before opening S8 truth.",
        "- Never access consumed system 3.",
        "",
        "## Training",
        "",
        f"- Reviewed noteheads: {training['noteheads']}",
        f"- Dense proposal recall: {training['dense_proposal_recall']['recall']:.3f}",
        f"- OOF selector F1: {training['selector_oof']['metrics']['f1']:.3f}",
        f"- OOF pitch method: {training['pitch_oof']['selected_method']}",
        "",
        "## Frozen System 8",
        "",
        f"- Strict note F1: {summary['note_f1']:.3f}",
        f"- Ordered pitch accuracy: {summary['ordered_pitch_accuracy']:.3f}",
        f"- Duration accuracy: {summary['ordered_duration_accuracy']:.3f}",
        f"- Rest F1: {summary['rest_f1']:.3f}",
        f"- Exact measures: {summary['exact_measures']}/{summary['targets']}",
        f"- Historical same-target note F1: {historical_f1}",
        f"- Gate: {report['validation_gate']['status']}",
        "",
        "The result is spike evidence only. A passing system-8 result still requires a new "
        "independent heldout score before pipeline integration.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _mean_squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Feature vectors must have equal lengths")
    return sum((a - b) ** 2 for a, b in zip(left, right, strict=True)) / max(1, len(left))


def _sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    return _ratio(2 * tp, 2 * tp + fp + fn)


if __name__ == "__main__":
    raise SystemExit(main())
