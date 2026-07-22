"""Compose automatic notehead selection with visual rhythm/rest parsing.

The experiment has three explicit data boundaries:

1. promoted S1M1-4 candidate decisions are training labels only;
2. every split's candidate, anchor, and event predictions are frozen to disk;
3. benchmark truth is loaded only after the split freeze manifest is verified.

Validation must beat both predeclared full-system VLM S7 baselines before the
heldout S3 request or truth files are opened.

Example:
    uv run python scripts/experiments/spike_composed_melody_chain.py out
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
from typing import Any, Callable, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_vlm_melody_event_benchmark as benchmark  # noqa: E402
from scripts.experiments import spike_anchored_rhythm_parser as rhythm  # noqa: E402
from scripts.experiments import spike_notehead_patch_templates as selector  # noqa: E402

DEFAULT_OUT_DIR = REPO_ROOT / "out"
DEFAULT_SLUG = selector.DEFAULT_SLUG
DEFAULT_REVIEWS_DIR = REPO_ROOT / "tests/fixtures/vlm_melody/notehead_reviews"
DEFAULT_OUTPUT_SUBDIR = Path("composed_melody_chain")
DEVELOPMENT_SYSTEM = 1
DEVELOPMENT_MEASURES = (1, 2, 3, 4)
SELECTOR_PATCH_ID = "binary_raw"
SELECTOR_SCORER_KIND = "class_knn3"
SELECTOR_METHOD_ID = f"{SELECTOR_SCORER_KIND}__{SELECTOR_PATCH_ID}"
AUTOMATIC_COUNT_SELECTOR = "automatic_count_selector"
THRESHOLD_SELECTOR = "threshold_selector"
VALIDATION_NOTE_F1_BASELINE = 0.17284
VALIDATION_ORDERED_PITCH_BASELINE = 0.130435

TruthLoader = Callable[[Path], list[dict[str, Any]]]
PitchPredictor = Callable[
    [selector.CandidatePatch, Mapping[str, Any], Image.Image],
    str,
]


@dataclass(frozen=True)
class SelectorModel:
    scorer: selector.PatchScorer
    learned_count: int
    probability_center: float
    probability_scale: float
    training_measures: tuple[int, ...]
    training_positive_count: int
    learned_threshold: float | None = None
    threshold_training_metrics: dict[str, Any] | None = None

    def rank(
        self,
        measure: selector.UnlabeledMeasure,
        *,
        selection_mode: str = AUTOMATIC_COUNT_SELECTOR,
    ) -> tuple[list[dict[str, Any]], list[selector.CandidatePatch]]:
        if selection_mode not in (AUTOMATIC_COUNT_SELECTOR, THRESHOLD_SELECTOR):
            raise ValueError(f"Unsupported selector mode: {selection_mode}")
        if selection_mode == THRESHOLD_SELECTOR and self.learned_threshold is None:
            raise ValueError("Threshold selection requires a fitted training threshold")
        scored = [
            (candidate, float(self.scorer.score(candidate))) for candidate in measure.candidates
        ]
        scored.sort(key=lambda item: (-item[1], item[0].rank, item[0].id))
        selected = [
            candidate
            for rank, (candidate, score) in enumerate(scored, start=1)
            if self._selected(rank=rank, score=score, selection_mode=selection_mode)
        ]
        rows = []
        for rank, (candidate, score) in enumerate(scored, start=1):
            probability = self.probability(score)
            rows.append(
                {
                    "candidate_id": candidate.id,
                    "detector_rank": candidate.rank,
                    "selection_rank": rank,
                    "score": round(score, 9),
                    "probability": round(probability, 9),
                    "selected": self._selected(
                        rank=rank,
                        score=score,
                        selection_mode=selection_mode,
                    ),
                    "selection_mode": selection_mode,
                    "selected_by_learned_count": rank <= self.learned_count,
                    "selected_by_learned_threshold": (
                        self.learned_threshold is not None and score >= self.learned_threshold
                    ),
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

    def _selected(self, *, rank: int, score: float, selection_mode: str) -> bool:
        if selection_mode == AUTOMATIC_COUNT_SELECTOR:
            return rank <= self.learned_count
        assert self.learned_threshold is not None
        return score >= self.learned_threshold

    def probability(self, score: float) -> float:
        z = max(-60.0, min(60.0, (score - self.probability_center) / self.probability_scale))
        return 1.0 / (1.0 + math.exp(-z))


@dataclass
class ComposedMeasure:
    request: dict[str, Any]
    image_path: Path
    image: Image.Image
    staff_spacing: float
    candidate_predictions: list[dict[str, Any]]
    anchors: list[dict[str, Any]]
    groups: list[dict[str, Any]]
    anchor_features: list[dict[str, Any]]
    rest_features: list[dict[str, Any]]
    visual_symbols: list[dict[str, Any]]
    decoded_symbols: list[dict[str, Any]]
    prediction: dict[str, Any]


@dataclass(frozen=True)
class SplitArtifacts:
    split: str
    prediction_path: Path
    inference_path: Path
    freeze_path: Path
    overlay_paths: tuple[Path, ...]
    freeze_sha256: str


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
    print(f"automatic-count validation gate: {report['validation_gate']['status']}")
    threshold = report["threshold_selector"]
    print(f"threshold-selector validation gate: {threshold['validation_gate']['status']}")
    print(f"threshold-selector heldout: {threshold['heldout']['status']}")
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
    truth_loader: TruthLoader | None = None,
) -> dict[str, Any]:
    truth_loader = truth_loader or _read_jsonl
    benchmark_dir = out_dir / slug / "vlm_melody_event_benchmark"
    output_dir = output_dir or benchmark_dir / DEFAULT_OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    development_requests_path = benchmark_dir / "development/requests.jsonl"
    development_requests = [
        row
        for row in _read_jsonl(development_requests_path)
        if int(row["identity"]["system_index"]) == DEVELOPMENT_SYSTEM
        and int(row["identity"]["system_measure_index"]) in DEVELOPMENT_MEASURES
    ]
    _validate_request_slice(
        development_requests,
        split="development",
        expected_count=len(DEVELOPMENT_MEASURES),
    )

    # Candidate generation and patch extraction complete before any review fixture is opened.
    development_unlabeled = _load_unlabeled_requests(out_dir, development_requests, slug=slug)
    development_labeled = [
        _attach_training_labels(item, reviews_dir=reviews_dir, slug=slug)
        for item in development_unlabeled
    ]

    development_composed: list[ComposedMeasure] = []
    threshold_development_composed: list[ComposedMeasure] = []
    development_folds = []
    threshold_development_folds = []
    for request, heldout_unlabeled in zip(development_requests, development_unlabeled, strict=True):
        heldout_measure = heldout_unlabeled.measure
        training = [item for item in development_labeled if item.measure != heldout_measure]
        model = fit_selector(training)
        composed = compose_measure(
            request,
            heldout_unlabeled,
            model,
            out_dir=out_dir,
        )
        threshold_composed = compose_measure(
            request,
            heldout_unlabeled,
            model,
            out_dir=out_dir,
            selection_mode=THRESHOLD_SELECTOR,
        )
        development_composed.append(composed)
        threshold_development_composed.append(threshold_composed)
        development_folds.append(
            {
                "heldout_measure": heldout_measure,
                "training_measures": list(model.training_measures),
                "learned_count": model.learned_count,
                "selected_candidate_ids": [
                    row["candidate_id"] for row in composed.candidate_predictions if row["selected"]
                ],
            }
        )
        threshold_development_folds.append(
            {
                "heldout_measure": heldout_measure,
                "training_measures": list(model.training_measures),
                "learned_score_threshold": round(_required_threshold(model), 9),
                "learned_probability_threshold": round(
                    model.probability(_required_threshold(model)), 9
                ),
                "training_threshold_metrics": model.threshold_training_metrics,
                "selected_count": len(threshold_composed.anchors),
                "selected_candidate_ids": [
                    row["candidate_id"]
                    for row in threshold_composed.candidate_predictions
                    if row["selected"]
                ],
            }
        )
    development_artifacts = freeze_split_predictions(
        split="development",
        composed=development_composed,
        output_dir=output_dir,
        requests_path=development_requests_path,
        training=development_labeled,
        selector_mode="leave_one_measure_out",
    )
    threshold_development_artifacts = freeze_split_predictions(
        split="development",
        composed=threshold_development_composed,
        output_dir=output_dir / THRESHOLD_SELECTOR,
        requests_path=development_requests_path,
        training=development_labeled,
        selector_mode="training_only_inner_loocv_threshold",
    )
    # Both development arms are frozen before development truth is loaded.
    development_metrics = evaluate_frozen_split(
        benchmark_dir,
        artifacts=development_artifacts,
        truth_loader=truth_loader,
    )
    threshold_development_metrics = evaluate_frozen_split(
        benchmark_dir,
        artifacts=threshold_development_artifacts,
        truth_loader=truth_loader,
    )

    full_model = fit_selector(development_labeled)
    validation_requests_path = benchmark_dir / "validation/requests.jsonl"
    validation_requests = _read_jsonl(validation_requests_path)
    _validate_request_slice(validation_requests, split="validation", expected_count=14)
    validation_unlabeled = _load_unlabeled_requests(out_dir, validation_requests, slug=slug)
    validation_composed = [
        compose_measure(request, item, full_model, out_dir=out_dir)
        for request, item in zip(validation_requests, validation_unlabeled, strict=True)
    ]
    threshold_validation_composed = [
        compose_measure(
            request,
            item,
            full_model,
            out_dir=out_dir,
            selection_mode=THRESHOLD_SELECTOR,
        )
        for request, item in zip(validation_requests, validation_unlabeled, strict=True)
    ]
    validation_artifacts = freeze_split_predictions(
        split="validation",
        composed=validation_composed,
        output_dir=output_dir,
        requests_path=validation_requests_path,
        training=development_labeled,
        selector_mode="fit_all_development_reviews",
    )
    threshold_validation_artifacts = freeze_split_predictions(
        split="validation",
        composed=threshold_validation_composed,
        output_dir=output_dir / THRESHOLD_SELECTOR,
        requests_path=validation_requests_path,
        training=development_labeled,
        selector_mode="training_only_loocv_threshold_fit_all_reviews",
    )
    # Both validation arms are frozen before this run invokes the validation truth loader.
    validation_metrics = evaluate_frozen_split(
        benchmark_dir,
        artifacts=validation_artifacts,
        truth_loader=truth_loader,
    )
    threshold_validation_metrics = evaluate_frozen_split(
        benchmark_dir,
        artifacts=threshold_validation_artifacts,
        truth_loader=truth_loader,
    )
    validation_gate = validation_success_gate(validation_metrics["summary"])
    threshold_validation_gate = validation_success_gate(threshold_validation_metrics["summary"])

    heldout: dict[str, Any] = {
        "status": "not_opened_validation_failed",
        "metrics": None,
        "artifacts": None,
    }
    if validation_gate["passed"]:
        heldout["status"] = "not_evaluated_followup_controlled_by_threshold_selector"

    threshold_heldout: dict[str, Any] = {
        "status": "not_opened_threshold_validation_failed",
        "metrics": None,
        "artifacts": None,
        "per_measure_counts": [],
    }
    if threshold_validation_gate["passed"]:
        # This is intentionally the first heldout path access in the experiment.
        heldout_requests_path = benchmark_dir / "heldout/requests.jsonl"
        heldout_requests = _read_jsonl(heldout_requests_path)
        _validate_request_slice(heldout_requests, split="heldout", expected_count=9)
        heldout_unlabeled = _load_unlabeled_requests(out_dir, heldout_requests, slug=slug)
        heldout_composed = [
            compose_measure(
                request,
                item,
                full_model,
                out_dir=out_dir,
                selection_mode=THRESHOLD_SELECTOR,
            )
            for request, item in zip(heldout_requests, heldout_unlabeled, strict=True)
        ]
        threshold_heldout_artifacts = freeze_split_predictions(
            split="heldout",
            composed=heldout_composed,
            output_dir=output_dir / THRESHOLD_SELECTOR,
            requests_path=heldout_requests_path,
            training=development_labeled,
            selector_mode="training_only_loocv_threshold_fit_all_reviews",
        )
        # Heldout predictions and their freeze exist before the first heldout truth read.
        previous_heldout_metrics = _load_previous_heldout_evaluation(
            output_dir,
            artifacts=threshold_heldout_artifacts,
        )
        if previous_heldout_metrics is None:
            threshold_heldout_metrics = evaluate_frozen_split(
                benchmark_dir,
                artifacts=threshold_heldout_artifacts,
                truth_loader=truth_loader,
            )
            evaluation_source = "truth_read_once_after_prediction_freeze"
        else:
            threshold_heldout_metrics = previous_heldout_metrics
            evaluation_source = "reused_matching_one_shot_evaluation_without_truth_read"
        evaluation_path = _write_evaluation_snapshot(
            threshold_heldout_artifacts,
            threshold_heldout_metrics,
        )
        threshold_heldout = {
            "status": "evaluated_once_after_threshold_validation_pass",
            "evaluation_source": evaluation_source,
            "evaluation_snapshot": {
                "path": _display_path(evaluation_path),
                "sha256": _sha256(evaluation_path),
            },
            "metrics": threshold_heldout_metrics,
            "artifacts": _artifact_summary(threshold_heldout_artifacts),
            "per_measure_counts": _per_measure_counts(heldout_composed),
        }

    report_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    report = {
        "schema_version": 2,
        "kind": "composed_automatic_melody_chain_spike",
        "slug": slug,
        "selector": {
            "method_id": SELECTOR_METHOD_ID,
            "selection_mode": "training_median_positive_count",
            "full_training_learned_count": full_model.learned_count,
            "training_measures": list(full_model.training_measures),
            "training_positive_count": full_model.training_positive_count,
            "probability_calibration": (
                "logistic transform centered and scaled from training-labeled scores only; "
                "probability is diagnostic and count selection uses rank"
            ),
        },
        "protocol": {
            "development": "S1M1-4 outer leave-one-measure-out",
            "validation": (
                "fit S1M1-4 reviews; freeze count and threshold S7+S8 predictions; " "then evaluate"
            ),
            "heldout": (
                "open and evaluate S3 once only if both threshold-selector validation " "gates pass"
            ),
        },
        "model_selection_disclosure": {
            "validation_truth_previously_opened_by_original_arm": True,
            "validation_truth_used_for_threshold_fitting": False,
            "threshold_hypothesis_origin": (
                "preregistered after observing that the original automatic-count arm emitted "
                "three notes for every validation measure"
            ),
            "interpretation": "validation-driven model selection; not a fresh validation claim",
            "system3_status_before_threshold_gate": "sealed",
            "system3_status_after_threshold_gate": (
                "evaluated_once_after_frozen_predictions"
                if threshold_validation_gate["passed"]
                else "sealed"
            ),
        },
        "leakage_audit": {
            "training_label_fields": ["candidates[].id", "candidates[].label"],
            "review_anchor_fields_used_for_inference": [],
            "review_pitch_fields_used_for_inference": [],
            "inference_inputs": [
                "cap-24 candidate pixels and geometry",
                "request staff geometry",
                "request allowed_context",
                "raw measure pixels",
            ],
            "forbidden_inference_inputs": [
                "expected note counts",
                "canonical pitch sequences",
                "canonical durations",
                "benchmark truth",
            ],
            "prediction_freeze_required_before_truth_loader": True,
            "network_used": False,
            "production_code_changed": False,
        },
        "development": {
            "folds": development_folds,
            "metrics": development_metrics,
            "artifacts": _artifact_summary(development_artifacts),
        },
        "validation": {
            "metrics": validation_metrics,
            "artifacts": _artifact_summary(validation_artifacts),
        },
        "validation_gate": validation_gate,
        "heldout": heldout,
        "threshold_selector": {
            "hypothesis": (
                "Select every candidate at or above the class_knn3__binary_raw score threshold "
                "learned only from training-review inner LOOCV scores"
            ),
            "selector": {
                "method_id": SELECTOR_METHOD_ID,
                "selection_mode": THRESHOLD_SELECTOR,
                "full_training_score_threshold": round(_required_threshold(full_model), 9),
                "full_training_probability_threshold": round(
                    full_model.probability(_required_threshold(full_model)), 9
                ),
                "full_training_threshold_metrics": full_model.threshold_training_metrics,
                "validation_expected_counts_used": False,
                "validation_truth_used_for_fit": False,
            },
            "development": {
                "folds": threshold_development_folds,
                "candidate_selection_metrics": _candidate_selection_metrics(
                    threshold_development_composed,
                    development_labeled,
                ),
                "per_measure_counts": _per_measure_counts(threshold_development_composed),
                "metrics": threshold_development_metrics,
                "artifacts": _artifact_summary(threshold_development_artifacts),
            },
            "validation": {
                "per_measure_counts": _per_measure_counts(threshold_validation_composed),
                "metrics": threshold_validation_metrics,
                "artifacts": _artifact_summary(threshold_validation_artifacts),
            },
            "validation_gate": threshold_validation_gate,
            "heldout": threshold_heldout,
        },
        "artifacts": {
            "report_json": _display_path(report_path),
            "report_markdown": _display_path(markdown_path),
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown_report(report, markdown_path)
    return report


def fit_selector(training: Sequence[selector.MeasureData]) -> SelectorModel:
    if not training:
        raise ValueError("Selector training requires at least one reviewed measure")
    learned_threshold, threshold_training_metrics = _fit_training_threshold(training)
    training_rows = [row for measure in training for row in measure.rows]
    scorer = selector._fit_patch_scorer(
        training_rows,
        patch_id=SELECTOR_PATCH_ID,
        scorer_kind=SELECTOR_SCORER_KIND,
    )
    positive_scores = [scorer.score(row.candidate) for row in training_rows if row.label]
    negative_scores = [scorer.score(row.candidate) for row in training_rows if not row.label]
    center = (statistics.median(positive_scores) + statistics.median(negative_scores)) / 2.0
    all_scores = positive_scores + negative_scores
    scale = max(statistics.pstdev(all_scores), 1e-9)
    return SelectorModel(
        scorer=scorer,
        learned_count=selector._learned_training_count(training),
        probability_center=center,
        probability_scale=scale,
        training_measures=tuple(measure.measure for measure in training),
        training_positive_count=sum(measure.positive_count for measure in training),
        learned_threshold=learned_threshold,
        threshold_training_metrics=threshold_training_metrics,
    )


def _fit_training_threshold(
    training: Sequence[selector.MeasureData],
) -> tuple[float, dict[str, Any]]:
    """Calibrate from inner held-measure scores, never final-scorer fit scores."""
    if len(training) < 3:
        raise ValueError("Threshold calibration requires at least three reviewed measures")
    calibration_scores: dict[tuple[int, str], float] = {}
    for calibration_measure in training:
        inner_training_rows = [
            row
            for measure in training
            if measure.measure != calibration_measure.measure
            for row in measure.rows
        ]
        inner_scorer = selector._fit_patch_scorer(
            inner_training_rows,
            patch_id=SELECTOR_PATCH_ID,
            scorer_kind=SELECTOR_SCORER_KIND,
        )
        for row in calibration_measure.rows:
            calibration_scores[(calibration_measure.measure, row.id)] = inner_scorer.score(
                row.candidate
            )
    return selector._select_training_threshold(training, calibration_scores)


def compose_measure(
    request: Mapping[str, Any],
    measure: selector.UnlabeledMeasure,
    model: SelectorModel,
    *,
    out_dir: Path,
    selection_mode: str = AUTOMATIC_COUNT_SELECTOR,
    selector_method_id: str = SELECTOR_METHOD_ID,
    pitch_predictor: PitchPredictor | None = None,
) -> ComposedMeasure:
    """Run automatic inference without accepting review anchors or truth."""
    identity = dict(request["identity"])
    _validate_unlabeled_identity(request, measure)
    image_path = _resolve_request_image(request, out_dir)
    if _sha256(image_path) != str(request["images"]["raw"]["sha256"]):
        raise ValueError(f"Request image hash mismatch: {image_path}")
    if image_path.resolve() != measure.source_image.resolve():
        raise ValueError("Candidate source image does not match benchmark request image")

    candidate_predictions, selected = model.rank(measure, selection_mode=selection_mode)
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    staff_lines = [int(value) for value in request["staff_geometry"]["raw_staff_lines_y_px"]]
    spacing = rhythm.staff_spacing(staff_lines)
    key_hint = request.get("allowed_context", {}).get("key_hint")
    ordered = sorted(selected, key=lambda candidate: (candidate.center_x, candidate.center_y))
    anchors = [
        {
            "order": order,
            "pitch": (
                pitch_predictor(candidate, request, image)
                if pitch_predictor is not None
                else _pitch_for_y(candidate.center_y, staff_lines, key_hint=key_hint)
            ),
            "center": {
                "x": round(candidate.center_x, 3),
                "y": round(candidate.center_y, 3),
            },
            "source": {
                "kind": "automatic_candidate",
                "candidate_id": candidate.id,
                "selector_method": selector_method_id,
                "selection_mode": selection_mode,
            },
        }
        for order, candidate in enumerate(ordered, start=1)
    ]
    groups = rhythm.group_simultaneous_heads(anchors, spacing)
    anchor_features = rhythm.extract_anchor_features(image, anchors, staff_lines)
    rest_features = rhythm.extract_residual_rest_features(image, groups, staff_lines)
    visual_symbols = rhythm.build_visual_symbols(groups, anchor_features, rest_features)
    decoded_symbols, decoder_status = rhythm.decode_meter(
        visual_symbols,
        expected_beats=float(request["allowed_context"]["expected_measure_beats"]),
        allow_pickup=bool(request["allowed_context"].get("allow_pickup", False)),
    )
    prediction = rhythm.symbols_to_hypothesis(
        decoded_symbols,
        identity=identity,
        decoder_status=decoder_status,
    )
    prediction["inference_provenance"] = {
        "notehead_selector": selector_method_id,
        "selection_mode": selection_mode,
        "automatic_anchor_count": len(anchors),
        "review_anchors_used": False,
        "truth_used": False,
    }
    if selection_mode == AUTOMATIC_COUNT_SELECTOR:
        prediction["inference_provenance"]["learned_count"] = model.learned_count
    else:
        threshold = _required_threshold(model)
        prediction["inference_provenance"].update(
            {
                "learned_score_threshold": round(threshold, 9),
                "learned_probability_threshold": round(model.probability(threshold), 9),
                "threshold_fit_from_training_reviews_only": True,
            }
        )
    return ComposedMeasure(
        request=dict(request),
        image_path=image_path,
        image=image,
        staff_spacing=spacing,
        candidate_predictions=candidate_predictions,
        anchors=anchors,
        groups=groups,
        anchor_features=anchor_features,
        rest_features=rest_features,
        visual_symbols=visual_symbols,
        decoded_symbols=decoded_symbols,
        prediction=prediction,
    )


def freeze_split_predictions(
    *,
    split: str,
    composed: Sequence[ComposedMeasure],
    output_dir: Path,
    requests_path: Path,
    training: Sequence[selector.MeasureData],
    selector_mode: str,
    selector_method_id: str = SELECTOR_METHOD_ID,
    training_review_fields: Sequence[str] = (
        "candidates[].id",
        "candidates[].label",
    ),
) -> SplitArtifacts:
    if not composed:
        raise ValueError(f"Cannot freeze empty {split} predictions")
    split_dir = output_dir / split
    prediction_path = split_dir / "predictions.jsonl"
    inference_path = split_dir / "inference.jsonl"
    overlay_dir = split_dir / "overlays"
    freeze_path = split_dir / "freeze.json"
    _write_jsonl(prediction_path, [item.prediction for item in composed])
    _write_jsonl(inference_path, [_inference_record(item) for item in composed])

    overlay_paths = []
    for item in composed:
        identity = item.request["identity"]
        path = overlay_dir / (
            f"system_{int(identity['system_index']):03d}_"
            f"measure_{int(identity['system_measure_index']):03d}.png"
        )
        _write_overlay(item, path)
        overlay_paths.append(path)

    frozen_files = [prediction_path, inference_path, *overlay_paths]
    freeze = {
        "schema_version": 1,
        "status": "frozen_before_truth",
        "split": split,
        "target_count": len(composed),
        "selector": {
            "method_id": selector_method_id,
            "mode": selector_mode,
        },
        "requests": {
            "path": _display_path(requests_path),
            "sha256": _sha256(requests_path),
        },
        "training_reviews": [
            {
                "measure": measure.measure,
                "path": _display_path(measure.review_path),
                "sha256": measure.review_sha256,
                "fields_used": list(training_review_fields),
            }
            for measure in training
        ],
        "artifacts": [
            {"path": _display_path(path), "sha256": _sha256(path)} for path in frozen_files
        ],
    }
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return SplitArtifacts(
        split=split,
        prediction_path=prediction_path,
        inference_path=inference_path,
        freeze_path=freeze_path,
        overlay_paths=tuple(overlay_paths),
        freeze_sha256=_sha256(freeze_path),
    )


def evaluate_frozen_split(
    benchmark_dir: Path,
    *,
    artifacts: SplitArtifacts,
    truth_loader: TruthLoader = lambda path: _read_jsonl(path),
) -> dict[str, Any]:
    """Verify the prediction freeze before invoking the supplied truth loader."""
    freeze = json.loads(artifacts.freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen_before_truth" or freeze.get("split") != artifacts.split:
        raise ValueError(f"Invalid prediction freeze manifest: {artifacts.freeze_path}")
    if _sha256(artifacts.freeze_path) != artifacts.freeze_sha256:
        raise ValueError(f"Prediction freeze manifest changed: {artifacts.freeze_path}")
    for item in freeze.get("artifacts", []):
        path = _path_from_display(str(item["path"]))
        if not path.exists() or _sha256(path) != str(item["sha256"]):
            raise ValueError(f"Frozen prediction artifact changed or is missing: {path}")

    predictions = _read_jsonl(artifacts.prediction_path)
    if len(predictions) != int(freeze["target_count"]):
        raise ValueError("Frozen prediction count does not match freeze manifest")

    # This call is deliberately below every freeze existence/hash assertion.
    truth_rows = truth_loader(benchmark_dir / artifacts.split / "truth.jsonl")
    prediction_keys = {_identity_key(row["identity"]) for row in predictions}
    selected_truth = [
        row for row in truth_rows if _identity_key(row["identity"]) in prediction_keys
    ]
    if len(selected_truth) != len(predictions):
        raise ValueError(
            f"Truth identities do not exactly cover frozen {artifacts.split} predictions"
        )
    return benchmark.evaluate_predictions(selected_truth, predictions)


def validation_success_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    note_f1 = float(summary["note_f1"])
    pitch_accuracy = float(summary["ordered_pitch_accuracy"])
    note_passed = note_f1 > VALIDATION_NOTE_F1_BASELINE
    pitch_passed = pitch_accuracy > VALIDATION_ORDERED_PITCH_BASELINE
    return {
        "status": "pass" if note_passed and pitch_passed else "fail",
        "passed": note_passed and pitch_passed,
        "predeclared_rule": ("strict note F1 > 0.17284 AND ordered pitch accuracy > 0.130435"),
        "note_f1": {
            "observed": note_f1,
            "baseline": VALIDATION_NOTE_F1_BASELINE,
            "comparison": ">",
            "passed": note_passed,
        },
        "ordered_pitch_accuracy": {
            "observed": pitch_accuracy,
            "baseline": VALIDATION_ORDERED_PITCH_BASELINE,
            "comparison": ">",
            "passed": pitch_passed,
        },
    }


def _required_threshold(model: SelectorModel) -> float:
    if model.learned_threshold is None:
        raise ValueError("Selector model has no learned threshold")
    return model.learned_threshold


def _per_measure_counts(composed: Sequence[ComposedMeasure]) -> list[dict[str, Any]]:
    rows = []
    for item in composed:
        provenance = item.prediction["inference_provenance"]
        identity = item.request["identity"]
        row = {
            "system_index": int(identity["system_index"]),
            "system_measure_index": int(identity["system_measure_index"]),
            "global_measure_index": int(identity["global_measure_index"]),
            "predicted_notehead_count": len(item.anchors),
            "selected_candidate_ids": [
                candidate_row["candidate_id"]
                for candidate_row in item.candidate_predictions
                if candidate_row["selected"]
            ],
        }
        if "learned_score_threshold" in provenance:
            row["learned_score_threshold"] = provenance["learned_score_threshold"]
            row["learned_probability_threshold"] = provenance["learned_probability_threshold"]
        rows.append(row)
    return rows


def _candidate_selection_metrics(
    composed: Sequence[ComposedMeasure],
    labeled: Sequence[selector.MeasureData],
) -> dict[str, Any]:
    labels_by_measure = {measure.measure: measure for measure in labeled}
    metric_rows = []
    for item in composed:
        measure_index = int(item.request["identity"]["system_measure_index"])
        measure = labels_by_measure[measure_index]
        selected_ids = {
            str(row["candidate_id"]) for row in item.candidate_predictions if row["selected"]
        }
        selected = [row for row in measure.rows if row.id in selected_ids]
        metric_rows.append(selector._selection_metrics(selected, measure))
    return selector._aggregate_metric_rows(metric_rows)


def _load_previous_heldout_evaluation(
    output_dir: Path,
    *,
    artifacts: SplitArtifacts,
) -> dict[str, Any] | None:
    """Reuse an identical one-shot evaluation; never reopen truth after a changed freeze."""
    snapshot_path = artifacts.freeze_path.parent / "evaluation.json"
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if snapshot.get("freeze_sha256") != artifacts.freeze_sha256:
            raise ValueError("Heldout prediction freeze changed; refusing to reopen heldout truth")
        metrics = snapshot.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"Invalid heldout evaluation snapshot: {snapshot_path}")
        return metrics

    # Migration path for the first follow-up run, whose one-shot result lived in report.json.
    report_path = output_dir / "report.json"
    if not report_path.exists():
        return None
    previous_report = json.loads(report_path.read_text(encoding="utf-8"))
    previous_heldout = previous_report.get("threshold_selector", {}).get("heldout", {})
    metrics = previous_heldout.get("metrics")
    previous_artifacts = previous_heldout.get("artifacts") or {}
    if not isinstance(metrics, dict):
        return None
    if previous_artifacts.get("freeze_sha256") != artifacts.freeze_sha256:
        raise ValueError("Heldout prediction freeze changed; refusing to reopen heldout truth")
    return metrics


def _write_evaluation_snapshot(
    artifacts: SplitArtifacts,
    metrics: Mapping[str, Any],
) -> Path:
    path = artifacts.freeze_path.parent / "evaluation.json"
    payload = {
        "schema_version": 1,
        "kind": "one_shot_heldout_evaluation",
        "split": artifacts.split,
        "freeze_sha256": artifacts.freeze_sha256,
        "metrics": dict(metrics),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _load_unlabeled_requests(
    out_dir: Path,
    requests: Sequence[Mapping[str, Any]],
    *,
    slug: str,
) -> list[selector.UnlabeledMeasure]:
    return [
        selector._load_unlabeled_measure(
            out_dir,
            slug=slug,
            system_index=int(request["identity"]["system_index"]),
            measure=int(request["identity"]["system_measure_index"]),
            max_candidates=selector.DEFAULT_MAX_CANDIDATES,
        )
        for request in requests
    ]


def _attach_training_labels(
    measure: selector.UnlabeledMeasure,
    *,
    reviews_dir: Path,
    slug: str,
) -> selector.MeasureData:
    review_path = reviews_dir / (
        f"{slug}_system_{DEVELOPMENT_SYSTEM:03d}_measure_{measure.measure:03d}.json"
    )
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    identity = payload.get("identity", {})
    if (
        int(identity.get("system_index", -1)) != DEVELOPMENT_SYSTEM
        or int(identity.get("system_measure_index", -1)) != measure.measure
    ):
        raise ValueError(f"Training review identity mismatch: {review_path}")
    source = payload.get("source", {})
    if source.get("image_sha256") != measure.source_sha256:
        raise ValueError(f"Training review image hash mismatch: {review_path}")
    if int(source.get("candidate_cap", -1)) != len(measure.candidates):
        raise ValueError(f"Training review candidate cap mismatch: {review_path}")

    # Only candidate decisions are training labels. final_noteheads and their pitches are ignored.
    decisions = payload.get("candidates")
    if not isinstance(decisions, list):
        raise ValueError(f"Training review has no candidates list: {review_path}")
    by_id = {str(item["id"]): item for item in decisions}
    expected_ids = {candidate.id for candidate in measure.candidates}
    if len(by_id) != len(decisions) or set(by_id) != expected_ids:
        raise ValueError(f"Training review candidate IDs mismatch: {review_path}")
    rows = []
    for candidate in measure.candidates:
        decision = by_id[candidate.id]
        label = decision.get("label")
        if label not in ("accepted", "rejected"):
            raise ValueError(f"Candidate {candidate.id} has invalid training label: {label!r}")
        if selector._review_bbox(decision.get("bbox")) != candidate.bbox:
            raise ValueError(f"Candidate {candidate.id} training geometry is stale: {review_path}")
        rows.append(selector.LabeledCandidate(candidate=candidate, label=int(label == "accepted")))
    return selector.MeasureData(
        measure=measure.measure,
        source_image=measure.source_image,
        source_sha256=measure.source_sha256,
        review_path=review_path,
        review_sha256=_sha256(review_path),
        staff_lines=measure.staff_lines,
        staff_spacing=measure.staff_spacing,
        rows=tuple(rows),
    )


def _inference_record(item: ComposedMeasure) -> dict[str, Any]:
    return {
        "identity": item.request["identity"],
        "source": {
            "image": _display_path(item.image_path),
            "image_sha256": _sha256(item.image_path),
        },
        "training_vs_inference": {
            "training_labels": "reviewed S1 candidate accepted/rejected decisions",
            "inference_anchors": "selected cap-24 candidate centers",
            "review_anchors_used": False,
            "review_pitches_used": False,
            "truth_used": False,
        },
        "candidate_predictions": item.candidate_predictions,
        "automatic_anchors": item.anchors,
        "anchor_features": item.anchor_features,
        "residual_rest_features": item.rest_features,
        "visual_symbols": item.visual_symbols,
        "decoded_symbols": item.decoded_symbols,
        "canonical_prediction": item.prediction,
    }


def _write_overlay(item: ComposedMeasure, path: Path) -> None:
    overlay = item.image.copy()
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    selected_ids = {str(anchor["source"]["candidate_id"]): anchor for anchor in item.anchors}
    radius = max(4, round(item.staff_spacing * 0.28))
    for row in item.candidate_predictions:
        x = float(row["center"]["x"])
        y = float(row["center"]["y"])
        candidate_id = str(row["candidate_id"])
        if candidate_id in selected_ids:
            anchor = selected_ids[candidate_id]
            color = (15, 105, 210)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
            draw.text(
                (x + radius + 2, max(1, y - radius)),
                f"{candidate_id} {anchor['pitch']} p={row['probability']:.2f}",
                fill=color,
                font=font,
            )
        else:
            color = (150, 150, 150)
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=color, width=1)
    for rest in item.rest_features:
        bbox = rest["bbox"]
        draw.rectangle(
            (bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]),
            outline=(155, 45, 175),
            width=2,
        )
    draw.rectangle((0, 0, min(390, overlay.width - 1), 14), fill="white")
    draw.text(
        (3, 2),
        "automatic candidate anchors + visual rhythm/rests (no truth)",
        fill="black",
        font=font,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(path)


def _pitch_for_y(
    y_px: float,
    staff_lines_y_px: Sequence[int],
    *,
    key_hint: Any,
) -> str:
    spacing = rhythm.staff_spacing(staff_lines_y_px)
    half_steps_down = _round_half_away_from_zero(
        (float(y_px) - float(staff_lines_y_px[0])) / (spacing / 2.0)
    )
    letters = ("C", "D", "E", "F", "G", "A", "B")
    top_line_number = 5 * 7 + letters.index("F")
    number = top_line_number - half_steps_down
    pitch = f"{letters[number % 7]}{number // 7}"
    hint = str(key_hint or "")
    for letter in letters:
        if f"{letter}b" in hint and pitch.startswith(letter):
            return f"{letter}b{pitch[1:]}"
        if f"{letter}#" in hint and pitch.startswith(letter):
            return f"{letter}#{pitch[1:]}"
    return pitch


def _round_half_away_from_zero(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def _validate_request_slice(
    requests: Sequence[Mapping[str, Any]], *, split: str, expected_count: int
) -> None:
    if len(requests) != expected_count:
        raise ValueError(f"Expected {expected_count} {split} requests, got {len(requests)}")
    if any(str(request.get("split")) != split for request in requests):
        raise ValueError(f"Request split mismatch in {split}")
    identities = [_identity_key(request["identity"]) for request in requests]
    if len(identities) != len(set(identities)):
        raise ValueError(f"Duplicate identities in {split} requests")


def _validate_unlabeled_identity(
    request: Mapping[str, Any], measure: selector.UnlabeledMeasure
) -> None:
    if int(request["identity"]["system_measure_index"]) != measure.measure:
        raise ValueError("Candidate measure does not match request identity")
    if measure.source_sha256 != str(request["images"]["raw"]["sha256"]):
        raise ValueError("Candidate image hash does not match request identity")


def _resolve_request_image(request: Mapping[str, Any], out_dir: Path) -> Path:
    raw = Path(str(request["images"]["raw"]["path_relative_to_out"]))
    path = raw if raw.is_absolute() else out_dir / raw
    if not path.exists():
        raise FileNotFoundError(path)
    return path.resolve()


def _artifact_summary(artifacts: SplitArtifacts) -> dict[str, Any]:
    return {
        "prediction_jsonl": _display_path(artifacts.prediction_path),
        "inference_jsonl": _display_path(artifacts.inference_path),
        "freeze_json": _display_path(artifacts.freeze_path),
        "freeze_sha256": artifacts.freeze_sha256,
        "overlays": [_display_path(path) for path in artifacts.overlay_paths],
    }


def _write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    development = report["development"]["metrics"]["summary"]
    validation = report["validation"]["metrics"]["summary"]
    threshold = report["threshold_selector"]
    threshold_development = threshold["development"]["metrics"]["summary"]
    threshold_validation = threshold["validation"]["metrics"]["summary"]
    candidate_metrics = threshold["development"]["candidate_selection_metrics"]
    lines = [
        "# Composed Melody Chain Spike",
        "",
        f"Automatic-count validation gate: "
        f"**{str(report['validation_gate']['status']).upper()}**",
        f"Threshold-selector validation gate: "
        f"**{str(threshold['validation_gate']['status']).upper()}**",
        f"Threshold-selector heldout: **{str(threshold['heldout']['status'])}**",
        "",
        "## Strict metrics",
        "",
        "| Split | Note F1 | Ordered pitch | Ordered onset | Ordered duration | Rest F1 |",
        "|---|---:|---:|---:|---:|---:|",
        _metric_row("Count: development S1M1-4 LOOCV", development),
        _metric_row("Count: validation S7+S8", validation),
        _metric_row("Threshold: development S1M1-4 LOOCV", threshold_development),
        _metric_row("Threshold: validation S7+S8", threshold_validation),
    ]
    threshold_heldout = threshold["heldout"]
    if threshold_heldout["metrics"] is not None:
        lines.append(_metric_row("Threshold: heldout S3", threshold_heldout["metrics"]["summary"]))
    lines.extend(
        [
            "",
            "Threshold candidate-selection development LOOCV P/R/F1: "
            f"`{float(candidate_metrics['precision']):.6f}` / "
            f"`{float(candidate_metrics['recall']):.6f}` / "
            f"`{float(candidate_metrics['f1']):.6f}`.",
        ]
    )
    lines.extend(
        [
            "",
            "## Threshold counts",
            "",
            "| Split | System | Measure | Predicted noteheads | Score threshold |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for split_label, rows in (
        ("development", threshold["development"]["per_measure_counts"]),
        ("validation", threshold["validation"]["per_measure_counts"]),
        ("heldout", threshold_heldout["per_measure_counts"]),
    ):
        lines.extend(_count_row(split_label, row) for row in rows)
    lines.extend(
        [
            "",
            "## Model-selection disclosure",
            "",
            "S7/S8 truth was opened by the original automatic-count arm before this follow-up "
            "was proposed. It was not used to fit the threshold: every threshold comes only "
            "from inner leave-one-measure-out scores on promoted S1 reviews. This is therefore "
            "validation-driven model selection, not a fresh validation claim. System 3 was "
            "sealed when the threshold gate was decided; after the pass, its predictions were "
            "frozen and truth was evaluated once. Matching reruns reuse that evaluation snapshot "
            "without reopening truth.",
            "",
            "## Leakage boundary",
            "",
            "Promoted review fixtures contribute only S1 candidate accepted/rejected training "
            "labels. Inference anchors are selected candidate centers; pitches come from request "
            "staff geometry and allowed key context. Candidate, anchor, rest, and canonical event "
            "predictions are hashed in each split freeze before truth is loaded.",
            "",
            "## Validation gate",
            "",
            f"- Count strict note F1: `{validation['note_f1']:.6f}` > "
            f"`{VALIDATION_NOTE_F1_BASELINE:.6f}`: "
            f"`{report['validation_gate']['note_f1']['passed']}`",
            f"- Count ordered pitch accuracy: `{validation['ordered_pitch_accuracy']:.6f}` > "
            f"`{VALIDATION_ORDERED_PITCH_BASELINE:.6f}`: "
            f"`{report['validation_gate']['ordered_pitch_accuracy']['passed']}`",
            f"- Threshold strict note F1: `{threshold_validation['note_f1']:.6f}` > "
            f"`{VALIDATION_NOTE_F1_BASELINE:.6f}`: "
            f"`{threshold['validation_gate']['note_f1']['passed']}`",
            f"- Threshold ordered pitch accuracy: "
            f"`{threshold_validation['ordered_pitch_accuracy']:.6f}` > "
            f"`{VALIDATION_ORDERED_PITCH_BASELINE:.6f}`: "
            f"`{threshold['validation_gate']['ordered_pitch_accuracy']['passed']}`",
            "",
            "## Artifacts",
            "",
            f"- Development predictions: "
            f"`{report['development']['artifacts']['prediction_jsonl']}`",
            f"- Validation predictions: "
            f"`{report['validation']['artifacts']['prediction_jsonl']}`",
            f"- Threshold development predictions: "
            f"`{threshold['development']['artifacts']['prediction_jsonl']}`",
            f"- Threshold validation predictions: "
            f"`{threshold['validation']['artifacts']['prediction_jsonl']}`",
            f"- Report JSON: `{report['artifacts']['report_json']}`",
        ]
    )
    if threshold_heldout["artifacts"] is not None:
        lines.append(
            f"- Threshold heldout predictions: "
            f"`{threshold_heldout['artifacts']['prediction_jsonl']}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metric_row(label: str, summary: Mapping[str, Any]) -> str:
    return (
        f"| {label} | {float(summary['note_f1']):.6f} | "
        f"{float(summary['ordered_pitch_accuracy']):.6f} | "
        f"{float(summary['ordered_onset_accuracy']):.6f} | "
        f"{float(summary['ordered_duration_accuracy']):.6f} | "
        f"{float(summary['rest_f1']):.6f} |"
    )


def _count_row(split: str, row: Mapping[str, Any]) -> str:
    return (
        f"| {split} | {int(row['system_index'])} | "
        f"{int(row['system_measure_index'])} | "
        f"{int(row['predicted_notehead_count'])} | "
        f"{float(row['learned_score_threshold']):.9f} |"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _identity_key(identity: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(identity["slug"]),
        int(identity["system_index"]),
        int(identity["system_measure_index"]),
        int(identity["global_measure_index"]),
    )


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


def _path_from_display(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
