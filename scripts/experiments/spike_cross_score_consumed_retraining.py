"""Retrain the selected notehead chain with consumed cross-score reviews.

This spike compares Aviador-only training with two Carrizal confidence policies.
Every reported evaluation fold is score-disjoint. Carrizal is consumed training
evidence after this experiment; its historical heldout result is preserved only
as benchmark history and is not claimed as heldout performance here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import spike_cap24_review_augmented_selector as cap  # noqa: E402
from scripts.experiments import spike_meter_gap_resolver as gap  # noqa: E402
from scripts.experiments import spike_notehead_patch_templates as patches  # noqa: E402
from scripts.experiments import spike_review_augmented_selector as dense  # noqa: E402

AVIADOR_SLUG = "jaime-llanos_12_aviador_pasillo_fulgencio-garcia"
CARRIZAL_SLUG = "jaime-llanos_19_carrizal_pasillo_emilio-murillo"
LA_CHATA_SLUG = "jaime-llanos_64_la-chata_pasillo_luis-a-calvo"
NAMESPACE = "carrizal_system_004_seg_v2"
OUTPUT_VERSION = "cross_score_notehead_v1"
DEFAULT_OUT_DIR = REPO_ROOT / "out"
DEFAULT_REVIEWS_DIR = REPO_ROOT / "tests/fixtures/vlm_melody/notehead_reviews"
DEFAULT_CARRIZAL_REVIEWS = (
    DEFAULT_OUT_DIR
    / CARRIZAL_SLUG
    / "vlm_melody_training_inputs"
    / NAMESPACE
    / "agent_reviews"
    / "manifest.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_OUT_DIR / "vlm_melody_consumed_training" / OUTPUT_VERSION
EPSILON = 1e-9
CONFIDENCE_POLICIES = {
    "B": frozenset({"high"}),
    "C": frozenset({"high", "medium"}),
}
# This cross-score spike deliberately uses a smaller preregistered search than
# the general dense selector. The endpoints and established middle values
# retain the selector's useful operating range while keeping each fit bounded.
LOCAL_THRESHOLD_BUDGET = 48
LOCAL_NMS_X_SPACES_GRID = (0.45, 0.68, 0.85)
LOCAL_MIN_SELECTED_COUNT_GRID = (0, 2, 4)
LOCAL_MAX_SELECTED_COUNT_GRID = (4, 5, 6)
LOCAL_CONFIGURATION_COUNT = sum(
    minimum <= maximum
    for _nms in LOCAL_NMS_X_SPACES_GRID
    for minimum in LOCAL_MIN_SELECTED_COUNT_GRID
    for maximum in LOCAL_MAX_SELECTED_COUNT_GRID
)
PROTECTED_TRUTH_PARTS = (
    "dataset/ground_truth",
    f"dataset/musicxml/{LA_CHATA_SLUG}",
)


@dataclass(frozen=True)
class FittedChain:
    model: gap.GapAwareSelectorModel
    method: dict[str, Any]
    pitch_cv: dict[str, Any]
    pitch_method: str
    accidental_model: dense.AccidentalKNN | None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--reviews-dir", type=Path, default=DEFAULT_REVIEWS_DIR)
    parser.add_argument("--carrizal-reviews", type=Path, default=DEFAULT_CARRIZAL_REVIEWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    try:
        report = run_experiment(
            args.out_dir,
            reviews_dir=args.reviews_dir,
            carrizal_reviews=args.carrizal_reviews,
            output_dir=args.output_dir,
        )
    except (FileExistsError, FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(report["artifacts"]["output_dir"])
    print(f"selected configuration: {report['selection']['selected_configuration']}")
    return 0


def run_experiment(
    out_dir: Path,
    *,
    reviews_dir: Path = DEFAULT_REVIEWS_DIR,
    carrizal_reviews: Path = DEFAULT_CARRIZAL_REVIEWS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    out_dir = out_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite create-once output: {output_dir}")
    _reject_protected_truth_paths((reviews_dir, carrizal_reviews, output_dir))

    aviador, aviador_sources = _load_aviador_examples(out_dir, reviews_dir=reviews_dir)
    carrizal_by_policy, carrizal_sources = _load_carrizal_examples(
        out_dir,
        manifest_path=carrizal_reviews,
    )

    baseline_model = _fit_selected_chain(aviador)
    baseline_all = _evaluate_score_fold(
        baseline_model,
        training=aviador,
        evaluation=carrizal_by_policy["C"],
        configuration="A",
        confidence_policy="high+medium",
    )
    baseline_high = _evaluate_score_fold(
        baseline_model,
        training=aviador,
        evaluation=carrizal_by_policy["B"],
        configuration="A",
        confidence_policy="high",
    )

    configurations = {
        "A": {
            "description": "Aviador S1+S7 only; cross-score Carrizal context baseline",
            "eligible_for_final_selection": False,
            "folds": [baseline_all],
            "high_confidence_view": baseline_high,
            "macro": _macro_score_metrics([baseline_all]),
        }
    }
    for configuration, confidences in CONFIDENCE_POLICIES.items():
        carrizal = carrizal_by_policy[configuration]
        aviador_to_carrizal = _evaluate_score_fold(
            baseline_model,
            training=aviador,
            evaluation=carrizal,
            configuration=configuration,
            confidence_policy="+".join(sorted(confidences)),
        )
        carrizal_model = _fit_selected_chain(carrizal)
        carrizal_to_aviador = _evaluate_score_fold(
            carrizal_model,
            training=carrizal,
            evaluation=aviador,
            configuration=configuration,
            confidence_policy="all_human_reviews",
        )
        folds = [aviador_to_carrizal, carrizal_to_aviador]
        configurations[configuration] = {
            "description": (
                "Aviador plus Carrizal high-confidence heads"
                if configuration == "B"
                else "Aviador plus Carrizal high- and medium-confidence heads"
            ),
            "eligible_for_final_selection": True,
            "carrizal_confidences": sorted(confidences),
            "folds": folds,
            "macro": _macro_score_metrics(folds),
        }
    selection = select_configuration(configurations["B"]["macro"], configurations["C"]["macro"])
    selected_configuration = str(selection["selected_configuration"])
    catastrophic = all(_catastrophic(configurations[key]["macro"]) for key in ("B", "C"))
    if catastrophic:
        selected_configuration = "A"
        selection = {
            **selection,
            "selected_configuration": "A",
            "status": "blocked_both_cross_score_configurations_catastrophic",
        }
        final_training = aviador
    else:
        final_training = [*aviador, *carrizal_by_policy[selected_configuration]]
        selection = {**selection, "status": "selected_cross_score_consumed_training_model"}

    final_model = _refit_with_fixed_configuration(final_training, template=baseline_model)
    training_selection = _training_selection_payload(
        selection=selection,
        configurations=configurations,
        final_training=final_training,
        aviador_sources=aviador_sources,
        carrizal_sources=carrizal_sources,
    )
    model_payload = _serialize_model(
        final_model,
        configuration=selected_configuration,
        training=final_training,
        blocked=catastrophic,
    )
    report = {
        "schema_version": 1,
        "kind": "vlm_melody_cross_score_consumed_retraining",
        "experiment_version": OUTPUT_VERSION,
        "protocol": {
            "component": "dense/cap24 class-template notehead selector with meter-gap recovery",
            "score_disjoint_evaluation": True,
            "configuration_A": "train Aviador; evaluate corrected Carrizal",
            "configurations_B_C": (
                "two leave-one-score-out folds: train Aviador/evaluate Carrizal and "
                "train Carrizal/evaluate Aviador"
            ),
            "selection_rule": selection["rule"],
            "final_refit": (
                "Refit the Aviador-selected patch scorer and pitch samples on all consumed "
                "reviews without re-running hyperparameter selection."
            ),
            "bounded_training_selector": {
                "threshold_budget": LOCAL_THRESHOLD_BUDGET,
                "nms_x_spaces_grid": list(LOCAL_NMS_X_SPACES_GRID),
                "minimum_selected_count_grid": list(LOCAL_MIN_SELECTED_COUNT_GRID),
                "maximum_selected_count_grid": list(LOCAL_MAX_SELECTED_COUNT_GRID),
                "configuration_count": LOCAL_CONFIGURATION_COUNT,
                "threshold_strategy": (
                    "sorted unique score values sampled by evenly spaced rank indices, "
                    "plus max_score+1e-9 sentinel"
                ),
                "tie_break_semantics": "unchanged from dense selector",
            },
            "truth_policy": (
                "Only consumed Aviador reviews and consumed Carrizal agent adjudication are read. "
                "No La Chata truth or MusicXML is accessed."
            ),
        },
        "configurations": configurations,
        "selection": selection,
        "historical_evidence_status": {
            "carrizal_old_heldout_metrics": "historical_only_not_claimable_after_consumption",
            "frozen_evidence_overwritten": False,
            "third_score_predictions_created": False,
        },
        "final_model": {
            "configuration": selected_configuration,
            "blocked": catastrophic,
            "training_scores": sorted(_score_slugs(final_training)),
            "training_measure_count": len(final_training),
            "training_notehead_count": sum(example.true_note_count for example in final_training),
        },
        "artifacts": {
            "output_dir": _display_path(output_dir),
            "report_json": _display_path(output_dir / "report.json"),
            "report_markdown": _display_path(output_dir / "report.md"),
            "training_selection": _display_path(output_dir / "training_selection.json"),
            "model": _display_path(output_dir / "model.json"),
            "manifest": _display_path(output_dir / "manifest.json"),
        },
    }
    _write_create_once_artifacts(
        output_dir,
        report=report,
        training_selection=training_selection,
        model=model_payload,
    )
    return report


def _load_aviador_examples(
    out_dir: Path,
    *,
    reviews_dir: Path,
) -> tuple[list[dense.TrainingExample], list[dict[str, str]]]:
    benchmark_dir = out_dir / AVIADOR_SLUG / "vlm_melody_event_benchmark"
    request_paths = [
        benchmark_dir / "development/requests.jsonl",
        benchmark_dir / "validation/requests.jsonl",
    ]
    requests = [row for path in request_paths for row in dense._read_jsonl(path)]
    selected = dense._select_requests(requests, dense.TRAINING_TARGETS)
    unlabeled = [dense._prepare_dense_measure(request, out_dir=out_dir) for request in selected]
    examples = [
        dense._attach_review(request, measure, reviews_dir=reviews_dir, slug=AVIADOR_SLUG)
        for request, measure in zip(selected, unlabeled, strict=True)
    ]
    examples = [_with_score_key(example, AVIADOR_SLUG) for example in examples]
    sources = [_file_record(path) for path in request_paths]
    sources.extend(_file_record(example.measure.review_path) for example in examples)
    _reject_protected_truth_paths(Path(row["path"]) for row in sources)
    return examples, sources


def _load_carrizal_examples(
    out_dir: Path,
    *,
    manifest_path: Path,
) -> tuple[dict[str, list[dense.TrainingExample]], list[dict[str, str]]]:
    manifest_path = manifest_path.resolve()
    manifest = _load_json_object(manifest_path, "Carrizal agent-review manifest")
    required = {
        "kind": "vlm_melody_agent_visual_adjudication_manifest",
        "split_status": "consumed_training",
        "reviewer_type": "agent_visual_adjudication",
        "parent_review_status": "accepted_for_spike_training",
        "eligible_for_spike_training": True,
        "eligible_for_human_promotion": False,
        "human_reviewed": False,
        "segmentation_namespace": NAMESPACE,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Carrizal manifest {key} must be {expected!r}")
    identity = manifest.get("identity")
    if identity != {"slug": CARRIZAL_SLUG, "system_index": 4}:
        raise ValueError("Carrizal manifest identity mismatch")
    source_records = [_file_record(manifest_path)]
    for record in manifest.get("source", {}).values():
        source_records.append(_validate_file_record(record, label="Carrizal manifest source"))
    review_records = manifest.get("outputs", {}).get("reviews")
    if not isinstance(review_records, list) or len(review_records) != 8:
        raise ValueError("Carrizal manifest must pin exactly eight review files")

    payloads = []
    for record in review_records:
        validated = _validate_file_record(record, label="Carrizal review")
        source_records.append(validated)
        path = REPO_ROOT / validated["path"]
        payloads.append(_load_json_object(path, "Carrizal review"))
    _reject_protected_truth_paths(Path(row["path"]) for row in source_records)

    by_policy = {}
    for configuration, confidences in CONFIDENCE_POLICIES.items():
        examples = [
            _attach_carrizal_review(
                payload,
                out_dir=out_dir,
                review_path=REPO_ROOT / review_record["path"],
                confidences=confidences,
            )
            for payload, review_record in zip(payloads, review_records, strict=True)
        ]
        by_policy[configuration] = examples
    return by_policy, source_records


def _attach_carrizal_review(
    payload: Mapping[str, Any],
    *,
    out_dir: Path,
    review_path: Path,
    confidences: frozenset[str],
) -> dense.TrainingExample:
    required = {
        "kind": "vlm_melody_agent_visual_adjudication_review",
        "split_status": "consumed_training",
        "reviewer_type": "agent_visual_adjudication",
        "parent_review_status": "accepted_for_spike_training",
        "eligible_for_spike_training": True,
        "eligible_for_human_promotion": False,
        "human_reviewed": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"Carrizal review {key} must be {expected!r}: {review_path}")
    identity = payload.get("identity", {})
    if identity.get("slug") != CARRIZAL_SLUG or int(identity.get("system_index", -1)) != 4:
        raise ValueError(f"Carrizal review identity mismatch: {review_path}")
    measure_index = int(identity["system_measure_index"])
    raw_record = _validate_file_record(
        payload.get("source", {}).get("raw_image"), label="Carrizal raw image"
    )
    candidate_record = _validate_file_record(
        payload.get("source", {}).get("candidate_artifact"),
        label="Carrizal candidate artifact",
    )
    raw_path = REPO_ROOT / raw_record["path"]
    candidate_payload = _load_json_object(
        REPO_ROOT / candidate_record["path"], "Carrizal candidates"
    )
    if int(candidate_payload.get("system_measure_index", -1)) != measure_index:
        raise ValueError(f"Carrizal candidate identity mismatch: {review_path}")
    staff_lines = tuple(int(value) for value in candidate_payload["staff_lines_y_px"])
    with Image.open(raw_path) as opened:
        width, height = opened.size
    request = {
        "schema_version": 1,
        "split": "consumed_training",
        "identity": {
            "slug": CARRIZAL_SLUG,
            "system_index": 4,
            "system_measure_index": measure_index,
            "global_measure_index": int(
                candidate_payload.get("global_measure_index", measure_index)
            ),
        },
        "images": {
            "raw": {
                "path_relative_to_out": raw_path.relative_to(out_dir).as_posix(),
                "sha256": raw_record["sha256"],
                "width_px": width,
                "height_px": height,
            }
        },
        "staff_geometry": {"raw_staff_lines_y_px": list(staff_lines)},
        "allowed_context": {
            "clef": "treble",
            "key_hint": None,
            "time_signature": None,
            "expected_measure_beats": None,
            "allow_pickup": False,
        },
    }
    unlabeled = dense._prepare_dense_measure(request, out_dir=out_dir)
    head_payloads = payload.get("heads")
    if not isinstance(head_payloads, list) or not head_payloads:
        raise ValueError(f"Carrizal review has no heads: {review_path}")
    selected_head_orders = {
        int(head["order"]) for head in _select_review_heads(head_payloads, confidences)
    }
    all_notes = []
    included_ids = set()
    for index, head in enumerate(head_payloads, start=1):
        confidence = str(head.get("confidence"))
        if confidence not in {"high", "medium"}:
            raise ValueError(f"Unexpected Carrizal confidence {confidence!r}: {review_path}")
        note = dense.ReviewNote(
            id=f"n{index:03d}",
            order=int(head.get("order", index)),
            x=float(head["center"]["x"]),
            annotated_y=float(head["center"]["y"]),
            pitch=str(head["sounding_pitch"]),
            pitch_row_y=dense._pitch_row_y(str(head["sounding_pitch"]), staff_lines),
            source_kind=f"agent_{head.get('selection', {}).get('kind', 'unknown')}_{confidence}",
        )
        all_notes.append(note)
        if note.order in selected_head_orders:
            included_ids.add(note.id)
    all_matches = dense._match_dense_candidates(all_notes, unlabeled)
    ignored_candidate_ids = {
        candidate_id for note_id, candidate_id in all_matches.items() if note_id not in included_ids
    }
    notes = tuple(note for note in all_notes if note.id in included_ids)
    matched = {
        note_id: candidate_id
        for note_id, candidate_id in all_matches.items()
        if note_id in included_ids
    }
    rows = tuple(
        patches.LabeledCandidate(
            candidate=candidate,
            label=int(candidate.id in matched.values()),
        )
        for candidate in unlabeled.candidates
        if candidate.id not in ignored_candidate_ids
    )
    labeled = patches.MeasureData(
        measure=measure_index,
        source_image=raw_path,
        source_sha256=raw_record["sha256"],
        review_path=review_path,
        review_sha256=_sha256(review_path),
        staff_lines=staff_lines,
        staff_spacing=unlabeled.staff_spacing,
        rows=rows,
    )
    return dense.TrainingExample(
        key=f"{CARRIZAL_SLUG}:S04M{measure_index:02d}",
        request=request,
        measure=labeled,
        notes=notes,
        matched_candidate_ids=frozenset(matched.values()),
        unmatched_note_ids=tuple(note.id for note in notes if note.id not in matched),
        pitch_vectors=dense._pitch_vectors(raw_path, staff_lines, notes),
    )


def _fit_selected_chain(examples: Sequence[dense.TrainingExample]) -> FittedChain:
    if len(examples) < 2:
        raise ValueError("Selected-chain fit requires at least two measures")
    methods = []
    for patch_spec in patches.PATCH_SPECS:
        scores = cap._out_of_fold_scores(
            examples,
            patch_id=patch_spec.id,
            scorer_kind="class_template",
        )
        selection = _select_bounded_training_configuration(examples, scores)
        recovery = gap._select_recovery_configuration(examples, scores, selection=selection)
        methods.append(
            {
                "method_id": f"class_template__{patch_spec.id}",
                "patch_id": patch_spec.id,
                "scorer_kind": "class_template",
                "selection": selection,
                "recovery": recovery,
            }
        )
    winner = max(
        methods,
        key=lambda method: (
            float(method["recovery"].metrics["f1"]),
            float(method["recovery"].metrics["recall"]),
            float(method["recovery"].metrics["precision"]),
            int(method["recovery"].metrics["exact_measures"]),
            str(method["method_id"]),
        ),
    )
    scorer = patches._fit_patch_scorer(
        [row for example in examples for row in example.measure.rows],
        patch_id=str(winner["patch_id"]),
        scorer_kind="class_template",
    )
    selection = winner["selection"]
    base_model = dense.DenseSelectorModel(
        scorer=scorer,  # type: ignore[arg-type]
        learned_threshold=float(selection["threshold"]),
        threshold_training_metrics=dict(selection["metrics"]),
        training_keys=tuple(example.key for example in examples),
        training_positive_count=sum(example.true_note_count for example in examples),
        learned_count=int(statistics.median(example.true_note_count for example in examples)),
        nms_x_spaces=float(selection["nms_x_spaces"]),
        minimum_selected_count=int(selection["minimum_selected_count"]),
        maximum_selected_count=int(selection["maximum_selected_count"]),
    )
    pitch_cv = dense._evaluate_pitch_methods(examples)
    pitch_method = str(pitch_cv["selected_method"])
    accidental_model = (
        dense._fit_accidental_model(examples) if pitch_method == "accidental_knn" else None
    )
    return FittedChain(
        model=gap.GapAwareSelectorModel(base=base_model, recovery=winner["recovery"]),
        method=gap._serializable_method(winner),
        pitch_cv=pitch_cv,
        pitch_method=pitch_method,
        accidental_model=accidental_model,
    )


def _threshold_candidates(
    scores: Mapping[tuple[str, str], float],
    *,
    budget: int = LOCAL_THRESHOLD_BUDGET,
) -> tuple[float, ...]:
    """Return a deterministic bounded threshold set, including no-selection."""
    if budget < 2:
        raise ValueError("Threshold budget must be at least two")
    values = sorted({float(value) for value in scores.values()})
    if not values:
        raise ValueError("Training threshold search requires at least one score")
    ordinary_budget = min(len(values), budget - 1)
    if len(values) > ordinary_budget:
        indices = [
            round(index * (len(values) - 1) / max(1, ordinary_budget - 1))
            for index in range(ordinary_budget)
        ]
        values = sorted({values[index] for index in indices})
    return tuple([values[-1] + 1e-9, *reversed(values)])


def _select_bounded_training_threshold(
    examples: Sequence[dense.TrainingExample],
    scores: Mapping[tuple[str, str], float],
    *,
    nms_x_spaces: float,
    minimum_selected_count: int,
    maximum_selected_count: int,
) -> tuple[float, dict[str, Any]]:
    thresholds = _threshold_candidates(scores)
    best_threshold = thresholds[0]
    best_metrics: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    for threshold in thresholds:
        metrics = dense._aggregate_selection_metrics(
            [
                dense._selection_metrics(
                    example,
                    dense._select_candidates(
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
        # Keep the dense selector's objective and deterministic tie-break order.
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


def _select_bounded_training_configuration(
    examples: Sequence[dense.TrainingExample],
    scores: Mapping[tuple[str, str], float],
) -> dict[str, Any]:
    rows = []
    threshold_candidate_count = len(_threshold_candidates(scores))
    for nms_x_spaces in LOCAL_NMS_X_SPACES_GRID:
        for minimum_selected_count in LOCAL_MIN_SELECTED_COUNT_GRID:
            for maximum_selected_count in LOCAL_MAX_SELECTED_COUNT_GRID:
                if minimum_selected_count > maximum_selected_count:
                    continue
                threshold, metrics = _select_bounded_training_threshold(
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
    search_budget = {
        "threshold_budget": LOCAL_THRESHOLD_BUDGET,
        "threshold_candidate_count": threshold_candidate_count,
        "threshold_strategy": (
            "sorted unique score values sampled by evenly spaced rank indices, "
            "plus max_score+1e-9 sentinel"
        ),
        "nms_x_spaces_grid": list(LOCAL_NMS_X_SPACES_GRID),
        "minimum_selected_count_grid": list(LOCAL_MIN_SELECTED_COUNT_GRID),
        "maximum_selected_count_grid": list(LOCAL_MAX_SELECTED_COUNT_GRID),
        "configuration_count": len(rows),
        "max_search_evaluations": len(rows) * threshold_candidate_count,
        "objective": "training-review out-of-fold notehead F1 only",
    }
    return {
        **winner,
        "selection_basis": "training-review out-of-fold notehead F1 only",
        "search_budget": search_budget,
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


def _refit_with_fixed_configuration(
    examples: Sequence[dense.TrainingExample],
    *,
    template: FittedChain,
) -> FittedChain:
    patch_id = str(template.method["patch_id"])
    scorer = patches._fit_patch_scorer(
        [row for example in examples for row in example.measure.rows],
        patch_id=patch_id,
        scorer_kind="class_template",
    )
    base = template.model.base
    refitted_base = dense.DenseSelectorModel(
        scorer=scorer,  # type: ignore[arg-type]
        learned_threshold=base.learned_threshold,
        threshold_training_metrics=base.threshold_training_metrics,
        training_keys=tuple(example.key for example in examples),
        training_positive_count=sum(example.true_note_count for example in examples),
        learned_count=int(statistics.median(example.true_note_count for example in examples)),
        nms_x_spaces=base.nms_x_spaces,
        minimum_selected_count=base.minimum_selected_count,
        maximum_selected_count=base.maximum_selected_count,
    )
    accidental_model = (
        dense._fit_accidental_model(examples) if template.pitch_method == "accidental_knn" else None
    )
    method = dict(template.method)
    method["final_refit"] = {
        "hyperparameters_from": "Aviador S1+S7 training-only selected chain",
        "scorer_refit_on_all_consumed_reviews": True,
        "hyperparameter_search_repeated_on_combined_data": False,
    }
    return FittedChain(
        model=gap.GapAwareSelectorModel(base=refitted_base, recovery=template.model.recovery),
        method=method,
        pitch_cv=template.pitch_cv,
        pitch_method=template.pitch_method,
        accidental_model=accidental_model,
    )


def _evaluate_score_fold(
    fitted: FittedChain,
    *,
    training: Sequence[dense.TrainingExample],
    evaluation: Sequence[dense.TrainingExample],
    configuration: str,
    confidence_policy: str,
) -> dict[str, Any]:
    training_scores = _score_slugs(training)
    evaluation_scores = _score_slugs(evaluation)
    _assert_score_disjoint(training_scores, evaluation_scores)
    predictor = dense._build_pitch_predictor(fitted.accidental_model)
    rows = []
    matched_notehead_pitch_correct = 0
    matched_notehead_pitch_total = 0
    truth_notehead_total = 0
    exact_with_pitch = 0
    for example in evaluation:
        unlabeled = patches.UnlabeledMeasure(
            measure=example.measure.measure,
            source_image=example.measure.source_image,
            source_sha256=example.measure.source_sha256,
            staff_lines=example.measure.staff_lines,
            staff_spacing=example.measure.staff_spacing,
            candidates=tuple(row.candidate for row in example.measure.rows),
        )
        _, selected = fitted.model.rank(unlabeled)
        coordinate = dense._selection_metrics(example, selected)
        matches = dense._match_dense_candidates(example.notes, unlabeled)
        notes_by_candidate = {
            candidate_id: next(note for note in example.notes if note.id == note_id)
            for note_id, candidate_id in matches.items()
        }
        measure_pitch_correct = 0
        measure_matched_notehead_total = 0
        with Image.open(example.measure.source_image) as opened:
            image = opened.convert("RGB")
        for candidate in selected:
            note = notes_by_candidate.get(candidate.id)
            if note is None:
                continue
            predicted = predictor(candidate, example.request, image)
            measure_matched_notehead_total += 1
            measure_pitch_correct += int(
                dense.rhythm.pitch_to_midi(predicted) == dense.rhythm.pitch_to_midi(note.pitch)
            )
        matched_notehead_pitch_correct += measure_pitch_correct
        matched_notehead_pitch_total += measure_matched_notehead_total
        truth_notehead_total += example.true_note_count
        pitch_exact = bool(coordinate["exact_set"]) and (
            measure_matched_notehead_total == example.true_note_count
            and measure_pitch_correct == example.true_note_count
        )
        exact_with_pitch += int(pitch_exact)
        rows.append(
            {
                **coordinate,
                "matched_notehead_pitch_correct": measure_pitch_correct,
                "matched_notehead_pitch_total": measure_matched_notehead_total,
                "conditional_pitch_accuracy_on_matched_noteheads": dense._ratio(
                    measure_pitch_correct, measure_matched_notehead_total
                ),
                "truth_notehead_total": example.true_note_count,
                "end_to_end_correct_pitch_recall": dense._ratio(
                    measure_pitch_correct, example.true_note_count
                ),
                "exact_with_pitch": pitch_exact,
            }
        )
    coordinate = dense._aggregate_selection_metrics(rows)
    return {
        "configuration": configuration,
        "training_scores": sorted(training_scores),
        "evaluation_score": next(iter(evaluation_scores)),
        "score_disjoint": True,
        "confidence_policy": confidence_policy,
        "training_measure_count": len(training),
        "evaluation_measure_count": len(evaluation),
        "coordinate_noteheads": coordinate,
        "conditional_pitch_on_matched_noteheads": {
            "correct": matched_notehead_pitch_correct,
            "matched_noteheads": matched_notehead_pitch_total,
            "accuracy": dense._ratio(matched_notehead_pitch_correct, matched_notehead_pitch_total),
        },
        "end_to_end_correct_pitch": {
            "correct": matched_notehead_pitch_correct,
            "truth_noteheads": truth_notehead_total,
            "recall": dense._ratio(matched_notehead_pitch_correct, truth_notehead_total),
        },
        "proposal_recall": dense._proposal_recall(evaluation),
        "measure_exactness": {
            "coordinate_exact": coordinate["exact_measures"],
            "coordinate_and_pitch_exact": exact_with_pitch,
            "total": len(evaluation),
            "coordinate_rate": dense._ratio(coordinate["exact_measures"], len(evaluation)),
            "coordinate_and_pitch_rate": dense._ratio(exact_with_pitch, len(evaluation)),
        },
        "training_selection": {
            "method_id": fitted.method["method_id"],
            "patch_id": fitted.method["patch_id"],
            "selection": fitted.method["selection"],
            "recovery": fitted.method["recovery"],
            "pitch": fitted.pitch_cv,
        },
    }


def _macro_score_metrics(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not folds:
        raise ValueError("Macro score metrics require at least one fold")
    conditional_pitch_correct = sum(
        int(fold["conditional_pitch_on_matched_noteheads"]["correct"]) for fold in folds
    )
    matched_noteheads = sum(
        int(fold["conditional_pitch_on_matched_noteheads"]["matched_noteheads"]) for fold in folds
    )
    end_to_end_pitch_correct = sum(
        int(fold["end_to_end_correct_pitch"]["correct"]) for fold in folds
    )
    truth_noteheads = sum(
        int(fold["end_to_end_correct_pitch"]["truth_noteheads"]) for fold in folds
    )
    return {
        "score_count": len(folds),
        "notehead_precision": statistics.mean(
            float(fold["coordinate_noteheads"]["precision"]) for fold in folds
        ),
        "notehead_recall": statistics.mean(
            float(fold["coordinate_noteheads"]["recall"]) for fold in folds
        ),
        "notehead_f1": statistics.mean(float(fold["coordinate_noteheads"]["f1"]) for fold in folds),
        "conditional_pitch_accuracy_on_matched_noteheads": statistics.mean(
            float(fold["conditional_pitch_on_matched_noteheads"]["accuracy"]) for fold in folds
        ),
        "end_to_end_correct_pitch_recall": statistics.mean(
            float(fold["end_to_end_correct_pitch"]["recall"]) for fold in folds
        ),
        "conditional_pitch_accuracy_on_matched_noteheads_micro": dense._ratio(
            conditional_pitch_correct, matched_noteheads
        ),
        "end_to_end_correct_pitch_recall_micro": dense._ratio(
            end_to_end_pitch_correct, truth_noteheads
        ),
        "proposal_recall": statistics.mean(
            float(fold["proposal_recall"]["recall"]) for fold in folds
        ),
        "coordinate_measure_exactness": statistics.mean(
            float(fold["measure_exactness"]["coordinate_rate"]) for fold in folds
        ),
        "coordinate_and_pitch_measure_exactness": statistics.mean(
            float(fold["measure_exactness"]["coordinate_and_pitch_rate"]) for fold in folds
        ),
    }


def select_configuration(
    metrics_b: Mapping[str, Any],
    metrics_c: Mapping[str, Any],
) -> dict[str, Any]:
    c_f1_ok = float(metrics_c["notehead_f1"]) >= float(metrics_b["notehead_f1"]) - EPSILON
    c_pitch_ok = float(metrics_c["conditional_pitch_accuracy_on_matched_noteheads"]) >= (
        float(metrics_b["conditional_pitch_accuracy_on_matched_noteheads"]) - EPSILON
    )
    selected = "C" if c_f1_ok and c_pitch_ok else "B"
    return {
        "selected_configuration": selected,
        "rule": (
            "Select C only when macro score-level notehead F1 is higher than B or equal "
            "within 1e-9, and macro conditional pitch accuracy on matched/localized "
            "noteheads does not regress; otherwise select B. End-to-end correct-pitch "
            "recall is reported but is not part of this preregistered selection rule."
        ),
        "epsilon": EPSILON,
        "c_notehead_f1_eligible": c_f1_ok,
        "c_conditional_pitch_on_matched_noteheads_non_regression": c_pitch_ok,
        "metrics_B": dict(metrics_b),
        "metrics_C": dict(metrics_c),
    }


def _select_review_heads(
    heads: Sequence[Mapping[str, Any]],
    confidences: frozenset[str],
) -> list[Mapping[str, Any]]:
    unsupported = confidences - {"high", "medium"}
    if unsupported:
        raise ValueError(f"Unsupported confidence policy: {sorted(unsupported)}")
    selected = []
    for head in heads:
        confidence = str(head.get("confidence"))
        if confidence not in {"high", "medium"}:
            raise ValueError(f"Unexpected review confidence: {confidence!r}")
        if confidence in confidences:
            selected.append(head)
    return selected


def _catastrophic(metrics: Mapping[str, Any]) -> bool:
    return float(metrics["notehead_f1"]) <= 0.05 or float(metrics["proposal_recall"]) <= 0.05


def _training_selection_payload(
    *,
    selection: Mapping[str, Any],
    configurations: Mapping[str, Any],
    final_training: Sequence[dense.TrainingExample],
    aviador_sources: Sequence[Mapping[str, str]],
    carrizal_sources: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "vlm_melody_cross_score_training_selection",
        "experiment_version": OUTPUT_VERSION,
        "selection": dict(selection),
        "configuration_metrics": {key: value["macro"] for key, value in configurations.items()},
        "bounded_training_selector": {
            "threshold_budget": LOCAL_THRESHOLD_BUDGET,
            "nms_x_spaces_grid": list(LOCAL_NMS_X_SPACES_GRID),
            "minimum_selected_count_grid": list(LOCAL_MIN_SELECTED_COUNT_GRID),
            "maximum_selected_count_grid": list(LOCAL_MAX_SELECTED_COUNT_GRID),
            "configuration_count": LOCAL_CONFIGURATION_COUNT,
        },
        "final_training": {
            "keys": [example.key for example in final_training],
            "scores": sorted(_score_slugs(final_training)),
            "measure_count": len(final_training),
            "notehead_count": sum(example.true_note_count for example in final_training),
            "review_hashes": [example.measure.review_sha256 for example in final_training],
        },
        "input_provenance": {
            "aviador": list(aviador_sources),
            "carrizal": list(carrizal_sources),
        },
        "historical_carrizal_heldout_status": "consumed_not_claimable_as_heldout",
        "la_chata_truth_accessed": False,
    }


def _serialize_model(
    fitted: FittedChain,
    *,
    configuration: str,
    training: Sequence[dense.TrainingExample],
    blocked: bool,
) -> dict[str, Any]:
    scorer = fitted.model.base.scorer
    if isinstance(scorer, patches.PatchScorer):
        scorer_payload = {
            "kind": "patch_scorer",
            "patch_id": scorer.patch_id,
            "scorer_kind": scorer.scorer_kind,
            "positive_vectors": [list(vector) for vector in scorer.positive_vectors],
            "negative_vectors": [list(vector) for vector in scorer.negative_vectors],
        }
    elif isinstance(scorer, dense.LogisticScorer):
        scorer_payload = {
            "kind": "dense_logistic",
            "means": list(scorer.means),
            "scales": list(scorer.scales),
            "weights": list(scorer.weights),
            "intercept": scorer.intercept,
        }
    else:
        raise ValueError(f"Unsupported replay scorer: {type(scorer).__name__}")
    accidental_samples = []
    if fitted.accidental_model is not None:
        accidental_samples = [
            {
                "key": sample.key,
                "pitch": sample.pitch,
                "base_pitch": sample.base_pitch,
                "delta": sample.delta,
                "vector": list(sample.vector),
            }
            for sample in fitted.accidental_model.samples
        ]
    return {
        "schema_version": 1,
        "kind": "vlm_melody_consumed_training_notehead_model",
        "experiment_version": OUTPUT_VERSION,
        "configuration": configuration,
        "blocked": blocked,
        "replay": {
            "feature_order": list(dense.DENSE_FEATURES),
            "selection_mode": dense.SELECTION_MODE,
            "method": fitted.method,
            "selector": {
                "scorer": scorer_payload,
                "threshold": fitted.model.learned_threshold,
                "nms_x_spaces": fitted.model.base.nms_x_spaces,
                "minimum_selected_count": fitted.model.base.minimum_selected_count,
                "maximum_selected_count": fitted.model.base.maximum_selected_count,
                "leading_gap_spaces": fitted.model.recovery.leading_gap_spaces,
                "score_margin": fitted.model.recovery.score_margin,
            },
            "pitch": {
                "method": fitted.pitch_method,
                "oof_selection": fitted.pitch_cv,
                "accidental_samples": accidental_samples,
            },
        },
        "training": {
            "keys": [example.key for example in training],
            "scores": sorted(_score_slugs(training)),
            "review_hashes": [example.measure.review_sha256 for example in training],
        },
        "third_score_truth_used": False,
    }


def _write_create_once_artifacts(
    output_dir: Path,
    *,
    report: Mapping[str, Any],
    training_selection: Mapping[str, Any],
    model: Mapping[str, Any],
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite create-once output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir.with_name(f".{output_dir.name}.tmp")
    if temp_dir.exists():
        raise FileExistsError(f"Refusing stale temporary output: {temp_dir}")
    temp_dir.mkdir()
    try:
        _write_json(temp_dir / "training_selection.json", training_selection)
        _write_json(temp_dir / "model.json", model)
        _write_json(temp_dir / "report.json", report)
        _write_markdown(report, temp_dir / "report.md")
        manifest = {
            "schema_version": 1,
            "kind": "vlm_melody_cross_score_consumed_retraining_manifest",
            "experiment_version": OUTPUT_VERSION,
            "create_once": True,
            "artifacts": {
                name: {
                    "path": f"{_display_path(output_dir)}/{name}",
                    "sha256": _sha256(temp_dir / name),
                }
                for name in ("report.json", "report.md", "training_selection.json", "model.json")
            },
            "implementation": _file_record(Path(__file__).resolve()),
            "la_chata_truth_accessed": False,
        }
        _write_json(temp_dir / "manifest.json", manifest)
        temp_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _write_markdown(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Cross-Score Consumed Retraining",
        "",
        "All reported folds are score-disjoint. Carrizal is consumed training evidence.",
        "Its previous heldout result remains historical only and was not overwritten.",
        "",
        (
            "| Config | Macro P | Macro R | Macro F1 | Conditional pitch on matched heads | "
            "End-to-end correct-pitch recall | Proposal R | Exact measures |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("A", "B", "C"):
        metrics = report["configurations"][key]["macro"]
        lines.append(
            "| {key} | {p:.3f} | {r:.3f} | {f1:.3f} | {conditional_pitch:.3f} | "
            "{end_to_end_pitch:.3f} | {proposal:.3f} | {exact:.3f} |".format(
                key=key,
                p=metrics["notehead_precision"],
                r=metrics["notehead_recall"],
                f1=metrics["notehead_f1"],
                conditional_pitch=metrics["conditional_pitch_accuracy_on_matched_noteheads"],
                end_to_end_pitch=metrics["end_to_end_correct_pitch_recall"],
                proposal=metrics["proposal_recall"],
                exact=metrics["coordinate_measure_exactness"],
            )
        )
    lines.extend(
        [
            "",
            f"Selected configuration: **{report['selection']['selected_configuration']}**",
            "",
            report["selection"]["rule"],
            "",
            "No La Chata truth, MusicXML, or third-score prediction was accessed or created.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _with_score_key(example: dense.TrainingExample, slug: str) -> dense.TrainingExample:
    return replace(example, key=f"{slug}:{example.key}")


def _score_slugs(examples: Sequence[dense.TrainingExample]) -> set[str]:
    return {str(example.request["identity"]["slug"]) for example in examples}


def _assert_score_disjoint(training_scores: set[str], evaluation_scores: set[str]) -> None:
    overlap = training_scores & evaluation_scores
    if overlap:
        raise ValueError(f"Score leakage detected: {sorted(overlap)}")
    if len(evaluation_scores) != 1:
        raise ValueError("Each fold must evaluate exactly one score")


def _reject_protected_truth_paths(paths: Sequence[Path] | Any) -> None:
    for path in paths:
        normalized = Path(path).as_posix()
        if any(part in normalized for part in PROTECTED_TRUTH_PARTS):
            raise ValueError(f"Protected third-score truth path is forbidden: {normalized}")


def _validate_file_record(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} record must be an object")
    path_value = value.get("path")
    expected = value.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected, str):
        raise ValueError(f"{label} record must contain path and sha256")
    _reject_protected_truth_paths((Path(path_value),))
    path = REPO_ROOT / path_value
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"{label} hash mismatch: {path_value}")
    return {"path": path_value, "sha256": expected}


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _file_record(path: Path) -> dict[str, str]:
    path = path.resolve()
    return {"path": _display_path(path), "sha256": _sha256(path)}


def _display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
