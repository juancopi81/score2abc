"""Resolve leading note/rest ambiguity after review-trained notehead selection.

Dense proposals recover every reviewed S1+S7 notehead, but frozen system-8
errors show a narrower ambiguity: a large leading gap can contain either a
missed first note or a leading eighth rest. This arm resolves that ambiguity in
two GT-free inference steps:

1. recover one near-threshold notehead candidate in the leading gap;
2. otherwise insert a leading half-beat rest only when visual note durations
   leave exactly half a beat unfilled in the request meter.

Patch method, threshold, NMS, count bounds, and recovery parameters are chosen
from S1+S7 review out-of-fold predictions. System 8 is frozen before its truth
is opened. System 3 is never accessed.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import spike_cap24_review_augmented_selector as cap  # noqa: E402
from scripts.experiments import spike_composed_melody_chain as composed  # noqa: E402
from scripts.experiments import spike_notehead_patch_templates as patches  # noqa: E402
from scripts.experiments import spike_review_augmented_selector as dense  # noqa: E402

DEFAULT_OUT_DIR = REPO_ROOT / "out"
DEFAULT_SLUG = patches.DEFAULT_SLUG
DEFAULT_REVIEWS_DIR = REPO_ROOT / "tests/fixtures/vlm_melody/notehead_reviews"
OUTPUT_SUBDIR = "meter_gap_resolver"
METHOD_PREFIX = "meter_gap_s1_s7"

# These grids are evaluated only against promoted S1+S7 notehead decisions.
RECOVERY_GAP_GRID = (2.75, 3.0, 3.25, 3.5)
RECOVERY_MARGIN_GRID = (0.0, 0.0025, 0.005, 0.01, 0.02)
MAX_METER_NOTE_COUNT = 6
LEADING_REST_GAP_SPACES = 3.0
HALF_BEAT = 0.5


@dataclass(frozen=True)
class RecoverySelection:
    leading_gap_spaces: float
    score_margin: float
    metrics: dict[str, Any]
    base_metrics: dict[str, Any]
    searched: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class GapAwareSelectorModel:
    base: dense.DenseSelectorModel
    recovery: RecoverySelection

    @property
    def learned_threshold(self) -> float:
        return self.base.learned_threshold

    @property
    def learned_count(self) -> int:
        return self.base.learned_count

    def probability(self, score: float) -> float:
        return self.base.probability(score)

    def rank(
        self,
        measure: patches.UnlabeledMeasure,
        *,
        selection_mode: str = dense.SELECTION_MODE,
    ) -> tuple[list[dict[str, Any]], list[patches.CandidatePatch]]:
        rows, selected = self.base.rank(measure, selection_mode=selection_mode)
        scores = {str(row["candidate_id"]): float(row["score"]) for row in rows}
        recovered = _recover_leading_candidate(
            measure,
            selected,
            scores,
            threshold=self.learned_threshold,
            leading_gap_spaces=self.recovery.leading_gap_spaces,
            score_margin=self.recovery.score_margin,
            maximum_selected_count=MAX_METER_NOTE_COUNT,
            nms_x_spaces=self.base.nms_x_spaces,
        )
        if recovered is None:
            return rows, selected

        selected = sorted(
            [*selected, recovered],
            key=lambda candidate: (candidate.center_x, candidate.center_y, candidate.id),
        )
        for row in rows:
            is_recovered = str(row["candidate_id"]) == recovered.id
            row["selected_by_gap_recovery"] = is_recovered
            if is_recovered:
                row["selected"] = True
        return rows, selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--reviews-dir", type=Path, default=DEFAULT_REVIEWS_DIR)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
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
    development_requests = dense._read_jsonl(development_requests_path)
    validation_requests = dense._read_jsonl(validation_requests_path)
    training_requests = dense._select_requests(
        [*development_requests, *validation_requests], dense.TRAINING_TARGETS
    )
    system8_requests = dense._select_requests(validation_requests, dense.VALIDATION_TARGETS)

    # Complete all image-derived proposals before opening review fixtures.
    training_unlabeled = [
        dense._prepare_dense_measure(request, out_dir=out_dir) for request in training_requests
    ]
    validation_unlabeled = [
        dense._prepare_dense_measure(request, out_dir=out_dir) for request in system8_requests
    ]
    training = [
        dense._attach_review(request, measure, reviews_dir=reviews_dir, slug=slug)
        for request, measure in zip(training_requests, training_unlabeled, strict=True)
    ]

    methods = []
    for patch_spec in patches.PATCH_SPECS:
        oof_scores = cap._out_of_fold_scores(
            training,
            patch_id=patch_spec.id,
            scorer_kind="class_template",
        )
        selection = dense._select_training_configuration(training, oof_scores)
        recovery = _select_recovery_configuration(training, oof_scores, selection=selection)
        methods.append(
            {
                "method_id": f"class_template__{patch_spec.id}",
                "patch_id": patch_spec.id,
                "scorer_kind": "class_template",
                "selection": selection,
                "recovery": recovery,
                "oof_scores": oof_scores,
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
    method_id = f"{METHOD_PREFIX}__{winner['method_id']}"
    full_scorer = patches._fit_patch_scorer(
        [row for example in training for row in example.measure.rows],
        patch_id=str(winner["patch_id"]),
        scorer_kind=str(winner["scorer_kind"]),
    )
    selection = winner["selection"]
    base_model = dense.DenseSelectorModel(
        scorer=full_scorer,  # type: ignore[arg-type]
        learned_threshold=float(selection["threshold"]),
        threshold_training_metrics=dict(selection["metrics"]),
        training_keys=tuple(example.key for example in training),
        training_positive_count=sum(example.true_note_count for example in training),
        learned_count=int(statistics.median(example.true_note_count for example in training)),
        nms_x_spaces=float(selection["nms_x_spaces"]),
        minimum_selected_count=int(selection["minimum_selected_count"]),
        maximum_selected_count=int(selection["maximum_selected_count"]),
    )
    model = GapAwareSelectorModel(base=base_model, recovery=winner["recovery"])
    pitch_cv = dense._evaluate_pitch_methods(training)
    pitch_method = str(pitch_cv["selected_method"])
    accidental_model = (
        dense._fit_accidental_model(training) if pitch_method == "accidental_knn" else None
    )
    pitch_predictor = dense._build_pitch_predictor(accidental_model)

    training_snapshot_path = output_dir / "training_selection.json"
    training_snapshot = {
        "status": "selected_before_validation_prediction",
        "training_targets": [example.key for example in training],
        "review_hashes": [example.measure.review_sha256 for example in training],
        "proposal_recall": dense._proposal_recall(training),
        "winner": _serializable_method(winner),
        "method_search": [_serializable_method(method) for method in methods],
        "pitch_oof": pitch_cv,
        "meter_gap_rule": {
            "leading_rest_gap_spaces": LEADING_REST_GAP_SPACES,
            "required_visual_deficit_beats": HALF_BEAT,
            "source": "fixed meter/layout rule; no canonical rhythm labels used",
        },
        "validation_request_sha256": dense._sha256(validation_requests_path),
    }
    training_snapshot_path.write_text(
        json.dumps(training_snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    validation_composed = [
        _compose_with_meter_gap_resolver(
            request,
            measure,
            model,
            out_dir=out_dir,
            selector_method_id=method_id,
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
        selector_mode="s1_s7_review_oof_meter_gap_resolver",
        selector_method_id=method_id,
    )
    validation_metrics = composed.evaluate_frozen_split(benchmark_dir, artifacts=artifacts)
    historical = dense._historical_system8_metrics(benchmark_dir)
    gate = dense._validation_gate(validation_metrics["summary"], historical)

    report_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    report = {
        "schema_version": 1,
        "kind": "meter_gap_resolver_spike",
        "slug": slug,
        "protocol": {
            "training": "promoted S1M1-4 and S7M1-7 coordinate/pitch reviews only",
            "model_selection": "patch/config/recovery selected on S1+S7 OOF notehead labels",
            "rhythm_repair": "request meter plus inferred visual durations and leading gap only",
            "validation": "freeze S8M1-7 before evaluation",
            "heldout": "not accessed; system 3 is consumed",
        },
        "training": {
            "proposal_recall": dense._proposal_recall(training),
            "winner": _serializable_method(winner),
            "method_search": [_serializable_method(method) for method in methods],
            "pitch_oof": pitch_cv,
        },
        "selector": {"method_id": method_id, "pitch_method": pitch_method},
        "leakage_audit": {
            "canonical_training_rhythm_used": False,
            "validation_truth_accessed_after_freeze": True,
            "prediction_freeze_sha256": artifacts.freeze_sha256,
            "system3_accessed": False,
            "interpretation": (
                "System 8 is consumed model-selection evidence, not a fresh heldout claim."
            ),
        },
        "validation": {
            "metrics": validation_metrics,
            "historical_threshold_selector_same_targets": historical,
            "per_measure_counts": composed._per_measure_counts(validation_composed),
            "per_measure_repairs": [_repair_summary(item) for item in validation_composed],
            "artifacts": composed._artifact_summary(artifacts),
        },
        "validation_gate": gate,
        "artifacts": {
            "report_json": dense._display_path(report_path),
            "report_markdown": dense._display_path(markdown_path),
            "training_selection": dense._display_path(training_snapshot_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report, markdown_path)
    return report


def _select_recovery_configuration(
    examples: Sequence[dense.TrainingExample],
    scores: Mapping[tuple[str, str], float],
    *,
    selection: Mapping[str, Any],
) -> RecoverySelection:
    threshold = float(selection["threshold"])
    base_rows = [
        dense._selection_metrics(
            example,
            dense._select_candidates(
                example.measure,
                {row.id: scores[(example.key, row.id)] for row in example.measure.rows},
                threshold=threshold,
                nms_x_spaces=float(selection["nms_x_spaces"]),
                minimum_selected_count=int(selection["minimum_selected_count"]),
                maximum_selected_count=int(selection["maximum_selected_count"]),
            ),
        )
        for example in examples
    ]
    base_metrics = dense._aggregate_selection_metrics(base_rows)
    searched = []
    for leading_gap_spaces in RECOVERY_GAP_GRID:
        for score_margin in RECOVERY_MARGIN_GRID:
            metric_rows = []
            recovered_count = 0
            for example in examples:
                measure_scores = {
                    row.id: scores[(example.key, row.id)] for row in example.measure.rows
                }
                selected = dense._select_candidates(
                    example.measure,
                    measure_scores,
                    threshold=threshold,
                    nms_x_spaces=float(selection["nms_x_spaces"]),
                    minimum_selected_count=int(selection["minimum_selected_count"]),
                    maximum_selected_count=int(selection["maximum_selected_count"]),
                )
                recovered = _recover_leading_candidate(
                    example.measure,
                    selected,
                    measure_scores,
                    threshold=threshold,
                    leading_gap_spaces=leading_gap_spaces,
                    score_margin=score_margin,
                    maximum_selected_count=MAX_METER_NOTE_COUNT,
                    nms_x_spaces=float(selection["nms_x_spaces"]),
                )
                if recovered is not None:
                    selected = [*selected, recovered]
                    recovered_count += 1
                metric_rows.append(dense._selection_metrics(example, selected))
            metrics = dense._aggregate_selection_metrics(metric_rows)
            searched.append(
                {
                    "leading_gap_spaces": leading_gap_spaces,
                    "score_margin": score_margin,
                    "recovered_count": recovered_count,
                    "metrics": metrics,
                }
            )
    winner = max(
        searched,
        key=lambda row: (
            float(row["metrics"]["f1"]),
            float(row["metrics"]["recall"]),
            float(row["metrics"]["precision"]),
            int(row["metrics"]["exact_measures"]),
            -int(row["recovered_count"]),
            -float(row["score_margin"]),
            float(row["leading_gap_spaces"]),
        ),
    )
    return RecoverySelection(
        leading_gap_spaces=float(winner["leading_gap_spaces"]),
        score_margin=float(winner["score_margin"]),
        metrics=dict(winner["metrics"]),
        base_metrics=base_metrics,
        searched=tuple(searched),
    )


def _recover_leading_candidate(
    measure: patches.UnlabeledMeasure | patches.MeasureData,
    selected: Sequence[patches.CandidatePatch],
    scores: Mapping[str, float],
    *,
    threshold: float,
    leading_gap_spaces: float,
    score_margin: float,
    maximum_selected_count: int,
    nms_x_spaces: float,
) -> patches.CandidatePatch | None:
    if not selected or len(selected) >= maximum_selected_count or score_margin <= 0:
        return None
    ordered = sorted(selected, key=lambda candidate: (candidate.center_x, candidate.center_y))
    first = ordered[0]
    spacing = float(measure.staff_spacing)
    if first.center_x / spacing < leading_gap_spaces:
        return None
    candidates = [
        row.candidate if isinstance(row, patches.LabeledCandidate) else row
        for row in (
            measure.rows if isinstance(measure, patches.MeasureData) else measure.candidates
        )
    ]
    selected_ids = {candidate.id for candidate in selected}
    eligible = [
        candidate
        for candidate in candidates
        if candidate.id not in selected_ids
        and candidate.center_x < first.center_x - spacing * nms_x_spaces
        and scores[candidate.id] >= threshold - score_margin
        and all(
            abs(candidate.center_x - other.center_x) >= spacing * nms_x_spaces for other in selected
        )
    ]
    return max(
        eligible,
        key=lambda candidate: (scores[candidate.id], -candidate.rank, candidate.id),
        default=None,
    )


def _compose_with_meter_gap_resolver(
    request: Mapping[str, Any],
    measure: patches.UnlabeledMeasure,
    model: GapAwareSelectorModel,
    *,
    out_dir: Path,
    selector_method_id: str,
    pitch_predictor: composed.PitchPredictor,
) -> composed.ComposedMeasure:
    item = composed.compose_measure(
        request,
        measure,
        model,  # type: ignore[arg-type]
        out_dir=out_dir,
        selection_mode=dense.SELECTION_MODE,
        selector_method_id=selector_method_id,
        pitch_predictor=pitch_predictor,
    )
    expected_beats = float(request["allowed_context"]["expected_measure_beats"])
    allow_pickup = bool(request["allowed_context"].get("allow_pickup", False))
    note_symbols = [dict(symbol) for symbol in item.visual_symbols if symbol["kind"] == "note"]
    note_extent = sum(float(symbol["duration_beats"]) for symbol in note_symbols)
    first_gap_spaces = (
        float(item.anchors[0]["center"]["x"]) / item.staff_spacing if item.anchors else 0.0
    )
    should_insert_rest = _should_insert_leading_rest(
        expected_beats=expected_beats,
        visual_note_extent_beats=note_extent,
        first_anchor_gap_spaces=first_gap_spaces,
        allow_pickup=allow_pickup,
        has_anchors=bool(item.anchors),
    )
    synthetic_rest = None
    if should_insert_rest:
        first_x = float(item.anchors[0]["center"]["x"])
        leading_gap_right = max(item.staff_spacing, first_x - 0.6 * item.staff_spacing)
        center_x = max(item.staff_spacing, leading_gap_right / 2.0)
        staff_lines = [int(value) for value in request["staff_geometry"]["raw_staff_lines_y_px"]]
        synthetic_rest = {
            "role": "leading",
            "center_x": round(center_x, 3),
            "duration_beats": HALF_BEAT,
            "bbox": {
                "left": round(center_x - 0.35 * item.staff_spacing),
                "top": round(staff_lines[1] - 0.45 * item.staff_spacing),
                "right": round(center_x + 0.35 * item.staff_spacing),
                "bottom": round(staff_lines[3] + 0.45 * item.staff_spacing),
            },
            "area_staff_squared": 0.0,
            "height_staff_spacing": 0.0,
            "width_staff_spacing": 0.0,
            "evidence": "request_meter_plus_leading_gap",
            "visual_component_detected": False,
        }
        item.rest_features = [synthetic_rest]
        item.visual_symbols = composed.rhythm.build_visual_symbols(
            item.groups,
            item.anchor_features,
            item.rest_features,
        )

    decoded, status = composed.rhythm.decode_meter(
        item.visual_symbols,
        expected_beats=expected_beats,
        allow_pickup=allow_pickup,
    )
    repaired_extent = sum(float(symbol["duration_beats"]) for symbol in decoded)
    if not math.isclose(repaired_extent, expected_beats) and not (
        allow_pickup and repaired_extent <= expected_beats
    ):
        return dense._compose_with_meter_fallback(
            request,
            measure,
            model,  # type: ignore[arg-type]
            out_dir=out_dir,
            selection_mode=dense.SELECTION_MODE,
            selector_method_id=selector_method_id,
            pitch_predictor=pitch_predictor,
        )
    item.decoded_symbols = decoded
    item.prediction = composed.rhythm.symbols_to_hypothesis(
        decoded,
        identity=request["identity"],
        decoder_status=f"meter_gap_resolver:{status}",
    )
    provenance = {
        "notehead_selector": selector_method_id,
        "selection_mode": dense.SELECTION_MODE,
        "automatic_anchor_count": len(item.anchors),
        "review_anchors_used": False,
        "truth_used": False,
        "learned_score_threshold": round(model.learned_threshold, 9),
        "learned_probability_threshold": round(model.probability(model.learned_threshold), 9),
        "threshold_fit_from_training_reviews_only": True,
        "leading_gap_recovery": {
            "candidate_ids": [
                str(row["candidate_id"])
                for row in item.candidate_predictions
                if row.get("selected_by_gap_recovery")
            ],
            "leading_gap_spaces": model.recovery.leading_gap_spaces,
            "score_margin": model.recovery.score_margin,
        },
        "meter_gap_repair": {
            "applied": synthetic_rest is not None,
            "first_anchor_gap_spaces": round(first_gap_spaces, 6),
            "visual_note_extent_beats": round(note_extent, 6),
            "inserted_leading_rest_beats": HALF_BEAT if synthetic_rest is not None else 0.0,
            "canonical_rhythm_used": False,
        },
    }
    item.prediction["inference_provenance"] = provenance
    return item


def _should_insert_leading_rest(
    *,
    expected_beats: float,
    visual_note_extent_beats: float,
    first_anchor_gap_spaces: float,
    allow_pickup: bool,
    has_anchors: bool,
) -> bool:
    return (
        not allow_pickup
        and has_anchors
        and math.isclose(expected_beats - visual_note_extent_beats, HALF_BEAT)
        and first_anchor_gap_spaces >= LEADING_REST_GAP_SPACES
    )


def _serializable_method(method: Mapping[str, Any]) -> dict[str, Any]:
    recovery: RecoverySelection = method["recovery"]
    return {
        "method_id": method["method_id"],
        "patch_id": method["patch_id"],
        "scorer_kind": method["scorer_kind"],
        "selection": method["selection"],
        "recovery": {
            "leading_gap_spaces": recovery.leading_gap_spaces,
            "score_margin": recovery.score_margin,
            "base_metrics": recovery.base_metrics,
            "metrics": recovery.metrics,
            "searched": list(recovery.searched),
        },
    }


def _repair_summary(item: composed.ComposedMeasure) -> dict[str, Any]:
    provenance = item.prediction["inference_provenance"]
    identity = item.request["identity"]
    return {
        "system_measure_index": int(identity["system_measure_index"]),
        "anchor_count": len(item.anchors),
        "leading_gap_recovery": provenance["leading_gap_recovery"],
        "meter_gap_repair": provenance["meter_gap_repair"],
    }


def _write_markdown(report: Mapping[str, Any], path: Path) -> None:
    winner = report["training"]["winner"]
    summary = report["validation"]["metrics"]["summary"]
    historical = report["validation"]["historical_threshold_selector_same_targets"]
    lines = [
        "# Meter-Gap Resolver",
        "",
        f"- Training proposal recall: {report['training']['proposal_recall']['recall']:.3f}",
        f"- OOF winner: {winner['method_id']}",
        f"- OOF base candidate F1: {winner['recovery']['base_metrics']['f1']:.3f}",
        f"- OOF recovered candidate F1: {winner['recovery']['metrics']['f1']:.3f}",
        f"- Frozen system-8 note F1: {summary['note_f1']:.3f}",
        f"- Frozen system-8 rest F1: {summary['rest_f1']:.3f}",
        f"- Frozen system-8 ordered pitch: {summary['ordered_pitch_accuracy']:.3f}",
        f"- Historical same-target note F1: {historical['summary']['note_f1']:.3f}",
        f"- Gate: {report['validation_gate']['status']}",
        "",
        "System 8 is consumed model-selection evidence. System 3 was not read.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
