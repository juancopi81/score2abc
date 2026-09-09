"""Build a comparable error breakdown from immutable heldout score reports.

The input manifest contains portable copies of four independent-score evaluation
reports. This command validates their hashes, normalizes only metrics supported
by each report, and writes a create-once diagnostic report. It never reruns
inference or opens MusicXML.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_MANIFEST = (
    REPO_ROOT / "tests/fixtures/vlm_melody/cross_score_error_breakdown/manifest.json"
)
DEFAULT_OUTPUT = Path("out/vlm_melody_cross_score_error_breakdown/v2")
INPUT_KIND = "vlm_melody_cross_score_error_breakdown_input_manifest"
REPORT_KIND = "vlm_melody_cross_score_error_breakdown"
OUTPUT_MANIFEST_KIND = f"{REPORT_KIND}_manifest"
MIN_TARGET_SUPPORT_SCORES = 2
MIN_TARGET_OPPORTUNITIES = 20
RANKING_ELIGIBLE_STATUSES = {"scored", "scored_on_full_event_subset"}
SCOPE_BY_REPORT_KIND = {
    "one_shot_second_score_evaluation": "full_event",
    "fifth_score_full_event_one_shot_evaluation": "full_event",
    "third_score_pitch_only_one_shot_evaluation": "pitch_only",
    "fourth_score_pitch_only_one_shot_evaluation": "pitch_only",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=DEFAULT_INPUT_MANIFEST,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args(argv)
    try:
        destination = build_cross_score_error_breakdown(
            input_manifest=args.input_manifest,
            output_dir=args.output_dir,
        )
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(destination)
    return 0


def build_cross_score_error_breakdown(
    *,
    input_manifest: Path = DEFAULT_INPUT_MANIFEST,
    output_dir: Path = DEFAULT_OUTPUT,
) -> Path:
    """Normalize hash-pinned heldout reports and publish one diagnostic bundle."""
    manifest_path = input_manifest.resolve()
    manifest = _load_object(manifest_path, "Input manifest")
    report_specs = _validate_input_manifest(manifest, manifest_path=manifest_path)
    normalized = [
        _normalize_report(spec, _load_object(spec["path"], f"{spec['score_id']} report"))
        for spec in report_specs
    ]
    aggregate = _aggregate(normalized)
    stage_breakdown = _stage_breakdown(normalized, aggregate)
    decision = _select_next_target(stage_breakdown)
    report = {
        "schema_version": 1,
        "kind": REPORT_KIND,
        "status": "diagnostic_heldout_report_only",
        "evidence": {
            "split_status": manifest["split_status"],
            "source_manifest": _file_record(manifest_path),
            "source_report_count": len(report_specs),
            "excluded_evidence": manifest["exclusions"],
            "predictions_rerun": False,
            "musicxml_opened": False,
        },
        "scores": normalized,
        "aggregate": aggregate,
        "stage_breakdown": stage_breakdown,
        "next_engineering_target": decision,
        "interpretation_limits": [
            "Stage error counts are overlapping diagnostic dimensions and must not be added.",
            "Note-count mismatch does not distinguish candidate coverage from selector ranking.",
            "Pitch mismatch combines staff geometry, key state, accidentals, and ordering.",
            "Onset, duration, rests, meter, and full exact measures are scored only on "
            "Carrizal and Coqueteos.",
            "The report ranks engineering leverage; it is not a dataset-level accuracy claim.",
        ],
    }

    destination = output_dir.resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite cross-score report: {destination}")
    temp_dir = destination.with_name(f".{destination.name}.tmp")
    if temp_dir.exists() or temp_dir.is_symlink():
        raise FileExistsError(f"Refusing stale cross-score temp directory: {temp_dir}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir()
    try:
        report_path = temp_dir / "report.json"
        _write_json(report_path, report)
        summary_path = temp_dir / "summary.md"
        _write_text(summary_path, _render_markdown(report))
        _write_json(
            temp_dir / "manifest.json",
            {
                "schema_version": 1,
                "kind": OUTPUT_MANIFEST_KIND,
                "create_once": True,
                "report": _local_record(report_path, temp_dir),
                "summary": _local_record(summary_path, temp_dir),
                "selected_target": decision["selected_target"],
            },
        )
        temp_dir.rename(destination)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return destination


def _validate_input_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> list[dict[str, Any]]:
    if manifest.get("kind") != INPUT_KIND:
        raise ValueError(f"Unexpected input manifest kind: {manifest.get('kind')!r}")
    if manifest.get("split_status") != "independent_score_heldout_reports":
        raise ValueError("Input manifest is not independent-score heldout evidence")
    if manifest.get("eligible_for_end_to_end_accuracy_claim") is not False:
        raise ValueError("Input manifest cannot support a dataset-level accuracy claim")
    exclusions = manifest.get("exclusions")
    if not isinstance(exclusions, list) or not exclusions:
        raise ValueError("Input manifest must document excluded evidence")
    raw_reports = manifest.get("reports")
    if not isinstance(raw_reports, list) or len(raw_reports) < 2:
        raise ValueError("Input manifest must contain at least two reports")

    result = []
    seen: set[str] = set()
    allowed_scopes = {"full_event", "pitch_only"}
    for index, raw in enumerate(raw_reports):
        label = f"Input manifest reports[{index}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} must be an object")
        score_id = _required_string(raw, "score_id", label)
        if score_id in seen:
            raise ValueError(f"Duplicate score_id in input manifest: {score_id}")
        seen.add(score_id)
        scope = _required_string(raw, "scope", label)
        if scope not in allowed_scopes:
            raise ValueError(f"Unsupported report scope: {scope!r}")
        expected_kind = _required_string(raw, "expected_kind", label)
        expected_scope = SCOPE_BY_REPORT_KIND.get(expected_kind)
        if expected_scope != scope:
            raise ValueError(
                f"Scope mismatch in {label}: report kind {expected_kind!r} "
                f"requires {expected_scope!r}, got {scope!r}"
            )
        digest = _required_sha256(raw, "sha256", label)
        target_slug = _required_string(raw, "target_slug", label)
        target_system_index = _positive_int(
            raw.get("target_system_index"),
            f"{label}.target_system_index",
        )
        expected_crop_count = _positive_int(
            raw.get("expected_automatic_crop_count"),
            f"{label}.expected_automatic_crop_count",
        )
        expected_physical_measures = _positive_int_list(
            raw.get("expected_physical_measure_numbers")
        )
        if expected_physical_measures != list(range(1, len(expected_physical_measures) + 1)):
            raise ValueError(f"{label}.expected_physical_measure_numbers must be contiguous from 1")
        relative = Path(_required_string(raw, "path", label))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe report path in {label}: {relative}")
        path = (manifest_path.parent / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Input report is missing: {path}")
        if _sha256(path) != digest:
            raise ValueError(f"Input report hash mismatch: {path}")
        payload = _load_object(path, f"{score_id} report")
        if payload.get("kind") != expected_kind:
            raise ValueError(
                f"Input report kind mismatch for {score_id}: "
                f"expected {expected_kind!r}, got {payload.get('kind')!r}"
            )
        _validate_report_identity(
            payload,
            expected_kind=expected_kind,
            target_slug=target_slug,
            target_system_index=target_system_index,
        )
        result.append(
            {
                "score_id": score_id,
                "scope": scope,
                "expected_kind": expected_kind,
                "target_slug": target_slug,
                "target_system_index": target_system_index,
                "expected_automatic_crop_count": expected_crop_count,
                "expected_physical_measure_numbers": expected_physical_measures,
                "path": path,
                "record": {"path": relative.as_posix(), "sha256": digest},
            }
        )
    return result


def _normalize_report(spec: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    kind = str(report["kind"])
    if kind == "one_shot_second_score_evaluation":
        return _normalize_carrizal(spec, report)
    if kind == "fifth_score_full_event_one_shot_evaluation":
        return _normalize_coqueteos(spec, report)
    if kind in {
        "third_score_pitch_only_one_shot_evaluation",
        "fourth_score_pitch_only_one_shot_evaluation",
    }:
        return _normalize_pitch_only(spec, report)
    raise ValueError(f"Unsupported heldout report kind: {kind!r}")


def _normalize_carrizal(
    spec: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    if report.get("status") != "evaluated_once_after_frozen_predictions":
        raise ValueError("Carrizal report is not a completed frozen evaluation")
    metrics = _required_object(report, "metrics", "Carrizal report")
    rows = _object_rows(metrics.get("results"), "Carrizal metric results")
    segmentation = _required_object(report, "segmentation", "Carrizal report")
    mapping = _required_object(
        segmentation,
        "automatic_crop_to_physical_measures",
        "Carrizal segmentation",
    )
    expected_keys = {str(index) for index in range(1, len(rows) + 1)}
    if set(mapping) != expected_keys:
        raise ValueError("Carrizal crop mapping indices do not match metric rows")
    physical_lists = []
    for index, row in enumerate(rows, start=1):
        identity = _required_object(row, "identity", "Carrizal metric row")
        if _positive_int(identity.get("system_measure_index"), "system_measure_index") != index:
            raise ValueError("Carrizal metric rows are not in automatic-crop order")
        physical_lists.append(_positive_int_list(mapping[str(index)]))
    _validate_physical_mapping(spec, physical_lists)
    return _normalize_full_event(
        spec,
        report,
        rows=rows,
        physical_lists=physical_lists,
        mapping_mode="post_review_explicit_mapping",
        mapping_defined_after_truth=bool(
            segmentation.get("mapping_defined_after_independent_transcription_review")
        ),
        meter=None,
    )


def _normalize_coqueteos(
    spec: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    if report.get("status") != "evaluated_exactly_once_after_frozen_predictions":
        raise ValueError("Coqueteos report is not a completed frozen evaluation")
    metrics = _required_object(report, "metrics", "Coqueteos report")
    rows = _object_rows(metrics.get("results"), "Coqueteos metric results")
    for row in rows:
        identity = _required_object(row, "identity", "Coqueteos metric row")
        if identity.get("slug") != spec["target_slug"]:
            raise ValueError("Coqueteos metric row slug does not match the input manifest")
        if identity.get("system_index") != spec["target_system_index"]:
            raise ValueError("Coqueteos metric row system does not match the input manifest")
    meter = _required_object(report, "meter", "Coqueteos report")
    meter_rows = _object_rows(meter.get("crops"), "Coqueteos meter crops")
    if len(meter_rows) != len(rows):
        raise ValueError("Coqueteos metric and meter crop counts differ")
    meter_by_index = _rows_by_index(
        meter_rows,
        index_getter=lambda row: row.get("automatic_crop_index"),
        label="Coqueteos meter crops",
    )
    metric_by_index = _rows_by_index(
        rows,
        index_getter=lambda row: _required_object(
            row,
            "identity",
            "Coqueteos metric row",
        ).get("automatic_measure_index"),
        label="Coqueteos metric rows",
    )
    expected_indices = set(range(1, len(rows) + 1))
    if set(meter_by_index) != expected_indices or set(metric_by_index) != expected_indices:
        raise ValueError("Coqueteos crop indices must be contiguous from 1")
    rows = [metric_by_index[index] for index in sorted(expected_indices)]
    physical_lists = [
        _positive_int_list(meter_by_index[index].get("physical_measure_numbers"))
        for index in sorted(expected_indices)
    ]
    _validate_physical_mapping(spec, physical_lists)
    return _normalize_full_event(
        spec,
        report,
        rows=rows,
        physical_lists=physical_lists,
        mapping_mode=str(report.get("mapping_mode")),
        mapping_defined_after_truth=True,
        meter=meter,
    )


def _normalize_full_event(
    spec: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    rows: Sequence[Mapping[str, Any]],
    physical_lists: Sequence[Sequence[int]],
    mapping_mode: str,
    mapping_defined_after_truth: bool,
    meter: Mapping[str, Any] | None,
) -> dict[str, Any]:
    counts = _count_metrics(rows)
    pitch_matches = sum(_nonnegative_int(row.get("pitch_matches"), "pitch_matches") for row in rows)
    onset_matches = sum(_nonnegative_int(row.get("onset_matches"), "onset_matches") for row in rows)
    duration_matches = sum(
        _nonnegative_int(row.get("duration_matches"), "duration_matches") for row in rows
    )
    for label, matches in (
        ("pitch", pitch_matches),
        ("onset", onset_matches),
        ("duration", duration_matches),
    ):
        if matches > counts["matched_capacity"]:
            raise ValueError(f"{label} matches exceed note-count capacity")
    rest_tp = sum(_nonnegative_int(row.get("rest_tp"), "rest_tp") for row in rows)
    rest_fp = sum(_nonnegative_int(row.get("rest_fp"), "rest_fp") for row in rows)
    rest_fn = sum(_nonnegative_int(row.get("rest_fn"), "rest_fn") for row in rows)
    exact_measures = sum(row.get("exact_measure") is True for row in rows)
    segmentation = _segmentation_metrics(physical_lists)
    units = [
        _full_event_unit(index, row, physical_measure_numbers)
        for index, (row, physical_measure_numbers) in enumerate(
            zip(rows, physical_lists, strict=True),
            start=1,
        )
    ]
    meter_metrics: dict[str, Any]
    if meter is None:
        meter_metrics = {"status": "not_scored_not_reported"}
    else:
        meter_summary = _required_object(meter, "summary", "Meter")
        valid = _nonnegative_int(
            meter_summary.get("valid_prediction_crops"),
            "meter valid_prediction_crops",
        )
        total = _positive_int(
            meter_summary.get("prediction_crop_count"),
            "meter prediction_crop_count",
        )
        meter_metrics = {
            "status": "scored",
            "valid_crop_count": valid,
            "invalid_crop_count": total - valid,
            "crop_count": total,
            "valid_crop_rate": _ratio(valid, total),
        }
    return {
        "score_id": spec["score_id"],
        "scope": spec["scope"],
        "evidence_tier": "independent_score_heldout",
        "source": {
            "kind": report["kind"],
            "status": report["status"],
            "report": _file_record(spec["path"]),
        },
        "segmentation": {
            **segmentation,
            "mapping_mode": mapping_mode,
            "mapping_defined_after_truth": mapping_defined_after_truth,
        },
        "note_count": counts,
        "pitch": _conditional_accuracy(pitch_matches, counts["matched_capacity"]),
        "units": units,
        "rhythm": {
            "onset": _conditional_accuracy(onset_matches, counts["matched_capacity"]),
            "duration": _conditional_accuracy(duration_matches, counts["matched_capacity"]),
            "rests": _rest_metrics(rest_tp, rest_fp, rest_fn),
            "meter": meter_metrics,
            "exact_full_measures": {
                "status": "scored",
                "count": exact_measures,
                "measure_crop_count": len(rows),
                "rate": _ratio(exact_measures, len(rows)),
            },
        },
    }


def _normalize_pitch_only(
    spec: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    if report.get("status") != "evaluated_exactly_once_after_frozen_predictions":
        raise ValueError(f"{spec['score_id']} report is not a completed frozen evaluation")
    metrics = _required_object(report, "metrics", f"{spec['score_id']} report")
    rows = _object_rows(metrics.get("crops"), f"{spec['score_id']} metric crops")
    counts = _count_metrics(rows, pred_key="predicted_note_count", truth_key="truth_note_count")
    pitch_matches = sum(
        _nonnegative_int(
            _required_object(row, "alignment", "Pitch crop").get("exact_pitch_matches"),
            "exact_pitch_matches",
        )
        for row in rows
    )
    if pitch_matches > counts["matched_capacity"]:
        raise ValueError("Exact pitch matches exceed note-count capacity")
    crop_by_index = _rows_by_index(
        rows,
        index_getter=lambda row: row.get("automatic_crop_index"),
        label=f"{spec['score_id']} pitch crops",
    )
    expected_indices = set(range(1, len(rows) + 1))
    if set(crop_by_index) != expected_indices:
        raise ValueError(f"{spec['score_id']} crop indices must be contiguous from 1")
    rows = [crop_by_index[index] for index in sorted(expected_indices)]
    physical_lists = [_positive_int_list(row.get("physical_measure_numbers")) for row in rows]
    _validate_physical_mapping(spec, physical_lists)
    exact_crops = sum(row.get("exact_automatic_crop") is True for row in rows)
    units = [
        _pitch_only_unit(index, row, physical_measure_numbers)
        for index, (row, physical_measure_numbers) in enumerate(
            zip(rows, physical_lists, strict=True),
            start=1,
        )
    ]
    return {
        "score_id": spec["score_id"],
        "scope": spec["scope"],
        "evidence_tier": "independent_score_heldout",
        "source": {
            "kind": report["kind"],
            "status": report["status"],
            "report": _file_record(spec["path"]),
        },
        "segmentation": {
            **_segmentation_metrics(physical_lists),
            "mapping_mode": str(report.get("mapping_mode")),
            "mapping_defined_after_truth": True,
        },
        "note_count": counts,
        "pitch": {
            **_conditional_accuracy(pitch_matches, counts["matched_capacity"]),
            "exact_pitch_crop_count": exact_crops,
            "pitch_crop_count": len(rows),
            "exact_pitch_crop_rate": _ratio(exact_crops, len(rows)),
        },
        "units": units,
        "rhythm": {
            "onset": {"status": "not_scored_missing_frozen_context"},
            "duration": {"status": "not_scored_missing_frozen_context"},
            "rests": {"status": "not_scored_missing_frozen_context"},
            "meter": {"status": "not_scored_missing_frozen_context"},
            "exact_full_measures": {"status": "not_scored_missing_frozen_context"},
        },
    }


def _count_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    pred_key: str = "pred_note_count",
    truth_key: str = "truth_note_count",
) -> dict[str, Any]:
    predicted = [_nonnegative_int(row.get(pred_key), pred_key) for row in rows]
    truth = [_nonnegative_int(row.get(truth_key), truth_key) for row in rows]
    pred_total = sum(predicted)
    truth_total = sum(truth)
    capacity = sum(min(pred, target) for pred, target in zip(predicted, truth, strict=True))
    surplus = sum(max(pred - target, 0) for pred, target in zip(predicted, truth, strict=True))
    deficit = sum(max(target - pred, 0) for pred, target in zip(predicted, truth, strict=True))
    return {
        "metric_semantics": (
            "count-capacity upper bound only; min(predicted, truth) does not imply "
            "note identity, pitch, or event matches"
        ),
        "predicted": pred_total,
        "truth": truth_total,
        "matched_capacity": capacity,
        "surplus": surplus,
        "deficit": deficit,
        "precision": _ratio(capacity, pred_total),
        "recall": _ratio(capacity, truth_total),
        "f1": _f1(capacity, pred_total, truth_total),
    }


def _segmentation_metrics(physical_lists: Sequence[Sequence[int]]) -> dict[str, Any]:
    flattened = [number for values in physical_lists for number in values]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Physical measure mapping contains duplicates")
    crop_count = len(physical_lists)
    physical_count = len(flattened)
    merged = sum(len(values) > 1 for values in physical_lists)
    missing_boundaries = sum(max(len(values) - 1, 0) for values in physical_lists)
    return {
        "automatic_crop_count": crop_count,
        "physical_measure_count": physical_count,
        "count_match": crop_count == physical_count,
        "one_to_one": merged == 0 and crop_count == physical_count,
        "merged_crop_count": merged,
        "missing_boundary_count": missing_boundaries,
        "mapping_limit": "split or empty automatic crops are not representable",
    }


def _conditional_accuracy(matches: int, capacity: int) -> dict[str, Any]:
    return {
        "status": "scored",
        "exact_matches": matches,
        "mismatches_within_count_capacity": capacity - matches,
        "matched_capacity": capacity,
        "conditional_accuracy": _ratio(matches, capacity),
    }


def _rest_metrics(tp: int, fp: int, fn: int) -> dict[str, Any]:
    return {
        "status": "scored",
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "truth_count": tp + fn,
        "predicted_count": tp + fp,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _binary_f1(tp, fp, fn),
    }


def _full_event_unit(
    automatic_crop_index: int,
    row: Mapping[str, Any],
    physical_measure_numbers: Sequence[int],
) -> dict[str, Any]:
    return {
        "automatic_crop_index": automatic_crop_index,
        "physical_measure_numbers": list(physical_measure_numbers),
        "segmentation_confounded": len(physical_measure_numbers) != 1,
        "predicted_note_count": _nonnegative_int(
            row.get("pred_note_count"),
            "pred_note_count",
        ),
        "truth_note_count": _nonnegative_int(
            row.get("truth_note_count"),
            "truth_note_count",
        ),
        "exact_pitch_matches": _nonnegative_int(
            row.get("pitch_matches"),
            "pitch_matches",
        ),
        "exact_onset_matches": _nonnegative_int(
            row.get("onset_matches"),
            "onset_matches",
        ),
        "exact_duration_matches": _nonnegative_int(
            row.get("duration_matches"),
            "duration_matches",
        ),
        "rest_tp": _nonnegative_int(row.get("rest_tp"), "rest_tp"),
        "rest_fp": _nonnegative_int(row.get("rest_fp"), "rest_fp"),
        "rest_fn": _nonnegative_int(row.get("rest_fn"), "rest_fn"),
        "exact_full_measure": row.get("exact_measure") is True,
        "scope": "full_event",
    }


def _pitch_only_unit(
    automatic_crop_index: int,
    row: Mapping[str, Any],
    physical_measure_numbers: Sequence[int],
) -> dict[str, Any]:
    alignment = _required_object(row, "alignment", "Pitch-only crop")
    return {
        "automatic_crop_index": automatic_crop_index,
        "physical_measure_numbers": list(physical_measure_numbers),
        "segmentation_confounded": len(physical_measure_numbers) != 1,
        "predicted_note_count": _nonnegative_int(
            row.get("predicted_note_count"),
            "predicted_note_count",
        ),
        "truth_note_count": _nonnegative_int(
            row.get("truth_note_count"),
            "truth_note_count",
        ),
        "exact_pitch_matches": _nonnegative_int(
            alignment.get("exact_pitch_matches"),
            "exact_pitch_matches",
        ),
        "exact_pitch_crop": row.get("exact_automatic_crop") is True,
        "scope": "pitch_only",
    }


def _aggregate(scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    units = [unit for score in scores for unit in score["units"]]
    clean_units = [unit for unit in units if unit["segmentation_confounded"] is False]
    segmentation = {
        "score_count": len(scores),
        "automatic_crop_count": sum(
            int(score["segmentation"]["automatic_crop_count"]) for score in scores
        ),
        "physical_measure_count": sum(
            int(score["segmentation"]["physical_measure_count"]) for score in scores
        ),
        "one_to_one_score_count": sum(
            score["segmentation"]["one_to_one"] is True for score in scores
        ),
        "merged_crop_count": sum(
            int(score["segmentation"]["merged_crop_count"]) for score in scores
        ),
        "missing_boundary_count": sum(
            int(score["segmentation"]["missing_boundary_count"]) for score in scores
        ),
        "confounded_crop_count": len(units) - len(clean_units),
        "root_cause_policy": (
            "exclude merged or otherwise non-one-to-one crops from downstream target ranking"
        ),
    }
    return {
        "independent_score_count": len(scores),
        "segmentation": segmentation,
        "all_units": _aggregate_units(units),
        "clean_one_to_one_units": _aggregate_units(clean_units),
    }


def _aggregate_units(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    predicted = sum(int(unit["predicted_note_count"]) for unit in units)
    truth = sum(int(unit["truth_note_count"]) for unit in units)
    capacity = sum(
        min(int(unit["predicted_note_count"]), int(unit["truth_note_count"])) for unit in units
    )
    count = {
        "metric_semantics": (
            "count-capacity upper bound only; min(predicted, truth) does not imply "
            "note identity, pitch, or event matches"
        ),
        "predicted": predicted,
        "truth": truth,
        "matched_capacity": capacity,
        "surplus": sum(
            max(int(unit["predicted_note_count"]) - int(unit["truth_note_count"]), 0)
            for unit in units
        ),
        "deficit": sum(
            max(int(unit["truth_note_count"]) - int(unit["predicted_note_count"]), 0)
            for unit in units
        ),
        "precision": _ratio(capacity, predicted),
        "recall": _ratio(capacity, truth),
        "f1": _f1(capacity, predicted, truth),
    }
    pitch_matches = sum(int(unit["exact_pitch_matches"]) for unit in units)
    if pitch_matches > capacity:
        raise ValueError("Aggregate pitch matches exceed note-count capacity")
    full_units = [unit for unit in units if unit["scope"] == "full_event"]
    return {
        "unit_count": len(units),
        "note_count": count,
        "pitch": _conditional_accuracy(pitch_matches, capacity),
        "full_event_subset": _aggregate_full_event_units(full_units),
    }


def _aggregate_full_event_units(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not units:
        return {"status": "not_scored"}
    predicted = sum(int(unit["predicted_note_count"]) for unit in units)
    truth = sum(int(unit["truth_note_count"]) for unit in units)
    capacity = sum(
        min(int(unit["predicted_note_count"]), int(unit["truth_note_count"])) for unit in units
    )
    count = {
        "metric_semantics": (
            "count-capacity upper bound only; min(predicted, truth) does not imply "
            "note identity, pitch, or event matches"
        ),
        "predicted": predicted,
        "truth": truth,
        "matched_capacity": capacity,
        "surplus": sum(
            max(int(unit["predicted_note_count"]) - int(unit["truth_note_count"]), 0)
            for unit in units
        ),
        "deficit": sum(
            max(int(unit["truth_note_count"]) - int(unit["predicted_note_count"]), 0)
            for unit in units
        ),
        "precision": _ratio(capacity, predicted),
        "recall": _ratio(capacity, truth),
        "f1": _f1(capacity, predicted, truth),
    }
    pitch_matches = sum(int(unit["exact_pitch_matches"]) for unit in units)
    onset_matches = sum(int(unit["exact_onset_matches"]) for unit in units)
    duration_matches = sum(int(unit["exact_duration_matches"]) for unit in units)
    rest_tp = sum(int(unit["rest_tp"]) for unit in units)
    rest_fp = sum(int(unit["rest_fp"]) for unit in units)
    rest_fn = sum(int(unit["rest_fn"]) for unit in units)
    exact_measures = sum(unit["exact_full_measure"] is True for unit in units)
    return {
        "status": "scored",
        "unit_count": len(units),
        "note_count": count,
        "pitch": _conditional_accuracy(pitch_matches, capacity),
        "onset": _conditional_accuracy(onset_matches, capacity),
        "duration": _conditional_accuracy(duration_matches, capacity),
        "rests": _rest_metrics(rest_tp, rest_fp, rest_fn),
        "exact_full_measures": {
            "count": exact_measures,
            "measure_crop_count": len(units),
            "rate": _ratio(exact_measures, len(units)),
        },
    }


def _stage_breakdown(
    scores: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    clean = aggregate["clean_one_to_one_units"]
    full = clean["full_event_subset"]
    count = clean["note_count"]
    pitch = clean["pitch"]
    stages = [
        {
            "stage": "segmentation",
            "status": "scored_but_confounded_units_excluded",
            "support_score_count": len(scores),
            "opportunity_count": aggregate["segmentation"]["physical_measure_count"],
            "observed_error_count": aggregate["segmentation"]["missing_boundary_count"],
            "observed_metric": "missing measure boundaries",
        },
        {
            "stage": "candidate_coverage",
            "status": "not_identifiable_from_frozen_reports",
            "support_score_count": 0,
            "opportunity_count": 0,
            "observed_error_count": 0,
            "observed_metric": "requires candidate-level truth matching",
        },
        {
            "stage": "note_count_output",
            "status": "scored_but_causally_ambiguous",
            "support_score_count": len(scores),
            "opportunity_count": count["truth"] + count["surplus"],
            "observed_error_count": count["deficit"] + count["surplus"],
            "observed_metric": (
                "per-crop note-count deficits plus surpluses; candidate coverage and "
                "selector ranking are not separable"
            ),
        },
        {
            "stage": "pitch_mapping_and_key_context",
            "status": "scored",
            "support_score_count": len(scores),
            "opportunity_count": pitch["matched_capacity"],
            "observed_error_count": pitch["mismatches_within_count_capacity"],
            "observed_metric": "wrong ordered pitches within note-count capacity",
        },
        {
            "stage": "onset_assignment",
            "status": "scored_on_full_event_subset",
            "support_score_count": 2,
            "opportunity_count": full["onset"]["matched_capacity"],
            "observed_error_count": full["onset"]["mismatches_within_count_capacity"],
            "observed_metric": "wrong ordered onsets within note-count capacity",
        },
        {
            "stage": "duration_quantization",
            "status": "scored_on_full_event_subset",
            "support_score_count": 2,
            "opportunity_count": full["duration"]["matched_capacity"],
            "observed_error_count": full["duration"]["mismatches_within_count_capacity"],
            "observed_metric": "wrong ordered durations within note-count capacity",
        },
        {
            "stage": "rest_detection",
            "status": "scored_on_full_event_subset",
            "support_score_count": 2,
            "opportunity_count": full["rests"]["truth_count"] + full["rests"]["fp"],
            "observed_error_count": full["rests"]["fn"] + full["rests"]["fp"],
            "observed_metric": "rest false positives plus false negatives",
        },
    ]
    meter_scores = [score for score in scores if score["rhythm"]["meter"]["status"] == "scored"]
    meter_crop_count = sum(int(score["rhythm"]["meter"]["crop_count"]) for score in meter_scores)
    meter_valid_count = sum(
        int(score["rhythm"]["meter"]["valid_crop_count"]) for score in meter_scores
    )
    stages.append(
        {
            "stage": "meter_validation",
            "status": "scored_on_limited_subset",
            "support_score_count": len(meter_scores),
            "opportunity_count": meter_crop_count,
            "observed_error_count": meter_crop_count - meter_valid_count,
            "observed_metric": "meter-invalid automatic crops",
        }
    )
    for stage in stages:
        stage["error_rate"] = _ratio(
            int(stage["observed_error_count"]),
            int(stage["opportunity_count"]),
        )
        stage["ranking_eligible"] = stage["status"] in RANKING_ELIGIBLE_STATUSES and (
            stage["support_score_count"] >= MIN_TARGET_SUPPORT_SCORES
            and stage["opportunity_count"] >= MIN_TARGET_OPPORTUNITIES
        )
        if not stage["ranking_eligible"]:
            if stage["status"] not in RANKING_ELIGIBLE_STATUSES:
                stage["ranking_exclusion_reason"] = (
                    "stage is unsupported, causally ambiguous, limited, or already "
                    "excluded through clean-unit filtering"
                )
            else:
                stage["ranking_exclusion_reason"] = (
                    "insufficient independent-score support or opportunities"
                )
    return stages


def _select_next_target(stages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [stage for stage in stages if stage["ranking_eligible"]]
    if not eligible:
        raise ValueError("No stage has enough independent support for target selection")
    ranked = sorted(
        eligible,
        key=lambda stage: (
            -int(stage["observed_error_count"]),
            -float(stage["error_rate"]),
            str(stage["stage"]),
        ),
    )
    selected = ranked[0]
    return {
        "selected_target": selected["stage"],
        "decision_rule": {
            "minimum_support_scores": MIN_TARGET_SUPPORT_SCORES,
            "minimum_opportunities": MIN_TARGET_OPPORTUNITIES,
            "eligible_statuses": sorted(RANKING_ELIGIBLE_STATUSES),
            "unit_policy": "rank downstream stages on clean one-to-one crops only",
            "ranking": "highest observed error count, then error rate",
        },
        "basis": {
            "support_score_count": selected["support_score_count"],
            "opportunity_count": selected["opportunity_count"],
            "observed_error_count": selected["observed_error_count"],
            "error_rate": selected["error_rate"],
        },
        "next_experiment": {
            "goal": (
                "isolate pitch from note selection by freezing candidate identities and "
                "coordinates, then compare automatic staff-geometry and key-state mapping"
            ),
            "gate": (
                "improve exact ordered pitch on at least two independent unseen scores "
                "without changing note counts or coordinates"
            ),
            "human_input_required_now": False,
        },
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    aggregate = report["aggregate"]
    clean = aggregate["clean_one_to_one_units"]
    lines = [
        "# Cross-Score Melody Error Breakdown",
        "",
        "Independent frozen score gates only. Unsupported fields are not treated as zero.",
        "",
        "| Score | Scope | Count-capacity F1 upper bound | Conditional pitch "
        "| Missing boundaries |",
        "|---|---|---:|---:|---:|",
    ]
    for score in report["scores"]:
        lines.append(
            "| {score_id} | {scope} | {count_f1:.3f} | {pitch:.3f} | {boundaries} |".format(
                score_id=score["score_id"],
                scope=score["scope"],
                count_f1=score["note_count"]["f1"],
                pitch=score["pitch"]["conditional_accuracy"],
                boundaries=score["segmentation"]["missing_boundary_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Clean one-to-one count-capacity F1 upper bound: "
            f"`{clean['note_count']['f1']:.6f}` "
            f"({clean['note_count']['deficit']} deficits, "
            f"{clean['note_count']['surplus']} surpluses).",
            f"- Clean conditional exact pitch: "
            f"`{clean['pitch']['conditional_accuracy']:.6f}` "
            f"({clean['pitch']['mismatches_within_count_capacity']} errors across "
            f"{clean['pitch']['matched_capacity']} count-alignable notes).",
            f"- Missing measure boundaries: "
            f"`{aggregate['segmentation']['missing_boundary_count']}`.",
            "",
            "## Next Target",
            "",
            f"`{report['next_engineering_target']['selected_target']}`",
            "",
            report["next_engineering_target"]["next_experiment"]["goal"] + ".",
            "",
        ]
    )
    return "\n".join(lines)


def _positive_int_list(value: Any) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError("Physical measure mapping must be a non-empty list")
    return [_positive_int(item, "physical measure number") for item in value]


def _rows_by_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    index_getter: Callable[[Mapping[str, Any]], Any],
    label: str,
) -> dict[int, Mapping[str, Any]]:
    indexed: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        index = _positive_int(index_getter(row), f"{label} index")
        if index in indexed:
            raise ValueError(f"{label} contains duplicate index {index}")
        indexed[index] = row
    return indexed


def _validate_physical_mapping(
    spec: Mapping[str, Any],
    physical_lists: Sequence[Sequence[int]],
) -> None:
    expected_crop_count = int(spec["expected_automatic_crop_count"])
    if len(physical_lists) != expected_crop_count:
        raise ValueError(
            f"{spec['score_id']} automatic crop count mismatch: "
            f"expected {expected_crop_count}, got {len(physical_lists)}"
        )
    for values in physical_lists:
        if list(values) != list(range(values[0], values[-1] + 1)):
            raise ValueError(f"{spec['score_id']} merged physical measures must be contiguous")
    flattened = [number for values in physical_lists for number in values]
    if flattened != list(spec["expected_physical_measure_numbers"]):
        raise ValueError(f"{spec['score_id']} physical measure coverage mismatch")


def _validate_report_identity(
    report: Mapping[str, Any],
    *,
    expected_kind: str,
    target_slug: str,
    target_system_index: int,
) -> None:
    if expected_kind == "one_shot_second_score_evaluation":
        metrics = _required_object(report, "metrics", "Carrizal report")
        rows = _object_rows(metrics.get("results"), "Carrizal metric results")
        identities = [_required_object(row, "identity", "Carrizal metric row") for row in rows]
        if any(identity.get("slug") != target_slug for identity in identities):
            raise ValueError("Carrizal target slug does not match the input manifest")
        if any(identity.get("system_index") != target_system_index for identity in identities):
            raise ValueError("Carrizal target system does not match the input manifest")
        return
    target = _required_object(report, "target", "Heldout report")
    if target.get("slug") != target_slug:
        raise ValueError("Heldout target slug does not match the input manifest")
    if target.get("system_index") != target_system_index:
        raise ValueError("Heldout target system does not match the input manifest")


def _object_rows(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty array")
    if any(not isinstance(row, Mapping) for row in value):
        raise ValueError(f"{label} must contain only objects")
    return [dict(row) for row in value]


def _required_object(payload: Mapping[str, Any], key: str, label: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}.{key} must be an object")
    return value


def _required_string(payload: Mapping[str, Any], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value


def _required_sha256(payload: Mapping[str, Any], key: str, label: str) -> str:
    value = _required_string(payload, key, label)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label}.{key} must be a lowercase SHA256")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _f1(capacity: int, predicted: int, truth: int) -> float | None:
    return round(2 * capacity / (predicted + truth), 6) if predicted + truth else None


def _binary_f1(tp: int, fp: int, fn: int) -> float | None:
    denominator = 2 * tp + fp + fn
    return round(2 * tp / denominator, 6) if denominator else None


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _file_record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Artifact is missing: {resolved}")
    try:
        display_path = resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        display_path = str(resolved)
    return {"path": display_path, "sha256": _sha256(resolved)}


def _local_record(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, ensure_ascii=False)
        output.write("\n")


def _write_text(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8") as output:
        output.write(content)


if __name__ == "__main__":
    raise SystemExit(main())
