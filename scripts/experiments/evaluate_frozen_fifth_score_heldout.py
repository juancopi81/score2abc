"""Score the sealed Coqueteos fifth-score gate with full event metrics.

The evaluator is intentionally post-freeze and create-once. It verifies the
sealed gate, prepared inputs, frozen predictions, model artifacts, training
artifacts, and inference binding before it opens MusicXML or mapping data.

When the frozen gate and MusicXML both contain six measures, crops are mapped
one-to-one in order. Otherwise ``--mapping`` must provide an explicit,
whole-measure mapping such as::

    {
      "schema_version": 1,
      "automatic_crops": [
        {"automatic_crop_index": 1, "physical_measure_numbers": [1, 2]},
        {"automatic_crop_index": 2, "physical_measure_numbers": [3]}
      ]
    }

Every crop and physical measure must appear exactly once and in score order.
Partial-measure note spans are rejected. Successful output is written atomically
under ``evaluation_<version>`` with source MusicXML, materialized truth, report,
mapping, frozen-input snapshots, evaluator snapshots, and a hash-pinned manifest.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from score2abc.events import measure_length_beats  # noqa: E402
from scripts import build_vlm_melody_event_benchmark as benchmark  # noqa: E402
from scripts.experiments import evaluate_frozen_third_score_heldout as heldout  # noqa: E402
from scripts.experiments import evaluate_second_score_heldout as second_score  # noqa: E402
from scripts.experiments import freeze_third_score_heldout as freezer  # noqa: E402

SCHEMA_VERSION = 1
DEFAULT_EVALUATION_VERSION = "v1"
EXPECTED_CROP_COUNT = 6
REPORT_KIND = "fifth_score_full_event_one_shot_evaluation"
MANIFEST_KIND = "fifth_score_full_event_evaluation_manifest"

TruthLoader = Callable[[Path], second_score.MusicXMLTruth]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "sealed_manifest",
        type=Path,
        help="Frozen fifth-score sealed_manifest.json; verified before MusicXML is opened.",
    )
    parser.add_argument("--musicxml", type=Path, required=True, help="Final human MusicXML.")
    parser.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help=(
            "Whole-measure crop mapping JSON. Required when the frozen crop count and "
            "physical MusicXML measure count are not both six."
        ),
    )
    parser.add_argument(
        "--evaluation-version",
        default=DEFAULT_EVALUATION_VERSION,
        help="Create-once output version (default: v1, written as evaluation_v1).",
    )
    args = parser.parse_args(argv)
    try:
        result = evaluate_frozen_fifth_score(
            args.sealed_manifest,
            musicxml_path=args.musicxml,
            mapping_path=args.mapping,
            evaluation_version=args.evaluation_version,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        OSError,
        ValueError,
        ET.ParseError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(result["report"])
    return 0


def evaluate_frozen_fifth_score(
    sealed_manifest_path: Path,
    *,
    musicxml_path: Path,
    mapping_path: Path | None = None,
    evaluation_version: str = DEFAULT_EVALUATION_VERSION,
    truth_loader: TruthLoader = second_score.load_musicxml_truth,
) -> dict[str, str]:
    """Verify, score, and atomically publish the fifth-score evaluation."""
    _validate_version(evaluation_version)
    sealed_manifest_path = sealed_manifest_path.expanduser().resolve()
    musicxml_path = musicxml_path.expanduser().resolve()
    mapping_path = mapping_path.expanduser().resolve() if mapping_path else None

    # No truth-bearing file may be opened before this completes.
    frozen = dict(heldout.verify_frozen_gate(sealed_manifest_path))
    frozen["sealed_manifest_path"] = sealed_manifest_path
    if frozen["evaluation_spec"].gate != heldout.FIFTH_SCORE_EVALUATION.gate:
        raise ValueError("Sealed manifest is not the fifth-score heldout gate")

    namespace_root = Path(frozen["namespace_root"])
    output_dir = namespace_root / f"evaluation_{evaluation_version}"
    temp_dir = namespace_root / f".evaluation_{evaluation_version}.tmp"
    prior_evaluations = sorted(namespace_root.glob("evaluation_*"))
    if prior_evaluations:
        raise FileExistsError(
            "One-shot fifth-score evaluation already exists: "
            + ", ".join(str(path) for path in prior_evaluations)
        )
    stale_temps = sorted(namespace_root.glob(".evaluation_*.tmp"))
    if stale_temps:
        raise FileExistsError(
            "Stale fifth-score evaluation directory exists: "
            + ", ".join(str(path) for path in stale_temps)
        )

    if not musicxml_path.is_file():
        raise FileNotFoundError(f"User MusicXML does not exist: {musicxml_path}")
    if mapping_path is not None and not mapping_path.is_file():
        raise FileNotFoundError(f"Mapping JSON does not exist: {mapping_path}")

    musicxml_sha256 = freezer._sha256(musicxml_path)
    mapping_sha256 = freezer._sha256(mapping_path) if mapping_path else None
    truth = truth_loader(musicxml_path)
    crop_indices = tuple(sorted(int(value) for value in frozen["predictions_by_crop"]))
    mapping, mapping_mode = _resolve_mapping(
        crop_indices,
        truth.measure_numbers,
        mapping_path=mapping_path,
    )
    allowed_context, context_path, context_sha256 = _load_allowed_context(frozen)
    measure_length = measure_length_beats(str(truth.payload["time_signature"]))

    request_rows = _benchmark_requests(frozen["requests_by_crop"])
    truth_rows = second_score.build_mapped_truth_rows(
        request_rows,
        truth,
        mapping=mapping,
        measure_length=measure_length,
    )
    prediction_rows = _benchmark_predictions(frozen["predictions_by_crop"])
    canonical_metrics = benchmark.evaluate_predictions(truth_rows, prediction_rows)
    meter_metrics = _evaluate_meter(
        truth=truth,
        truth_rows=truth_rows,
        prediction_rows=prediction_rows,
        mapping=mapping,
        allowed_context=allowed_context,
        measure_length=measure_length,
    )
    canonical_metrics["summary"].update(
        {
            "meter_context_match": meter_metrics["summary"]["context_match"],
            "meter_valid_crops": meter_metrics["summary"]["valid_prediction_crops"],
            "meter_valid_crop_rate": meter_metrics["summary"]["valid_prediction_crop_rate"],
            "meter_truth_valid_measures": meter_metrics["summary"]["valid_truth_measures"],
            "meter_truth_valid_measure_rate": meter_metrics["summary"]["valid_truth_measure_rate"],
        }
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "status": "evaluated_exactly_once_after_frozen_predictions",
        "target": dict(frozen["target"]),
        "mapping_mode": mapping_mode,
        "metric_support": {
            "note_f1": "scored",
            "ordered_pitch": "scored",
            "ordered_onset": "scored",
            "ordered_duration": "scored",
            "rests": "scored",
            "exact_measures": "scored",
            "meter_validity": "scored",
            "meter_context_match": "scored",
        },
        "metrics": canonical_metrics,
        "meter": meter_metrics,
        "source_musicxml_context": {
            "time_signature": truth.payload["time_signature"],
            "key_fifths": truth.key_fifths,
            "clef": list(truth.clef) if truth.clef is not None else None,
        },
        "truth_materialization": {
            "loader": "evaluate_second_score_heldout.load_musicxml_truth",
            "mapping_semantics": "whole_physical_measures_only",
            "partial_measure_spans_allowed": False,
        },
    }

    frozen_paths = _frozen_input_paths(frozen)
    evaluator_sources = {
        "evaluator": Path(__file__).resolve(),
        "heldout_verifier": Path(heldout.__file__).resolve(),
        "musicxml_truth_loader": Path(second_score.__file__).resolve(),
        "event_benchmark": Path(benchmark.__file__).resolve(),
    }

    temp_dir.mkdir(parents=False, exist_ok=False)
    try:
        snapshots = {
            "source_musicxml": _copy_snapshot(
                musicxml_path,
                temp_dir / "source.musicxml",
                source_sha256=musicxml_sha256,
            ),
            "mapping": _write_mapping_snapshot(
                temp_dir / "mapping.json",
                mapping,
                source_path=mapping_path,
                source_sha256=mapping_sha256,
            ),
            "truth": _write_jsonl_snapshot(temp_dir / "truth.jsonl", truth_rows),
            "allowed_context": _copy_snapshot(
                context_path,
                temp_dir / "frozen_allowed_context.json",
                source_sha256=context_sha256,
            ),
        }
        for role, source in frozen_paths.items():
            snapshots[role] = _copy_snapshot(
                source["path"],
                temp_dir / source["snapshot_name"],
                source_sha256=source["sha256"],
            )
        for role, source in evaluator_sources.items():
            snapshots[role] = _copy_snapshot(
                source,
                temp_dir / f"{role}.py",
                source_sha256=freezer._sha256(source),
            )

        report["pins"] = snapshots
        report_path = temp_dir / "report.json"
        heldout._write_json(report_path, report)
        snapshots["report"] = heldout._snapshot_record(report_path)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": MANIFEST_KIND,
            "status": "evaluated_exactly_once_after_frozen_predictions",
            "create_once": True,
            "evaluation_version": evaluation_version,
            "target": dict(frozen["target"]),
            "truth_opened_after_all_frozen_hashes_verified": True,
            "mapping_mode": mapping_mode,
            "partial_measure_mapping_allowed": False,
            "pins": snapshots,
        }
        heldout._write_json(temp_dir / "manifest.json", manifest)

        verified_again = heldout.verify_frozen_gate(sealed_manifest_path)
        if (
            verified_again["freeze_sha256"] != frozen["freeze_sha256"]
            or verified_again["sealed_sha256"] != frozen["sealed_sha256"]
            or verified_again["prepared_sha256"] != frozen["prepared_sha256"]
        ):
            raise ValueError("Frozen fifth-score gate changed during evaluation")
        temp_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return {
        "evaluation_dir": str(output_dir),
        "manifest": str(output_dir / "manifest.json"),
        "report": str(output_dir / "report.json"),
        "truth": str(output_dir / "truth.jsonl"),
        "mapping": str(output_dir / "mapping.json"),
    }


def _resolve_mapping(
    crop_indices: Sequence[int],
    measure_numbers: Sequence[int],
    *,
    mapping_path: Path | None,
) -> tuple[dict[int, tuple[int, ...]], str]:
    if mapping_path is None:
        if len(crop_indices) != EXPECTED_CROP_COUNT or len(measure_numbers) != EXPECTED_CROP_COUNT:
            raise ValueError(
                "Default mapping requires exactly six frozen crops and six physical "
                "MusicXML measures; provide --mapping with whole physical measures"
            )
        return (
            {crop: (measure,) for crop, measure in zip(crop_indices, measure_numbers, strict=True)},
            "deterministic_default_one_to_one",
        )

    payload = heldout._read_json(mapping_path)
    if int(payload.get("schema_version", SCHEMA_VERSION)) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported mapping schema version: {payload.get('schema_version')}")
    entries = payload.get("automatic_crops")
    if not isinstance(entries, list):
        raise ValueError("Mapping must contain an automatic_crops list")

    mapping: dict[int, tuple[int, ...]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("Each automatic_crops entry must be an object")
        crop = int(entry["automatic_crop_index"])
        if crop in mapping:
            raise ValueError(f"Duplicate automatic crop in mapping: {crop}")
        if "physical_measure_spans" in entry:
            spans = entry["physical_measure_spans"]
            if not isinstance(spans, list) or not spans:
                raise ValueError(f"Automatic crop {crop} has no physical measures")
            measures = []
            for span in spans:
                if not isinstance(span, Mapping):
                    raise ValueError(f"Automatic crop {crop} contains a non-object span")
                if "note_start" in span or "note_end" in span:
                    raise ValueError(
                        "Partial-measure note spans are not supported; map whole measures only"
                    )
                if set(span) != {"measure_number"}:
                    raise ValueError(f"Automatic crop {crop} span must contain only measure_number")
                measures.append(int(span["measure_number"]))
        else:
            raw_measures = entry.get("physical_measure_numbers")
            if not isinstance(raw_measures, list) or not raw_measures:
                raise ValueError(f"Automatic crop {crop} has no physical measures")
            measures = [int(value) for value in raw_measures]
        mapping[crop] = tuple(measures)

    if set(mapping) != set(crop_indices):
        raise ValueError(
            "Mapping crop identities differ from frozen predictions: "
            f"mapping={sorted(mapping)}, frozen={list(crop_indices)}"
        )
    flattened = tuple(measure for crop in sorted(mapping) for measure in mapping[crop])
    if flattened != tuple(measure_numbers):
        raise ValueError(
            "Whole-measure mapping must cover every MusicXML measure exactly once in score "
            f"order: mapping={flattened}, musicxml={tuple(measure_numbers)}"
        )
    return mapping, "explicit_whole_measure_mapping"


def _load_allowed_context(frozen: Mapping[str, Any]) -> tuple[dict[str, Any], Path, str]:
    prepared = heldout._read_json(Path(frozen["prepared_path"]))
    context_record = prepared.get("artifacts", {}).get("context", {}).get("allowed_context")
    if not isinstance(context_record, Mapping):
        raise ValueError("Fifth-score prepared gate is missing hash-pinned allowed context")
    context_path = heldout._safe_child(
        Path(frozen["namespace_root"]),
        str(context_record["path"]),
    )
    context_sha256 = str(context_record["sha256"])
    if freezer._sha256(context_path) != context_sha256:
        raise ValueError("Prepared allowed-context hash drift")
    payload = heldout._read_json(context_path)
    if payload.get("truth_accessed") is not False or payload.get("truth_used") is not False:
        raise ValueError("Prepared musical context is not truth-blind")
    allowed = payload.get("allowed_context")
    if not isinstance(allowed, Mapping):
        raise ValueError("Prepared allowed_context is malformed")
    return dict(allowed), context_path, context_sha256


def _benchmark_requests(
    requests_by_crop: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for global_index, crop in enumerate(sorted(requests_by_crop), start=1):
        row = dict(requests_by_crop[crop])
        identity = dict(row["identity"])
        identity["system_measure_index"] = crop
        identity["global_measure_index"] = global_index
        row["identity"] = identity
        rows.append(row)
    return rows


def _benchmark_predictions(
    predictions_by_crop: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for global_index, crop in enumerate(sorted(predictions_by_crop), start=1):
        row = dict(predictions_by_crop[crop])
        identity = dict(row["identity"])
        identity["system_measure_index"] = crop
        identity["global_measure_index"] = global_index
        row["identity"] = identity
        rows.append(row)
    return rows


def _evaluate_meter(
    *,
    truth: second_score.MusicXMLTruth,
    truth_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    mapping: Mapping[int, Sequence[int]],
    allowed_context: Mapping[str, Any],
    measure_length: Fraction,
) -> dict[str, Any]:
    context_time_signature = allowed_context.get("time_signature")
    context_beats = _optional_fraction(allowed_context.get("expected_measure_beats"))
    context_match = (
        context_time_signature == truth.payload.get("time_signature")
        and context_beats == measure_length
    )

    valid_truth = {
        measure: extent == measure_length for measure, extent in truth.measure_extents.items()
    }
    truth_by_identity = {benchmark._identity_key(row["identity"]): row for row in truth_rows}
    crop_rows = []
    for prediction in prediction_rows:
        identity = dict(prediction["identity"])
        crop = int(identity["system_measure_index"])
        truth_row = truth_by_identity[benchmark._identity_key(identity)]
        expected_extent = Fraction(str(truth_row["measure_extent_beats"]))
        declared_extent = _optional_fraction(prediction.get("measure_extent_beats"))
        coverage = _event_coverage(
            list(prediction.get("notes") or []) + list(prediction.get("rests") or []),
            expected_extent,
        )
        extent_match = declared_extent == expected_extent
        crop_rows.append(
            {
                "automatic_crop_index": crop,
                "physical_measure_numbers": list(mapping[crop]),
                "expected_extent_beats": _fraction_number(expected_extent),
                "predicted_extent_beats": (
                    _fraction_number(declared_extent) if declared_extent is not None else None
                ),
                "predicted_extent_match": extent_match,
                "event_coverage_beats": _fraction_number(coverage),
                "event_coverage_complete": coverage == expected_extent,
                "meter_valid": extent_match and coverage == expected_extent,
            }
        )

    valid_prediction_count = sum(bool(row["meter_valid"]) for row in crop_rows)
    valid_truth_count = sum(valid_truth.values())
    return {
        "context": {
            "allowed_time_signature": context_time_signature,
            "truth_time_signature": truth.payload.get("time_signature"),
            "allowed_expected_measure_beats": (
                _fraction_number(context_beats) if context_beats is not None else None
            ),
            "truth_measure_length_beats": _fraction_number(measure_length),
        },
        "summary": {
            "context_match": context_match,
            "valid_prediction_crops": valid_prediction_count,
            "prediction_crop_count": len(crop_rows),
            "valid_prediction_crop_rate": _ratio(valid_prediction_count, len(crop_rows)),
            "valid_truth_measures": valid_truth_count,
            "truth_measure_count": len(valid_truth),
            "valid_truth_measure_rate": _ratio(valid_truth_count, len(valid_truth)),
        },
        "truth_measures": [
            {
                "measure_number": measure,
                "extent_beats": _fraction_number(truth.measure_extents[measure]),
                "meter_valid": valid_truth[measure],
            }
            for measure in truth.measure_numbers
        ],
        "crops": crop_rows,
    }


def _event_coverage(events: Sequence[Mapping[str, Any]], extent: Fraction) -> Fraction:
    intervals = []
    for event in events:
        onset = Fraction(str(event["onset_beats"]))
        duration = Fraction(str(event["duration_beats"]))
        if onset < 0 or duration <= 0 or onset + duration > extent:
            return Fraction(-1)
        intervals.append((onset, onset + duration))
    if not intervals:
        return Fraction(0)
    intervals.sort()
    merged: list[list[Fraction]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    if merged[0][0] != 0 or merged[-1][1] != extent:
        return sum((end - start for start, end in merged), Fraction(0))
    return sum((end - start for start, end in merged), Fraction(0))


def _frozen_input_paths(frozen: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    namespace_root = Path(frozen["namespace_root"])
    freeze = frozen["freeze"]
    requests_record = freeze["requests"]
    predictions_record = freeze["predictions"]
    requests_path = heldout._safe_child(namespace_root, str(requests_record["path"]))
    predictions_path = heldout._safe_child(
        namespace_root,
        str(predictions_record["snapshot_path_relative_to_namespace"]),
    )
    return {
        "frozen_prepared_manifest": {
            "path": Path(frozen["prepared_path"]),
            "sha256": str(frozen["prepared_sha256"]),
            "snapshot_name": "frozen_prepared_manifest.json",
        },
        "frozen_requests": {
            "path": requests_path,
            "sha256": str(requests_record["sha256"]),
            "snapshot_name": "frozen_requests.jsonl",
        },
        "frozen_predictions": {
            "path": predictions_path,
            "sha256": str(predictions_record["snapshot_sha256"]),
            "snapshot_name": "frozen_predictions.jsonl",
        },
        "frozen_freeze_manifest": {
            "path": Path(frozen["freeze_path"]),
            "sha256": str(frozen["freeze_sha256"]),
            "snapshot_name": "frozen_freeze.json",
        },
        "frozen_sealed_manifest": {
            "path": Path(frozen["sealed_manifest_path"]),
            "sha256": str(frozen["sealed_sha256"]),
            "snapshot_name": "frozen_sealed_manifest.json",
        },
    }


def _copy_snapshot(source: Path, destination: Path, *, source_sha256: str) -> dict[str, Any]:
    shutil.copyfile(source, destination)
    return heldout._snapshot_record(
        destination,
        source_path=source,
        source_sha256=source_sha256,
    )


def _write_mapping_snapshot(
    path: Path,
    mapping: Mapping[int, Sequence[int]],
    *,
    source_path: Path | None,
    source_sha256: str | None,
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "whole_measure_crop_mapping",
        "automatic_crops": [
            {
                "automatic_crop_index": crop,
                "physical_measure_numbers": list(mapping[crop]),
            }
            for crop in sorted(mapping)
        ],
    }
    heldout._write_json(path, payload)
    return heldout._snapshot_record(
        path,
        source_path=source_path,
        source_sha256=source_sha256,
        require_source_match=False,
    )


def _write_jsonl_snapshot(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    heldout._write_jsonl(path, rows)
    return heldout._snapshot_record(path)


def _optional_fraction(value: Any) -> Fraction | None:
    if value is None:
        return None
    return Fraction(str(value))


def _fraction_number(value: Fraction) -> int | float:
    return value.numerator if value.denominator == 1 else float(value)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


def _validate_version(value: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError(f"Invalid evaluation version: {value!r}")


if __name__ == "__main__":
    raise SystemExit(main())
