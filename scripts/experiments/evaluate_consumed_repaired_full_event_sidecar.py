"""Evaluate a repaired full-event sidecar as consumed postmortem evidence.

This evaluator is deliberately create-once and consumed-only. It verifies the
truth-blind sidecar, its upstream candidate/model pins, and the canonical
baseline prediction pin before opening the supplied truth snapshot. Its output
must never be interpreted as held-out or runtime-promotion evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_vlm_melody_event_benchmark as benchmark  # noqa: E402
from scripts.experiments import (  # noqa: E402
    materialize_repaired_full_event_sidecar as materializer,
)

SCHEMA_VERSION = 1
REPORT_KIND = "consumed_repaired_full_event_sidecar_evaluation"
MANIFEST_KIND = "consumed_repaired_full_event_sidecar_evaluation_manifest"
EVIDENCE_SCOPE = "consumed_postmortem"

TruthLoader = Callable[[Path], list[dict[str, Any]]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sidecar_dir", type=Path)
    parser.add_argument("--truth-snapshot", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = evaluate_consumed_repaired_full_event_sidecar(
            args.sidecar_dir,
            truth_snapshot=args.truth_snapshot,
            mapping=args.mapping,
            output_dir=args.output_dir,
        )
    except (FileExistsError, FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(result["report"])
    return 0


def evaluate_consumed_repaired_full_event_sidecar(
    sidecar_dir: Path,
    *,
    truth_snapshot: Path,
    mapping: Path,
    output_dir: Path,
    truth_loader: TruthLoader | None = None,
) -> dict[str, str]:
    """Verify, score, and atomically publish one consumed-only comparison."""
    sidecar_dir = sidecar_dir.expanduser().resolve()
    truth_snapshot = truth_snapshot.expanduser().resolve()
    mapping = mapping.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    temp_dir = output_dir.parent / f".{output_dir.name}.tmp"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite create-once evaluation: {output_dir}")
    if temp_dir.exists():
        raise FileExistsError(f"Refusing stale temporary evaluation: {temp_dir}")
    if _is_relative_to(output_dir, sidecar_dir):
        raise ValueError("Evaluation output must not be created inside the verified sidecar")

    verification_order: list[str] = []

    # No truth-bearing path is opened before both verification steps complete.
    verified_sidecar = materializer.verify_repaired_full_event_sidecar(sidecar_dir)
    verification_order.append("candidate_model_and_sidecar_hashes_verified")
    sidecar_manifest_path = sidecar_dir / "manifest.json"
    sidecar_manifest_sha256 = _sha256(sidecar_manifest_path)
    sidecar_manifest = _read_json(sidecar_manifest_path)
    baseline_path, baseline_sha256 = _verify_baseline_canonical_predictions_pin(
        sidecar_dir, sidecar_manifest
    )
    repaired_path, repaired_sha256 = _verify_repaired_predictions_pin(sidecar_dir, sidecar_manifest)
    verification_order.append("baseline_canonical_predictions_pin_verified")

    if not mapping.is_file():
        raise FileNotFoundError(f"Consumed crop mapping does not exist: {mapping}")
    mapping_sha256 = _sha256(mapping)
    crop_mapping = _validate_crop_mapping(_read_json(mapping))
    verification_order.append("explicit_crop_mapping_opened")
    if not truth_snapshot.is_file():
        raise FileNotFoundError(f"Consumed truth snapshot does not exist: {truth_snapshot}")
    verification_order.append("truth_snapshot_opened")
    truth_sha256 = _sha256(truth_snapshot)
    load_truth = truth_loader or _read_jsonl
    truth_rows = load_truth(truth_snapshot)
    baseline_rows = _read_jsonl(baseline_path)
    repaired_rows = _read_jsonl(repaired_path)

    adapted_truth, adapted_baseline, adapted_repaired = _adapt_aligned_rows(
        truth_rows,
        baseline_rows,
        repaired_rows,
        target=sidecar_manifest.get("target"),
        crop_mapping=crop_mapping,
    )
    baseline_metrics = benchmark.evaluate_predictions(adapted_truth, adapted_baseline)
    repaired_metrics = benchmark.evaluate_predictions(adapted_truth, adapted_repaired)
    baseline_meter = _evaluate_meter_validity(adapted_truth, adapted_baseline)
    repaired_meter = _evaluate_meter_validity(adapted_truth, adapted_repaired)
    metric_deltas = _summary_deltas(baseline_metrics["summary"], repaired_metrics["summary"])

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "status": "evaluated_consumed_postmortem",
        "evidence_scope": EVIDENCE_SCOPE,
        "runtime_promotion_supported": False,
        "heldout_claim_supported": False,
        "target": dict(sidecar_manifest.get("target") or {}),
        "measure_count": len(adapted_truth),
        "verification_order": verification_order,
        "truth_opened_after_all_candidate_model_sidecar_hashes_verified": True,
        "identity_adaptation": {
            "ordering": "automatic_crop_ascending",
            "system_measure_index": "automatic_crop_index",
            "global_measure_index": "one_based_automatic_crop_order",
            "explicit_physical_measure_mapping": {
                str(crop): physical for crop, physical in sorted(crop_mapping.items())
            },
        },
        "lanes": {
            "baseline": {"metrics": baseline_metrics, "meter_validity": baseline_meter},
            "repaired": {"metrics": repaired_metrics, "meter_validity": repaired_meter},
        },
        "metric_deltas_repaired_minus_baseline": metric_deltas,
        "interpretation_limits": {
            "meter_validity": (
                "Declared prediction measure_extent_beats equals consumed truth extent; "
                "this does not establish duration or rest accuracy."
            ),
            "duration_and_rest_accuracy": "Reported separately by the event benchmark.",
            "promotion": "Consumed postmortem evidence cannot support runtime promotion.",
        },
        "verified_sidecar": dict(verified_sidecar),
    }

    source_records = {
        "sidecar_manifest": _source_record(sidecar_manifest_path, sidecar_manifest_sha256),
        "baseline_predictions": _source_record(baseline_path, baseline_sha256),
        "repaired_predictions": _source_record(repaired_path, repaired_sha256),
        "truth": _source_record(truth_snapshot, truth_sha256),
        "mapping": _source_record(mapping, mapping_sha256),
    }
    implementation_records = {
        "evaluator": _implementation_record(Path(__file__)),
        "sidecar_verifier": _implementation_record(Path(materializer.__file__)),
        "event_benchmark": _implementation_record(Path(benchmark.__file__)),
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=False, exist_ok=False)
    try:
        snapshot_records = {
            "baseline_predictions": _copy_snapshot(
                baseline_path,
                temp_dir / "baseline_predictions.jsonl",
                source_sha256=baseline_sha256,
            ),
            "repaired_predictions": _copy_snapshot(
                repaired_path,
                temp_dir / "repaired_predictions.jsonl",
                source_sha256=repaired_sha256,
            ),
            "truth": _copy_snapshot(
                truth_snapshot,
                temp_dir / "truth.jsonl",
                source_sha256=truth_sha256,
            ),
            "mapping": _copy_snapshot(
                mapping,
                temp_dir / "mapping.json",
                source_sha256=mapping_sha256,
            ),
            "sidecar_manifest": _copy_snapshot(
                sidecar_manifest_path,
                temp_dir / "sidecar_manifest.json",
                source_sha256=sidecar_manifest_sha256,
            ),
        }
        report_path = temp_dir / "report.json"
        report_md_path = temp_dir / "report.md"
        _write_json(report_path, report)
        report_md_path.write_text(_render_report_markdown(report), encoding="utf-8")
        snapshot_records["report"] = _snapshot_record(report_path)
        snapshot_records["report_markdown"] = _snapshot_record(report_md_path)

        _assert_source_hashes_unchanged(
            {
                sidecar_manifest_path: sidecar_manifest_sha256,
                baseline_path: baseline_sha256,
                repaired_path: repaired_sha256,
                truth_snapshot: truth_sha256,
                mapping: mapping_sha256,
            }
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": MANIFEST_KIND,
            "status": "evaluated_consumed_postmortem",
            "create_once": True,
            "evidence_scope": EVIDENCE_SCOPE,
            "runtime_promotion_supported": False,
            "heldout_claim_supported": False,
            "target": dict(sidecar_manifest.get("target") or {}),
            "measure_count": len(adapted_truth),
            "truth_opened_after_all_candidate_model_sidecar_hashes_verified": True,
            "verification_order": verification_order,
            "pins": {
                "sources": source_records,
                "snapshots": snapshot_records,
                "implementations": implementation_records,
            },
        }
        _write_json(temp_dir / "manifest.json", manifest)
        temp_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return {
        "evaluation_dir": str(output_dir),
        "manifest": str(output_dir / "manifest.json"),
        "report": str(output_dir / "report.json"),
        "report_markdown": str(output_dir / "report.md"),
    }


def _verify_baseline_canonical_predictions_pin(
    sidecar_dir: Path,
    manifest: Mapping[str, Any],
) -> tuple[Path, str]:
    canonical = manifest.get("canonical")
    record = canonical.get("predictions") if isinstance(canonical, Mapping) else None
    expected = (sidecar_dir.parent / "predictions.jsonl").resolve()
    return _verify_exact_relative_pin(
        sidecar_dir,
        record,
        expected=expected,
        label="Baseline canonical predictions",
    )


def _verify_repaired_predictions_pin(
    sidecar_dir: Path,
    manifest: Mapping[str, Any],
) -> tuple[Path, str]:
    artifacts = manifest.get("artifacts")
    record = artifacts.get("predictions.jsonl") if isinstance(artifacts, Mapping) else None
    expected = (sidecar_dir / "predictions.jsonl").resolve()
    return _verify_exact_relative_pin(
        sidecar_dir,
        record,
        expected=expected,
        label="Repaired predictions",
    )


def _verify_exact_relative_pin(
    root: Path,
    record: Any,
    *,
    expected: Path,
    label: str,
) -> tuple[Path, str]:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} pin is missing")
    raw_path = Path(str(record.get("path", "")))
    if raw_path.is_absolute():
        raise ValueError(f"{label} pin must be relative")
    actual = (root / raw_path).resolve()
    if actual != expected:
        raise ValueError(f"{label} path substitution: expected {expected}, got {actual}")
    expected_sha256 = str(record.get("sha256", ""))
    if _sha256(actual) != expected_sha256:
        raise ValueError(f"{label} source hash drift: {actual}")
    return actual, expected_sha256


def _adapt_aligned_rows(
    truth_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    repaired_rows: Sequence[Mapping[str, Any]],
    *,
    target: Any,
    crop_mapping: Mapping[int, Sequence[int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    truth_by_target = _rows_by_target(truth_rows, label="Truth")
    baseline_by_target = _rows_by_target(baseline_rows, label="Baseline")
    repaired_by_target = _rows_by_target(repaired_rows, label="Repaired")
    counts = {len(truth_by_target), len(baseline_by_target), len(repaired_by_target)}
    if len(counts) != 1 or not truth_by_target:
        raise ValueError(
            "Truth, baseline, and repaired predictions require identical nonzero row counts"
        )
    truth_keys = set(truth_by_target)
    if truth_keys != set(baseline_by_target) or truth_keys != set(repaired_by_target):
        raise ValueError("Truth, baseline, and repaired target identities are not aligned")
    crop_indices = {key[2] for key in truth_keys}
    if crop_indices != set(crop_mapping):
        raise ValueError("Explicit crop mapping does not match evaluated automatic crops")

    score_targets = {(slug, system_index) for slug, system_index, _ in truth_keys}
    if len(score_targets) != 1:
        raise ValueError("Consumed evaluation must contain exactly one score system")
    slug, system_index = next(iter(score_targets))
    if (
        not isinstance(target, Mapping)
        or str(target.get("slug")) != slug
        or int(target.get("system_index", -1)) != system_index
    ):
        raise ValueError("Sidecar manifest target does not match evaluated identities")

    ordered_keys = sorted(truth_keys, key=lambda item: item[2])
    return (
        _adapt_rows(truth_by_target, ordered_keys, crop_mapping=crop_mapping),
        _adapt_rows(baseline_by_target, ordered_keys, crop_mapping=crop_mapping),
        _adapt_rows(repaired_by_target, ordered_keys, crop_mapping=crop_mapping),
    )


def _rows_by_target(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> dict[tuple[str, int, int], Mapping[str, Any]]:
    result: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} row is not an object")
        identity = row.get("identity")
        if not isinstance(identity, Mapping):
            raise ValueError(f"{label} row identity is missing")
        crop = _automatic_crop_index(identity, label=label)
        key = (str(identity.get("slug")), int(identity.get("system_index", -1)), crop)
        if not key[0] or key[1] <= 0:
            raise ValueError(f"{label} target identity is invalid: {key}")
        if key in result:
            raise ValueError(f"Duplicate {label.lower()} target identity: {key}")
        result[key] = row
    return result


def _automatic_crop_index(identity: Mapping[str, Any], *, label: str) -> int:
    automatic = identity.get("automatic_measure_index")
    system_measure = identity.get("system_measure_index")
    value = automatic if automatic is not None else system_measure
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} automatic crop index must be a positive integer")
    if automatic is not None and system_measure is not None and int(system_measure) != value:
        raise ValueError(f"{label} automatic/system measure indices conflict")
    return value


def _adapt_rows(
    rows_by_target: Mapping[tuple[str, int, int], Mapping[str, Any]],
    ordered_keys: Sequence[tuple[str, int, int]],
    *,
    crop_mapping: Mapping[int, Sequence[int]],
) -> list[dict[str, Any]]:
    adapted = []
    for global_index, key in enumerate(ordered_keys, start=1):
        row = dict(rows_by_target[key])
        identity = dict(row["identity"])
        identity["automatic_measure_index"] = key[2]
        identity["system_measure_index"] = key[2]
        identity["global_measure_index"] = global_index
        identity["physical_measure_numbers"] = list(crop_mapping[key[2]])
        row["identity"] = identity
        adapted.append(row)
    return adapted


def _validate_crop_mapping(payload: Mapping[str, Any]) -> dict[int, list[int]]:
    entries = payload.get("automatic_crops")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Explicit crop mapping has no automatic_crops")
    result: dict[int, list[int]] = {}
    flattened: list[int] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("Explicit crop mapping entry is not an object")
        crop = entry.get("automatic_crop_index")
        measures = entry.get("physical_measure_numbers")
        if isinstance(crop, bool) or not isinstance(crop, int) or crop <= 0 or crop in result:
            raise ValueError("Explicit crop mapping indices must be unique positive integers")
        if (
            not isinstance(measures, list)
            or not measures
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in measures
            )
            or measures != sorted(set(measures))
        ):
            raise ValueError("Physical measure numbers must be sorted unique positive integers")
        result[crop] = list(measures)
        flattened.extend(measures)
    if set(result) != set(range(1, max(result) + 1)):
        raise ValueError("Automatic crop mapping must be contiguous from one")
    if flattened != sorted(set(flattened)):
        raise ValueError("Physical measure mappings must be strictly increasing without overlap")
    return result


def _evaluate_meter_validity(
    truth_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    truth_by_identity = {benchmark._identity_key(row["identity"]): row for row in truth_rows}
    rows = []
    for prediction in prediction_rows:
        identity = dict(prediction["identity"])
        truth = truth_by_identity[benchmark._identity_key(identity)]
        truth_extent = _required_fraction(truth.get("measure_extent_beats"), "truth")
        predicted_extent = _optional_fraction(prediction.get("measure_extent_beats"))
        matches = predicted_extent == truth_extent
        rows.append(
            {
                "identity": identity,
                "truth_measure_extent_beats": _fraction_number(truth_extent),
                "prediction_measure_extent_beats": (
                    _fraction_number(predicted_extent) if predicted_extent is not None else None
                ),
                "measure_extent_match": matches,
                "meter_valid": matches,
            }
        )
    valid = sum(bool(row["meter_valid"]) for row in rows)
    return {
        "definition": "prediction measure_extent_beats equals truth measure_extent_beats",
        "duration_or_rest_accuracy_included": False,
        "summary": {
            "valid_measures": valid,
            "measure_count": len(rows),
            "valid_measure_rate": _ratio(valid, len(rows)),
        },
        "measures": rows,
    }


def _summary_deltas(
    baseline: Mapping[str, Any],
    repaired: Mapping[str, Any],
) -> dict[str, int | float]:
    result: dict[str, int | float] = {}
    for name in sorted(set(baseline) & set(repaired)):
        before = baseline[name]
        after = repaired[name]
        if isinstance(before, bool) or isinstance(after, bool):
            continue
        if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
            continue
        delta = after - before
        result[name] = delta if isinstance(delta, int) else round(delta, 6)
    return result


def _render_report_markdown(report: Mapping[str, Any]) -> str:
    baseline = report["lanes"]["baseline"]["metrics"]["summary"]
    repaired = report["lanes"]["repaired"]["metrics"]["summary"]
    deltas = report["metric_deltas_repaired_minus_baseline"]
    rows = [
        "# Consumed repaired full-event sidecar evaluation",
        "",
        "This is consumed postmortem evidence only. It is neither held-out evidence nor "
        "runtime-promotion evidence.",
        "",
        "| Metric | Baseline | Repaired | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in (
        "note_f1",
        "ordered_pitch_accuracy",
        "ordered_onset_accuracy",
        "ordered_duration_accuracy",
        "rest_f1",
        "exact_measure_rate",
    ):
        rows.append(
            f"| {metric} | {baseline.get(metric)} | {repaired.get(metric)} | "
            f"{deltas.get(metric)} |"
        )
    rows.extend(
        [
            "",
            "## Meter validity",
            "",
            (
                "Meter validity only checks declared `measure_extent_beats` against truth. "
                "Duration and rest accuracy remain separate metrics above."
            ),
            "",
            (
                f"- Baseline: "
                f"{report['lanes']['baseline']['meter_validity']['summary']['valid_measure_rate']}"
            ),
            (
                f"- Repaired: "
                f"{report['lanes']['repaired']['meter_validity']['summary']['valid_measure_rate']}"
            ),
            "",
        ]
    )
    return "\n".join(rows)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSON objects: {path}")
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _source_record(path: Path, sha256: str) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256}


def _implementation_record(path: Path) -> dict[str, str]:
    path = path.resolve()
    try:
        display = path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        display = path.as_posix()
    return {"path": display, "sha256": _sha256(path)}


def _copy_snapshot(source: Path, destination: Path, *, source_sha256: str) -> dict[str, str]:
    shutil.copyfile(source, destination)
    record = _snapshot_record(destination)
    if record["sha256"] != source_sha256:
        raise ValueError(f"Snapshot differs from verified source: {source}")
    record["source_path"] = str(source.resolve())
    record["source_sha256"] = source_sha256
    return record


def _snapshot_record(path: Path) -> dict[str, str]:
    return {"path": path.name, "sha256": _sha256(path)}


def _assert_source_hashes_unchanged(records: Mapping[Path, str]) -> None:
    for path, expected in records.items():
        if _sha256(path) != expected:
            raise ValueError(f"Evaluation source changed while scoring: {path}")


def _required_fraction(value: Any, label: str) -> Fraction:
    result = _optional_fraction(value)
    if result is None or result <= 0:
        raise ValueError(f"{label} measure_extent_beats must be positive")
    return result


def _optional_fraction(value: Any) -> Fraction | None:
    if value is None:
        return None
    result = Fraction(str(value))
    if result <= 0:
        raise ValueError("measure_extent_beats must be positive")
    return result


def _fraction_number(value: Fraction) -> int | float:
    return value.numerator if value.denominator == 1 else float(value)


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
