"""Score two sealed Astra submissions against the frozen Aviador S8 baseline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_vlm_melody_event_benchmark as benchmark  # noqa: E402
from scripts.experiments.spike_vlm_system_transcription import (  # noqa: E402
    pitch_name_to_midi,
)

SEAL_KIND = "astra_music_benchmark_submission_seal"
REPORT_KIND = "astra_music_benchmark_evaluation"
ARMS = ("a", "b")
TruthLoader = Callable[[Path], list[dict[str, Any]]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        result = evaluate_astra_music_benchmark(args.experiment_dir, output_dir=args.output_dir)
    except (FileExistsError, FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(result["report"])
    return 0


def evaluate_astra_music_benchmark(
    experiment_dir: Path,
    *,
    output_dir: Path | None = None,
    truth_loader: TruthLoader | None = None,
) -> dict[str, str]:
    """Verify all seals before truth access, then score the three arms once."""
    root = experiment_dir.expanduser().resolve()
    output = (output_dir or root / "evaluation_v1").expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {output}")

    protocol_path, seal_path = root / "protocol.json", root / "submission_seal.json"
    protocol, seal = _json(protocol_path), _json(seal_path)
    _validate_protocol(protocol)
    if (
        seal.get("kind") != SEAL_KIND
        or seal.get("status") != "frozen_before_truth"
        or seal.get("truth_accessed") is not False
    ):
        raise ValueError("Submission seal is not truth-blind and frozen")
    if seal.get("protocol_sha256") != benchmark._sha256(protocol_path):
        raise ValueError("Protocol hash does not match submission seal")
    submissions = seal.get("submissions")
    if not isinstance(submissions, dict) or tuple(sorted(submissions)) != ARMS:
        raise ValueError(f"Submission seal must contain exactly {list(ARMS)}")

    # Submission and source-only input hashes precede every reference/truth read.
    submission_paths = {arm: _verify(root, submissions[arm], arm) for arm in ARMS}
    if len(set(submission_paths.values())) != len(ARMS):
        raise ValueError("Astra arms must use distinct archived submission paths")
    original_paths = {}
    for arm in ARMS:
        original = Path(protocol["arms"][arm]["directory"]) / "result/submission.json"
        original_paths[arm] = _verify(
            root,
            {"path": str(original), "sha256": submissions[arm]["sha256"]},
            f"arm {arm} preregistered original submission",
        )
    for arm in ARMS:
        _verify_many(root, protocol["arms"][arm]["input_pins"], f"arm {arm} input")
    sealed_artifacts = seal.get("artifacts")
    if not isinstance(sealed_artifacts, list):
        raise ValueError("Submission seal artifacts must be an array")
    _verify_many(root, sealed_artifacts, "sealed archive artifact")

    pins = protocol["reference_pins"]
    requests_path = _verify(root, pins["requests"], "requests")
    freeze_path = _verify(root, pins["baseline_freeze"], "baseline freeze")
    baseline_path = _verify(root, pins["baseline_predictions"], "baseline predictions")
    report_path = _verify(root, pins["baseline_report"], "baseline report")
    scorer_path = _verify(root, pins["evaluator"], "historical evaluator")
    if scorer_path != Path(benchmark.__file__).resolve():
        raise ValueError("Protocol evaluator pin differs from imported historical scorer")
    freeze_records = _verify_freeze(freeze_path, requests_path, baseline_path)

    target = protocol["target"]
    requests = _target_rows(benchmark._read_jsonl(requests_path), target)
    identities = [row["identity"] for row in requests]
    baseline = benchmark._read_jsonl(baseline_path)
    _same_identities(baseline, identities, "baseline predictions")
    expected = target["expected_measure_indices"]
    predictions, validations = {}, {}
    for arm in ARMS:
        predictions[arm], validations[arm] = _adapt(
            _json(submission_paths[arm]), identities, expected
        )

    # The truth pin is checked only after both immutable submissions and every
    # non-truth protocol pin. Parsing follows the hash check.
    truth_path = _verify(root, pins["truth"], "truth")
    truth = _target_rows((truth_loader or benchmark._read_jsonl)(truth_path), target)
    _same_identities(truth, identities, "truth rows")

    baseline_metrics = benchmark.evaluate_predictions(truth, baseline)
    historical = _json(report_path)["validation"]["metrics"]["summary"]
    if baseline_metrics["summary"] != historical:
        raise ValueError("Runtime baseline metrics do not reproduce the pinned report")
    lanes: dict[str, Any] = {
        "baseline": {"metrics": baseline_metrics, "historical_summary_reproduced": True}
    }
    for arm in ARMS:
        errors = _geometry_errors(truth, predictions[arm])
        if errors:
            validations[arm].update(
                status="invalid_event_geometry",
                headline_metrics_valid=False,
                event_geometry_errors=errors,
            )
            scored = []
        else:
            validations[arm]["event_geometry_errors"] = []
            scored = predictions[arm]
        lanes[arm] = {
            "submission_validation": validations[arm],
            "metrics": benchmark.evaluate_predictions(truth, scored),
        }

    all_records = [
        {"path": str(protocol_path), "sha256": seal["protocol_sha256"]},
        {"path": str(seal_path), "sha256": benchmark._sha256(seal_path)},
        *pins.values(),
        *submissions.values(),
        *freeze_records,
        *sealed_artifacts,
        *({"path": str(original_paths[arm]), "sha256": submissions[arm]["sha256"]} for arm in ARMS),
        *(record for arm in ARMS for record in protocol["arms"][arm]["input_pins"]),
    ]
    _verify_many(root, all_records, "source recheck")
    result = {
        "schema_version": 1,
        "kind": REPORT_KIND,
        "status": "evaluated_after_both_astra_submissions_frozen",
        "target": target,
        "dataset_status": protocol["dataset_status"],
        "scorer": "scripts.build_vlm_melody_event_benchmark.evaluate_predictions",
        "scored_scope": "structured_events_only",
        "separate_coordinator_audits": {
            "abc": "not scored here",
            "context": "not scored here",
            "source": "not scored here",
            "notation": "not scored here",
        },
        "arm_input_descriptions": {arm: protocol["arms"][arm]["input"] for arm in ARMS},
        "truth_opened_after_all_submission_and_baseline_pins_verified": True,
        "lanes": lanes,
        "pins": {
            "protocol_sha256": seal["protocol_sha256"],
            "submission_seal_sha256": benchmark._sha256(seal_path),
            "references": pins,
            "submissions": submissions,
            "preregistered_original_submissions": {
                arm: {
                    "path": str(original_paths[arm]),
                    "sha256": submissions[arm]["sha256"],
                }
                for arm in ARMS
            },
            "arm_inputs": {arm: protocol["arms"][arm]["input_pins"] for arm in ARMS},
        },
        "implementations": {
            "adapter_sha256": benchmark._sha256(Path(__file__)),
            "historical_scorer_sha256": benchmark._sha256(Path(benchmark.__file__)),
        },
    }
    output.mkdir(parents=True)
    result_path = output / "report.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"evaluation_dir": str(output), "report": str(result_path)}


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    target, arms, pins = (
        protocol.get("target"),
        protocol.get("arms"),
        protocol.get("reference_pins"),
    )
    required = {
        "requests",
        "truth",
        "baseline_predictions",
        "baseline_freeze",
        "baseline_report",
        "evaluator",
    }
    if protocol.get("status") != "preregistered_before_new_inference":
        raise ValueError("Invalid Astra benchmark protocol status")
    if not isinstance(target, dict) or target.get("expected_measure_indices") != list(range(1, 8)):
        raise ValueError("Protocol must predeclare Aviador measures 1..7")
    if int(target.get("system_index", -1)) != 8:
        raise ValueError("Protocol must predeclare Aviador system 8")
    if not isinstance(arms, dict) or tuple(sorted(arms)) != ARMS:
        raise ValueError(f"Protocol arms must be exactly {list(ARMS)}")
    if any(not isinstance(arms[arm].get("input_pins"), list) for arm in ARMS):
        raise ValueError("Each protocol arm must contain input_pins")
    if not isinstance(pins, dict) or set(pins) != required:
        raise ValueError(f"Protocol reference_pins must be exactly {sorted(required)}")


def _verify_freeze(path: Path, requests: Path, baseline: Path) -> list[Mapping[str, Any]]:
    freeze = _json(path)
    if (
        freeze.get("status") != "frozen_before_truth"
        or freeze.get("split") != "validation"
        or int(freeze.get("target_count", -1)) != 7
    ):
        raise ValueError("Invalid seven-target validation baseline freeze")
    if freeze.get("requests", {}).get("sha256") != benchmark._sha256(requests):
        raise ValueError("Baseline freeze requests hash mismatch")
    records = freeze.get("artifacts")
    if not isinstance(records, list):
        raise ValueError("Baseline freeze artifacts must be an array")
    paths = _verify_many(path.parent, records, "baseline artifact")
    if paths.get(baseline) != benchmark._sha256(baseline):
        raise ValueError("Baseline predictions are not pinned by their freeze")
    return records


def _adapt(
    payload: Mapping[str, Any], identities: Sequence[Mapping[str, Any]], expected: list[int]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    measures = payload.get("measures") if isinstance(payload, dict) else None
    if not isinstance(measures, list) or not isinstance(payload.get("context"), dict):
        raise ValueError("Submission requires measures and context")
    _strings(payload.get("notation_uncertainties"), "notation_uncertainties")
    indexed, duplicates = {}, []
    for measure in measures:
        if not isinstance(measure, dict) or not isinstance(measure.get("events"), list):
            raise ValueError("Each submission measure requires an events array")
        index = measure.get("measure_index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("measure_index must be an integer")
        _strings(measure.get("uncertainties"), f"measure {index} uncertainties")
        if index in indexed:
            duplicates.append(index)
        else:
            indexed[index] = measure
        for event in measure["events"]:
            _validate_event(event)
    missing = sorted(set(expected) - set(indexed))
    extra = sorted(set(indexed) - set(expected))
    headline_valid = not extra and not duplicates
    validation = {
        "status": "valid" if headline_valid and not missing else "invalid_measure_coverage",
        "headline_metrics_valid": headline_valid,
        "expected_measure_indices": expected,
        "submitted_measure_indices": [row["measure_index"] for row in measures],
        "missing_measure_indices": missing,
        "extra_measure_indices": extra,
        "duplicate_measure_indices": sorted(set(duplicates)),
        "submitted_event_count": sum(len(row["events"]) for row in measures),
    }
    if not headline_valid:
        return [], validation
    rows = []
    for identity in identities:
        index = int(identity["system_measure_index"])
        if index not in indexed:  # The historical scorer supplies an empty prediction.
            continue
        notes, rests = [], []
        for event in indexed[index]["events"]:
            item = {key: event[key] for key in ("onset_beats", "duration_beats")}
            if event["kind"] == "rest":
                rests.append(item)
            else:
                notes.append({**item, "pitch_midi": pitch_name_to_midi(event["pitch"])})
        rows.append({"identity": dict(identity), "notes": notes, "rests": rests})
    return rows, validation


def _validate_event(event: Any) -> None:
    if not isinstance(event, dict) or event.get("kind") not in ("note", "rest"):
        raise ValueError("Each event must be a note or rest object")
    onset, duration = event.get("onset_beats"), event.get("duration_beats")
    numbers = (onset, duration)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in numbers):
        raise ValueError("Event onset and duration must be numbers")
    if not math.isfinite(onset) or not math.isfinite(duration) or onset < 0 or duration <= 0:
        raise ValueError("Event onset/duration must be finite and non-negative/positive")
    pitch = event.get("pitch")
    if event["kind"] == "rest" and pitch is not None:
        raise ValueError("Rest pitch must be null")
    if event["kind"] == "note" and not isinstance(pitch, str):
        raise ValueError("Note pitch must be a scientific pitch-name string")
    if event["kind"] == "note":
        pitch_name_to_midi(pitch)
    if event.get("accidental") is not None and not isinstance(event["accidental"], str):
        raise ValueError("Event accidental must be a string or null")


def _geometry_errors(
    truth: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    truths = {benchmark._identity_key(row["identity"]): row for row in truth}
    errors = []
    for row in predictions:
        identity = row["identity"]
        try:
            benchmark._validate_prediction_events(
                row["notes"],
                row["rests"],
                measure_extent=Fraction(
                    str(truths[benchmark._identity_key(identity)]["measure_extent_beats"])
                ),
                identity=identity,
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append({"measure_index": identity["system_measure_index"], "error": str(exc)})
    return errors


def _target_rows(rows: Sequence[Mapping[str, Any]], target: Mapping[str, Any]) -> list[dict]:
    expected = target["expected_measure_indices"]
    selected = [
        dict(row)
        for row in rows
        if row["identity"]["slug"] == target["slug"]
        and int(row["identity"]["system_index"]) == int(target["system_index"])
        and int(row["identity"]["system_measure_index"]) in expected
    ]
    selected.sort(key=lambda row: int(row["identity"]["system_measure_index"]))
    actual = [int(row["identity"]["system_measure_index"]) for row in selected]
    if actual != expected:
        raise ValueError(f"Target rows do not exactly cover measures {expected}: {actual}")
    return selected


def _same_identities(rows: Sequence[Mapping], identities: Sequence[Mapping], label: str) -> None:
    actual = [benchmark._identity_key(row["identity"]) for row in rows]
    expected = [benchmark._identity_key(identity) for identity in identities]
    if actual != expected:
        raise ValueError(f"{label} do not exactly match target request identities")


def _verify(root: Path, record: Mapping[str, Any], label: str) -> Path:
    raw = Path(str(record.get("path", ""))).expanduser()
    path = (raw if raw.is_absolute() else root / raw).resolve()
    if not path.is_file() and not raw.is_absolute():
        path = (REPO_ROOT / raw).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pinned {label} does not exist: {path}")
    if benchmark._sha256(path) != record.get("sha256"):
        raise ValueError(f"{label} hash mismatch: {path}")
    return path


def _verify_many(root: Path, records: Sequence[Mapping], label: str) -> dict[Path, str]:
    return {_verify(root, record, label): str(record["sha256"]) for record in records}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _strings(value: Any, label: str) -> None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be an array of strings")


if __name__ == "__main__":
    raise SystemExit(main())
