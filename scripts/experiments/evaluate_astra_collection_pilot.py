"""Score sealed source-only excerpts without changing previous experiments."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from score2abc.events import STEP_TO_SEMITONE  # noqa: E402
from scripts import build_vlm_melody_event_benchmark as benchmark  # noqa: E402
from scripts.experiments.evaluate_astra_music_benchmark import (  # noqa: E402
    _adapt,
    _geometry_errors,
)

TARGET_IDS = ("carrizal_s4", "gatoe_fique_s3", "coqueteos_s2")
PROTOCOL_STATUS = "preregistered_before_inference"
MODEL = "gpt-6-astra"
SEAL_KIND = "astra_collection_pilot_submission_seal"
SEAL_STATUS = "all_submissions_frozen_before_truth"
REPORT_LABEL = "consumed assisted diagnostic pilot; source-only isolation failed"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(path: Path, digest: str) -> None:
    if not path.is_file() or benchmark._sha256(path) != digest:
        raise ValueError(f"Pinned file missing or changed: {path}")


def verify_inputs(root: Path) -> tuple[dict, dict, dict]:
    """Verify every submission and reference hash before opening any reference."""
    protocol = read_json(root / "protocol.json")
    seal = read_json(root / "submission_seal.json")
    if protocol.get("status") != PROTOCOL_STATUS or protocol.get("model") != MODEL:
        raise ValueError("Unexpected pilot protocol status or model")
    if set(protocol.get("workers", {})) != set(TARGET_IDS):
        raise ValueError("Pilot protocol workers must exactly match the three targets")
    if seal.get("kind") != SEAL_KIND or seal.get("status") != SEAL_STATUS:
        raise ValueError("Submission seal is not the completed pilot seal")
    verify(root / "protocol.json", seal["protocol_sha256"])
    deviation_path = root / "protocol_deviation.json"
    verify(deviation_path, seal["protocol_deviation_sha256"])
    deviation = read_json(deviation_path)
    if deviation.get("source_only_protocol_passed") is not False:
        raise ValueError("This pilot must record the failed source-only isolation contract")
    if deviation.get("allowed_label") != REPORT_LABEL:
        raise ValueError("Protocol deviation does not authorize the required diagnostic label")
    verify(root / "selection_plan.json", protocol["selection_plan_sha256"])
    plan = read_json(root / "selection_plan.json")
    if tuple(row["id"] for row in plan["targets"]) != TARGET_IDS:
        raise ValueError("Unexpected pilot targets or ordering")
    if set(seal["submissions"]) != set(TARGET_IDS):
        raise ValueError("All three submissions must be sealed before truth access")
    for record in seal["files"]:
        verify(root / record["path"], record["sha256"])
    provenance = {
        "worker_inputs": {},
        "submissions": {},
        "preregistered_original_submissions": {},
        "sources": {},
        "human_truth": {},
        "historical_references": {},
        "scorers": dict(protocol["scorer_pins"]),
        "protocol_deviation": {
            "path": str(deviation_path),
            "sha256": seal["protocol_deviation_sha256"],
        },
    }
    archived_paths = set()
    for target_id in TARGET_IDS:
        original = Path(protocol["workers"][target_id]["directory"])
        provenance["worker_inputs"][target_id] = protocol["workers"][target_id]["inputs"]
        for record in protocol["workers"][target_id]["inputs"]:
            verify(Path(record["path"]), record["sha256"])
        provenance["submissions"][target_id] = seal["submissions"][target_id]
        provenance["preregistered_original_submissions"][target_id] = {}
        for filename, record in seal["submissions"][target_id].items():
            if filename not in ("submission.json", "transcription.abc", "uncertainties.md"):
                raise ValueError("Unexpected submission file")
            archived = (root / record["path"]).resolve()
            if archived in archived_paths:
                raise ValueError("Submission archives must use distinct paths")
            archived_paths.add(archived)
            original_path = original / "result" / filename
            verify(archived, record["sha256"])
            verify(original_path, record["sha256"])
            provenance["preregistered_original_submissions"][target_id][filename] = {
                "path": str(original_path),
                "sha256": record["sha256"],
            }
        if set(seal["submissions"][target_id]) != {
            "submission.json",
            "transcription.abc",
            "uncertainties.md",
        }:
            raise ValueError("Incomplete submission")
    for relative, digest in protocol["scorer_pins"].items():
        verify(REPO_ROOT / relative, digest)
    if benchmark._sha256(Path(__file__)) != seal["evaluator_sha256"]:
        correction_path = root / "evaluator_correction_v2.json"
        correction = read_json(correction_path)
        verify(root / "evaluator_correction_v1.json", correction["previous_correction_sha256"])
        verify(root / "evaluator_corrected_v1.py", correction["previous_implementation_sha256"])
        verify(root / "submission_seal.json", correction["submission_seal_sha256"])
        if correction["original_evaluator_sha256"] != seal["evaluator_sha256"]:
            raise ValueError("Evaluator correction does not preserve the sealed implementation")
        verify(root / "evaluator_sealed.py", seal["evaluator_sha256"])
        verify(Path(__file__), correction["corrected_evaluator_sha256"])
        provenance["evaluator_correction"] = {
            "path": str(correction_path),
            "sha256": benchmark._sha256(correction_path),
        }
    else:
        verify(Path(__file__), seal["evaluator_sha256"])
    for target in plan["targets"]:
        provenance["sources"][target["id"]] = target["source_pdf"]
        provenance["human_truth"][target["id"]] = target["human_truth"]
        provenance["historical_references"][target["id"]] = {
            "mapping": target["historical_mapping"],
            "baseline": target["historical_baseline"],
        }
        verify(REPO_ROOT / target["source_pdf"]["path"], target["source_pdf"]["sha256"])
        for reference in target["human_truth"].values():
            verify(REPO_ROOT / reference["path"], reference["sha256"])
        for section in (target["historical_mapping"], target["historical_baseline"]):
            for name, value in section.items():
                if name.endswith("_path"):
                    verify(REPO_ROOT / value, section[name[:-5] + "_sha256"])
    return plan, seal, provenance


def written_truth(path: Path, slug: str, system: int, offset: int = 0) -> list[dict]:
    """Read written durations/rests, retaining both ends of cross-bar ties."""
    xml = ET.parse(path).getroot()
    for element in xml.iter():
        element.tag = element.tag.rsplit("}", 1)[-1]
    parts = xml.findall("part")
    if len(parts) != 1:
        raise ValueError("The pilot requires a single MusicXML part")
    divisions, meter = 1, None
    rows = []
    for index, measure in enumerate(parts[0].findall("measure"), 1):
        cursor = last = extent = Fraction(0)
        notes, rests = [], []
        for item in measure:
            if item.tag == "attributes":
                divisions = int(item.findtext("divisions", str(divisions)))
                if divisions <= 0:
                    raise ValueError("Non-positive divisions")
                time = item.find("time")
                if time is not None:
                    meter = Fraction(
                        int(time.findtext("beats")) * 4, int(time.findtext("beat-type"))
                    )
                continue
            if item.tag in ("backup", "forward"):
                cursor += Fraction(int(item.findtext("duration", "0")), divisions) * (
                    -1 if item.tag == "backup" else 1
                )
                if cursor < 0:
                    raise ValueError("MusicXML cursor before measure start")
                extent = max(extent, cursor)
                continue
            if item.tag != "note":
                continue
            if item.find("grace") is not None or item.find("unpitched") is not None:
                raise ValueError(
                    "Grace/unpitched reference events need a separate scoring contract"
                )
            duration = Fraction(int(item.findtext("duration", "0")), divisions)
            if duration <= 0:
                raise ValueError("Reference note/rest lacks written duration")
            chord = item.find("chord") is not None
            onset = last if chord else cursor
            event = {"onset_beats": float(onset), "duration_beats": float(duration)}
            if item.find("rest") is not None:
                rests.append(event)
            else:
                pitch = item.find("pitch")
                if pitch is None:
                    raise ValueError("Reference event lacks pitch/rest")
                alter = Fraction(pitch.findtext("alter", "0"))
                if alter.denominator != 1:
                    raise ValueError("Microtonal pitch is outside this MIDI benchmark")
                event["pitch_midi"] = (
                    (int(pitch.findtext("octave")) + 1) * 12
                    + STEP_TO_SEMITONE[pitch.findtext("step")]
                    + int(alter)
                )
                notes.append(event)
            extent = max(extent, onset + duration)
            if not chord:
                last, cursor = onset, cursor + duration
        if meter is None:
            raise ValueError("Reference has no explicit meter")
        rows.append(
            {
                "identity": {
                    "slug": slug,
                    "system_index": system,
                    "system_measure_index": index,
                    "global_measure_index": offset + index,
                },
                "source_measure_number": measure.get("number"),
                "measure_extent_beats": float(extent),
                "nominal_meter_beats": float(meter),
                "notes": sorted(notes, key=lambda e: (e["onset_beats"], e["pitch_midi"])),
                "rests": sorted(rests, key=lambda e: e["onset_beats"]),
            }
        )
    return rows


def compose(rows: list[dict], historical_truth: list[dict], mapping: dict) -> list[dict]:
    """Compose fixed physical measures into historical crops; never retune events."""
    indexed = {row["identity"]["system_measure_index"]: row for row in rows}
    composed = []
    for historical in historical_truth:
        crop = historical["identity"]["system_measure_index"]
        output = {"identity": historical["identity"], "notes": [], "rests": []}
        for shift, physical in enumerate(mapping[str(crop)]):
            row = indexed.get(physical, {})
            for kind in ("notes", "rests"):
                output[kind].extend(
                    {**event, "onset_beats": event["onset_beats"] + shift * 3}
                    for event in row.get(kind, [])
                )
        composed.append(output)
    return composed


def validate_mapping(mapping: dict, physical_count: int, historical_truth: list[dict]) -> None:
    """Require a one-use physical partition and exact historical crop coverage."""
    if not isinstance(mapping, dict):
        raise ValueError("Historical mapping must be an object")
    historical_crops = [row["identity"]["system_measure_index"] for row in historical_truth]
    if len(historical_crops) != len(set(historical_crops)):
        raise ValueError("Historical truth has duplicate crop identities")
    try:
        mapping_crops = [int(value) for value in mapping]
    except (TypeError, ValueError) as exc:
        raise ValueError("Historical mapping crop keys must be integers") from exc
    if set(mapping_crops) != set(historical_crops) or len(mapping_crops) != len(historical_crops):
        raise ValueError("Historical mapping does not exactly cover historical crops")
    physical = []
    for values in mapping.values():
        if not isinstance(values, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise ValueError("Historical mapping physical measures must be integer arrays")
        physical.extend(values)
    if sorted(physical) != list(range(1, physical_count + 1)):
        raise ValueError("Historical mapping must partition every physical measure exactly once")


def _normalized_events(rows: list[dict]) -> list[dict]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "notes": sorted(
                    (
                        float(event["onset_beats"]),
                        float(event["duration_beats"]),
                        int(event["pitch_midi"]),
                    )
                    for event in row.get("notes", [])
                ),
                "rests": sorted(
                    (float(event["onset_beats"]), float(event["duration_beats"]))
                    for event in row.get("rests", [])
                ),
            }
        )
    return normalized


def adapt_submission(
    submission: Any, identities: list[dict], expected: list[int]
) -> tuple[list[dict], dict]:
    """Convert one target without allowing malformed input to abort the pilot."""
    try:
        return _adapt(submission, identities, expected)
    except (KeyError, TypeError, ValueError) as exc:
        return invalid_submission(exc, expected)


def invalid_submission(error: Exception, expected: list[int]) -> tuple[list[dict], dict]:
    """Represent a syntactically unreadable sealed JSON file as one invalid target."""
    return [], {
        "status": "invalid_submission",
        "headline_metrics_valid": False,
        "expected_measure_indices": expected,
        "submitted_measure_indices": None,
        "missing_measure_indices": None,
        "extra_measure_indices": None,
        "duplicate_measure_indices": None,
        "submitted_event_count": None,
        "validation_errors": [str(error)],
        "event_geometry_errors": [],
    }


def meter_valid_count(rows: list[dict], truth: list[dict]) -> int:
    indexed = {row["identity"]["system_measure_index"]: row for row in rows}
    valid = 0
    for reference in truth:
        row = indexed.get(reference["identity"]["system_measure_index"], {})
        intervals = sorted(
            (event["onset_beats"], event["onset_beats"] + event["duration_beats"])
            for kind in ("notes", "rests")
            for event in row.get(kind, [])
        )
        cursor, gap = 0.0, False
        for start, end in intervals:
            gap |= start > cursor + 1e-6
            cursor = max(cursor, end)
        valid += (
            bool(intervals) and not gap and abs(cursor - reference["nominal_meter_beats"]) < 1e-6
        )
    return valid


def pitch_comparison(target: dict, predictions: list[dict], headline_valid: bool) -> dict:
    """Replay the archived pitch/count scorer for the one-to-one Gato excerpt."""
    path = REPO_ROOT / target["historical_baseline"]["evaluator_path"]
    spec = importlib.util.spec_from_file_location("astra_pilot_archived_pitch_scorer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    truth = benchmark._read_jsonl(
        REPO_ROOT / target["human_truth"]["existing_pitch_count_snapshot"]["path"]
    )
    expected_count = target["historical_mapping"]["physical_measure_count"]
    reference_crops = [row["automatic_crop_index"] for row in truth]
    if reference_crops != list(range(1, expected_count + 1)):
        raise ValueError("Gato historical truth does not exactly cover the expected crops")
    old_rows = benchmark._read_jsonl(REPO_ROOT / target["historical_baseline"]["predictions_path"])
    previous = read_json(REPO_ROOT / target["historical_baseline"]["report_path"])
    old_by_crop = {row["identity"]["system_measure_index"]: row for row in old_rows}
    by_crop = {row["identity"]["system_measure_index"]: row for row in predictions}
    by_crop = {
        row["automatic_crop_index"]: by_crop.get(row["automatic_crop_index"], {}) for row in truth
    }
    kwargs = {"target": previous["target"], "mapping_mode": previous["mapping_mode"]}
    old = module.evaluate_pitch_only(truth, old_by_crop, **kwargs)["metrics"]
    if old != previous["metrics"]:
        raise ValueError("Gato historical pitch/count baseline not reproduced")
    result = {
        "baseline_reproduced": True,
        "baseline": old,
        "astra": None,
        "metric_contract": "archived ordered-pitch alignment and note count only",
        "headline_role": "secondary_only; never substitutes for withheld full-event metrics",
    }
    if not headline_valid:
        result.update(
            supported=False,
            unsupported_reason=(
                "Native submission is invalid; Astra historical headline is suppressed."
            ),
        )
        return result
    result.update(
        supported=True,
        astra=module.evaluate_pitch_only(truth, by_crop, **kwargs)["metrics"],
    )
    return result


def full_event_comparison(
    target: dict,
    native_truth: list[dict],
    predictions: list[dict],
    headline_valid: bool,
    unsupported_reason: str | None = None,
) -> dict:
    """Reproduce a baseline and compare only when written-event semantics match exactly."""
    snapshot = target["human_truth"]["canonical_full_event_snapshot"]
    old_truth = benchmark._read_jsonl(REPO_ROOT / snapshot["path"])
    baseline = benchmark._read_jsonl(REPO_ROOT / target["historical_baseline"]["predictions_path"])
    if target["id"] == "coqueteos_s2":
        baseline = coqueteos_benchmark_identities(baseline)
    mapping = target["historical_mapping"]["automatic_crop_to_physical_measures"]
    validate_mapping(mapping, len(native_truth), old_truth)
    old = benchmark.evaluate_predictions(old_truth, baseline)
    previous = read_json(REPO_ROOT / target["historical_baseline"]["report_path"])
    if target["id"] == "coqueteos_s2":
        old["summary"].update(coqueteos_meter_summary(target, old_truth, baseline, previous))
    previous_metrics = previous.get("metrics", previous)
    if old["summary"] != previous_metrics.get("summary"):
        raise ValueError(f"Historical baseline not reproduced: {target['id']}")
    result = {
        "baseline_reproduced": True,
        "baseline": old,
        "astra": None,
        "mapping": mapping,
        "limitation": target["metric_comparability"]["limitation"],
    }
    composed_truth = compose(native_truth, old_truth, mapping)
    if _normalized_events(composed_truth) != _normalized_events(old_truth):
        result.update(
            supported=False,
            unsupported_reason=(
                "Composed native written-event truth differs from the pinned historical truth; "
                "the historical lane may use different tie or event semantics."
            ),
        )
        return result
    if not headline_valid:
        result.update(
            supported=False,
            unsupported_reason=unsupported_reason
            or "Native submission is invalid; Astra historical headline is suppressed.",
        )
        return result
    result.update(
        supported=True,
        astra=benchmark.evaluate_predictions(old_truth, compose(predictions, old_truth, mapping)),
    )
    return result


def coqueteos_meter_summary(target: dict, truth_rows: list, baseline: list, previous: dict) -> dict:
    """Replay the archived extra meter summary using its recorded allowed context."""
    from scripts.experiments.evaluate_second_score_heldout import load_musicxml_truth

    path = REPO_ROOT / target["historical_baseline"]["evaluator_path"]
    spec = importlib.util.spec_from_file_location("astra_pilot_archived_meter_scorer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    context = previous["meter"]["context"]
    truth = load_musicxml_truth(REPO_ROOT / target["human_truth"]["musicxml"]["path"])
    meter = module._evaluate_meter(
        truth=truth,
        truth_rows=truth_rows,
        prediction_rows=baseline,
        mapping={
            int(k): v
            for k, v in target["historical_mapping"]["automatic_crop_to_physical_measures"].items()
        },
        allowed_context={
            "time_signature": context["allowed_time_signature"],
            "expected_measure_beats": context["allowed_expected_measure_beats"],
        },
        measure_length=Fraction(3),
    )
    if meter != previous["meter"]:
        raise ValueError("Historical Coqueteos meter report not reproduced")
    summary = meter["summary"]
    return {
        "meter_context_match": summary["context_match"],
        "meter_valid_crops": summary["valid_prediction_crops"],
        "meter_valid_crop_rate": summary["valid_prediction_crop_rate"],
        "meter_truth_valid_measures": summary["valid_truth_measures"],
        "meter_truth_valid_measure_rate": summary["valid_truth_measure_rate"],
    }


def coqueteos_benchmark_identities(rows: list[dict]) -> list[dict]:
    """Replay the archived evaluator's _benchmark_predictions identity adaptation."""
    by_crop = {int(row["identity"]["system_measure_index"]): row for row in rows}
    if len(by_crop) != len(rows):
        raise ValueError("Duplicate Coqueteos baseline crop")
    return [
        {
            **by_crop[crop],
            "identity": {
                **by_crop[crop]["identity"],
                "system_measure_index": crop,
                "global_measure_index": index,
            },
        }
        for index, crop in enumerate(sorted(by_crop), 1)
    ]


def evaluate(root: Path) -> dict[str, Any]:
    destination = root / "evaluation_v1"
    if destination.exists():
        raise ValueError("Refusing to overwrite a prior or partial evaluation")
    plan, seal, provenance = verify_inputs(root)
    reports, all_truth, all_predictions = {}, [], []
    artifacts = {}
    for ordinal, target in enumerate(plan["targets"]):
        key = target["id"]
        submission_record = seal["submissions"][key]["submission.json"]
        submission_path = root / submission_record["path"]
        verify(submission_path, submission_record["sha256"])
        submission_error = None
        try:
            submission = read_json(submission_path)
        except (UnicodeError, json.JSONDecodeError) as exc:
            submission, submission_error = None, exc
        # All three outputs and every reference hash have passed verification above.
        truth_record = target["human_truth"]["musicxml"]
        truth_path = REPO_ROOT / truth_record["path"]
        verify(truth_path, truth_record["sha256"])
        truth = written_truth(
            truth_path,
            target["slug"],
            target["system_index"],
            ordinal * 100,
        )
        mapping_contract = target["historical_mapping"]
        expected_count = mapping_contract.get("physical_measure_count") or sum(
            len(values)
            for values in mapping_contract["automatic_crop_to_physical_measures"].values()
        )
        if len(truth) != expected_count:
            raise ValueError("Reference measure count disagrees with preregistered mapping")
        expected = list(range(1, len(truth) + 1))
        if submission_error is None:
            predictions, validation = adapt_submission(
                submission, [row["identity"] for row in truth], expected
            )
        else:
            predictions, validation = invalid_submission(submission_error, expected)
        comparison_valid = validation["headline_metrics_valid"]
        reference_geometry = _geometry_errors(truth, truth)
        reference_supported = not reference_geometry
        geometry = (
            _geometry_errors(truth, predictions)
            if reference_supported and validation["headline_metrics_valid"]
            else []
        )
        validation["event_geometry_errors"] = geometry
        if geometry:
            validation["status"] = "invalid_event_geometry"
            validation["headline_metrics_valid"] = False
        report = {
            "validation": validation,
            "physical_measure_count": len(truth),
            "truth_note_count": sum(len(row["notes"]) for row in truth),
            "truth_rest_count": sum(len(row["rests"]) for row in truth),
            "native_metrics": None,
            "historical_comparison": None,
            "reference_validation": {
                "full_event_metrics_supported": reference_supported,
                "event_geometry_errors": reference_geometry,
            },
            "native_metrics_status": None,
        }
        native_headline_valid = validation["headline_metrics_valid"] and reference_supported
        if not reference_supported:
            report["native_metrics_status"] = "unsupported_reference_event_geometry"
        elif not validation["headline_metrics_valid"]:
            report["native_metrics_status"] = "invalid_submission"
        else:
            report["native_metrics_status"] = "supported"
        if native_headline_valid:
            report["native_metrics"] = benchmark.evaluate_predictions(truth, predictions)
            report["meter_valid_measures"] = meter_valid_count(predictions, truth)
            all_truth.extend(truth)
            all_predictions.extend(predictions)
        snapshot = target["human_truth"].get("canonical_full_event_snapshot")
        if snapshot:
            report["historical_comparison"] = full_event_comparison(
                target,
                truth,
                predictions,
                native_headline_valid,
                (
                    "Native written-event reference geometry is unsupported by the strict scorer; "
                    "the Astra historical full-event headline is suppressed."
                    if not reference_supported
                    else None
                ),
            )
        else:
            report["historical_comparison"] = pitch_comparison(
                target, predictions, comparison_valid
            )
        reports[key] = report
        artifacts[f"{key}_truth.jsonl"] = truth
        artifacts[f"{key}_predictions.jsonl"] = predictions
    all_valid = all(report["native_metrics"] is not None for report in reports.values())
    report = {
        "schema_version": 1,
        "label": REPORT_LABEL,
        "source_only_protocol_passed": False,
        "truth_opened_after_all_submissions_verified": True,
        "protocol_sha256": seal["protocol_sha256"],
        "submission_seal_sha256": benchmark._sha256(root / "submission_seal.json"),
        "evaluator_sha256": benchmark._sha256(Path(__file__)),
        "targets": reports,
        "pooled": benchmark.evaluate_predictions(all_truth, all_predictions) if all_valid else None,
        "pooled_status": "valid" if all_valid else "withheld_due_to_target_or_reference_contract",
        "macro_note_f1": (
            sum(r["native_metrics"]["summary"]["note_f1"] for r in reports.values()) / 3
            if all_valid
            else None
        ),
        "macro_exact_measure_rate": (
            sum(r["native_metrics"]["summary"]["exact_measure_rate"] for r in reports.values()) / 3
            if all_valid
            else None
        ),
        "pins": provenance,
        "limitations": [
            "Source-only tool access passed, but inherited repository context broke isolation.",
            "Previously consumed excerpts; not fresh generalization or complete scores.",
            "Written notes/rests only; chords, spelling, form, ties and slurs need separate audit.",
            "Gato'e Fique baseline covers pitch/count only; full-event metrics are standalone.",
        ],
    }
    rechecked_plan, rechecked_seal, rechecked_provenance = verify_inputs(root)
    if (rechecked_plan, rechecked_seal, rechecked_provenance) != (plan, seal, provenance):
        raise ValueError("Pilot inputs changed during evaluation")
    destination.mkdir()
    for filename, rows in artifacts.items():
        benchmark._write_jsonl(destination / filename, rows)
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    outcome = evaluate(args.root)
    print(
        json.dumps(
            {key: value["native_metrics"] for key, value in outcome["targets"].items()}, indent=2
        )
    )
