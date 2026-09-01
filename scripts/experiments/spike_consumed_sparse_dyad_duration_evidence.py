"""Evaluate sparse-dyad dotted-half duration evidence on consumed scores.

The fixed sparse repair already requires a shared-stem head pair and aligned
augmentation-dot candidates. This postmortem checks whether that accepted
visual pattern consistently denotes one three-beat dyad in already-opened
MusicXML. It does not alter inference or support a heldout claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import evaluate_second_score_heldout as musicxml_truth  # noqa: E402
from scripts.experiments import sparse_dyad_duration_evidence as duration_evidence  # noqa: E402

SCHEMA_VERSION = 1
OUTPUT_VERSION = "consumed_sparse_dyad_duration_evidence_v1"
DEFAULT_OUTPUT = REPO_ROOT / "out/vlm_melody_consumed_training/sparse_dyad_duration_v1"

SPARSE_REPORT = Path("out/vlm_melody_consumed_training/sparse_stem_dyad_repair_v3/report.json")
A_MEDIO_ROOT = Path(
    "out/local_restricted/jaime-llanos_7_a-medio-palo_pasillo_m-garavito-w/"
    "vlm_melody_independent_multihead_recovery_gate/v1/system_007/evaluation_v1"
)
DESDE_ROOT = Path(
    "out/local_restricted/jaime-llanos_26_desde-lejos_pasillo_b-b/"
    "vlm_melody_independent_sparse_dyad_repair_gate/v1/system_007/evaluation_v1"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", type=Path, default=REPO_ROOT / "out")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        report = run_spike(out_dir=args.out_dir, output_dir=args.output_dir)
    except (FileExistsError, FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(report["artifacts"]["report_json"])
    return 0


def run_spike(*, out_dir: Path, output_dir: Path) -> dict[str, Any]:
    out_dir = out_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite duration-evidence spike: {output_dir}")
    temp_dir = output_dir.with_name(f".{output_dir.name}.tmp")
    if temp_dir.exists():
        raise FileExistsError(f"Refusing stale duration-evidence temp dir: {temp_dir}")

    sparse_report_path = out_dir / SPARSE_REPORT.relative_to("out")
    sparse_report = _read_json(sparse_report_path)
    _validate_sparse_report(sparse_report)
    _verify_pins(sparse_report["provenance"])

    a_medio_root = out_dir / A_MEDIO_ROOT.relative_to("out")
    desde_root = out_dir / DESDE_ROOT.relative_to("out")
    positives = [
        *_a_medio_positive_records(sparse_report, a_medio_root),
        _desde_positive_record(desde_root),
    ]
    negative_records = _negative_records(sparse_report, desde_root)
    exact_positive_count = sum(record["truth_exact_dotted_half_dyad"] for record in positives)
    false_application_count = sum(record["evidence"]["applied"] for record in negative_records)
    gate = {
        "all_visual_positives_are_exact_three_beat_dyads": exact_positive_count == len(positives),
        "all_nonaccepted_sparse_decisions_remain_unchanged": false_application_count == 0,
        "a_medio_consumed_rule_previously_passed": sparse_report["gate"][
            "passed_consumed_model_selection"
        ]
        is True,
        "desde_independent_visual_identity_previously_passed": _read_json(
            desde_root / "report.json"
        )["raw_image_review"]["head_pixel_identity_passed"]
        is True,
    }
    gate["passed_consumed_duration_evidence"] = all(gate.values())
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "vlm_melody_consumed_sparse_dyad_duration_evidence_report",
        "version": OUTPUT_VERSION,
        "status": "consumed_model_selection_only",
        "truth_used_for_model_selection": True,
        "eligible_for_heldout_claim": False,
        "runtime_promotion_supported": False,
        "rule": {
            "evidence_kind": duration_evidence.EVIDENCE_KIND,
            "duration_beats": duration_evidence.DOTTED_HALF_BEATS,
            "scope": "accepted sparse shared-stem dyad in a three-beat measure",
            "residual_rest_policy": "suppress only inside this full-measure visual pattern",
        },
        "summary": {
            "positive_measure_count": len(positives),
            "exact_positive_count": exact_positive_count,
            "negative_decision_count": len(negative_records),
            "false_application_count": false_application_count,
        },
        "gate": gate,
        "decision": (
            "eligible_for_opt_in_full_event_v2"
            if gate["passed_consumed_duration_evidence"]
            else "not_selected"
        ),
        "positive_records": positives,
        "negative_records": negative_records,
        "pins": {
            "sparse_report": _pin(sparse_report_path),
            "a_medio_musicxml": _pin(a_medio_root / "source.musicxml"),
            "a_medio_mapping": _pin(a_medio_root / "mapping.json"),
            "desde_report": _pin(desde_root / "report.json"),
            "desde_diagnostics": _pin(desde_root / "frozen_diagnostics.jsonl"),
            "desde_musicxml": _pin(desde_root / "source.musicxml"),
            "desde_mapping": _pin(desde_root / "mapping.json"),
        },
        "implementation": {
            "evaluator": _pin(Path(__file__)),
            "duration_rule": _pin(Path(duration_evidence.__file__)),
        },
    }

    temp_dir.mkdir(parents=True)
    try:
        _write_json(temp_dir / "report.json", report)
        (temp_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
        temp_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    report["artifacts"] = {
        "report_json": str(output_dir / "report.json"),
        "report_markdown": str(output_dir / "report.md"),
    }
    return report


def _a_medio_positive_records(sparse_report: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    musicxml_path = root / "source.musicxml"
    truth = musicxml_truth.load_musicxml_truth(musicxml_path)
    written_durations = _written_note_durations_by_measure(musicxml_path)
    mapping = _mapping_by_crop(_read_json(root / "mapping.json"))
    records = sparse_report["results"]["a_medio_palo"]["records"]
    return [
        _positive_record(
            score="a_medio_palo",
            crop_index=int(record["automatic_crop_index"]),
            decision=record["decision"],
            candidate_ids=record["repaired_candidate_ids"],
            truth=truth,
            physical_measure=mapping[int(record["automatic_crop_index"])],
            written_durations=written_durations[mapping[int(record["automatic_crop_index"])]],
        )
        for record in records
        if record["decision"]["accepted"] is True
    ]


def _desde_positive_record(root: Path) -> dict[str, Any]:
    diagnostics = _read_jsonl(root / "frozen_diagnostics.jsonl")
    accepted = [row for row in diagnostics if row["sparse_repair"]["accepted"] is True]
    if len(accepted) != 1:
        raise ValueError("Expected exactly one accepted Desde Lejos sparse repair")
    row = accepted[0]
    crop = int(row["identity"]["automatic_measure_index"])
    decision = row["sparse_repair"]
    mapping = _mapping_by_crop(_read_json(root / "mapping.json"))
    musicxml_path = root / "source.musicxml"
    truth = musicxml_truth.load_musicxml_truth(musicxml_path)
    written_durations = _written_note_durations_by_measure(musicxml_path)
    return _positive_record(
        score="desde_lejos",
        crop_index=crop,
        decision=decision,
        candidate_ids=decision["proposed_ids"],
        truth=truth,
        physical_measure=mapping[crop],
        written_durations=written_durations[mapping[crop]],
    )


def _positive_record(
    *,
    score: str,
    crop_index: int,
    decision: Mapping[str, Any],
    candidate_ids: Sequence[str],
    truth: Any,
    physical_measure: int,
    written_durations: Sequence[float],
) -> dict[str, Any]:
    lane = [
        {"candidate_id": candidate_id, "onset_group_index": 1} for candidate_id in candidate_ids
    ]
    evidence = duration_evidence.derive_dotted_half_duration_evidence(
        decision,
        lane,
        expected_measure_beats=3.0,
    )
    notes = [
        dict(note)
        for note in truth.payload.get("notes") or []
        if int(note["measure"]) == physical_measure
    ]
    onsets = sorted({float(note["onset_beats"]) for note in notes})
    sounding_durations = [float(note["duration_beats"]) for note in notes]
    written_durations = [float(value) for value in written_durations]
    rests = list(truth.rests_by_measure.get(physical_measure, ()))
    exact = (
        len(notes) == 2
        and onsets == [0.0]
        and written_durations == [3.0, 3.0]
        and float(truth.measure_extents[physical_measure]) == 3.0
        and not rests
    )
    return {
        "score": score,
        "automatic_crop_index": crop_index,
        "physical_measure_number": physical_measure,
        "evidence": evidence,
        "truth_note_count": len(notes),
        "truth_onsets_beats": onsets,
        "truth_written_durations_beats": written_durations,
        "truth_sounding_durations_beats": sounding_durations,
        "truth_measure_extent_beats": float(truth.measure_extents[physical_measure]),
        "truth_rest_count": len(rests),
        "truth_exact_dotted_half_dyad": exact,
    }


def _negative_records(sparse_report: Mapping[str, Any], desde_root: Path) -> list[dict[str, Any]]:
    rows = []
    for score, result in sparse_report["results"].items():
        for record in result["records"]:
            decision = record["decision"]
            if decision["accepted"] is True:
                continue
            evidence = duration_evidence.derive_dotted_half_duration_evidence(
                decision,
                [],
                expected_measure_beats=3.0,
            )
            rows.append(
                {
                    "score": score,
                    "automatic_crop_index": int(record["automatic_crop_index"]),
                    "evidence": evidence,
                }
            )
    for row in _read_jsonl(desde_root / "frozen_diagnostics.jsonl"):
        decision = row["sparse_repair"]
        if decision["accepted"] is True:
            continue
        rows.append(
            {
                "score": "desde_lejos",
                "automatic_crop_index": int(row["identity"]["automatic_measure_index"]),
                "evidence": duration_evidence.derive_dotted_half_duration_evidence(
                    decision,
                    [],
                    expected_measure_beats=3.0,
                ),
            }
        )
    return rows


def _validate_sparse_report(report: Mapping[str, Any]) -> None:
    if (
        report.get("version") != "consumed_sparse_shared_stem_dyad_repair_v3"
        or report.get("status") != "consumed_model_selection_only"
        or report.get("truth_used_for_model_selection") is not True
        or report.get("gate", {}).get("passed_consumed_model_selection") is not True
    ):
        raise ValueError("Sparse dyad consumed report contract mismatch")


def _verify_pins(provenance: Mapping[str, Any]) -> None:
    for score, records in provenance.items():
        for name, record in records.items():
            path = Path(str(record["path"]))
            if not path.is_file() or _sha256(path) != str(record["sha256"]):
                raise ValueError(f"Sparse report provenance drift: {score}.{name}")


def _mapping_by_crop(payload: Mapping[str, Any]) -> dict[int, int]:
    mapping = {}
    for crop in payload.get("automatic_crops") or []:
        spans = crop.get("physical_measure_spans") or []
        if len(spans) != 1:
            raise ValueError("Duration evidence requires one physical measure per crop")
        mapping[int(crop["automatic_crop_index"])] = int(spans[0]["measure_number"])
    if not mapping:
        raise ValueError("Duration evidence mapping is empty")
    return mapping


def _written_note_durations_by_measure(path: Path) -> dict[int, list[float]]:
    root = ET.parse(path).getroot()
    musicxml_truth._strip_namespaces(root)
    parts = root.findall("part")
    if len(parts) != 1:
        raise ValueError(f"Expected exactly one MusicXML part, found {len(parts)}")
    divisions = 1
    durations: dict[int, list[float]] = {}
    for fallback, measure in enumerate(parts[0].findall("measure"), start=1):
        number = musicxml_truth._measure_number(measure, fallback)
        attributes = measure.find("attributes")
        if attributes is not None:
            divisions = int(attributes.findtext("divisions", str(divisions)))
        if divisions <= 0:
            raise ValueError(f"Invalid MusicXML divisions in measure {number}")
        durations[number] = [
            int(note.findtext("duration", "0")) / divisions
            for note in measure.findall("note")
            if note.find("rest") is None and int(note.findtext("duration", "0")) > 0
        ]
    return durations


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.resolve().read_text(encoding="utf-8").splitlines() if line
    ]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected non-empty JSONL objects: {path}")
    return rows


def _pin(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": _sha256(resolved), "bytes": resolved.stat().st_size}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    gate = report["gate"]
    lines = [
        "# Sparse-dyad duration evidence",
        "",
        f"Decision: `{report['decision']}`.",
        "",
        f"- Visual positives: `{summary['positive_measure_count']}`",
        f"- Exact three-beat dyads: `{summary['exact_positive_count']}`",
        f"- Non-accepted decisions: `{summary['negative_decision_count']}`",
        f"- False applications: `{summary['false_application_count']}`",
        "",
        "This is consumed model-selection evidence. It does not change inference or support "
        "a heldout claim.",
        "",
        f"Consumed gate passed: `{gate['passed_consumed_duration_evidence']}`.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
