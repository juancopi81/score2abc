"""Materialize and score a supported sealed heldout exactly once.

This spike-only evaluator verifies every prepared and frozen hash before it
opens user-supplied MusicXML or mapping data. Frozen predictions contain only
ordered pitches and coordinates, so rhythm, rest, onset, duration, and meter
metrics are deliberately reported as unsupported rather than as zero.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from score2abc import musicxml as musicxml_utils  # noqa: E402
from scripts.experiments import freeze_fourth_score_heldout as fourth_freezer  # noqa: E402
from scripts.experiments import freeze_third_score_heldout as freezer  # noqa: E402
from scripts.experiments import run_third_score_heldout_inference as inference  # noqa: E402

SCHEMA_VERSION = 1
DEFAULT_EVALUATION_VERSION = "v1"
NOT_SCORED = "not_scored_missing_frozen_context"


@dataclass(frozen=True)
class HeldoutEvaluationSpec:
    gate: freezer.HeldoutGateSpec
    default_one_to_one_count: int

    @property
    def report_kind(self) -> str:
        return f"{self.gate.key}_pitch_only_one_shot_evaluation"

    @property
    def manifest_kind(self) -> str:
        return f"{self.gate.key}_post_freeze_evaluation_manifest"


THIRD_SCORE_EVALUATION = HeldoutEvaluationSpec(
    gate=freezer.THIRD_SCORE_GATE,
    default_one_to_one_count=7,
)
FOURTH_SCORE_EVALUATION = HeldoutEvaluationSpec(
    gate=fourth_freezer.FOURTH_SCORE_GATE,
    default_one_to_one_count=6,
)
EVALUATION_SPECS = {
    spec.gate.sealed_kind: spec
    for spec in (
        THIRD_SCORE_EVALUATION,
        FOURTH_SCORE_EVALUATION,
    )
}

# Backward-compatible names used by the established third-score fixtures.
EXPECTED_SEALED_KIND = THIRD_SCORE_EVALUATION.gate.sealed_kind
EXPECTED_FREEZE_KIND = THIRD_SCORE_EVALUATION.gate.freeze_kind
EXPECTED_PREPARED_KIND = THIRD_SCORE_EVALUATION.gate.prepare_kind


@dataclass(frozen=True)
class VisibleMusicXMLTruth:
    measure_numbers: tuple[int, ...]
    notes_by_measure: dict[int, tuple[dict[str, Any], ...]]
    time_signature: str
    key_fifths: int | None
    clef: tuple[str, int] | None


TruthLoader = Callable[[Path], VisibleMusicXMLTruth]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sealed_manifest", type=Path)
    parser.add_argument("--musicxml", type=Path, required=True)
    parser.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help=(
            "Optional crop-to-physical-measure mapping JSON. Required unless the "
            "MusicXML and frozen prediction set match the gate's configured one-to-one count."
        ),
    )
    parser.add_argument(
        "--evaluation-version",
        default=DEFAULT_EVALUATION_VERSION,
        help="Create-once output version (default: v1, written as evaluation_v1).",
    )
    args = parser.parse_args(argv)
    try:
        result = evaluate_frozen_heldout(
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


def evaluate_frozen_heldout(
    sealed_manifest_path: Path,
    *,
    musicxml_path: Path,
    mapping_path: Path | None = None,
    evaluation_version: str = DEFAULT_EVALUATION_VERSION,
    truth_loader: TruthLoader = lambda path: load_visible_musicxml_truth(path),
) -> dict[str, Any]:
    """Verify, materialize truth, and score one sealed heldout exactly once."""
    _validate_version(evaluation_version)
    sealed_manifest_path = sealed_manifest_path.expanduser().resolve()
    musicxml_path = musicxml_path.expanduser().resolve()
    mapping_path = mapping_path.expanduser().resolve() if mapping_path is not None else None

    # This is intentionally the first operation that opens any input file.
    frozen = verify_frozen_gate(sealed_manifest_path)
    namespace_root = frozen["namespace_root"]
    output_dir = namespace_root / f"evaluation_{evaluation_version}"
    temp_dir = namespace_root / f".evaluation_{evaluation_version}.tmp"
    prior_evaluations = sorted(namespace_root.glob("evaluation_*"))
    if prior_evaluations:
        raise FileExistsError(
            "One-shot evaluation already exists: "
            + ", ".join(str(path) for path in prior_evaluations)
        )
    stale_temps = sorted(namespace_root.glob(".evaluation_*.tmp"))
    if stale_temps:
        raise FileExistsError(
            "Stale one-shot evaluation directory exists: "
            + ", ".join(str(path) for path in stale_temps)
        )

    # Truth and mapping existence/content checks occur only after the freeze gate.
    if not musicxml_path.is_file():
        raise FileNotFoundError(f"User MusicXML does not exist: {musicxml_path}")
    if mapping_path is not None and not mapping_path.is_file():
        raise FileNotFoundError(f"Mapping JSON does not exist: {mapping_path}")

    musicxml_source_sha256 = freezer._sha256(musicxml_path)
    mapping_source_sha256 = freezer._sha256(mapping_path) if mapping_path else None
    truth = truth_loader(musicxml_path)
    crop_indices = tuple(sorted(frozen["predictions_by_crop"]))
    if mapping_path is None:
        mapping = _default_mapping(
            truth.measure_numbers,
            crop_indices,
            expected_count=frozen["evaluation_spec"].default_one_to_one_count,
        )
        mapping_mode = "deterministic_default_one_to_one"
    else:
        mapping = _load_mapping(mapping_path)
        mapping_mode = "explicit_user_mapping"
    materialized_mapping = validate_and_materialize_mapping(
        mapping,
        truth=truth,
        crop_indices=crop_indices,
        mode=mapping_mode,
    )
    truth_rows = build_truth_rows(
        frozen["requests_by_crop"],
        truth,
        materialized_mapping,
    )
    report = evaluate_pitch_only(
        truth_rows,
        frozen["predictions_by_crop"],
        target=frozen["target"],
        mapping_mode=mapping_mode,
        report_kind=frozen["evaluation_spec"].report_kind,
    )
    report["source_musicxml_context"] = {
        "time_signature": truth.time_signature,
        "key_fifths": truth.key_fifths,
        "clef": list(truth.clef) if truth.clef is not None else None,
    }

    temp_dir.mkdir(parents=False, exist_ok=False)
    try:
        source_snapshot = temp_dir / "source.musicxml"
        mapping_snapshot = temp_dir / "mapping.json"
        freeze_snapshot = temp_dir / "frozen_freeze.json"
        sealed_snapshot = temp_dir / "frozen_sealed_manifest.json"
        evaluator_snapshot = temp_dir / "evaluator.py"
        truth_path = temp_dir / "truth.jsonl"
        report_path = temp_dir / "report.json"

        shutil.copyfile(musicxml_path, source_snapshot)
        _write_json(mapping_snapshot, materialized_mapping)
        shutil.copyfile(frozen["freeze_path"], freeze_snapshot)
        shutil.copyfile(sealed_manifest_path, sealed_snapshot)
        shutil.copyfile(Path(__file__).resolve(), evaluator_snapshot)
        _write_jsonl(truth_path, truth_rows)

        pins = {
            "source_musicxml": _snapshot_record(
                source_snapshot,
                source_path=musicxml_path,
                source_sha256=musicxml_source_sha256,
            ),
            "mapping": _snapshot_record(
                mapping_snapshot,
                source_path=mapping_path,
                source_sha256=mapping_source_sha256,
                require_source_match=False,
            ),
            "freeze_manifest": _snapshot_record(
                freeze_snapshot,
                source_path=frozen["freeze_path"],
                source_sha256=frozen["freeze_sha256"],
            ),
            "sealed_manifest": _snapshot_record(
                sealed_snapshot,
                source_path=sealed_manifest_path,
                source_sha256=frozen["sealed_sha256"],
            ),
            "evaluator": _snapshot_record(
                evaluator_snapshot,
                source_path=Path(__file__).resolve(),
                source_sha256=freezer._sha256(Path(__file__).resolve()),
            ),
            "truth": _snapshot_record(truth_path),
        }
        report["pins"] = pins
        report["truth_materialization"] = {
            "visible_noteheads_include_tied_continuations": True,
            "pitch_uses_musicxml_sounding_alter": True,
            "physical_mapping_explicit": True,
        }
        _write_json(report_path, report)
        pins["report"] = _snapshot_record(report_path)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": frozen["evaluation_spec"].manifest_kind,
            "status": "evaluated_exactly_once_after_frozen_predictions",
            "create_once": True,
            "evaluation_version": evaluation_version,
            "target": frozen["target"],
            "truth_opened_after_all_frozen_hashes_verified": True,
            "mapping_mode": mapping_mode,
            "pins": pins,
        }
        _write_json(temp_dir / "manifest.json", manifest)

        verified_again = verify_frozen_gate(sealed_manifest_path)
        if (
            verified_again["freeze_sha256"] != frozen["freeze_sha256"]
            or verified_again["sealed_sha256"] != frozen["sealed_sha256"]
        ):
            raise ValueError("Frozen gate changed during one-shot evaluation")
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


def verify_frozen_gate(sealed_manifest_path: Path) -> dict[str, Any]:
    """Verify the sealed manifest, prepared inputs, and every frozen snapshot."""
    if not sealed_manifest_path.is_file():
        raise FileNotFoundError(f"Sealed manifest does not exist: {sealed_manifest_path}")
    sealed = _read_json(sealed_manifest_path)
    evaluation_spec = EVALUATION_SPECS.get(str(sealed.get("kind")))
    if evaluation_spec is None:
        raise ValueError(f"Unexpected sealed manifest kind: {sealed.get('kind')}")
    if sealed.get("status") != "frozen_awaiting_truth":
        raise ValueError("Sealed manifest is not awaiting truth")
    if sealed.get("split") != freezer.SPLIT_NAME or sealed.get("truth_accessed") is not False:
        raise ValueError("Sealed manifest is not a truth-blind fresh_heldout gate")

    frozen_dir = sealed_manifest_path.parent
    namespace_root = frozen_dir.parent
    inference.verify_frozen_outputs(frozen_dir)
    freeze_path = _safe_child(frozen_dir, str(sealed["freeze"]["path"]))
    freeze_sha256 = freezer._sha256(freeze_path)
    if freeze_sha256 != str(sealed["freeze"]["sha256"]):
        raise ValueError("Freeze manifest hash drift")
    freeze = _read_json(freeze_path)
    if freeze.get("kind") != evaluation_spec.gate.freeze_kind:
        raise ValueError(f"Unexpected freeze manifest kind: {freeze.get('kind')}")
    if freeze.get("status") != "frozen_awaiting_truth":
        raise ValueError("Freeze manifest is not awaiting truth")
    if freeze.get("split") != freezer.SPLIT_NAME or freeze.get("truth_accessed") is not False:
        raise ValueError("Freeze manifest is not a truth-blind fresh_heldout gate")
    if freeze.get("target") != sealed.get("target"):
        raise ValueError("Freeze and sealed targets differ")

    prepared_path = _safe_child(
        frozen_dir,
        str(freeze["prepared_manifest"]["path"]),
        allowed_root=namespace_root,
    )
    prepared_sha256 = freezer._sha256(prepared_path)
    if prepared_sha256 != str(freeze["prepared_manifest"]["sha256"]):
        raise ValueError("Prepared manifest hash drift against freeze")
    if prepared_sha256 != str(sealed["prepared_manifest_sha256"]):
        raise ValueError("Prepared manifest hash drift against sealed manifest")
    prepared = _read_json(prepared_path)
    if prepared.get("kind") != evaluation_spec.gate.prepare_kind:
        raise ValueError(f"Unexpected prepared manifest kind: {prepared.get('kind')}")
    if prepared.get("target") != freeze.get("target"):
        raise ValueError("Prepared and frozen targets differ")
    freezer._verify_prepared_manifest(
        namespace_root,
        prepared_path,
        prepared,
        expected_kind=evaluation_spec.gate.prepare_kind,
    )
    if str(freeze.get("selection_sha256")) != str(prepared["artifacts"]["selection"]["sha256"]):
        raise ValueError("Frozen selection hash differs from prepared manifest")

    requests_record = freeze["requests"]
    requests_path = _safe_child(namespace_root, str(requests_record["path"]))
    if freezer._sha256(requests_path) != str(requests_record["sha256"]):
        raise ValueError("Frozen request hash drift")
    request_rows = _read_jsonl(requests_path)
    expected_row_hashes = tuple(str(value) for value in requests_record.get("row_sha256") or [])
    actual_row_hashes = tuple(freezer._hash_json(row) for row in request_rows)
    if actual_row_hashes != expected_row_hashes:
        raise ValueError("Frozen request row hash drift")

    for role in ("predictions", "model_artifacts", "training_artifacts"):
        records = freeze[role] if isinstance(freeze[role], list) else [freeze[role]]
        if not records:
            raise ValueError(f"Frozen {role} snapshots are missing")
        for record in records:
            snapshot = _safe_child(
                namespace_root,
                str(record["snapshot_path_relative_to_namespace"]),
            )
            if freezer._sha256(snapshot) != str(record["snapshot_sha256"]):
                raise ValueError(f"Frozen {role} snapshot hash drift: {snapshot}")

    predictions_record = freeze["predictions"]
    predictions_path = _safe_child(
        namespace_root,
        str(predictions_record["snapshot_path_relative_to_namespace"]),
    )
    prediction_rows = _read_jsonl(predictions_path)
    requests_by_crop = _rows_by_crop(request_rows, label="requests")
    predictions_by_crop = _rows_by_crop(prediction_rows, label="predictions")
    if set(requests_by_crop) != set(predictions_by_crop):
        raise ValueError("Frozen request and prediction crop identities differ")

    return {
        "namespace_root": namespace_root,
        "sealed_sha256": freezer._sha256(sealed_manifest_path),
        "sealed": sealed,
        "freeze_path": freeze_path,
        "freeze_sha256": freeze_sha256,
        "freeze": freeze,
        "prepared_path": prepared_path,
        "prepared_sha256": prepared_sha256,
        "target": dict(freeze["target"]),
        "requests_by_crop": requests_by_crop,
        "predictions_by_crop": predictions_by_crop,
        "evaluation_spec": evaluation_spec,
    }


def load_visible_musicxml_truth(path: Path) -> VisibleMusicXMLTruth:
    """Load each visible pitched head, including tied continuation heads."""
    canonical = musicxml_utils.parse_musicxml_events(path)
    root = ET.parse(path).getroot()
    _strip_namespaces(root)
    parts = root.findall("part")
    if len(parts) != 1:
        raise ValueError(f"Expected exactly one MusicXML part, found {len(parts)}")

    key_fifths: int | None = None
    clef: tuple[str, int] | None = None
    measure_numbers: list[int] = []
    notes_by_measure: dict[int, tuple[dict[str, Any], ...]] = {}
    for fallback, measure in enumerate(parts[0].findall("measure"), start=1):
        number = _measure_number(measure, fallback)
        if number in notes_by_measure:
            raise ValueError(f"Duplicate physical MusicXML measure number: {number}")
        measure_numbers.append(number)
        cursor = 0
        last_note_onset = 0
        raw_notes: list[dict[str, Any]] = []
        serial = 0
        for child in measure:
            if child.tag == "attributes":
                fifths_text = child.findtext("key/fifths")
                if fifths_text is not None:
                    key_fifths = int(fifths_text)
                sign = child.findtext("clef/sign")
                line = child.findtext("clef/line")
                if sign is not None and line is not None:
                    clef = (sign.strip(), int(line))
                continue
            if child.tag == "backup":
                cursor -= int(child.findtext("duration", "0"))
                continue
            if child.tag == "forward":
                cursor += int(child.findtext("duration", "0"))
                continue
            if child.tag != "note":
                continue

            duration = int(child.findtext("duration", "0"))
            is_chord_tone = child.find("chord") is not None
            onset = last_note_onset if is_chord_tone else cursor
            if child.find("rest") is None:
                pitch = child.find("pitch")
                if pitch is None:
                    raise ValueError("Visible non-rest MusicXML note is missing pitch")
                alter = int(pitch.findtext("alter", "0"))
                tie_types = set(musicxml_utils._iter_tie_types(child))
                raw_notes.append(
                    {
                        "xml_order": serial,
                        "onset_divisions": onset,
                        "duration_divisions": duration,
                        "pitch_midi": musicxml_utils._pitch_to_midi(pitch),
                        "pitch": _pitch_name(pitch, alter),
                        "sounding_alter": alter,
                        "display_accidental": child.findtext("accidental"),
                        "tie_start": "start" in tie_types,
                        "tie_stop": "stop" in tie_types,
                    }
                )
                serial += 1
            if not is_chord_tone:
                cursor += duration
                last_note_onset = onset

        raw_notes.sort(key=lambda row: (int(row["onset_divisions"]), int(row["xml_order"])))
        notes_by_measure[number] = tuple(
            {
                **note,
                "physical_measure_number": number,
                "physical_note_index": index,
            }
            for index, note in enumerate(raw_notes)
        )

    return VisibleMusicXMLTruth(
        measure_numbers=tuple(measure_numbers),
        notes_by_measure=notes_by_measure,
        time_signature=str(canonical["time_signature"]),
        key_fifths=key_fifths,
        clef=clef,
    )


def validate_and_materialize_mapping(
    payload: Mapping[str, Any],
    *,
    truth: VisibleMusicXMLTruth,
    crop_indices: Sequence[int],
    mode: str,
) -> dict[str, Any]:
    entries = payload.get("automatic_crops")
    if not isinstance(entries, list):
        raise ValueError("Mapping must contain an automatic_crops list")
    entries_by_crop: dict[int, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("Each automatic_crops entry must be an object")
        crop = int(entry["automatic_crop_index"])
        if crop in entries_by_crop:
            raise ValueError(f"Duplicate automatic crop in mapping: {crop}")
        entries_by_crop[crop] = entry
    if set(entries_by_crop) != set(crop_indices):
        raise ValueError(
            "Mapping crop identities differ from frozen predictions: "
            f"mapping={sorted(entries_by_crop)}, frozen={sorted(crop_indices)}"
        )

    physical_set = set(truth.measure_numbers)
    materialized_entries: list[dict[str, Any]] = []
    assigned_note_keys: list[tuple[int, int]] = []
    covered_measures: set[int] = set()
    for crop in sorted(entries_by_crop):
        spans = entries_by_crop[crop].get("physical_measure_spans")
        if not isinstance(spans, list) or not spans:
            raise ValueError(f"Automatic crop {crop} must contain physical_measure_spans")
        materialized_spans = []
        for span in spans:
            if not isinstance(span, Mapping):
                raise ValueError(f"Automatic crop {crop} contains a non-object span")
            measure = int(span["measure_number"])
            if measure not in physical_set:
                raise ValueError(f"Mapping references unknown physical measure {measure}")
            note_count = len(truth.notes_by_measure[measure])
            start = int(span.get("note_start", 0))
            raw_end = span.get("note_end")
            end = note_count if raw_end is None else int(raw_end)
            if start < 0 or end < start or end > note_count:
                raise ValueError(
                    f"Invalid note slice for physical measure {measure}: "
                    f"[{start}, {end}) with {note_count} visible notes"
                )
            covered_measures.add(measure)
            assigned_note_keys.extend((measure, index) for index in range(start, end))
            materialized_spans.append(
                {
                    "measure_number": measure,
                    "note_start": start,
                    "note_end": end,
                }
            )
        materialized_entries.append(
            {
                "automatic_crop_index": crop,
                "physical_measure_spans": materialized_spans,
            }
        )

    if covered_measures != physical_set:
        raise ValueError(
            "Mapping does not cover every physical MusicXML measure: "
            f"covered={sorted(covered_measures)}, musicxml={list(truth.measure_numbers)}"
        )
    canonical_note_keys = [
        (measure, index)
        for measure in truth.measure_numbers
        for index in range(len(truth.notes_by_measure[measure]))
    ]
    if assigned_note_keys != canonical_note_keys:
        raise ValueError(
            "Mapping note spans must cover every visible MusicXML note exactly once in order"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "automatic_crop_to_physical_measure_mapping",
        "mode": mode,
        "note_slice_semantics": "zero_based_half_open_visible_noteheads",
        "automatic_crops": materialized_entries,
    }


def build_truth_rows(
    requests_by_crop: Mapping[int, Mapping[str, Any]],
    truth: VisibleMusicXMLTruth,
    mapping: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for entry in mapping["automatic_crops"]:
        crop = int(entry["automatic_crop_index"])
        notes = []
        physical_measures = []
        for span in entry["physical_measure_spans"]:
            measure = int(span["measure_number"])
            physical_measures.append(measure)
            notes.extend(
                dict(note)
                for note in truth.notes_by_measure[measure][
                    int(span["note_start"]) : int(span["note_end"])
                ]
            )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "identity": dict(requests_by_crop[crop]["identity"]),
                "automatic_crop_index": crop,
                "physical_measure_numbers": physical_measures,
                "physical_measure_spans": list(entry["physical_measure_spans"]),
                "visible_notehead_count": len(notes),
                "notes": notes,
            }
        )
    return rows


def evaluate_pitch_only(
    truth_rows: Sequence[Mapping[str, Any]],
    predictions_by_crop: Mapping[int, Mapping[str, Any]],
    *,
    target: Mapping[str, Any],
    mapping_mode: str,
    report_kind: str = THIRD_SCORE_EVALUATION.report_kind,
) -> dict[str, Any]:
    crop_reports = []
    total_predicted = 0
    total_truth = 0
    count_matches = 0
    pitch_matches = 0
    substitutions = 0
    insertions = 0
    deletions = 0
    exact_crops = 0
    for truth_row in truth_rows:
        crop = int(truth_row["automatic_crop_index"])
        prediction = predictions_by_crop[crop]
        predicted_pitches = [int(note["pitch_midi"]) for note in prediction.get("notes") or []]
        truth_pitches = [int(note["pitch_midi"]) for note in truth_row.get("notes") or []]
        alignment = _align_pitches(predicted_pitches, truth_pitches)
        exact = predicted_pitches == truth_pitches
        pred_count = len(predicted_pitches)
        truth_count = len(truth_pitches)
        matched_count = min(pred_count, truth_count)
        total_predicted += pred_count
        total_truth += truth_count
        count_matches += matched_count
        pitch_matches += alignment["exact_pitch_matches"]
        substitutions += alignment["substitutions"]
        insertions += alignment["insertions"]
        deletions += alignment["deletions"]
        exact_crops += int(exact)
        crop_reports.append(
            {
                "automatic_crop_index": crop,
                "physical_measure_numbers": truth_row["physical_measure_numbers"],
                "predicted_note_count": pred_count,
                "truth_note_count": truth_count,
                "predicted_ordered_pitches": predicted_pitches,
                "truth_ordered_pitches": truth_pitches,
                "alignment": alignment,
                "exact_automatic_crop": exact,
            }
        )

    count_precision = _ratio(count_matches, total_predicted)
    count_recall = _ratio(count_matches, total_truth)
    count_f1 = _f1(count_precision, count_recall)
    alignment_total = pitch_matches + substitutions + insertions + deletions
    crop_count = len(crop_reports)
    metric_support = {
        "note_count_precision_recall_f1": "scored",
        "ordered_pitch_alignment": "scored",
        "exact_pitch_matches": "scored",
        "exact_automatic_crops": "scored",
        "onset": NOT_SCORED,
        "duration": NOT_SCORED,
        "rests": NOT_SCORED,
        "meter": NOT_SCORED,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": report_kind,
        "status": "evaluated_exactly_once_after_frozen_predictions",
        "target": dict(target),
        "mapping_mode": mapping_mode,
        "metric_support": metric_support,
        "metrics": {
            "summary": {
                "automatic_crop_count": crop_count,
                "predicted_note_count": total_predicted,
                "truth_note_count": total_truth,
                "note_count_matched_capacity": count_matches,
                "note_count_precision": count_precision,
                "note_count_recall": count_recall,
                "note_count_f1": count_f1,
                "exact_pitch_matches": pitch_matches,
                "ordered_pitch_substitutions": substitutions,
                "ordered_pitch_insertions": insertions,
                "ordered_pitch_deletions": deletions,
                "ordered_pitch_alignment_accuracy": _ratio(pitch_matches, alignment_total),
                "exact_automatic_crops": exact_crops,
                "exact_automatic_crop_rate": _ratio(exact_crops, crop_count),
            },
            "crops": crop_reports,
            "unsupported": {
                name: {"status": status}
                for name, status in metric_support.items()
                if status == NOT_SCORED
            },
        },
    }


def _align_pitches(predicted: Sequence[int], truth: Sequence[int]) -> dict[str, Any]:
    rows = len(predicted) + 1
    columns = len(truth) + 1
    costs = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        costs[row][0] = row
    for column in range(columns):
        costs[0][column] = column
    for row in range(1, rows):
        for column in range(1, columns):
            substitution = 0 if predicted[row - 1] == truth[column - 1] else 1
            costs[row][column] = min(
                costs[row - 1][column - 1] + substitution,
                costs[row - 1][column] + 1,
                costs[row][column - 1] + 1,
            )

    row = len(predicted)
    column = len(truth)
    operations = []
    while row or column:
        if row and column and predicted[row - 1] == truth[column - 1]:
            if costs[row][column] == costs[row - 1][column - 1]:
                operations.append(
                    {
                        "operation": "exact",
                        "predicted": predicted[row - 1],
                        "truth": truth[column - 1],
                    }
                )
                row -= 1
                column -= 1
                continue
        if row and column and costs[row][column] == costs[row - 1][column - 1] + 1:
            operations.append(
                {
                    "operation": "substitution",
                    "predicted": predicted[row - 1],
                    "truth": truth[column - 1],
                }
            )
            row -= 1
            column -= 1
            continue
        if row and costs[row][column] == costs[row - 1][column] + 1:
            operations.append(
                {"operation": "insertion", "predicted": predicted[row - 1], "truth": None}
            )
            row -= 1
            continue
        operations.append({"operation": "deletion", "predicted": None, "truth": truth[column - 1]})
        column -= 1
    operations.reverse()
    return {
        "edit_distance": costs[-1][-1],
        "exact_pitch_matches": sum(row["operation"] == "exact" for row in operations),
        "substitutions": sum(row["operation"] == "substitution" for row in operations),
        "insertions": sum(row["operation"] == "insertion" for row in operations),
        "deletions": sum(row["operation"] == "deletion" for row in operations),
        "operations": operations,
    }


def _default_mapping(
    measure_numbers: Sequence[int],
    crop_indices: Sequence[int],
    *,
    expected_count: int = THIRD_SCORE_EVALUATION.default_one_to_one_count,
) -> dict[str, Any]:
    if len(measure_numbers) != expected_count or len(crop_indices) != expected_count:
        raise ValueError(
            "Automatic default mapping is available only when MusicXML measures and "
            f"frozen crops both contain the configured count ({expected_count}); "
            "provide --mapping"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "automatic_crops": [
            {
                "automatic_crop_index": crop,
                "physical_measure_spans": [{"measure_number": measure}],
            }
            for crop, measure in zip(crop_indices, measure_numbers, strict=True)
        ],
    }


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if int(payload.get("schema_version", SCHEMA_VERSION)) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported mapping schema version: {payload.get('schema_version')}")
    return payload


def _rows_by_crop(rows: Iterable[Mapping[str, Any]], *, label: str) -> dict[int, Mapping[str, Any]]:
    result = {}
    for row in rows:
        identity = row.get("identity") or {}
        raw_crop = identity.get("automatic_measure_index", identity.get("system_measure_index"))
        if raw_crop is None:
            raise ValueError(f"Frozen {label} row is missing an automatic crop identity")
        crop = int(raw_crop)
        if crop in result:
            raise ValueError(f"Duplicate frozen {label} crop identity: {crop}")
        result[crop] = row
    if not result:
        raise ValueError(f"Frozen {label} rows are empty")
    return result


def _snapshot_record(
    snapshot: Path,
    *,
    source_path: Path | None = None,
    source_sha256: str | None = None,
    require_source_match: bool = True,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "snapshot_path": snapshot.name,
        "snapshot_sha256": freezer._sha256(snapshot),
    }
    if source_path is not None:
        record["source_path"] = freezer._repo_display_path(source_path)
        record["source_sha256"] = source_sha256
        if require_source_match and source_sha256 != record["snapshot_sha256"]:
            raise ValueError(f"Snapshot hash mismatch for {source_path}")
    else:
        record["source_path"] = None
        record["source_sha256"] = None
    return record


def _safe_child(
    parent: Path,
    relative_path: str,
    *,
    allowed_root: Path | None = None,
) -> Path:
    path = (parent / relative_path).resolve()
    allowed_root = allowed_root or parent
    try:
        path.relative_to(allowed_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Manifest path escapes its namespace: {relative_path}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Pinned artifact does not exist: {path}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        rows.append(payload)
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _strip_namespaces(root: ET.Element) -> None:
    for element in root.iter():
        if isinstance(element.tag, str) and "}" in element.tag:
            element.tag = element.tag.split("}", 1)[1]


def _measure_number(measure: ET.Element, fallback: int) -> int:
    try:
        return int(measure.get("number", str(fallback)))
    except ValueError:
        return fallback


def _pitch_name(pitch: ET.Element, alter: int) -> str:
    step = (pitch.findtext("step") or "").strip()
    octave = (pitch.findtext("octave") or "").strip()
    accidental = {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}.get(alter, f"({alter:+d})")
    return f"{step}{accidental}{octave}"


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return round(numerator / denominator, 6)


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 6)


def _validate_version(value: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError(f"Invalid evaluation version: {value!r}")


if __name__ == "__main__":
    raise SystemExit(main())
