"""Evaluate the future independent dyad-recovery gate exactly once.

The evaluator is deliberately separate from the future freezer and runner. It
verifies the create-once fresh-heldout seal, every prepared artifact, every
frozen snapshot, and the paired baseline/recovered prediction contract before
opening user-supplied MusicXML or mapping data.

Expected transcription path:
  out/local_restricted/jaime-llanos_73_no-lo-creas_pasillo_a-vasquez-pedrero/
  vlm_melody_independent_dyad_recovery_gate/v1/system_008/
  no_lo_creas_system_008.musicxml
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import evaluate_frozen_third_score_heldout as heldout  # noqa: E402
from scripts.experiments import freeze_independent_dyad_recovery_gate as dyad_gate  # noqa: E402
from scripts.experiments import freeze_third_score_heldout as freezer  # noqa: E402
from scripts.experiments import run_independent_dyad_recovery_gate as dyad_runner  # noqa: E402

SCHEMA_VERSION = 1
DEFAULT_EVALUATION_VERSION = "v1"
TARGET_SLUG = "jaime-llanos_73_no-lo-creas_pasillo_a-vasquez-pedrero"
TARGET_SYSTEM_INDEX = 8
OUTPUT_SUBDIR = "vlm_melody_independent_dyad_recovery_gate"
DEFAULT_NAMESPACE = "v1"
EXPECTED_TRANSCRIPTION_FILENAME = "no_lo_creas_system_008.musicxml"
EXPECTED_TRANSCRIPTION_PATH = (
    Path("out/local_restricted")
    / TARGET_SLUG
    / OUTPUT_SUBDIR
    / DEFAULT_NAMESPACE
    / f"system_{TARGET_SYSTEM_INDEX:03d}"
    / EXPECTED_TRANSCRIPTION_FILENAME
)

GATE_SPEC = dyad_gate.INDEPENDENT_DYAD_RECOVERY_GATE
LANE_BASELINE = "baseline_generic"
LANE_RECOVERED = "edge_safe_recovery"
LANES = (LANE_BASELINE, LANE_RECOVERED)
NOT_SCORED_LOCALIZATION_GATE = "not_scored_localization_focused_gate"
NOT_SCORED_AMBIGUOUS_MAPPING = "not_scored_mapping_splits_onset_group"
NOT_SCORED_MISSING_GROUPS = "not_scored_missing_frozen_onset_group_ids"

TruthLoader = Callable[[Path], heldout.VisibleMusicXMLTruth]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Expected local transcription path:\n"
            f"  {EXPECTED_TRANSCRIPTION_PATH.as_posix()}\n\n"
            "When automatic crops do not map one-to-one to physical measures, pass "
            "--mapping with automatic_crops/physical_measure_spans entries."
        ),
    )
    parser.add_argument("sealed_manifest", type=Path, help="Future gate sealed_manifest.json")
    parser.add_argument(
        "--musicxml",
        type=Path,
        required=True,
        help=(
            "User transcription MusicXML. Expected location: "
            f"{EXPECTED_TRANSCRIPTION_PATH.as_posix()}"
        ),
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help=(
            "Optional crop-to-physical-measure mapping JSON. Required when frozen crop "
            "count and physical MusicXML measure count differ or a crop boundary splits a measure."
        ),
    )
    parser.add_argument(
        "--evaluation-version",
        default=DEFAULT_EVALUATION_VERSION,
        help="Create-once output version (default: v1, written as evaluation_v1).",
    )
    args = parser.parse_args(argv)
    try:
        result = evaluate_independent_dyad_recovery_gate(
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


def evaluate_independent_dyad_recovery_gate(
    sealed_manifest_path: Path,
    *,
    musicxml_path: Path,
    mapping_path: Path | None = None,
    evaluation_version: str = DEFAULT_EVALUATION_VERSION,
    truth_loader: TruthLoader = heldout.load_visible_musicxml_truth,
) -> dict[str, str]:
    """Verify the blind paired gate, then materialize and score truth exactly once."""
    heldout._validate_version(evaluation_version)
    sealed_manifest_path = sealed_manifest_path.expanduser().resolve()
    musicxml_path = musicxml_path.expanduser().resolve()
    mapping_path = mapping_path.expanduser().resolve() if mapping_path is not None else None

    # This must remain the first operation that opens any caller-supplied input.
    frozen = verify_frozen_dyad_gate(sealed_manifest_path)
    namespace_root = frozen["namespace_root"]
    output_dir = namespace_root / f"evaluation_{evaluation_version}"
    temp_dir = namespace_root / f".evaluation_{evaluation_version}.tmp"
    prior_evaluations = sorted(namespace_root.glob("evaluation_*"))
    if prior_evaluations:
        raise FileExistsError(
            "Independent dyad evaluation already exists: "
            + ", ".join(str(path) for path in prior_evaluations)
        )
    stale_temps = sorted(namespace_root.glob(".evaluation_*.tmp"))
    if stale_temps:
        raise FileExistsError(
            "Stale independent dyad evaluation directory exists: "
            + ", ".join(str(path) for path in stale_temps)
        )

    # Only now may truth or post-freeze mapping data be opened.
    if not musicxml_path.is_file():
        raise FileNotFoundError(f"User MusicXML does not exist: {musicxml_path}")
    if mapping_path is not None and not mapping_path.is_file():
        raise FileNotFoundError(f"Mapping JSON does not exist: {mapping_path}")
    musicxml_sha256 = freezer._sha256(musicxml_path)
    mapping_sha256 = freezer._sha256(mapping_path) if mapping_path is not None else None
    truth = truth_loader(musicxml_path)

    crop_indices = tuple(sorted(frozen["paired_rows_by_crop"]))
    if mapping_path is None:
        mapping_payload = _default_mapping(truth.measure_numbers, crop_indices)
        mapping_mode = "deterministic_equal_count_one_to_one"
    else:
        mapping_payload = heldout._load_mapping(mapping_path)
        mapping_mode = "explicit_user_mapping"
    mapping = heldout.validate_and_materialize_mapping(
        mapping_payload,
        truth=truth,
        crop_indices=crop_indices,
        mode=mapping_mode,
    )
    truth_rows = heldout.build_truth_rows(frozen["requests_by_crop"], truth, mapping)
    structure_support = _mapping_structure_support(mapping, truth)
    lane_reports = {
        lane: _score_lane(
            truth_rows,
            frozen["paired_rows_by_crop"],
            lane=lane,
            structure_support=structure_support,
        )
        for lane in LANES
    }
    report = _build_report(
        target=frozen["target"],
        mapping_mode=mapping_mode,
        truth=truth,
        lane_reports=lane_reports,
        structure_support=structure_support,
    )

    temp_dir.mkdir(parents=False, exist_ok=False)
    try:
        source_snapshot = temp_dir / "source.musicxml"
        mapping_snapshot = temp_dir / "mapping.json"
        truth_snapshot = temp_dir / "truth.jsonl"
        paired_snapshot = temp_dir / "frozen_paired_predictions.jsonl"
        freeze_snapshot = temp_dir / "frozen_freeze.json"
        sealed_snapshot = temp_dir / "frozen_sealed_manifest.json"
        evaluator_snapshot = temp_dir / "evaluator.py"
        report_path = temp_dir / "report.json"

        shutil.copyfile(musicxml_path, source_snapshot)
        heldout._write_json(mapping_snapshot, mapping)
        heldout._write_jsonl(truth_snapshot, truth_rows)
        shutil.copyfile(frozen["predictions_path"], paired_snapshot)
        shutil.copyfile(frozen["freeze_path"], freeze_snapshot)
        shutil.copyfile(sealed_manifest_path, sealed_snapshot)
        shutil.copyfile(Path(__file__).resolve(), evaluator_snapshot)

        pins = {
            "source_musicxml": heldout._snapshot_record(
                source_snapshot,
                source_path=musicxml_path,
                source_sha256=musicxml_sha256,
            ),
            "mapping": heldout._snapshot_record(
                mapping_snapshot,
                source_path=mapping_path,
                source_sha256=mapping_sha256,
                require_source_match=False,
            ),
            "truth": heldout._snapshot_record(truth_snapshot),
            "paired_predictions": heldout._snapshot_record(
                paired_snapshot,
                source_path=frozen["predictions_path"],
                source_sha256=frozen["predictions_sha256"],
            ),
            "freeze_manifest": heldout._snapshot_record(
                freeze_snapshot,
                source_path=frozen["freeze_path"],
                source_sha256=frozen["freeze_sha256"],
            ),
            "sealed_manifest": heldout._snapshot_record(
                sealed_snapshot,
                source_path=sealed_manifest_path,
                source_sha256=frozen["sealed_sha256"],
            ),
            "evaluator": heldout._snapshot_record(
                evaluator_snapshot,
                source_path=Path(__file__).resolve(),
                source_sha256=freezer._sha256(Path(__file__).resolve()),
            ),
        }
        report["pins"] = pins
        heldout._write_json(report_path, report)
        pins["report"] = heldout._snapshot_record(report_path)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "independent_dyad_recovery_post_freeze_evaluation_manifest",
            "status": "evaluated_exactly_once_after_frozen_predictions",
            "create_once": True,
            "evaluation_version": evaluation_version,
            "target": frozen["target"],
            "truth_opened_after_all_frozen_hashes_verified": True,
            "mapping_mode": mapping_mode,
            "pins": pins,
        }
        heldout._write_json(temp_dir / "manifest.json", manifest)

        verified_again = verify_frozen_dyad_gate(sealed_manifest_path)
        if (
            verified_again["freeze_sha256"] != frozen["freeze_sha256"]
            or verified_again["sealed_sha256"] != frozen["sealed_sha256"]
        ):
            raise ValueError("Frozen independent dyad gate changed during evaluation")
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


def verify_frozen_dyad_gate(sealed_manifest_path: Path) -> dict[str, Any]:
    """Verify every create-once artifact without opening transcription truth."""
    if not sealed_manifest_path.is_file():
        raise FileNotFoundError(f"Sealed manifest does not exist: {sealed_manifest_path}")
    sealed = heldout._read_json(sealed_manifest_path)
    expected_target = {"slug": TARGET_SLUG, "system_index": TARGET_SYSTEM_INDEX}
    if sealed.get("kind") != "independent_dyad_recovery_paired_sealed_manifest":
        raise ValueError(f"Unexpected independent dyad sealed kind: {sealed.get('kind')}")
    if sealed.get("status") != "frozen_awaiting_truth":
        raise ValueError("Independent dyad seal is not awaiting truth")
    if sealed.get("truth_accessed") is not False or sealed.get("truth_used") is not False:
        raise ValueError("Independent dyad seal is not truth-blind")
    if sealed.get("target") != expected_target:
        raise ValueError(f"Unexpected independent dyad target: {sealed.get('target')}")

    frozen_dir = sealed_manifest_path.parent.resolve()
    namespace_root = frozen_dir.parent.resolve()
    freeze_path = heldout._safe_child(frozen_dir, str(sealed["freeze"]["path"]))
    if freeze_path != frozen_dir / "freeze.json":
        raise ValueError(f"Unexpected paired dyad freeze path: {freeze_path}")
    freeze_sha256 = freezer._sha256(freeze_path)
    if freeze_sha256 != str(sealed["freeze"]["sha256"]):
        raise ValueError("Independent dyad freeze manifest hash drift")
    freeze = heldout._read_json(freeze_path)
    if freeze.get("kind") != "independent_dyad_recovery_paired_freeze":
        raise ValueError(f"Unexpected independent dyad freeze kind: {freeze.get('kind')}")
    if freeze.get("status") != "frozen_awaiting_truth":
        raise ValueError("Independent dyad freeze is not awaiting truth")
    if freeze.get("truth_accessed") is not False or freeze.get("truth_used") is not False:
        raise ValueError("Independent dyad freeze is not truth-blind")
    if freeze.get("target") != expected_target:
        raise ValueError("Independent dyad frozen target mismatch")

    # The runner owns the paired-freeze format and verifies every role pin.
    # Keep this call before any truth path opens.
    dyad_runner.verify_paired_freeze(frozen_dir)
    pins = freeze.get("pins")
    if not isinstance(pins, Mapping):
        raise ValueError("Independent dyad paired freeze has no role-based pins")

    predictions_pin = _only_pin(pins, "paired_predictions")
    predictions_path = _pin_snapshot(namespace_root, predictions_pin)
    prepared_pin = _find_pin(
        pins,
        "prepared_and_source",
        filename="prepared_manifest.json",
        expected_sha256=str(freeze["prepared_manifest_sha256"]),
    )
    prepared_path = _pin_snapshot(namespace_root, prepared_pin)
    prepared = heldout._read_json(prepared_path)
    _validate_prepared_snapshot(prepared, expected_target=expected_target)

    prepared_source = _pin_source_path(prepared_pin)
    requests_record = prepared["artifacts"]["requests"]
    requests_source = prepared_source.parent / str(requests_record["path"])
    requests_pin = _find_pin_by_source(pins, "prepared_and_source", requests_source)
    if str(requests_pin["snapshot_sha256"]) != str(requests_record["sha256"]):
        raise ValueError("Independent dyad prepared request pin hash mismatch")
    requests_path = _pin_snapshot(namespace_root, requests_pin)
    request_rows = heldout._read_jsonl(requests_path)
    expected_row_hashes = tuple(str(value) for value in requests_record.get("row_sha256") or [])
    actual_row_hashes = tuple(freezer._hash_json(row) for row in request_rows)
    if actual_row_hashes != expected_row_hashes:
        raise ValueError("Independent dyad request row hash drift")
    requests_by_crop = heldout._rows_by_crop(request_rows, label="requests")
    _validate_row_targets(request_rows, label="request")

    paired_rows = heldout._read_jsonl(predictions_path)
    paired_rows_by_crop = heldout._rows_by_crop(paired_rows, label="paired predictions")
    if set(requests_by_crop) != set(paired_rows_by_crop):
        raise ValueError("Independent dyad request/prediction crop identities differ")
    _validate_row_targets(paired_rows, label="paired prediction")
    _validate_paired_prediction_contract(paired_rows)

    _verify_role_hash_binding(
        pins,
        "paired_artifacts",
        filename="manifest.json",
        expected_sha256=str(freeze["paired_manifest_sha256"]),
    )
    invariance_pin = _verify_role_hash_binding(
        pins,
        "paired_artifacts",
        filename="additive_invariance.json",
        expected_sha256=str(freeze["additive_invariance_sha256"]),
    )
    invariance = heldout._read_json(_pin_snapshot(namespace_root, invariance_pin))
    if (
        invariance.get("passed") is not True
        or invariance.get("truth_accessed") is not False
        or invariance.get("truth_used") is not False
    ):
        raise ValueError("Independent dyad additive-invariance pin is not passed/truth-blind")
    _verify_role_hash_binding(
        pins,
        "baseline_inference",
        filename="predictions.jsonl",
        expected_sha256=str(freeze["generic_baseline_predictions_sha256"]),
    )
    source_system = prepared["artifacts"]["source_system"]
    _verify_role_hash_binding(
        pins,
        "prepared_and_source",
        filename=Path(str(source_system["path_relative_to_out"])).name,
        expected_sha256=str(freeze["source_system_sha256"]),
    )

    return {
        "namespace_root": namespace_root,
        "sealed_sha256": freezer._sha256(sealed_manifest_path),
        "freeze_path": freeze_path,
        "freeze_sha256": freeze_sha256,
        "prepared_path": prepared_path,
        "prepared_sha256": str(prepared_pin["snapshot_sha256"]),
        "predictions_path": predictions_path,
        "predictions_sha256": str(predictions_pin["snapshot_sha256"]),
        "target": expected_target,
        "requests_by_crop": requests_by_crop,
        "paired_rows_by_crop": paired_rows_by_crop,
    }


def _validate_prepared_snapshot(
    prepared: Mapping[str, Any], *, expected_target: Mapping[str, Any]
) -> None:
    if prepared.get("kind") != GATE_SPEC.prepare_kind:
        raise ValueError("Independent dyad pinned prepared-manifest kind mismatch")
    if prepared.get("status") != "prepared_awaiting_model_predictions":
        raise ValueError("Independent dyad pinned preparation is not awaiting predictions")
    if prepared.get("split") != freezer.SPLIT_NAME or prepared.get("truth_accessed") is not False:
        raise ValueError("Independent dyad pinned preparation is not truth-blind")
    if prepared.get("target") != expected_target:
        raise ValueError("Independent dyad pinned prepared target mismatch")
    config = prepared.get("independent_dyad_recovery_gate")
    if not isinstance(config, Mapping):
        raise ValueError("Independent dyad pinned preparation has no recovery contract")
    if config.get("truth_accessed") is not False or config.get("truth_used") is not False:
        raise ValueError("Independent dyad pinned recovery contract used truth")


def _only_pin(pins: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    records = pins.get(role)
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], Mapping):
        raise ValueError(f"Independent dyad freeze requires one {role} pin")
    return records[0]


def _find_pin(
    pins: Mapping[str, Any],
    role: str,
    *,
    filename: str,
    expected_sha256: str,
) -> Mapping[str, Any]:
    records = pins.get(role)
    if not isinstance(records, list):
        raise ValueError(f"Independent dyad freeze has no {role} role")
    matches = [
        pin
        for pin in records
        if isinstance(pin, Mapping)
        and Path(str(pin.get("source_path", ""))).name == filename
        and str(pin.get("snapshot_sha256")) == expected_sha256
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Independent dyad freeze cannot uniquely locate {role}/{filename} at pinned hash"
        )
    return matches[0]


def _find_pin_by_source(
    pins: Mapping[str, Any], role: str, expected_source: Path
) -> Mapping[str, Any]:
    records = pins.get(role)
    if not isinstance(records, list):
        raise ValueError(f"Independent dyad freeze has no {role} role")
    expected = expected_source.expanduser().resolve()
    matches = [
        pin for pin in records if isinstance(pin, Mapping) and _pin_source_path(pin) == expected
    ]
    if len(matches) != 1:
        raise ValueError(f"Independent dyad freeze cannot locate pinned source: {expected}")
    return matches[0]


def _pin_source_path(pin: Mapping[str, Any]) -> Path:
    source = Path(str(pin["source_path"])).expanduser()
    if not source.is_absolute():
        source = REPO_ROOT / source
    return source.resolve()


def _pin_snapshot(namespace_root: Path, pin: Mapping[str, Any]) -> Path:
    return heldout._safe_child(
        namespace_root,
        str(pin["snapshot_path_relative_to_namespace"]),
    )


def _verify_role_hash_binding(
    pins: Mapping[str, Any],
    role: str,
    *,
    filename: str,
    expected_sha256: str,
) -> Mapping[str, Any]:
    return _find_pin(
        pins,
        role,
        filename=filename,
        expected_sha256=expected_sha256,
    )


def _validate_row_targets(rows: Sequence[Mapping[str, Any]], *, label: str) -> None:
    for row in rows:
        identity = row.get("identity")
        if not isinstance(identity, Mapping):
            raise ValueError(f"Independent dyad {label} has no identity")
        if (
            identity.get("slug") != TARGET_SLUG
            or int(identity.get("system_index", -1)) != TARGET_SYSTEM_INDEX
        ):
            raise ValueError(f"Independent dyad {label} target identity mismatch")
        if row.get("truth_accessed") is not False:
            raise ValueError(f"Independent dyad {label} is not truth-blind")


def _validate_paired_prediction_contract(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        if row.get("truth_used") is not False:
            raise ValueError("Independent dyad paired prediction used truth")
        lanes = row.get("lanes")
        if not isinstance(lanes, Mapping) or set(lanes) != set(LANES):
            raise ValueError(f"Independent dyad predictions require exactly these lanes: {LANES}")
        baseline_lane = lanes[LANE_BASELINE]
        recovered_lane = lanes[LANE_RECOVERED]
        baseline_notes = _lane_notes(baseline_lane, lane=LANE_BASELINE)
        recovered_notes = _lane_notes(recovered_lane, lane=LANE_RECOVERED)
        canonical = baseline_lane.get("canonical_prediction")
        if not isinstance(canonical, Mapping) or not isinstance(canonical.get("notes"), list):
            raise ValueError("Baseline lane has no canonical prediction notes")
        if baseline_lane.get("canonical_prediction_sha256") != dyad_runner.inference._hash_json(
            canonical
        ):
            raise ValueError("Baseline canonical prediction hash drift")
        baseline_by_id = _notes_by_candidate_id(baseline_notes, lane=LANE_BASELINE)
        recovered_by_id = _notes_by_candidate_id(recovered_notes, lane=LANE_RECOVERED)
        canonical_by_id = _notes_by_candidate_id(
            canonical["notes"], lane="baseline canonical prediction"
        )
        if set(canonical_by_id) != set(baseline_by_id):
            raise ValueError("Baseline note identities differ from canonical prediction")
        for candidate_id, canonical_note in canonical_by_id.items():
            baseline_note = baseline_by_id[candidate_id]
            if _identity_coordinate_signature(canonical_note) != _identity_coordinate_signature(
                baseline_note
            ):
                raise ValueError(
                    f"Baseline note identity/coordinates drifted from canonical prediction: "
                    f"{candidate_id}"
                )
            if baseline_note.get("recovered") is not False:
                raise ValueError(f"Baseline note is not marked recovered=false: {candidate_id}")
            if _note_group_identity(baseline_note) is None:
                raise ValueError(f"Baseline note has no frozen onset group: {candidate_id}")
        if not set(baseline_by_id).issubset(recovered_by_id):
            raise ValueError("Recovered lane removed a baseline candidate")
        for candidate_id, baseline in baseline_by_id.items():
            recovered = recovered_by_id[candidate_id]
            if _localization_signature(baseline) != _localization_signature(recovered):
                raise ValueError(f"Recovered lane changed baseline localization: {candidate_id}")
        recovered_ids = recovered_lane.get("recovered_candidate_ids")
        if not isinstance(recovered_ids, list) or len(recovered_ids) != len(set(recovered_ids)):
            raise ValueError("Recovered lane has invalid recovered_candidate_ids")
        additional_ids = set(recovered_by_id) - set(baseline_by_id)
        if additional_ids != set(str(value) for value in recovered_ids):
            raise ValueError("Recovered pitch lane additions differ from recovered_candidate_ids")
        if int(recovered_lane.get("recovered_head_count", -1)) != len(recovered_ids):
            raise ValueError("Recovered head count differs from recovered_candidate_ids")
        baseline_group_ids = {_note_group_identity(note) for note in baseline_notes}
        if None in baseline_group_ids:
            raise ValueError("Baseline lane has an incomplete onset-group contract")
        if {_note_group_identity(note) for note in recovered_notes} != baseline_group_ids:
            raise ValueError("Recovered lane changed the frozen onset-group identities")
        for candidate_id in additional_ids:
            if recovered_by_id[candidate_id].get("recovered") is not True:
                raise ValueError(f"Recovered note is not marked recovered=true: {candidate_id}")

        baseline_candidates = _candidate_lane(baseline_lane, lane=LANE_BASELINE)
        recovered_candidates = _candidate_lane(recovered_lane, lane=LANE_RECOVERED)
        baseline_candidate_by_id = _notes_by_candidate_id(
            baseline_candidates, lane=f"{LANE_BASELINE} candidates"
        )
        recovered_candidate_by_id = _notes_by_candidate_id(
            recovered_candidates, lane=f"{LANE_RECOVERED} candidates"
        )
        if set(baseline_candidate_by_id) != set(baseline_by_id):
            raise ValueError("Baseline candidate and pitch lanes differ")
        if set(recovered_candidate_by_id) != set(recovered_by_id):
            raise ValueError("Recovered candidate and pitch lanes differ")
        if not set(baseline_candidate_by_id).issubset(recovered_candidate_by_id):
            raise ValueError("Recovered candidate lane removed a baseline candidate")
        for candidate_id, baseline in baseline_candidate_by_id.items():
            if recovered_candidate_by_id[candidate_id] != baseline:
                raise ValueError(f"Recovered lane changed baseline candidate: {candidate_id}")
        if set(recovered_candidate_by_id) - set(baseline_candidate_by_id) != additional_ids:
            raise ValueError("Recovered candidate and pitch additions differ")
        for candidate_id in additional_ids:
            if recovered_candidate_by_id[candidate_id].get("recovered") is not True:
                raise ValueError(
                    f"Recovered candidate is not marked recovered=true: {candidate_id}"
                )
        for candidate_id, note in recovered_by_id.items():
            candidate = recovered_candidate_by_id.get(candidate_id)
            if candidate is None or _note_center(note) != _note_center(candidate):
                raise ValueError(f"Pitch/candidate localization mismatch: {candidate_id}")
            if _note_group_identity(note) != _note_group_identity(candidate):
                raise ValueError(f"Pitch/candidate onset-group mismatch: {candidate_id}")


def _lane_notes(lane_payload: Any, *, lane: str) -> list[Mapping[str, Any]]:
    if not isinstance(lane_payload, Mapping):
        raise ValueError(f"Independent dyad {lane} lane is not an object")
    available = [
        name for name in ("notes", "pitch_lane") if isinstance(lane_payload.get(name), list)
    ]
    if len(available) != 1:
        raise ValueError(f"Independent dyad {lane} lane requires exactly one frozen note list")
    notes = lane_payload[available[0]]
    if any(not isinstance(note, Mapping) for note in notes):
        raise ValueError(f"Independent dyad {lane} lane contains a non-object note")
    return notes


def _candidate_lane(lane_payload: Any, *, lane: str) -> list[Mapping[str, Any]]:
    if not isinstance(lane_payload, Mapping) or not isinstance(
        lane_payload.get("candidate_lane"), list
    ):
        raise ValueError(f"Independent dyad {lane} lane has no candidate_lane list")
    candidates = lane_payload["candidate_lane"]
    if any(not isinstance(candidate, Mapping) for candidate in candidates):
        raise ValueError(f"Independent dyad {lane} candidate lane contains a non-object")
    return candidates


def _notes_by_candidate_id(
    notes: Sequence[Mapping[str, Any]], *, lane: str
) -> dict[str, Mapping[str, Any]]:
    result = {}
    for note in notes:
        candidate_id = str(note.get("candidate_id", ""))
        if not candidate_id:
            raise ValueError(f"Independent dyad {lane} note has no candidate_id")
        if candidate_id in result:
            raise ValueError(f"Duplicate {lane} candidate_id: {candidate_id}")
        result[candidate_id] = note
    return result


def _localization_signature(note: Mapping[str, Any]) -> tuple[Any, ...]:
    return (*_identity_coordinate_signature(note), _prediction_staff_position(note))


def _identity_coordinate_signature(note: Mapping[str, Any]) -> tuple[Any, ...]:
    x, y = _note_center(note)
    return (
        str(note["candidate_id"]),
        round(x, 6),
        round(y, 6),
    )


def _note_center(note: Mapping[str, Any]) -> tuple[float, float]:
    center = note.get("center")
    if isinstance(center, Mapping) and center.get("x") is not None and center.get("y") is not None:
        return float(center["x"]), float(center["y"])
    if note.get("x") is not None and note.get("y") is not None:
        return float(note["x"]), float(note["y"])
    raise ValueError(f"Prediction note has no center: {note.get('candidate_id')}")


def _note_group_identity(note: Mapping[str, Any]) -> str | None:
    value = note.get("onset_group_index", note.get("onset_group_id"))
    return str(value) if value is not None else None


def _prediction_groups(lane_payload: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    notes = _lane_notes(lane_payload, lane="prediction")
    group_ids = [_note_group_identity(note) for note in notes]
    if not any(group_ids):
        return None
    if any(group_id is None for group_id in group_ids):
        raise ValueError("Frozen prediction mixes grouped and ungrouped pitch notes")
    groups: list[dict[str, Any]] = []
    seen: set[str] = set()
    for note, group_id in zip(notes, group_ids, strict=True):
        assert group_id is not None
        if not groups or groups[-1]["group_id"] != group_id:
            if group_id in seen:
                raise ValueError(f"Prediction onset group is non-contiguous: {group_id}")
            seen.add(group_id)
            groups.append({"group_id": group_id, "notes": []})
        groups[-1]["notes"].append(note)
    return groups


def _default_mapping(measure_numbers: Sequence[int], crop_indices: Sequence[int]) -> dict[str, Any]:
    if len(measure_numbers) != len(crop_indices):
        raise ValueError(
            "Automatic default mapping requires equal MusicXML measure and frozen crop counts; "
            "provide --mapping to preserve one-to-many or many-to-one segmentation"
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


def _mapping_structure_support(
    mapping: Mapping[str, Any], truth: heldout.VisibleMusicXMLTruth
) -> dict[int, dict[str, Any]]:
    result = {}
    for entry in mapping["automatic_crops"]:
        crop = int(entry["automatic_crop_index"])
        split_boundaries = []
        for span in entry["physical_measure_spans"]:
            measure = int(span["measure_number"])
            notes = truth.notes_by_measure[measure]
            start = int(span["note_start"])
            end = int(span["note_end"])
            if 0 < start < len(notes) and int(notes[start - 1]["onset_divisions"]) == int(
                notes[start]["onset_divisions"]
            ):
                split_boundaries.append(
                    {"measure_number": measure, "boundary": "start", "note_index": start}
                )
            if 0 < end < len(notes) and int(notes[end - 1]["onset_divisions"]) == int(
                notes[end]["onset_divisions"]
            ):
                split_boundaries.append(
                    {"measure_number": measure, "boundary": "end", "note_index": end}
                )
        result[crop] = {
            "well_defined": not split_boundaries,
            "split_onset_group_boundaries": split_boundaries,
        }
    return result


def _score_lane(
    truth_rows: Sequence[Mapping[str, Any]],
    paired_rows_by_crop: Mapping[int, Mapping[str, Any]],
    *,
    lane: str,
    structure_support: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    crops = []
    totals = {
        "predicted_notes": 0,
        "truth_notes": 0,
        "count_capacity_matches": 0,
        "diatonic_exact": 0,
        "diatonic_substitutions": 0,
        "diatonic_insertions": 0,
        "diatonic_deletions": 0,
        "structure_supported_crops": 0,
        "predicted_groups": 0,
        "truth_groups": 0,
        "chord_size_exact": 0,
        "chord_size_substitutions": 0,
        "chord_size_insertions": 0,
        "chord_size_deletions": 0,
        "exact_structure_crops": 0,
    }
    for truth_row in truth_rows:
        crop = int(truth_row["automatic_crop_index"])
        lane_payload = paired_rows_by_crop[crop]["lanes"][lane]
        predicted_notes = _lane_notes(lane_payload, lane=lane)
        predicted_groups = _prediction_groups(lane_payload)
        truth_groups = _truth_groups(truth_row.get("notes") or [])
        if predicted_groups is None:
            predicted_positions = [
                _prediction_staff_position(note)
                for note in sorted(
                    predicted_notes,
                    key=lambda note: (
                        _note_center(note)[0],
                        _prediction_staff_position(note),
                        str(note["candidate_id"]),
                    ),
                )
            ]
        else:
            predicted_positions = [
                position
                for group in predicted_groups
                for position in sorted(_prediction_staff_position(note) for note in group["notes"])
            ]
        truth_positions = [
            position
            for group in truth_groups
            for position in sorted(_truth_staff_position(note) for note in group["notes"])
        ]
        diatonic_alignment = heldout._align_pitches(predicted_positions, truth_positions)
        predicted_count = len(predicted_positions)
        truth_count = len(truth_positions)
        totals["predicted_notes"] += predicted_count
        totals["truth_notes"] += truth_count
        totals["count_capacity_matches"] += min(predicted_count, truth_count)
        totals["diatonic_exact"] += diatonic_alignment["exact_pitch_matches"]
        totals["diatonic_substitutions"] += diatonic_alignment["substitutions"]
        totals["diatonic_insertions"] += diatonic_alignment["insertions"]
        totals["diatonic_deletions"] += diatonic_alignment["deletions"]

        crop_report = {
            "automatic_crop_index": crop,
            "physical_measure_numbers": truth_row["physical_measure_numbers"],
            "predicted_note_count": predicted_count,
            "truth_note_count": truth_count,
            "predicted_ordered_staff_positions": predicted_positions,
            "truth_ordered_staff_positions": truth_positions,
            "ordered_diatonic_alignment": diatonic_alignment,
            "exact_ordered_diatonic_positions": predicted_positions == truth_positions,
            "onset_group_chord_size": {
                "status": (
                    NOT_SCORED_AMBIGUOUS_MAPPING
                    if not structure_support[crop]["well_defined"]
                    else NOT_SCORED_MISSING_GROUPS
                ),
                "mapping_diagnostics": structure_support[crop],
            },
        }
        if structure_support[crop]["well_defined"] and predicted_groups is not None:
            predicted_sizes = [len(group["notes"]) for group in predicted_groups]
            truth_sizes = [len(group["notes"]) for group in truth_groups]
            size_alignment = heldout._align_pitches(predicted_sizes, truth_sizes)
            totals["structure_supported_crops"] += 1
            totals["predicted_groups"] += len(predicted_sizes)
            totals["truth_groups"] += len(truth_sizes)
            totals["chord_size_exact"] += size_alignment["exact_pitch_matches"]
            totals["chord_size_substitutions"] += size_alignment["substitutions"]
            totals["chord_size_insertions"] += size_alignment["insertions"]
            totals["chord_size_deletions"] += size_alignment["deletions"]
            totals["exact_structure_crops"] += int(predicted_sizes == truth_sizes)
            crop_report["onset_group_chord_size"] = {
                "status": "scored",
                "predicted_ordered_chord_sizes": predicted_sizes,
                "truth_ordered_chord_sizes": truth_sizes,
                "alignment": size_alignment,
                "exact_structure": predicted_sizes == truth_sizes,
                "mapping_diagnostics": structure_support[crop],
            }
        crops.append(crop_report)

    count_precision = heldout._ratio(totals["count_capacity_matches"], totals["predicted_notes"])
    count_recall = heldout._ratio(totals["count_capacity_matches"], totals["truth_notes"])
    diatonic_total = (
        totals["diatonic_exact"]
        + totals["diatonic_substitutions"]
        + totals["diatonic_insertions"]
        + totals["diatonic_deletions"]
    )
    chord_size_total = (
        totals["chord_size_exact"]
        + totals["chord_size_substitutions"]
        + totals["chord_size_insertions"]
        + totals["chord_size_deletions"]
    )
    return {
        "summary": {
            "automatic_crop_count": len(crops),
            "predicted_note_count": totals["predicted_notes"],
            "truth_note_count": totals["truth_notes"],
            "note_count_precision": count_precision,
            "note_count_recall": count_recall,
            "note_count_f1": heldout._f1(count_precision, count_recall),
            "exact_diatonic_staff_position_matches": totals["diatonic_exact"],
            "ordered_diatonic_substitutions": totals["diatonic_substitutions"],
            "ordered_diatonic_insertions": totals["diatonic_insertions"],
            "ordered_diatonic_deletions": totals["diatonic_deletions"],
            "ordered_diatonic_alignment_accuracy": heldout._ratio(
                totals["diatonic_exact"], diatonic_total
            ),
            "structure_supported_crop_count": totals["structure_supported_crops"],
            "structure_unsupported_crop_count": len(crops) - totals["structure_supported_crops"],
            "predicted_onset_group_count": totals["predicted_groups"],
            "truth_onset_group_count": totals["truth_groups"],
            "exact_chord_size_matches": totals["chord_size_exact"],
            "chord_size_substitutions": totals["chord_size_substitutions"],
            "chord_size_insertions": totals["chord_size_insertions"],
            "chord_size_deletions": totals["chord_size_deletions"],
            "chord_size_alignment_accuracy": heldout._ratio(
                totals["chord_size_exact"], chord_size_total
            ),
            "exact_structure_crops": totals["exact_structure_crops"],
        },
        "crops": crops,
    }


def _truth_groups(notes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups = []
    for note in notes:
        key = (int(note["physical_measure_number"]), int(note["onset_divisions"]))
        if not groups or groups[-1]["key"] != key:
            groups.append({"key": key, "notes": []})
        groups[-1]["notes"].append(note)
    return groups


def _prediction_staff_position(note: Mapping[str, Any]) -> int:
    if note.get("staff_position") is not None:
        return int(note["staff_position"])
    return _pitch_to_treble_staff_position(str(note.get("pitch", "")))


def _truth_staff_position(note: Mapping[str, Any]) -> int:
    return _pitch_to_treble_staff_position(str(note["pitch"]))


def _pitch_to_treble_staff_position(pitch: str) -> int:
    match = re.fullmatch(r"([A-G])(?:bb|##|b|#|\([^)]+\))?(-?\d+)", pitch)
    if match is None:
        raise ValueError(f"Cannot derive diatonic staff position from pitch: {pitch!r}")
    step, octave_text = match.groups()
    step_index = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}[step]
    absolute_diatonic = int(octave_text) * 7 + step_index
    treble_bottom_e4 = 4 * 7 + 2
    return absolute_diatonic - treble_bottom_e4


def _build_report(
    *,
    target: Mapping[str, Any],
    mapping_mode: str,
    truth: heldout.VisibleMusicXMLTruth,
    lane_reports: Mapping[str, Mapping[str, Any]],
    structure_support: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = lane_reports[LANE_BASELINE]["summary"]
    recovered = lane_reports[LANE_RECOVERED]["summary"]
    mapping_unsupported_count = sum(
        not bool(record["well_defined"]) for record in structure_support.values()
    )
    structure_supported_count = min(
        baseline["structure_supported_crop_count"],
        recovered["structure_supported_crop_count"],
    )
    structure_status = "scored"
    if mapping_unsupported_count == len(structure_support):
        structure_status = NOT_SCORED_AMBIGUOUS_MAPPING
    elif structure_supported_count == 0:
        structure_status = NOT_SCORED_MISSING_GROUPS
    elif structure_supported_count < len(structure_support):
        structure_status = "partially_scored_mapping_safe_crops_only"
    metric_support = {
        "note_count_precision_recall_f1": "scored",
        "ordered_diatonic_pitch_staff_position": "scored_ignore_key_signature_accidentals",
        "onset_group_chord_size_structure": structure_status,
        "key_signature_and_accidentals": NOT_SCORED_LOCALIZATION_GATE,
        "meter": NOT_SCORED_LOCALIZATION_GATE,
        "absolute_onset_and_rhythm": NOT_SCORED_LOCALIZATION_GATE,
        "duration": NOT_SCORED_LOCALIZATION_GATE,
        "rests": NOT_SCORED_LOCALIZATION_GATE,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "independent_dyad_recovery_one_shot_evaluation",
        "status": "evaluated_exactly_once_after_frozen_predictions",
        "target": dict(target),
        "mapping_mode": mapping_mode,
        "localization_contract": (
            "Diatonic pitch is scored as treble-clef staff position; MusicXML key-signature "
            "and note accidental alterations are intentionally ignored."
        ),
        "prediction_group_contract": (
            "Chord-size structure is scored only when frozen pitch-lane notes carry explicit "
            "onset_group_index/onset_group_id values; x proximity is never fabricated by the "
            "evaluator."
        ),
        "metric_support": metric_support,
        "frozen_context": {
            "key_signature": "unknown",
            "meter": "unsupported_not_frozen",
            "rhythm": "unsupported_not_frozen",
            "rests": "unsupported_not_frozen",
        },
        "source_musicxml_context_not_scored": {
            "time_signature": truth.time_signature,
            "key_fifths": truth.key_fifths,
            "key_events": [list(event) for event in truth.key_events],
            "clef": list(truth.clef) if truth.clef is not None else None,
        },
        "lanes": dict(lane_reports),
        "comparison": {
            "predicted_note_count_delta": recovered["predicted_note_count"]
            - baseline["predicted_note_count"],
            "note_count_f1_delta": round(recovered["note_count_f1"] - baseline["note_count_f1"], 6),
            "exact_diatonic_staff_position_match_delta": (
                recovered["exact_diatonic_staff_position_matches"]
                - baseline["exact_diatonic_staff_position_matches"]
            ),
            "ordered_diatonic_alignment_accuracy_delta": round(
                recovered["ordered_diatonic_alignment_accuracy"]
                - baseline["ordered_diatonic_alignment_accuracy"],
                6,
            ),
            "exact_chord_size_match_delta": recovered["exact_chord_size_matches"]
            - baseline["exact_chord_size_matches"],
            "exact_structure_crop_delta": recovered["exact_structure_crops"]
            - baseline["exact_structure_crops"],
        },
        "unsupported": {
            name: {"status": status}
            for name, status in metric_support.items()
            if status.startswith("not_scored")
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
