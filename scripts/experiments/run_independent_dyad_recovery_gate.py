"""Run and seal the independent No Lo Creas dyad-recovery gate.

Generic score-disjoint inference is materialized once and frozen unchanged.
The fixed recovery rule then creates a second, additive lane. Target truth and
MusicXML are forbidden until both lanes and all provenance have been sealed.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import freeze_independent_dyad_recovery_gate as gate  # noqa: E402
from scripts.experiments import freeze_third_score_heldout as base  # noqa: E402
from scripts.experiments import run_third_score_heldout_inference as inference  # noqa: E402
from scripts.experiments import spike_consumed_polyphonic_pitch_repair as recovery  # noqa: E402

SCHEMA_VERSION = 2
PAIR_VERSION = "independent-dyad-recovery-paired-predictions-v2"
DEFAULT_INFERENCE_DIRNAME = "baseline_inference_v1"
PAIR_DIRNAME = "dyad_recovery_v1"
PAIR_FREEZE_DIRNAME = "frozen"
DEFAULT_MODEL_DIR = (
    REPO_ROOT / "out/vlm_melody_consumed_training/cross_score_notehead_v1_replay_20260722"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared_manifest", type=Path)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args(argv)
    try:
        result = run_and_seal(args.prepared_manifest, model_dir=args.model_dir)
    except (FileExistsError, FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def run_and_seal(prepared_manifest_path: Path, *, model_dir: Path) -> dict[str, Any]:
    """Materialize one baseline, one additive recovery lane, and both freezes."""
    prepared_manifest_path = prepared_manifest_path.resolve()
    model_dir = model_dir.resolve()
    prepared = base._read_json(prepared_manifest_path)
    _validate_prepared_gate(prepared_manifest_path, prepared)

    baseline = _materialize_or_resume_baseline(
        prepared_manifest_path,
        model_dir=model_dir,
    )
    inference_dir = Path(baseline["inference_dir"])
    generic_manifest = inference._read_json(inference_dir / "manifest.json")
    validated = inference._verify_inference_binding(
        prepared_manifest_path,
        model_dir=model_dir,
        inference_dir=inference_dir,
        manifest=generic_manifest,
    )
    model_payload = validated["model"]
    detailed_rows = inference._read_jsonl(inference_dir / "inference.jsonl")
    pair_result = materialize_paired_recovery(
        detailed_rows,
        model_payload=model_payload,
        inference_dir=inference_dir,
        expected_target=prepared["target"],
    )

    paired_freeze = freeze_paired_recovery(
        prepared_manifest_path,
        model_dir=model_dir,
        inference_dir=inference_dir,
        pair_dir=Path(pair_result["pair_dir"]),
    )
    return {
        "target": prepared["target"],
        "baseline_inference": baseline,
        "paired_recovery": pair_result,
        "paired_freeze": paired_freeze,
    }


def _materialize_or_resume_baseline(
    prepared_manifest_path: Path,
    *,
    model_dir: Path,
) -> dict[str, Any]:
    """Create generic inference once, or validate and reuse a pre-pair failure."""
    namespace_root = prepared_manifest_path.parent
    inference_dir = namespace_root / DEFAULT_INFERENCE_DIRNAME
    if not inference_dir.exists():
        return inference.materialize_third_score_inference(
            prepared_manifest_path,
            model_dir=model_dir,
            inference_dirname=DEFAULT_INFERENCE_DIRNAME,
        )

    pair_dir = inference_dir / PAIR_DIRNAME
    frozen_dir = namespace_root / PAIR_FREEZE_DIRNAME
    if pair_dir.exists() or frozen_dir.exists():
        raise FileExistsError(
            "Refusing dyad baseline resume after paired or frozen output already exists"
        )
    manifest_path = inference_dir / "manifest.json"
    manifest = inference._read_json(manifest_path)
    expected_gate = inference.GATE_CONFIGS[gate.INDEPENDENT_DYAD_RECOVERY_GATE.prepare_kind]
    prepared = base._read_json(prepared_manifest_path)
    if (
        manifest.get("kind") != expected_gate["manifest_kind"]
        or manifest.get("version") != expected_gate["inference_version"]
        or manifest.get("status") != "inferred_awaiting_freeze"
        or manifest.get("target") != prepared.get("target")
        or manifest.get("truth_accessed") is not False
        or manifest.get("truth_used") is not False
    ):
        raise ValueError("Existing dyad baseline inference is not resumable")
    validated = inference._verify_inference_binding(
        prepared_manifest_path,
        model_dir=model_dir,
        inference_dir=inference_dir,
        manifest=manifest,
    )
    expected_count = int(validated["expected_count"])
    if expected_count != gate.EXPECTED_CROP_COUNT:
        raise ValueError("Existing dyad baseline inference count contract drift")
    return {
        "inference_dir": str(inference_dir),
        "manifest": str(manifest_path),
        "manifest_sha256": inference._sha256(manifest_path),
        "predictions": str(inference_dir / "predictions.jsonl"),
        "predictions_sha256": inference._sha256(inference_dir / "predictions.jsonl"),
        "inference_sha256": inference._sha256(inference_dir / "inference.jsonl"),
        "output_count": expected_count,
        "warnings": list(manifest.get("warnings", [])),
        "resumed_after_validated_pre_pair_failure": True,
    }


def materialize_paired_recovery(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_payload: Mapping[str, Any],
    inference_dir: Path,
    expected_target: Mapping[str, Any],
) -> dict[str, Any]:
    """Write paired lanes exactly once while preserving generic baseline bytes."""
    output_dir = inference_dir / PAIR_DIRNAME
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite create-once dyad recovery: {output_dir}")
    if len(rows) != gate.EXPECTED_CROP_COUNT:
        raise ValueError(
            f"Independent dyad gate requires {gate.EXPECTED_CROP_COUNT} inference rows, "
            f"got {len(rows)}"
        )
    selector = recovery.selector_config_from_model(model_payload)
    temp_dir = output_dir.with_name(f".{output_dir.name}.tmp")
    if temp_dir.exists():
        raise FileExistsError(f"Refusing stale dyad-recovery temporary output: {temp_dir}")
    temp_dir.mkdir(parents=False)
    try:
        pair_rows: list[dict[str, Any]] = []
        diagnostic_rows: list[dict[str, Any]] = []
        overlay_dir = temp_dir / "overlays"
        overlay_dir.mkdir()
        overlay_paths = []
        for row in rows:
            paired, diagnostics, baseline_candidates, recovered_candidates = _pair_row(
                row,
                selector=selector,
                expected_target=expected_target,
            )
            pair_rows.append(paired)
            diagnostic_rows.append(diagnostics)
            measure = int(row["identity"]["automatic_measure_index"])
            overlay_path = overlay_dir / f"measure_{measure:03d}.png"
            _render_overlay(
                row,
                baseline=baseline_candidates,
                recovered=recovered_candidates,
                output_path=overlay_path,
            )
            overlay_paths.append(overlay_path)

        invariance = verify_paired_contract(
            pair_rows,
            rows,
            expected_target=expected_target,
        )
        parameters_path = temp_dir / "recovery_parameters.json"
        inference._write_json(
            parameters_path,
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "independent_dyad_recovery_fixed_parameters",
                "config_id": recovery.EDGE_SAFE_STEM_DYAD_CONFIG_ID,
                "parameters": dict(recovery.EDGE_SAFE_STEM_DYAD_PARAMETERS),
                "truth_accessed": False,
                "truth_used": False,
            },
        )
        paired_path = temp_dir / "paired_predictions.jsonl"
        diagnostics_path = temp_dir / "recovery_diagnostics.jsonl"
        invariance_path = temp_dir / "additive_invariance.json"
        inference._write_jsonl(paired_path, pair_rows)
        inference._write_jsonl(diagnostics_path, diagnostic_rows)
        inference._write_json(invariance_path, invariance)
        inference._write_contact_sheet(overlay_paths, temp_dir / "contact_sheet.png")

        manifest_path = temp_dir / "manifest.json"
        artifacts = _artifact_records(temp_dir)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "independent_dyad_recovery_paired_prediction_manifest",
            "version": PAIR_VERSION,
            "status": "predicted_before_truth",
            "truth_accessed": False,
            "truth_used": False,
            "target": dict(expected_target),
            "create_once": True,
            "baseline_contract": (
                "canonical_prediction copied unchanged from generic score-disjoint inference"
            ),
            "recovery_contract": (
                "add at most one companion per existing x group; never delete or reposition "
                "baseline candidates"
            ),
            "paired_note_contract": (
                "each lane has one notes list with staff_position, onset_group_index, and "
                "recovered; both lanes use direct natural-treble mapping from frozen candidate "
                "coordinates and recovery reuses baseline group IDs"
            ),
            "evaluation_scope": {
                "supported": ["candidate_localization", "note_count", "diatonic_pitch"],
                "unsupported": [
                    "chromatic_key_accuracy",
                    "onset",
                    "duration",
                    "rests",
                    "meter",
                ],
            },
            "measure_count": len(pair_rows),
            "recovered_head_count": sum(
                int(row["lanes"]["edge_safe_recovery"]["recovered_head_count"]) for row in pair_rows
            ),
            "implementation": inference._file_record(Path(__file__)),
            "recovery_implementation": inference._file_record(Path(recovery.__file__)),
            "baseline_inference": {
                "manifest": inference._file_record(inference_dir / "manifest.json"),
                "predictions": inference._file_record(inference_dir / "predictions.jsonl"),
                "detailed_inference": inference._file_record(inference_dir / "inference.jsonl"),
            },
            "artifacts": artifacts,
        }
        inference._write_json(manifest_path, manifest)
        temp_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return {
        "pair_dir": str(output_dir),
        "manifest": str(output_dir / "manifest.json"),
        "manifest_sha256": inference._sha256(output_dir / "manifest.json"),
        "paired_predictions": str(output_dir / "paired_predictions.jsonl"),
        "paired_predictions_sha256": inference._sha256(output_dir / "paired_predictions.jsonl"),
        "invariance": invariance,
    }


def _pair_row(
    row: Mapping[str, Any],
    *,
    selector: Mapping[str, Any],
    expected_target: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    _validate_row_context(row, expected_target=expected_target)
    baseline = recovery.select_candidates(row, selector)
    canonical = row.get("canonical_prediction")
    if not isinstance(canonical, Mapping):
        raise ValueError("Detailed inference row has no canonical_prediction")
    _verify_generic_baseline(row, baseline, canonical)

    stem_features, stem_metadata = recovery.candidate_local_stem_features(row)
    recovered = recovery.recover_edge_safe_stem_aware_chord_candidates(
        row,
        selector,
        baseline,
        stem_features=stem_features,
        **recovery.EDGE_SAFE_STEM_DYAD_PARAMETERS,
    )
    _verify_additive_recovery(row, selector=selector, baseline=baseline, recovered=recovered)

    baseline_group_by_id = _baseline_group_indices(row, selector=selector, baseline=baseline)
    baseline_candidate_lane = [
        _candidate_record(
            candidate,
            recovered=False,
            onset_group_index=baseline_group_by_id[str(candidate["candidate_id"])],
        )
        for candidate in baseline
    ]
    recovery_candidate_lane = [
        *copy.deepcopy(baseline_candidate_lane),
        *[
            _candidate_record(
                candidate,
                recovered=True,
                onset_group_index=int(candidate["recovery_group_index"]),
            )
            for candidate in recovered
        ],
    ]
    recovery_candidate_lane.sort(key=_candidate_sort_key)
    generic_note_by_id = {str(note["candidate_id"]): note for note in canonical["notes"]}
    baseline_notes = [
        _normalized_note(
            row,
            candidate,
            onset_group_index=baseline_group_by_id[str(candidate["candidate_id"])],
            recovered=False,
            generic_note=generic_note_by_id[str(candidate["candidate_id"])],
        )
        for candidate in baseline
    ]
    baseline_notes.sort(key=_note_sort_key)
    recovered_notes = [
        _normalized_note(
            row,
            candidate,
            onset_group_index=int(candidate["recovery_group_index"]),
            recovered=True,
        )
        for candidate in recovered
    ]
    recovery_notes = [*copy.deepcopy(baseline_notes), *recovered_notes]
    recovery_notes.sort(key=_note_sort_key)
    identity = dict(row["identity"])
    paired = {
        "schema_version": SCHEMA_VERSION,
        "identity": identity,
        "truth_accessed": False,
        "truth_used": False,
        "source": dict(row["source"]),
        "context": {
            "clef": "treble",
            "key_hint": None,
            "time_signature": None,
            "rhythm_rest_supported": False,
        },
        "lanes": {
            "baseline_generic": {
                "canonical_prediction": dict(canonical),
                "canonical_prediction_sha256": inference._hash_json(canonical),
                "candidate_lane": sorted(baseline_candidate_lane, key=_candidate_sort_key),
                "notes": baseline_notes,
                "recovered_head_count": 0,
            },
            "edge_safe_recovery": {
                "config_id": recovery.EDGE_SAFE_STEM_DYAD_CONFIG_ID,
                "candidate_lane": recovery_candidate_lane,
                "notes": recovery_notes,
                "recovered_head_count": len(recovered),
                "recovered_candidate_ids": [
                    str(candidate["candidate_id"]) for candidate in recovered
                ],
            },
        },
    }
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "identity": identity,
        "truth_accessed": False,
        "truth_used": False,
        "baseline_candidate_ids": [str(item["candidate_id"]) for item in baseline],
        "recovered_candidate_ids": [str(item["candidate_id"]) for item in recovered],
        "baseline_onset_group_count": recovery._onset_group_count(
            baseline,
            recovery._staff_spacing(row) * float(selector["nms_x_spaces"]),
        ),
        "baseline_onset_groups": [
            {
                "onset_group_index": group_index,
                "candidate_ids": sorted(
                    candidate_id
                    for candidate_id, assigned_group in baseline_group_by_id.items()
                    if assigned_group == group_index
                ),
            }
            for group_index in sorted(set(baseline_group_by_id.values()))
        ],
        "recovery": [
            {
                "candidate_id": str(item["candidate_id"]),
                "center": dict(item["center"]),
                "score": float(item["score"]),
                "recovery_group_index": int(item["recovery_group_index"]),
                "y_gap_staff_spaces": float(item["recovery_y_gap_staff_spaces"]),
                "score_ratio": item["recovery_score_ratio"],
                "stem_attachment_score": float(item["stem_attachment_score"]),
                "leading_edge_distance_staff_spaces": float(
                    item["leading_edge_distance_staff_spaces"]
                ),
            }
            for item in recovered
        ],
        "candidate_stem_features": {
            candidate_id: dict(stem_features[candidate_id])
            for candidate_id in sorted(stem_features)
        },
        "stem_feature_metadata": stem_metadata,
    }
    return paired, diagnostics, baseline, recovered


def _validate_row_context(row: Mapping[str, Any], *, expected_target: Mapping[str, Any]) -> None:
    if row.get("truth_used") is not False:
        raise ValueError("Independent dyad inference row is not truth-blind")
    identity = row.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("Independent dyad inference row has no identity")
    if str(identity.get("slug")) != str(expected_target["slug"]) or int(
        identity.get("system_index", -1)
    ) != int(expected_target["system_index"]):
        raise ValueError("Independent dyad inference target substitution")
    context = row.get("allowed_context")
    expected = {
        "allow_pickup": False,
        "clef": "treble",
        "expected_measure_beats": None,
        "key_hint": None,
        "time_signature": None,
    }
    if context != expected:
        raise ValueError("Independent dyad gate requires unknown key and unsupported meter context")
    if row.get("decoder_status") != "not_applied_missing_expected_measure_beats":
        raise ValueError("Independent dyad gate must not run rhythm/rest decoding")


def _verify_generic_baseline(
    row: Mapping[str, Any],
    baseline: Sequence[Mapping[str, Any]],
    canonical: Mapping[str, Any],
) -> None:
    notes = canonical.get("notes")
    if not isinstance(notes, list):
        raise ValueError("Generic canonical prediction has no notes")
    expected = [
        (
            str(candidate["candidate_id"]),
            round(float(candidate["center"]["x"]), 3),
            round(float(candidate["center"]["y"]), 3),
        )
        for candidate in sorted(baseline, key=_candidate_sort_key)
    ]
    actual = [
        (
            str(note["candidate_id"]),
            float(note["center"]["x"]),
            float(note["center"]["y"]),
        )
        for note in notes
    ]
    if actual != expected:
        raise ValueError("Generic baseline candidate identity or coordinates drifted")
    for note in notes:
        if note.get("onset_beats") is not None or note.get("duration_beats") is not None:
            raise ValueError("Generic baseline unexpectedly contains rhythm inference")
    if canonical.get("rests") or canonical.get("rhythm_tokens"):
        raise ValueError("Generic baseline unexpectedly contains rests or rhythm tokens")


def _verify_additive_recovery(
    row: Mapping[str, Any],
    *,
    selector: Mapping[str, Any],
    baseline: Sequence[Mapping[str, Any]],
    recovered: Sequence[Mapping[str, Any]],
) -> None:
    baseline_ids = {str(item["candidate_id"]) for item in baseline}
    recovered_ids = [str(item["candidate_id"]) for item in recovered]
    if baseline_ids.intersection(recovered_ids) or len(recovered_ids) != len(set(recovered_ids)):
        raise ValueError("Recovery candidate identity overlaps or duplicates the baseline")
    group_indices = [int(item["recovery_group_index"]) for item in recovered]
    if len(group_indices) != len(set(group_indices)):
        raise ValueError("Recovery added more than one companion to an existing x group")
    spacing = recovery._staff_spacing(row)
    x_radius = spacing * float(selector["nms_x_spaces"])
    baseline_groups = recovery._horizontal_candidate_groups(baseline, x_radius)
    for item in recovered:
        group_index = int(item["recovery_group_index"])
        if not 1 <= group_index <= len(baseline_groups):
            raise ValueError("Recovery referenced a nonexistent baseline x group")
        claimed_group = baseline_groups[group_index - 1]
        candidate_x = float(item["center"]["x"])
        if not any(
            abs(candidate_x - float(anchor["center"]["x"])) < x_radius for anchor in claimed_group
        ):
            raise ValueError("Recovery candidate does not belong to its baseline x group")
    before = recovery._onset_group_count(baseline, x_radius)
    after = recovery._onset_group_count([*baseline, *recovered], x_radius)
    if before != after:
        raise ValueError("Recovery introduced a new x group")
    if len(recovered) > before:
        raise ValueError("Recovery exceeded one companion per existing x group")


def verify_paired_contract(
    paired_rows: Sequence[Mapping[str, Any]],
    generic_rows: Sequence[Mapping[str, Any]],
    *,
    expected_target: Mapping[str, Any],
) -> dict[str, Any]:
    if len(paired_rows) != len(generic_rows) or len(paired_rows) != gate.EXPECTED_CROP_COUNT:
        raise ValueError("Paired/generic dyad row count mismatch")
    recovered_total = 0
    for paired, generic in zip(paired_rows, generic_rows, strict=True):
        _validate_row_context(generic, expected_target=expected_target)
        if paired.get("identity") != generic.get("identity"):
            raise ValueError("Paired/generic dyad identity mismatch")
        lanes = paired.get("lanes")
        if not isinstance(lanes, Mapping):
            raise ValueError("Paired dyad row has no lanes")
        baseline_lane = lanes["baseline_generic"]
        recovery_lane = lanes["edge_safe_recovery"]
        canonical = generic["canonical_prediction"]
        if baseline_lane["canonical_prediction"] != canonical or baseline_lane[
            "canonical_prediction_sha256"
        ] != inference._hash_json(canonical):
            raise ValueError("Paired baseline is not byte-equivalent generic inference")
        baseline_candidates = baseline_lane["candidate_lane"]
        recovered_candidates = recovery_lane["candidate_lane"]
        baseline_by_id = {str(item["candidate_id"]): item for item in baseline_candidates}
        recovered_by_id = {str(item["candidate_id"]): item for item in recovered_candidates}
        if not set(baseline_by_id).issubset(recovered_by_id):
            raise ValueError("Recovery deleted a baseline candidate")
        for candidate_id, baseline_candidate in baseline_by_id.items():
            if recovered_by_id[candidate_id] != baseline_candidate:
                raise ValueError("Recovery repositioned or changed a baseline candidate")
        recovered_ids = list(recovery_lane["recovered_candidate_ids"])
        if set(recovered_by_id) != set(baseline_by_id).union(recovered_ids):
            raise ValueError("Recovery candidate lane contains unrecorded additions")
        baseline_group_by_id = {
            str(candidate["candidate_id"]): int(candidate["onset_group_index"])
            for candidate in baseline_candidates
        }
        expected_group_indices = list(range(1, len(baseline_candidates) + 1))
        if [
            baseline_group_by_id[str(candidate["candidate_id"])]
            for candidate in sorted(baseline_candidates, key=_candidate_sort_key)
        ] != expected_group_indices:
            raise ValueError("Baseline onset group IDs are not deterministic left-to-right IDs")
        if any(bool(candidate.get("recovered")) for candidate in baseline_candidates):
            raise ValueError("Baseline candidate lane contains a recovered candidate")
        baseline_notes = baseline_lane["notes"]
        generic_note_by_id = {str(note["candidate_id"]): note for note in canonical["notes"]}
        expected_baseline_notes = sorted(
            [
                _normalized_note(
                    generic,
                    candidate,
                    onset_group_index=baseline_group_by_id[str(candidate["candidate_id"])],
                    recovered=False,
                    generic_note=generic_note_by_id[str(candidate["candidate_id"])],
                )
                for candidate in baseline_candidates
            ],
            key=_note_sort_key,
        )
        if baseline_notes != expected_baseline_notes:
            raise ValueError("Baseline note records drifted from generic inference")

        recovered_note_by_id = {str(note["candidate_id"]): note for note in recovery_lane["notes"]}
        if len(recovered_note_by_id) != len(recovery_lane["notes"]):
            raise ValueError("Recovery note lane contains duplicate candidate IDs")
        if not set(baseline_group_by_id).issubset(recovered_note_by_id):
            raise ValueError("Recovery deleted a baseline note")
        for baseline_note in baseline_notes:
            candidate_id = str(baseline_note["candidate_id"])
            if recovered_note_by_id[candidate_id] != baseline_note:
                raise ValueError("Recovery changed a baseline note")
        baseline_group_ids = set(baseline_group_by_id.values())
        expected_recovery_notes = copy.deepcopy(baseline_notes)
        for candidate_id in recovered_ids:
            candidate = recovered_by_id[candidate_id]
            group_index = int(candidate["onset_group_index"])
            if not bool(candidate.get("recovered")) or group_index not in baseline_group_ids:
                raise ValueError("Recovered candidate does not reuse a baseline group ID")
            expected_recovery_notes.append(
                _normalized_note(
                    generic,
                    candidate,
                    onset_group_index=group_index,
                    recovered=True,
                )
            )
        expected_recovery_notes.sort(key=_note_sort_key)
        if recovery_lane["notes"] != expected_recovery_notes:
            raise ValueError("Recovery note records or onset groups drifted")
        if {
            int(note["onset_group_index"]) for note in recovery_lane["notes"]
        } != baseline_group_ids:
            raise ValueError("Recovery note groups do not exactly match baseline group IDs")
        recovered_total += len(recovered_ids)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "independent_dyad_recovery_additive_invariance",
        "passed": True,
        "truth_accessed": False,
        "truth_used": False,
        "measure_count": len(paired_rows),
        "baseline_candidate_identity_preserved": True,
        "baseline_candidate_coordinates_preserved": True,
        "baseline_canonical_prediction_preserved": True,
        "baseline_candidate_identity_and_coordinates_preserved_in_normalized_notes": True,
        "paired_lane_pitches_use_direct_natural_treble_mapping": True,
        "deterministic_baseline_onset_group_ids": True,
        "recovered_notes_reuse_baseline_group_ids": True,
        "maximum_one_companion_per_existing_x_group": True,
        "recovered_head_count": recovered_total,
    }


def freeze_paired_recovery(
    prepared_manifest_path: Path,
    *,
    model_dir: Path,
    inference_dir: Path,
    pair_dir: Path,
) -> dict[str, Any]:
    """Create an independent immutable snapshot of the paired dyad gate."""
    namespace_root = prepared_manifest_path.parent
    prepared = base._read_json(prepared_manifest_path)
    _validate_prepared_gate(prepared_manifest_path, prepared)
    generic_manifest = inference._read_json(inference_dir / "manifest.json")
    validated = inference._verify_inference_binding(
        prepared_manifest_path,
        model_dir=model_dir,
        inference_dir=inference_dir,
        manifest=generic_manifest,
    )
    pair_manifest = inference._read_json(pair_dir / "manifest.json")
    pair_rows = inference._read_jsonl(pair_dir / "paired_predictions.jsonl")
    generic_rows = inference._read_jsonl(inference_dir / "inference.jsonl")
    invariance = verify_paired_contract(pair_rows, generic_rows, expected_target=prepared["target"])
    if inference._read_json(pair_dir / "additive_invariance.json") != invariance:
        raise ValueError("Dyad additive-invariance artifact drift")
    if (
        pair_manifest.get("target") != prepared["target"]
        or pair_manifest.get("version") != PAIR_VERSION
    ):
        raise ValueError("Dyad paired manifest target/version mismatch")

    output_dir = namespace_root / PAIR_FREEZE_DIRNAME
    temp_dir = namespace_root / f".{PAIR_FREEZE_DIRNAME}.tmp"
    if output_dir.exists():
        raise FileExistsError(f"Paired dyad freeze already exists: {output_dir}")
    if temp_dir.exists():
        raise FileExistsError(f"Stale paired dyad freeze exists: {temp_dir}")

    source_system = base._find_out_dir(namespace_root) / str(
        prepared["artifacts"]["source_system"]["path_relative_to_out"]
    )
    metadata_path = inference._resolve_record_path(validated["metadata_record"])
    prepared_paths = [
        metadata_path,
        *_prepared_artifact_paths(prepared_manifest_path, prepared),
    ]
    generic_inference_paths = sorted(
        path for path in inference_dir.rglob("*") if path.is_file() and pair_dir not in path.parents
    )
    pair_paths = sorted(path for path in pair_dir.rglob("*") if path.is_file())
    implementation_paths = [
        Path(__file__),
        Path(gate.__file__),
        Path(base.__file__),
        Path(inference.__file__),
        Path(recovery.__file__),
        validated["model_implementation_path"],
    ]
    model_and_training_paths = [
        model_dir / "manifest.json",
        *validated["model_artifact_paths"],
        *validated["training_input_paths"],
    ]
    groups = {
        "paired_predictions": [pair_dir / "paired_predictions.jsonl"],
        "paired_artifacts": [
            path for path in pair_paths if path != pair_dir / "paired_predictions.jsonl"
        ],
        "prepared_and_source": [source_system, *prepared_paths],
        "baseline_inference": generic_inference_paths,
        "model_and_training": model_and_training_paths,
        "implementations": implementation_paths,
    }

    temp_dir.mkdir(parents=False)
    try:
        pins = {
            role: [
                base._snapshot_artifact(
                    path,
                    frozen_dir=temp_dir,
                    published_frozen_dir=output_dir,
                    role=role,
                    index=index,
                )
                for index, path in enumerate(_deduplicate(paths), start=1)
            ]
            for role, paths in groups.items()
        }
        freeze_path = temp_dir / "freeze.json"
        freeze_payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": "independent_dyad_recovery_paired_freeze",
            "status": "frozen_awaiting_truth",
            "truth_accessed": False,
            "truth_used": False,
            "target": prepared["target"],
            "config_id": recovery.EDGE_SAFE_STEM_DYAD_CONFIG_ID,
            "parameters": dict(recovery.EDGE_SAFE_STEM_DYAD_PARAMETERS),
            "paired_manifest_sha256": inference._sha256(pair_dir / "manifest.json"),
            "generic_baseline_predictions_sha256": inference._sha256(
                inference_dir / "predictions.jsonl"
            ),
            "prepared_manifest_sha256": inference._sha256(prepared_manifest_path),
            "source_system_sha256": inference._sha256(source_system),
            "additive_invariance_sha256": inference._sha256(pair_dir / "additive_invariance.json"),
            "pins": pins,
        }
        inference._write_json(freeze_path, freeze_payload)
        sealed_path = temp_dir / "sealed_manifest.json"
        inference._write_json(
            sealed_path,
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "independent_dyad_recovery_paired_sealed_manifest",
                "status": "frozen_awaiting_truth",
                "truth_accessed": False,
                "truth_used": False,
                "target": prepared["target"],
                "freeze": {"path": "freeze.json", "sha256": inference._sha256(freeze_path)},
                "next_gate": (
                    "verify all hashes, then transcribe and evaluate localization/count/"
                    "diatonic pitch only"
                ),
            },
        )
        temp_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    verify_paired_freeze(output_dir)
    return {
        "freeze": str(output_dir / "freeze.json"),
        "freeze_sha256": inference._sha256(output_dir / "freeze.json"),
        "sealed_manifest": str(output_dir / "sealed_manifest.json"),
        "sealed_manifest_sha256": inference._sha256(output_dir / "sealed_manifest.json"),
    }


def verify_paired_freeze(output_dir: Path) -> None:
    freeze = inference._read_json(output_dir / "freeze.json")
    sealed = inference._read_json(output_dir / "sealed_manifest.json")
    if freeze.get("status") != "frozen_awaiting_truth" or freeze.get("truth_accessed") is not False:
        raise ValueError("Paired dyad freeze is not truth-blind/frozen")
    if sealed.get("status") != "frozen_awaiting_truth" or sealed.get("truth_accessed") is not False:
        raise ValueError("Paired dyad seal is not truth-blind/frozen")
    if inference._sha256(output_dir / "freeze.json") != str(sealed["freeze"]["sha256"]):
        raise ValueError("Paired dyad sealed freeze hash drift")
    if (
        freeze.get("config_id") != recovery.EDGE_SAFE_STEM_DYAD_CONFIG_ID
        or freeze.get("parameters") != recovery.EDGE_SAFE_STEM_DYAD_PARAMETERS
    ):
        raise ValueError("Paired dyad fixed parameters drift")
    pins = freeze.get("pins")
    if not isinstance(pins, Mapping):
        raise ValueError("Paired dyad freeze has no provenance pins")
    required_roles = {
        "paired_predictions",
        "paired_artifacts",
        "prepared_and_source",
        "baseline_inference",
        "model_and_training",
        "implementations",
    }
    if set(pins) != required_roles:
        raise ValueError("Paired dyad freeze provenance roles drift")
    for role in sorted(pins):
        if not pins[role]:
            raise ValueError(f"Paired dyad freeze provenance role is empty: {role}")
        for pin in pins[role]:
            snapshot = output_dir.parent / str(pin["snapshot_path_relative_to_namespace"])
            if inference._sha256(snapshot) != str(pin["snapshot_sha256"]):
                raise ValueError(f"Paired dyad snapshot hash drift: {snapshot}")
            if pin["source_sha256"] != pin["snapshot_sha256"]:
                raise ValueError(f"Paired dyad source/snapshot mismatch: {snapshot}")


def _validate_prepared_gate(path: Path, prepared: Mapping[str, Any]) -> None:
    expected_target = {"slug": gate.NO_LO_CREAS_SLUG, "system_index": gate.TARGET_SYSTEM_INDEX}
    if prepared.get("kind") != gate.INDEPENDENT_DYAD_RECOVERY_GATE.prepare_kind:
        raise ValueError("Prepared manifest is not the independent dyad-recovery gate")
    if prepared.get("target") != expected_target:
        raise ValueError("Independent dyad-recovery target substitution")
    config = prepared.get("independent_dyad_recovery_gate")
    if not isinstance(config, Mapping):
        raise ValueError("Prepared manifest has no dyad-recovery contract")
    if (
        config.get("config_id") != recovery.EDGE_SAFE_STEM_DYAD_CONFIG_ID
        or config.get("parameters") != recovery.EDGE_SAFE_STEM_DYAD_PARAMETERS
    ):
        raise ValueError("Prepared dyad-recovery parameters drift")
    if config.get("truth_accessed") is not False or prepared.get("truth_accessed") is not False:
        raise ValueError("Prepared dyad-recovery gate is not truth-blind")
    base._verify_prepared_manifest(
        path.parent,
        path,
        prepared,
        expected_kind=gate.INDEPENDENT_DYAD_RECOVERY_GATE.prepare_kind,
    )


def _prepared_artifact_paths(
    prepared_manifest_path: Path,
    prepared: Mapping[str, Any],
) -> list[Path]:
    root = prepared_manifest_path.parent
    paths = [prepared_manifest_path]
    for role in ("selection", "requests", "evaluator"):
        paths.append(root / str(prepared["artifacts"][role]["path"]))
    for record in prepared["artifacts"].get("context", {}).values():
        paths.append(root / str(record["path"]))
    for record in prepared["artifacts"]["crops"]:
        paths.append(root / str(record["path"]))
    return _deduplicate(paths)


def _deduplicate(paths: Sequence[Path]) -> list[Path]:
    result = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        if not resolved.is_file():
            raise ValueError(f"Dyad freeze artifact is missing: {resolved}")
        seen.add(resolved)
        result.append(resolved)
    return result


def _artifact_records(root: Path) -> dict[str, dict[str, str]]:
    return {
        path.relative_to(root).as_posix(): {
            "path": path.relative_to(root).as_posix(),
            "sha256": inference._sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _baseline_group_indices(
    row: Mapping[str, Any],
    *,
    selector: Mapping[str, Any],
    baseline: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    x_radius = recovery._staff_spacing(row) * float(selector["nms_x_spaces"])
    groups = recovery._horizontal_candidate_groups(baseline, x_radius)
    result: dict[str, int] = {}
    for group_index, group in enumerate(groups, start=1):
        for candidate in group:
            candidate_id = str(candidate["candidate_id"])
            if candidate_id in result:
                raise ValueError(f"Baseline candidate appears in multiple x groups: {candidate_id}")
            result[candidate_id] = group_index
    if len(result) != len(baseline):
        raise ValueError("Baseline x grouping lost candidate identity")
    return result


def _candidate_record(
    candidate: Mapping[str, Any],
    *,
    recovered: bool,
    onset_group_index: int,
) -> dict[str, Any]:
    source = candidate.get("source")
    bbox = source.get("bbox") if isinstance(source, Mapping) else None
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "center": {
            "x": round(float(candidate["center"]["x"]), 3),
            "y": round(float(candidate["center"]["y"]), 3),
        },
        "score": float(candidate["score"]),
        "recovered": recovered,
        "onset_group_index": int(onset_group_index),
        **({"bbox": dict(bbox)} if isinstance(bbox, Mapping) else {}),
    }


def _normalized_note(
    row: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    onset_group_index: int,
    recovered: bool,
    generic_note: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    x = round(float(candidate["center"]["x"]), 3)
    y = round(float(candidate["center"]["y"]), 3)
    pitch = recovery._staff_pitch(y, row["staff_geometry"]["raw_staff_lines_y_px"], {})
    result = {
        "pitch": pitch["pitch"],
        "pitch_midi": pitch["pitch_midi"],
        "staff_position": int(pitch["staff_position"]),
        "onset_group_index": int(onset_group_index),
        "recovered": recovered,
        "onset_beats": None,
        "duration_beats": None,
        "candidate_id": str(candidate["candidate_id"]),
        "center": {"x": x, "y": y},
    }
    if generic_note is not None:
        result["generic_pitch_diagnostic"] = {
            "pitch": generic_note.get("pitch"),
            "pitch_midi": generic_note.get("pitch_midi"),
        }
    return result


def _candidate_sort_key(candidate: Mapping[str, Any]) -> tuple[float, float, str]:
    return (
        float(candidate["center"]["x"]),
        float(candidate["center"]["y"]),
        str(candidate["candidate_id"]),
    )


def _note_sort_key(note: Mapping[str, Any]) -> tuple[float, float, str]:
    return (
        float(note["center"]["x"]),
        float(note["center"]["y"]),
        str(note["candidate_id"]),
    )


def _render_overlay(
    row: Mapping[str, Any],
    *,
    baseline: Sequence[Mapping[str, Any]],
    recovered: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    image_path = Path(str(row["source"]["image"]))
    if not image_path.is_absolute():
        image_path = REPO_ROOT / image_path
    if inference._sha256(image_path) != str(row["source"]["sha256"]):
        raise ValueError(f"Dyad overlay source hash drift: {image_path}")
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    for color, candidates in (((20, 90, 220), baseline), ((0, 160, 70), recovered)):
        for candidate in candidates:
            source = candidate.get("source")
            bbox = source.get("bbox") if isinstance(source, Mapping) else None
            if not isinstance(bbox, Mapping):
                continue
            bounds = tuple(int(bbox[key]) for key in ("left", "top", "right", "bottom"))
            draw.rectangle(bounds, outline=color, width=2)
            draw.text(
                (bounds[0], max(0, bounds[1] - 11)),
                str(candidate["candidate_id"]),
                fill=color,
            )
    image.save(output_path)


if __name__ == "__main__":
    raise SystemExit(main())
