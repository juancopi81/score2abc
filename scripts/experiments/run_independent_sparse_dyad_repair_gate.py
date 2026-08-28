"""Run and seal the independent Desde Lejos sparse-dyad repair gate.

The score-disjoint selector and fixed multi-head recovery form the comparison
lane. The consumed-selected dotted-hollow replacement forms the second lane.
Both are frozen before target MusicXML or pixel annotations may be opened.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import freeze_independent_sparse_dyad_repair_gate as gate  # noqa: E402
from scripts.experiments import freeze_third_score_heldout as base  # noqa: E402
from scripts.experiments import run_independent_multihead_recovery_gate as multihead  # noqa: E402
from scripts.experiments import run_third_score_heldout_inference as inference  # noqa: E402
from scripts.experiments import spike_consumed_polyphonic_pitch_repair as recovery  # noqa: E402
from scripts.experiments import spike_consumed_sparse_stem_dyad_repair as repair  # noqa: E402

SCHEMA_VERSION = 1
PAIR_VERSION = "independent-sparse-dyad-repair-paired-predictions-v1"
DEFAULT_INFERENCE_DIRNAME = "baseline_inference_v1"
PAIR_DIRNAME = "sparse_dyad_repair_v1"
FROZEN_DIRNAME = "frozen"
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
    prepared_manifest_path = prepared_manifest_path.resolve()
    model_dir = model_dir.resolve()
    prepared = base._read_json(prepared_manifest_path)
    _validate_prepared_gate(prepared_manifest_path, prepared)

    baseline = _materialize_or_resume_baseline(prepared_manifest_path, model_dir=model_dir)
    inference_dir = Path(baseline["inference_dir"])
    manifest = inference._read_json(inference_dir / "manifest.json")
    validated = inference._verify_inference_binding(
        prepared_manifest_path,
        model_dir=model_dir,
        inference_dir=inference_dir,
        manifest=manifest,
    )
    rows = inference._read_jsonl(inference_dir / "inference.jsonl")
    pair_result = materialize_paired_repair(
        rows,
        model_payload=validated["model"],
        inference_dir=inference_dir,
        expected_target=prepared["target"],
    )
    frozen = freeze_paired_repair(
        prepared_manifest_path,
        model_dir=model_dir,
        inference_dir=inference_dir,
        pair_dir=Path(pair_result["pair_dir"]),
    )
    return {
        "target": prepared["target"],
        "baseline_inference": baseline,
        "paired_repair": pair_result,
        "freeze": frozen,
    }


def _materialize_or_resume_baseline(
    prepared_manifest_path: Path,
    *,
    model_dir: Path,
) -> dict[str, Any]:
    namespace_root = prepared_manifest_path.parent
    inference_dir = namespace_root / DEFAULT_INFERENCE_DIRNAME
    if not inference_dir.exists():
        return inference.materialize_third_score_inference(
            prepared_manifest_path,
            model_dir=model_dir,
            inference_dirname=DEFAULT_INFERENCE_DIRNAME,
        )
    if (inference_dir / PAIR_DIRNAME).exists() or (namespace_root / FROZEN_DIRNAME).exists():
        raise FileExistsError("Refusing baseline resume after repair or frozen output exists")
    manifest = inference._read_json(inference_dir / "manifest.json")
    prepared = base._read_json(prepared_manifest_path)
    config = inference.GATE_CONFIGS[gate.INDEPENDENT_SPARSE_DYAD_REPAIR_GATE.prepare_kind]
    if (
        manifest.get("kind") != config["manifest_kind"]
        or manifest.get("version") != config["inference_version"]
        or manifest.get("status") != "inferred_awaiting_freeze"
        or manifest.get("target") != prepared.get("target")
        or manifest.get("truth_accessed") is not False
        or manifest.get("truth_used") is not False
    ):
        raise ValueError("Existing sparse-dyad baseline inference is not resumable")
    validated = inference._verify_inference_binding(
        prepared_manifest_path,
        model_dir=model_dir,
        inference_dir=inference_dir,
        manifest=manifest,
    )
    if int(validated["expected_count"]) != gate.EXPECTED_CROP_COUNT:
        raise ValueError("Existing sparse-dyad baseline crop-count contract drift")
    return {
        "inference_dir": str(inference_dir),
        "manifest": str(inference_dir / "manifest.json"),
        "manifest_sha256": inference._sha256(inference_dir / "manifest.json"),
        "predictions": str(inference_dir / "predictions.jsonl"),
        "predictions_sha256": inference._sha256(inference_dir / "predictions.jsonl"),
        "inference_sha256": inference._sha256(inference_dir / "inference.jsonl"),
        "output_count": gate.EXPECTED_CROP_COUNT,
        "resumed_after_validated_pre_pair_failure": True,
    }


def materialize_paired_repair(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_payload: Mapping[str, Any],
    inference_dir: Path,
    expected_target: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = inference_dir / PAIR_DIRNAME
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite create-once sparse-dyad repair: {output_dir}")
    if len(rows) != gate.EXPECTED_CROP_COUNT:
        raise ValueError(
            f"Sparse-dyad gate requires {gate.EXPECTED_CROP_COUNT} rows, got {len(rows)}"
        )
    selector = recovery.selector_config_from_model(model_payload)
    temp_dir = output_dir.with_name(f".{output_dir.name}.tmp")
    if temp_dir.exists():
        raise FileExistsError(f"Stale sparse-dyad temporary output exists: {temp_dir}")
    temp_dir.mkdir(parents=False)
    try:
        paired_rows = []
        diagnostics_rows = []
        overlay_dir = temp_dir / "overlays"
        overlay_dir.mkdir()
        overlay_paths = []
        for row in rows:
            paired, diagnostics, current, repaired = _pair_row(
                row,
                selector=selector,
                expected_target=expected_target,
            )
            paired_rows.append(paired)
            diagnostics_rows.append(diagnostics)
            measure = int(row["identity"]["automatic_measure_index"])
            overlay_path = overlay_dir / f"measure_{measure:03d}.png"
            repair._render_overlay(
                row,
                current_candidates=current,
                repaired_candidates=repaired,
                path=overlay_path,
            )
            overlay_paths.append(overlay_path)

        contract = verify_paired_contract(
            paired_rows,
            rows,
            selector=selector,
            expected_target=expected_target,
        )
        inference._write_jsonl(temp_dir / "paired_predictions.jsonl", paired_rows)
        inference._write_jsonl(temp_dir / "diagnostics.jsonl", diagnostics_rows)
        inference._write_json(temp_dir / "truth_blind_contract.json", contract)
        inference._write_contact_sheet(overlay_paths, temp_dir / "contact_sheet.png")
        artifacts = inference._recursive_artifact_records(temp_dir)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "independent_sparse_dyad_repair_paired_prediction_manifest",
            "version": PAIR_VERSION,
            "status": "predicted_before_truth",
            "truth_accessed": False,
            "truth_used": False,
            "target": dict(expected_target),
            "create_once": True,
            "comparison_lane": recovery.EDGE_SAFE_STEM_MULTIHEAD_CONFIG_ID,
            "repair_lane": repair.CONFIG_ID,
            "repair_parameters": dict(repair.PARAMETERS),
            "measure_count": len(paired_rows),
            "accepted_repair_count": int(contract["accepted_repair_count"]),
            "evaluation_scope": {
                "supported": [
                    "candidate_pixel_identity",
                    "augmentation_dot_evidence",
                    "note_count",
                    "diatonic_pitch",
                    "onset_group_chord_size",
                ],
                "unsupported": [
                    "chromatic_key_accuracy",
                    "duration",
                    "rests",
                    "meter",
                ],
            },
            "baseline_inference": {
                "manifest": inference._file_record(inference_dir / "manifest.json"),
                "predictions": inference._file_record(inference_dir / "predictions.jsonl"),
                "detailed_inference": inference._file_record(inference_dir / "inference.jsonl"),
            },
            "implementations": {
                "runner": inference._file_record(Path(__file__)),
                "multihead": inference._file_record(Path(multihead.__file__)),
                "sparse_repair": inference._file_record(Path(repair.__file__)),
            },
            "artifacts": artifacts,
        }
        inference._write_json(temp_dir / "manifest.json", manifest)
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
        "accepted_repair_count": int(contract["accepted_repair_count"]),
        "measure_count": len(paired_rows),
    }


def _pair_row(
    row: Mapping[str, Any],
    *,
    selector: Mapping[str, Any],
    expected_target: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    multihead_row, multihead_diagnostics, baseline, recovered = multihead._pair_row(
        row,
        selector=selector,
        expected_target=expected_target,
    )
    current = [*baseline, *recovered]
    stem_features, stem_metadata = recovery.candidate_local_stem_features(row)
    decision = repair.propose_sparse_shared_stem_dyad(
        row,
        selector,
        current,
        stem_features,
    )
    repaired = current
    if decision["accepted"]:
        candidate_by_id = {
            str(candidate["candidate_id"]): candidate
            for candidate in repair._normalized_candidates(row)
        }
        repaired = [candidate_by_id[candidate_id] for candidate_id in decision["proposed_ids"]]

    current_ids = {str(candidate["candidate_id"]) for candidate in current}
    repaired_group_by_id = _group_indices(row, selector=selector, candidates=repaired)
    repaired_candidate_lane = [
        _candidate_record(
            candidate,
            onset_group_index=repaired_group_by_id[str(candidate["candidate_id"])],
            sparse_repair_added=str(candidate["candidate_id"]) not in current_ids,
        )
        for candidate in repaired
    ]
    repaired_candidate_lane.sort(key=multihead._candidate_sort_key)
    repaired_notes = [
        multihead._normalized_note(
            row,
            candidate,
            onset_group_index=repaired_group_by_id[str(candidate["candidate_id"])],
            recovered=str(candidate["candidate_id"]) not in current_ids,
        )
        for candidate in repaired
    ]
    repaired_notes.sort(key=multihead._note_sort_key)
    current_lane = multihead_row["lanes"]["multihead_recovery"]
    paired = {
        "schema_version": SCHEMA_VERSION,
        "identity": dict(row["identity"]),
        "truth_accessed": False,
        "truth_used": False,
        "source": dict(row["source"]),
        "context": dict(multihead_row["context"]),
        "lanes": {
            "multihead_recovery": copy.deepcopy(current_lane),
            "sparse_dyad_repair": {
                "config_id": repair.CONFIG_ID,
                "candidate_lane": repaired_candidate_lane,
                "notes": repaired_notes,
                "accepted": bool(decision["accepted"]),
                "added_candidate_ids": sorted(
                    str(candidate["candidate_id"])
                    for candidate in repaired
                    if str(candidate["candidate_id"]) not in current_ids
                ),
                "displaced_candidate_ids": sorted(
                    str(candidate["candidate_id"])
                    for candidate in current
                    if str(candidate["candidate_id"])
                    not in {str(item["candidate_id"]) for item in repaired}
                ),
            },
        },
    }
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "identity": dict(row["identity"]),
        "truth_accessed": False,
        "truth_used": False,
        "multihead": multihead_diagnostics,
        "sparse_repair": decision,
        "candidate_stem_features": {
            candidate_id: dict(stem_features[candidate_id])
            for candidate_id in sorted(stem_features)
        },
        "stem_feature_metadata": stem_metadata,
    }
    return paired, diagnostics, current, repaired


def _group_indices(
    row: Mapping[str, Any],
    *,
    selector: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    x_radius = recovery._staff_spacing(row) * float(selector["nms_x_spaces"])
    groups = recovery._horizontal_candidate_groups(candidates, x_radius)
    result = {
        str(candidate["candidate_id"]): group_index
        for group_index, group in enumerate(groups, start=1)
        for candidate in group
    }
    if len(result) != len(candidates):
        raise ValueError("Sparse-dyad onset grouping lost candidate identity")
    return result


def _candidate_record(
    candidate: Mapping[str, Any],
    *,
    onset_group_index: int,
    sparse_repair_added: bool,
) -> dict[str, Any]:
    source = candidate.get("source")
    bbox = source.get("bbox") if isinstance(source, Mapping) else candidate.get("bbox")
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "center": {
            "x": round(float(candidate["center"]["x"]), 3),
            "y": round(float(candidate["center"]["y"]), 3),
        },
        "score": float(candidate["score"]),
        "onset_group_index": int(onset_group_index),
        "sparse_repair_added": sparse_repair_added,
        **({"bbox": dict(bbox)} if isinstance(bbox, Mapping) else {}),
    }


def verify_paired_contract(
    paired_rows: Sequence[Mapping[str, Any]],
    generic_rows: Sequence[Mapping[str, Any]],
    *,
    selector: Mapping[str, Any],
    expected_target: Mapping[str, Any],
) -> dict[str, Any]:
    if len(paired_rows) != len(generic_rows) or len(paired_rows) != gate.EXPECTED_CROP_COUNT:
        raise ValueError("Sparse-dyad paired/generic row count mismatch")
    accepted = 0
    for actual, generic in zip(paired_rows, generic_rows, strict=True):
        expected, diagnostics, _, _ = _pair_row(
            generic,
            selector=selector,
            expected_target=expected_target,
        )
        if actual != expected:
            raise ValueError("Sparse-dyad paired prediction drifted from fixed rule")
        decision = diagnostics["sparse_repair"]
        if decision["accepted"]:
            accepted += 1
            chosen = decision.get("chosen_pair")
            if not isinstance(chosen, Mapping) or not chosen.get("augmentation_dot_pairs"):
                raise ValueError("Accepted sparse-dyad repair lacks augmentation-dot evidence")
            lane = actual["lanes"]["sparse_dyad_repair"]
            if len({int(item["onset_group_index"]) for item in lane["candidate_lane"]}) != 1:
                raise ValueError("Accepted sparse-dyad pair was not grouped as one onset")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "independent_sparse_dyad_repair_truth_blind_contract",
        "passed": True,
        "truth_accessed": False,
        "truth_used": False,
        "measure_count": len(paired_rows),
        "accepted_repair_count": accepted,
        "fixed_multihead_comparison_lane_reproduced": True,
        "fixed_sparse_repair_reproduced": True,
        "accepted_repairs_have_paired_augmentation_dot_evidence": True,
        "accepted_repairs_form_one_onset_group": True,
    }


def freeze_paired_repair(
    prepared_manifest_path: Path,
    *,
    model_dir: Path,
    inference_dir: Path,
    pair_dir: Path,
) -> dict[str, Any]:
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
    paired_rows = inference._read_jsonl(pair_dir / "paired_predictions.jsonl")
    generic_rows = inference._read_jsonl(inference_dir / "inference.jsonl")
    selector = recovery.selector_config_from_model(validated["model"])
    contract = verify_paired_contract(
        paired_rows,
        generic_rows,
        selector=selector,
        expected_target=prepared["target"],
    )
    if inference._read_json(pair_dir / "truth_blind_contract.json") != contract:
        raise ValueError("Sparse-dyad truth-blind contract artifact drift")
    if (
        pair_manifest.get("target") != prepared["target"]
        or pair_manifest.get("version") != PAIR_VERSION
        or int(pair_manifest.get("accepted_repair_count", -1))
        != int(contract["accepted_repair_count"])
    ):
        raise ValueError("Sparse-dyad paired manifest target/version/count mismatch")

    output_dir = namespace_root / FROZEN_DIRNAME
    temp_dir = namespace_root / f".{FROZEN_DIRNAME}.tmp"
    if output_dir.exists():
        raise FileExistsError(f"Sparse-dyad freeze already exists: {output_dir}")
    if temp_dir.exists():
        raise FileExistsError(f"Stale sparse-dyad freeze exists: {temp_dir}")

    source_system = base._find_out_dir(namespace_root) / str(
        prepared["artifacts"]["source_system"]["path_relative_to_out"]
    )
    metadata_path = inference._resolve_record_path(validated["metadata_record"])
    prepared_paths = [
        metadata_path,
        *multihead._prepared_artifact_paths(prepared_manifest_path, prepared),
    ]
    inference_paths = sorted(
        path for path in inference_dir.rglob("*") if path.is_file() and pair_dir not in path.parents
    )
    pair_paths = sorted(path for path in pair_dir.rglob("*") if path.is_file())
    implementation_paths = [
        Path(__file__),
        Path(gate.__file__),
        Path(base.__file__),
        Path(inference.__file__),
        Path(multihead.__file__),
        Path(recovery.__file__),
        Path(repair.__file__),
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
        "baseline_inference": inference_paths,
        "model_and_training": model_and_training_paths,
        "implementations": implementation_paths,
    }
    for paths in groups.values():
        for path in multihead._deduplicate(paths):
            base._validate_external_artifact(path, target_slug=gate.DESDE_LEJOS_SLUG)

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
                for index, path in enumerate(multihead._deduplicate(paths), start=1)
            ]
            for role, paths in groups.items()
        }
        freeze_path = temp_dir / "freeze.json"
        inference._write_json(
            freeze_path,
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "independent_sparse_dyad_repair_freeze",
                "status": "frozen_awaiting_truth",
                "truth_accessed": False,
                "truth_used": False,
                "target": prepared["target"],
                "comparison_config_id": recovery.EDGE_SAFE_STEM_MULTIHEAD_CONFIG_ID,
                "repair_config_id": repair.CONFIG_ID,
                "repair_parameters": dict(repair.PARAMETERS),
                "accepted_repair_count": int(contract["accepted_repair_count"]),
                "paired_manifest_sha256": inference._sha256(pair_dir / "manifest.json"),
                "prepared_manifest_sha256": inference._sha256(prepared_manifest_path),
                "source_system_sha256": inference._sha256(source_system),
                "truth_blind_contract_sha256": inference._sha256(
                    pair_dir / "truth_blind_contract.json"
                ),
                "pins": pins,
            },
        )
        sealed_path = temp_dir / "sealed_manifest.json"
        inference._write_json(
            sealed_path,
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "independent_sparse_dyad_repair_sealed_manifest",
                "status": "frozen_awaiting_truth",
                "truth_accessed": False,
                "truth_used": False,
                "target": prepared["target"],
                "freeze": {"path": "freeze.json", "sha256": inference._sha256(freeze_path)},
                "next_gate": (
                    "verify all hashes, then collect MusicXML plus raw-image-only head/dot review"
                ),
            },
        )
        temp_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    verify_freeze(output_dir)
    return {
        "freeze": str(output_dir / "freeze.json"),
        "freeze_sha256": inference._sha256(output_dir / "freeze.json"),
        "sealed_manifest": str(output_dir / "sealed_manifest.json"),
        "sealed_manifest_sha256": inference._sha256(output_dir / "sealed_manifest.json"),
        "accepted_repair_count": int(contract["accepted_repair_count"]),
    }


def verify_freeze(output_dir: Path) -> None:
    freeze = inference._read_json(output_dir / "freeze.json")
    sealed = inference._read_json(output_dir / "sealed_manifest.json")
    if freeze.get("status") != "frozen_awaiting_truth" or freeze.get("truth_accessed") is not False:
        raise ValueError("Sparse-dyad freeze is not truth-blind/frozen")
    if sealed.get("status") != "frozen_awaiting_truth" or sealed.get("truth_accessed") is not False:
        raise ValueError("Sparse-dyad seal is not truth-blind/frozen")
    if inference._sha256(output_dir / "freeze.json") != str(sealed["freeze"]["sha256"]):
        raise ValueError("Sparse-dyad sealed freeze hash drift")
    if (
        freeze.get("repair_config_id") != repair.CONFIG_ID
        or freeze.get("repair_parameters") != repair.PARAMETERS
    ):
        raise ValueError("Sparse-dyad frozen parameter drift")
    pins = freeze.get("pins")
    required_roles = {
        "paired_predictions",
        "paired_artifacts",
        "prepared_and_source",
        "baseline_inference",
        "model_and_training",
        "implementations",
    }
    if not isinstance(pins, Mapping) or set(pins) != required_roles:
        raise ValueError("Sparse-dyad freeze provenance roles drift")
    for role in sorted(pins):
        if not pins[role]:
            raise ValueError(f"Sparse-dyad freeze provenance role is empty: {role}")
        for pin in pins[role]:
            snapshot = output_dir.parent / str(pin["snapshot_path_relative_to_namespace"])
            if inference._sha256(snapshot) != str(pin["snapshot_sha256"]):
                raise ValueError(f"Sparse-dyad frozen snapshot hash drift: {snapshot}")


def _validate_prepared_gate(prepared_path: Path, prepared: Mapping[str, Any]) -> None:
    if prepared.get("kind") != gate.INDEPENDENT_SPARSE_DYAD_REPAIR_GATE.prepare_kind:
        raise ValueError(f"Not an independent sparse-dyad prepared manifest: {prepared_path}")
    if prepared.get("target") != {
        "slug": gate.DESDE_LEJOS_SLUG,
        "system_index": gate.TARGET_SYSTEM_INDEX,
    }:
        raise ValueError("Independent sparse-dyad prepared target drift")
    if prepared.get("truth_accessed") is not False:
        raise ValueError("Independent sparse-dyad prepared artifact is not truth-blind")
    if prepared["independent_sparse_dyad_repair_gate"].get("truth_used") is not False:
        raise ValueError("Independent sparse-dyad gate declaration is not truth-blind")
    if len(prepared["artifacts"]["crops"]) != gate.EXPECTED_CROP_COUNT:
        raise ValueError("Independent sparse-dyad prepared crop-count drift")
    base._verify_prepared_manifest(
        prepared_path.parent,
        prepared_path,
        prepared,
        expected_kind=gate.INDEPENDENT_SPARSE_DYAD_REPAIR_GATE.prepare_kind,
    )


if __name__ == "__main__":
    raise SystemExit(main())
