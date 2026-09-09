"""Seal the complete truth-blind Tio Climaco full-event prediction chain.

The seal is intentionally separate from inference. It accepts only the fixed
Tio Climaco gate after baseline inference, multi-head recovery, sparse-dyad
repair, and full-event composition all verify without target MusicXML.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import freeze_independent_full_event_gate as gate  # noqa: E402
from scripts.experiments import freeze_third_score_heldout as base  # noqa: E402
from scripts.experiments import materialize_repaired_full_event_sidecar as sidecar  # noqa: E402
from scripts.experiments import run_third_score_heldout_inference as runner  # noqa: E402

SCHEMA_VERSION = 1
SEAL_VERSION = "independent-full-event-seal-v1"
OUTPUT_DIRNAME = "frozen_full_event"
EXPECTED_TRANSCRIPTION_FILENAME = "tio_climaco_system_007.musicxml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sidecar_dir", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = seal_independent_full_event_gate(
            args.sidecar_dir,
            model_dir=args.model_dir,
        )
    except (FileExistsError, FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(result["sealed_manifest"])
    return 0


def seal_independent_full_event_gate(
    sidecar_dir: Path,
    *,
    model_dir: Path,
) -> dict[str, Any]:
    """Create the immutable pre-truth seal after all full-event lanes verify."""
    sidecar_dir = sidecar_dir.expanduser().resolve()
    model_dir = model_dir.expanduser().resolve()
    validated = _validate_upstream(sidecar_dir, model_dir=model_dir)
    namespace_root = validated["prepared_manifest_path"].parent
    _reject_existing_target_truth(namespace_root)

    output_dir = namespace_root / OUTPUT_DIRNAME
    temp_dir = namespace_root / f".{OUTPUT_DIRNAME}.tmp"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite full-event seal: {output_dir}")
    if temp_dir.exists():
        raise FileExistsError(f"Refusing stale full-event seal directory: {temp_dir}")

    temp_dir.mkdir(parents=False, exist_ok=False)
    try:
        snapshot_records = _snapshot_upstream(validated, temp_dir=temp_dir)
        freeze_path = temp_dir / "freeze.json"
        freeze = {
            "schema_version": SCHEMA_VERSION,
            "kind": "independent_full_event_freeze",
            "version": SEAL_VERSION,
            "status": "frozen_awaiting_truth",
            "create_once": True,
            "truth_accessed": False,
            "truth_used": False,
            "target": dict(validated["target"]),
            "contract": {
                "baseline_inference_frozen": True,
                "multihead_recovery_frozen": True,
                "sparse_dyad_repair_frozen": True,
                "repaired_full_events_frozen": True,
                "all_measures_meter_valid": True,
                "target_musicxml_forbidden_before_seal": True,
                "canonical_inference_unchanged": True,
                "spike_only": True,
            },
            "snapshots": snapshot_records,
            "implementation": runner._file_record(Path(__file__).resolve()),
        }
        base._write_json(freeze_path, freeze)

        sealed_path = temp_dir / "sealed_manifest.json"
        sealed = {
            "schema_version": SCHEMA_VERSION,
            "kind": "independent_full_event_sealed_manifest",
            "version": SEAL_VERSION,
            "status": "frozen_awaiting_truth",
            "create_once": True,
            "truth_accessed": False,
            "truth_used": False,
            "target": dict(validated["target"]),
            "freeze": {"path": "freeze.json", "sha256": base._sha256(freeze_path)},
            "next_truth_artifact": {
                "filename": EXPECTED_TRANSCRIPTION_FILENAME,
                "path_relative_to_namespace": EXPECTED_TRANSCRIPTION_FILENAME,
                "must_be_created_after_seal": True,
            },
        }
        base._write_json(sealed_path, sealed)
        temp_dir.rename(output_dir)
        verify_independent_full_event_gate(
            output_dir / "sealed_manifest.json",
            model_dir=model_dir,
        )
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        raise

    return {
        "freeze": str(output_dir / "freeze.json"),
        "freeze_sha256": base._sha256(output_dir / "freeze.json"),
        "sealed_manifest": str(output_dir / "sealed_manifest.json"),
        "sealed_manifest_sha256": base._sha256(output_dir / "sealed_manifest.json"),
        "truth_accessed": False,
        "truth_used": False,
    }


def verify_independent_full_event_gate(
    sealed_manifest_path: Path,
    *,
    model_dir: Path,
) -> dict[str, Any]:
    """Verify the seal, snapshots, and deterministic upstream replay."""
    sealed_manifest_path = sealed_manifest_path.expanduser().resolve()
    model_dir = model_dir.expanduser().resolve()
    if not sealed_manifest_path.is_file():
        raise FileNotFoundError(f"Sealed manifest does not exist: {sealed_manifest_path}")
    output_dir = sealed_manifest_path.parent
    namespace_root = output_dir.parent
    sealed = base._read_json(sealed_manifest_path)
    expected_target = {
        "slug": gate.TIO_CLIMACO_SLUG,
        "system_index": gate.TARGET_SYSTEM_INDEX,
    }
    if (
        sealed.get("kind") != "independent_full_event_sealed_manifest"
        or sealed.get("version") != SEAL_VERSION
        or sealed.get("status") != "frozen_awaiting_truth"
        or sealed.get("truth_accessed") is not False
        or sealed.get("truth_used") is not False
        or sealed.get("target") != expected_target
    ):
        raise ValueError("Independent full-event sealed-manifest contract mismatch")

    freeze_record = sealed.get("freeze")
    if not isinstance(freeze_record, Mapping) or freeze_record.get("path") != "freeze.json":
        raise ValueError("Independent full-event freeze record is invalid")
    freeze_path = output_dir / "freeze.json"
    if base._sha256(freeze_path) != str(freeze_record.get("sha256")):
        raise ValueError("Independent full-event freeze hash drift")
    freeze = base._read_json(freeze_path)
    if (
        freeze.get("kind") != "independent_full_event_freeze"
        or freeze.get("version") != SEAL_VERSION
        or freeze.get("status") != "frozen_awaiting_truth"
        or freeze.get("truth_accessed") is not False
        or freeze.get("truth_used") is not False
        or freeze.get("target") != expected_target
    ):
        raise ValueError("Independent full-event freeze contract mismatch")
    runner._verify_file_record(
        freeze["implementation"],
        Path(__file__).resolve(),
        label="Independent full-event sealer implementation",
    )

    source_paths = _verify_snapshots(freeze["snapshots"], output_dir=output_dir)
    prepared_path = source_paths["prepared_manifest"]
    sidecar_path = source_paths["repaired_full_event_manifest"].parent
    validated = _validate_upstream(sidecar_path, model_dir=model_dir)
    if validated["prepared_manifest_path"] != prepared_path:
        raise ValueError("Independent full-event prepared-manifest substitution")
    if validated["target"] != expected_target:
        raise ValueError("Independent full-event target substitution")

    transcription = sealed.get("next_truth_artifact")
    if (
        not isinstance(transcription, Mapping)
        or transcription.get("filename") != EXPECTED_TRANSCRIPTION_FILENAME
        or transcription.get("must_be_created_after_seal") is not True
    ):
        raise ValueError("Independent full-event next-truth contract mismatch")
    return {
        "namespace_root": namespace_root,
        "output_dir": output_dir,
        "target": expected_target,
        "prepared_manifest_path": prepared_path,
        "sidecar_dir": sidecar_path,
        "freeze_sha256": base._sha256(freeze_path),
        "sealed_sha256": base._sha256(sealed_manifest_path),
        "verified": True,
    }


def _validate_upstream(sidecar_dir: Path, *, model_dir: Path) -> dict[str, Any]:
    runner._reject_truth_path(sidecar_dir)
    runner._reject_truth_path(model_dir)
    sidecar.verify_repaired_full_event_sidecar(sidecar_dir, model_dir=model_dir)
    inference_dir = sidecar_dir.parent
    main_path = inference_dir / "manifest.json"
    main = base._read_json(main_path)
    prepared_record = main.get("prepared_manifest")
    if not isinstance(prepared_record, Mapping):
        raise ValueError("Full-event inference has no prepared-manifest pin")
    prepared_path = runner._resolve_record_path(prepared_record).resolve()
    prepared = base._read_json(prepared_path)
    if prepared.get("kind") != gate.INDEPENDENT_FULL_EVENT_GATE.prepare_kind:
        raise ValueError("Full-event inference does not belong to the Tio Climaco gate")
    target = prepared.get("target")
    expected_target = {
        "slug": gate.TIO_CLIMACO_SLUG,
        "system_index": gate.TARGET_SYSTEM_INDEX,
    }
    if target != expected_target or main.get("target") != expected_target:
        raise ValueError("Independent full-event target mismatch")
    runner._verify_inference_binding(
        prepared_path,
        model_dir=model_dir,
        inference_dir=inference_dir,
        manifest=main,
    )
    multihead_dir = inference_dir / runner.MULTIHEAD_RECOVERY_DIRNAME
    sparse_dir = inference_dir / runner.SPARSE_DYAD_REPAIR_DIRNAME
    runner._verify_multihead_recovery_sidecar(multihead_dir)
    runner._verify_sparse_dyad_repair_sidecar(sparse_dir)
    return {
        "target": expected_target,
        "prepared_manifest_path": prepared_path,
        "inference_manifest_path": main_path,
        "multihead_manifest_path": multihead_dir / "manifest.json",
        "sparse_manifest_path": sparse_dir / "manifest.json",
        "repaired_full_event_manifest_path": sidecar_dir / "manifest.json",
        "model_manifest_path": model_dir / "manifest.json",
    }


def _snapshot_upstream(validated: Mapping[str, Any], *, temp_dir: Path) -> dict[str, Any]:
    snapshots_dir = temp_dir / "snapshots"
    snapshots_dir.mkdir()
    records = {}
    for role in (
        "prepared_manifest",
        "inference_manifest",
        "multihead_manifest",
        "sparse_manifest",
        "repaired_full_event_manifest",
        "model_manifest",
    ):
        source = Path(validated[f"{role}_path"]).resolve()
        snapshot = snapshots_dir / f"{role}.json"
        shutil.copyfile(source, snapshot)
        if base._sha256(source) != base._sha256(snapshot):
            raise ValueError(f"Independent full-event snapshot copy mismatch: {role}")
        records[role] = {
            "source": runner._file_record(source),
            "snapshot_path": snapshot.relative_to(temp_dir).as_posix(),
            "snapshot_sha256": base._sha256(snapshot),
        }
    return records


def _verify_snapshots(
    records: Mapping[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Path]:
    expected_roles = {
        "prepared_manifest",
        "inference_manifest",
        "multihead_manifest",
        "sparse_manifest",
        "repaired_full_event_manifest",
        "model_manifest",
    }
    if set(records) != expected_roles:
        raise ValueError("Independent full-event snapshot role mismatch")
    paths = {}
    for role in sorted(expected_roles):
        record = records[role]
        if not isinstance(record, Mapping) or not isinstance(record.get("source"), Mapping):
            raise ValueError(f"Invalid independent full-event snapshot record: {role}")
        source = runner._resolve_record_path(record["source"]).resolve()
        runner._verify_file_record(record["source"], source, label=f"Full-event {role}")
        snapshot = output_dir / str(record.get("snapshot_path"))
        if base._sha256(snapshot) != str(record.get("snapshot_sha256")):
            raise ValueError(f"Independent full-event snapshot hash drift: {role}")
        if base._sha256(source) != base._sha256(snapshot):
            raise ValueError(f"Independent full-event source/snapshot drift: {role}")
        paths[role] = source
    return paths


def _reject_existing_target_truth(namespace_root: Path) -> None:
    forbidden = []
    for path in namespace_root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.casefold()
        if name.endswith((".musicxml", ".mxl")) or "truth" in name:
            forbidden.append(path)
    if forbidden:
        raise ValueError(
            "Target truth/MusicXML exists before the full-event seal: "
            + ", ".join(str(path) for path in sorted(forbidden))
        )


if __name__ == "__main__":
    raise SystemExit(main())
