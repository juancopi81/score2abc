"""Materialize the opt-in duration-aware repaired full-event sidecar v2.

This is additive to v1. It consumes the same hash-pinned candidate lanes and
also verifies the sparse-repair diagnostics used by the bounded dotted-half
duration classifier. Canonical inference and v1 artifacts remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import compose_repaired_candidate_events as base_compositor  # noqa: E402
from scripts.experiments import compose_repaired_candidate_events_v2 as compositor  # noqa: E402
from scripts.experiments import materialize_repaired_full_event_sidecar as v1  # noqa: E402
from scripts.experiments import sparse_dyad_duration_evidence as duration_evidence  # noqa: E402

SCHEMA_VERSION = 1
SIDECAR_VERSION = "repaired-full-event-sidecar-v2"
DEFAULT_OUTPUT_DIRNAME = "repaired_full_event_v2"
MANIFEST_KIND = "repaired_full_event_sidecar_v2_manifest"
CONFIG_KIND = "repaired_full_event_sidecar_v2_configuration"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inference_dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output-dirname", default=DEFAULT_OUTPUT_DIRNAME)
    args = parser.parse_args(argv)
    try:
        result = materialize_repaired_full_event_sidecar_v2(
            args.inference_dir,
            model_dir=args.model_dir,
            output_dirname=args.output_dirname,
        )
    except (FileExistsError, FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def materialize_repaired_full_event_sidecar_v2(
    inference_dir: Path,
    *,
    model_dir: Path | None = None,
    output_dirname: str = DEFAULT_OUTPUT_DIRNAME,
) -> dict[str, Any]:
    """Create the duration-aware full-event sidecar exactly once."""
    v1.runner._validate_output_name(output_dirname)
    inference_dir = inference_dir.resolve()
    v1.runner._reject_truth_path(inference_dir)
    output_dir = inference_dir / output_dirname
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite create-once sidecar: {output_dir}")
    temp_dir = output_dir.with_name(f".{output_dir.name}.tmp")
    if temp_dir.exists():
        raise FileExistsError(f"Refusing stale temporary sidecar: {temp_dir}")

    validated = _validate_inputs(inference_dir, model_dir=model_dir)
    canonical_before = v1._canonical_bytes(inference_dir)
    existing_v1_before = _existing_v1_bytes(inference_dir)
    predictions, details, diagnostics, applied_count = _compose_all(validated)
    config = _expected_config(validated)

    temp_dir.mkdir()
    try:
        v1._write_json(temp_dir / "config.json", config)
        v1._write_jsonl(temp_dir / "predictions.jsonl", predictions)
        v1._write_jsonl(temp_dir / "inference.jsonl", details)
        v1._write_jsonl(temp_dir / "diagnostics.jsonl", diagnostics)

        overlay_dir = temp_dir / "overlays"
        overlay_dir.mkdir()
        overlay_paths = []
        seen_measure_indices: set[int] = set()
        for request, repair_row in zip(
            validated["requests"], validated["repair_rows"], strict=True
        ):
            measure_index = v1._measure_index(request["identity"])
            if measure_index in seen_measure_indices:
                raise ValueError(f"Duplicate automatic measure index: {measure_index}")
            seen_measure_indices.add(measure_index)
            overlay_path = overlay_dir / f"measure_{measure_index:03d}.png"
            v1._write_repaired_overlay(
                request,
                v1._sparse_candidate_lane(repair_row),
                out_dir=validated["out_dir"],
                output_path=overlay_path,
            )
            overlay_paths.append(overlay_path)
        v1.runner._write_contact_sheet(overlay_paths, temp_dir / "contact_sheet.png")

        manifest = _expected_manifest(
            validated,
            output_dir=temp_dir,
            measure_count=len(predictions),
            duration_evidence_applied_count=applied_count,
        )
        v1._write_json(temp_dir / "manifest.json", manifest)
        if v1._canonical_bytes(inference_dir) != canonical_before:
            raise ValueError("Canonical inference bytes changed during sidecar materialization")
        if _existing_v1_bytes(inference_dir) != existing_v1_before:
            raise ValueError(
                "Existing repaired full-event v1 bytes changed during v2 materialization"
            )
        verify_repaired_full_event_sidecar_v2(temp_dir, model_dir=validated["model_dir"])
        temp_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return {
        "output_dir": str(output_dir),
        "manifest": str(output_dir / "manifest.json"),
        "manifest_sha256": v1._sha256(output_dir / "manifest.json"),
        "measure_count": len(predictions),
        "duration_evidence_applied_count": applied_count,
        "truth_used": False,
        "freeze_supported": False,
    }


def verify_repaired_full_event_sidecar_v2(
    output_dir: Path,
    model_dir: Path | None = None,
) -> dict[str, Any]:
    """Verify every v2 pin and deterministically recompose every row."""
    output_dir = output_dir.resolve()
    v1.runner._reject_truth_path(output_dir)
    manifest = v1._read_json(output_dir / "manifest.json")
    _verify_manifest_contract(manifest)

    validated = _validate_inputs(output_dir.parent, model_dir=model_dir)
    predictions, details, diagnostics, applied_count = _compose_all(validated)
    expected_manifest = _expected_manifest(
        validated,
        output_dir=output_dir,
        measure_count=len(predictions),
        duration_evidence_applied_count=applied_count,
    )
    if manifest != expected_manifest:
        raise ValueError("Repaired full-event v2 manifest drift")
    v1._verify_artifact_inventory(output_dir, manifest["artifacts"])
    if v1._read_json(output_dir / "config.json") != _expected_config(validated):
        raise ValueError("Repaired full-event v2 configuration drift")
    if v1._read_jsonl(output_dir / "predictions.jsonl") != predictions:
        raise ValueError("Repaired full-event v2 predictions drifted from deterministic replay")
    if v1._read_jsonl(output_dir / "inference.jsonl") != details:
        raise ValueError("Repaired full-event v2 inference drifted from deterministic replay")
    if v1._read_jsonl(output_dir / "diagnostics.jsonl") != diagnostics:
        raise ValueError("Repaired full-event v2 diagnostics drifted from deterministic replay")
    return {
        "output_dir": str(output_dir),
        "measure_count": len(predictions),
        "duration_evidence_applied_count": applied_count,
        "verified": True,
        "truth_used": False,
    }


def _validate_inputs(inference_dir: Path, *, model_dir: Path | None) -> dict[str, Any]:
    validated = dict(v1._validate_inputs(inference_dir, model_dir=model_dir))
    diagnostics_path = validated["sparse_dir"] / "diagnostics.jsonl"
    diagnostic_rows = v1._read_jsonl(diagnostics_path)
    if len(diagnostic_rows) != len(validated["requests"]):
        raise ValueError("Request/sparse diagnostic row count mismatch")
    decisions = []
    for request, repair_row, diagnostic_row in zip(
        validated["requests"],
        validated["repair_rows"],
        diagnostic_rows,
        strict=True,
    ):
        identity = request["identity"]
        if repair_row.get("identity") != identity or diagnostic_row.get("identity") != identity:
            raise ValueError("Request/repair/sparse diagnostic identity mismatch")
        if (
            diagnostic_row.get("truth_accessed") is not False
            or diagnostic_row.get("truth_used") is not False
        ):
            raise ValueError("Sparse diagnostic row is not explicitly truth-blind")
        decision = diagnostic_row.get("sparse_repair")
        if not isinstance(decision, Mapping):
            raise ValueError("Sparse diagnostic row has no repair decision")
        decisions.append(dict(decision))
    validated["sparse_diagnostics_path"] = diagnostics_path
    validated["sparse_diagnostic_rows"] = diagnostic_rows
    validated["sparse_repair_decisions"] = decisions
    return validated


def _compose_all(
    validated: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int]:
    predictions = []
    details = []
    diagnostics = []
    applied_count = 0
    for request, inference_row, repair_row, decision in zip(
        validated["requests"],
        validated["inference_rows"],
        validated["repair_rows"],
        validated["sparse_repair_decisions"],
        strict=True,
    ):
        lane = v1._sparse_candidate_lane(repair_row)
        result = compositor.compose_repaired_candidate_events_v2(
            request,
            inference_row,
            lane,
            decision,
            pitch_predictor=validated["pitch_predictor"],
            out_dir=validated["out_dir"],
            selector_method_id=validated["selector_method_id"],
        )
        evidence = result.get("duration_evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError("V2 compositor omitted duration evidence diagnostics")
        applied_count += int(evidence.get("applied") is True)
        if (
            result.get("status") != base_compositor.STATUS_MATERIALIZED
            or not isinstance(result.get("prediction"), Mapping)
            or result.get("meter_valid") is not True
        ):
            raise ValueError(
                "Full-event v2 composition failed closed: every row requires materialized "
                "events and valid meter"
            )
        if result.get("identity") != request.get("identity"):
            raise ValueError("V2 compositor output identity mismatch")
        prediction = dict(result["prediction"])
        if prediction.get("identity") != request.get("identity"):
            raise ValueError("V2 composed prediction identity mismatch")
        diagnostic = v1._diagnostic_row(result, lane)
        diagnostic["duration_evidence"] = dict(evidence)
        if "v1_base_composition" in result:
            diagnostic["v1_base_composition"] = dict(result["v1_base_composition"])
        predictions.append(prediction)
        details.append(dict(result))
        diagnostics.append(diagnostic)
    return predictions, details, diagnostics, applied_count


def _expected_config(validated: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CONFIG_KIND,
        "version": SIDECAR_VERSION,
        "selector_method_id": validated["selector_method_id"],
        "candidate_source": "sparse_stem_dyad_repair_v1",
        "composition": compositor.COMPOSITOR_VERSION,
        "duration_evidence": duration_evidence.EVIDENCE_KIND,
        "truth_accessed": False,
        "truth_used": False,
        "freeze_supported": False,
    }


def _expected_manifest(
    validated: Mapping[str, Any],
    *,
    output_dir: Path,
    measure_count: int,
    duration_evidence_applied_count: int,
) -> dict[str, Any]:
    inference_dir = validated["inference_dir"]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "version": SIDECAR_VERSION,
        "status": "spike_only_full_events_v2_materialized",
        "truth_accessed": False,
        "truth_used": False,
        "create_once": True,
        "freeze_supported": False,
        "target": dict(validated["main"]["target"]),
        "measure_count": measure_count,
        "duration_evidence_applied_count": duration_evidence_applied_count,
        "contract": {
            "canonical_bytes_unchanged": True,
            "existing_v1_artifacts_unchanged": True,
            "exact_sparse_candidate_lane_consumed": True,
            "candidate_reranking_applied": False,
            "duration_override_scope": duration_evidence.EVIDENCE_KIND,
            "full_events_materialized": True,
            "all_measures_meter_valid": True,
            "truth_used": False,
            "freeze_supported": False,
            "spike_only": True,
        },
        "canonical": {
            "main_manifest": v1._relative_record(output_dir, inference_dir / "manifest.json"),
            "requests": v1._relative_record(output_dir, inference_dir / "requests.jsonl"),
            "predictions": v1._relative_record(output_dir, inference_dir / "predictions.jsonl"),
            "inference": v1._relative_record(output_dir, inference_dir / "inference.jsonl"),
        },
        "upstream": {
            "multihead_manifest": v1._relative_record(
                output_dir, validated["multihead_dir"] / "manifest.json"
            ),
            "multihead_lane": v1._relative_record(
                output_dir, validated["multihead_dir"] / "recovery_lane.jsonl"
            ),
            "sparse_manifest": v1._relative_record(
                output_dir, validated["sparse_dir"] / "manifest.json"
            ),
            "sparse_lane": v1._relative_record(
                output_dir, validated["sparse_dir"] / "repair_lane.jsonl"
            ),
            "sparse_diagnostics": v1._relative_record(
                output_dir, validated["sparse_diagnostics_path"]
            ),
        },
        "model_and_training": dict(validated["pins"]),
        "implementation": v1._file_record(Path(__file__)),
        "v1_materializer_implementation": v1._file_record(Path(v1.__file__)),
        "v2_compositor_implementation": v1._file_record(Path(compositor.__file__)),
        "v1_compositor_implementation": v1._file_record(Path(base_compositor.__file__)),
        "duration_evidence_implementation": v1._file_record(Path(duration_evidence.__file__)),
        "artifacts": v1._artifact_records(output_dir),
    }


def _verify_manifest_contract(manifest: Mapping[str, Any]) -> None:
    if (
        manifest.get("kind") != MANIFEST_KIND
        or manifest.get("version") != SIDECAR_VERSION
        or manifest.get("status") != "spike_only_full_events_v2_materialized"
        or manifest.get("truth_accessed") is not False
        or manifest.get("truth_used") is not False
        or manifest.get("create_once") is not True
        or manifest.get("freeze_supported") is not False
    ):
        raise ValueError("Repaired full-event v2 manifest contract mismatch")


def _existing_v1_bytes(inference_dir: Path) -> dict[str, bytes]:
    v1_dir = inference_dir / v1.DEFAULT_OUTPUT_DIRNAME
    if not v1_dir.exists():
        return {}
    if not v1_dir.is_dir():
        raise ValueError(f"Existing v1 sidecar path is not a directory: {v1_dir}")
    return {
        path.relative_to(v1_dir).as_posix(): path.read_bytes()
        for path in sorted(v1_dir.rglob("*"))
        if path.is_file()
    }


if __name__ == "__main__":
    raise SystemExit(main())
