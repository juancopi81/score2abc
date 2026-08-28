"""Materialize a truth-blind full-event sidecar from the fixed repaired lane.

The command consumes an existing no-freeze inference with both optional
candidate-recovery sidecars. It never mutates or freezes canonical inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import compose_repaired_candidate_events as compositor  # noqa: E402
from scripts.experiments import run_third_score_heldout_inference as runner  # noqa: E402

SCHEMA_VERSION = 1
SIDECAR_VERSION = "repaired-full-event-sidecar-v1"
DEFAULT_OUTPUT_DIRNAME = "repaired_full_event_v1"
MANIFEST_KIND = "repaired_full_event_sidecar_manifest"
CONFIG_KIND = "repaired_full_event_sidecar_configuration"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inference_dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output-dirname", default=DEFAULT_OUTPUT_DIRNAME)
    args = parser.parse_args(argv)
    try:
        result = materialize_repaired_full_event_sidecar(
            args.inference_dir,
            model_dir=args.model_dir,
            output_dirname=args.output_dirname,
        )
    except (FileExistsError, FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def materialize_repaired_full_event_sidecar(
    inference_dir: Path,
    *,
    model_dir: Path | None = None,
    output_dirname: str = DEFAULT_OUTPUT_DIRNAME,
) -> dict[str, Any]:
    """Create the full-event sidecar exactly once after all rows pass."""
    runner._validate_output_name(output_dirname)
    inference_dir = inference_dir.resolve()
    runner._reject_truth_path(inference_dir)
    output_dir = inference_dir / output_dirname
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite create-once sidecar: {output_dir}")
    temp_dir = output_dir.with_name(f".{output_dir.name}.tmp")
    if temp_dir.exists():
        raise FileExistsError(f"Refusing stale temporary sidecar: {temp_dir}")

    validated = _validate_inputs(inference_dir, model_dir=model_dir)
    canonical_before = _canonical_bytes(inference_dir)
    predictions, details, diagnostics = _compose_all(validated)
    config = _expected_config(validated)

    temp_dir.mkdir()
    try:
        _write_json(temp_dir / "config.json", config)
        _write_jsonl(temp_dir / "predictions.jsonl", predictions)
        _write_jsonl(temp_dir / "inference.jsonl", details)
        _write_jsonl(temp_dir / "diagnostics.jsonl", diagnostics)

        overlay_dir = temp_dir / "overlays"
        overlay_dir.mkdir()
        overlay_paths = []
        seen_measure_indices: set[int] = set()
        for request, repair_row in zip(
            validated["requests"], validated["repair_rows"], strict=True
        ):
            measure_index = _measure_index(request["identity"])
            if measure_index in seen_measure_indices:
                raise ValueError(f"Duplicate automatic measure index: {measure_index}")
            seen_measure_indices.add(measure_index)
            overlay_path = overlay_dir / f"measure_{measure_index:03d}.png"
            _write_repaired_overlay(
                request,
                _sparse_candidate_lane(repair_row),
                out_dir=validated["out_dir"],
                output_path=overlay_path,
            )
            overlay_paths.append(overlay_path)
        runner._write_contact_sheet(overlay_paths, temp_dir / "contact_sheet.png")

        manifest = _expected_manifest(
            validated,
            output_dir=temp_dir,
            measure_count=len(predictions),
        )
        _write_json(temp_dir / "manifest.json", manifest)
        if _canonical_bytes(inference_dir) != canonical_before:
            raise ValueError("Canonical inference bytes changed during sidecar materialization")
        verify_repaired_full_event_sidecar(temp_dir, model_dir=validated["model_dir"])
        temp_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return {
        "output_dir": str(output_dir),
        "manifest": str(output_dir / "manifest.json"),
        "manifest_sha256": _sha256(output_dir / "manifest.json"),
        "measure_count": len(predictions),
        "truth_used": False,
        "freeze_supported": False,
    }


def verify_repaired_full_event_sidecar(
    output_dir: Path,
    model_dir: Path | None = None,
) -> dict[str, Any]:
    """Verify every pin and deterministically recompose every output row."""
    output_dir = output_dir.resolve()
    runner._reject_truth_path(output_dir)
    manifest = _read_json(output_dir / "manifest.json")
    _verify_manifest_contract(manifest)

    inference_dir = output_dir.parent
    validated = _validate_inputs(inference_dir, model_dir=model_dir)
    expected_manifest = _expected_manifest(
        validated,
        output_dir=output_dir,
        measure_count=len(validated["requests"]),
    )
    if manifest != expected_manifest:
        raise ValueError("Repaired full-event manifest drift")
    _verify_artifact_inventory(output_dir, manifest["artifacts"])

    expected_config = _expected_config(validated)
    if _read_json(output_dir / "config.json") != expected_config:
        raise ValueError("Repaired full-event configuration drift")
    expected_predictions, expected_details, expected_diagnostics = _compose_all(validated)
    if _read_jsonl(output_dir / "predictions.jsonl") != expected_predictions:
        raise ValueError("Repaired full-event predictions drifted from deterministic replay")
    if _read_jsonl(output_dir / "inference.jsonl") != expected_details:
        raise ValueError("Repaired full-event inference drifted from deterministic replay")
    if _read_jsonl(output_dir / "diagnostics.jsonl") != expected_diagnostics:
        raise ValueError("Repaired full-event diagnostics drifted from deterministic replay")
    return {
        "output_dir": str(output_dir),
        "measure_count": len(expected_predictions),
        "verified": True,
        "truth_used": False,
    }


def _validate_inputs(inference_dir: Path, *, model_dir: Path | None) -> dict[str, Any]:
    main_path = inference_dir / "manifest.json"
    main = _read_json(main_path)
    if (
        main.get("status") != "inferred_spike_only_no_freeze"
        or main.get("create_once") is not True
        or main.get("truth_accessed") is not False
        or main.get("truth_used") is not False
    ):
        raise ValueError("Input must be an existing truth-blind no-freeze inference")

    optional_lanes = main.get("optional_lanes")
    if not isinstance(optional_lanes, Mapping):
        raise ValueError("Input inference has no optional recovery lanes")
    expected_optional = {
        "edge_safe_stem_multihead_recovery": (
            runner.MULTIHEAD_RECOVERY_DIRNAME,
            "Multi-head recovery",
        ),
        "sparse_stem_dyad_repair": (
            runner.SPARSE_DYAD_REPAIR_DIRNAME,
            "Sparse-dyad repair",
        ),
    }
    for name, (dirname, label) in expected_optional.items():
        record = optional_lanes.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"Input inference is missing {label} optional lane")
        _verify_relative_record(
            record,
            inference_dir,
            f"{dirname}/manifest.json",
            label=f"{label} optional manifest",
        )

    canonical_names = ("requests.jsonl", "predictions.jsonl", "inference.jsonl")
    artifacts = main.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Input inference artifact manifest is missing")
    for name in canonical_names:
        record = artifacts.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"Input inference is missing canonical artifact: {name}")
        _verify_relative_record(record, inference_dir, name, label=f"Canonical {name}")

    multihead_dir = inference_dir / runner.MULTIHEAD_RECOVERY_DIRNAME
    sparse_dir = inference_dir / runner.SPARSE_DYAD_REPAIR_DIRNAME
    runner._verify_multihead_recovery_sidecar(multihead_dir)
    runner._verify_sparse_dyad_repair_sidecar(sparse_dir)
    multihead_manifest = _read_json(multihead_dir / "manifest.json")
    sparse_manifest = _read_json(sparse_dir / "manifest.json")
    pins = sparse_manifest.get("model_and_training")
    if not isinstance(pins, Mapping):
        raise ValueError("Sparse sidecar has no selected model/training pins")
    if main.get("model_and_training") != pins:
        raise ValueError("Main inference model/training pins differ from sparse sidecar")
    if multihead_manifest.get("model_and_training") != pins:
        raise ValueError("Multi-head and sparse model/training pins differ")

    resolved_model_dir, model_payload, pitch_predictor = _verify_model_pins(
        pins,
        model_dir=model_dir,
    )
    requests = _read_jsonl(inference_dir / "requests.jsonl")
    inference_rows = _read_jsonl(inference_dir / "inference.jsonl")
    repair_rows = _read_jsonl(sparse_dir / "repair_lane.jsonl")
    expected_count = int(main.get("output_count", -1))
    if not (
        expected_count > 0
        and len(requests) == len(inference_rows) == len(repair_rows) == expected_count
    ):
        raise ValueError("Request/inference/sparse lane row count mismatch")
    for request, inference_row, repair_row in zip(
        requests, inference_rows, repair_rows, strict=True
    ):
        identity = request.get("identity")
        if not isinstance(identity, Mapping):
            raise ValueError("Request identity is missing")
        if inference_row.get("identity") != identity or repair_row.get("identity") != identity:
            raise ValueError("Request/inference/sparse lane identity mismatch")
        if repair_row.get("source") != inference_row.get("source"):
            raise ValueError("Inference/sparse lane source mismatch")
        if (
            request.get("truth_accessed", False) is not False
            or inference_row.get("truth_used") is not False
            or repair_row.get("truth_used") is not False
        ):
            raise ValueError("Input rows are not explicitly truth-blind")
        target = main.get("target", {})
        if str(identity.get("slug")) != str(target.get("slug")) or int(
            identity.get("system_index", -1)
        ) != int(target.get("system_index", -2)):
            raise ValueError("Input row target substitution")
        _sparse_candidate_lane(repair_row)

    return {
        "inference_dir": inference_dir,
        "main_manifest_path": main_path,
        "main": main,
        "multihead_dir": multihead_dir,
        "multihead_manifest": multihead_manifest,
        "sparse_dir": sparse_dir,
        "sparse_manifest": sparse_manifest,
        "pins": dict(pins),
        "model_dir": resolved_model_dir,
        "model_payload": model_payload,
        "pitch_predictor": pitch_predictor,
        "selector_method_id": str(model_payload["replay"]["method"]["method_id"]),
        "requests": requests,
        "inference_rows": inference_rows,
        "repair_rows": repair_rows,
        "out_dir": runner.freezer._find_out_dir(inference_dir.parent),
    }


def _verify_model_pins(
    pins: Mapping[str, Any],
    *,
    model_dir: Path | None,
) -> tuple[Path, dict[str, Any], Any]:
    model_manifest_record = pins.get("model_manifest")
    implementation_record = pins.get("implementation")
    artifact_records = pins.get("artifacts")
    if not isinstance(model_manifest_record, Mapping):
        raise ValueError("Model manifest pin is missing")
    if not isinstance(implementation_record, Mapping) or not isinstance(artifact_records, Mapping):
        raise ValueError("Model/training pins are incomplete")
    pinned_manifest_path = runner._resolve_record_path(model_manifest_record).resolve()
    resolved_model_dir = (
        model_dir.resolve() if model_dir is not None else pinned_manifest_path.parent
    )
    runner._reject_truth_path(resolved_model_dir)
    if pinned_manifest_path != (resolved_model_dir / "manifest.json").resolve():
        raise ValueError("Supplied model directory does not match sparse sidecar pins")
    _verify_file_record(model_manifest_record, pinned_manifest_path, label="Model manifest")

    model_manifest = _read_json(pinned_manifest_path)
    if model_manifest.get("implementation") != implementation_record:
        raise ValueError("Pinned model implementation differs from model manifest")
    if model_manifest.get("artifacts") != artifact_records:
        raise ValueError("Pinned model artifacts differ from model manifest")
    _verify_file_record(
        implementation_record,
        runner._resolve_record_path(implementation_record),
        label="Model implementation",
    )
    for name, record in artifact_records.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"Invalid model artifact pin: {name}")
        expected = (resolved_model_dir / str(name)).resolve()
        _verify_file_record(record, expected, label=f"Model artifact {name}")
    model_record = artifact_records.get("model.json")
    if not isinstance(model_record, Mapping):
        raise ValueError("Pinned model.json is missing")
    model_payload = _read_json(resolved_model_dir / "model.json")
    _, pitch_predictor, _ = runner.reconstruct_model(model_payload)
    return resolved_model_dir, model_payload, pitch_predictor


def _compose_all(
    validated: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    predictions = []
    details = []
    diagnostics = []
    for request, inference_row, repair_row in zip(
        validated["requests"],
        validated["inference_rows"],
        validated["repair_rows"],
        strict=True,
    ):
        lane = _sparse_candidate_lane(repair_row)
        result = compositor.compose_repaired_candidate_events(
            request,
            inference_row,
            lane,
            pitch_predictor=validated["pitch_predictor"],
            out_dir=validated["out_dir"],
            selector_method_id=validated["selector_method_id"],
        )
        if (
            result.get("status") != compositor.STATUS_MATERIALIZED
            or not isinstance(result.get("prediction"), Mapping)
            or result.get("meter_valid") is not True
        ):
            raise ValueError(
                "Full-event composition failed closed: every row requires materialized "
                "events and valid meter"
            )
        if result.get("identity") != request.get("identity"):
            raise ValueError("Compositor output identity mismatch")
        prediction = dict(result["prediction"])
        if prediction.get("identity") != request.get("identity"):
            raise ValueError("Composed prediction identity mismatch")
        predictions.append(prediction)
        details.append(dict(result))
        diagnostics.append(_diagnostic_row(result, lane))
    return predictions, details, diagnostics


def _diagnostic_row(
    result: Mapping[str, Any],
    lane: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    groups = []
    for group in result["groups"]:
        groups.append(
            {
                "group_id": str(group["group_id"]),
                "onset_group_index": int(group["onset_group_index"]),
                "candidate_ids": [
                    str(anchor["source"]["candidate_id"]) for anchor in group["anchors"]
                ],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "identity": dict(result["identity"]),
        "truth_used": False,
        "candidate_ids": [str(row["candidate_id"]) for row in lane],
        "onset_groups": groups,
        "automatic_key_context": dict(result["automatic_key_context"]),
        "decoder": {
            "status": str(result["decoder_status"]),
            "expected_measure_beats": result["expected_measure_beats"],
            "observed_extent_beats": result["observed_extent_beats"],
            "decoded_extent_beats": result["decoded_extent_beats"],
            "meter_valid": result["meter_valid"],
        },
        "candidate_reranking_applied": False,
    }


def _expected_config(validated: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CONFIG_KIND,
        "version": SIDECAR_VERSION,
        "selector_method_id": validated["selector_method_id"],
        "candidate_source": "sparse_stem_dyad_repair_v1",
        "composition": "compose_repaired_candidate_events",
        "truth_accessed": False,
        "truth_used": False,
        "freeze_supported": False,
    }


def _expected_manifest(
    validated: Mapping[str, Any],
    *,
    output_dir: Path,
    measure_count: int,
) -> dict[str, Any]:
    inference_dir = validated["inference_dir"]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "version": SIDECAR_VERSION,
        "status": "spike_only_full_events_materialized",
        "truth_accessed": False,
        "truth_used": False,
        "create_once": True,
        "freeze_supported": False,
        "target": dict(validated["main"]["target"]),
        "measure_count": measure_count,
        "contract": {
            "canonical_bytes_unchanged": True,
            "exact_sparse_candidate_lane_consumed": True,
            "candidate_reranking_applied": False,
            "full_events_materialized": True,
            "all_measures_meter_valid": True,
            "truth_used": False,
            "freeze_supported": False,
            "spike_only": True,
        },
        "canonical": {
            "main_manifest": _relative_record(output_dir, inference_dir / "manifest.json"),
            "requests": _relative_record(output_dir, inference_dir / "requests.jsonl"),
            "predictions": _relative_record(output_dir, inference_dir / "predictions.jsonl"),
            "inference": _relative_record(output_dir, inference_dir / "inference.jsonl"),
        },
        "upstream": {
            "multihead_manifest": _relative_record(
                output_dir, validated["multihead_dir"] / "manifest.json"
            ),
            "multihead_lane": _relative_record(
                output_dir, validated["multihead_dir"] / "recovery_lane.jsonl"
            ),
            "sparse_manifest": _relative_record(
                output_dir, validated["sparse_dir"] / "manifest.json"
            ),
            "sparse_lane": _relative_record(
                output_dir, validated["sparse_dir"] / "repair_lane.jsonl"
            ),
        },
        "model_and_training": dict(validated["pins"]),
        "implementation": _file_record(Path(__file__)),
        "compositor_implementation": _file_record(Path(compositor.__file__)),
        "artifacts": _artifact_records(output_dir),
    }


def _verify_manifest_contract(manifest: Mapping[str, Any]) -> None:
    if (
        manifest.get("kind") != MANIFEST_KIND
        or manifest.get("version") != SIDECAR_VERSION
        or manifest.get("status") != "spike_only_full_events_materialized"
        or manifest.get("truth_accessed") is not False
        or manifest.get("truth_used") is not False
        or manifest.get("create_once") is not True
        or manifest.get("freeze_supported") is not False
    ):
        raise ValueError("Repaired full-event manifest contract mismatch")


def _sparse_candidate_lane(repair_row: Mapping[str, Any]) -> list[dict[str, Any]]:
    lanes = repair_row.get("lanes")
    sparse = lanes.get("sparse_dyad_repair") if isinstance(lanes, Mapping) else None
    lane = sparse.get("candidate_lane") if isinstance(sparse, Mapping) else None
    if not isinstance(lane, list) or not lane or not all(isinstance(row, dict) for row in lane):
        raise ValueError("Sparse repair row has no valid candidate_lane")
    return [dict(row) for row in lane]


def _write_repaired_overlay(
    request: Mapping[str, Any],
    lane: Sequence[Mapping[str, Any]],
    *,
    out_dir: Path,
    output_path: Path,
) -> None:
    image_path = compositor.composed._resolve_request_image(request, out_dir)
    if _sha256(image_path) != str(request["images"]["raw"]["sha256"]):
        raise ValueError(f"Repaired overlay source hash drift: {image_path}")
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = ("#0066cc", "#cc3300", "#008844", "#8a2be2", "#b36b00", "#008b8b")
    for row in lane:
        bbox = row.get("bbox")
        if not isinstance(bbox, Mapping):
            raise ValueError(f"Candidate overlay bbox is missing: {row.get('candidate_id')}")
        bounds = tuple(int(bbox[name]) for name in ("left", "top", "right", "bottom"))
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            raise ValueError(f"Candidate overlay bbox is invalid: {row.get('candidate_id')}")
        group_index = int(row["onset_group_index"])
        color = colors[(group_index - 1) % len(colors)]
        draw.rectangle(bounds, outline=color, width=2)
        label = f"{row['candidate_id']} g{group_index}"
        label_y = max(0, bounds[1] - 12)
        label_box = draw.textbbox((bounds[0], label_y), label, font=font)
        draw.rectangle(label_box, fill="white")
        draw.text((bounds[0], label_y), label, fill=color, font=font)
    image.save(output_path)


def _verify_artifact_inventory(output_dir: Path, records: Mapping[str, Any]) -> None:
    recorded = set(records)
    actual = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if recorded != actual:
        raise ValueError("Repaired full-event artifact inventory drift")
    for relative, record in records.items():
        if not isinstance(record, Mapping) or record.get("path") != relative:
            raise ValueError(f"Invalid repaired full-event artifact record: {relative}")
        path = output_dir / relative
        if _sha256(path) != str(record.get("sha256")):
            raise ValueError(f"Repaired full-event artifact hash drift: {path}")


def _artifact_records(root: Path) -> dict[str, dict[str, str]]:
    return {
        path.relative_to(root).as_posix(): {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def _canonical_bytes(inference_dir: Path) -> dict[str, bytes]:
    return {
        name: (inference_dir / name).read_bytes()
        for name in ("requests.jsonl", "predictions.jsonl", "inference.jsonl")
    }


def _measure_index(identity: Mapping[str, Any]) -> int:
    value = identity.get("automatic_measure_index")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("Automatic measure index must be a positive integer")
    return value


def _relative_record(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": Path(os.path.relpath(path.resolve(), root.resolve())).as_posix(),
        "sha256": _sha256(path),
    }


def _verify_relative_record(
    record: Mapping[str, Any],
    root: Path,
    expected_name: str,
    *,
    label: str,
) -> None:
    if str(record.get("path")) != expected_name:
        raise ValueError(f"{label} path substitution: expected {expected_name}")
    path = (root / expected_name).resolve()
    if _sha256(path) != str(record.get("sha256")):
        raise ValueError(f"{label} hash drift: {path}")


def _verify_file_record(record: Mapping[str, Any], path: Path, *, label: str) -> None:
    expected = path.resolve()
    recorded = runner._resolve_record_path(record).resolve()
    if recorded != expected:
        raise ValueError(f"{label} path substitution: expected {expected}, got {recorded}")
    if _sha256(expected) != str(record.get("sha256")):
        raise ValueError(f"{label} hash drift: {expected}")


def _read_json(path: Path) -> dict[str, Any]:
    runner._reject_truth_path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    runner._reject_truth_path(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
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


def _file_record(path: Path) -> dict[str, str]:
    path = path.resolve()
    try:
        display = path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        display = path.as_posix()
    return {"path": display, "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
