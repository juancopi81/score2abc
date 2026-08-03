"""Create a truth-blind meter-deficit review sidecar for a sealed held-out gate.

The sidecar verifies the complete held-out freeze before loading any inference
inputs. It replays the existing visual meter-deficit observer and writes only
review flags and diagnostics; canonical predictions are never modified.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import evaluate_frozen_third_score_heldout as evaluator  # noqa: E402
from scripts.experiments import freeze_third_score_heldout as freezer  # noqa: E402
from scripts.experiments import run_third_score_heldout_inference as heldout  # noqa: E402
from scripts.experiments import spike_consumed_meter_deficit_validator as meter  # noqa: E402
from scripts.experiments import spike_consumed_onset_group_selector as onset  # noqa: E402
from scripts.experiments import spike_consumed_polyphonic_pitch_repair as recovery  # noqa: E402

SCHEMA_VERSION = 1
DEFAULT_OUTPUT_DIRNAME = "pretruth_meter_triage_v1"
OUTPUT_KIND = "frozen_heldout_truth_blind_meter_deficit_triage"
MANIFEST_KIND = "frozen_heldout_truth_blind_meter_deficit_triage_manifest"
SAFE_OUTPUT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
TRUTH_SUFFIXES = {".musicxml", ".mxl", ".xml"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sealed_manifest", type=Path)
    parser.add_argument(
        "--output-dirname",
        default=DEFAULT_OUTPUT_DIRNAME,
        help=(
            "Create-once sibling output directory under the held-out namespace "
            f"(default: {DEFAULT_OUTPUT_DIRNAME})."
        ),
    )
    args = parser.parse_args(argv)
    try:
        result = triage_frozen_heldout_meter_deficits(
            args.sealed_manifest,
            output_dirname=args.output_dirname,
        )
    except (FileExistsError, FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(result["report"])
    print(
        "flagged automatic measures: "
        + (
            ", ".join(str(value) for value in result["flagged_measure_indices"])
            if result["flagged_measure_indices"]
            else "none"
        )
    )
    return 0


def triage_frozen_heldout_meter_deficits(
    sealed_manifest_path: Path,
    *,
    output_dirname: str = DEFAULT_OUTPUT_DIRNAME,
) -> dict[str, Any]:
    """Verify a sealed gate and atomically publish non-mutating review flags."""
    _validate_output_dirname(output_dirname)
    sealed_manifest_path = sealed_manifest_path.expanduser().resolve()
    _reject_truth_looking_path(sealed_manifest_path)

    # This verifier must remain the first operation that opens held-out inputs.
    frozen = evaluator.verify_frozen_gate(sealed_manifest_path)
    namespace_root = Path(frozen["namespace_root"]).resolve()
    target = dict(frozen["target"])
    target_slug = str(target["slug"])
    output_dir = namespace_root / output_dirname
    temp_dir = namespace_root / f".{output_dirname}.tmp"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite create-once meter triage: {output_dir}")
    if temp_dir.exists():
        raise FileExistsError(f"Refusing stale temporary meter triage: {temp_dir}")

    inputs = _load_verified_inputs(frozen, target_slug=target_slug)
    canonical_predictions_path = inputs["paths"]["canonical_predictions"]
    canonical_predictions_sha256 = freezer._sha256(canonical_predictions_path)
    canonical_row_hashes = {
        crop: freezer._hash_json(row) for crop, row in sorted(frozen["predictions_by_crop"].items())
    }

    replay_model, _pitch_predictor, model_audit = heldout.reconstruct_model(inputs["model_payload"])
    selector = recovery.selector_config_from_model(inputs["model_payload"])
    replays = onset._holdout_replays(
        inputs["inference_rows"],
        request_rows=inputs["request_rows"],
        out_dir=inputs["out_dir"],
        model=replay_model,
        selector=selector,
    )
    requests_by_measure = {
        int(row["identity"]["system_measure_index"]): row for row in inputs["request_rows"]
    }
    predictions_by_measure = {
        int(row["identity"]["system_measure_index"]): row
        for row in inputs["canonical_prediction_rows"]
    }
    if {replay.measure_index for replay in replays} != set(requests_by_measure):
        raise ValueError("Replay and frozen request measure identities differ")
    if set(requests_by_measure) != set(predictions_by_measure):
        raise ValueError("Frozen request and canonical prediction measure identities differ")

    rows = []
    for replay in replays:
        measure_index = replay.measure_index
        observation = meter.observe_replay(
            replay,
            request=requests_by_measure[measure_index],
            metadata=inputs["metadata"],
        )
        if observation.get("truth_used") is not False:
            raise ValueError("Meter observer did not preserve truth_used=false")
        canonical = predictions_by_measure[measure_index]
        rows.append(
            {
                **observation,
                "schema_version": SCHEMA_VERSION,
                "canonical_prediction_sha256": freezer._hash_json(canonical),
                "canonical_prediction_mutated": False,
                "triage_only": True,
            }
        )
    rows.sort(key=lambda row: int(row["identity"]["system_measure_index"]))
    flagged = [int(row["identity"]["system_measure_index"]) for row in rows if row["review_flag"]]

    temp_dir.mkdir(parents=False, exist_ok=False)
    try:
        predictions_path = temp_dir / "predictions.jsonl"
        _write_jsonl(predictions_path, rows)

        overlays_dir = temp_dir / "overlays"
        overlays_dir.mkdir()
        rows_by_key = {str(row["key"]): row for row in rows}
        overlay_paths = []
        for replay in replays:
            overlay_path = overlays_dir / f"measure_{replay.measure_index:03d}.png"
            meter._write_overlay(rows_by_key[replay.key], replay, overlay_path)
            overlay_paths.append(overlay_path)
        contact_sheet_path = temp_dir / "contact_sheet.png"
        heldout._write_contact_sheet(overlay_paths, contact_sheet_path)

        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": OUTPUT_KIND,
            "status": "triaged_awaiting_truth",
            "target": target,
            "truth_used": False,
            "mutation_policy": "review_sidecar_only; canonical predictions are unchanged",
            "canonical_predictions_mutated": False,
            "measure_count": len(rows),
            "flagged_measure_count": len(flagged),
            "flagged_automatic_measure_indices": flagged,
            "review_flag_semantics": (
                "true only when the existing visual rhythm observer underfills "
                "a non-pickup measure with available meter context"
            ),
            "model_audit": model_audit,
            "input_hashes": inputs["pins"],
            "source_images": inputs["source_image_pins"],
        }
        report_path = temp_dir / "report.json"
        _write_json(report_path, report)

        # Re-verify the gate and every consumed input immediately before publication.
        verified_again = evaluator.verify_frozen_gate(sealed_manifest_path)
        if (
            verified_again["sealed_sha256"] != frozen["sealed_sha256"]
            or verified_again["freeze_sha256"] != frozen["freeze_sha256"]
            or verified_again["prepared_sha256"] != frozen["prepared_sha256"]
        ):
            raise ValueError("Frozen held-out gate changed during meter triage")
        _verify_consumed_hashes(inputs["paths"], inputs["pins"])
        if freezer._sha256(canonical_predictions_path) != canonical_predictions_sha256:
            raise ValueError("Canonical frozen predictions changed during meter triage")
        if {
            crop: freezer._hash_json(row)
            for crop, row in sorted(verified_again["predictions_by_crop"].items())
        } != canonical_row_hashes:
            raise ValueError("Canonical frozen prediction rows changed during meter triage")

        output_pins = {
            "predictions": _file_pin(predictions_path, root=temp_dir),
            "report": _file_pin(report_path, root=temp_dir),
            "contact_sheet": _file_pin(contact_sheet_path, root=temp_dir),
            "overlays": [_file_pin(path, root=temp_dir) for path in sorted(overlay_paths)],
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": MANIFEST_KIND,
            "status": "triaged_awaiting_truth",
            "create_once": True,
            "atomic_publication": True,
            "target": target,
            "truth_used": False,
            "truth_or_musicxml_opened": False,
            "review_flags_only": True,
            "canonical_predictions_mutated": False,
            "sealed_manifest": _file_pin(sealed_manifest_path),
            "freeze_manifest": _file_pin(Path(frozen["freeze_path"])),
            "prepared_manifest": _file_pin(Path(frozen["prepared_path"])),
            "implementation": _file_pin(Path(__file__).resolve()),
            "inputs": inputs["pins"],
            "source_images": inputs["source_image_pins"],
            "outputs": output_pins,
        }
        _write_json(temp_dir / "manifest.json", manifest)
        temp_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return {
        "output_dir": str(output_dir),
        "manifest": str(output_dir / "manifest.json"),
        "report": str(output_dir / "report.json"),
        "predictions": str(output_dir / "predictions.jsonl"),
        "contact_sheet": str(output_dir / "contact_sheet.png"),
        "flagged_measure_indices": flagged,
    }


def _load_verified_inputs(
    frozen: Mapping[str, Any],
    *,
    target_slug: str,
) -> dict[str, Any]:
    namespace_root = Path(frozen["namespace_root"]).resolve()
    freeze = frozen["freeze"]
    binding = freeze.get("inference_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("Sealed gate has no hash-pinned inference binding")
    selected_model = binding.get("selected_model")
    inference_binding = binding.get("inference")
    if not isinstance(selected_model, Mapping) or not isinstance(inference_binding, Mapping):
        raise ValueError("Sealed gate inference binding is incomplete")
    model_artifacts = selected_model.get("artifacts")
    if not isinstance(model_artifacts, Mapping) or "model.json" not in model_artifacts:
        raise ValueError("Sealed gate has no hash-pinned serialized model")

    records = {
        "model": model_artifacts["model.json"],
        "requests": inference_binding["requests"],
        "detailed_inference": inference_binding["detailed_inference"],
        "metadata": inference_binding["metadata"],
        "canonical_predictions": inference_binding["predictions"],
    }
    paths = {
        label: _resolve_snapshot_record(
            namespace_root,
            record,
            label=label,
            target_slug=target_slug,
        )
        for label, record in records.items()
    }
    pins = {label: _file_pin(path) for label, path in paths.items()}

    model_payload = _read_json(paths["model"])
    request_rows = _read_jsonl(paths["requests"])
    inference_rows = _read_jsonl(paths["detailed_inference"])
    metadata = _read_json(paths["metadata"])
    canonical_prediction_rows = _read_jsonl(paths["canonical_predictions"])
    if not (
        len(request_rows) == len(inference_rows) == len(canonical_prediction_rows) and request_rows
    ):
        raise ValueError("Frozen inference inputs are empty or have mismatched row counts")

    out_dir = freezer._find_out_dir(namespace_root)
    source_image_pins = _verify_rows_and_images(
        request_rows,
        inference_rows,
        canonical_prediction_rows,
        out_dir=out_dir,
        target_slug=target_slug,
    )
    return {
        "paths": paths,
        "pins": pins,
        "model_payload": model_payload,
        "request_rows": request_rows,
        "inference_rows": inference_rows,
        "metadata": metadata,
        "canonical_prediction_rows": canonical_prediction_rows,
        "out_dir": out_dir,
        "source_image_pins": source_image_pins,
    }


def _resolve_snapshot_record(
    namespace_root: Path,
    record: Any,
    *,
    label: str,
    target_slug: str,
) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError(f"Frozen {label} pin is not an object")
    relative = Path(str(record["snapshot_path_relative_to_namespace"]))
    if relative.is_absolute():
        raise ValueError(f"Frozen {label} snapshot path must be namespace-relative")
    path = (namespace_root / relative).resolve()
    try:
        path.relative_to(namespace_root)
    except ValueError as exc:
        raise ValueError(f"Frozen {label} snapshot escapes its namespace") from exc
    _reject_truth_looking_path(path, target_slug=target_slug)
    if not path.is_file():
        raise FileNotFoundError(f"Frozen {label} snapshot does not exist: {path}")
    expected = str(record["snapshot_sha256"])
    if freezer._sha256(path) != expected:
        raise ValueError(f"Frozen {label} snapshot hash drift: {path}")
    return path


def _verify_rows_and_images(
    requests: Sequence[Mapping[str, Any]],
    inference_rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    out_dir: Path,
    target_slug: str,
) -> list[dict[str, Any]]:
    source_pins = []
    seen_measures: set[int] = set()
    for request, detail, prediction in zip(requests, inference_rows, predictions, strict=True):
        identity = request.get("identity")
        if not isinstance(identity, Mapping):
            raise ValueError("Frozen request identity is missing")
        measure = int(identity["system_measure_index"])
        if measure in seen_measures:
            raise ValueError(f"Duplicate frozen measure identity: {measure}")
        seen_measures.add(measure)
        if detail.get("identity") != identity or prediction.get("identity") != identity:
            raise ValueError("Frozen request, inference, and prediction identities differ")
        if detail.get("canonical_prediction") != prediction:
            raise ValueError("Frozen detailed inference does not match canonical prediction")
        if detail.get("allowed_context") != request.get("allowed_context"):
            raise ValueError("Frozen detailed inference does not match request context")
        if any(row.get("truth_used") is not False for row in (request, detail)):
            raise ValueError("Frozen inference input did not preserve truth_used=false")
        provenance = prediction.get("inference_provenance", {})
        if not isinstance(provenance, Mapping) or provenance.get("truth_used") is not False:
            raise ValueError("Canonical prediction did not preserve truth_used=false")

        raw = request.get("images", {}).get("raw", {})
        if not isinstance(raw, Mapping):
            raise ValueError(f"Frozen request {measure} has no raw image record")
        image_path = (out_dir / str(raw["path_relative_to_out"])).resolve()
        try:
            image_path.relative_to(out_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"Frozen request image escapes output root: {image_path}") from exc
        _reject_truth_looking_path(image_path, target_slug=target_slug)
        if not image_path.is_file():
            raise FileNotFoundError(f"Frozen request image does not exist: {image_path}")
        image_sha256 = freezer._sha256(image_path)
        if image_sha256 != str(raw["sha256"]):
            raise ValueError(f"Frozen request image hash drift: {image_path}")
        source = detail.get("source")
        if not isinstance(source, Mapping) or str(source.get("sha256")) != image_sha256:
            raise ValueError("Frozen detailed inference source hash differs from request image")
        source_pins.append(
            {
                "automatic_measure_index": measure,
                **_file_pin(image_path),
            }
        )
    return source_pins


def _verify_consumed_hashes(
    paths: Mapping[str, Path],
    pins: Mapping[str, Mapping[str, Any]],
) -> None:
    for label, path in paths.items():
        if freezer._sha256(path) != str(pins[label]["sha256"]):
            raise ValueError(f"Consumed frozen input changed during meter triage: {label}")


def _validate_output_dirname(value: str) -> None:
    if not SAFE_OUTPUT_NAME.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"Invalid meter-triage output directory name: {value!r}")
    lowered = value.casefold()
    if ("truth" in lowered and not lowered.startswith("pretruth_")) or "musicxml" in lowered:
        raise ValueError(f"Truth/MusicXML-looking output directory is forbidden: {value!r}")


def _reject_truth_looking_path(path: Path, *, target_slug: str | None = None) -> None:
    parts = tuple(part.casefold() for part in path.parts)
    if path.suffix.casefold() in TRUTH_SUFFIXES:
        raise ValueError(f"Truth/MusicXML-looking path is forbidden: {path}")
    if any(part in {"ground_truth", "truth"} for part in parts):
        raise ValueError(f"Truth/MusicXML-looking path is forbidden: {path}")
    if any("musicxml" in part for part in parts):
        raise ValueError(f"Truth/MusicXML-looking path is forbidden: {path}")
    if target_slug and freezer._is_forbidden_target_truth_path(
        parts,
        target_slug=target_slug,
    ):
        raise ValueError(f"Target truth/MusicXML path is forbidden: {path}")


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


def _file_pin(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    display = (
        resolved.relative_to(root.resolve()).as_posix()
        if root is not None
        else heldout._display_path(resolved)
    )
    return {
        "path": display,
        "sha256": freezer._sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


if __name__ == "__main__":
    raise SystemExit(main())
