"""Replay the selected model on a prepared truth-blind third-score gate.

This spike validates every prepared/model/training hash before inference,
materializes deterministic per-crop predictions and diagnostics, and then uses
the existing third-score freezer. Target truth and MusicXML are forbidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_vlm_melody_inputs as melody_inputs  # noqa: E402
from scripts.experiments import freeze_fifth_score_heldout as fifth_freezer  # noqa: E402
from scripts.experiments import freeze_fourth_score_heldout as fourth_freezer  # noqa: E402
from scripts.experiments import freeze_independent_key_state_gates as key_freezer  # noqa: E402
from scripts.experiments import freeze_third_score_heldout as freezer  # noqa: E402
from scripts.experiments import spike_composed_melody_chain as composed  # noqa: E402
from scripts.experiments import spike_meter_gap_resolver as gap  # noqa: E402
from scripts.experiments import spike_notehead_patch_templates as patches  # noqa: E402
from scripts.experiments import spike_review_augmented_selector as dense  # noqa: E402

SCHEMA_VERSION = 1
INFERENCE_VERSION = "third-score-inference-v2"
FOURTH_SCORE_INFERENCE_VERSION = "fourth-score-inference-v1"
FIFTH_SCORE_INFERENCE_VERSION = "fifth-score-inference-v1"
INDEPENDENT_KEY_INFERENCE_VERSION = "independent-key-inference-v1"
DEFAULT_INFERENCE_DIRNAME = "inference_v2"
DEFAULT_MODEL_DIR = REPO_ROOT / "out/vlm_melody_consumed_training/cross_score_notehead_v1"
LA_CHATA_SLUG = "jaime-llanos_64_la-chata_pasillo_luis-a-calvo"
TRUTH_PATH_MARKERS = (
    "/dataset/ground_truth/",
    "/dataset/musicxml/",
)


GATE_CONFIGS = {
    freezer.THIRD_SCORE_GATE.prepare_kind: {
        "gate": freezer.THIRD_SCORE_GATE,
        "inference_version": INFERENCE_VERSION,
        "manifest_kind": "third_score_truth_blind_inference_manifest",
        "binding_kind": "third_score_inference_provenance_binding",
    },
    fourth_freezer.FOURTH_SCORE_GATE.prepare_kind: {
        "gate": fourth_freezer.FOURTH_SCORE_GATE,
        "inference_version": FOURTH_SCORE_INFERENCE_VERSION,
        "manifest_kind": "fourth_score_truth_blind_inference_manifest",
        "binding_kind": "fourth_score_inference_provenance_binding",
    },
    fifth_freezer.FIFTH_SCORE_GATE.prepare_kind: {
        "gate": fifth_freezer.FIFTH_SCORE_GATE,
        "inference_version": FIFTH_SCORE_INFERENCE_VERSION,
        "manifest_kind": "fifth_score_truth_blind_inference_manifest",
        "binding_kind": "fifth_score_inference_provenance_binding",
    },
    **{
        gate.prepare_kind: {
            "gate": gate,
            "inference_version": INDEPENDENT_KEY_INFERENCE_VERSION,
            "manifest_kind": "independent_key_truth_blind_inference_manifest",
            "binding_kind": "independent_key_inference_provenance_binding",
        }
        for gate in key_freezer.gates()
    },
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared_manifest", type=Path)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--inference-dirname", default=DEFAULT_INFERENCE_DIRNAME)
    parser.add_argument(
        "--no-freeze",
        action="store_true",
        help="Materialize inference only; intended for focused replay tests.",
    )
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        result = materialize_third_score_inference(
            args.prepared_manifest,
            model_dir=args.model_dir,
            inference_dirname=args.inference_dirname,
        )
        if not args.no_freeze:
            result["freeze"] = freeze_inference(
                args.prepared_manifest,
                inference_dir=Path(result["inference_dir"]),
                model_dir=args.model_dir,
            )
    except (FileExistsError, FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    result["runtime_seconds"] = round(time.perf_counter() - started, 6)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def materialize_third_score_inference(
    prepared_manifest_path: Path,
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
    inference_dirname: str = DEFAULT_INFERENCE_DIRNAME,
) -> dict[str, Any]:
    """Create deterministic truth-blind inference artifacts exactly once."""
    _validate_output_name(inference_dirname)
    prepared_manifest_path = prepared_manifest_path.resolve()
    model_dir = model_dir.resolve()
    _reject_truth_path(prepared_manifest_path)
    _reject_truth_path(model_dir)
    namespace_root = prepared_manifest_path.parent
    output_dir = namespace_root / inference_dirname
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite create-once inference: {output_dir}")

    validated = _validate_inputs(prepared_manifest_path, model_dir=model_dir)
    prepared = validated["prepared"]
    gate_config = validated["gate_config"]
    expected_count = validated["expected_count"]
    model_payload = validated["model"]
    requests = validated["prepared_requests"]
    metadata = validated["metadata"]
    replay_model, pitch_predictor, replay_audit = reconstruct_model(model_payload)
    out_dir = freezer._find_out_dir(namespace_root)

    temp_dir = output_dir.with_name(f".{output_dir.name}.tmp")
    if temp_dir.exists():
        raise FileExistsError(f"Refusing stale temporary inference: {temp_dir}")
    temp_dir.mkdir(parents=False)
    try:
        assumptions = _context_assumptions(
            metadata,
            validated["metadata_record"],
            prepared_context=validated.get("prepared_context"),
        )
        materialized_requests = [
            _materialize_request(
                row,
                namespace_root=namespace_root,
                out_dir=out_dir,
                assumptions=assumptions,
                row_sha256=prepared["artifacts"]["requests"]["row_sha256"][index],
            )
            for index, row in enumerate(requests)
        ]
        _write_jsonl(temp_dir / "requests.jsonl", materialized_requests)
        _write_json(temp_dir / "assumptions.json", assumptions)
        _write_json(temp_dir / "replay.json", replay_audit)

        items = [
            _infer_request(
                request,
                model=replay_model,
                pitch_predictor=pitch_predictor,
                out_dir=out_dir,
                selector_method_id=str(model_payload["replay"]["method"]["method_id"]),
            )
            for request in materialized_requests
        ]
        if len(items) != expected_count:
            raise ValueError(
                f"Held-out inference requires exactly {expected_count} outputs, got {len(items)}"
            )
        predictions = [item.prediction for item in items]
        inference_rows = [_inference_record(item) for item in items]
        if any(row.get("truth_used") is not False for row in inference_rows):
            raise ValueError("Inference output did not preserve truth_used=false")
        _write_jsonl(temp_dir / "predictions.jsonl", predictions)
        _write_jsonl(temp_dir / "inference.jsonl", inference_rows)
        overlay_dir = temp_dir / "overlays"
        overlay_dir.mkdir()
        overlay_paths = []
        for index, item in enumerate(items, start=1):
            overlay_path = overlay_dir / f"measure_{index:03d}.png"
            composed._write_overlay(item, overlay_path)
            overlay_paths.append(overlay_path)
        _write_contact_sheet(overlay_paths, temp_dir / "contact_sheet.png")

        artifacts = _artifact_records(temp_dir)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": gate_config["manifest_kind"],
            "version": gate_config["inference_version"],
            "status": "inferred_awaiting_freeze",
            "split": "fresh_heldout",
            "truth_accessed": False,
            "truth_used": False,
            "target": prepared["target"],
            "create_once": True,
            "prepared_manifest": _file_record(prepared_manifest_path),
            "model_and_training": validated["pins"],
            "implementation": _file_record(Path(__file__)),
            "context": {
                "metadata": validated["metadata_record"],
                **(
                    {"prepared_context": validated["prepared_context_record"]}
                    if validated.get("prepared_context_record")
                    else {}
                ),
                "assumptions": {
                    "path": "assumptions.json",
                    "sha256": artifacts["assumptions.json"]["sha256"],
                },
                "requests": {
                    "path": "requests.jsonl",
                    "sha256": artifacts["requests.jsonl"]["sha256"],
                },
                "replay": {
                    "path": "replay.json",
                    "sha256": artifacts["replay.json"]["sha256"],
                },
                "detailed_inference": {
                    "path": "inference.jsonl",
                    "sha256": artifacts["inference.jsonl"]["sha256"],
                },
            },
            "output_count": len(predictions),
            "artifacts": artifacts,
            "warnings": assumptions["warnings"],
        }
        _write_json(temp_dir / "manifest.json", manifest)
        temp_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return {
        "inference_dir": str(output_dir),
        "manifest": str(output_dir / "manifest.json"),
        "manifest_sha256": _sha256(output_dir / "manifest.json"),
        "predictions": str(output_dir / "predictions.jsonl"),
        "predictions_sha256": _sha256(output_dir / "predictions.jsonl"),
        "inference_sha256": _sha256(output_dir / "inference.jsonl"),
        "output_count": expected_count,
        "warnings": assumptions["warnings"],
    }


def freeze_inference(
    prepared_manifest_path: Path,
    *,
    inference_dir: Path,
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    """Freeze the prediction file and all selected model/training provenance."""
    prepared_manifest_path = prepared_manifest_path.resolve()
    inference_dir = inference_dir.resolve()
    model_dir = model_dir.resolve()
    for path in (prepared_manifest_path, inference_dir, model_dir):
        _reject_truth_path(path)
    manifest = _read_json(inference_dir / "manifest.json")
    validated = _verify_inference_binding(
        prepared_manifest_path,
        model_dir=model_dir,
        inference_dir=inference_dir,
        manifest=manifest,
    )
    model_paths = _deduplicate_paths(
        [model_dir / "model.json", validated["model_implementation_path"]]
    )
    selected_model_training_outputs = [
        path
        for path in validated["model_artifact_paths"]
        if path != (model_dir / "model.json").resolve()
    ]
    training_paths = _deduplicate_paths(
        [
            *selected_model_training_outputs,
            model_dir / "manifest.json",
            *validated["training_input_paths"],
            inference_dir / "manifest.json",
            Path(__file__),
            _resolve_record_path(validated["metadata_record"]),
            inference_dir / "assumptions.json",
            inference_dir / "requests.jsonl",
            inference_dir / "replay.json",
            inference_dir / "inference.jsonl",
            *validated.get("prepared_context_paths", ()),
        ]
    )
    result = freezer.freeze_prepared_heldout_score(
        prepared_manifest_path,
        predictions_path=inference_dir / "predictions.jsonl",
        model_artifact_paths=model_paths,
        training_artifact_paths=training_paths,
        gate=validated["gate_config"]["gate"],
    )
    frozen_dir = prepared_manifest_path.parent / "frozen"
    _seal_inference_binding(
        frozen_dir,
        manifest=manifest,
        prepared_manifest_path=prepared_manifest_path,
        model_dir=model_dir,
        inference_dir=inference_dir,
        validated=validated,
    )
    verify_frozen_outputs(frozen_dir)
    result.update(
        {
            "freeze_sha256": _sha256(frozen_dir / "freeze.json"),
            "sealed_manifest_sha256": _sha256(frozen_dir / "sealed_manifest.json"),
        }
    )
    return result


def reconstruct_model(
    payload: Mapping[str, Any],
) -> tuple[gap.GapAwareSelectorModel, composed.PitchPredictor, dict[str, Any]]:
    """Reconstruct every serialized inference-bearing model component."""
    if payload.get("kind") != "vlm_melody_consumed_training_notehead_model":
        raise ValueError("Unsupported selected model kind")
    if payload.get("configuration") != "C" or payload.get("blocked") is not False:
        raise ValueError("Selected model must be unblocked configuration C")
    if payload.get("third_score_truth_used") is not False:
        raise ValueError("Selected model is not truth-blind for the third score")
    replay = payload["replay"]
    if tuple(replay["feature_order"]) != tuple(dense.DENSE_FEATURES):
        raise ValueError("Serialized dense feature order drift")
    if replay["selection_mode"] != dense.SELECTION_MODE:
        raise ValueError("Serialized selection mode drift")
    selector = replay["selector"]
    scorer_payload = selector["scorer"]
    scorer_kind = str(scorer_payload["kind"])
    if scorer_kind == "patch_scorer":
        scorer = patches.PatchScorer(
            patch_id=str(scorer_payload["patch_id"]),
            scorer_kind=str(scorer_payload["scorer_kind"]),  # type: ignore[arg-type]
            positive_vectors=tuple(
                tuple(float(value) for value in vector)
                for vector in scorer_payload["positive_vectors"]
            ),
            negative_vectors=tuple(
                tuple(float(value) for value in vector)
                for vector in scorer_payload["negative_vectors"]
            ),
        )
    elif scorer_kind == "dense_logistic":
        scorer = dense.LogisticScorer(
            means=tuple(float(value) for value in scorer_payload["means"]),
            scales=tuple(float(value) for value in scorer_payload["scales"]),
            weights=tuple(float(value) for value in scorer_payload["weights"]),
            intercept=float(scorer_payload["intercept"]),
        )
    else:
        raise ValueError(f"Unsupported serialized scorer: {scorer_kind}")
    selection_metrics = dict(replay["method"]["selection"].get("metrics", {}))
    base = dense.DenseSelectorModel(
        scorer=scorer,  # type: ignore[arg-type]
        learned_threshold=float(selector["threshold"]),
        threshold_training_metrics=selection_metrics,
        training_keys=tuple(str(value) for value in payload["training"]["keys"]),
        training_positive_count=len(payload["training"]["review_hashes"]),
        # The historical count selector field was not serialized and is not used by
        # the selected threshold mode. Zero keeps its diagnostic flag inert.
        learned_count=0,
        nms_x_spaces=float(selector["nms_x_spaces"]),
        minimum_selected_count=int(selector["minimum_selected_count"]),
        maximum_selected_count=int(selector["maximum_selected_count"]),
    )
    recovery_payload = replay["method"]["recovery"]
    recovery = gap.RecoverySelection(
        leading_gap_spaces=float(selector["leading_gap_spaces"]),
        score_margin=float(selector["score_margin"]),
        metrics=dict(recovery_payload.get("metrics", {})),
        base_metrics=dict(recovery_payload.get("base_metrics", {})),
        searched=tuple(dict(row) for row in recovery_payload.get("searched", [])),
    )
    model = gap.GapAwareSelectorModel(base=base, recovery=recovery)

    pitch_payload = replay["pitch"]
    pitch_method = str(pitch_payload["method"])
    accidental_model = None
    if pitch_method == "accidental_knn":
        accidental_model = dense.AccidentalKNN(
            tuple(
                dense.PitchSample(
                    key=str(row["key"]),
                    pitch=str(row["pitch"]),
                    base_pitch=str(row["base_pitch"]),
                    delta=int(row["delta"]),
                    vector=tuple(float(value) for value in row["vector"]),
                )
                for row in pitch_payload["accidental_samples"]
            )
        )
    elif pitch_method != "key_signature_only":
        raise ValueError(f"Unsupported serialized pitch method: {pitch_method}")
    predictor = dense._build_pitch_predictor(accidental_model)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "kind": "serialized_model_replay_audit",
        "truth_used": False,
        "selection_mode": replay["selection_mode"],
        "selector": {
            key: selector[key]
            for key in (
                "threshold",
                "nms_x_spaces",
                "minimum_selected_count",
                "maximum_selected_count",
                "leading_gap_spaces",
                "score_margin",
            )
        },
        "scorer": {
            "kind": scorer_kind,
            "patch_id": scorer_payload.get("patch_id"),
            "scorer_kind": scorer_payload.get("scorer_kind"),
            "positive_vector_count": len(scorer_payload.get("positive_vectors", [])),
            "negative_vector_count": len(scorer_payload.get("negative_vectors", [])),
        },
        "recovery": {
            "leading_gap_spaces": recovery.leading_gap_spaces,
            "score_margin": recovery.score_margin,
        },
        "pitch": {
            "method": pitch_method,
            "accidental_sample_count": len(pitch_payload["accidental_samples"]),
        },
        "nonserialized_adapter_fields": {
            "learned_count": {
                "value": 0,
                "reason": "not serialized and unused by threshold_selector",
            }
        },
    }
    return model, predictor, audit


def _validate_inputs(prepared_path: Path, *, model_dir: Path) -> dict[str, Any]:
    prepared = _read_json(prepared_path)
    gate_config = GATE_CONFIGS.get(str(prepared.get("kind")))
    if gate_config is None:
        raise ValueError(f"Unsupported held-out prepared manifest kind: {prepared.get('kind')}")
    freezer._verify_prepared_manifest(
        prepared_path.parent,
        prepared_path,
        prepared,
        expected_kind=str(prepared["kind"]),
    )
    target = prepared.get("target")
    if not isinstance(target, Mapping) or not target.get("slug"):
        raise ValueError("Prepared target is missing")
    target_slug = str(target["slug"])
    _reject_target_truth_path(prepared_path, target_slug=target_slug)
    requests_path = prepared_path.parent / str(prepared["artifacts"]["requests"]["path"])
    requests = _read_jsonl(requests_path)
    expected_count = len(prepared["artifacts"]["requests"]["row_sha256"])
    if expected_count <= 0 or len(requests) != expected_count:
        raise ValueError(
            f"Prepared request count drift: expected {expected_count}, got {len(requests)}"
        )
    row_hashes = [_hash_json(row) for row in requests]
    if row_hashes != list(prepared["artifacts"]["requests"]["row_sha256"]):
        raise ValueError("Prepared request row hash drift")
    if any(row.get("truth_accessed") is not False for row in requests):
        raise ValueError("Prepared requests are not truth-blind")

    model_manifest_path = model_dir / "manifest.json"
    model_manifest = _read_json(model_manifest_path)
    if model_manifest.get("kind") != "vlm_melody_cross_score_consumed_retraining_manifest":
        raise ValueError("Selected model manifest kind mismatch")
    if model_manifest.get("la_chata_truth_accessed") is not False:
        raise ValueError("Selected model manifest is not La Chata truth-blind")
    pins: dict[str, dict[str, str]] = {}
    required_artifacts = {"model.json", "training_selection.json", "report.json"}
    if not required_artifacts.issubset(model_manifest["artifacts"]):
        missing = sorted(required_artifacts - set(model_manifest["artifacts"]))
        raise ValueError(f"Selected model manifest is missing artifacts: {missing}")
    for name, record in model_manifest["artifacts"].items():
        source = _resolve_record_path(record)
        _reject_truth_path(source)
        expected_source = (model_dir / str(name)).resolve()
        if source.resolve() != expected_source:
            raise ValueError(
                f"Selected model manifest artifact path substitution: {name}: "
                f"expected {expected_source}, got {source}"
            )
        if _sha256(source) != str(record["sha256"]):
            raise ValueError(f"Selected model artifact hash drift: {source}")
        pins[name] = _file_record(source)
    implementation = _resolve_record_path(model_manifest["implementation"])
    if _sha256(implementation) != str(model_manifest["implementation"]["sha256"]):
        raise ValueError("Selected model implementation hash drift")

    model_payload = _read_json(model_dir / "model.json")
    training_selection = _read_json(model_dir / "training_selection.json")
    report = _read_json(model_dir / "report.json")
    selected = str(training_selection["selection"]["selected_configuration"])
    if selected != str(model_payload["configuration"]):
        raise ValueError("Model/training selected configuration drift")
    if selected != str(report["selection"]["selected_configuration"]):
        raise ValueError("Model/report selected configuration drift")
    final = training_selection["final_training"]
    if list(final["keys"]) != list(model_payload["training"]["keys"]):
        raise ValueError("Model/training key drift")
    if list(final["review_hashes"]) != list(model_payload["training"]["review_hashes"]):
        raise ValueError("Model/training review hash drift")
    training_scores = {str(value) for value in final.get("scores", [])}
    training_scores.update(str(value) for value in model_payload["training"].get("scores", []))
    training_scores.update(str(value).split(":", 1)[0] for value in final["keys"])
    if target_slug in training_scores:
        raise ValueError(f"Selected model was trained on held-out target: {target_slug}")
    training_input_paths = []
    for group in training_selection["input_provenance"].values():
        for record in group:
            source = _resolve_record_path(record)
            _reject_truth_path(source)
            if _sha256(source) != str(record["sha256"]):
                raise ValueError(f"Training input provenance hash drift: {source}")
            training_input_paths.append(source.resolve())

    out_dir = freezer._find_out_dir(prepared_path.parent)
    metadata_path = out_dir / target_slug / "metadata.json"
    _reject_truth_path(metadata_path)
    _reject_target_truth_path(metadata_path, target_slug=target_slug)
    metadata = _read_json(metadata_path)
    prepared_context = None
    prepared_context_record = None
    prepared_context_paths: tuple[Path, ...] = ()
    context_artifacts = prepared["artifacts"].get("context", {})
    if context_artifacts:
        prepared_context_paths = tuple(
            (prepared_path.parent / str(record["path"])).resolve()
            for record in context_artifacts.values()
        )
        allowed_record = context_artifacts.get("allowed_context")
        if not isinstance(allowed_record, Mapping):
            raise ValueError("Prepared context is missing allowed_context")
        allowed_path = prepared_path.parent / str(allowed_record["path"])
        prepared_context = _read_json(allowed_path)
        prepared_context_record = _file_record(allowed_path)
    return {
        "prepared": prepared,
        "prepared_requests": requests,
        "model": model_payload,
        "training_selection": training_selection,
        "report": report,
        "metadata": metadata,
        "metadata_record": _file_record(metadata_path),
        "prepared_context": prepared_context,
        "prepared_context_record": prepared_context_record,
        "prepared_context_paths": prepared_context_paths,
        "expected_count": expected_count,
        "gate_config": gate_config,
        "model_artifact_paths": tuple(
            (model_dir / name).resolve() for name in sorted(model_manifest["artifacts"])
        ),
        "model_implementation_path": implementation.resolve(),
        "training_input_paths": tuple(sorted(set(training_input_paths))),
        "pins": {
            "model_manifest": _file_record(model_manifest_path),
            "implementation": _file_record(implementation),
            "artifacts": pins,
        },
    }


def _context_assumptions(
    metadata: Mapping[str, Any],
    metadata_record: Mapping[str, str],
    *,
    prepared_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if prepared_context is not None:
        if (
            prepared_context.get("truth_used") is not False
            or prepared_context.get("truth_accessed") is not False
        ):
            raise ValueError("Prepared musical context is not truth-blind")
        allowed = prepared_context.get("allowed_context")
        provenance = prepared_context.get("provenance")
        if not isinstance(allowed, Mapping) or not isinstance(provenance, Mapping):
            raise ValueError("Prepared musical context is malformed")
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "fresh_score_allowed_context_assumptions",
            "truth_used": False,
            "metadata": dict(metadata_record),
            "allowed_context": dict(allowed),
            "provenance": dict(provenance),
            "warnings": list(prepared_context.get("warnings", [])),
        }
    time_signature = metadata.get("time_signature")
    expected_beats = _expected_beats(time_signature)
    warnings = []
    if expected_beats is None:
        warnings.append(
            "Pipeline metadata has no usable time signature; meter decoding, onsets, "
            "durations, and rest inference are not applied."
        )
    if metadata.get("key_hint") is None:
        warnings.append(
            "Pipeline metadata has no key hint; pitch mapping uses natural treble-clef pitches."
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "third_score_allowed_context_assumptions",
        "truth_used": False,
        "metadata": dict(metadata_record),
        "allowed_context": {
            "clef": "treble",
            "time_signature": time_signature,
            "key_hint": metadata.get("key_hint"),
            "expected_measure_beats": expected_beats,
            "allow_pickup": False,
        },
        "provenance": {
            "clef": "melody-spike fixed treble-clef contract; not target truth",
            "time_signature": "hash-pinned pipeline metadata",
            "key_hint": "hash-pinned pipeline metadata",
            "expected_measure_beats": "derived only when metadata time_signature is present",
            "allow_pickup": "false because automatic system-7 crops cannot establish pickup status",
        },
        "warnings": warnings,
    }


def _materialize_request(
    prepared_row: Mapping[str, Any],
    *,
    namespace_root: Path,
    out_dir: Path,
    assumptions: Mapping[str, Any],
    row_sha256: str,
) -> dict[str, Any]:
    crop_path = namespace_root / str(prepared_row["input"]["path_relative_to_namespace"])
    _reject_truth_path(crop_path)
    if _sha256(crop_path) != str(prepared_row["input"]["sha256"]):
        raise ValueError(f"Prepared crop hash drift: {crop_path}")
    with Image.open(crop_path) as opened:
        image = opened.convert("RGB")
        width, height = image.size
        staff = melody_inputs._estimate_staff(image)
    if len(staff.line_ys) != 5:
        raise ValueError(f"Expected five detected staff lines: {crop_path}")
    identity = dict(prepared_row["identity"])
    identity["system_measure_index"] = int(identity.pop("automatic_measure_index"))
    identity["automatic_measure_index"] = identity["system_measure_index"]
    return {
        "schema_version": SCHEMA_VERSION,
        "split": "fresh_heldout",
        "truth_accessed": False,
        "truth_used": False,
        "identity": identity,
        "images": {
            "raw": {
                "path_relative_to_out": crop_path.relative_to(out_dir).as_posix(),
                "sha256": _sha256(crop_path),
                "width_px": width,
                "height_px": height,
            }
        },
        "staff_geometry": {
            "raw_staff_lines_y_px": list(staff.line_ys),
            "method": "build_vlm_melody_inputs._estimate_staff",
        },
        "allowed_context": dict(assumptions["allowed_context"]),
        "allowed_context_provenance": dict(assumptions["provenance"]),
        "prepared_provenance": {
            "prepared_request_sha256": row_sha256,
            "layout_provenance": dict(prepared_row["layout_provenance"]),
            "bbox_px": list(prepared_row["input"]["bbox_px"]),
        },
    }


def _infer_request(
    request: Mapping[str, Any],
    *,
    model: gap.GapAwareSelectorModel,
    pitch_predictor: composed.PitchPredictor,
    out_dir: Path,
    selector_method_id: str,
) -> composed.ComposedMeasure:
    measure = dense._prepare_dense_measure(request, out_dir=out_dir)
    if request["allowed_context"].get("expected_measure_beats") is not None:
        return gap._compose_with_meter_gap_resolver(
            request,
            measure,
            model,
            out_dir=out_dir,
            selector_method_id=selector_method_id,
            pitch_predictor=pitch_predictor,
        )
    image_path = composed._resolve_request_image(request, out_dir)
    candidate_predictions, selected = model.rank(measure, selection_mode=dense.SELECTION_MODE)
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    staff_lines = [int(value) for value in request["staff_geometry"]["raw_staff_lines_y_px"]]
    spacing = composed.rhythm.staff_spacing(staff_lines)
    ordered = sorted(selected, key=lambda row: (row.center_x, row.center_y, row.id))
    anchors = [
        {
            "order": index,
            "pitch": pitch_predictor(candidate, request, image),
            "center": {"x": round(candidate.center_x, 3), "y": round(candidate.center_y, 3)},
            "source": {
                "kind": "automatic_candidate",
                "candidate_id": candidate.id,
                "selector_method": selector_method_id,
                "selection_mode": dense.SELECTION_MODE,
            },
        }
        for index, candidate in enumerate(ordered, start=1)
    ]
    groups = composed.rhythm.group_simultaneous_heads(anchors, spacing)
    anchor_features = composed.rhythm.extract_anchor_features(image, anchors, staff_lines)
    rest_features = composed.rhythm.extract_residual_rest_features(image, groups, staff_lines)
    visual_symbols = composed.rhythm.build_visual_symbols(groups, anchor_features, rest_features)
    ordered_notes = [
        {
            "order": anchor["order"],
            "pitch": anchor["pitch"],
            "pitch_midi": composed.rhythm.pitch_to_midi(str(anchor["pitch"])),
            "onset_beats": None,
            "duration_beats": None,
            "candidate_id": anchor["source"]["candidate_id"],
            "center": anchor["center"],
        }
        for anchor in anchors
    ]
    prediction = {
        "identity": dict(request["identity"]),
        "notes": ordered_notes,
        "rests": [],
        "rhythm_tokens": [],
        "measure_extent_beats": None,
        "decoder_status": "not_applied_missing_expected_measure_beats",
        "inference_provenance": {
            "notehead_selector": selector_method_id,
            "selection_mode": dense.SELECTION_MODE,
            "automatic_anchor_count": len(anchors),
            "review_anchors_used": False,
            "truth_used": False,
            "learned_score_threshold": round(model.learned_threshold, 9),
            "threshold_fit_from_training_reviews_only": True,
            "rhythm_decoding_applied": False,
        },
    }
    return composed.ComposedMeasure(
        request=dict(request),
        image_path=image_path,
        image=image,
        staff_spacing=spacing,
        candidate_predictions=candidate_predictions,
        anchors=anchors,
        groups=groups,
        anchor_features=anchor_features,
        rest_features=rest_features,
        visual_symbols=visual_symbols,
        decoded_symbols=[],
        prediction=prediction,
    )


def _inference_record(item: composed.ComposedMeasure) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "identity": item.request["identity"],
        "truth_used": False,
        "source": {"image": _display_path(item.image_path), "sha256": _sha256(item.image_path)},
        "allowed_context": item.request["allowed_context"],
        "allowed_context_provenance": item.request["allowed_context_provenance"],
        "staff_geometry": item.request["staff_geometry"],
        "candidate_predictions": item.candidate_predictions,
        "automatic_anchors": item.anchors,
        "anchor_features": item.anchor_features,
        "residual_rest_features": item.rest_features,
        "visual_symbols": item.visual_symbols,
        "decoded_symbols": item.decoded_symbols,
        "decoder_status": item.prediction["decoder_status"],
        "canonical_prediction": item.prediction,
    }


def _verify_inference_binding(
    prepared_manifest_path: Path,
    *,
    model_dir: Path,
    inference_dir: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove that an inference run belongs to the exact supplied inputs."""
    validated = _validate_inputs(prepared_manifest_path, model_dir=model_dir)
    gate_config = validated["gate_config"]
    if manifest.get("kind") != gate_config["manifest_kind"]:
        raise ValueError("Inference manifest kind mismatch")
    if manifest.get("version") != gate_config["inference_version"]:
        raise ValueError(
            f"Inference manifest version mismatch: expected {gate_config['inference_version']}, "
            f"got {manifest.get('version')}"
        )
    _verify_inference_manifest(
        inference_dir,
        manifest,
        expected_count=validated["expected_count"],
    )
    if manifest.get("target") != validated["prepared"].get("target"):
        raise ValueError("Inference/prepared target substitution")
    _verify_file_record(
        manifest["prepared_manifest"],
        prepared_manifest_path,
        label="Inference prepared manifest",
    )
    if manifest.get("model_and_training") != validated["pins"]:
        raise ValueError("Inference selected-model/training provenance substitution")
    _verify_file_record(
        manifest["implementation"],
        Path(__file__),
        label="Inference implementation",
    )

    context = manifest["context"]
    if context.get("metadata") != validated["metadata_record"]:
        raise ValueError("Inference metadata/context provenance substitution")
    _verify_file_record(
        context["metadata"],
        _resolve_record_path(validated["metadata_record"]),
        label="Inference metadata",
    )
    expected_prepared_context = validated.get("prepared_context_record")
    if expected_prepared_context:
        if context.get("prepared_context") != expected_prepared_context:
            raise ValueError("Inference prepared-context provenance substitution")
        _verify_file_record(
            context["prepared_context"],
            _resolve_record_path(expected_prepared_context),
            label="Inference prepared context",
        )
    elif "prepared_context" in context:
        raise ValueError("Inference introduced an unprepared musical context")
    for key, filename in (
        ("assumptions", "assumptions.json"),
        ("requests", "requests.jsonl"),
        ("replay", "replay.json"),
        ("detailed_inference", "inference.jsonl"),
    ):
        _verify_relative_record(context[key], inference_dir, filename, label=f"Inference {key}")

    assumptions = _read_json(inference_dir / "assumptions.json")
    expected_assumptions = _context_assumptions(
        validated["metadata"],
        validated["metadata_record"],
        prepared_context=validated.get("prepared_context"),
    )
    if assumptions != expected_assumptions:
        raise ValueError("Inference assumptions do not match hash-pinned metadata/context")
    _, _, expected_replay = reconstruct_model(validated["model"])
    if _read_json(inference_dir / "replay.json") != expected_replay:
        raise ValueError("Inference replay audit does not match the supplied selected model")

    requests = _read_jsonl(inference_dir / "requests.jsonl")
    predictions = _read_jsonl(inference_dir / "predictions.jsonl")
    detailed = _read_jsonl(inference_dir / "inference.jsonl")
    expected_count = validated["expected_count"]
    if not (len(requests) == len(predictions) == len(detailed) == expected_count):
        raise ValueError(
            f"Inference replay artifacts must contain exactly {expected_count} aligned rows"
        )
    for request, prediction, detail in zip(requests, predictions, detailed, strict=True):
        identity = request.get("identity")
        if prediction.get("identity") != identity or detail.get("identity") != identity:
            raise ValueError("Inference request/prediction identity substitution")
        if detail.get("canonical_prediction") != prediction:
            raise ValueError("Detailed inference/prediction substitution")
        if detail.get("allowed_context") != request.get("allowed_context"):
            raise ValueError("Detailed inference/request context substitution")
    return validated


def _seal_inference_binding(
    frozen_dir: Path,
    *,
    manifest: Mapping[str, Any],
    prepared_manifest_path: Path,
    model_dir: Path,
    inference_dir: Path,
    validated: Mapping[str, Any],
) -> None:
    freeze_path = frozen_dir / "freeze.json"
    sealed_path = frozen_dir / "sealed_manifest.json"
    freeze = _read_json(freeze_path)
    all_pins = [freeze["predictions"], *freeze["model_artifacts"], *freeze["training_artifacts"]]

    def pin(path: Path) -> dict[str, Any]:
        display = _display_path(path)
        matches = [item for item in all_pins if item.get("source_path") == display]
        if not matches:
            raise ValueError(f"Frozen provenance snapshot missing for: {path}")
        if any(item.get("source_sha256") != _sha256(path) for item in matches):
            raise ValueError(f"Frozen provenance source hash mismatch for: {path}")
        return dict(matches[0])

    model_artifacts = {
        name: pin(_resolve_record_path(record))
        for name, record in sorted(validated["pins"]["artifacts"].items())
    }
    training_inputs = [pin(path) for path in validated["training_input_paths"]]
    binding = {
        "schema_version": SCHEMA_VERSION,
        "kind": validated["gate_config"]["binding_kind"],
        "version": validated["gate_config"]["inference_version"],
        "prepared_manifest": _file_record(prepared_manifest_path),
        "selected_model": {
            "manifest": pin(model_dir / "manifest.json"),
            "artifacts": model_artifacts,
            "implementation": pin(validated["model_implementation_path"]),
            "training_inputs": training_inputs,
        },
        "inference": {
            "manifest": pin(inference_dir / "manifest.json"),
            "implementation": pin(Path(__file__)),
            "metadata": pin(_resolve_record_path(validated["metadata_record"])),
            **(
                {
                    "prepared_context": pin(
                        _resolve_record_path(validated["prepared_context_record"])
                    )
                }
                if validated.get("prepared_context_record")
                else {}
            ),
            "assumptions": pin(inference_dir / "assumptions.json"),
            "requests": pin(inference_dir / "requests.jsonl"),
            "replay": pin(inference_dir / "replay.json"),
            "detailed_inference": pin(inference_dir / "inference.jsonl"),
            "predictions": pin(inference_dir / "predictions.jsonl"),
        },
        "manifest_sha256": _sha256(inference_dir / "manifest.json"),
        "context_sha256": _hash_json(manifest["context"]),
    }
    freeze["inference_binding"] = binding
    _write_json(freeze_path, freeze)

    sealed = _read_json(sealed_path)
    sealed["freeze"]["sha256"] = _sha256(freeze_path)
    sealed["inference_binding_sha256"] = _hash_json(binding)
    _write_json(sealed_path, sealed)


def verify_frozen_outputs(frozen_dir: Path) -> None:
    freeze = _read_json(frozen_dir / "freeze.json")
    sealed = _read_json(frozen_dir / "sealed_manifest.json")
    if freeze.get("status") != "frozen_awaiting_truth" or freeze.get("truth_accessed") is not False:
        raise ValueError("Third-score freeze status is not frozen_awaiting_truth")
    if sealed.get("status") != "frozen_awaiting_truth" or sealed.get("truth_accessed") is not False:
        raise ValueError("Third-score sealed status is not frozen_awaiting_truth")
    if _sha256(frozen_dir / "freeze.json") != str(sealed["freeze"]["sha256"]):
        raise ValueError("Sealed freeze hash drift")
    for pin in [freeze["predictions"], *freeze["model_artifacts"], *freeze["training_artifacts"]]:
        snapshot = frozen_dir.parent / str(pin["snapshot_path_relative_to_namespace"])
        if _sha256(snapshot) != str(pin["snapshot_sha256"]):
            raise ValueError(f"Frozen snapshot hash drift: {snapshot}")
        if str(pin["source_sha256"]) != str(pin["snapshot_sha256"]):
            raise ValueError(f"Frozen source/snapshot hash mismatch: {snapshot}")
    binding = freeze.get("inference_binding")
    allowed_versions = {config["inference_version"] for config in GATE_CONFIGS.values()}
    if not isinstance(binding, dict) or binding.get("version") not in allowed_versions:
        raise ValueError("Frozen inference provenance binding is missing or has the wrong version")
    if _hash_json(binding) != str(sealed.get("inference_binding_sha256")):
        raise ValueError("Sealed inference provenance binding hash drift")
    all_pins = [freeze["predictions"], *freeze["model_artifacts"], *freeze["training_artifacts"]]
    known_pins = {_hash_json(pin) for pin in all_pins}
    bound_pins = [
        binding["selected_model"]["manifest"],
        *binding["selected_model"]["artifacts"].values(),
        binding["selected_model"]["implementation"],
        *binding["selected_model"]["training_inputs"],
        *binding["inference"].values(),
    ]
    if any(_hash_json(pin) not in known_pins for pin in bound_pins):
        raise ValueError("Inference provenance binding references an unsealed artifact")
    inference_manifest_pin = binding["inference"]["manifest"]
    inference_manifest_path = frozen_dir.parent / str(
        inference_manifest_pin["snapshot_path_relative_to_namespace"]
    )
    inference_manifest = json.loads(inference_manifest_path.read_text(encoding="utf-8"))
    if inference_manifest.get("prepared_manifest") != binding["prepared_manifest"]:
        raise ValueError("Frozen inference/prepared-manifest binding mismatch")
    _verify_frozen_model_and_training_binding(
        frozen_dir,
        inference_manifest=inference_manifest,
        selected_model=binding["selected_model"],
    )
    if inference_manifest.get("implementation") != _source_record(
        binding["inference"]["implementation"]
    ):
        raise ValueError("Frozen inference implementation binding mismatch")
    _verify_frozen_inference_context_binding(
        inference_manifest=inference_manifest,
        inference_binding=binding["inference"],
        frozen_predictions=freeze["predictions"],
        context_sha256=str(binding["context_sha256"]),
    )
    if str(binding["manifest_sha256"]) != str(inference_manifest_pin["source_sha256"]):
        raise ValueError("Frozen inference manifest hash binding mismatch")


def _verify_frozen_model_and_training_binding(
    frozen_dir: Path,
    *,
    inference_manifest: Mapping[str, Any],
    selected_model: Mapping[str, Any],
) -> None:
    provenance = inference_manifest.get("model_and_training")
    if not isinstance(provenance, Mapping):
        raise ValueError("Frozen inference selected-model provenance binding is missing")
    _verify_source_record_matches_pin(
        provenance.get("model_manifest"),
        selected_model.get("manifest"),
        label="Frozen inference selected-model manifest binding",
    )
    _verify_source_record_matches_pin(
        provenance.get("implementation"),
        selected_model.get("implementation"),
        label="Frozen inference selected-model implementation binding",
    )

    manifest_artifacts = provenance.get("artifacts")
    bound_artifacts = selected_model.get("artifacts")
    if not isinstance(manifest_artifacts, Mapping) or not isinstance(bound_artifacts, Mapping):
        raise ValueError("Frozen inference selected-model artifact binding is missing")
    if set(manifest_artifacts) != set(bound_artifacts):
        raise ValueError("Frozen inference selected-model artifact roles differ")
    for name in sorted(bound_artifacts):
        _verify_source_record_matches_pin(
            manifest_artifacts[name],
            bound_artifacts[name],
            label=f"Frozen inference selected-model artifact binding ({name})",
        )

    model_manifest = _read_frozen_json(frozen_dir, selected_model["manifest"])
    model_manifest_artifacts = model_manifest.get("artifacts")
    if not isinstance(model_manifest_artifacts, Mapping):
        raise ValueError("Frozen selected-model manifest artifact provenance is missing")
    if set(model_manifest_artifacts) != set(bound_artifacts):
        raise ValueError("Frozen selected-model manifest artifact roles differ")
    for name in sorted(bound_artifacts):
        _verify_source_record_matches_pin(
            model_manifest_artifacts[name],
            bound_artifacts[name],
            label=f"Frozen selected-model manifest artifact binding ({name})",
        )
    _verify_source_record_matches_pin(
        model_manifest.get("implementation"),
        selected_model.get("implementation"),
        label="Frozen selected-model manifest implementation binding",
    )

    training_selection_pin = bound_artifacts.get("training_selection.json")
    if not isinstance(training_selection_pin, Mapping):
        raise ValueError("Frozen selected-model training-selection binding is missing")
    training_selection = _read_frozen_json(frozen_dir, training_selection_pin)
    input_provenance = training_selection.get("input_provenance")
    if not isinstance(input_provenance, Mapping):
        raise ValueError("Frozen selected-model training-input provenance is missing")
    recorded_training_inputs: list[Mapping[str, Any]] = []
    for group in input_provenance.values():
        if not isinstance(group, list) or any(not isinstance(record, Mapping) for record in group):
            raise ValueError("Frozen selected-model training-input provenance is malformed")
        recorded_training_inputs.extend(group)
    bound_training_inputs = selected_model.get("training_inputs")
    if not isinstance(bound_training_inputs, list) or any(
        not isinstance(pin, Mapping) for pin in bound_training_inputs
    ):
        raise ValueError("Frozen selected-model training-input binding is malformed")
    recorded_identities = sorted(
        _source_record_identity(record) for record in recorded_training_inputs
    )
    bound_identities = sorted(
        _source_record_identity(_source_record(pin)) for pin in bound_training_inputs
    )
    if recorded_identities != bound_identities:
        raise ValueError("Frozen selected-model training-input provenance binding mismatch")


def _verify_frozen_inference_context_binding(
    *,
    inference_manifest: Mapping[str, Any],
    inference_binding: Mapping[str, Any],
    frozen_predictions: Mapping[str, Any],
    context_sha256: str,
) -> None:
    expected_roles = {
        "manifest",
        "implementation",
        "metadata",
        "assumptions",
        "requests",
        "replay",
        "detailed_inference",
        "predictions",
    }
    if "prepared_context" in inference_manifest.get("context", {}):
        expected_roles.add("prepared_context")
    if set(inference_binding) != expected_roles:
        raise ValueError("Frozen inference provenance roles differ")
    if inference_binding["predictions"] != frozen_predictions:
        raise ValueError("Frozen inference predictions role binding mismatch")

    context = inference_manifest.get("context")
    if not isinstance(context, Mapping):
        raise ValueError("Frozen inference context binding is missing")
    expected_context_roles = {
        "metadata",
        "assumptions",
        "requests",
        "replay",
        "detailed_inference",
    }
    if "prepared_context" in context:
        expected_context_roles.add("prepared_context")
    if set(context) != expected_context_roles:
        raise ValueError("Frozen inference context roles differ")
    _verify_source_record_matches_pin(
        context["metadata"],
        inference_binding["metadata"],
        label="Frozen inference metadata binding",
    )
    if "prepared_context" in context:
        _verify_source_record_matches_pin(
            context["prepared_context"],
            inference_binding["prepared_context"],
            label="Frozen inference prepared-context binding",
        )

    relative_roles = {
        "assumptions": "assumptions.json",
        "requests": "requests.jsonl",
        "replay": "replay.json",
        "detailed_inference": "inference.jsonl",
    }
    for role, filename in relative_roles.items():
        expected = {
            "path": filename,
            "sha256": str(inference_binding[role]["source_sha256"]),
        }
        if context[role] != expected:
            raise ValueError(f"Frozen inference {role} context binding mismatch")
    if _hash_json(context) != context_sha256:
        raise ValueError("Frozen inference context hash binding mismatch")

    artifacts = inference_manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Frozen inference artifact provenance is missing")
    artifact_roles = {
        "assumptions.json": "assumptions",
        "requests.jsonl": "requests",
        "replay.json": "replay",
        "inference.jsonl": "detailed_inference",
        "predictions.jsonl": "predictions",
    }
    for filename, role in artifact_roles.items():
        expected = {
            "path": filename,
            "sha256": str(inference_binding[role]["source_sha256"]),
        }
        if artifacts.get(filename) != expected:
            raise ValueError(f"Frozen inference artifact binding mismatch ({filename})")


def _verify_source_record_matches_pin(
    record: Any,
    pin: Any,
    *,
    label: str,
) -> None:
    if not isinstance(record, Mapping) or not isinstance(pin, Mapping):
        raise ValueError(f"{label} is missing")
    if _source_record_identity(record) != _source_record_identity(_source_record(pin)):
        raise ValueError(f"{label} mismatch")


def _source_record_identity(record: Mapping[str, Any]) -> tuple[str, str]:
    if set(record) != {"path", "sha256"}:
        raise ValueError("Frozen source record fields differ")
    return str(_resolve_record_path(record).resolve()), str(record["sha256"])


def _read_frozen_json(frozen_dir: Path, pin: Mapping[str, Any]) -> dict[str, Any]:
    namespace_root = frozen_dir.parent.resolve()
    path = (namespace_root / str(pin["snapshot_path_relative_to_namespace"])).resolve()
    if not path.is_relative_to(namespace_root):
        raise ValueError(f"Frozen snapshot escapes namespace: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_inference_manifest(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    expected_count: int,
) -> None:
    if manifest.get("output_count") != expected_count or manifest.get("truth_used") is not False:
        raise ValueError("Inference manifest count/truth gate mismatch")
    for record in manifest["artifacts"].values():
        if isinstance(record, list):
            records = record
        else:
            records = [record]
        for item in records:
            path = root / str(item["path"])
            if _sha256(path) != str(item["sha256"]):
                raise ValueError(f"Inference artifact hash drift: {path}")


def _artifact_records(root: Path) -> dict[str, Any]:
    names = (
        "requests.jsonl",
        "predictions.jsonl",
        "inference.jsonl",
        "assumptions.json",
        "replay.json",
        "contact_sheet.png",
    )
    return {
        **{name: {"path": name, "sha256": _sha256(root / name)} for name in names},
        "overlays": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
            }
            for path in sorted((root / "overlays").glob("measure_*.png"))
        ],
    }


def _write_contact_sheet(paths: Sequence[Path], output_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    try:
        width = max(image.width for image in images)
        title_height = 20
        height = sum(image.height + title_height for image in images)
        sheet = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default()
        y = 0
        for index, image in enumerate(images, start=1):
            draw.text((4, y + 4), f"automatic crop {index} (truth blind)", fill="black", font=font)
            y += title_height
            sheet.paste(image, (0, y))
            y += image.height
        sheet.save(output_path)
    finally:
        for image in images:
            image.close()


def _expected_beats(time_signature: Any) -> float | None:
    if not time_signature:
        return None
    value = str(time_signature)
    try:
        numerator_text, denominator_text = value.split("/", 1)
        return float(Fraction(int(numerator_text) * 4, int(denominator_text)))
    except (ValueError, ZeroDivisionError):
        return None


def _resolve_record_path(record: Mapping[str, Any]) -> Path:
    path = Path(str(record["path"]))
    return path if path.is_absolute() else REPO_ROOT / path


def _verify_file_record(record: Mapping[str, Any], path: Path, *, label: str) -> None:
    expected = path.resolve()
    recorded = _resolve_record_path(record).resolve()
    if recorded != expected:
        raise ValueError(f"{label} path substitution: expected {expected}, got {recorded}")
    if str(record.get("sha256")) != _sha256(expected):
        raise ValueError(f"{label} hash drift: {expected}")


def _verify_relative_record(
    record: Mapping[str, Any],
    root: Path,
    expected_name: str,
    *,
    label: str,
) -> None:
    if str(record.get("path")) != expected_name:
        raise ValueError(f"{label} path substitution: expected {expected_name}")
    path = root / expected_name
    if str(record.get("sha256")) != _sha256(path):
        raise ValueError(f"{label} hash drift: {path}")


def _source_record(pin: Mapping[str, Any]) -> dict[str, str]:
    return {
        "path": str(pin["source_path"]),
        "sha256": str(pin["source_sha256"]),
    }


def _deduplicate_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    deduplicated: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduplicated.append(resolved)
    return tuple(deduplicated)


def _reject_truth_path(path: Path) -> None:
    normalized = "/" + path.resolve().as_posix().lower().lstrip("/")
    if any(marker in normalized for marker in TRUTH_PATH_MARKERS):
        raise ValueError(f"Truth/MusicXML path is forbidden during held-out inference: {path}")


def _reject_target_truth_path(path: Path, *, target_slug: str) -> None:
    for candidate in (path.absolute(), path.resolve()):
        parts = tuple(part.casefold() for part in candidate.parts)
        if freezer._is_forbidden_target_truth_path(parts, target_slug=target_slug):
            raise ValueError(f"Held-out target truth/MusicXML path is forbidden: {path}")


def _validate_output_name(value: str) -> None:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Invalid inference directory name: {value!r}")


def _read_json(path: Path) -> dict[str, Any]:
    _reject_truth_path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    _reject_truth_path(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSON objects in: {path}")
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _file_record(path: Path) -> dict[str, str]:
    return {"path": _display_path(path), "sha256": _sha256(path)}


def _display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _hash_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
