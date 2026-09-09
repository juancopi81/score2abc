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
from scripts.experiments import freeze_independent_dyad_recovery_gate as dyad_freezer  # noqa: E402
from scripts.experiments import (  # noqa: E402
    freeze_independent_full_event_gate as full_event_freezer,
)
from scripts.experiments import freeze_independent_key_state_gates as key_freezer  # noqa: E402
from scripts.experiments import (  # noqa: E402
    freeze_independent_multihead_recovery_gate as multihead_freezer,
)
from scripts.experiments import (  # noqa: E402
    freeze_independent_sparse_dyad_repair_gate as sparse_dyad_freezer,
)
from scripts.experiments import freeze_third_score_heldout as freezer  # noqa: E402
from scripts.experiments import spike_composed_melody_chain as composed  # noqa: E402
from scripts.experiments import spike_consumed_polyphonic_pitch_repair as recovery  # noqa: E402
from scripts.experiments import spike_meter_gap_resolver as gap  # noqa: E402
from scripts.experiments import spike_notehead_patch_templates as patches  # noqa: E402
from scripts.experiments import spike_review_augmented_selector as dense  # noqa: E402

SCHEMA_VERSION = 1
INFERENCE_VERSION = "third-score-inference-v2"
FOURTH_SCORE_INFERENCE_VERSION = "fourth-score-inference-v1"
FIFTH_SCORE_INFERENCE_VERSION = "fifth-score-inference-v1"
INDEPENDENT_KEY_INFERENCE_VERSION = "independent-key-inference-v1"
INDEPENDENT_DYAD_INFERENCE_VERSION = "independent-dyad-baseline-inference-v1"
INDEPENDENT_MULTIHEAD_INFERENCE_VERSION = "independent-multihead-baseline-inference-v1"
INDEPENDENT_SPARSE_DYAD_INFERENCE_VERSION = "independent-sparse-dyad-baseline-inference-v1"
INDEPENDENT_FULL_EVENT_INFERENCE_VERSION = "independent-full-event-baseline-inference-v1"
MULTIHEAD_RECOVERY_SIDECAR_VERSION = "edge-safe-stem-multihead-sidecar-v1"
MULTIHEAD_RECOVERY_DIRNAME = "edge_safe_stem_multihead_recovery_v1"
SPARSE_DYAD_REPAIR_SIDECAR_VERSION = "sparse-stem-dyad-repair-sidecar-v1"
SPARSE_DYAD_REPAIR_DIRNAME = "sparse_stem_dyad_repair_v1"
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
    dyad_freezer.INDEPENDENT_DYAD_RECOVERY_GATE.prepare_kind: {
        "gate": dyad_freezer.INDEPENDENT_DYAD_RECOVERY_GATE,
        "inference_version": INDEPENDENT_DYAD_INFERENCE_VERSION,
        "manifest_kind": "independent_dyad_truth_blind_baseline_inference_manifest",
        "binding_kind": "independent_dyad_baseline_inference_provenance_binding",
    },
    multihead_freezer.INDEPENDENT_MULTIHEAD_RECOVERY_GATE.prepare_kind: {
        "gate": multihead_freezer.INDEPENDENT_MULTIHEAD_RECOVERY_GATE,
        "inference_version": INDEPENDENT_MULTIHEAD_INFERENCE_VERSION,
        "manifest_kind": "independent_multihead_truth_blind_baseline_inference_manifest",
        "binding_kind": "independent_multihead_baseline_inference_provenance_binding",
    },
    sparse_dyad_freezer.INDEPENDENT_SPARSE_DYAD_REPAIR_GATE.prepare_kind: {
        "gate": sparse_dyad_freezer.INDEPENDENT_SPARSE_DYAD_REPAIR_GATE,
        "inference_version": INDEPENDENT_SPARSE_DYAD_INFERENCE_VERSION,
        "manifest_kind": "independent_sparse_dyad_truth_blind_baseline_inference_manifest",
        "binding_kind": "independent_sparse_dyad_baseline_inference_provenance_binding",
    },
    full_event_freezer.INDEPENDENT_FULL_EVENT_GATE.prepare_kind: {
        "gate": full_event_freezer.INDEPENDENT_FULL_EVENT_GATE,
        "inference_version": INDEPENDENT_FULL_EVENT_INFERENCE_VERSION,
        "manifest_kind": "independent_full_event_truth_blind_baseline_inference_manifest",
        "binding_kind": "independent_full_event_baseline_inference_provenance_binding",
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
    parser.add_argument(
        "--multihead-recovery",
        action="store_true",
        help=(
            "Materialize the spike-only edge-safe stem-aware multi-head recovery sidecar; "
            "requires --no-freeze."
        ),
    )
    parser.add_argument(
        "--sparse-dyad-repair",
        action="store_true",
        help=(
            "Chain the passed sparse dotted-hollow dyad repair after the spike-only "
            "multi-head sidecar; requires --multihead-recovery and --no-freeze."
        ),
    )
    args = parser.parse_args(argv)
    if args.sparse_dyad_repair and (not args.multihead_recovery or not args.no_freeze):
        print(
            "error: --sparse-dyad-repair is spike-only and requires "
            "--multihead-recovery and --no-freeze",
            file=sys.stderr,
        )
        return 1
    if args.multihead_recovery and not args.no_freeze:
        print(
            "error: --multihead-recovery is spike-only and requires --no-freeze; "
            "the canonical baseline freeze does not include optional recovery artifacts",
            file=sys.stderr,
        )
        return 1
    started = time.perf_counter()
    try:
        result = materialize_third_score_inference(
            args.prepared_manifest,
            model_dir=args.model_dir,
            inference_dirname=args.inference_dirname,
            multihead_recovery=args.multihead_recovery,
            sparse_dyad_repair=args.sparse_dyad_repair,
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
    multihead_recovery: bool = False,
    sparse_dyad_repair: bool = False,
) -> dict[str, Any]:
    """Create deterministic truth-blind inference artifacts exactly once."""
    if sparse_dyad_repair and not multihead_recovery:
        raise ValueError("Sparse-dyad repair requires the multi-head recovery sidecar")
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
        optional_lanes = {}
        multihead_result = None
        sparse_dyad_result = None
        if multihead_recovery:
            multihead_result = _materialize_multihead_recovery_sidecar(
                inference_rows,
                model_payload=model_payload,
                model_and_training=validated["pins"],
                inference_root=temp_dir,
                expected_target=prepared["target"],
            )
            optional_lanes["edge_safe_stem_multihead_recovery"] = {
                "path": f"{MULTIHEAD_RECOVERY_DIRNAME}/manifest.json",
                "sha256": _sha256(temp_dir / MULTIHEAD_RECOVERY_DIRNAME / "manifest.json"),
            }
        if sparse_dyad_repair:
            sparse_dyad_result = _materialize_sparse_dyad_repair_sidecar(
                inference_rows,
                model_payload=model_payload,
                model_and_training=validated["pins"],
                inference_root=temp_dir,
                expected_target=prepared["target"],
            )
            optional_lanes["sparse_stem_dyad_repair"] = {
                "path": f"{SPARSE_DYAD_REPAIR_DIRNAME}/manifest.json",
                "sha256": _sha256(temp_dir / SPARSE_DYAD_REPAIR_DIRNAME / "manifest.json"),
            }
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
        if optional_lanes:
            manifest["status"] = "inferred_spike_only_no_freeze"
            manifest["optional_lanes"] = optional_lanes
        _write_json(temp_dir / "manifest.json", manifest)
        temp_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    result = {
        "inference_dir": str(output_dir),
        "manifest": str(output_dir / "manifest.json"),
        "manifest_sha256": _sha256(output_dir / "manifest.json"),
        "predictions": str(output_dir / "predictions.jsonl"),
        "predictions_sha256": _sha256(output_dir / "predictions.jsonl"),
        "inference_sha256": _sha256(output_dir / "inference.jsonl"),
        "output_count": expected_count,
        "warnings": assumptions["warnings"],
    }
    if multihead_result is not None:
        result["multihead_recovery"] = {
            **multihead_result,
            "sidecar_dir": str(output_dir / MULTIHEAD_RECOVERY_DIRNAME),
            "manifest": str(output_dir / MULTIHEAD_RECOVERY_DIRNAME / "manifest.json"),
        }
    if sparse_dyad_result is not None:
        result["sparse_dyad_repair"] = {
            **sparse_dyad_result,
            "sidecar_dir": str(output_dir / SPARSE_DYAD_REPAIR_DIRNAME),
            "manifest": str(output_dir / SPARSE_DYAD_REPAIR_DIRNAME / "manifest.json"),
        }
    return result


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
    if manifest.get("optional_lanes"):
        raise ValueError(
            "Inference contains spike-only optional recovery artifacts and cannot use the "
            "canonical freeze; materialize the baseline without optional recovery flags"
        )
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


def _materialize_multihead_recovery_sidecar(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_payload: Mapping[str, Any],
    model_and_training: Mapping[str, Any],
    inference_root: Path,
    expected_target: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize an additive candidate lane without changing canonical predictions."""
    output_dir = inference_root / MULTIHEAD_RECOVERY_DIRNAME
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite multi-head recovery sidecar: {output_dir}")
    selector = recovery.selector_config_from_model(model_payload)
    output_dir.mkdir()
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir()
    lane_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    overlay_paths: list[Path] = []
    recovered_head_count = 0

    config = {
        "schema_version": SCHEMA_VERSION,
        "kind": "edge_safe_stem_multihead_recovery_configuration",
        "version": MULTIHEAD_RECOVERY_SIDECAR_VERSION,
        "config_id": recovery.EDGE_SAFE_STEM_MULTIHEAD_CONFIG_ID,
        "parameters": dict(recovery.EDGE_SAFE_STEM_MULTIHEAD_PARAMETERS),
        "selector": selector,
        "truth_accessed": False,
        "truth_used": False,
    }
    _write_json(output_dir / "config.json", config)

    for row in rows:
        lane, diagnostics, baseline, recovered_candidates = _multihead_recovery_row(
            row,
            selector=selector,
            expected_target=expected_target,
        )
        lane_rows.append(lane)
        diagnostics_rows.append(diagnostics)
        recovered_head_count += len(recovered_candidates)
        measure_index = int(row["identity"]["automatic_measure_index"])
        overlay_path = overlay_dir / f"measure_{measure_index:03d}.png"
        _write_multihead_recovery_overlay(
            row,
            baseline=baseline,
            recovered_candidates=recovered_candidates,
            output_path=overlay_path,
        )
        overlay_paths.append(overlay_path)

    _write_jsonl(output_dir / "recovery_lane.jsonl", lane_rows)
    _write_jsonl(output_dir / "diagnostics.jsonl", diagnostics_rows)
    _write_contact_sheet(overlay_paths, output_dir / "contact_sheet.png")
    artifacts = _recursive_artifact_records(output_dir)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "edge_safe_stem_multihead_recovery_sidecar_manifest",
        "version": MULTIHEAD_RECOVERY_SIDECAR_VERSION,
        "status": "spike_only_not_canonical",
        "truth_accessed": False,
        "truth_used": False,
        "create_once": True,
        "target": dict(expected_target),
        "measure_count": len(lane_rows),
        "recovered_head_count": recovered_head_count,
        "contract": {
            "baseline_predictions_unchanged": True,
            "baseline_canonical_predictions_unchanged": True,
            "additive_candidate_selection_only": True,
            "recovered_candidates_reuse_existing_onset_groups": True,
            "canonical_pitch_and_rhythm_recomposition_applied": False,
            "freeze_supported": False,
        },
        "baseline": {
            "predictions": {
                "path": "../predictions.jsonl",
                "sha256": _sha256(inference_root / "predictions.jsonl"),
            },
            "detailed_inference": {
                "path": "../inference.jsonl",
                "sha256": _sha256(inference_root / "inference.jsonl"),
            },
        },
        "model_and_training": dict(model_and_training),
        "implementation": _file_record(Path(__file__)),
        "recovery_implementation": _file_record(Path(recovery.__file__)),
        "artifacts": artifacts,
    }
    _write_json(output_dir / "manifest.json", manifest)
    _verify_multihead_recovery_sidecar(output_dir)
    return {
        "manifest_sha256": _sha256(output_dir / "manifest.json"),
        "recovery_lane_sha256": _sha256(output_dir / "recovery_lane.jsonl"),
        "diagnostics_sha256": _sha256(output_dir / "diagnostics.jsonl"),
        "recovered_head_count": recovered_head_count,
        "measure_count": len(lane_rows),
    }


def _multihead_recovery_row(
    row: Mapping[str, Any],
    *,
    selector: Mapping[str, Any],
    expected_target: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if row.get("truth_used") is not False:
        raise ValueError("Multi-head recovery input is not truth-blind")
    identity = row.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("Multi-head recovery input has no identity")
    if str(identity.get("slug")) != str(expected_target["slug"]) or int(
        identity.get("system_index", -1)
    ) != int(expected_target["system_index"]):
        raise ValueError("Multi-head recovery target substitution")

    canonical = row.get("canonical_prediction")
    if not isinstance(canonical, Mapping):
        raise ValueError("Multi-head recovery input has no canonical prediction")
    baseline = recovery.select_candidates(row, selector)
    _verify_multihead_baseline(row, baseline, canonical)
    stem_features, stem_metadata = recovery.candidate_local_stem_features(row)
    recovered_candidates = recovery.recover_edge_safe_stem_aware_multihead_candidates(
        row,
        selector,
        baseline,
        stem_features=stem_features,
        **recovery.EDGE_SAFE_STEM_MULTIHEAD_PARAMETERS,
    )
    group_by_id = _multihead_group_indices(row, selector=selector, baseline=baseline)
    maximum_per_group = int(
        recovery.EDGE_SAFE_STEM_MULTIHEAD_PARAMETERS["maximum_recovered_heads_per_group"]
    )
    recovered_group_counts: dict[int, int] = {}
    baseline_ids = {str(candidate["candidate_id"]) for candidate in baseline}
    recovered_ids: set[str] = set()
    for candidate in recovered_candidates:
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in baseline_ids or candidate_id in recovered_ids:
            raise ValueError("Multi-head recovery duplicated a candidate")
        recovered_ids.add(candidate_id)
        group_index = int(candidate["recovery_group_index"])
        if group_index not in set(group_by_id.values()):
            raise ValueError("Multi-head recovery referenced a nonexistent baseline group")
        recovered_group_counts[group_index] = recovered_group_counts.get(group_index, 0) + 1
    if any(count > maximum_per_group for count in recovered_group_counts.values()):
        raise ValueError("Multi-head recovery exceeded the per-group companion cap")

    baseline_lane = [
        _multihead_candidate_record(
            candidate,
            recovered=False,
            onset_group_index=group_by_id[str(candidate["candidate_id"])],
        )
        for candidate in baseline
    ]
    additive_lane = [
        *baseline_lane,
        *[
            _multihead_candidate_record(
                candidate,
                recovered=True,
                onset_group_index=int(candidate["recovery_group_index"]),
            )
            for candidate in recovered_candidates
        ],
    ]
    additive_lane.sort(key=_multihead_candidate_sort_key)
    baseline_lane.sort(key=_multihead_candidate_sort_key)
    lane = {
        "schema_version": SCHEMA_VERSION,
        "identity": dict(identity),
        "truth_accessed": False,
        "truth_used": False,
        "source": dict(row["source"]),
        "lanes": {
            "baseline_generic": {
                "canonical_prediction": dict(canonical),
                "canonical_prediction_sha256": _hash_json(canonical),
                "candidate_lane": baseline_lane,
            },
            "edge_safe_stem_multihead_recovery": {
                "config_id": recovery.EDGE_SAFE_STEM_MULTIHEAD_CONFIG_ID,
                "candidate_lane": additive_lane,
                "recovered_candidate_ids": sorted(recovered_ids),
                "recovered_head_count": len(recovered_candidates),
                "canonical_prediction_materialized": False,
            },
        },
    }
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "identity": dict(identity),
        "truth_accessed": False,
        "truth_used": False,
        "baseline_candidate_ids": sorted(baseline_ids),
        "recovered_candidate_ids": sorted(recovered_ids),
        "baseline_onset_group_count": len(set(group_by_id.values())),
        "recovered_onset_group_count": len(set(group_by_id.values())),
        "baseline_anchor_count": len(baseline),
        "canonical_note_count": len(canonical["notes"]),
        "meter_decoder_dropped_anchor_count": len(baseline) - len(canonical["notes"]),
        "recovery": [
            {
                "candidate_id": str(candidate["candidate_id"]),
                "center": dict(candidate["center"]),
                "score": float(candidate["score"]),
                "onset_group_index": int(candidate["recovery_group_index"]),
                "y_gap_staff_spaces": float(candidate["recovery_y_gap_staff_spaces"]),
                "score_ratio": candidate["recovery_score_ratio"],
                "stem_attachment_score": float(candidate["stem_attachment_score"]),
                "leading_edge_distance_staff_spaces": float(
                    candidate["leading_edge_distance_staff_spaces"]
                ),
            }
            for candidate in recovered_candidates
        ],
        "candidate_stem_features": {
            candidate_id: dict(stem_features[candidate_id])
            for candidate_id in sorted(stem_features)
        },
        "stem_feature_metadata": stem_metadata,
    }
    return lane, diagnostics, baseline, recovered_candidates


def _verify_multihead_baseline(
    row: Mapping[str, Any],
    baseline: Sequence[Mapping[str, Any]],
    canonical: Mapping[str, Any],
) -> None:
    notes = canonical.get("notes")
    if not isinstance(notes, list):
        raise ValueError("Canonical baseline has no notes")
    anchors = row.get("automatic_anchors")
    if not isinstance(anchors, list):
        raise ValueError("Detailed inference has no automatic anchors")
    expected = {
        str(candidate["candidate_id"]): (
            round(float(candidate["center"]["x"]), 3),
            round(float(candidate["center"]["y"]), 3),
        )
        for candidate in baseline
    }
    actual = {
        str(anchor["source"]["candidate_id"]): (
            round(float(anchor["center"]["x"]), 3),
            round(float(anchor["center"]["y"]), 3),
        )
        for anchor in anchors
    }
    if len(actual) != len(anchors) or actual != expected:
        raise ValueError("Recovery selector does not reproduce the canonical baseline")
    # Meter decoding may drop selected trailing anchors that cannot fit the
    # request meter. Recovery remains anchored to the exact pre-decoder IDs and
    # coordinates, while the canonical event lane stays byte-for-byte unchanged.
    if len(notes) > len(anchors):
        raise ValueError("Canonical note count exceeds the automatic baseline anchors")
    if row.get("truth_used") is not False:
        raise ValueError("Canonical baseline is not truth-blind")


def _multihead_group_indices(
    row: Mapping[str, Any],
    *,
    selector: Mapping[str, Any],
    baseline: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    x_radius = recovery._staff_spacing(row) * float(selector["nms_x_spaces"])
    groups = recovery._horizontal_candidate_groups(baseline, x_radius)
    result = {
        str(candidate["candidate_id"]): group_index
        for group_index, group in enumerate(groups, start=1)
        for candidate in group
    }
    if len(result) != len(baseline):
        raise ValueError("Multi-head baseline grouping lost candidate identity")
    return result


def _multihead_candidate_record(
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


def _multihead_candidate_sort_key(
    candidate: Mapping[str, Any],
) -> tuple[float, float, str]:
    return (
        float(candidate["center"]["x"]),
        float(candidate["center"]["y"]),
        str(candidate["candidate_id"]),
    )


def _write_multihead_recovery_overlay(
    row: Mapping[str, Any],
    *,
    baseline: Sequence[Mapping[str, Any]],
    recovered_candidates: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    image_path = Path(str(row["source"]["image"]))
    if not image_path.is_absolute():
        image_path = REPO_ROOT / image_path
    if _sha256(image_path) != str(row["source"]["sha256"]):
        raise ValueError(f"Multi-head overlay source hash drift: {image_path}")
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.text((4, 4), "blue: baseline | green: recovered", fill=(0, 0, 0))
    for color, candidates in (
        ((20, 90, 220), baseline),
        ((0, 160, 70), recovered_candidates),
    ):
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


def _recursive_artifact_records(root: Path) -> dict[str, dict[str, str]]:
    return {
        path.relative_to(root).as_posix(): {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def _verify_multihead_recovery_sidecar(output_dir: Path) -> None:
    manifest = _read_json(output_dir / "manifest.json")
    if (
        manifest.get("kind") != "edge_safe_stem_multihead_recovery_sidecar_manifest"
        or manifest.get("version") != MULTIHEAD_RECOVERY_SIDECAR_VERSION
        or manifest.get("truth_used") is not False
    ):
        raise ValueError("Multi-head recovery sidecar manifest contract mismatch")
    _verify_sidecar_artifact_inventory(
        output_dir,
        manifest.get("artifacts"),
        label="Multi-head recovery sidecar",
    )
    _verify_named_relative_records(
        manifest.get("baseline"),
        output_dir,
        {
            "predictions": "../predictions.jsonl",
            "detailed_inference": "../inference.jsonl",
        },
        label="Multi-head recovery baseline",
    )


def _materialize_sparse_dyad_repair_sidecar(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_payload: Mapping[str, Any],
    model_and_training: Mapping[str, Any],
    inference_root: Path,
    expected_target: Mapping[str, Any],
) -> dict[str, Any]:
    """Chain the fixed sparse-dyad replacement rule after the multi-head lane."""
    sparse_repair = _sparse_repair_module()
    upstream_dir = inference_root / MULTIHEAD_RECOVERY_DIRNAME
    _verify_multihead_recovery_sidecar(upstream_dir)
    upstream_rows = _read_jsonl(upstream_dir / "recovery_lane.jsonl")
    if len(upstream_rows) != len(rows):
        raise ValueError("Sparse-dyad input/multi-head row count mismatch")

    output_dir = inference_root / SPARSE_DYAD_REPAIR_DIRNAME
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite sparse-dyad repair sidecar: {output_dir}")
    selector = recovery.selector_config_from_model(model_payload)
    output_dir.mkdir()
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir()
    lane_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    overlay_paths: list[Path] = []
    accepted_repair_count = 0

    config = {
        "schema_version": SCHEMA_VERSION,
        "kind": "sparse_stem_dyad_repair_configuration",
        "version": SPARSE_DYAD_REPAIR_SIDECAR_VERSION,
        "config_id": sparse_repair.CONFIG_ID,
        "parameters": dict(sparse_repair.PARAMETERS),
        "selector": selector,
        "truth_accessed": False,
        "truth_used": False,
    }
    _write_json(output_dir / "config.json", config)

    for row, upstream_row in zip(rows, upstream_rows, strict=True):
        lane, diagnostics, current, repaired = _sparse_dyad_repair_row(
            row,
            upstream_row=upstream_row,
            selector=selector,
            expected_target=expected_target,
        )
        lane_rows.append(lane)
        diagnostics_rows.append(diagnostics)
        accepted_repair_count += int(lane["lanes"]["sparse_dyad_repair"]["accepted"])
        measure_index = int(row["identity"]["automatic_measure_index"])
        overlay_path = overlay_dir / f"measure_{measure_index:03d}.png"
        _write_sparse_dyad_repair_overlay(
            row,
            current=current,
            repaired=repaired,
            output_path=overlay_path,
        )
        overlay_paths.append(overlay_path)

    _write_jsonl(output_dir / "repair_lane.jsonl", lane_rows)
    _write_jsonl(output_dir / "diagnostics.jsonl", diagnostics_rows)
    _write_contact_sheet(overlay_paths, output_dir / "contact_sheet.png")
    artifacts = _recursive_artifact_records(output_dir)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "sparse_stem_dyad_repair_sidecar_manifest",
        "version": SPARSE_DYAD_REPAIR_SIDECAR_VERSION,
        "status": "spike_only_not_canonical",
        "truth_accessed": False,
        "truth_used": False,
        "create_once": True,
        "target": dict(expected_target),
        "measure_count": len(lane_rows),
        "accepted_repair_count": accepted_repair_count,
        "contract": {
            "baseline_predictions_unchanged": True,
            "baseline_canonical_predictions_unchanged": True,
            "chains_exact_multihead_candidate_lane": True,
            "replacement_candidate_selection_only": True,
            "accepted_repairs_require_paired_augmentation_dot_evidence": True,
            "accepted_repairs_form_one_onset_group": True,
            "canonical_pitch_and_rhythm_recomposition_applied": False,
            "freeze_supported": False,
        },
        "baseline": {
            "predictions": {
                "path": "../predictions.jsonl",
                "sha256": _sha256(inference_root / "predictions.jsonl"),
            },
            "detailed_inference": {
                "path": "../inference.jsonl",
                "sha256": _sha256(inference_root / "inference.jsonl"),
            },
        },
        "upstream_multihead": {
            "manifest": {
                "path": f"../{MULTIHEAD_RECOVERY_DIRNAME}/manifest.json",
                "sha256": _sha256(upstream_dir / "manifest.json"),
            },
            "recovery_lane": {
                "path": f"../{MULTIHEAD_RECOVERY_DIRNAME}/recovery_lane.jsonl",
                "sha256": _sha256(upstream_dir / "recovery_lane.jsonl"),
            },
        },
        "model_and_training": dict(model_and_training),
        "implementation": _file_record(Path(__file__)),
        "repair_implementation": _file_record(Path(sparse_repair.__file__)),
        "artifacts": artifacts,
    }
    _write_json(output_dir / "manifest.json", manifest)
    _verify_sparse_dyad_repair_sidecar(output_dir)
    return {
        "manifest_sha256": _sha256(output_dir / "manifest.json"),
        "repair_lane_sha256": _sha256(output_dir / "repair_lane.jsonl"),
        "diagnostics_sha256": _sha256(output_dir / "diagnostics.jsonl"),
        "accepted_repair_count": accepted_repair_count,
        "measure_count": len(lane_rows),
    }


def _sparse_dyad_repair_row(
    row: Mapping[str, Any],
    *,
    upstream_row: Mapping[str, Any],
    selector: Mapping[str, Any],
    expected_target: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    sparse_repair = _sparse_repair_module()
    if row.get("truth_used") is not False or upstream_row.get("truth_used") is not False:
        raise ValueError("Sparse-dyad repair input is not truth-blind")
    identity = row.get("identity")
    if not isinstance(identity, Mapping) or upstream_row.get("identity") != identity:
        raise ValueError("Sparse-dyad repair identity mismatch")
    if str(identity.get("slug")) != str(expected_target["slug"]) or int(
        identity.get("system_index", -1)
    ) != int(expected_target["system_index"]):
        raise ValueError("Sparse-dyad repair target substitution")

    upstream_lane = upstream_row.get("lanes", {}).get("edge_safe_stem_multihead_recovery")
    if not isinstance(upstream_lane, Mapping):
        raise ValueError("Sparse-dyad repair has no upstream multi-head lane")
    upstream_candidates = upstream_lane.get("candidate_lane")
    if not isinstance(upstream_candidates, list):
        raise ValueError("Sparse-dyad repair upstream candidate lane is invalid")
    candidate_by_id = {
        str(candidate["candidate_id"]): candidate
        for candidate in sparse_repair._normalized_candidates(row)
    }
    current = []
    for recorded in upstream_candidates:
        candidate_id = str(recorded["candidate_id"])
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            raise ValueError("Sparse-dyad upstream lane introduced an unknown candidate")
        expected_center = (
            round(float(candidate["center"]["x"]), 3),
            round(float(candidate["center"]["y"]), 3),
        )
        actual_center = (
            round(float(recorded["center"]["x"]), 3),
            round(float(recorded["center"]["y"]), 3),
        )
        if actual_center != expected_center or float(recorded["score"]) != float(
            candidate["score"]
        ):
            raise ValueError("Sparse-dyad upstream candidate identity drift")
        current.append(candidate)

    stem_features, stem_metadata = recovery.candidate_local_stem_features(row)
    decision = sparse_repair.propose_sparse_shared_stem_dyad(
        row,
        selector,
        current,
        stem_features,
    )
    repaired = current
    if decision["accepted"]:
        repaired = [candidate_by_id[candidate_id] for candidate_id in decision["proposed_ids"]]
        chosen = decision.get("chosen_pair")
        if not isinstance(chosen, Mapping) or not chosen.get("augmentation_dot_pairs"):
            raise ValueError("Accepted sparse-dyad repair lacks augmentation-dot evidence")

    current_ids = {str(candidate["candidate_id"]) for candidate in current}
    repaired_ids = {str(candidate["candidate_id"]) for candidate in repaired}
    group_by_id = _sparse_group_indices(row, selector=selector, candidates=repaired)
    if decision["accepted"] and len(set(group_by_id.values())) != 1:
        raise ValueError("Accepted sparse-dyad repair did not form one onset group")
    repaired_lane = [
        _sparse_candidate_record(
            candidate,
            onset_group_index=group_by_id[str(candidate["candidate_id"])],
            sparse_repair_added=str(candidate["candidate_id"]) not in current_ids,
        )
        for candidate in repaired
    ]
    repaired_lane.sort(key=_multihead_candidate_sort_key)
    lane = {
        "schema_version": SCHEMA_VERSION,
        "identity": dict(identity),
        "truth_accessed": False,
        "truth_used": False,
        "source": dict(row["source"]),
        "lanes": {
            "edge_safe_stem_multihead_recovery": dict(upstream_lane),
            "sparse_dyad_repair": {
                "config_id": sparse_repair.CONFIG_ID,
                "candidate_lane": repaired_lane,
                "accepted": bool(decision["accepted"]),
                "added_candidate_ids": sorted(repaired_ids - current_ids),
                "displaced_candidate_ids": sorted(current_ids - repaired_ids),
                "canonical_prediction_materialized": False,
            },
        },
    }
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "identity": dict(identity),
        "truth_accessed": False,
        "truth_used": False,
        "upstream_candidate_ids": sorted(current_ids),
        "repaired_candidate_ids": sorted(repaired_ids),
        "sparse_repair": decision,
        "candidate_stem_features": {
            candidate_id: dict(stem_features[candidate_id])
            for candidate_id in sorted(stem_features)
        },
        "stem_feature_metadata": stem_metadata,
    }
    return lane, diagnostics, current, repaired


def _sparse_group_indices(
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


def _sparse_candidate_record(
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


def _write_sparse_dyad_repair_overlay(
    row: Mapping[str, Any],
    *,
    current: Sequence[Mapping[str, Any]],
    repaired: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    image_path = Path(str(row["source"]["image"]))
    if not image_path.is_absolute():
        image_path = REPO_ROOT / image_path
    if _sha256(image_path) != str(row["source"]["sha256"]):
        raise ValueError(f"Sparse-dyad overlay source hash drift: {image_path}")
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.text((4, 4), "blue: retained | green: added | red: displaced", fill=(0, 0, 0))
    current_by_id = {str(candidate["candidate_id"]): candidate for candidate in current}
    repaired_by_id = {str(candidate["candidate_id"]): candidate for candidate in repaired}
    categories = (
        ((20, 90, 220), set(current_by_id) & set(repaired_by_id), current_by_id),
        ((0, 160, 70), set(repaired_by_id) - set(current_by_id), repaired_by_id),
        ((210, 35, 35), set(current_by_id) - set(repaired_by_id), current_by_id),
    )
    for color, candidate_ids, candidates in categories:
        for candidate_id in sorted(candidate_ids):
            source = candidates[candidate_id].get("source")
            bbox = source.get("bbox") if isinstance(source, Mapping) else None
            if not isinstance(bbox, Mapping):
                continue
            bounds = tuple(int(bbox[key]) for key in ("left", "top", "right", "bottom"))
            draw.rectangle(bounds, outline=color, width=2)
            draw.text((bounds[0], max(0, bounds[1] - 11)), candidate_id, fill=color)
    image.save(output_path)


def _verify_sparse_dyad_repair_sidecar(output_dir: Path) -> None:
    sparse_repair = _sparse_repair_module()
    manifest = _read_json(output_dir / "manifest.json")
    if (
        manifest.get("kind") != "sparse_stem_dyad_repair_sidecar_manifest"
        or manifest.get("version") != SPARSE_DYAD_REPAIR_SIDECAR_VERSION
        or manifest.get("truth_used") is not False
    ):
        raise ValueError("Sparse-dyad repair sidecar manifest contract mismatch")
    _verify_sidecar_artifact_inventory(
        output_dir,
        manifest.get("artifacts"),
        label="Sparse-dyad repair sidecar",
    )
    _verify_named_relative_records(
        manifest.get("baseline"),
        output_dir,
        {
            "predictions": "../predictions.jsonl",
            "detailed_inference": "../inference.jsonl",
        },
        label="Sparse-dyad repair baseline",
    )
    _verify_named_relative_records(
        manifest.get("upstream_multihead"),
        output_dir,
        {
            "manifest": f"../{MULTIHEAD_RECOVERY_DIRNAME}/manifest.json",
            "recovery_lane": f"../{MULTIHEAD_RECOVERY_DIRNAME}/recovery_lane.jsonl",
        },
        label="Sparse-dyad repair upstream_multihead",
    )

    inference_root = output_dir.parent
    upstream_dir = inference_root / MULTIHEAD_RECOVERY_DIRNAME
    _verify_multihead_recovery_sidecar(upstream_dir)
    config = _read_json(output_dir / "config.json")
    if (
        config.get("config_id") != sparse_repair.CONFIG_ID
        or config.get("parameters") != sparse_repair.PARAMETERS
        or config.get("truth_used") is not False
    ):
        raise ValueError("Sparse-dyad repair configuration drift")
    rows = _read_jsonl(inference_root / "inference.jsonl")
    upstream_rows = _read_jsonl(upstream_dir / "recovery_lane.jsonl")
    lane_rows = _read_jsonl(output_dir / "repair_lane.jsonl")
    diagnostics_rows = _read_jsonl(output_dir / "diagnostics.jsonl")
    if not (len(rows) == len(upstream_rows) == len(lane_rows) == len(diagnostics_rows)):
        raise ValueError("Sparse-dyad repair sidecar row count mismatch")
    accepted_repair_count = 0
    for row, upstream_row, lane, diagnostics in zip(
        rows,
        upstream_rows,
        lane_rows,
        diagnostics_rows,
        strict=True,
    ):
        expected_lane, expected_diagnostics, _, _ = _sparse_dyad_repair_row(
            row,
            upstream_row=upstream_row,
            selector=config["selector"],
            expected_target=manifest["target"],
        )
        if lane != expected_lane or diagnostics != expected_diagnostics:
            raise ValueError("Sparse-dyad repair sidecar drifted from the fixed rule")
        accepted_repair_count += int(lane["lanes"]["sparse_dyad_repair"]["accepted"])
    if int(manifest.get("measure_count", -1)) != len(lane_rows):
        raise ValueError("Sparse-dyad repair manifest measure count mismatch")
    if int(manifest.get("accepted_repair_count", -1)) != accepted_repair_count:
        raise ValueError("Sparse-dyad repair manifest accepted count mismatch")


def _sparse_repair_module() -> Any:
    # The consumed experiment imports held-out evaluators, so load it only after this
    # reusable runner is fully initialized to avoid a module-import cycle.
    from scripts.experiments import spike_consumed_sparse_stem_dyad_repair

    return spike_consumed_sparse_stem_dyad_repair


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


def _verify_named_relative_records(
    records: Any,
    root: Path,
    expected: Mapping[str, str],
    *,
    label: str,
) -> None:
    if not isinstance(records, Mapping) or set(records) != set(expected):
        raise ValueError(f"{label} record inventory drift")
    for name, relative_path in expected.items():
        record = records[name]
        if not isinstance(record, Mapping):
            raise ValueError(f"{label} record is invalid: {name}")
        _verify_relative_record(record, root, relative_path, label=f"{label} artifact")


def _verify_sidecar_artifact_inventory(
    output_dir: Path,
    records: Any,
    *,
    label: str,
) -> None:
    if not isinstance(records, Mapping):
        raise ValueError(f"{label} artifact inventory is missing")
    actual = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(records) != actual:
        raise ValueError(f"{label} artifact inventory drift")
    for relative_path in sorted(actual):
        record = records[relative_path]
        if not isinstance(record, Mapping):
            raise ValueError(f"{label} artifact record is invalid: {relative_path}")
        _verify_relative_record(
            record,
            output_dir,
            relative_path,
            label=f"{label} artifact",
        )


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
