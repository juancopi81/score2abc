"""Replay Coqueteos with corrected seven-measure segmentation.

This is a create-once consumed-evidence postmortem. It reuses the exact model
and musical context frozen for the fifth-score gate, writes predictions before
opening the now-consumed MusicXML, and compares the corrected seven-crop result
with the immutable six-crop heldout baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from score2abc.events import measure_length_beats  # noqa: E402
from scripts import build_vlm_melody_event_benchmark as benchmark  # noqa: E402
from scripts import build_vlm_melody_inputs as melody_inputs  # noqa: E402
from scripts.experiments import evaluate_frozen_fifth_score_heldout as fifth_eval  # noqa: E402
from scripts.experiments import evaluate_second_score_heldout as truth_tools  # noqa: E402
from scripts.experiments import run_third_score_heldout_inference as inference  # noqa: E402
from scripts.experiments import spike_consumed_meter_deficit_validator as meter  # noqa: E402
from scripts.experiments import spike_consumed_onset_group_selector as onset  # noqa: E402
from scripts.experiments import spike_consumed_polyphonic_pitch_repair as recovery  # noqa: E402

SCHEMA_VERSION = 1
KIND = "consumed_coqueteos_corrected_segmentation_replay"
SLUG = "jaime-llanos_22_coqueteos_pasillo_fulgencio-garcia"
SYSTEM_INDEX = 2
MEASURE_COUNT = 7
DEFAULT_NAMESPACE = "coqueteos_system_002_seg_v2"
DEFAULT_MODEL_DIRNAME = "cross_score_notehead_v1_replay_20260722"
DEFAULT_OUTPUT_DIRNAME = "consumed_corrected_replay_v1"

TruthLoader = Callable[[Path], truth_tools.MusicXMLTruth]
InferenceRunner = Callable[
    [Sequence[Mapping[str, Any]], Mapping[str, Any], Path, Path],
    dict[str, Any],
]
TriageRunner = Callable[..., list[dict[str, Any]]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", type=Path, default=Path("out"))
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        result = run_consumed_replay(
            args.out_dir,
            namespace=args.namespace,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
        )
    except (FileExistsError, FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(result["report"])
    return 0


def run_consumed_replay(
    out_dir: Path,
    *,
    namespace: str = DEFAULT_NAMESPACE,
    output_dir: Path | None = None,
    model_dir: Path | None = None,
    truth_loader: TruthLoader = truth_tools.load_musicxml_truth,
    inference_runner: InferenceRunner | None = None,
    triage_runner: TriageRunner | None = None,
) -> dict[str, str]:
    """Run the corrected replay and publish one atomic consumed bundle."""
    out_root = out_dir.resolve()
    namespace_root = (out_root / SLUG / "vlm_melody_training_inputs" / namespace).resolve()
    heldout_root = (
        out_root / SLUG / "vlm_melody_fifth_score_heldout" / "v1" / "system_002"
    ).resolve()
    destination = (
        output_dir.resolve() if output_dir is not None else namespace_root / DEFAULT_OUTPUT_DIRNAME
    )
    model_root = (
        model_dir.resolve()
        if model_dir is not None
        else out_root / "vlm_melody_consumed_training" / DEFAULT_MODEL_DIRNAME
    )
    temp_dir = destination.with_name(f".{destination.name}.tmp")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite consumed replay: {destination}")
    if temp_dir.exists():
        raise FileExistsError(f"Refusing stale consumed replay temp: {temp_dir}")

    inputs = _validate_prediction_inputs(
        out_root=out_root,
        namespace_root=namespace_root,
        heldout_root=heldout_root,
        model_root=model_root,
    )
    requests = _materialize_requests(
        inputs["input_rows"],
        out_root=out_root,
        allowed_context=inputs["allowed_context"],
        context_provenance=inputs["context"]["provenance"],
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir()
    try:
        _write_jsonl(temp_dir / "requests.jsonl", requests)
        _write_json(temp_dir / "allowed_context.json", inputs["context"])
        runner = inference_runner or _run_inference
        inferred = runner(requests, inputs["model"], out_root, temp_dir)
        predictions = list(inferred["predictions"])
        details = list(inferred["inference"])
        if len(predictions) != MEASURE_COUNT or len(details) != MEASURE_COUNT:
            raise ValueError("Corrected replay must produce exactly seven aligned outputs")
        _write_jsonl(temp_dir / "predictions.jsonl", predictions)
        _write_jsonl(temp_dir / "inference.jsonl", details)
        _write_json(temp_dir / "replay.json", inferred["replay"])
        triage = triage_runner or _triage_predictions
        triage_rows = triage(
            requests,
            details,
            model_payload=inputs["model"],
            metadata=inputs["metadata"],
            out_root=out_root,
            output_dir=temp_dir,
        )
        _write_jsonl(temp_dir / "triage_predictions.jsonl", triage_rows)

        pretruth = {
            "schema_version": SCHEMA_VERSION,
            "kind": f"{KIND}_pretruth",
            "status": "predictions_materialized_before_consumed_truth_open",
            "truth_accessed_for_prediction": False,
            "truth_used_for_prediction": False,
            "target": {"slug": SLUG, "system_index": SYSTEM_INDEX},
            "measure_count": MEASURE_COUNT,
            "pins": {
                "training_namespace": _file_record(namespace_root / "manifest.json"),
                "inputs": _file_record(namespace_root / "inputs_manifest.jsonl"),
                "model": _file_record(model_root / "model.json"),
                "allowed_context": _file_record(inputs["context_path"]),
                "requests": _local_record(temp_dir / "requests.jsonl", temp_dir),
                "predictions": _local_record(temp_dir / "predictions.jsonl", temp_dir),
                "inference": _local_record(temp_dir / "inference.jsonl", temp_dir),
                "replay": _local_record(temp_dir / "replay.json", temp_dir),
                "triage_predictions": _local_record(
                    temp_dir / "triage_predictions.jsonl", temp_dir
                ),
            },
        }
        _write_json(temp_dir / "pretruth_manifest.json", pretruth)

        # This is intentionally the first read of target MusicXML or prior target metrics.
        truth = truth_loader(inputs["musicxml_path"])
        if tuple(truth.measure_numbers) != tuple(range(1, MEASURE_COUNT + 1)):
            raise ValueError(
                f"Expected seven physical MusicXML measures, got {truth.measure_numbers}"
            )
        measure_length = measure_length_beats(str(truth.payload["time_signature"]))
        mapping = {index: (index,) for index in range(1, MEASURE_COUNT + 1)}
        truth_rows = truth_tools.build_mapped_truth_rows(
            requests,
            truth,
            mapping=mapping,
            measure_length=measure_length,
        )
        metrics = benchmark.evaluate_predictions(truth_rows, predictions)
        meter = fifth_eval._evaluate_meter(
            truth=truth,
            truth_rows=truth_rows,
            prediction_rows=predictions,
            mapping=mapping,
            allowed_context=inputs["allowed_context"],
            measure_length=measure_length,
        )
        metrics["summary"].update(
            {
                "meter_context_match": meter["summary"]["context_match"],
                "meter_valid_crops": meter["summary"]["valid_prediction_crops"],
                "meter_valid_crop_rate": meter["summary"]["valid_prediction_crop_rate"],
            }
        )
        baseline = _read_json(inputs["baseline_report_path"])
        baseline_summary = dict(baseline["metrics"]["summary"])
        corrected_summary = dict(metrics["summary"])
        comparison = _comparison(baseline_summary, corrected_summary)
        corrected_triage = _evaluate_triage(triage_rows, metrics["results"])
        baseline_triage_report = _read_json(inputs["baseline_triage_report_path"])
        baseline_flags = {
            int(value) for value in baseline_triage_report["flagged_automatic_measure_indices"]
        }
        baseline_triage = _evaluate_baseline_triage(
            baseline_flags,
            baseline["metrics"]["results"],
        )
        priorities = _coordinate_review_priorities(metrics["results"], triage_rows)

        _write_jsonl(temp_dir / "truth.jsonl", truth_rows)
        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "status": "evaluated_consumed_postmortem_create_once",
            "target": {"slug": SLUG, "system_index": SYSTEM_INDEX},
            "evidence_scope": {
                "independent_heldout": False,
                "consumed_postmortem": True,
                "truth_used_for_prediction": False,
                "truth_used_for_evaluation": True,
                "selector_training_changed": False,
                "frozen_baseline_mutated": False,
            },
            "experiment": {
                "question": (
                    "How much does corrected one-crop-per-measure segmentation improve "
                    "the exact frozen Coqueteos recognizer?"
                ),
                "baseline_crop_count": 6,
                "corrected_crop_count": 7,
                "mapping": "one corrected crop to one physical MusicXML measure",
                "model_sha256": _sha256(model_root / "model.json"),
                "allowed_context": inputs["allowed_context"],
            },
            "baseline": {
                "report": _file_record(inputs["baseline_report_path"]),
                "summary": baseline_summary,
            },
            "corrected": {"summary": corrected_summary, "metrics": metrics, "meter": meter},
            "comparison": comparison,
            "review_triage": {
                "baseline": baseline_triage,
                "corrected": corrected_triage,
                "policy": "review sidecar only; canonical predictions are unchanged",
            },
            "coordinate_review_priorities": priorities,
            "next_gate": _next_gate(comparison),
        }
        _write_json(temp_dir / "report.json", report)
        _write_markdown(temp_dir / "report.md", report)
        _write_json(
            temp_dir / "consumption_mapping.json",
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "vlm_melody_consumed_training_mapping",
                "split_status": "consumed_training",
                "identity": {"slug": SLUG, "system_index": SYSTEM_INDEX},
                "consumption": {
                    "source_split_status": "fresh_heldout",
                    "reason": (
                        "The sealed fifth-score result is explicitly consumed for corrected "
                        "segmentation replay and unreviewed candidate proposals."
                    ),
                    "evaluation_evidence": _local_record(temp_dir / "report.json", temp_dir),
                },
                "source": {
                    "musicxml": _file_record(inputs["musicxml_path"]),
                    "requests": _local_record(temp_dir / "requests.jsonl", temp_dir),
                },
                "crops": [
                    {
                        "system_measure_index": index,
                        "physical_measure_numbers": [index],
                    }
                    for index in range(1, MEASURE_COUNT + 1)
                ],
            },
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": f"{KIND}_manifest",
            "status": "evaluated_consumed_postmortem_create_once",
            "create_once": True,
            "target": report["target"],
            "pretruth_manifest_sha256": _sha256(temp_dir / "pretruth_manifest.json"),
            "pins": {
                "source_musicxml": _file_record(inputs["musicxml_path"]),
                "baseline_report": _file_record(inputs["baseline_report_path"]),
                "baseline_triage_report": _file_record(inputs["baseline_triage_report_path"]),
                "pretruth_manifest": _local_record(temp_dir / "pretruth_manifest.json", temp_dir),
                "consumption_mapping": _local_record(
                    temp_dir / "consumption_mapping.json", temp_dir
                ),
                "truth": _local_record(temp_dir / "truth.jsonl", temp_dir),
                "report": _local_record(temp_dir / "report.json", temp_dir),
                "implementation": _file_record(Path(__file__)),
            },
        }
        _write_json(temp_dir / "manifest.json", manifest)
        temp_dir.rename(destination)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return {
        "output_dir": str(destination),
        "report": str(destination / "report.json"),
        "contact_sheet": str(destination / "contact_sheet.png"),
        "pretruth_manifest": str(destination / "pretruth_manifest.json"),
        "consumption_mapping": str(destination / "consumption_mapping.json"),
    }


def _validate_prediction_inputs(
    *,
    out_root: Path,
    namespace_root: Path,
    heldout_root: Path,
    model_root: Path,
) -> dict[str, Any]:
    required = {
        "namespace_manifest": namespace_root / "manifest.json",
        "inputs": namespace_root / "inputs_manifest.jsonl",
        "context": heldout_root / "context/allowed_context.json",
        "musicxml": heldout_root / "coqueteos_system_002.musicxml",
        "baseline_report": heldout_root / "evaluation_v1/report.json",
        "baseline_triage_report": heldout_root / "pretruth_meter_triage_v1/report.json",
        "frozen_freeze": heldout_root / "frozen/freeze.json",
        "model": model_root / "model.json",
        "metadata": out_root / SLUG / "metadata.json",
    }
    for label, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    namespace = _read_json(required["namespace_manifest"])
    if namespace.get("kind") != "vlm_melody_consumed_cross_score_training_inputs":
        raise ValueError("Unsupported corrected training-input namespace")
    if namespace.get("identity") != {"slug": SLUG, "system_index": SYSTEM_INDEX}:
        raise ValueError("Corrected namespace target mismatch")
    if int(namespace.get("measure_count", -1)) != MEASURE_COUNT:
        raise ValueError("Corrected namespace must contain exactly seven measures")
    if namespace.get("eligible_for_training") is not False:
        raise ValueError("Unreviewed corrected namespace must remain ineligible for training")

    rows = _read_jsonl(required["inputs"])
    identities = [
        (row.get("slug"), int(row["system_index"]), int(row["system_measure_index"]))
        for row in rows
    ]
    expected = [(SLUG, SYSTEM_INDEX, index) for index in range(1, MEASURE_COUNT + 1)]
    if identities != expected:
        raise ValueError(f"Corrected input identities changed: {identities}")
    for row in rows:
        image_path = Path(str(row["paths"]["measure_raw"])).resolve()
        if not image_path.is_relative_to(out_root) or not image_path.is_file():
            raise ValueError(f"Corrected crop escaped output root: {image_path}")

    context = _read_json(required["context"])
    if context.get("truth_accessed") is not False or context.get("truth_used") is not False:
        raise ValueError("Frozen fifth-score context is not truth-blind")
    allowed_context = context.get("allowed_context")
    if not isinstance(allowed_context, Mapping):
        raise ValueError("Frozen fifth-score allowed_context is malformed")

    model = _read_json(required["model"])
    metadata = _read_json(required["metadata"])
    frozen = _read_json(required["frozen_freeze"])
    frozen_model_hashes = {
        str(record["source_sha256"])
        for record in frozen.get("model_artifacts", [])
        if str(record.get("source_path", "")).endswith("/model.json")
    }
    if frozen_model_hashes != {_sha256(required["model"])}:
        raise ValueError("Replay model differs from the model frozen for Coqueteos")
    return {
        "input_rows": rows,
        "context": context,
        "allowed_context": dict(allowed_context),
        "context_path": required["context"],
        "model": model,
        "metadata": metadata,
        "musicxml_path": required["musicxml"],
        "baseline_report_path": required["baseline_report"],
        "baseline_triage_report_path": required["baseline_triage_report"],
    }


def _materialize_requests(
    rows: Sequence[Mapping[str, Any]],
    *,
    out_root: Path,
    allowed_context: Mapping[str, Any],
    context_provenance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    requests = []
    for index, row in enumerate(rows, start=1):
        image_path = Path(str(row["paths"]["measure_raw"])).resolve()
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            staff = melody_inputs._estimate_staff(image)
            width, height = image.size
        if len(staff.line_ys) != 5:
            raise ValueError(f"Expected five staff lines in corrected crop: {image_path}")
        requests.append(
            {
                "schema_version": SCHEMA_VERSION,
                "split": "consumed_postmortem",
                "truth_accessed_for_prediction": False,
                "truth_used": False,
                "identity": {
                    "slug": SLUG,
                    "system_index": SYSTEM_INDEX,
                    "automatic_measure_index": index,
                    "system_measure_index": index,
                    "global_measure_index": index,
                },
                "images": {
                    "raw": {
                        "path_relative_to_out": image_path.relative_to(out_root).as_posix(),
                        "sha256": _sha256(image_path),
                        "width_px": width,
                        "height_px": height,
                    }
                },
                "staff_geometry": {
                    "raw_staff_lines_y_px": list(staff.line_ys),
                    "method": "build_vlm_melody_inputs._estimate_staff",
                },
                "allowed_context": dict(allowed_context),
                "allowed_context_provenance": dict(context_provenance),
                "segmentation_provenance": {
                    "x_bounds_px": dict(row["x_bounds_px"]),
                    "source_record_sha256": _hash_json(row),
                },
            }
        )
    return requests


def _run_inference(
    requests: Sequence[Mapping[str, Any]],
    model_payload: Mapping[str, Any],
    out_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    model, pitch_predictor, replay = inference.reconstruct_model(model_payload)
    selector_method = str(model_payload["replay"]["method"]["method_id"])
    items = [
        inference._infer_request(
            request,
            model=model,
            pitch_predictor=pitch_predictor,
            out_dir=out_root,
            selector_method_id=selector_method,
        )
        for request in requests
    ]
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir()
    overlays = []
    for index, item in enumerate(items, start=1):
        path = overlay_dir / f"measure_{index:03d}.png"
        inference.composed._write_overlay(item, path)
        overlays.append(path)
    inference._write_contact_sheet(overlays, output_dir / "contact_sheet.png")
    return {
        "predictions": [item.prediction for item in items],
        "inference": [inference._inference_record(item) for item in items],
        "replay": replay,
    }


def _triage_predictions(
    requests: Sequence[Mapping[str, Any]],
    details: Sequence[Mapping[str, Any]],
    *,
    model_payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    out_root: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    model, _pitch_predictor, _replay = inference.reconstruct_model(model_payload)
    selector = recovery.selector_config_from_model(model_payload)
    replays = onset._holdout_replays(
        details,
        request_rows=requests,
        out_dir=out_root,
        model=model,
        selector=selector,
    )
    request_by_measure = {int(row["identity"]["system_measure_index"]): row for row in requests}
    rows = [
        meter.observe_replay(
            replay,
            request=request_by_measure[replay.measure_index],
            metadata=metadata,
        )
        for replay in replays
    ]
    triage_overlay_dir = output_dir / "triage_overlays"
    triage_overlay_dir.mkdir()
    overlay_paths = []
    rows_by_key = {str(row["key"]): row for row in rows}
    for replay in replays:
        path = triage_overlay_dir / f"measure_{replay.measure_index:03d}.png"
        meter._write_overlay(rows_by_key[replay.key], replay, path)
        overlay_paths.append(path)
    inference._write_contact_sheet(overlay_paths, output_dir / "triage_contact_sheet.png")
    return rows


def _evaluate_triage(
    rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metrics_by_measure = {int(row["identity"]["system_measure_index"]): row for row in metric_rows}
    evaluated = []
    for row in rows:
        measure = int(row["identity"]["system_measure_index"])
        result = metrics_by_measure[measure]
        compared = int(result["compared_notes"])
        has_onset_error = (
            int(result["pred_note_count"]) != int(result["truth_note_count"])
            or int(result["onset_matches"]) != compared
        )
        evaluated.append({**dict(row), "has_onset_error": has_onset_error})
    return {
        "metrics": meter.validation_metrics(evaluated),
        "flagged_measure_indices": [
            int(row["identity"]["system_measure_index"]) for row in evaluated if row["review_flag"]
        ],
        "error_measure_indices": [
            int(row["identity"]["system_measure_index"])
            for row in evaluated
            if row["has_onset_error"]
        ],
    }


def _evaluate_baseline_triage(
    flagged_measures: set[int],
    metric_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = []
    for row in metric_rows:
        measure = int(row["identity"]["system_measure_index"])
        compared = int(row["compared_notes"])
        rows.append(
            {
                "identity": {"system_measure_index": measure},
                "review_flag": measure in flagged_measures,
                "has_onset_error": (
                    int(row["pred_note_count"]) != int(row["truth_note_count"])
                    or int(row["onset_matches"]) != compared
                ),
            }
        )
    return {
        "metrics": meter.validation_metrics(rows),
        "flagged_measure_indices": sorted(flagged_measures),
        "error_measure_indices": [
            int(row["identity"]["system_measure_index"]) for row in rows if row["has_onset_error"]
        ],
    }


def _coordinate_review_priorities(
    metric_rows: Sequence[Mapping[str, Any]],
    triage_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    triage_by_measure = {int(row["identity"]["system_measure_index"]): row for row in triage_rows}
    priorities = []
    for row in metric_rows:
        measure = int(row["identity"]["system_measure_index"])
        count_delta = int(row["pred_note_count"]) - int(row["truth_note_count"])
        priorities.append(
            {
                "system_measure_index": measure,
                "pred_note_count": int(row["pred_note_count"]),
                "truth_note_count": int(row["truth_note_count"]),
                "count_delta": count_delta,
                "absolute_count_error": abs(count_delta),
                "meter_review_flag": bool(triage_by_measure[measure]["review_flag"]),
                "ordered_pitch_accuracy": (
                    round(int(row["pitch_matches"]) / int(row["compared_notes"]), 6)
                    if int(row["compared_notes"])
                    else 0.0
                ),
            }
        )
    return sorted(
        priorities,
        key=lambda row: (
            -int(row["absolute_count_error"]),
            not bool(row["meter_review_flag"]),
            int(row["system_measure_index"]),
        ),
    )


def _comparison(baseline: Mapping[str, Any], corrected: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "note_f1",
        "note_precision",
        "note_recall",
        "ordered_pitch_accuracy",
        "ordered_onset_accuracy",
        "ordered_duration_accuracy",
        "rest_f1",
        "exact_measure_rate",
        "meter_valid_crop_rate",
    )
    deltas = {}
    for key in keys:
        before = baseline.get(key)
        after = corrected.get(key)
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            deltas[key] = round(float(after) - float(before), 6)
    return {"metric_deltas_corrected_minus_baseline": deltas}


def _next_gate(comparison: Mapping[str, Any]) -> dict[str, Any]:
    deltas = comparison["metric_deltas_corrected_minus_baseline"]
    recall_gain = float(deltas.get("note_recall", 0.0))
    precision_gain = float(deltas.get("note_precision", 0.0))
    return {
        "segmentation_materially_helped": recall_gain >= 0.05 and precision_gain >= -0.02,
        "coordinate_review_required_before_retraining": True,
        "reason": (
            "MusicXML supplies events but not notehead pixel coordinates; selector training "
            "remains blocked until corrected-crop candidates are visually adjudicated."
        ),
    }


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    baseline = report["baseline"]["summary"]
    corrected = report["corrected"]["summary"]
    deltas = report["comparison"]["metric_deltas_corrected_minus_baseline"]
    lines = [
        "# Coqueteos Corrected Segmentation Replay",
        "",
        "Consumed postmortem only; the heldout result remains unchanged.",
        "",
        "| Metric | Sealed 6 crops | Corrected 7 crops | Delta |",
        "|---|---:|---:|---:|",
    ]
    for key in (
        "note_f1",
        "note_precision",
        "note_recall",
        "ordered_pitch_accuracy",
        "ordered_onset_accuracy",
        "ordered_duration_accuracy",
        "rest_f1",
        "exact_measure_rate",
        "meter_valid_crop_rate",
    ):
        lines.append(
            f"| `{key}` | {baseline.get(key)} | {corrected.get(key)} | {deltas.get(key)} |"
        )
    lines.extend(
        [
            "",
            "The exact frozen model and truth-blind fifth-score context were reused.",
            "No Coqueteos coordinate labels were inferred from MusicXML.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError(f"Expected JSON objects: {path}")
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": _display_path(resolved),
        "sha256": _sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _local_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_json(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
