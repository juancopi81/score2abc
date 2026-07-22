"""Replay consumed pitch predictions with detector-derived key state.

This is a bounded, consumed-evidence postmortem. It proves that the automatic
visual key-state artifact can supply exact fifths to the existing x-only pitch
replay without changing candidate selection. Predictions are persisted before
the consumed truth is opened, and the result is not an independent heldout
claim or a pipeline-ready integration.

Example::

    uv run python scripts/experiments/spike_consumed_visual_key_pitch_replay.py \
        --detector-report /tmp/score2abc-key-state-detector/final2/la-chata/report.json \
        --context-hints out/jaime-llanos_64_la-chata_pasillo_luis-a-calvo/\
vlm_melody_third_score_heldout/v2/system_007/context_hints_v1.json \
        --inference out/jaime-llanos_64_la-chata_pasillo_luis-a-calvo/\
vlm_melody_third_score_heldout/v2/system_007/inference_v2/inference.jsonl \
        --model out/jaime-llanos_64_la-chata_pasillo_luis-a-calvo/\
vlm_melody_third_score_heldout/v2/system_007/frozen/artifacts/model/001_model.json \
        --truth out/jaime-llanos_64_la-chata_pasillo_luis-a-calvo/\
vlm_melody_third_score_heldout/v2/system_007/evaluation_v1/truth.jsonl \
        --output-dir /tmp/score2abc-visual-key-pitch-replay
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import spike_consumed_polyphonic_pitch_repair as replay  # noqa: E402

SCHEMA_VERSION = 1
KIND = "consumed_visual_key_pitch_replay_postmortem"
LANE_BASELINE = "x_only_no_context"
LANE_VISUAL = "x_only_visual_key"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pin(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_object(path: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    value, pin = replay._read_json_object(path, label=label)
    return value, pin


def _read_detector_report(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    report, pin = _read_object(path, "detector report")
    if report.get("truth_used_for_prediction") is not False:
        raise ValueError("Detector report must declare truth_used_for_prediction=false")
    measures = report.get("measures")
    predictions = report.get("predictions")
    if isinstance(measures, list) and measures:
        if any(not isinstance(item, Mapping) for item in measures):
            raise ValueError("Detector report measures must be objects")
        if any(item.get("truth_used_for_prediction") is True for item in measures):
            raise ValueError("Detector measure must not be marked truth_used_for_prediction")
    elif isinstance(predictions, list) and predictions:
        if any(not isinstance(item, Mapping) for item in predictions):
            raise ValueError("Detector report predictions must be objects")
        if any(item.get("truth_used_for_prediction") is True for item in predictions):
            raise ValueError("Detector prediction must not be marked truth_used_for_prediction")
        hints = report.get("context_hints")
        if not isinstance(hints, Mapping) or hints.get("truth_used") is not False:
            raise ValueError("Expanded detector report needs truth-free context_hints")
    else:
        raise ValueError("Detector report has no measures or predictions")
    return report, pin


def _legacy_detector_context(
    report: Mapping[str, Any],
) -> tuple[dict[int, dict[str, int]], list[dict[str, Any]], dict[str, Any]]:
    """Validate state propagation and turn confirmed fifths into replay hints."""
    measures = sorted(report["measures"], key=lambda item: int(item.get("sequence_index", 10**9)))
    context: dict[int, dict[str, int]] = {}
    state_rows: list[dict[str, Any]] = []
    current_fifths: int | None = None
    explicit_count = 0
    for measure in measures:
        if not isinstance(measure, Mapping):
            raise ValueError("Detector report measures must be objects")
        measure_index = int(measure.get("sequence_index", -1)) + 1
        identity = int(measure.get("automatic_measure_index", measure_index))
        state = measure.get("state")
        if not isinstance(state, Mapping):
            raise ValueError(f"Detector measure {measure_index} has no propagated state")
        kind = state.get("kind")
        fifths = state.get("fifths")
        measure_fifths = measure.get("fifths")
        if kind == "explicit_change":
            if measure.get("fifths_status") != "confirmed_explicit":
                raise ValueError(f"Detector measure {measure_index} lacks confirmed fifths status")
            if not isinstance(fifths, int) or isinstance(fifths, bool):
                raise ValueError(f"Detector measure {measure_index} has no exact explicit fifths")
            if measure_fifths != fifths:
                raise ValueError(f"Detector measure {measure_index} has inconsistent fifths")
            current_fifths = int(fifths)
            explicit_count += 1
        elif kind == "inherited":
            if current_fifths is None or fifths != current_fifths:
                raise ValueError(f"Detector measure {measure_index} has invalid inherited fifths")
        elif kind == "unknown_initial":
            if fifths is not None or current_fifths is not None:
                raise ValueError(f"Detector measure {measure_index} has invalid unknown state")
        else:
            raise ValueError(f"Detector measure {measure_index} has unsupported state kind: {kind}")

        if current_fifths is not None:
            context[identity] = {"key_fifths": current_fifths}
        state_rows.append(
            {
                "automatic_measure_index": identity,
                "kind": kind,
                "fifths": current_fifths,
                "pitch_mapping_ready": current_fifths is not None,
            }
        )
    if explicit_count == 0:
        raise ValueError("Detector report has no confirmed explicit key change")
    return (
        context,
        state_rows,
        {
            "explicit_change_count": explicit_count,
            "final_fifths": current_fifths,
            "mapped_measure_count": len(context),
        },
    )


def _expanded_detector_context(
    report: Mapping[str, Any],
    *,
    measure_indices: Sequence[int],
    slug: str,
) -> tuple[dict[int, dict[str, int]], list[dict[str, Any]], dict[str, Any]]:
    hints = report.get("context_hints")
    events = hints.get("events") if isinstance(hints, Mapping) else None
    if not isinstance(events, list) or not events:
        raise ValueError("Expanded detector report has no context events")

    scoped_events: list[tuple[int, int]] = []
    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError("Expanded detector context events must be objects")
        source = event.get("source")
        event_slug = source.get("slug") if isinstance(source, Mapping) else None
        if event_slug != slug:
            continue
        start = event.get("start_measure")
        hint = event.get("key_hint")
        fifths = hint.get("fifths") if isinstance(hint, Mapping) else None
        if not isinstance(start, int) or isinstance(start, bool) or start <= 0:
            raise ValueError(f"Detector event for {slug} has invalid start_measure")
        if not isinstance(fifths, int) or isinstance(fifths, bool):
            raise ValueError(f"Detector event for {slug} has invalid fifths")
        scoped_events.append((start, fifths))
    if not scoped_events:
        raise ValueError(f"Expanded detector report has no context events for slug: {slug}")
    scoped_events.sort()
    starts = [start for start, _ in scoped_events]
    if len(starts) != len(set(starts)):
        raise ValueError(f"Expanded detector report has duplicate event starts for slug: {slug}")

    context: dict[int, dict[str, int]] = {}
    state_rows = []
    for measure in sorted(set(int(value) for value in measure_indices)):
        active = [(start, fifths) for start, fifths in scoped_events if start <= measure]
        if active:
            start, fifths = active[-1]
            context[measure] = {"key_fifths": fifths}
            kind = "explicit_change" if start == measure else "inherited"
        else:
            fifths = None
            kind = "unknown_initial"
        state_rows.append(
            {
                "automatic_measure_index": measure,
                "kind": kind,
                "fifths": fifths,
                "pitch_mapping_ready": fifths is not None,
            }
        )
    return (
        context,
        state_rows,
        {
            "source_format": "expanded_signature_events",
            "target_slug": slug,
            "explicit_change_count": len(scoped_events),
            "final_fifths": scoped_events[-1][1],
            "mapped_measure_count": len(context),
        },
    )


def _detector_context(
    report: Mapping[str, Any],
    *,
    measure_indices: Sequence[int],
    slug: str,
) -> tuple[dict[int, dict[str, int]], list[dict[str, Any]], dict[str, Any]]:
    if isinstance(report.get("measures"), list):
        return _legacy_detector_context(report)
    return _expanded_detector_context(report, measure_indices=measure_indices, slug=slug)


def _verify_detector_images(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    pins = []
    entries = report.get("measures", report.get("predictions"))
    if not isinstance(entries, list):
        raise ValueError("Detector report has no image-bearing entries")
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, Mapping):
            raise ValueError("Detector report image entries must be objects")
        input_pin = entry.get("input")
        if not isinstance(input_pin, Mapping):
            raise ValueError(f"Detector entry {index} has no input image pin")
        path_value = input_pin.get("path")
        expected = input_pin.get("sha256")
        if not isinstance(path_value, str) or not isinstance(expected, str):
            raise ValueError(f"Detector entry {index} has incomplete input image pin")
        path = Path(path_value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Detector source image is missing: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"Detector source image hash mismatch: {path}")
        pins.append({"measure_sequence_index": index - 1, "path": str(path), "sha256": actual})
    return pins


def _inference_slug(rows: Sequence[Mapping[str, Any]]) -> str:
    slugs = {
        str(identity["slug"])
        for row in rows
        if isinstance((identity := row.get("identity")), Mapping) and identity.get("slug")
    }
    if len(slugs) != 1:
        raise ValueError("Inference rows must identify exactly one work slug")
    return next(iter(slugs))


def _prediction_row(
    lane: str, prediction: Mapping[str, Any], key_fifths: int | None
) -> dict[str, Any]:
    return {
        "automatic_measure_index": int(prediction["automatic_measure_index"]),
        "candidate_ids": list(prediction["selected_candidate_ids"]),
        "key_fifths": key_fifths,
        "lane": lane,
        "notes": [
            {
                "candidate_id": note["candidate_id"],
                "pitch": note["pitch"],
                "pitch_midi": int(note["pitch_midi"]),
                "x": float(note["x"]),
                "y": float(note["y"]),
            }
            for note in prediction["notes"]
        ],
        "onset_group_count": int(prediction["onset_group_count"]),
        "total_head_count": int(prediction["total_head_count"]),
        "truth_used_for_prediction": False,
    }


def _persist_predictions(
    output_dir: Path,
    lane: str,
    predictions: Mapping[int, Mapping[str, Any]],
    visual_context: Mapping[int, Mapping[str, int]],
) -> dict[str, Any]:
    path = output_dir / f"predictions_{lane}.jsonl"
    lines = []
    rows = []
    for measure in sorted(predictions):
        key_fifths = (
            visual_context.get(measure, {}).get("key_fifths") if lane == LANE_VISUAL else None
        )
        row = _prediction_row(lane, predictions[measure], key_fifths)
        rows.append(row)
        lines.append(json.dumps(row, sort_keys=True, separators=(",", ":")))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "rows": rows,
    }


def _selection_comparison(
    baseline: Mapping[int, Mapping[str, Any]], visual: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    per_measure = []
    for measure in sorted(baseline):
        left = baseline[measure]
        right = visual.get(measure)
        if right is None:
            raise ValueError(f"Visual prediction missing measure {measure}")
        ids_equal = list(left["selected_candidate_ids"]) == list(right["selected_candidate_ids"])
        counts_equal = int(left["total_head_count"]) == int(right["total_head_count"])
        groups_equal = int(left["onset_group_count"]) == int(right["onset_group_count"])
        per_measure.append(
            {
                "automatic_measure_index": measure,
                "candidate_ids_equal": ids_equal,
                "baseline_candidate_ids": list(left["selected_candidate_ids"]),
                "visual_candidate_ids": list(right["selected_candidate_ids"]),
                "counts_equal": counts_equal,
                "baseline_total_head_count": int(left["total_head_count"]),
                "visual_total_head_count": int(right["total_head_count"]),
                "onset_groups_equal": groups_equal,
                "baseline_onset_group_count": int(left["onset_group_count"]),
                "visual_onset_group_count": int(right["onset_group_count"]),
            }
        )
    return {
        "all_candidate_ids_equal": all(item["candidate_ids_equal"] for item in per_measure),
        "all_counts_equal": all(item["counts_equal"] for item in per_measure),
        "all_onset_groups_equal": all(item["onset_groups_equal"] for item in per_measure),
        "identical_selection": all(
            item["candidate_ids_equal"] and item["counts_equal"] and item["onset_groups_equal"]
            for item in per_measure
        ),
        "per_measure": per_measure,
    }


def _x_only_score(
    predictions: Mapping[int, Mapping[str, Any]],
    truth_by_measure: Mapping[int, Mapping[str, Any]],
    selector: Mapping[str, Any],
) -> dict[str, Any]:
    result = replay._score_lane({"x_only": predictions}, truth_by_measure, selector=selector)
    if len(result) != 1:
        raise ValueError("Expected exactly one x-only score")
    return result[0]


def _metrics_summary(score: Mapping[str, Any]) -> dict[str, Any]:
    metrics = score["metrics"]
    return {
        "exact_pitch_matches": metrics["legacy_ordered_pitch_metrics"]["exact_pitch_matches"],
        "ordered_pitch_edit_distance": metrics["legacy_ordered_pitch_metrics"][
            "ordered_pitch_edit_distance"
        ],
        "exact_group_count": metrics["pitch_group_metrics"]["exact_group_count"],
        "group_edit_distance": metrics["pitch_group_metrics"]["group_edit_distance"],
        "predicted_total_head_count": metrics["selection_count_metrics"][
            "predicted_total_head_count"
        ],
        "truth_total_head_count": metrics["selection_count_metrics"]["truth_total_head_count"],
    }


def _label_context(
    path: Path,
    measure_indices: Sequence[int],
) -> tuple[dict[int, Any], dict[str, Any]]:
    labels, pin = replay._load_context_hints(path, measure_indices=measure_indices)
    return labels, pin


def _hint_fifths(hint: Any) -> int | None:
    if isinstance(hint, Mapping):
        for key in ("key_fifths", "fifths"):
            if key in hint:
                return int(hint[key])
    accidentals = replay._key_accidentals(hint)
    values = list(accidentals.values())
    if not values or any(value != values[0] for value in values):
        return None
    return len(values) if values[0] > 0 else -len(values)


def _write_markdown(report: Mapping[str, Any]) -> str:
    comparison = report["comparison"]
    baseline = comparison["baseline_metrics"]
    visual = comparison["visual_metrics"]
    gate = report["consumed_evidence_gate"]
    lines = [
        "# Consumed Visual-Key Pitch Replay",
        "",
        "This is a consumed postmortem, not an independent heldout result or a "
        "pipeline-ready change.",
        "",
        "## X-only result",
        "",
        "| Lane | Exact ordered pitches | Exact pitch groups | Ordered edit | Group edit |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| No context | {baseline['exact_pitch_matches']} | {baseline['exact_group_count']} | "
        f"{baseline['ordered_pitch_edit_distance']} | {baseline['group_edit_distance']} |",
        f"| Visual key | {visual['exact_pitch_matches']} | {visual['exact_group_count']} | "
        f"{visual['ordered_pitch_edit_distance']} | {visual['group_edit_distance']} |",
        "",
        "## Gate",
        "",
        f"- Identical candidate selection: `{gate['identical_selection']}`.",
        f"- Ordered exact pitch improvement: `{gate['ordered_exact_pitch_improved']}`.",
        f"- Exact pitch-group improvement: `{gate['exact_pitch_group_improved']}`.",
        f"- Consumed-evidence gate passed: `{gate['passed']}`.",
        "",
        "The visual lane uses only exact fifths propagated from a confirmed detector change; "
        "context hints are evaluation labels and do not drive prediction materialization.",
        "",
    ]
    return "\n".join(lines)


def run_replay(
    detector_report_path: Path,
    context_hints_path: Path,
    inference_path: Path,
    model_path: Path,
    truth_path: Path,
    output_dir: Path,
    *,
    slug: str | None = None,
) -> dict[str, Any]:
    """Run the bounded replay and create deterministic artifacts once."""
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite replay output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    detector_report_path = detector_report_path.expanduser().resolve()
    context_hints_path = context_hints_path.expanduser().resolve()
    inference_path = inference_path.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    truth_path = truth_path.expanduser().resolve()

    detector_report, detector_pin = _read_detector_report(detector_report_path)
    source_images = _verify_detector_images(detector_report)
    inference_rows, inference_pin = replay._read_jsonl(inference_path, label="inference")
    model, model_pin = _read_object(model_path, "model")
    if any(row.get("truth_used") is True for row in inference_rows):
        raise ValueError("Inference rows must not be marked truth_used")
    selector = replay.selector_config_from_model(model)
    measures = [replay._identity_measure(row) for row in inference_rows]
    target_slug = slug or _inference_slug(inference_rows)
    visual_context, state_rows, state_summary = _detector_context(
        detector_report,
        measure_indices=measures,
        slug=target_slug,
    )
    if isinstance(detector_report.get("measures"), list):
        detector_measure_indices = {
            int(item.get("automatic_measure_index", int(item.get("sequence_index", -1)) + 1))
            for item in detector_report["measures"]
        }
        if set(measures) != detector_measure_indices:
            raise ValueError("Detector and inference measure identities do not match")

    baseline_predictions = replay._materialize_predictions(
        inference_rows, selector, [None], lane=LANE_BASELINE
    )["x_only"]
    visual_predictions = replay._materialize_predictions(
        inference_rows,
        selector,
        [None],
        lane=LANE_VISUAL,
        context_hints=visual_context,
    )["x_only"]
    baseline_parity = None
    if all(isinstance(row.get("canonical_prediction"), Mapping) for row in inference_rows):
        baseline_parity = replay._verify_x_only_replay_parity(
            inference_rows, {"x_only": baseline_predictions}
        )

    output_dir.mkdir()
    baseline_artifact = _persist_predictions(
        output_dir, LANE_BASELINE, baseline_predictions, visual_context
    )
    visual_artifact = _persist_predictions(
        output_dir, LANE_VISUAL, visual_predictions, visual_context
    )
    prediction_manifest = {
        "schema_version": SCHEMA_VERSION,
        "truth_used_for_prediction": False,
        "predictions_materialized_before_truth_read": True,
        "lanes": {
            LANE_BASELINE: {
                key: value for key, value in baseline_artifact.items() if key != "rows"
            },
            LANE_VISUAL: {key: value for key, value in visual_artifact.items() if key != "rows"},
        },
    }
    _write_json(output_dir / "prediction_manifest.json", prediction_manifest)
    prediction_manifest_pin = _pin(output_dir / "prediction_manifest.json")

    # This is intentionally the first consumed-truth access in this function.
    truth_rows, truth_pin = replay._read_jsonl(truth_path, label="truth")
    truth_by_measure = {
        int(row["automatic_crop_index"]): row
        for row in truth_rows
        if row.get("automatic_crop_index") is not None
    }
    if len(truth_by_measure) != len(truth_rows):
        raise ValueError("Truth rows need unique automatic_crop_index values")
    baseline_score = _x_only_score(baseline_predictions, truth_by_measure, selector)
    visual_score = _x_only_score(visual_predictions, truth_by_measure, selector)

    label_context, context_pin = _label_context(context_hints_path, measures)
    label_agreement = {
        "used_for_prediction": False,
        "detector_and_context_hint_fifths_match": all(
            _hint_fifths(label_context[measure])
            == visual_context.get(measure, {}).get("key_fifths")
            for measure in label_context
            if measure in visual_context
        ),
        "context_hint_measure_count": len(label_context),
    }
    selection = _selection_comparison(baseline_predictions, visual_predictions)
    baseline_metrics = _metrics_summary(baseline_score)
    visual_metrics = _metrics_summary(visual_score)
    gate = {
        "identical_selection": selection["identical_selection"],
        "ordered_exact_pitch_improved": visual_metrics["exact_pitch_matches"]
        > baseline_metrics["exact_pitch_matches"],
        "exact_pitch_group_improved": visual_metrics["exact_group_count"]
        > baseline_metrics["exact_group_count"],
    }
    gate["passed"] = all(gate.values())

    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "evaluated_consumed_postmortem_create_once",
        "claim_boundary": {
            "independent_heldout_claim": False,
            "pipeline_ready": False,
            "description": f"Consumed {target_slug} postmortem over already-used evidence.",
        },
        "detector": {
            "report": detector_pin,
            "truth_used_for_prediction": detector_report["truth_used_for_prediction"],
            "source_images": source_images,
            "state_sequence": state_rows,
            "state_summary": state_summary,
            "target_slug": target_slug,
            "visual_context_used_for_prediction": visual_context,
        },
        "inputs": {
            "inference_jsonl": inference_pin,
            "frozen_model_json": model_pin,
            "truth_jsonl": truth_pin,
            "context_hints_json": context_pin,
        },
        "selector": selector,
        "baseline_parity": baseline_parity,
        "prediction_artifacts": {
            "manifest": prediction_manifest_pin,
            "baseline": {key: value for key, value in baseline_artifact.items() if key != "rows"},
            "visual_key": {key: value for key, value in visual_artifact.items() if key != "rows"},
        },
        "comparison": {
            "baseline_metrics": baseline_metrics,
            "visual_metrics": visual_metrics,
            "delta": {
                "exact_pitch_matches": visual_metrics["exact_pitch_matches"]
                - baseline_metrics["exact_pitch_matches"],
                "exact_group_count": visual_metrics["exact_group_count"]
                - baseline_metrics["exact_group_count"],
                "ordered_pitch_edit_distance": visual_metrics["ordered_pitch_edit_distance"]
                - baseline_metrics["ordered_pitch_edit_distance"],
                "group_edit_distance": visual_metrics["group_edit_distance"]
                - baseline_metrics["group_edit_distance"],
            },
            "baseline_score": baseline_score,
            "visual_score": visual_score,
        },
        "candidate_selection": selection,
        "context_hint_evaluation_labels": label_agreement,
        "consumed_evidence_gate": gate,
        "access_audit": {
            "detector_truth_used_verified_false": True,
            "detector_context_used_for_prediction": True,
            "source_image_hashes_verified": True,
            "predictions_persisted_before_truth_read": True,
            "truth_used_for_prediction": False,
            "truth_used_for_scoring": True,
            "evaluation_label_context_used_for_prediction": False,
        },
    }
    _write_json(output_dir / "report.json", report)
    (output_dir / "report.md").write_text(_write_markdown(report), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-report", type=Path, required=True)
    parser.add_argument("--context-hints", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slug")
    args = parser.parse_args(argv)
    try:
        report = run_replay(
            args.detector_report,
            args.context_hints,
            args.inference,
            args.model,
            args.truth,
            args.output_dir,
            slug=args.slug,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(report["prediction_artifacts"]["baseline"]["path"])
    print(report["prediction_artifacts"]["visual_key"]["path"])
    print(report["inputs"]["truth_jsonl"]["path"])
    print(str(Path(report["prediction_artifacts"]["baseline"]["path"]).parent / "report.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
