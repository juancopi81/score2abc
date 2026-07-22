"""Flag automatic onset groups whose visual durations underfill the meter.

The rejected onset-group filter attempted to delete low-confidence groups and
could not improve work-disjoint F1 without losing real notes. This replacement
does not mutate transcription. It uses automatic onset anchors, the existing
visual rhythm parser, and available meter context to prioritize measures for
human review. Evaluation truth is opened only after every review flag has been
materialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import run_third_score_heldout_inference as heldout  # noqa: E402
from scripts.experiments import spike_anchored_rhythm_parser as rhythm  # noqa: E402
from scripts.experiments import spike_consumed_onset_group_selector as onset  # noqa: E402
from scripts.experiments import spike_consumed_polyphonic_pitch_repair as recovery  # noqa: E402
from scripts.experiments import spike_cross_score_consumed_retraining as cross_score  # noqa: E402

SCHEMA_VERSION = 1
OUTPUT_VERSION = "consumed_meter_deficit_validator_v1"
DEFAULT_OUT_DIR = REPO_ROOT / "out"
DEFAULT_OUTPUT_DIR = DEFAULT_OUT_DIR / "vlm_melody_consumed_training" / OUTPUT_VERSION
RHYTHM_METER_PRIORS = {"pasillo": 3.0}
EPSILON = 1e-9


@dataclass(frozen=True)
class MeterContext:
    expected_beats: float
    allow_pickup: bool
    source: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _file_pin(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": _display_path(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _metadata_path(out_dir: Path, slug: str) -> Path:
    return (out_dir / slug / "metadata.json").resolve()


def meter_context(
    request: Mapping[str, Any], *, metadata: Mapping[str, Any]
) -> MeterContext | None:
    allowed = request.get("allowed_context", {})
    if not isinstance(allowed, Mapping):
        allowed = {}
    expected = allowed.get("expected_measure_beats")
    allow_pickup = bool(allowed.get("allow_pickup", False))
    if expected is not None:
        value = float(expected)
        if math.isfinite(value) and value > 0:
            return MeterContext(value, allow_pickup, "request.allowed_context")
    score_rhythm = str(metadata.get("rhythm", "")).strip().lower()
    prior = RHYTHM_METER_PRIORS.get(score_rhythm)
    if prior is None:
        return None
    return MeterContext(prior, allow_pickup, f"metadata.rhythm_prior:{score_rhythm}")


def decide_meter_deficit(
    visual_beats: float, context: MeterContext | None
) -> tuple[bool, str, float | None]:
    if context is None:
        return False, "not_evaluable_missing_meter_context", None
    deficit = context.expected_beats - visual_beats
    if context.allow_pickup:
        return False, "pickup_exempt", deficit
    if deficit > EPSILON:
        return True, "review_visual_meter_deficit", deficit
    return False, "meter_not_underfilled", deficit


def _automatic_anchors(replay: onset.ExampleReplay) -> list[dict[str, Any]]:
    anchors = []
    for group in replay.baseline_groups:
        for candidate in group["candidates"]:
            x, y = onset._candidate_center(candidate)
            anchors.append(
                {
                    "order": len(anchors) + 1,
                    "pitch": "C4",
                    "center": {"x": x, "y": y},
                    "candidate_id": onset._candidate_id(candidate),
                }
            )
    return anchors


def observe_replay(
    replay: onset.ExampleReplay,
    *,
    request: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    anchors = _automatic_anchors(replay)
    staff_lines = [
        int(value) for value in replay.inference_row["staff_geometry"]["raw_staff_lines_y_px"]
    ]
    spacing = rhythm.staff_spacing(staff_lines)
    with Image.open(replay.source_image) as opened:
        image = opened.convert("RGB")
    groups = rhythm.group_simultaneous_heads(anchors, spacing)
    anchor_features = rhythm.extract_anchor_features(image, anchors, staff_lines)
    rest_features = rhythm.extract_residual_rest_features(image, groups, staff_lines)
    symbols = rhythm.build_visual_symbols(groups, anchor_features, rest_features)
    visual_beats = sum(float(symbol["duration_beats"]) for symbol in symbols)
    context = meter_context(request, metadata=metadata)
    flagged, status, deficit = decide_meter_deficit(visual_beats, context)
    return {
        "key": replay.key,
        "identity": {
            "slug": replay.work_id,
            "system_index": replay.system_index,
            "system_measure_index": replay.measure_index,
        },
        "source_image": _display_path(replay.source_image),
        "source_sha256": _sha256(replay.source_image),
        "baseline_group_count": len(replay.observations),
        "automatic_anchor_count": len(anchors),
        "visual_note_symbol_count": sum(symbol["kind"] == "note" for symbol in symbols),
        "visual_rest_symbol_count": sum(symbol["kind"] == "rest" for symbol in symbols),
        "visual_total_beats": visual_beats,
        "expected_measure_beats": context.expected_beats if context else None,
        "meter_context_source": context.source if context else None,
        "allow_pickup": context.allow_pickup if context else None,
        "deficit_beats": deficit,
        "review_flag": flagged,
        "status": status,
        "symbols": symbols,
        "truth_used": False,
    }


def validation_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    evaluable = [row for row in rows if row.get("has_onset_error") is not None]
    tp = sum(bool(row["review_flag"]) and bool(row["has_onset_error"]) for row in evaluable)
    fp = sum(bool(row["review_flag"]) and not bool(row["has_onset_error"]) for row in evaluable)
    fn = sum(not bool(row["review_flag"]) and bool(row["has_onset_error"]) for row in evaluable)
    tn = sum(not bool(row["review_flag"]) and not bool(row["has_onset_error"]) for row in evaluable)
    return {
        "measure_count": len(evaluable),
        "error_measure_count": tp + fn,
        "flagged_measure_count": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "review_load": (tp + fp) / len(evaluable) if evaluable else 0.0,
    }


def _evaluate_consumed(
    observations: Sequence[dict[str, Any]],
    replays: Sequence[onset.ExampleReplay],
) -> list[dict[str, Any]]:
    replay_by_key = {replay.key: replay for replay in replays}
    evaluated = []
    for prediction in observations:
        replay = replay_by_key[str(prediction["key"])]
        metrics = onset._predicted_metrics(replay, replay.observations)
        evaluated.append(
            {
                **prediction,
                "has_onset_error": bool(metrics["fp"] or metrics["fn"]),
                "baseline_onset_metrics": metrics,
            }
        )
    return evaluated


def _truth_counts(rows: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    result = {}
    for row in rows:
        identity = row.get("identity", {})
        measure = int(identity.get("automatic_measure_index", 0))
        if measure <= 0:
            raise ValueError("Truth row has no automatic_measure_index")
        notes = row.get("notes", [])
        result[measure] = len({int(note["onset_divisions"]) for note in notes})
    return result


def _evaluate_count_only(
    observations: Sequence[dict[str, Any]], truth_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    counts = _truth_counts(truth_rows)
    evaluated = []
    for prediction in observations:
        measure = int(prediction["identity"]["system_measure_index"])
        truth_count = counts[measure]
        evaluated.append(
            {
                **prediction,
                "truth_group_count": truth_count,
                "has_onset_error": int(prediction["baseline_group_count"]) != truth_count,
                "evaluation_scope": "count_only_truth_has_no_pixel_x",
            }
        )
    return evaluated


def _write_overlay(observation: Mapping[str, Any], replay: onset.ExampleReplay, path: Path) -> None:
    with Image.open(replay.source_image) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    color = (220, 80, 30) if observation["review_flag"] else (20, 150, 60)
    for group in replay.baseline_groups:
        x = round(float(group["x_center"]))
        draw.line((x, 0, x, image.height - 1), fill=color, width=2)
    label = (
        f"{observation['status']} visual={observation['visual_total_beats']:g} "
        f"meter={observation['expected_measure_beats']}"
    )
    draw.rectangle((0, 0, min(image.width - 1, 420), 15), fill=(255, 255, 255))
    draw.text((2, 2), label, fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    tn = sum(int(row["tn"]) for row in rows)
    total = tp + fp + fn + tn
    return {
        "measure_count": total,
        "error_measure_count": tp + fn,
        "flagged_measure_count": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "review_load": (tp + fp) / total if total else 0.0,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    aggregate = report["consumed_evaluation"]["aggregate"]
    lines = [
        "# Consumed Meter-Deficit Validator",
        "",
        "This validator never deletes onset groups. It flags visually underfilled "
        "measures for review.",
        "",
        f"- Gate: **{report['adoption_gate']['status']}**",
        f"- Precision: `{aggregate['precision']:.3f}`",
        f"- Recall: `{aggregate['recall']:.3f}`",
        f"- F1: `{aggregate['f1']:.3f}`",
        f"- Review load: `{aggregate['review_load']:.3f}`",
        "",
        "| Work | Errors caught | False alerts | Errors missed | Review load |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for work in report["consumed_evaluation"]["works"]:
        metrics = work["metrics"]
        lines.append(
            f"| `{work['work_id']}` | {metrics['tp']} | {metrics['fp']} | "
            f"{metrics['fn']} | {metrics['review_load']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The `Pasillo -> 3 quarter-note beats` fallback is an explicit score-metadata prior, "
            "not a value read from MusicXML truth.",
            "",
            f"La Chata count-only status: `{report['la_chata']['status']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def run_experiment(
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    model_path: Path = onset.DEFAULT_MODEL,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    la_chata_inference: Path | None = onset.DEFAULT_LA_CHATA_INFERENCE,
    la_chata_requests: Path | None = onset.DEFAULT_LA_CHATA_REQUESTS,
    la_chata_truth: Path | None = onset.DEFAULT_LA_CHATA_TRUTH,
    overlays: bool = False,
) -> dict[str, Any]:
    out_dir = out_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_payload = onset._read_json(model_path, label="frozen model")
    model, _pitch_predictor, model_audit = heldout.reconstruct_model(model_payload)
    selector = recovery.selector_config_from_model(model_payload)

    aviador, aviador_sources = cross_score._load_aviador_examples(
        out_dir, reviews_dir=cross_score.DEFAULT_REVIEWS_DIR
    )
    carrizal_by_policy, carrizal_sources = cross_score._load_carrizal_examples(
        out_dir, manifest_path=cross_score.DEFAULT_CARRIZAL_REVIEWS
    )
    examples = [*aviador, *carrizal_by_policy["C"]]
    replays = [onset._example_replay(example, model, selector) for example in examples]
    example_by_key = {example.key: example for example in examples}
    metadata_by_slug = {
        replay.work_id: onset._read_json(
            _metadata_path(out_dir, replay.work_id), label="score metadata"
        )
        for replay in replays
    }

    # Complete the truth-free observation boundary for both consumed works.
    observations = [
        observe_replay(
            replay,
            request=example_by_key[replay.key].request,
            metadata=metadata_by_slug[replay.work_id],
        )
        for replay in replays
    ]
    _write_jsonl(output_dir / "consumed_predictions.jsonl", observations)
    evaluated = _evaluate_consumed(observations, replays)
    work_rows = []
    for work_id in sorted({replay.work_id for replay in replays}):
        rows = [row for row in evaluated if row["identity"]["slug"] == work_id]
        work_rows.append({"work_id": work_id, "metrics": validation_metrics(rows), "rows": rows})
    aggregate = _aggregate([work["metrics"] for work in work_rows])

    la_chata: dict[str, Any] = {"status": "not_run"}
    holdout_replays: list[onset.ExampleReplay] = []
    holdout_observations: list[dict[str, Any]] = []
    if la_chata_inference is not None and la_chata_requests is not None:
        inference_rows = onset._read_jsonl(la_chata_inference, label="La Chata inference")
        request_rows = onset._read_jsonl(la_chata_requests, label="La Chata requests")
        requests_by_measure = {
            int(row["identity"]["system_measure_index"]): row for row in request_rows
        }
        holdout_replays = onset._holdout_replays(
            inference_rows,
            request_rows=request_rows,
            out_dir=out_dir,
            model=model,
            selector=selector,
        )
        holdout_metadata = onset._read_json(
            _metadata_path(out_dir, holdout_replays[0].work_id), label="La Chata metadata"
        )
        holdout_observations = [
            observe_replay(
                replay,
                request=requests_by_measure[replay.measure_index],
                metadata=holdout_metadata,
            )
            for replay in holdout_replays
        ]
        prediction_path = output_dir / "la_chata_predictions.jsonl"
        _write_jsonl(prediction_path, holdout_observations)
        la_chata = {
            "status": "predicted_before_truth",
            "predictions_materialized_before_truth_read": True,
            "prediction_path": _display_path(prediction_path),
            "rows": holdout_observations,
        }
        if la_chata_truth is not None:
            truth_rows = onset._read_jsonl(la_chata_truth, label="La Chata truth")
            count_evaluation = _evaluate_count_only(holdout_observations, truth_rows)
            la_chata = {
                **la_chata,
                "status": "evaluated_count_only_truth_has_no_x",
                "metrics": validation_metrics(count_evaluation),
                "rows": count_evaluation,
            }

    if overlays:
        observation_by_key = {
            str(row["key"]): row for row in [*observations, *holdout_observations]
        }
        for replay in [*replays, *holdout_replays]:
            name = f"{replay.work_id}_s{replay.system_index:02d}_m{replay.measure_index:02d}.png"
            _write_overlay(observation_by_key[replay.key], replay, output_dir / "overlays" / name)

    consumed_gate_passed = (
        aggregate["precision"] >= 0.95 and aggregate["recall"] >= 0.80 and aggregate["fp"] == 0
    )
    la_chata_metrics = la_chata.get("metrics")
    generalization_gate_passed = bool(
        isinstance(la_chata_metrics, Mapping)
        and float(la_chata_metrics["precision"]) >= 0.95
        and float(la_chata_metrics["recall"]) >= 0.80
        and int(la_chata_metrics["fp"]) == 0
    )
    runtime_gate_passed = consumed_gate_passed and generalization_gate_passed
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "consumed_meter_deficit_validator_spike",
        "output_version": OUTPUT_VERSION,
        "protocol": {
            "mutation_policy": "validation_only; predicted onset groups are never removed",
            "visual_duration_parser": "existing anchored rhythm parser over automatic anchors",
            "decision": "flag non-pickup measures when visual total is below expected beats",
            "meter_sources": ["request.allowed_context", "metadata rhythm prior"],
            "rhythm_meter_priors": RHYTHM_METER_PRIORS,
            "predictions_materialized_before_truth": True,
        },
        "inputs": {
            "model": _file_pin(model_path),
            "consumed_sources": sorted([*aviador_sources, *carrizal_sources], key=str),
            "metadata": {
                slug: _file_pin(_metadata_path(out_dir, slug)) for slug in metadata_by_slug
            },
        },
        "model_audit": model_audit,
        "consumed_evaluation": {"works": work_rows, "aggregate": aggregate},
        "la_chata": la_chata,
        "adoption_gate": {
            "status": (
                "adopt_for_review_triage"
                if runtime_gate_passed
                else (
                    "retain_consumed_signal_generalization_blocked"
                    if consumed_gate_passed
                    else "reject"
                )
            ),
            "runtime_transcription_mutation_eligible": False,
            "requires_precision": 0.95,
            "requires_recall": 0.80,
            "requires_zero_false_alerts": True,
            "consumed_gate_passed": consumed_gate_passed,
            "la_chata_generalization_gate_passed": generalization_gate_passed,
            "passed": runtime_gate_passed,
        },
        "artifacts": {
            "report_json": _display_path(output_dir / "report.json"),
            "report_markdown": _display_path(output_dir / "report.md"),
            "consumed_predictions": _display_path(output_dir / "consumed_predictions.jsonl"),
            "overlays": _display_path(output_dir / "overlays") if overlays else None,
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", type=Path, default=onset.DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-la-chata", action="store_true")
    parser.add_argument("--overlays", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = run_experiment(
            args.out_dir,
            model_path=args.model,
            output_dir=args.output_dir,
            la_chata_inference=None if args.no_la_chata else onset.DEFAULT_LA_CHATA_INFERENCE,
            la_chata_requests=None if args.no_la_chata else onset.DEFAULT_LA_CHATA_REQUESTS,
            la_chata_truth=None if args.no_la_chata else onset.DEFAULT_LA_CHATA_TRUTH,
            overlays=args.overlays,
        )
    except (FileNotFoundError, KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(report["artifacts"]["report_json"])
    print(report["artifacts"]["report_markdown"])
    print(f"adoption gate: {report['adoption_gate']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
