"""Replay fixed noteheads through key-state and local-staff pitch mapping.

This is a consumed, post-truth experiment. The committed fixture keeps requests
and truth separate. Every prediction lane is materialized and hash-pinned before
truth is opened, and candidate IDs plus coordinates must remain identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.evaluate_frozen_third_score_heldout import (  # noqa: E402
    _align_pitches,
)
from scripts.experiments.spike_consumed_key_signature_detector import (  # noqa: E402
    detect_signature,
)
from scripts.experiments.spike_consumed_polyphonic_pitch_repair import (  # noqa: E402
    _fifths_accidentals,
    _staff_pitch,
)
from scripts.experiments.spike_local_staff_tracking import (  # noqa: E402
    track_common_staff_shift,
)

SCHEMA_VERSION = 1
FIXTURE_KIND = "consumed_cross_score_pitch_mapping_fixture"
REPORT_KIND = "consumed_cross_score_pitch_mapping_spike"
DEFAULT_FIXTURE_DIR = REPO_ROOT / "tests/fixtures/vlm_melody/cross_score_pitch_mapping"
DEFAULT_OUTPUT_DIR = Path("out/vlm_melody_consumed_training/cross_score_pitch_mapping_v2")
LANE_BASELINE = "frozen_baseline"
LANE_GLOBAL_FROZEN = "global_staff_frozen_key"
LANE_GLOBAL_AUTOMATIC = "global_staff_automatic_key"
LANE_TRACKED_FROZEN = "tracked_staff_frozen_key"
LANE_TRACKED_AUTOMATIC = "tracked_staff_automatic_key"
LANES = (
    LANE_BASELINE,
    LANE_GLOBAL_FROZEN,
    LANE_GLOBAL_AUTOMATIC,
    LANE_TRACKED_FROZEN,
    LANE_TRACKED_AUTOMATIC,
)
PITCH_RE = re.compile(r"^([A-G])([#b]?)(-?\d+)$")
NATURAL_PITCH_CLASSES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    try:
        destination = run_experiment(
            fixture_dir=args.fixture_dir,
            output_dir=args.output_dir,
        )
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(destination)
    return 0


def run_experiment(*, fixture_dir: Path, output_dir: Path) -> Path:
    fixture_root = fixture_dir.expanduser().resolve()
    manifest_path = fixture_root / "manifest.json"
    manifest = _load_object(manifest_path, "Fixture manifest")
    _validate_manifest(manifest, fixture_root)
    requests_path = _pinned_path(fixture_root, manifest["requests"], "requests")
    truth_path = _pinned_path(fixture_root, manifest["truth"], "truth")
    requests = _load_jsonl(requests_path, "Pitch requests")
    _validate_requests(requests, manifest, fixture_root)

    destination = output_dir.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite pitch experiment: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"Refusing stale pitch experiment temp directory: {temporary}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()

    try:
        key_state = _materialize_key_state(manifest, fixture_root, temporary)
        predictions, tracking = _materialize_all_predictions(
            requests,
            fixture_root=fixture_root,
            key_state=key_state,
        )
        invariance = _selection_invariance(predictions)
        if not invariance["all_candidate_ids_equal"]:
            raise ValueError("Pitch lanes changed candidate IDs")
        if not invariance["all_coordinates_equal"]:
            raise ValueError("Pitch lanes changed notehead coordinates")
        if not invariance["all_note_counts_equal"]:
            raise ValueError("Pitch lanes changed note counts")

        prediction_records = {}
        for lane in LANES:
            path = temporary / f"predictions_{lane}.jsonl"
            _write_jsonl(path, predictions[lane])
            prediction_records[lane] = _local_record(path, temporary)
        tracking_path = temporary / "tracking_diagnostics.json"
        _write_json(tracking_path, tracking)
        tracking_record = _local_record(tracking_path, temporary)
        prediction_seal_path = temporary / "prediction_seal.json"
        _write_json(
            prediction_seal_path,
            {
                "schema_version": SCHEMA_VERSION,
                "kind": f"{REPORT_KIND}_prediction_seal",
                "truth_opened": False,
                "automatic_key_state": key_state["artifact"],
                "predictions": prediction_records,
                "tracking_diagnostics": tracking_record,
                "selection_invariance": invariance,
            },
        )
        prediction_seal_record = _local_record(prediction_seal_path, temporary)

        # Truth is intentionally opened only after every prediction file exists.
        truth = _load_jsonl(truth_path, "Pitch truth")
        evaluation = _evaluate(predictions, truth)
        replay_difference = _pitch_lane_differences(
            predictions[LANE_BASELINE], predictions[LANE_GLOBAL_FROZEN]
        )
        decisions = _decisions(evaluation, invariance, replay_difference)
        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": REPORT_KIND,
            "status": "evaluated_consumed_evidence_not_heldout",
            "protocol": {
                "predictions_materialized_before_truth_open": True,
                "prediction_seal_written_before_truth_open": True,
                "candidate_ids_frozen": True,
                "coordinates_frozen": True,
                "note_counts_frozen": True,
                "segmentation_confounded_crops_excluded_in_fixture": True,
                "primary_key_lane": LANE_GLOBAL_AUTOMATIC,
                "primary_geometry_lane": LANE_TRACKED_FROZEN,
                "combined_lane": LANE_TRACKED_AUTOMATIC,
                "heldout_promotion_requirement": (
                    "repeat the frozen winning component on at least two new independent scores"
                ),
            },
            "source": {
                "fixture_manifest": _file_record(manifest_path),
                "requests": _file_record(requests_path),
                "truth": _file_record(truth_path),
                "score_count": len(manifest["scores"]),
                "measure_count": len(requests),
            },
            "automatic_key_state": key_state,
            "selection_invariance": invariance,
            "evaluation": evaluation,
            "decisions": decisions,
            "artifacts": {
                "predictions": prediction_records,
                "prediction_seal": prediction_seal_record,
                "tracking_diagnostics": tracking_record,
            },
            "interpretation_limits": [
                "All four scores are consumed; no lane is eligible for runtime promotion.",
                "Ordered-pitch alignment still confounds wrong selected heads with pitch mapping.",
                "Explicit in-measure accidentals are not detected by this experiment.",
                "Local staff tracking follows one common vertical shift for all five lines.",
            ],
        }
        report_path = temporary / "report.json"
        _write_json(report_path, report)
        summary_path = temporary / "summary.md"
        summary_path.write_text(_render_markdown(report), encoding="utf-8")
        _write_json(
            temporary / "artifact_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "kind": f"{REPORT_KIND}_artifact_manifest",
                "create_once": True,
                "report": _local_record(report_path, temporary),
                "summary": _local_record(summary_path, temporary),
                "truth_opened_after_prediction_hashes": True,
            },
        )
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _validate_manifest(manifest: Mapping[str, Any], fixture_root: Path) -> None:
    if manifest.get("kind") != FIXTURE_KIND:
        raise ValueError(f"Unexpected fixture kind: {manifest.get('kind')!r}")
    if manifest.get("evidence_tier") != "consumed_postmortem":
        raise ValueError("Pitch fixture must be labeled consumed_postmortem")
    if manifest.get("eligible_for_heldout_claim") is not False:
        raise ValueError("Pitch fixture cannot support a heldout claim")
    scores = manifest.get("scores")
    if not isinstance(scores, list) or len(scores) < 2:
        raise ValueError("Pitch fixture must contain at least two scores")
    score_ids = [str(score.get("score_id")) for score in scores if isinstance(score, Mapping)]
    if len(score_ids) != len(scores) or len(set(score_ids)) != len(score_ids):
        raise ValueError("Pitch fixture score IDs must be unique")
    for key in ("requests", "truth"):
        _pinned_path(fixture_root, manifest.get(key), key)
    key_events = manifest.get("key_events")
    if not isinstance(key_events, list) or not key_events:
        raise ValueError("Pitch fixture needs automatic key events")
    for event in key_events:
        if not isinstance(event, Mapping):
            raise ValueError("Pitch key event must be an object")
        if event.get("score_id") not in score_ids:
            raise ValueError("Pitch key event references an unknown score")
        if event.get("mode") not in {"initial", "change"}:
            raise ValueError("Pitch key event mode must be initial or change")
        _positive_int(event.get("start_measure"), "key event start_measure")
        _pinned_path(fixture_root, event.get("image"), "key event image")
        fallback = event.get("fallback_fifths")
        if fallback is not None and (isinstance(fallback, bool) or not isinstance(fallback, int)):
            raise ValueError("fallback_fifths must be an integer or null")


def _validate_requests(
    requests: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    fixture_root: Path,
) -> None:
    score_ids = {str(score["score_id"]) for score in manifest["scores"]}
    identities = set()
    for request in requests:
        score_id = str(request.get("score_id"))
        if score_id not in score_ids:
            raise ValueError(f"Request references unknown score: {score_id}")
        measure = _positive_int(request.get("measure_index"), "measure_index")
        identity = (score_id, measure)
        if identity in identities:
            raise ValueError(f"Duplicate pitch request identity: {identity}")
        identities.add(identity)
        if request.get("segmentation_confounded") is not False:
            raise ValueError("Pitch requests must exclude segmentation-confounded crops")
        lines = request.get("staff_lines_y_px")
        if not _ordered_numbers(lines, expected_count=5):
            raise ValueError(f"Request {identity} has invalid staff lines")
        image_path = _pinned_path(fixture_root, request.get("image"), "request image")
        with Image.open(image_path) as opened:
            width, height = opened.size
        notes = request.get("notes")
        if not isinstance(notes, list):
            raise ValueError(f"Request {identity} notes must be an array")
        candidate_ids = set()
        for note in notes:
            if not isinstance(note, Mapping):
                raise ValueError(f"Request {identity} note must be an object")
            candidate_id = str(note.get("candidate_id"))
            if not candidate_id or candidate_id in candidate_ids:
                raise ValueError(f"Request {identity} candidate IDs must be unique")
            candidate_ids.add(candidate_id)
            x = _finite_number(note.get("x"), "note x")
            y = _finite_number(note.get("y"), "note y")
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"Request {identity} note coordinate is outside the image")
            _pitch_to_midi(str(note.get("baseline_pitch")))


def _materialize_key_state(
    manifest: Mapping[str, Any],
    fixture_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    predictions = []
    for event in manifest["key_events"]:
        image_path = _pinned_path(fixture_root, event["image"], "key event image")
        error = None
        try:
            prediction = detect_signature(image_path, mode=str(event["mode"]))
            predicted_fifths = prediction.get("fifths")
            detector = {
                "gate_passed": bool(prediction["gate_passed"]),
                "predicted_signature_family": prediction["predicted_signature_family"],
                "selected_glyph_ids": list(prediction["selected_glyph_ids"]),
            }
        except ValueError as exc:
            predicted_fifths = None
            detector = {
                "gate_passed": False,
                "predicted_signature_family": None,
                "selected_glyph_ids": [],
            }
            error = str(exc)
        effective = (
            int(predicted_fifths) if predicted_fifths is not None else event.get("fallback_fifths")
        )
        predictions.append(
            {
                "score_id": event["score_id"],
                "mode": event["mode"],
                "start_measure": int(event["start_measure"]),
                "predicted_fifths": predicted_fifths,
                "fallback_fifths": event.get("fallback_fifths"),
                "effective_fifths": effective,
                "fallback_used": predicted_fifths is None,
                "detector_error": error,
                "detector": detector,
                "image": _file_record(image_path),
                "truth_used_for_prediction": False,
            }
        )
    path = output_dir / "automatic_key_state.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "cross_score_automatic_key_state",
        "truth_used": False,
        "events": predictions,
    }
    _write_json(path, payload)
    return {**payload, "artifact": _local_record(path, output_dir)}


def _materialize_all_predictions(
    requests: Sequence[Mapping[str, Any]],
    *,
    fixture_root: Path,
    key_state: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    predictions = {lane: [] for lane in LANES}
    tracking_rows = []
    for request in requests:
        score_id = str(request["score_id"])
        measure = int(request["measure_index"])
        image_path = _pinned_path(fixture_root, request["image"], "request image")
        lines = [float(value) for value in request["staff_lines_y_px"]]
        spacing = sum(right - left for left, right in zip(lines, lines[1:], strict=False)) / 4.0
        with Image.open(image_path) as opened:
            image = opened.convert("L")
        shifts = track_common_staff_shift(image, base_lines=lines, spacing=spacing)
        frozen_fifths = request.get("frozen_fifths")
        automatic_fifths = _effective_fifths(
            key_state,
            score_id=score_id,
            measure=measure,
            default=frozen_fifths,
        )
        lane_notes = {lane: [] for lane in LANES}
        note_diagnostics = []
        for note in request["notes"]:
            x = float(note["x"])
            y = float(note["y"])
            x_index = min(max(round(x), 0), len(shifts) - 1)
            local_shift = int(shifts[x_index])
            local_lines = [line + local_shift for line in lines]
            baseline_midi = _pitch_to_midi(str(note["baseline_pitch"]))
            pitch_by_lane = {
                LANE_BASELINE: baseline_midi,
                LANE_GLOBAL_FROZEN: _mapped_midi(y, lines, frozen_fifths),
                LANE_GLOBAL_AUTOMATIC: _mapped_midi(y, lines, automatic_fifths),
                LANE_TRACKED_FROZEN: _mapped_midi(y, local_lines, frozen_fifths),
                LANE_TRACKED_AUTOMATIC: _mapped_midi(y, local_lines, automatic_fifths),
            }
            for lane, pitch_midi in pitch_by_lane.items():
                lane_notes[lane].append(
                    {
                        "candidate_id": note["candidate_id"],
                        "x": x,
                        "y": y,
                        "pitch_midi": pitch_midi,
                    }
                )
            note_diagnostics.append(
                {
                    "candidate_id": note["candidate_id"],
                    "x": x,
                    "y": y,
                    "local_shift_px": local_shift,
                    "pitch_midi_by_lane": pitch_by_lane,
                }
            )
        for lane in LANES:
            predictions[lane].append(
                {
                    "score_id": score_id,
                    "measure_index": measure,
                    "lane": lane,
                    "frozen_fifths": frozen_fifths,
                    "automatic_fifths": automatic_fifths,
                    "notes": lane_notes[lane],
                    "truth_used": False,
                }
            )
        tracking_rows.append(
            {
                "score_id": score_id,
                "measure_index": measure,
                "image": _file_record(image_path),
                "base_staff_lines_y_px": lines,
                "staff_spacing_px": spacing,
                "shift_min_px": min(shifts),
                "shift_max_px": max(shifts),
                "shift_at_noteheads_px": [row["local_shift_px"] for row in note_diagnostics],
                "notes": note_diagnostics,
                "truth_used": False,
            }
        )
    return predictions, {"schema_version": SCHEMA_VERSION, "rows": tracking_rows}


def _effective_fifths(
    key_state: Mapping[str, Any],
    *,
    score_id: str,
    measure: int,
    default: Any,
) -> int | None:
    effective = default
    events = sorted(
        (
            event
            for event in key_state["events"]
            if event["score_id"] == score_id and int(event["start_measure"]) <= measure
        ),
        key=lambda event: int(event["start_measure"]),
    )
    for event in events:
        effective = event["effective_fifths"]
    return int(effective) if effective is not None else None


def _mapped_midi(y: float, lines: Sequence[float], fifths: Any) -> int:
    alterations = _fifths_accidentals(int(fifths)) if fifths is not None else {}
    return int(_staff_pitch(y, lines, alterations)["pitch_midi"])


def _selection_invariance(
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    baseline = {
        (str(row["score_id"]), int(row["measure_index"])): row for row in predictions[LANE_BASELINE]
    }
    comparisons = []
    for lane in LANES[1:]:
        lane_rows = {
            (str(row["score_id"]), int(row["measure_index"])): row for row in predictions[lane]
        }
        if set(lane_rows) != set(baseline):
            raise ValueError(f"Prediction identities differ for lane {lane}")
        for identity in sorted(baseline):
            left_notes = baseline[identity]["notes"]
            right_notes = lane_rows[identity]["notes"]
            left_ids = [note["candidate_id"] for note in left_notes]
            right_ids = [note["candidate_id"] for note in right_notes]
            left_coordinates = [(float(note["x"]), float(note["y"])) for note in left_notes]
            right_coordinates = [(float(note["x"]), float(note["y"])) for note in right_notes]
            comparisons.append(
                {
                    "score_id": identity[0],
                    "measure_index": identity[1],
                    "lane": lane,
                    "candidate_ids_equal": left_ids == right_ids,
                    "coordinates_equal": left_coordinates == right_coordinates,
                    "note_counts_equal": len(left_notes) == len(right_notes),
                }
            )
    return {
        "all_candidate_ids_equal": all(row["candidate_ids_equal"] for row in comparisons),
        "all_coordinates_equal": all(row["coordinates_equal"] for row in comparisons),
        "all_note_counts_equal": all(row["note_counts_equal"] for row in comparisons),
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
    }


def _pitch_lane_differences(
    baseline: Sequence[Mapping[str, Any]],
    replay: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    replay_by_identity = {(str(row["score_id"]), int(row["measure_index"])): row for row in replay}
    changed = []
    for baseline_row in baseline:
        identity = (str(baseline_row["score_id"]), int(baseline_row["measure_index"]))
        replay_row = replay_by_identity.get(identity)
        if replay_row is None:
            raise ValueError(f"Global replay is missing identity {identity}")
        for baseline_note, replay_note in zip(
            baseline_row["notes"], replay_row["notes"], strict=True
        ):
            if int(baseline_note["pitch_midi"]) != int(replay_note["pitch_midi"]):
                changed.append(
                    {
                        "score_id": identity[0],
                        "measure_index": identity[1],
                        "candidate_id": baseline_note["candidate_id"],
                        "baseline_pitch_midi": int(baseline_note["pitch_midi"]),
                        "replay_pitch_midi": int(replay_note["pitch_midi"]),
                    }
                )
    return {
        "pitch_value_parity": not changed,
        "changed_pitch_count": len(changed),
        "changed_measure_count": len({(row["score_id"], row["measure_index"]) for row in changed}),
        "changes": changed,
    }


def _evaluate(
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    truth: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    truth_by_identity = {(str(row["score_id"]), int(row["measure_index"])): row for row in truth}
    if len(truth_by_identity) != len(truth):
        raise ValueError("Pitch truth contains duplicate identities")
    lanes = {}
    for lane in LANES:
        per_measure = []
        for prediction in predictions[lane]:
            identity = (str(prediction["score_id"]), int(prediction["measure_index"]))
            truth_row = truth_by_identity.get(identity)
            if truth_row is None:
                raise ValueError(f"Pitch truth is missing identity {identity}")
            predicted_pitches = [int(note["pitch_midi"]) for note in prediction["notes"]]
            truth_pitches = [int(value) for value in truth_row["pitch_midi"]]
            alignment = _align_pitches(predicted_pitches, truth_pitches)
            per_measure.append(
                {
                    "score_id": identity[0],
                    "measure_index": identity[1],
                    "predicted_pitches": predicted_pitches,
                    "truth_pitches": truth_pitches,
                    "alignment": alignment,
                }
            )
        lanes[lane] = _lane_summary(per_measure)
    if set(truth_by_identity) != {
        (str(row["score_id"]), int(row["measure_index"])) for row in predictions[LANE_BASELINE]
    }:
        raise ValueError("Pitch truth identities do not exactly match prediction requests")
    return {"truth_opened": True, "lanes": lanes}


def _lane_summary(per_measure: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_score: dict[str, list[Mapping[str, Any]]] = {}
    for row in per_measure:
        by_score.setdefault(str(row["score_id"]), []).append(row)

    def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        matches = sum(int(row["alignment"]["exact_pitch_matches"]) for row in rows)
        edits = sum(int(row["alignment"]["edit_distance"]) for row in rows)
        predicted = sum(len(row["predicted_pitches"]) for row in rows)
        expected = sum(len(row["truth_pitches"]) for row in rows)
        capacity = sum(
            min(len(row["predicted_pitches"]), len(row["truth_pitches"])) for row in rows
        )
        return {
            "measure_count": len(rows),
            "predicted_note_count": predicted,
            "truth_note_count": expected,
            "count_capacity": capacity,
            "exact_pitch_matches": matches,
            "conditional_exact_pitch_accuracy": round(matches / capacity, 6) if capacity else None,
            "ordered_pitch_edit_distance": edits,
        }

    return {
        "aggregate": summarize(per_measure),
        "by_score": {score_id: summarize(rows) for score_id, rows in sorted(by_score.items())},
        "per_measure": list(per_measure),
    }


def _decisions(
    evaluation: Mapping[str, Any],
    invariance: Mapping[str, Any],
    replay_difference: Mapping[str, Any],
) -> dict[str, Any]:
    lanes = evaluation["lanes"]
    baseline = lanes[LANE_BASELINE]
    global_frozen = lanes[LANE_GLOBAL_FROZEN]
    key_lane = lanes[LANE_GLOBAL_AUTOMATIC]
    geometry_lane = lanes[LANE_TRACKED_FROZEN]
    combined_lane = lanes[LANE_TRACKED_AUTOMATIC]
    invariant = bool(
        invariance["all_candidate_ids_equal"]
        and invariance["all_coordinates_equal"]
        and invariance["all_note_counts_equal"]
    )
    metric_parity = (
        baseline["aggregate"]["exact_pitch_matches"]
        == global_frozen["aggregate"]["exact_pitch_matches"]
        and baseline["aggregate"]["ordered_pitch_edit_distance"]
        == global_frozen["aggregate"]["ordered_pitch_edit_distance"]
    )
    key_decision = _component_decision(
        baseline=global_frozen,
        candidate=key_lane,
        invariant=invariant,
        component="automatic key-state mapping",
    )
    key_decision["aggregate_exact_pitch_delta_vs_frozen_baseline"] = int(
        key_lane["aggregate"]["exact_pitch_matches"]
    ) - int(baseline["aggregate"]["exact_pitch_matches"])
    return {
        "global_mapper_reference": {
            "lane": LANE_GLOBAL_FROZEN,
            "pitch_value_parity": replay_difference["pitch_value_parity"],
            "truth_metric_parity": metric_parity,
            "exact_pitch_delta_vs_frozen_baseline": int(
                global_frozen["aggregate"]["exact_pitch_matches"]
            )
            - int(baseline["aggregate"]["exact_pitch_matches"]),
            "pitch_differences": replay_difference,
            "status": "identical" if metric_parity else "diagnostic_difference",
            "interpretation": (
                "Component gates compare against this replay lane so key and geometry effects "
                "remain isolated from historical mapper differences."
            ),
        },
        "automatic_key_state": key_decision,
        "local_staff_tracking": _component_decision(
            baseline=global_frozen,
            candidate=geometry_lane,
            invariant=invariant,
            component="local common-shift staff tracking",
        ),
        "combined": _component_decision(
            baseline=global_frozen,
            candidate=combined_lane,
            invariant=invariant,
            component="automatic key-state plus local staff tracking",
        ),
        "runtime_promotion": {
            "eligible": False,
            "status": "blocked_requires_two_new_independent_scores",
            "reason": "this fixture contains consumed postmortem evidence only",
        },
    }


def _component_decision(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    invariant: bool,
    component: str,
) -> dict[str, Any]:
    baseline_scores = baseline["by_score"]
    candidate_scores = candidate["by_score"]
    improvements = []
    regressions = []
    unchanged = []
    for score_id in sorted(baseline_scores):
        delta = int(candidate_scores[score_id]["exact_pitch_matches"]) - int(
            baseline_scores[score_id]["exact_pitch_matches"]
        )
        if delta > 0:
            improvements.append({"score_id": score_id, "exact_pitch_delta": delta})
        elif delta < 0:
            regressions.append({"score_id": score_id, "exact_pitch_delta": delta})
        else:
            unchanged.append(score_id)
    aggregate_delta = int(candidate["aggregate"]["exact_pitch_matches"]) - int(
        baseline["aggregate"]["exact_pitch_matches"]
    )
    consumed_gate = invariant and aggregate_delta > 0 and len(improvements) >= 2 and not regressions
    return {
        "component": component,
        "aggregate_exact_pitch_delta": aggregate_delta,
        "improved_scores": improvements,
        "regressed_scores": regressions,
        "unchanged_scores": unchanged,
        "localization_invariant": invariant,
        "consumed_development_gate_passed": consumed_gate,
        "heldout_status": "not_run",
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    lanes = report["evaluation"]["lanes"]
    decisions = report["decisions"]
    lines = [
        "# Cross-Score Pitch-Mapping Spike",
        "",
        "Consumed postmortem evidence only. Candidate IDs, coordinates, and note counts are fixed.",
        "",
        "| Lane | Exact pitches | Capacity | Accuracy | Edit distance |",
        "|---|---:|---:|---:|---:|",
    ]
    for lane in LANES:
        metrics = lanes[lane]["aggregate"]
        lines.append(
            f"| {lane} | {metrics['exact_pitch_matches']} | {metrics['count_capacity']} | "
            f"{metrics['conditional_exact_pitch_accuracy']:.3f} | "
            f"{metrics['ordered_pitch_edit_distance']} |"
        )
    lines.extend(["", "## Component decisions", ""])
    reference = decisions["global_mapper_reference"]
    lines.append(
        "- Global mapper reference: "
        f"`{reference['pitch_differences']['changed_pitch_count']}` changed pitch values and "
        f"truth-metric delta `{reference['exact_pitch_delta_vs_frozen_baseline']:+d}` versus the "
        "historical frozen baseline."
    )
    for key in ("automatic_key_state", "local_staff_tracking", "combined"):
        decision = decisions[key]
        lines.append(
            f"- {decision['component']}: delta `{decision['aggregate_exact_pitch_delta']:+d}`, "
            f"consumed gate `{decision['consumed_development_gate_passed']}`."
        )
    lines.extend(
        [
            "",
            "## Promotion status",
            "",
            "Runtime promotion remains blocked until the winning component improves exact ordered "
            "pitch on at least two new independent scores with identical localization.",
            "",
        ]
    )
    return "\n".join(lines)


def _pitch_to_midi(pitch: str) -> int:
    match = PITCH_RE.fullmatch(pitch)
    if match is None:
        raise ValueError(f"Invalid scientific pitch: {pitch!r}")
    letter, accidental, raw_octave = match.groups()
    alteration = 1 if accidental == "#" else -1 if accidental == "b" else 0
    return 12 * (int(raw_octave) + 1) + NATURAL_PITCH_CLASSES[letter] + alteration


def _ordered_numbers(value: Any, *, expected_count: int) -> bool:
    if not isinstance(value, list) or len(value) != expected_count:
        return False
    try:
        numbers = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(item) for item in numbers) and all(
        left < right for left, right in zip(numbers, numbers[1:], strict=False)
    )


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _pinned_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} pin must be an object")
    raw_path = value.get("path")
    expected = value.get("sha256")
    if not isinstance(raw_path, str) or not raw_path or not isinstance(expected, str):
        raise ValueError(f"{label} pin is incomplete")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} pin path must be fixture-relative")
    path = (root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} artifact is missing: {path}")
    if _sha256(path) != expected:
        raise ValueError(f"{label} artifact hash mismatch: {path}")
    return path


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} has invalid JSON on line {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label} line {line_number} must be an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{label} is empty")
    return rows


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display = resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        display = str(resolved)
    return {"path": display, "sha256": _sha256(resolved), "bytes": resolved.stat().st_size}


def _local_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
