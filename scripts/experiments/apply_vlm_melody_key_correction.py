"""Replay frozen melody pitches with explicit human-reviewed key events.

This spike-only review action never reruns notehead selection. It binds each
frozen canonical note to its automatic anchor, preserves candidate identity,
coordinates, order, and rhythm, and recomputes only pitch spelling/MIDI from
staff geometry plus the supplied key signature.

Example::

    uv run python scripts/experiments/apply_vlm_melody_key_correction.py \
        out/<slug>/.../inference.jsonl \
        --key-event 1=2 \
        --output-dir out/<slug>/.../review_key_correction_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import spike_consumed_polyphonic_pitch_repair as replay  # noqa: E402

SCHEMA_VERSION = 1
KIND = "vlm_melody_human_key_correction"


@dataclass(frozen=True)
class KeyEvent:
    start_measure: int
    fifths: int


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pin(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _read_inference(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Inference line {line_number} must be a JSON object")
        if value.get("truth_used") is not False:
            raise ValueError(f"Inference line {line_number} must declare truth_used=false")
        rows.append(value)
    if not rows:
        raise ValueError("Inference JSONL is empty")
    measures = [replay._identity_measure(row) for row in rows]
    if len(measures) != len(set(measures)):
        raise ValueError("Inference rows contain duplicate automatic_measure_index values")
    return sorted(rows, key=replay._identity_measure), _pin(path)


def _source_image(row: Mapping[str, Any]) -> Path:
    source = row.get("source")
    image_value = source.get("image") if isinstance(source, Mapping) else None
    expected_hash = source.get("sha256") if isinstance(source, Mapping) else None
    if not isinstance(image_value, str) or not isinstance(expected_hash, str):
        raise ValueError("Inference row has no pinned source image")
    image = Path(image_value).expanduser()
    if not image.is_absolute():
        image = REPO_ROOT / image
    image = image.resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    if _sha256(image) != expected_hash:
        raise ValueError(f"Inference source image hash mismatch: {image}")
    return image


def _bound_notes(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    measure = replay._identity_measure(row)
    canonical = row.get("canonical_prediction")
    notes = canonical.get("notes") if isinstance(canonical, Mapping) else None
    anchors = row.get("automatic_anchors")
    candidates = row.get("candidate_predictions")
    if not isinstance(notes, list):
        raise ValueError(f"Inference row {measure} has no canonical notes")
    if not isinstance(anchors, list) or len(anchors) != len(notes):
        raise ValueError(f"Inference row {measure} cannot bind notes to automatic anchors")
    if not isinstance(candidates, list):
        raise ValueError(f"Inference row {measure} has no candidate predictions")

    candidate_by_id = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError(f"Inference row {measure} has a non-object candidate")
        candidate_id = replay._candidate_id(candidate)
        if candidate_id in candidate_by_id:
            raise ValueError(f"Inference row {measure} repeats candidate {candidate_id}")
        candidate_by_id[candidate_id] = candidate

    bound = []
    for index, (note, anchor) in enumerate(zip(notes, anchors, strict=True), start=1):
        if not isinstance(note, Mapping) or not isinstance(anchor, Mapping):
            raise ValueError(f"Inference row {measure} has invalid note/anchor {index}")
        source = anchor.get("source")
        center = anchor.get("center")
        candidate_id = source.get("candidate_id") if isinstance(source, Mapping) else None
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(f"Inference row {measure} anchor {index} has no candidate_id")
        if note.get("candidate_id") not in {None, candidate_id}:
            raise ValueError(f"Inference row {measure} note/anchor candidate mismatch")
        if not isinstance(center, Mapping):
            raise ValueError(f"Inference row {measure} anchor {index} has no center")
        x = float(center["x"])
        y = float(center["y"])
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            raise ValueError(f"Inference row {measure} anchor candidate is missing: {candidate_id}")
        candidate_center = candidate.get("center")
        if not isinstance(candidate_center, Mapping) or (
            float(candidate_center["x"]),
            float(candidate_center["y"]),
        ) != (x, y):
            raise ValueError(f"Inference row {measure} candidate center changed: {candidate_id}")
        bound.append(
            {
                "candidate_id": candidate_id,
                "x": x,
                "y": y,
                "order": index,
                "note": dict(note),
            }
        )
    return bound


def _key_for_measure(events: Sequence[KeyEvent], measure: int) -> int:
    active = [event for event in events if event.start_measure <= measure]
    if not active:
        raise ValueError(f"No key event applies to automatic measure {measure}")
    return active[-1].fifths


def _correct_row(row: Mapping[str, Any], events: Sequence[KeyEvent]) -> dict[str, Any]:
    measure = replay._identity_measure(row)
    fifths = _key_for_measure(events, measure)
    alterations = replay._fifths_accidentals(fifths)
    staff_lines = replay._row_staff_lines(row)
    canonical = row["canonical_prediction"]
    bound = _bound_notes(row)
    corrected_notes = []
    pitch_changes = []
    for item in bound:
        corrected = replay._staff_pitch(item["y"], staff_lines, alterations)
        original = item["note"]
        note = {
            **original,
            "candidate_id": item["candidate_id"],
            "x": item["x"],
            "y": item["y"],
            "pitch": corrected["pitch"],
            "pitch_midi": corrected["pitch_midi"],
            "staff_position": corrected["staff_position"],
        }
        corrected_notes.append(note)
        pitch_changes.append(
            {
                "candidate_id": item["candidate_id"],
                "original_pitch_midi": int(original["pitch_midi"]),
                "corrected_pitch_midi": int(corrected["pitch_midi"]),
                "corrected_pitch": corrected["pitch"],
                "changed": int(original["pitch_midi"]) != int(corrected["pitch_midi"]),
            }
        )

    before_ids = [item["candidate_id"] for item in bound]
    after_ids = [str(note["candidate_id"]) for note in corrected_notes]
    before_coordinates = [[item["x"], item["y"]] for item in bound]
    after_coordinates = [[float(note["x"]), float(note["y"])] for note in corrected_notes]
    rhythm_fields = ("onset_beats", "duration_beats")
    rhythm_unchanged = all(
        all(original["note"].get(field) == corrected.get(field) for field in rhythm_fields)
        for original, corrected in zip(bound, corrected_notes, strict=True)
    )
    invariant = {
        "candidate_ids_unchanged": before_ids == after_ids,
        "coordinates_unchanged": before_coordinates == after_coordinates,
        "note_count_unchanged": len(bound) == len(corrected_notes),
        "note_rhythm_unchanged": rhythm_unchanged,
    }
    if not all(invariant.values()):
        raise ValueError(f"Key correction changed non-pitch data in measure {measure}")

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "identity": dict(row["identity"]),
        "key_fifths": fifths,
        "candidate_ids": after_ids,
        "notes": corrected_notes,
        "rests": list(canonical.get("rests", [])),
        "rhythm_tokens": list(canonical.get("rhythm_tokens", [])),
        "measure_extent_beats": canonical.get("measure_extent_beats"),
        "pitch_changes": pitch_changes,
        "invariant": invariant,
        "truth_used": False,
    }


def _draw_overlay(row: Mapping[str, Any], prediction: Mapping[str, Any], path: Path) -> None:
    with Image.open(_source_image(row)) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    radius = max(4, round(image.height * 0.025))
    for note in prediction["notes"]:
        x = round(float(note["x"]))
        y = round(float(note["y"]))
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=(20, 150, 60),
            width=2,
        )
        draw.text((x + radius + 2, max(0, y - radius - 4)), str(note["pitch"]), fill=(10, 90, 35))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _write_create_once(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    inference_pin: Mapping[str, Any],
    events: Sequence[KeyEvent],
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite create-once output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"Temporary create-once output already exists: {temporary}")
    temporary.mkdir()
    try:
        predictions = [_correct_row(row, events) for row in rows]
        overlay_dir = temporary / "overlays"
        for row, prediction in zip(rows, predictions, strict=True):
            measure = replay._identity_measure(row)
            _draw_overlay(row, prediction, overlay_dir / f"measure_{measure:03d}.png")

        predictions_path = temporary / "predictions.jsonl"
        predictions_path.write_text(
            "\n".join(
                json.dumps(prediction, sort_keys=True, separators=(",", ":"))
                for prediction in predictions
            )
            + "\n",
            encoding="utf-8",
        )
        changed = sum(
            bool(change["changed"])
            for prediction in predictions
            for change in prediction["pitch_changes"]
        )
        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "truth_used": False,
            "protocol": {
                "selection_replayed": False,
                "only_pitch_fields_may_change": True,
                "key_events": [
                    {"start_measure": event.start_measure, "fifths": event.fifths}
                    for event in events
                ],
            },
            "source": {"inference": dict(inference_pin)},
            "summary": {
                "measure_count": len(predictions),
                "note_count": sum(len(prediction["notes"]) for prediction in predictions),
                "changed_pitch_count": changed,
                "all_candidate_ids_unchanged": all(
                    prediction["invariant"]["candidate_ids_unchanged"] for prediction in predictions
                ),
                "all_coordinates_unchanged": all(
                    prediction["invariant"]["coordinates_unchanged"] for prediction in predictions
                ),
                "all_note_counts_unchanged": all(
                    prediction["invariant"]["note_count_unchanged"] for prediction in predictions
                ),
                "all_note_rhythm_unchanged": all(
                    prediction["invariant"]["note_rhythm_unchanged"] for prediction in predictions
                ),
            },
            "artifacts": {
                "predictions_jsonl": str(output_dir / "predictions.jsonl"),
                "overlays": str(output_dir / "overlays"),
                "report_json": str(output_dir / "report.json"),
            },
        }
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def apply_key_correction(
    inference_path: Path,
    *,
    key_events: Sequence[KeyEvent],
    output_dir: Path,
) -> dict[str, Any]:
    if not key_events:
        raise ValueError("At least one key event is required")
    ordered = sorted(key_events, key=lambda event: event.start_measure)
    starts = [event.start_measure for event in ordered]
    if len(starts) != len(set(starts)):
        raise ValueError("Key event start measures must be unique")
    if any(event.start_measure <= 0 or not -7 <= event.fifths <= 7 for event in ordered):
        raise ValueError("Key events require positive starts and fifths between -7 and 7")
    rows, inference_pin = _read_inference(inference_path)
    return _write_create_once(output_dir, rows, inference_pin, ordered)


def _key_event(value: str) -> KeyEvent:
    if "=" not in value:
        raise argparse.ArgumentTypeError("key event must be START_MEASURE=FIFTHS")
    raw_start, raw_fifths = value.split("=", 1)
    try:
        event = KeyEvent(start_measure=int(raw_start), fifths=int(raw_fifths))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("key event must be START_MEASURE=FIFTHS") from exc
    if event.start_measure <= 0 or not -7 <= event.fifths <= 7:
        raise argparse.ArgumentTypeError("key event needs a positive start and fifths from -7 to 7")
    return event


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inference", type=Path)
    parser.add_argument("--key-event", action="append", type=_key_event, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = apply_key_correction(
            args.inference,
            key_events=args.key_event,
            output_dir=args.output_dir,
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(report["artifacts"]["report_json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
