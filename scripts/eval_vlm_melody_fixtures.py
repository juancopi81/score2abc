"""Evaluate VLM melody fixtures against committed note ground truth.

This is spike tooling for comparing crop variants and models. It reads fixtures
created by `scripts/record_vlm_melody_fixtures.py` and reports simple ordered
pitch/duration sanity metrics for selected measure crops.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from score2abc.melody.vlm import (  # noqa: E402
    DEFAULT_MELODY_VLM_MODEL,
    DEFAULT_OPENAI_MELODY_VLM_MODEL,
    INPUT_KINDS,
    MELODY_VLM_PROMPT_VERSION,
    NOTEHEAD_Y_PROMPT_VERSION,
    STAFF_POSITION_PROMPT_VERSION,
    TRANSCRIPTION_MODES,
    VLM_PROVIDERS,
    InputKind,
    TranscriptionMode,
    default_model_for_provider,
    fixture_model_id,
    melody_fixture_key,
    read_melody_fixture,
)
from score2abc.utils import get_logger  # noqa: E402
from scripts.record_vlm_melody_fixtures import (  # noqa: E402
    DEFAULT_FIXTURES_DIR,
    INPUT_KIND_TO_PATH_KEY,
    _apply_context_overrides,
    _load_manifest,
    _resolve_path,
    _selected_records,
)

PITCH_RE = re.compile(r"^\s*([A-Ga-g])([#b♯♭]?)(-?\d+)\s*$")
SHARP_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FLAT_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
PITCH_CLASS = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.model = args.model or default_model_for_provider(args.provider)
    logger = get_logger("score2abc.eval_vlm_melody_fixtures")

    records = list(
        _selected_records(
            _load_manifest(args.out_dir),
            selected_slugs=set(args.slug) if args.slug else None,
            selected_systems=set(args.system) if args.system else None,
            selected_measures=set(args.measure) if args.measure else None,
        )
    )
    input_kinds = INPUT_KINDS if args.input_kind == "all" else (args.input_kind,)
    report = evaluate_records(
        records,
        out_dir=args.out_dir,
        ground_truth_dir=args.ground_truth,
        fixtures_dir=args.fixtures_dir,
        input_kinds=input_kinds,
        model_id=fixture_model_id(
            args.provider,
            args.model,
            openai_reasoning_effort=args.openai_reasoning_effort,
        ),
        transcription_mode=args.transcription_mode,
        context_overrides={
            "clef_hint": args.clef_hint,
            "time_signature_hint": args.time_signature_hint,
            "key_hint": args.key_hint,
            "expected_measure_beats": args.expected_measure_beats,
        },
    )

    report_dir = args.out_dir / "vlm_melody_eval"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / (
        f"report_{args.provider}_{_safe_name(args.model)}_"
        f"{_reasoning_report_part(args.provider, args.openai_reasoning_effort)}"
        f"{args.transcription_mode}_{args.input_kind}.json"
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote evaluation report: %s", report_path)
    logger.info("Summary: %s", json.dumps(report["summary"], sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="Pipeline output directory.")
    parser.add_argument("--slug", action="append", default=None, help="Limit to a work slug.")
    parser.add_argument(
        "--system", action="append", type=int, default=None, help="Limit to a system."
    )
    parser.add_argument(
        "--measure",
        action="append",
        type=int,
        default=None,
        help="Limit to a system-local measure index.",
    )
    parser.add_argument(
        "--input-kind",
        choices=(*INPUT_KINDS, "all"),
        default="staff",
        help="Crop variant to evaluate.",
    )
    parser.add_argument(
        "--provider",
        choices=VLM_PROVIDERS,
        default="gemini",
        help="Fixture provider to evaluate. Defaults to gemini.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Provider model id. Defaults to "
            f"{DEFAULT_MELODY_VLM_MODEL} for Gemini and {DEFAULT_OPENAI_MELODY_VLM_MODEL} "
            "for OpenAI."
        ),
    )
    parser.add_argument(
        "--transcription-mode",
        choices=TRANSCRIPTION_MODES,
        default="pitch",
        help="Prompt/schema mode to evaluate.",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=DEFAULT_FIXTURES_DIR,
        help="Fixture/cache directory.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("dataset/ground_truth"),
        help="Ground-truth directory.",
    )
    parser.add_argument("--clef-hint", default=None, help="Override the context clef hint.")
    parser.add_argument(
        "--time-signature-hint",
        default=None,
        help="Override the context time-signature hint, e.g. 3/4.",
    )
    parser.add_argument("--key-hint", default=None, help="Override the context key hint.")
    parser.add_argument(
        "--expected-measure-beats",
        default=None,
        help="Override the expected measure duration in beats.",
    )
    parser.add_argument(
        "--openai-reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh"),
        default="none",
        help="OpenAI reasoning effort used when the fixture was recorded.",
    )
    return parser


def evaluate_records(
    records: Iterable[dict[str, Any]],
    *,
    out_dir: Path,
    ground_truth_dir: Path,
    fixtures_dir: Path,
    input_kinds: tuple[InputKind, ...],
    model_id: str,
    transcription_mode: TranscriptionMode = "pitch",
    context_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    truth_cache: dict[str, dict[int, list[dict[str, Any]]]] = {}
    results: list[dict[str, Any]] = []
    for record in records:
        slug = record["slug"]
        if slug not in truth_cache:
            truth_cache[slug] = _load_truth_by_measure(ground_truth_dir / f"{slug}.json")
        truth_by_measure = truth_cache[slug]
        truth_notes = truth_by_measure.get(int(record["global_measure_index"]), [])
        context = _load_context(out_dir, record, context_overrides)
        for input_kind in input_kinds:
            image_path = _resolve_path(out_dir, record["paths"][INPUT_KIND_TO_PATH_KEY[input_kind]])
            fixture_name = melody_fixture_key_for_model(
                image_path,
                model_id,
                input_kind,
                context,
                transcription_mode,
            )
            fixture_path = fixtures_dir / f"{fixture_name}.json"
            if not fixture_path.exists():
                results.append(_missing_result(record, input_kind, fixture_path, truth_notes))
                continue
            transcription = read_melody_fixture(fixture_path)
            results.append(
                _compare_fixture(
                    record,
                    input_kind,
                    image_path,
                    fixture_path,
                    transcription,
                    truth_notes,
                    transcription_mode,
                )
            )

    return {
        "summary": _summarize(results),
        "results": results,
    }


def melody_fixture_key_for_model(
    image_path: Path,
    model_id: str,
    input_kind: InputKind,
    context: dict[str, Any],
    transcription_mode: TranscriptionMode,
) -> str:
    return melody_fixture_key(
        image_path,
        prompt_version=_prompt_version_for_mode(transcription_mode),
        model_id=model_id,
        input_kind=input_kind,
        context=context,
    )


def _prompt_version_for_mode(transcription_mode: TranscriptionMode) -> str:
    if transcription_mode == "notehead_y":
        return NOTEHEAD_Y_PROMPT_VERSION
    if transcription_mode == "staff_position":
        return STAFF_POSITION_PROMPT_VERSION
    return MELODY_VLM_PROMPT_VERSION


def _load_context(
    out_dir: Path,
    record: dict[str, Any],
    context_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    context_path = _resolve_path(out_dir, record["paths"]["context"])
    context = json.loads(context_path.read_text(encoding="utf-8"))
    return _apply_context_overrides(context, context_overrides)


def _compare_fixture(
    record: dict[str, Any],
    input_kind: InputKind,
    image_path: Path,
    fixture_path: Path,
    transcription,
    truth_notes: list[dict[str, Any]],
    transcription_mode: TranscriptionMode,
) -> dict[str, Any]:
    pred_notes = [item for item in transcription.items if item.kind == "note"]
    pred_midis = [_pitch_to_midi(item.pitch) for item in pred_notes]
    truth_midis = [int(note["pitch_midi"]) for note in truth_notes]
    pred_durations = [_duration_to_beats(item.duration) for item in pred_notes]
    truth_durations = [float(note["duration_beats"]) for note in truth_notes]
    pred_staff_positions = _pred_staff_positions(
        record,
        input_kind,
        image_path,
        pred_notes,
        transcription_mode,
    )
    truth_staff_positions = [_truth_staff_position(note) for note in truth_notes]
    pitch_matches = sum(
        1 for pred, truth in zip(pred_midis, truth_midis, strict=False) if pred == truth
    )
    duration_matches = sum(
        1
        for pred, truth in zip(pred_durations, truth_durations, strict=False)
        if pred is not None and abs(pred - truth) < 1e-6
    )
    compared = max(len(pred_midis), len(truth_midis))
    result = {
        **_record_identity(record, input_kind),
        "fixture": str(fixture_path),
        "transcription_mode": transcription_mode,
        "status": "evaluated",
        "pred_note_count": len(pred_midis),
        "truth_note_count": len(truth_midis),
        "note_count_match": len(pred_midis) == len(truth_midis),
        "pitch_order_matches": pitch_matches,
        "pitch_order_accuracy": round(pitch_matches / compared, 6) if compared else 1.0,
        "duration_order_matches": duration_matches,
        "duration_order_accuracy": round(duration_matches / compared, 6) if compared else 1.0,
        "pred_pitches": [_midi_to_name(midi) if midi is not None else None for midi in pred_midis],
        "truth_pitches": [_truth_pitch_name(note) for note in truth_notes],
        "pred_durations_beats": pred_durations,
        "truth_durations_beats": truth_durations,
        "pred_confidences": [item.confidence for item in pred_notes],
        "pred_evidence": [item.evidence for item in pred_notes],
        "comments": transcription.comments,
        "overall_confidence": transcription.overall_confidence,
        "uncertainties": list(transcription.uncertainties),
    }
    if transcription_mode in {"staff_position", "notehead_y"}:
        staff_position_matches = sum(
            1
            for pred, truth in zip(pred_staff_positions, truth_staff_positions, strict=False)
            if pred == truth
        )
        result.update(
            {
                "staff_position_order_matches": staff_position_matches,
                "staff_position_order_accuracy": (
                    round(staff_position_matches / compared, 6) if compared else 1.0
                ),
                "pred_staff_positions": pred_staff_positions,
                "truth_staff_positions": truth_staff_positions,
                "pred_notehead_x_fractions": [item.notehead_x_fraction for item in pred_notes],
                "pred_notehead_y_fractions": [item.notehead_y_fraction for item in pred_notes],
            }
        )
    return result


def _pred_staff_positions(
    record: dict[str, Any],
    input_kind: InputKind,
    image_path: Path,
    pred_notes,
    transcription_mode: TranscriptionMode,
) -> list[int | None]:
    if transcription_mode == "staff_position":
        return [item.staff_position for item in pred_notes]
    if transcription_mode != "notehead_y":
        return [None for _ in pred_notes]

    line_ys = _staff_lines_for_input(record, input_kind)
    if len(line_ys) < 2:
        return [None for _ in pred_notes]
    spacing = (line_ys[-1] - line_ys[0]) / (len(line_ys) - 1)
    if spacing <= 0:
        return [None for _ in pred_notes]

    bottom_line_y = line_ys[-1]
    with Image.open(image_path) as image:
        image_height = image.height
    return [
        _notehead_y_to_staff_position(
            item.notehead_y_fraction,
            image_height=image_height,
            bottom_line_y=bottom_line_y,
            spacing=spacing,
        )
        for item in pred_notes
    ]


def _staff_lines_for_input(record: dict[str, Any], input_kind: InputKind) -> list[float]:
    if input_kind in {"pitch_ruler", "pitch_ruler_soft", "pitch_ruler_panel"}:
        pitch_ruler = record.get("pitch_ruler")
        if isinstance(pitch_ruler, dict):
            return [float(value) for value in pitch_ruler.get("staff_lines_y_px", [])]
    key = "staff_lines_y_px_in_system" if input_kind == "raw" else "staff_lines_y_px_in_staff_crop"
    return [float(value) for value in record.get(key, [])]


def _notehead_y_to_staff_position(
    y_fraction: float | None,
    *,
    image_height: int,
    bottom_line_y: float,
    spacing: float,
) -> int | None:
    if y_fraction is None or image_height <= 0:
        return None
    y = y_fraction * max(1, image_height - 1)
    return round((bottom_line_y - y) / (spacing / 2))


def _missing_result(
    record: dict[str, Any],
    input_kind: InputKind,
    fixture_path: Path,
    truth_notes: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        **_record_identity(record, input_kind),
        "fixture": str(fixture_path),
        "status": "missing_fixture",
        "truth_note_count": len(truth_notes),
    }


def _record_identity(record: dict[str, Any], input_kind: InputKind) -> dict[str, Any]:
    return {
        "slug": record["slug"],
        "system_index": record["system_index"],
        "system_measure_index": record["system_measure_index"],
        "global_measure_index": record["global_measure_index"],
        "input_kind": input_kind,
    }


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [result for result in results if result["status"] == "evaluated"]
    if not evaluated:
        return {
            "targets": len(results),
            "evaluated": 0,
            "missing": len(results),
            "note_count_match_rate": 0.0,
            "pitch_order_accuracy_avg": 0.0,
            "duration_order_accuracy_avg": 0.0,
            "staff_position_order_accuracy_avg": 0.0,
        }
    staff_position_results = [
        result for result in evaluated if "staff_position_order_accuracy" in result
    ]
    summary = {
        "targets": len(results),
        "evaluated": len(evaluated),
        "missing": len(results) - len(evaluated),
        "note_count_match_rate": round(
            sum(1 for result in evaluated if result["note_count_match"]) / len(evaluated), 6
        ),
        "pitch_order_accuracy_avg": round(
            sum(float(result["pitch_order_accuracy"]) for result in evaluated) / len(evaluated),
            6,
        ),
        "duration_order_accuracy_avg": round(
            sum(float(result["duration_order_accuracy"]) for result in evaluated) / len(evaluated),
            6,
        ),
    }
    if staff_position_results:
        summary["staff_position_order_accuracy_avg"] = round(
            sum(float(result["staff_position_order_accuracy"]) for result in staff_position_results)
            / len(staff_position_results),
            6,
        )
    return summary


def _load_truth_by_measure(path: Path) -> dict[int, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_measure: dict[int, list[dict[str, Any]]] = {}
    for note in payload.get("notes") or []:
        by_measure.setdefault(int(note["measure"]), []).append(note)
    for notes in by_measure.values():
        notes.sort(key=lambda item: float(item.get("onset_beats", 0.0)))
    return by_measure


def _pitch_to_midi(pitch: str | None) -> int | None:
    if not pitch:
        return None
    match = PITCH_RE.match(pitch.replace("♯", "#").replace("♭", "b"))
    if not match:
        return None
    letter, accidental, octave_text = match.groups()
    name = letter.upper() + accidental.replace("♯", "#").replace("♭", "b")
    pitch_class = PITCH_CLASS.get(name)
    if pitch_class is None:
        return None
    octave = int(octave_text)
    return 12 * (octave + 1) + pitch_class


def _midi_to_name(midi: int | None) -> str | None:
    if midi is None:
        return None
    octave = midi // 12 - 1
    return f"{SHARP_NAMES[midi % 12]}{octave}"


def _truth_pitch_name(note: dict[str, Any]) -> str:
    midi = int(note["pitch_midi"])
    octave = midi // 12 - 1
    names = FLAT_NAMES if int(note.get("accidental", 0)) < 0 else SHARP_NAMES
    return f"{names[midi % 12]}{octave}"


def _truth_staff_position(note: dict[str, Any]) -> int:
    name = _truth_pitch_name(note)
    letter = name[0]
    octave = int(re.search(r"-?\d+$", name).group(0))
    letter_index = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}
    e4_diatonic = 4 * 7 + letter_index["E"]
    return octave * 7 + letter_index[letter] - e4_diatonic


def _duration_to_beats(value: str) -> float | None:
    text = value.strip().lower()
    dotted = text.startswith("dotted-")
    if dotted:
        text = text.removeprefix("dotted-")
    try:
        fraction = Fraction(text)
    except ValueError:
        return None
    beats = float(fraction / Fraction(1, 4))
    if dotted:
        beats *= 1.5
    return beats


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value)


def _reasoning_report_part(provider: str, effort: str) -> str:
    if provider == "openai" and effort != "none":
        return f"reasoning-{_safe_name(effort)}_"
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
