from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List

_STEP_TO_SEMITONE = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}

_ALTER_TO_SYMBOL = {
    -2: "bb",
    -1: "b",
    0: "",
    1: "#",
    2: "##",
}

_KIND_TO_SYMBOL = {
    "major": "",
    "minor": "m",
    "dominant": "7",
    "minor-sixth": "m6",
}


def parse_musicxml_events(path: Path) -> Dict[str, object]:
    tree = ET.parse(path)
    root = tree.getroot()
    _strip_namespaces(root)

    parts = root.findall("part")
    if not parts:
        raise ValueError(f"No <part> elements found in MusicXML: {path}")
    if len(parts) > 1:
        raise ValueError(f"Expected exactly one part in MusicXML, found {len(parts)}: {path}")

    notes: List[Dict[str, object]] = []
    chords: List[Dict[str, object]] = []
    active_ties: Dict[int, Dict[str, object]] = {}

    divisions = 1
    time_signature: str | None = None

    for measure_index, measure in enumerate(parts[0].findall("measure")):
        measure_number = _parse_measure_number(measure, fallback=measure_index)
        cursor_divisions = 0
        last_note_onset_divisions = 0

        for child in measure:
            if child.tag == "attributes":
                divisions = _parse_divisions(child, divisions)
                time_signature = _parse_time_signature(child, time_signature)
                continue

            if child.tag == "backup":
                cursor_divisions -= _int_from_child(child, "duration", default=0)
                continue

            if child.tag == "forward":
                cursor_divisions += _int_from_child(child, "duration", default=0)
                continue

            if child.tag == "harmony":
                onset_divisions = cursor_divisions + _int_from_child(child, "offset", default=0)
                chords.append(
                    {
                        "measure": measure_number,
                        "onset_beats": _round_beats(onset_divisions / divisions),
                        "symbol": _parse_harmony_symbol(child),
                    }
                )
                continue

            if child.tag != "note":
                continue

            duration_divisions = _int_from_child(child, "duration", default=0)
            if duration_divisions <= 0:
                continue

            is_chord_tone = child.find("chord") is not None
            onset_divisions = last_note_onset_divisions if is_chord_tone else cursor_divisions

            if child.find("rest") is not None:
                if not is_chord_tone:
                    cursor_divisions += duration_divisions
                    last_note_onset_divisions = onset_divisions
                continue

            pitch = child.find("pitch")
            if pitch is None:
                continue

            pitch_midi = _pitch_to_midi(pitch)
            tie_types = set(_iter_tie_types(child))
            has_tie_start = "start" in tie_types
            has_tie_stop = "stop" in tie_types
            duration_beats = _round_beats(duration_divisions / divisions)

            if has_tie_stop and pitch_midi in active_ties:
                active_ties[pitch_midi]["duration_beats"] = _round_beats(
                    float(active_ties[pitch_midi]["duration_beats"]) + duration_beats
                )
                if not has_tie_start:
                    active_ties.pop(pitch_midi, None)
            else:
                note_event: Dict[str, object] = {
                    "measure": measure_number,
                    "onset_beats": _round_beats(onset_divisions / divisions),
                    "duration_beats": duration_beats,
                    "pitch_midi": pitch_midi,
                }
                alter = _int_from_child(pitch, "alter")
                if alter is not None:
                    note_event["accidental"] = alter
                if has_tie_start:
                    note_event["tie"] = True
                notes.append(note_event)

                if has_tie_start:
                    active_ties[pitch_midi] = note_event

            if not is_chord_tone:
                cursor_divisions += duration_divisions
                last_note_onset_divisions = onset_divisions

    if time_signature is None:
        raise ValueError(f"MusicXML does not define a time signature: {path}")

    return {
        "time_signature": time_signature,
        "notes": notes,
        "chords": chords,
    }


def write_musicxml_ground_truth(input_path: Path, output_path: Path) -> Dict[str, object]:
    payload = parse_musicxml_events(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _strip_namespaces(root: ET.Element) -> None:
    for elem in root.iter():
        if isinstance(elem.tag, str) and "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]


def _parse_measure_number(measure: ET.Element, *, fallback: int) -> int:
    raw_number = measure.get("number")
    if raw_number is None:
        return fallback
    try:
        return int(raw_number)
    except ValueError:
        return fallback


def _parse_divisions(attributes: ET.Element, current: int) -> int:
    return _int_from_child(attributes, "divisions", default=current)


def _parse_time_signature(attributes: ET.Element, current: str | None) -> str | None:
    time = attributes.find("time")
    if time is None:
        return current

    beats = time.findtext("beats")
    beat_type = time.findtext("beat-type")
    if not beats or not beat_type:
        return current
    return f"{beats.strip()}/{beat_type.strip()}"


def _parse_harmony_symbol(harmony: ET.Element) -> str:
    root = harmony.find("root")
    if root is None:
        raise ValueError("Harmony element missing <root>")

    root_step = (root.findtext("root-step") or "").strip()
    if root_step not in _STEP_TO_SEMITONE:
        raise ValueError(f"Unsupported harmony root step: {root_step!r}")

    root_alter = _int_from_child(root, "root-alter", default=0)
    kind = harmony.find("kind")
    if kind is None:
        raise ValueError("Harmony element missing <kind>")

    quality = (kind.get("text") or "").strip()
    if not quality:
        quality = _KIND_TO_SYMBOL.get((kind.text or "").strip(), (kind.text or "").strip())

    symbol = f"{root_step}{_ALTER_TO_SYMBOL.get(root_alter, '')}{quality}"

    bass_step = harmony.findtext("bass/bass-step")
    if bass_step:
        bass_alter = _int_from_child(harmony.find("bass"), "bass-alter", default=0)
        symbol += f"/{bass_step.strip()}{_ALTER_TO_SYMBOL.get(bass_alter, '')}"

    return symbol


def _pitch_to_midi(pitch: ET.Element) -> int:
    step = (pitch.findtext("step") or "").strip()
    if step not in _STEP_TO_SEMITONE:
        raise ValueError(f"Unsupported pitch step: {step!r}")

    alter = _int_from_child(pitch, "alter", default=0)
    octave_text = pitch.findtext("octave")
    if octave_text is None:
        raise ValueError("Pitch element missing <octave>")

    octave = int(octave_text)
    return (octave + 1) * 12 + _STEP_TO_SEMITONE[step] + alter


def _iter_tie_types(note: ET.Element) -> Iterable[str]:
    for tie in note.findall("tie"):
        tie_type = tie.get("type")
        if tie_type:
            yield tie_type
    for tied in note.findall("notations/tied"):
        tie_type = tied.get("type")
        if tie_type:
            yield tie_type


def _int_from_child(parent: ET.Element | None, tag: str, default: int | None = None) -> int | None:
    if parent is None:
        return default
    text = parent.findtext(tag)
    if text is None:
        return default
    return int(text.strip())


def _round_beats(value: float) -> float:
    return round(value, 4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m score2abc.musicxml",
        description="Convert a MusicXML score into score2abc ground-truth events JSON.",
    )
    parser.add_argument("input_path", help="Path to the source .musicxml file")
    parser.add_argument("output_path", help="Path to the output events JSON file")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    write_musicxml_ground_truth(Path(args.input_path), Path(args.output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
