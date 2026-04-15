from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterable, List, Mapping


@dataclass(frozen=True)
class CanonicalNote:
    measure: int
    onset_beats: Fraction
    duration_beats: Fraction
    pitch_midi: int
    accidental: int | None = None
    tie: bool = False


@dataclass(frozen=True)
class NoteGroup:
    measure: int
    onset_beats: Fraction
    duration_beats: Fraction
    notes: tuple[CanonicalNote, ...]


def measure_length_beats(time_signature: str) -> Fraction:
    beats_text, beat_type_text = time_signature.split("/", maxsplit=1)
    beats = int(beats_text.strip())
    beat_type = int(beat_type_text.strip())
    if beats <= 0 or beat_type <= 0:
        raise ValueError(f"Invalid time signature: {time_signature!r}")
    return Fraction(beats * 4, beat_type)


def normalize_note_groups(raw_notes: Iterable[Mapping[str, object]]) -> Dict[int, List[NoteGroup]]:
    grouped_notes: Dict[tuple[int, Fraction], List[CanonicalNote]] = {}

    for raw_note in raw_notes:
        note = CanonicalNote(
            measure=int(raw_note["measure"]),
            onset_beats=_to_fraction(raw_note["onset_beats"]),
            duration_beats=_to_fraction(raw_note["duration_beats"]),
            pitch_midi=int(raw_note["pitch_midi"]),
            accidental=(
                int(raw_note["accidental"]) if raw_note.get("accidental") is not None else None
            ),
            tie=bool(raw_note.get("tie", False)),
        )
        if note.duration_beats <= 0:
            raise ValueError(f"Note duration must be positive: {raw_note!r}")
        key = (note.measure, note.onset_beats)
        grouped_notes.setdefault(key, []).append(note)

    by_measure: Dict[int, List[NoteGroup]] = {}
    for (measure, onset_beats), notes in grouped_notes.items():
        sorted_notes = tuple(
            sorted(notes, key=lambda note: (note.pitch_midi, note.accidental or 0))
        )
        durations = {note.duration_beats for note in sorted_notes}
        if len(durations) != 1:
            raise ValueError(
                "Simultaneous notes must share the same duration: "
                f"measure={measure}, onset={float(onset_beats)}"
            )
        group = NoteGroup(
            measure=measure,
            onset_beats=onset_beats,
            duration_beats=next(iter(durations)),
            notes=sorted_notes,
        )
        by_measure.setdefault(measure, []).append(group)

    for measure, groups in by_measure.items():
        ordered_groups = sorted(groups, key=lambda group: group.onset_beats)
        previous_end = Fraction(0)
        for group in ordered_groups:
            if group.onset_beats < previous_end:
                raise ValueError(
                    "Overlapping note groups are not supported: "
                    f"measure={measure}, onset={float(group.onset_beats)}"
                )
            previous_end = group.onset_beats + group.duration_beats
        by_measure[measure] = ordered_groups

    return by_measure


def normalize_chord_map(
    raw_chords: Iterable[Mapping[str, object]],
) -> Dict[int, Dict[Fraction, str]]:
    chord_map: Dict[int, Dict[Fraction, str]] = {}
    for raw_chord in raw_chords:
        measure = int(raw_chord["measure"])
        onset_beats = _to_fraction(raw_chord["onset_beats"])
        symbol = str(raw_chord["symbol"]).strip()
        existing = chord_map.setdefault(measure, {}).get(onset_beats)
        if existing is not None and existing != symbol:
            raise ValueError(
                "Multiple chord symbols at the same onset are not supported: "
                f"measure={measure}, onset={float(onset_beats)}"
            )
        chord_map[measure][onset_beats] = symbol
    return chord_map


def _to_fraction(value: object) -> Fraction:
    if isinstance(value, Fraction):
        return value
    return Fraction(str(value))
