from __future__ import annotations

from fractions import Fraction
from typing import Dict, List

from score2abc.schemas import WorkMetadata

_PITCH_CLASS = {
    0: "C",
    1: "^C",
    2: "D",
    3: "^D",
    4: "E",
    5: "F",
    6: "^F",
    7: "G",
    8: "^G",
    9: "A",
    10: "^A",
    11: "B",
}


def events_to_abc(events: Dict, metadata: WorkMetadata) -> str:
    """Convert a minimal events dict into ABC notation."""
    time_signature = events.get("time_signature") or metadata.time_signature or "4/4"
    key_hint = metadata.key_hint or "C"

    notes = events.get("notes") or []
    chords = {c.get("measure"): c.get("symbol") for c in events.get("chords", [])}

    lines: List[str] = [
        "X:1",
        f"T:{metadata.title}",
        f"C:{metadata.composer}",
        f"M:{time_signature}",
        "L:1/4",
        f"K:{key_hint}",
    ]

    measures: Dict[int, List[Dict]] = {}
    for note in notes:
        measures.setdefault(int(note["measure"]), []).append(note)

    if not measures:
        body = "C D E F |"
    else:
        body_parts: List[str] = []
        for measure in sorted(measures.keys()):
            if measure in chords:
                body_parts.append(f"\"{chords[measure]}\"")
            for note in sorted(measures[measure], key=lambda n: n.get("onset_beats", 0)):
                body_parts.append(_note_to_abc(note))
            body_parts.append("|")
        body = " ".join(body_parts)

    lines.append(body)
    return "\n".join(lines) + "\n"


def _note_to_abc(note: Dict) -> str:
    midi = int(note["pitch_midi"])
    duration = note.get("duration_beats", 1)
    pitch = _midi_to_abc(midi)
    length = _duration_to_abc(duration)
    return f"{pitch}{length}"


def _duration_to_abc(duration_beats: float) -> str:
    fraction = Fraction(duration_beats).limit_denominator(8)
    if fraction == 1:
        return ""
    if fraction.numerator == 1:
        return f"/{fraction.denominator}"
    if fraction.denominator == 1:
        return str(fraction.numerator)
    return f"{fraction.numerator}/{fraction.denominator}"


def _midi_to_abc(midi: int) -> str:
    pitch_class = midi % 12
    name = _PITCH_CLASS[pitch_class]
    octave = (midi // 12) - 1

    if octave == 4:
        return name
    if octave == 5:
        return name.lower()
    if octave > 5:
        return name.lower() + ("'" * (octave - 5))

    return name + ("," * (4 - octave))
