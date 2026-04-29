from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List

from score2abc.events import (
    STEP_TO_SEMITONE,
    CanonicalNote,
    NoteGroup,
    measure_length_beats,
    normalize_chord_map,
    normalize_note_groups,
)
from score2abc.schemas import WorkMetadata

_DEFAULT_PITCH_CLASS = {
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

_ACCIDENTAL_PREFIX = {
    -2: "__",
    -1: "_",
    0: "",
    1: "^",
    2: "^^",
}


@dataclass(frozen=True)
class RenderSegment:
    onset_beats: Fraction
    duration_beats: Fraction
    notes: tuple[CanonicalNote, ...]
    tie_to_next: bool = False


def events_to_abc(events: Dict, metadata: WorkMetadata) -> str:
    """Convert canonical note/chord events into ABC notation."""
    time_signature = events.get("time_signature") or metadata.time_signature or "4/4"
    measure_beats = measure_length_beats(time_signature)
    key_hint = metadata.key_hint or "C"

    note_groups = normalize_note_groups(events.get("notes") or [])
    chord_map = normalize_chord_map(events.get("chords", []) or [])
    segments_by_measure = _expand_note_groups(note_groups, chord_map, measure_beats)

    lines: List[str] = [
        "X:1",
        f"T:{metadata.title}",
        f"C:{metadata.composer}",
        f"M:{time_signature}",
        "L:1/4",
        f"K:{key_hint}",
    ]

    all_measures = sorted(set(segments_by_measure) | set(chord_map))
    if not all_measures:
        body = "C D E F |"
    else:
        measure_strings: List[str] = []
        for measure in range(all_measures[0], all_measures[-1] + 1):
            tokens = _render_measure_tokens(
                measure=measure,
                segments=segments_by_measure.get(measure, []),
                chords=chord_map.get(measure, {}),
                measure_beats=measure_beats,
            )
            measure_strings.append(" ".join(tokens))
        body = " | ".join(measure_strings) + " |"

    lines.append(body)
    return "\n".join(lines) + "\n"


def _expand_note_groups(
    note_groups: Dict[int, List[NoteGroup]],
    chord_map: Dict[int, Dict[Fraction, str]],
    measure_beats: Fraction,
) -> Dict[int, List[RenderSegment]]:
    sorted_chord_onsets: Dict[int, List[Fraction]] = {
        measure: sorted(onsets) for measure, onsets in chord_map.items()
    }
    raw: Dict[int, List[RenderSegment]] = {}
    for groups in note_groups.values():
        for group in groups:
            remaining = group.duration_beats
            current_measure = group.measure
            current_onset = group.onset_beats

            while remaining > 0:
                while current_onset >= measure_beats:
                    current_measure += 1
                    current_onset -= measure_beats

                segment_duration = min(remaining, measure_beats - current_onset)
                for chord_onset in sorted_chord_onsets.get(current_measure, ()):
                    if current_onset < chord_onset < current_onset + segment_duration:
                        segment_duration = chord_onset - current_onset
                        break

                if segment_duration <= 0:
                    raise ValueError(
                        "Cannot expand note group: non-positive segment duration at "
                        f"measure={current_measure}, onset={float(current_onset)}"
                    )

                remaining -= segment_duration
                tie_to_next = remaining > 0
                raw.setdefault(current_measure, []).append(
                    RenderSegment(
                        onset_beats=current_onset,
                        duration_beats=segment_duration,
                        notes=group.notes,
                        tie_to_next=tie_to_next,
                    )
                )

                if not tie_to_next:
                    break
                if current_onset + segment_duration >= measure_beats:
                    current_measure += 1
                    current_onset = Fraction(0)
                else:
                    current_onset += segment_duration

    return {
        measure: sorted(segments, key=lambda segment: segment.onset_beats)
        for measure, segments in raw.items()
    }


def _render_measure_tokens(
    *,
    measure: int,
    segments: List[RenderSegment],
    chords: Dict[Fraction, str],
    measure_beats: Fraction,
) -> List[str]:
    tokens: List[str] = []
    cursor = Fraction(0)
    chord_onsets = sorted(chords)

    for segment in segments:
        if segment.onset_beats > cursor:
            tokens.extend(_render_rest_tokens(cursor, segment.onset_beats, chords, chord_onsets))
        tokens.append(_attach_chord(chords.get(segment.onset_beats), _segment_to_abc(segment)))
        cursor = segment.onset_beats + segment.duration_beats

    fill_until = _measure_fill_limit(measure, segments, chords, measure_beats)
    if cursor < fill_until:
        tokens.extend(_render_rest_tokens(cursor, fill_until, chords, chord_onsets))

    if tokens:
        return tokens
    fallback_duration = fill_until if fill_until > 0 else measure_beats
    return ["z" + _duration_to_abc(fallback_duration)]


def _measure_fill_limit(
    measure: int,
    segments: List[RenderSegment],
    chords: Dict[Fraction, str],
    measure_beats: Fraction,
) -> Fraction:
    if measure != 0:
        return measure_beats

    content_end = Fraction(0)
    if segments:
        content_end = max(content_end, max(s.onset_beats + s.duration_beats for s in segments))
    if chords:
        content_end = max(content_end, max(chords))
    return content_end


def _render_rest_tokens(
    start: Fraction,
    end: Fraction,
    chords: Dict[Fraction, str],
    sorted_chord_onsets: List[Fraction],
) -> List[str]:
    tokens: List[str] = []
    cursor = start
    boundaries = [onset for onset in sorted_chord_onsets if start < onset < end]
    for boundary in boundaries + [end]:
        duration = boundary - cursor
        if duration > 0:
            rest_token = "z" + _duration_to_abc(duration)
            tokens.append(_attach_chord(chords.get(cursor), rest_token))
        cursor = boundary
    return tokens


def _segment_to_abc(segment: RenderSegment) -> str:
    duration = _duration_to_abc(segment.duration_beats)
    if len(segment.notes) == 1:
        token = _note_symbol(segment.notes[0]) + duration
    else:
        token = "[" + "".join(_note_symbol(note) for note in segment.notes) + "]" + duration
    if segment.tie_to_next:
        token += "-"
    return token


def _note_symbol(note: CanonicalNote) -> str:
    if note.accidental is None:
        return _midi_to_abc(note.pitch_midi)

    spelling = _spelled_step_octave(note.pitch_midi, note.accidental)
    if spelling is None:
        return _midi_to_abc(note.pitch_midi)

    step, octave = spelling
    return _ACCIDENTAL_PREFIX.get(note.accidental, "") + _format_octave(step, octave)


def _duration_to_abc(duration_beats: Fraction) -> str:
    fraction = duration_beats.limit_denominator(64)
    if fraction == 1:
        return ""
    if fraction.numerator == 1:
        return f"/{fraction.denominator}"
    if fraction.denominator == 1:
        return str(fraction.numerator)
    return f"{fraction.numerator}/{fraction.denominator}"


def _midi_to_abc(midi: int) -> str:
    pitch_class = midi % 12
    return _format_octave(_DEFAULT_PITCH_CLASS[pitch_class], (midi // 12) - 1)


def _spelled_step_octave(midi: int, accidental: int) -> tuple[str, int] | None:
    pitch_class = midi % 12
    for step, natural_pitch_class in STEP_TO_SEMITONE.items():
        if (natural_pitch_class + accidental) % 12 != pitch_class:
            continue
        natural_midi = midi - accidental
        octave = (natural_midi // 12) - 1
        if (octave + 1) * 12 + natural_pitch_class == natural_midi:
            return step, octave
    return None


def _format_octave(name: str, octave: int) -> str:
    if octave == 4:
        return name
    if octave == 5:
        return name.lower()
    if octave > 5:
        return name.lower() + ("'" * (octave - 5))
    return name + ("," * (4 - octave))


def _attach_chord(symbol: str | None, token: str) -> str:
    if not symbol:
        return token
    return f'"{symbol}"{token}'
