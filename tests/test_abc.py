import json
from pathlib import Path

import pytest

from score2abc.abc import events_to_abc
from score2abc.schemas import WorkMetadata


def test_events_to_abc_formats_headers_and_body() -> None:
    metadata = WorkMetadata(
        title="Test Tune",
        composer="Composer",
        rhythm="Pasillo",
        time_signature="3/4",
        key_hint="Em",
    )
    events = {
        "time_signature": "3/4",
        "notes": [
            {"measure": 1, "onset_beats": 0.0, "duration_beats": 1.0, "pitch_midi": 60},
            {"measure": 1, "onset_beats": 1.0, "duration_beats": 1.0, "pitch_midi": 62},
            {"measure": 2, "onset_beats": 0.0, "duration_beats": 2.0, "pitch_midi": 64},
        ],
        "chords": [
            {"measure": 1, "onset_beats": 0.0, "symbol": "Em"},
            {"measure": 2, "onset_beats": 0.0, "symbol": "B7"},
        ],
    }

    abc = events_to_abc(events, metadata)
    assert "M:3/4" in abc
    assert "K:Em" in abc
    assert '"Em"C D z | "B7"E2 z |' in abc


def test_events_to_abc_supports_rests_ties_and_simultaneous_notes() -> None:
    metadata = WorkMetadata(title="Layered Tune", composer="Composer", rhythm="Pasillo")
    events = {
        "time_signature": "3/4",
        "notes": [
            {"measure": 1, "onset_beats": 1.0, "duration_beats": 1.0, "pitch_midi": 60},
            {
                "measure": 1,
                "onset_beats": 2.0,
                "duration_beats": 4.0,
                "pitch_midi": 66,
                "accidental": 1,
            },
            {"measure": 1, "onset_beats": 2.0, "duration_beats": 4.0, "pitch_midi": 74},
        ],
        "chords": [
            {"measure": 1, "onset_beats": 0.0, "symbol": "Cm"},
            {"measure": 1, "onset_beats": 1.0, "symbol": "G7"},
            {"measure": 2, "onset_beats": 0.0, "symbol": "D"},
        ],
    }

    abc = events_to_abc(events, metadata)
    assert '"Cm"z "G7"C [^Fd]- | "D"[^Fd]3 |' in abc


def test_events_to_abc_splits_notes_when_chords_change_mid_sustain() -> None:
    metadata = WorkMetadata(title="Chord Shift", composer="Composer", rhythm="Pasillo")
    events = {
        "time_signature": "3/4",
        "notes": [
            {"measure": 1, "onset_beats": 1.0, "duration_beats": 2.0, "pitch_midi": 60},
        ],
        "chords": [
            {"measure": 1, "onset_beats": 0.0, "symbol": "Cm"},
            {"measure": 1, "onset_beats": 2.0, "symbol": "G7"},
        ],
    }

    abc = events_to_abc(events, metadata)
    assert '"Cm"z C- "G7"C |' in abc


def test_events_to_abc_merges_different_chords_at_same_onset() -> None:
    metadata = WorkMetadata(title="Chord Collision", composer="Composer", rhythm="Pasillo")
    events = {
        "time_signature": "3/4",
        "notes": [
            {"measure": 1, "onset_beats": 0.0, "duration_beats": 1.0, "pitch_midi": 60},
        ],
        "chords": [
            {"measure": 1, "onset_beats": 0.0, "symbol": "Am"},
            {"measure": 1, "onset_beats": 0.0, "symbol": "Gm"},
            {"measure": 1, "onset_beats": 0.0, "symbol": "Am"},
        ],
    }

    abc = events_to_abc(events, metadata)

    assert '"Am Gm"C z2 |' in abc


def test_events_to_abc_carries_out_of_range_onsets_forward() -> None:
    metadata = WorkMetadata(title="Malformed OMR", composer="Engine", rhythm="Pasillo")
    events = {
        "time_signature": "1/4",
        "notes": [
            {"measure": 1, "onset_beats": 1.0, "duration_beats": 0.5, "pitch_midi": 60},
            {"measure": 1, "onset_beats": 2.0, "duration_beats": 0.5, "pitch_midi": 62},
        ],
        "chords": [],
    }

    abc = events_to_abc(events, metadata)

    assert "C/2 z/2 | D/2 z/2 |" in abc


def test_events_to_abc_rejects_overlapping_note_groups() -> None:
    metadata = WorkMetadata(title="Broken Tune", composer="Composer", rhythm="Pasillo")
    events = {
        "time_signature": "3/4",
        "notes": [
            {"measure": 1, "onset_beats": 0.0, "duration_beats": 2.0, "pitch_midi": 60},
            {"measure": 1, "onset_beats": 1.0, "duration_beats": 1.0, "pitch_midi": 62},
        ],
        "chords": [],
    }

    with pytest.raises(ValueError, match="Overlapping note groups"):
        events_to_abc(events, metadata)


def test_events_to_abc_handles_real_aviador_ground_truth() -> None:
    metadata = WorkMetadata(
        title="Aviador",
        composer="Fulgencio García",
        rhythm="Pasillo",
        time_signature="3/4",
    )
    events_path = Path("dataset/ground_truth/jaime-llanos_12_aviador_pasillo_fulgencio-garcia.json")
    events = json.loads(events_path.read_text(encoding="utf-8"))

    abc = events_to_abc(events, metadata)

    assert abc.startswith("X:1\nT:Aviador\nC:Fulgencio García\nM:3/4\nL:1/4\nK:C\n")
    assert "A,/2 _B,/2 ^C/2 E/2 G/2 |" in abc
    assert '"Gm6"_B3/2 A/2 c |' in abc
    assert "[^Fd]3- | [^Fd] z2 |" in abc
    assert "z" in abc
    assert "[" in abc
    assert "-" in abc
