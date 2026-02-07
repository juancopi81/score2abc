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
    assert '"Em" C D | "B7" E2 |' in abc
