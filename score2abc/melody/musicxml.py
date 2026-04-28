from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from score2abc.musicxml import parse_musicxml_events

MELODY_NOTE_FIELDS: tuple[str, ...] = (
    "measure",
    "onset_beats",
    "duration_beats",
    "pitch_midi",
)


def extract_melody_events(musicxml_path: Path) -> Dict[str, Any]:
    """Parse a MusicXML file into normalized melody-only events.

    Output schema:
        {
            "time_signature": str,
            "notes": [
                {"measure", "onset_beats", "duration_beats", "pitch_midi"}, ...
            ],
        }

    This is the thin Melody Engine A integration slice: chords, accidentals, and
    ties are dropped from the per-note view. Future passes can extend the schema.
    """
    payload = parse_musicxml_events(musicxml_path)
    notes: List[Dict[str, Any]] = [
        {field: raw_note[field] for field in MELODY_NOTE_FIELDS}
        for raw_note in payload.get("notes") or []
    ]
    return {
        "time_signature": payload["time_signature"],
        "notes": notes,
    }


def write_melody_events(musicxml_path: Path, output_path: Path) -> Dict[str, Any]:
    """Parse MusicXML and write the melody-only events JSON next to the score."""
    payload = extract_melody_events(musicxml_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
