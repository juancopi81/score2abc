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

CANONICAL_NOTE_OPTIONAL_FIELDS: tuple[str, ...] = ("accidental",)


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
    notes = _project_notes(payload.get("notes") or [])
    return {
        "time_signature": payload["time_signature"],
        "notes": notes,
    }


def extract_canonical_melody_events(musicxml_path: Path) -> Dict[str, Any]:
    """Parse MusicXML into canonical events-ready melody notes.

    `melody.json` intentionally stays at the four-field extraction contract, but
    `events.json` can retain MusicXML spelling metadata needed by ABC rendering.
    """
    payload = parse_musicxml_events(musicxml_path)
    notes = _project_notes(
        payload.get("notes") or [],
        optional_fields=CANONICAL_NOTE_OPTIONAL_FIELDS,
    )
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


def _project_notes(
    raw_notes: List[Dict[str, Any]],
    *,
    optional_fields: tuple[str, ...] = (),
) -> List[Dict[str, Any]]:
    notes: List[Dict[str, Any]] = []
    for raw_note in raw_notes:
        note = {field: raw_note[field] for field in MELODY_NOTE_FIELDS}
        for field in optional_fields:
            if field in raw_note:
                note[field] = raw_note[field]
        notes.append(note)
    return notes
