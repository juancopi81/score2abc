from __future__ import annotations

import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from score2abc.melody import (
    MELODY_NOTE_FIELDS,
    extract_melody_events,
    write_melody_events,
)
from score2abc.pipeline import (
    _build_events_from_melody,
    _find_musicxml_source,
)
from score2abc.schemas import WorkMetadata

_TINY_MUSICXML = """\
<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1"><part-name>Music</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>2</divisions>
        <time><beats>3</beats><beat-type>4</beat-type></time>
      </attributes>
      <harmony>
        <root><root-step>D</root-step></root>
        <kind>major</kind>
      </harmony>
      <note>
        <pitch><step>E</step><octave>4</octave></pitch>
        <duration>2</duration>
      </note>
      <note>
        <pitch><step>F</step><alter>1</alter><octave>4</octave></pitch>
        <duration>2</duration>
      </note>
      <note>
        <rest/>
        <duration>2</duration>
      </note>
    </measure>
    <measure number="2">
      <note>
        <pitch><step>G</step><octave>4</octave></pitch>
        <duration>4</duration>
      </note>
      <note>
        <pitch><step>A</step><octave>4</octave></pitch>
        <duration>2</duration>
      </note>
    </measure>
  </part>
</score-partwise>
"""


def _write_musicxml(path: Path, contents: str) -> None:
    path.write_text(textwrap.dedent(contents).lstrip(), encoding="utf-8")


def test_extract_melody_events_returns_normalized_notes(tmp_path: Path) -> None:
    xml_path = tmp_path / "tiny.musicxml"
    _write_musicxml(xml_path, _TINY_MUSICXML)

    payload = extract_melody_events(xml_path)

    assert payload["time_signature"] == "3/4"
    assert payload["notes"] == [
        {"measure": 1, "onset_beats": 0.0, "duration_beats": 1.0, "pitch_midi": 64},
        {"measure": 1, "onset_beats": 1.0, "duration_beats": 1.0, "pitch_midi": 66},
        {"measure": 2, "onset_beats": 0.0, "duration_beats": 2.0, "pitch_midi": 67},
        {"measure": 2, "onset_beats": 2.0, "duration_beats": 1.0, "pitch_midi": 69},
    ]


def test_extract_melody_events_drops_chords_and_extra_fields(tmp_path: Path) -> None:
    xml_path = tmp_path / "tiny.musicxml"
    _write_musicxml(xml_path, _TINY_MUSICXML)

    payload = extract_melody_events(xml_path)

    assert "chords" not in payload
    for note in payload["notes"]:
        assert set(note.keys()) == set(MELODY_NOTE_FIELDS)


def test_write_melody_events_writes_json(tmp_path: Path) -> None:
    xml_path = tmp_path / "tiny.musicxml"
    output_path = tmp_path / "melody.json"
    _write_musicxml(xml_path, _TINY_MUSICXML)

    payload = write_melody_events(xml_path, output_path)

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written == payload
    assert written["notes"][0]["pitch_midi"] == 64


def test_extract_melody_events_handles_score_with_no_notes(tmp_path: Path) -> None:
    xml_path = tmp_path / "empty.musicxml"
    _write_musicxml(
        xml_path,
        """
        <?xml version="1.0" encoding="UTF-8"?>
        <score-partwise version="4.0">
          <part-list>
            <score-part id="P1"><part-name>Music</part-name></score-part>
          </part-list>
          <part id="P1">
            <measure number="1">
              <attributes>
                <divisions>1</divisions>
                <time><beats>4</beats><beat-type>4</beat-type></time>
              </attributes>
              <note>
                <rest/>
                <duration>4</duration>
              </note>
            </measure>
          </part>
        </score-partwise>
        """,
    )

    payload = extract_melody_events(xml_path)

    assert payload["time_signature"] == "4/4"
    assert payload["notes"] == []


def _metadata() -> WorkMetadata:
    return WorkMetadata(
        title="Demo",
        composer="Composer",
        rhythm="Pasillo",
        time_signature="3/4",
        key_hint="Em",
    )


def test_find_musicxml_source_returns_path_when_present(tmp_path: Path) -> None:
    work_dir = tmp_path / "slug"
    intermediate = work_dir / "intermediate"
    intermediate.mkdir(parents=True)
    musicxml_path = intermediate / "musicxml.xml"
    musicxml_path.write_text("<score/>", encoding="utf-8")

    assert _find_musicxml_source(work_dir) == musicxml_path


def test_find_musicxml_source_returns_none_when_absent(tmp_path: Path) -> None:
    work_dir = tmp_path / "slug"
    work_dir.mkdir()

    assert _find_musicxml_source(work_dir) is None


def test_build_events_from_melody_uses_extracted_notes_and_chords() -> None:
    item = SimpleNamespace(metadata=_metadata())
    melody = {
        "time_signature": "3/4",
        "notes": [
            {"measure": 1, "onset_beats": 0.0, "duration_beats": 1.0, "pitch_midi": 64},
        ],
    }
    chords = [{"measure": 1, "onset_beats": 0.0, "symbol": "Em", "confidence": 0.9}]

    events = _build_events_from_melody(item, melody=melody, chords=chords)

    assert events["time_signature"] == "3/4"
    assert events["notes"] == melody["notes"]
    assert events["chords"] == [{"measure": 1, "onset_beats": 0.0, "symbol": "Em"}]


def test_build_events_from_melody_keeps_empty_notes_without_stub_fallback() -> None:
    item = SimpleNamespace(metadata=_metadata())
    melody = {"time_signature": "3/4", "notes": []}

    events = _build_events_from_melody(item, melody=melody, chords=[])

    assert events["notes"] == []
    assert events["chords"] == []


def test_build_events_from_melody_falls_back_to_metadata_time_signature() -> None:
    item = SimpleNamespace(metadata=_metadata())
    melody = {"time_signature": None, "notes": []}

    events = _build_events_from_melody(item, melody=melody, chords=[])

    assert events["time_signature"] == "3/4"


def test_extract_melody_events_raises_on_missing_time_signature(tmp_path: Path) -> None:
    xml_path = tmp_path / "no_time.musicxml"
    _write_musicxml(
        xml_path,
        """
        <?xml version="1.0" encoding="UTF-8"?>
        <score-partwise version="4.0">
          <part-list>
            <score-part id="P1"><part-name>Music</part-name></score-part>
          </part-list>
          <part id="P1">
            <measure number="1">
              <note>
                <pitch><step>C</step><octave>4</octave></pitch>
                <duration>1</duration>
              </note>
            </measure>
          </part>
        </score-partwise>
        """,
    )

    with pytest.raises(ValueError, match="time signature"):
        extract_melody_events(xml_path)
