from __future__ import annotations

import json
import textwrap
from pathlib import Path

from score2abc.musicxml import parse_musicxml_events, write_musicxml_ground_truth


def _write_musicxml(path: Path, contents: str) -> None:
    path.write_text(textwrap.dedent(contents).lstrip(), encoding="utf-8")


def test_parse_musicxml_events_extracts_notes_chords_and_time_signature(tmp_path: Path) -> None:
    xml_path = tmp_path / "demo.musicxml"
    _write_musicxml(
        xml_path,
        """
        <?xml version="1.0" encoding="UTF-8"?>
        <score-partwise version="4.0">
          <part-list>
            <score-part id="P1"><part-name>Music</part-name></score-part>
          </part-list>
          <part id="P1">
            <measure number="0" implicit="yes">
              <attributes>
                <divisions>2</divisions>
                <time><beats>3</beats><beat-type>4</beat-type></time>
              </attributes>
              <note>
                <pitch><step>A</step><octave>4</octave></pitch>
                <duration>1</duration>
              </note>
              <note>
                <rest/>
                <duration>1</duration>
              </note>
              <harmony>
                <root><root-step>D</root-step></root>
                <kind>major</kind>
              </harmony>
              <note>
                <pitch><step>C</step><alter>1</alter><octave>5</octave></pitch>
                <duration>2</duration>
              </note>
              <note>
                <chord/>
                <pitch><step>E</step><octave>5</octave></pitch>
                <duration>2</duration>
              </note>
            </measure>
          </part>
        </score-partwise>
        """,
    )

    payload = parse_musicxml_events(xml_path)

    assert payload["time_signature"] == "3/4"
    assert payload["chords"] == [{"measure": 0, "onset_beats": 1.0, "symbol": "D"}]
    assert payload["notes"] == [
        {"measure": 0, "onset_beats": 0.0, "duration_beats": 0.5, "pitch_midi": 69},
        {
            "measure": 0,
            "onset_beats": 1.0,
            "duration_beats": 1.0,
            "pitch_midi": 73,
            "accidental": 1,
        },
        {"measure": 0, "onset_beats": 1.0, "duration_beats": 1.0, "pitch_midi": 76},
    ]


def test_parse_musicxml_events_merges_ties_and_uses_kind_text(tmp_path: Path) -> None:
    xml_path = tmp_path / "tied.musicxml"
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
              <harmony>
                <root><root-step>G</root-step></root>
                <kind text="m6">minor-sixth</kind>
              </harmony>
              <note>
                <pitch><step>B</step><alter>-1</alter><octave>4</octave></pitch>
                <duration>2</duration>
                <tie type="start"/>
              </note>
            </measure>
            <measure number="2">
              <note>
                <pitch><step>B</step><alter>-1</alter><octave>4</octave></pitch>
                <duration>1</duration>
                <tie type="stop"/>
              </note>
            </measure>
          </part>
        </score-partwise>
        """,
    )

    payload = parse_musicxml_events(xml_path)

    assert payload["chords"] == [{"measure": 1, "onset_beats": 0.0, "symbol": "Gm6"}]
    assert payload["notes"] == [
        {
            "measure": 1,
            "onset_beats": 0.0,
            "duration_beats": 3.0,
            "pitch_midi": 70,
            "accidental": -1,
            "tie": True,
        }
    ]


def test_write_musicxml_ground_truth_writes_json(tmp_path: Path) -> None:
    xml_path = tmp_path / "demo.musicxml"
    output_path = tmp_path / "demo.json"
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
                <time><beats>2</beats><beat-type>4</beat-type></time>
              </attributes>
              <note>
                <pitch><step>C</step><octave>4</octave></pitch>
                <duration>1</duration>
              </note>
            </measure>
          </part>
        </score-partwise>
        """,
    )

    payload = write_musicxml_ground_truth(xml_path, output_path)

    assert payload["time_signature"] == "2/4"
    assert output_path.exists()
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written == payload
