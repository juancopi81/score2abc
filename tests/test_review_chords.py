"""Explicit supplied-harmony upgrades preserve only safe accidental corrections."""

import json

import pytest

from score2abc.abc import events_to_abc
from score2abc.review import ReviewApp, ReviewError, _hash
from score2abc.review_chords import strip_chords, transfer_accidentals


@pytest.fixture
def legacy(tmp_path, monkeypatch):
    metadata = {"title": "Sample", "composer": "Composer", "rhythm": "Pasillo", "key_hint": "C"}
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps({"slug": "sample", "pdf_path": "source.pdf", "metadata": metadata})
    )
    app = ReviewApp(tmp_path)
    for folder in ("final", "stages", "intermediate", "overrides"):
        app.path("sample", folder).mkdir(parents=True)
    app.path("sample", "stages/extract_melody.json").write_text('{"status":"success"}')
    app.path("sample", "stages/extract_musicxml.json").write_text(
        '{"status":"success","params":{"backend":"fixture"}}'
    )
    events = {
        "time_signature": "3/4",
        "notes": [
            {"measure": 1, "onset_beats": 0, "duration_beats": 1.5, "pitch_midi": 72},
            {"measure": 1, "onset_beats": 1.5, "duration_beats": 1.5, "pitch_midi": 74},
        ],
        "chords": [{"measure": 1, "onset_beats": 0.0, "symbol": "C"}],
    }
    app.path("sample", "intermediate/events.json").write_text(json.dumps(events))
    app.path("sample", "intermediate/chords.json").write_text(
        json.dumps({"chords": events["chords"]})
    )
    raw = events_to_abc(events, app.works["sample"].metadata)
    app.path("sample", "final/melody_with_chords.abc").write_text(raw)
    app.path("sample", "intermediate/musicxml.xml").write_text("""<score-partwise version="4.0">
    <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
    <part id="P1"><measure number="1"><attributes><divisions>2</divisions>
    <time><beats>3</beats><beat-type>4</beat-type></time></attributes>
    <harmony><root><root-step>G</root-step></root><kind>major</kind></harmony>
    <note><pitch><step>C</step><octave>5</octave></pitch><duration>3</duration></note>
    <harmony><root><root-step>D</root-step></root><kind>minor</kind></harmony>
    <note><pitch><step>D</step><octave>5</octave></pitch><duration>3</duration></note>
    </measure></part></score-partwise>""")
    old = {
        "version": 1,
        "revision": 3,
        "abc": raw.replace('"C"c', '"C"^c'),
        "base_abc": raw,
        "base_abc_sha256": _hash(raw),
        "review_state": "reviewed",
        "unresolved": ["Check ending"],
        "active_review_ms": 12345,
    }
    app.path("sample", "overrides/review.json").write_text(json.dumps(old, indent=3))
    monkeypatch.setattr(app, "validate", lambda abc: {"valid": True, "note_count": 2, "errors": []})
    return app


def test_explicit_refresh_preserves_accidental_and_exact_backup(legacy):
    target = legacy.path("sample", "overrides/review.json")
    previous = target.read_bytes()
    canonical = legacy.path("sample", "final/melody_with_chords.abc").read_bytes()
    result = legacy.refresh_supplied_chords("sample", 3)
    assert result["revision"] == 4 and result["review_state"] == "draft"
    assert '"G"^c' in result["abc"] and '"Dm"d' in result["abc"]
    assert result["chord_source"] == "supplied_musicxml"
    assert result["unresolved"] == ["Check ending"] and result["active_review_ms"] == 12345
    assert (
        legacy.path("sample", "overrides/review.before_supplied_chords.json").read_bytes()
        == previous
    )
    assert legacy.path("sample", "final/melody_with_chords.abc").read_bytes() == canonical
    saved = json.loads(target.read_text())
    assert saved["base_abc_sha256"] == _hash(saved["base_abc"])
    assert not result["base_changed"]
    with pytest.raises(ReviewError, match="Stale"):
        legacy.refresh_supplied_chords("sample", 3)
    with pytest.raises(ReviewError, match="backup already exists"):
        legacy.refresh_supplied_chords("sample", 4)
    assert target.read_text() == json.dumps(saved, ensure_ascii=False, indent=2)
    reopened = ReviewApp(legacy.out_dir)
    assert reopened.work("sample")["chord_source"] == "supplied_musicxml"


def test_reads_keep_saved_legacy_chords_and_provenance(legacy):
    paths = {p: p.read_bytes() for p in legacy.out_dir.rglob("*") if p.is_file()}
    result = legacy.work("sample")
    assert '"C"^c' in result["abc"] and result["chord_source"] == "automatic_ocr"
    assert result["base_changed"]
    assert {p: p.read_bytes() for p in legacy.out_dir.rglob("*") if p.is_file()} == paths
    legacy.save("sample", {"revision": 3, "abc": result["abc"], "review_state": "draft"})
    assert legacy.work("sample")["chord_source"] == "automatic_ocr"


def test_unsaved_work_uses_restored_base_without_writing(legacy):
    legacy.path("sample", "overrides/review.json").unlink()
    result = legacy.work("sample")
    assert '"G"c' in result["abc"] and '"Dm"d' in result["abc"]
    assert result["chord_source"] == "supplied_musicxml"
    assert not legacy.path("sample", "overrides/review.json").exists()


def test_saved_chord_edit_conflicts_without_backup_or_write(legacy):
    target = legacy.path("sample", "overrides/review.json")
    saved = json.loads(target.read_text())
    saved["abc"] = saved["abc"].replace('"C"', '"Am"')
    target.write_text(json.dumps(saved))
    before = target.read_bytes()
    with pytest.raises(ReviewError, match="conflicts"):
        legacy.refresh_supplied_chords("sample", 3)
    assert target.read_bytes() == before
    assert not legacy.path("sample", "overrides/review.before_supplied_chords.json").exists()


def test_resegmentation_and_unreproduced_events_fail_closed(legacy):
    target = legacy.path("sample", "intermediate/events.json")
    events = json.loads(target.read_text())
    events["notes"][0]["duration_beats"] = 1
    target.write_text(json.dumps(events))
    with pytest.raises(ReviewError, match="do not reproduce"):
        legacy.refresh_supplied_chords("sample", 3)
    assert legacy.work("sample")["chord_source"] == "automatic_ocr"
    with pytest.raises(ValueError, match="structure"):
        transfer_accidentals("X:1\nK:C\nc2|\n", "X:1\nK:C\n^c2|\n", 'X:1\nK:C\n"G"c- c|\n')


def test_provenance_never_inferred_from_xml_alone(legacy):
    legacy.path("sample", "intermediate/chords.json").write_text("{}")
    assert legacy.work("sample")["chord_source"] == "unknown"
    saved = json.loads(legacy.path("sample", "overrides/review.json").read_text())
    saved["chord_source"] = "recognized_musicxml"
    legacy.path("sample", "overrides/review.json").write_text(json.dumps(saved))
    assert legacy.work("sample")["chord_source"] == "recognized_musicxml"


def test_accidental_transfer_is_right_biased_and_rejects_other_edits():
    old = "X:1\nK:C\nc d|\n"
    new = 'X:1\nK:C\n"G"c "Dm"d|\n'
    assert transfer_accidentals(old, old.replace("c d", "^c _d"), new) == new.replace(
        'c "', '^c "'
    ).replace('"d', '"_d')
    assert strip_chords(old) == strip_chords(new)
    for edited in (
        old.replace("K:C", "K:G"),
        old.replace("c d", "d d"),
        old.replace("c d", "c2 d"),
    ):
        with pytest.raises(ValueError):
            transfer_accidentals(old, edited, new)


def test_xml_chord_onset_splitting_note_is_rejected(legacy):
    xml = legacy.path("sample", "intermediate/musicxml.xml")
    xml.write_text(
        xml.read_text().replace(
            "<kind>minor</kind></harmony>", "<kind>minor</kind><offset>-1</offset></harmony>"
        )
    )
    target = legacy.path("sample", "overrides/review.json")
    before = target.read_bytes()
    with pytest.raises(ReviewError, match="structure"):
        legacy.refresh_supplied_chords("sample", 3)
    assert target.read_bytes() == before
    assert not legacy.path("sample", "overrides/review.before_supplied_chords.json").exists()


def test_refresh_validation_failure_leaves_no_backup(legacy, monkeypatch):
    monkeypatch.setattr(legacy, "validate", lambda abc: {"valid": False, "note_count": 0})
    target = legacy.path("sample", "overrides/review.json")
    before = target.read_bytes()
    with pytest.raises(ReviewError, match="did not validate"):
        legacy.refresh_supplied_chords("sample", 3)
    assert target.read_bytes() == before
    assert not legacy.path("sample", "overrides/review.before_supplied_chords.json").exists()


@pytest.mark.parametrize(
    "old,new", [("<step>C</step>", "<step>E</step>"), ("<beats>3</beats>", "<beats>4</beats>")]
)
def test_changed_xml_melody_or_meter_cannot_supply_harmonies(legacy, old, new):
    xml = legacy.path("sample", "intermediate/musicxml.xml")
    xml.write_text(xml.read_text().replace(old, new))
    target = legacy.path("sample", "overrides/review.json")
    before = target.read_bytes()
    with pytest.raises(ReviewError, match="melody does not match"):
        legacy.refresh_supplied_chords("sample", 3)
    assert target.read_bytes() == before
    assert not legacy.path("sample", "overrides/review.before_supplied_chords.json").exists()
    target.unlink()
    result = legacy.work("sample")
    assert '"C"c' in result["abc"] and '"Dm"' not in result["abc"]
    assert result["chord_source"] == "automatic_ocr"
