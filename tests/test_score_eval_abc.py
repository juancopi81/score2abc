"""Semantic ABC normalization for complete-score evaluation."""

import shutil

import pytest

from score2abc.score_eval_abc import normalize_abc

RENDERER = bool(shutil.which("node") and shutil.which("abc2svg"))
needs_renderer = pytest.mark.skipif(not RENDERER, reason="optional local ABC renderer unavailable")


@needs_renderer
def test_equivalent_layout_and_note_lengths_normalize_identically():
    a = 'X:1\nT:One\nM:3/4\nL:1/4\nK:C\n"C"[CE]2 z | D E F |]\n'
    b = 'X:1\nT:Two\nM:3/4\nL:1/8\nK:C\n% layout\n"C"[CE]4 z2|\nD2 E2 F2|]\n'
    left, right = normalize_abc(a), normalize_abc(b)
    assert left["valid"] and right["valid"]
    assert left["measures"] == right["measures"]
    assert left["route"] == right["route"] == [1, 2]
    assert left["counts"] == right["counts"]


@needs_renderer
def test_written_repeats_and_numbered_endings_have_distinct_played_route():
    abc = "X:1\nM:3/4\nL:1/4\nK:C\n" '|: "C"C3 | [1 "G"D3 :| [2 "Am"E3 |]\n'
    normalized = normalize_abc(abc)
    assert normalized["valid"], normalized["errors"]
    assert normalized["counts"]["measures"] == 3
    assert normalized["route"] == [1, 2, 1, 3]
    assert normalized["route_status"] == "resolved"
    assert [m["chords"][0]["symbol"] for m in normalized["measures"]] == ["C", "G", "Am"]


@needs_renderer
def test_part_order_uses_playback_not_written_section_concatenation():
    abc = "X:1\nP:ABA\nM:3/4\nL:1/4\nK:C\n" 'P:A\n"C"C3 |\nP:B\n"G"D3 |]\n'
    normalized = normalize_abc(abc)
    assert normalized["valid"], normalized["errors"]
    assert normalized["route"] == [1, 2, 1]
    assert normalized["counts"]["performed_measures"] == 3


@needs_renderer
def test_polyphonic_heads_rests_and_meter_change_stay_written():
    abc = "X:1\nM:3/4\nL:1/4\nK:C\n" '[CE]2 z & G3 | [CE]2 z & G3 | [M:2/4] "F"F G |]\n'
    normalized = normalize_abc(abc)
    assert normalized["valid"], normalized["errors"]
    first, second, third = normalized["measures"]
    assert sorted(n["pitch_midi"] for n in first["notes"]) == [60, 64, 67]
    assert {(n["pitch_midi"], n["voice"]) for n in first["notes"]} == {
        (60, "1"),
        (64, "1"),
        (67, "1o"),
    }
    assert first["notes"][0]["tie_start"] is False
    assert second["notes"][0]["onset_ticks"] == 0
    assert first["rests"][0]["onset_ticks"] == 768
    assert first["rests"][0]["duration_ticks"] == 384
    assert third["meter"] == {"numerator": 2, "denominator": 4}
    assert third["expected_ticks"] == 768


@needs_renderer
def test_tied_continuations_remain_separate_written_heads():
    normalized = normalize_abc("X:1\nM:3/4\nL:1/4\nK:C\nC3- | C3 |]\n")
    assert normalized["valid"], normalized["errors"]
    first, second = normalized["measures"]
    assert first["notes"][0]["pitch_midi"] == second["notes"][0]["pitch_midi"] == 60
    assert first["notes"][0]["tie_start"] is True
    assert second["notes"][0]["tie_end"] is True


@needs_renderer
def test_conflicting_explicit_voice_barlines_cannot_be_called_pickup_pair():
    abc = "X:1\nM:3/4\nL:1/4\nK:C\nV:1\nC D E |]\nV:2\nG A |]\n"
    result = normalize_abc(abc)
    assert result["valid"] is False
    assert any("barlines are not synchronized" in error for error in result["errors"])
    assert result["measures"] == []
    assert result["route_status"] == "unavailable"


@needs_renderer
def test_synchronized_explicit_voice_barlines_remain_valid():
    abc = "X:1\nM:3/4\nL:1/4\nK:C\nV:1\nC D E |]\nV:2\nG A B |]\n"
    result = normalize_abc(abc)
    assert result["valid"], result["errors"]
    assert result["counts"]["measures"] == 1
    assert result["measures"][0]["meter_valid"] is True


@needs_renderer
def test_underfull_first_bar_is_candidate_not_asserted_valid_pickup():
    abc = "X:1\nM:3/4\nL:1/4\nK:C\nC | D E F | G3 |]\n"
    normalized = normalize_abc(abc)
    assert normalized["valid"]
    first = normalized["measures"][0]
    assert first["duration_ticks"] == 384
    assert first["expected_ticks"] == 1152
    assert first["partial_reason"] == "pickup_candidate"
    assert first["meter_valid"] is None


@needs_renderer
def test_wrong_pitch_duration_and_chord_change_semantic_result():
    base = 'X:1\nM:3/4\nL:1/4\nK:C\n"C"C D E |]\n'
    changed = 'X:1\nM:3/4\nL:1/4\nK:C\n"G"C D F2 |]\n'
    expected, actual = normalize_abc(base), normalize_abc(changed)
    assert expected["valid"] and actual["valid"]
    assert [n["pitch_midi"] for n in expected["measures"][0]["notes"]] == [60, 62, 64]
    assert [n["pitch_midi"] for n in actual["measures"][0]["notes"]] == [60, 62, 65]
    assert expected["measures"][0]["chords"][0]["symbol"] == "C"
    assert actual["measures"][0]["chords"][0]["symbol"] == "G"
    assert actual["measures"][0]["meter_valid"] is False


@needs_renderer
def test_unsupported_part_order_has_no_invented_route():
    abc = "X:1\nP:AZ\nM:3/4\nL:1/4\nK:C\nP:A\nC3 |]\n"
    result = normalize_abc(abc)
    assert result["valid"]
    assert result["route"] is None
    assert result["route_status"] == "unsupported"
    assert result["counts"]["performed_measures"] is None
    assert any("missing section label" in warning for warning in result["warnings"])


@needs_renderer
@pytest.mark.parametrize(
    "abc",
    [
        "X:1\nM:3/4\nL:1/4\nK:C\n%%include secret.abc\nC3 |]\n",
        "X:1\nM:3/4\nL:1/4\nK:C\n%%beginjs\nC3 |]\n",
        "X:1\nM:3/4\nL:1/4\nK:C\n<script>alert(1)</script>\nC3 |]\n",
    ],
)
def test_executable_notation_and_includes_rejected(abc):
    result = normalize_abc(abc)
    assert not result["valid"]
    assert any("not allowed" in error for error in result["errors"])
    assert result["measures"] == []


def test_renderer_unavailable_is_explicit(monkeypatch):
    monkeypatch.setattr("score2abc.score_eval_abc.shutil.which", lambda _: None)
    result = normalize_abc("X:1\nM:3/4\nK:C\nC3 |]\n")
    assert result["valid"] is False
    assert result["errors"] == ["Local abc2svg and Node renderer unavailable."]
    assert result["route_status"] == "unavailable"
