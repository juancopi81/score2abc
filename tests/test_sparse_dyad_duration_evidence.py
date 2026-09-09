from __future__ import annotations

from pathlib import Path

import pytest

from scripts.experiments import sparse_dyad_duration_evidence as evidence
from scripts.experiments import spike_consumed_sparse_dyad_duration_evidence as consumed


def test_accepts_three_beat_shared_stem_dotted_dyad() -> None:
    result = evidence.derive_dotted_half_duration_evidence(
        _accepted_decision(),
        _lane(),
        expected_measure_beats=3.0,
    )

    assert result == {
        "schema_version": 1,
        "kind": "sparse_shared_stem_dotted_half",
        "applied": True,
        "reason": "accepted_visual_dotted_half_dyad",
        "candidate_ids": ["d001", "d002"],
        "onset_group_index": 1,
        "augmentation_dot_pairs": [["d010", "d011"]],
        "duration_beats": 3.0,
        "expected_measure_beats": 3.0,
        "suppress_residual_rest_hypotheses": True,
        "truth_used": False,
    }


def test_rejected_sparse_decision_is_a_noop() -> None:
    result = evidence.derive_dotted_half_duration_evidence(
        {"accepted": False, "reason": "rejected_current_candidate_count"},
        [],
        expected_measure_beats=3.0,
    )

    assert result["applied"] is False
    assert result["reason"] == "sparse_repair_not_accepted"


def test_accepted_pattern_fails_closed_outside_three_beat_meter() -> None:
    result = evidence.derive_dotted_half_duration_evidence(
        _accepted_decision(),
        _lane(),
        expected_measure_beats=4.0,
    )

    assert result["applied"] is False
    assert result["reason"] == "expected_meter_is_not_three_beats"


def test_written_duration_reader_does_not_merge_a_cross_bar_tie(tmp_path: Path) -> None:
    musicxml = tmp_path / "tied.musicxml"
    musicxml.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part id="P1">
    <measure number="1">
      <attributes><divisions>2</divisions></attributes>
      <note>
        <pitch><step>G</step><octave>4</octave></pitch>
        <duration>6</duration><tie type="start"/>
      </note>
      <note>
        <chord/><pitch><step>B</step><octave>4</octave></pitch>
        <duration>6</duration><tie type="start"/>
      </note>
    </measure>
    <measure number="2">
      <note>
        <pitch><step>G</step><octave>4</octave></pitch>
        <duration>2</duration><tie type="stop"/>
      </note>
      <note>
        <chord/><pitch><step>B</step><octave>4</octave></pitch>
        <duration>2</duration><tie type="stop"/>
      </note>
    </measure>
  </part>
</score-partwise>
""",
        encoding="utf-8",
    )

    assert consumed._written_note_durations_by_measure(musicxml) == {
        1: [3.0, 3.0],
        2: [1.0, 1.0],
    }


def test_duration_mapping_rejects_merged_physical_measures() -> None:
    with pytest.raises(ValueError, match="one physical measure per crop"):
        consumed._mapping_by_crop(
            {
                "automatic_crops": [
                    {
                        "automatic_crop_index": 1,
                        "physical_measure_spans": [
                            {"measure_number": 1},
                            {"measure_number": 2},
                        ],
                    }
                ]
            }
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda decision, lane: decision.pop("chosen_pair"), "no chosen_pair"),
        (
            lambda decision, lane: decision["chosen_pair"].update(augmentation_dot_pairs=[]),
            "no augmentation-dot pair",
        ),
        (lambda decision, lane: lane[1].update(candidate_id="different"), "does not match"),
        (lambda decision, lane: lane[1].update(onset_group_index=2), "share one"),
        (lambda decision, lane: decision.update(truth={"duration": 3}), "Forbidden truth"),
    ],
)
def test_malformed_or_truth_bearing_accepted_evidence_is_rejected(mutate, message: str) -> None:
    decision = _accepted_decision()
    lane = _lane()
    mutate(decision, lane)

    with pytest.raises(ValueError, match=message):
        evidence.derive_dotted_half_duration_evidence(
            decision,
            lane,
            expected_measure_beats=3.0,
        )


def _accepted_decision() -> dict:
    return {
        "accepted": True,
        "reason": "accepted",
        "chosen_pair": {
            "candidate_ids": ["d001", "d002"],
            "augmentation_dot_pairs": [{"candidate_ids": ["d010", "d011"]}],
        },
    }


def _lane() -> list[dict]:
    return [
        {"candidate_id": "d001", "onset_group_index": 1},
        {"candidate_id": "d002", "onset_group_index": 1},
    ]
