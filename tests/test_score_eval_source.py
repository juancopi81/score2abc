"""Source metrics must not turn editorial editions into blanket ground truth."""

from copy import deepcopy

import pytest

from score2abc.score_eval_source import NOTE_FIELDS, note_events_sha256, score_source


def _note(pitch=60, *, duration=384, onset=0):
    return {"pitch_midi": pitch, "duration_ticks": duration, "onset_ticks": onset}


def _score(*bars):
    return {
        "valid": True,
        "ticks_per_quarter": 384,
        "measures": [{"number": i, "notes": notes} for i, notes in enumerate(bars, 1)],
    }


def _labels(reference, numbers=(1,)):
    return {
        "source_audit": {
            "ticks_per_quarter": 384,
            "total_source_measures": len(reference["measures"]),
            "measures": [
                {
                    "source_bar": n,
                    "edition_bar": n,
                    "source_location": {"page": 1, "system": 1, "measure": n},
                    "dimensions": ["notes"],
                    "note_fields": list(NOTE_FIELDS),
                    "note_events_sha256": note_events_sha256(reference["measures"][n - 1]),
                    "expected_noteheads": len(reference["measures"][n - 1]["notes"]),
                }
                for n in numbers
            ],
        }
    }


def _alignment(*pairs):
    return [{"reference": r, "candidate": p} for r, p in pairs]


def test_only_audited_notes_score_and_editorial_changes_stay_out():
    reference = _score([_note()], [_note(62)])
    candidate = _score([_note()], [_note(90), _note(91)])
    result = score_source(reference, candidate, _alignment((1, 1), (2, 2)), _labels(reference))
    assert result["metrics"]["source_note_f1"] == 1
    assert result["coverage"]["source_scored_measures"] == 1
    assert result["coverage"]["source_total_measures"] == 2
    assert result["metric_status"]["source_note_f1"] == "model_audited_source_subset"


def test_missing_audited_bar_counts_as_false_negative():
    reference = _score([_note()], [_note(62)])
    candidate = _score([_note(62)])
    result = score_source(reference, candidate, _alignment((1, None), (2, 1)), _labels(reference))
    assert result["metrics"]["source_note_f1"] == 0
    assert result["coverage"]["source_expected_noteheads"] == 1
    assert result["coverage"]["source_matched_measures"] == 0
    assert result["mismatches"][0]["source_bar"] == 1


def test_added_head_and_duplicate_multiplicity_lower_precision():
    reference = _score([_note(), _note()])
    candidate = _score([_note(), _note(), _note()])
    result = score_source(reference, candidate, _alignment((1, 1)), _labels(reference))
    assert result["metrics"]["source_note_f1"] == 0.8
    assert result["coverage"]["source_matched_noteheads"] == 2


@pytest.mark.parametrize(
    "field,value",
    [("pitch_midi", 61), ("duration_ticks", 192), ("onset_ticks", 1), ("tie_start", True)],
)
def test_each_audited_field_is_scored(field, value):
    reference = _score([_note()])
    candidate = deepcopy(reference)
    candidate["measures"][0]["notes"][0][field] = value
    result = score_source(reference, candidate, _alignment((1, 1)), _labels(reference))
    assert result["metrics"]["source_note_f1"] == 0


def test_voice_spelling_order_are_not_note_identity():
    reference = _score([_note(60), _note(64)])
    candidate = _score([dict(_note(64), voice="b"), dict(_note(60), spelling="B#")])
    assert (
        score_source(reference, candidate, _alignment((1, 1)), _labels(reference))["metrics"][
            "source_note_f1"
        ]
        == 1
    )


def test_unknown_regions_and_invalid_candidates_are_not_zero_accuracy():
    reference = _score([_note()])
    assert score_source(reference, reference, [], {})["metrics"]["source_note_f1"] is None
    result = score_source(reference, {"valid": False}, [], _labels(reference))
    assert result["metrics"]["source_note_f1"] is None
    assert result["metric_status"]["source_note_f1"] == "invalid_candidate"
    assert result["coverage"]["source_eligible_measures"] == 1


def test_changed_reference_or_normalizer_requires_reauditing():
    reference = _score([_note()])
    labels = _labels(reference)
    reference["measures"][0]["notes"][0]["pitch_midi"] = 61
    with pytest.raises(ValueError, match="Audited source-note events"):
        score_source(reference, reference, _alignment((1, 1)), labels)


@pytest.mark.parametrize("alignment", [_alignment((1, 1), (1, 1)), [], _alignment((1, 99))])
def test_incomplete_duplicate_or_unknown_alignment_rejected(alignment):
    reference = _score([_note()])
    with pytest.raises(ValueError, match="alignment"):
        score_source(reference, reference, alignment, _labels(reference))


def test_unlocated_extra_bar_stays_out_of_subset_metric():
    reference = _score([_note()])
    candidate = _score([_note()], [_note(99)])
    result = score_source(reference, candidate, _alignment((1, 1), (None, 2)), _labels(reference))
    assert result["metrics"]["source_note_f1"] == 1
    assert result["coverage"]["source_predicted_noteheads"] == 1
