from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.experiments import strict_initial_key_context as key_context

FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "vlm_melody" / "strict_initial_key_context_cases.json"
)


@pytest.mark.parametrize(
    "case",
    json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["cases"],
    ids=lambda case: case["case_id"],
)
def test_strict_initial_key_context_regression_fixture(case: dict) -> None:
    if "prediction" in case:
        state = key_context.strict_initial_key_state(
            case["prediction"],
            source_system_index=case["source_system_index"],
        )
    else:
        state = key_context.unknown_key_state(
            source_system_index=case["source_system_index"],
            reasons=[f"detector_rejected_image:{case['detector_error']}"],
        )

    assert state["status"] == case["expected_status"]
    assert state["fifths"] == case["expected_fifths"]


def test_strict_initial_key_state_accepts_two_sharps() -> None:
    state = key_context.strict_initial_key_state(
        _prediction(fifths=2, selected_ids=["g1", "g2"], right_px=328),
        source_system_index=6,
    )

    assert state["status"] == key_context.STATUS_CONFIRMED
    assert state["fifths"] == 2
    assert state["applies_after_system_x_px"] == 328


def test_strict_initial_key_state_fails_closed_for_change_or_unknown() -> None:
    changed = _prediction(fifths=-1, selected_ids=["g1"], right_px=200)
    changed["mode"] = "change"
    unknown = _prediction(fifths=None, selected_ids=[], right_px=200)
    unknown["accidental_count"] = 0

    changed_state = key_context.strict_initial_key_state(changed, source_system_index=2)
    unknown_state = key_context.strict_initial_key_state(unknown, source_system_index=2)

    assert changed_state["status"] == key_context.STATUS_UNKNOWN
    assert "detector_mode_is_not_initial" in changed_state["rejection_reasons"]
    assert unknown_state["status"] == key_context.STATUS_UNKNOWN
    assert unknown_state["fifths"] is None


def test_strict_initial_key_state_rejects_count_mismatch_and_truth_access() -> None:
    prediction = _prediction(fifths=2, selected_ids=["g1"], right_px=328)
    prediction["truth_used_for_prediction"] = True

    state = key_context.strict_initial_key_state(prediction, source_system_index=1)

    assert state["status"] == key_context.STATUS_UNKNOWN
    assert "prediction_is_not_truth_blind" in state["rejection_reasons"]
    assert "selected_glyph_count_does_not_match_fifths" in state["rejection_reasons"]


def test_candidate_key_context_applies_only_after_system_boundary() -> None:
    state = key_context.strict_initial_key_state(
        _prediction(fifths=2, selected_ids=["g1", "g2"], right_px=328),
        source_system_index=6,
    )
    request = {
        "allowed_context": {"key_hint": "legacy", "visual_key_state": state},
        "prepared_provenance": {"bbox_px": [166, 0, 578, 200]},
    }

    assert key_context.key_hint_for_candidate(request, candidate_x_px=33) is None
    assert key_context.key_hint_for_candidate(request, candidate_x_px=208) == "2 sharp(s): F#, C#"


def test_candidate_key_context_preserves_legacy_key_hint_without_visual_state() -> None:
    request = {"allowed_context": {"key_hint": "1 flat(s): Bb"}}

    assert key_context.key_hint_for_candidate(request, candidate_x_px=10) == "1 flat(s): Bb"


def test_candidate_key_context_fails_closed_without_crop_provenance() -> None:
    state = key_context.strict_initial_key_state(
        _prediction(fifths=-1, selected_ids=["g1"], right_px=200),
        source_system_index=1,
    )
    request = {"allowed_context": {"visual_key_state": state}}

    assert key_context.key_hint_for_candidate(request, candidate_x_px=250) is None


def test_unknown_detection_inherits_previous_supported_state() -> None:
    state = key_context.unknown_key_state(
        source_system_index=7,
        reasons=["detector_structural_gate_failed"],
        inherited_fifths=-1,
    )
    request = {"allowed_context": {"visual_key_state": state}}

    assert state["status"] == key_context.STATUS_INHERITED
    assert key_context.key_hint_for_candidate(request, candidate_x_px=10) == "1 flat(s): Bb"


def test_unsupported_signature_count_fails_closed() -> None:
    state = key_context.strict_initial_key_state(
        _prediction(
            fifths=-4,
            selected_ids=["g1", "g2", "g3", "g4"],
            right_px=328,
        ),
        source_system_index=3,
    )

    assert state["status"] == key_context.STATUS_UNKNOWN
    assert "detector_fifths_is_not_independently_supported" in state["rejection_reasons"]


def _prediction(*, fifths: int | None, selected_ids: list[str], right_px: int) -> dict:
    return {
        "mode": "initial",
        "fifths": fifths,
        "gate_passed": True,
        "truth_used_for_prediction": False,
        "selection_method": "standard_position_sequence",
        "predicted_signature_family": "flat" if fifths is not None and fifths < 0 else "sharp",
        "accidental_count": len(selected_ids),
        "selected_glyph_ids": selected_ids,
        "search_region": {"left_px": 100, "right_px": right_px},
        "input": {"path": "system.png", "sha256": "abc"},
    }
