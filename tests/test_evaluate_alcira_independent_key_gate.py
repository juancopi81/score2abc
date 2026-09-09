from __future__ import annotations

from scripts.experiments import evaluate_alcira_independent_key_gate as evaluator
from scripts.experiments import evaluate_independent_key_state_gates as paired


def test_alcira_mapping_is_six_one_to_one_physical_measures() -> None:
    payload = paired._mapping_payload(evaluator.CASE)

    assert payload["automatic_crops"] == [
        {
            "automatic_crop_index": index,
            "physical_measure_spans": [
                {
                    "measure_number": index,
                    "note_start": 0,
                    "note_end": note_count,
                }
            ],
        }
        for index, note_count in enumerate((6, 6, 6, 3, 6, 6), start=1)
    ]


def test_alcira_gate_passes_only_for_strict_matching_improvement() -> None:
    result = _result(delta=5)
    context = {"strict_detector_fifths": 2, "automatic_lane_kind": "strict_automatic_key"}

    decision = evaluator._gate_decision(result, context)

    assert decision["status"] == "passed"
    assert decision["promotion_scope"] == "strict_initial_or_system_entry_key_state"
    assert decision["internal_change_scope"] == "not_evaluated_by_this_gate"


def test_alcira_gate_fails_on_regression_or_localization_change() -> None:
    context = {"strict_detector_fifths": 2, "automatic_lane_kind": "strict_automatic_key"}

    assert evaluator._gate_decision(_result(delta=0), context)["status"] == "failed"
    changed = _result(delta=5)
    changed["selection_invariance"]["passed"] = False
    assert evaluator._gate_decision(changed, context)["status"] == "failed"


def test_alcira_gate_fails_when_detector_does_not_match_truth() -> None:
    result = _result(delta=5)
    context = {"strict_detector_fifths": -1, "automatic_lane_kind": "strict_automatic_key"}

    decision = evaluator._gate_decision(result, context)

    assert decision["status"] == "failed"
    assert decision["strict_key_matches_truth"] is False


def _result(*, delta: int) -> dict:
    return {
        "source_musicxml_context": {"key_fifths": 2},
        "selection_invariance": {"passed": True},
        "comparison": {"exact_pitch_match_delta": delta},
    }
