from __future__ import annotations

from scripts.experiments import evaluate_chispazo_internal_key_gate as chispazo_evaluator
from scripts.experiments import evaluate_frozen_third_score_heldout as heldout
from scripts.experiments import evaluate_independent_key_state_gates as evaluator
from scripts.experiments import freeze_independent_key_state_gates as freezer


def test_shared_verifier_registers_independent_key_gates() -> None:
    sealed_kinds = {
        case.gate.sealed_kind
        for case in (*freezer.CASES.values(), *freezer.CHALLENGE_CASES.values())
    }

    assert sealed_kinds <= set(heldout.EVALUATION_SPECS)


def test_chispazo_mapping_preserves_missed_key_change_barline() -> None:
    payload = evaluator._mapping_payload(evaluator.CHISPAZO_INTERNAL_CHANGE_CASE)

    assert len(payload["automatic_crops"]) == 8
    assert payload["automatic_crops"][4] == {
        "automatic_crop_index": 5,
        "physical_measure_spans": [
            {"measure_number": 5, "note_start": 0, "note_end": 2},
            {"measure_number": 6, "note_start": 0, "note_end": 1},
        ],
    }


def test_chispazo_diagnostic_can_never_promote_runtime() -> None:
    decision = chispazo_evaluator._diagnostic_decision({"exact_pitch_match_delta": 4})

    assert decision["status"] == "diagnostic_supported"
    assert decision["promotable"] is False
    assert decision["runtime_action"] == "keep automatic internal key changes out of runtime"


def test_post_transcription_mappings_record_both_false_splits() -> None:
    estrella, sobre = evaluator.CASES

    assert evaluator._mapping_payload(estrella)["automatic_crops"][3:] == [
        {
            "automatic_crop_index": 4,
            "physical_measure_spans": [{"measure_number": 4, "note_start": 0, "note_end": 5}],
        },
        {
            "automatic_crop_index": 5,
            "physical_measure_spans": [{"measure_number": 4, "note_start": 5, "note_end": 7}],
        },
        {
            "automatic_crop_index": 6,
            "physical_measure_spans": [{"measure_number": 5, "note_start": 0, "note_end": 7}],
        },
    ]
    assert evaluator._mapping_payload(sobre)["automatic_crops"][:2] == [
        {
            "automatic_crop_index": 1,
            "physical_measure_spans": [{"measure_number": 1, "note_start": 0, "note_end": 2}],
        },
        {
            "automatic_crop_index": 2,
            "physical_measure_spans": [{"measure_number": 1, "note_start": 2, "note_end": 6}],
        },
    ]


def test_promotion_requires_improvement_on_every_score() -> None:
    results = [
        _case_result("improved", baseline=10, automatic=12),
        _case_result("regressed", baseline=10, automatic=3),
    ]

    decision = evaluator._promotion_decision(results)

    assert decision["status"] == "not_promoted"
    assert decision["improved_cases"] == ["improved"]
    assert decision["regressed_cases"] == ["regressed"]
    assert decision["runtime_action"] == "keep automatic key state out of runtime"


def test_promotion_passes_only_with_invariant_localization() -> None:
    results = [
        _case_result("first", baseline=10, automatic=11),
        _case_result("second", baseline=10, automatic=12),
    ]

    assert evaluator._promotion_decision(results)["status"] == "promoted"
    results[1]["selection_invariance"]["passed"] = False
    assert evaluator._promotion_decision(results)["status"] == "not_promoted"


def _case_result(case_id: str, *, baseline: int, automatic: int) -> dict:
    return {
        "case_id": case_id,
        "selection_invariance": {"passed": True},
        "comparison": {
            "score_improved": automatic > baseline,
            "score_regressed": automatic < baseline,
            "baseline_exact_pitch_matches": baseline,
            "automatic_exact_pitch_matches": automatic,
        },
    }
