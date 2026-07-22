from __future__ import annotations

from scripts.experiments import spike_consumed_meter_deficit_validator as validator


def test_meter_context_prefers_request_value() -> None:
    context = validator.meter_context(
        {
            "allowed_context": {
                "expected_measure_beats": 2.0,
                "allow_pickup": True,
            }
        },
        metadata={"rhythm": "Pasillo"},
    )

    assert context == validator.MeterContext(2.0, True, "request.allowed_context")


def test_meter_context_uses_explicit_pasillo_prior() -> None:
    context = validator.meter_context(
        {"allowed_context": {"expected_measure_beats": None, "allow_pickup": False}},
        metadata={"rhythm": "Pasillo"},
    )

    assert context == validator.MeterContext(
        3.0,
        False,
        "metadata.rhythm_prior:pasillo",
    )


def test_meter_deficit_flags_only_non_pickup_underfill() -> None:
    regular = validator.MeterContext(3.0, False, "test")
    pickup = validator.MeterContext(3.0, True, "test")

    assert validator.decide_meter_deficit(2.5, regular) == (
        True,
        "review_visual_meter_deficit",
        0.5,
    )
    assert validator.decide_meter_deficit(3.0, regular) == (
        False,
        "meter_not_underfilled",
        0.0,
    )
    assert validator.decide_meter_deficit(2.5, pickup) == (
        False,
        "pickup_exempt",
        0.5,
    )


def test_validation_metrics_treat_flags_as_review_triage() -> None:
    metrics = validator.validation_metrics(
        [
            {"review_flag": True, "has_onset_error": True},
            {"review_flag": False, "has_onset_error": True},
            {"review_flag": False, "has_onset_error": False},
            {"review_flag": True, "has_onset_error": False},
        ]
    )

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["review_load"] == 0.5


def test_truth_counts_group_polyphonic_notes_by_onset() -> None:
    counts = validator._truth_counts(
        [
            {
                "identity": {"automatic_measure_index": 1},
                "notes": [
                    {"onset_divisions": 0},
                    {"onset_divisions": 0},
                    {"onset_divisions": 2},
                ],
            }
        ]
    )

    assert counts == {1: 2}
