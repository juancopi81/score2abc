from score2abc.metrics import compare_events


def _events(
    *,
    chords: list[dict] | None = None,
    notes: list[dict] | None = None,
    time_signature: str = "3/4",
) -> dict:
    return {
        "time_signature": time_signature,
        "notes": notes or [],
        "chords": chords or [],
    }


def test_measure_only_chord_metric_ignores_onset_mismatch() -> None:
    pred = _events(
        chords=[
            {"measure": 1, "onset_beats": 2.0, "symbol": "Em"},
            {"measure": 2, "onset_beats": 1.5, "symbol": "B7"},
        ]
    )
    truth = _events(
        chords=[
            {"measure": 1, "onset_beats": 0.0, "symbol": "Em"},
            {"measure": 2, "onset_beats": 0.0, "symbol": "B7"},
        ]
    )

    result = compare_events(pred, truth)

    # Strict metric penalizes the onset mismatch.
    assert result["chord_f1"] == 0.0
    # Measure-only metric matches both chords.
    assert result["chord_precision_measure_only"] == 1.0
    assert result["chord_recall_measure_only"] == 1.0
    assert result["chord_f1_measure_only"] == 1.0
    assert result["chord_true_positives_measure_only"] == 2
    assert result["chord_false_positives_measure_only"] == 0
    assert result["chord_false_negatives_measure_only"] == 0


def test_measure_only_chord_metric_penalizes_wrong_measure() -> None:
    pred = _events(chords=[{"measure": 1, "onset_beats": 0.0, "symbol": "Em"}])
    truth = _events(chords=[{"measure": 2, "onset_beats": 0.0, "symbol": "Em"}])

    result = compare_events(pred, truth)

    assert result["chord_f1_measure_only"] == 0.0
    assert result["chord_false_positives_measure_only"] == 1
    assert result["chord_false_negatives_measure_only"] == 1


def test_measure_only_chord_metric_partial_overlap() -> None:
    pred = _events(
        chords=[
            {"measure": 1, "onset_beats": 0.0, "symbol": "C"},
            {"measure": 2, "onset_beats": 0.0, "symbol": "G"},
            {"measure": 3, "onset_beats": 0.0, "symbol": "F"},
        ]
    )
    truth = _events(
        chords=[
            {"measure": 1, "onset_beats": 0.0, "symbol": "C"},
            {"measure": 2, "onset_beats": 0.0, "symbol": "G"},
            {"measure": 3, "onset_beats": 0.0, "symbol": "Am"},
        ]
    )

    result = compare_events(pred, truth)

    assert result["chord_true_positives_measure_only"] == 2
    assert result["chord_false_positives_measure_only"] == 1
    assert result["chord_false_negatives_measure_only"] == 1
    assert result["chord_precision_measure_only"] == round(2 / 3, 6)
    assert result["chord_recall_measure_only"] == round(2 / 3, 6)


def test_measure_only_chord_metric_empty_inputs() -> None:
    result = compare_events(_events(), _events())
    assert result["chord_precision_measure_only"] == 0.0
    assert result["chord_recall_measure_only"] == 0.0
    assert result["chord_f1_measure_only"] == 0.0
