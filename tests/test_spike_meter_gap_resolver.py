from pathlib import Path

from scripts.experiments import spike_meter_gap_resolver as spike
from scripts.experiments import spike_notehead_patch_templates as patches


def test_recovers_near_threshold_candidate_from_large_leading_gap(tmp_path: Path) -> None:
    leading = _candidate("d001", rank=2, x=30.0, y=60.0, score=0.495)
    selected = _candidate("d002", rank=1, x=80.0, y=40.0, score=0.8)
    measure = _measure(tmp_path, leading, selected)

    recovered = spike._recover_leading_candidate(
        measure,
        [selected],
        {"d001": 0.495, "d002": 0.8},
        threshold=0.5,
        leading_gap_spaces=3.0,
        score_margin=0.01,
        maximum_selected_count=6,
        nms_x_spaces=0.45,
    )

    assert recovered == leading


def test_leading_recovery_is_disabled_without_training_selected_margin(tmp_path: Path) -> None:
    leading = _candidate("d001", rank=2, x=30.0, y=60.0, score=0.495)
    selected = _candidate("d002", rank=1, x=80.0, y=40.0, score=0.8)
    measure = _measure(tmp_path, leading, selected)

    recovered = spike._recover_leading_candidate(
        measure,
        [selected],
        {"d001": 0.495, "d002": 0.8},
        threshold=0.5,
        leading_gap_spaces=3.0,
        score_margin=0.0,
        maximum_selected_count=6,
        nms_x_spaces=0.45,
    )

    assert recovered is None


def test_leading_rest_requires_half_beat_deficit_and_nonpickup_gap() -> None:
    assert spike._should_insert_leading_rest(
        expected_beats=3.0,
        visual_note_extent_beats=2.5,
        first_anchor_gap_spaces=5.0,
        allow_pickup=False,
        has_anchors=True,
    )
    assert not spike._should_insert_leading_rest(
        expected_beats=3.0,
        visual_note_extent_beats=3.0,
        first_anchor_gap_spaces=5.0,
        allow_pickup=False,
        has_anchors=True,
    )
    assert not spike._should_insert_leading_rest(
        expected_beats=3.0,
        visual_note_extent_beats=2.5,
        first_anchor_gap_spaces=5.0,
        allow_pickup=True,
        has_anchors=True,
    )
    assert not spike._should_insert_leading_rest(
        expected_beats=3.0,
        visual_note_extent_beats=2.5,
        first_anchor_gap_spaces=2.9,
        allow_pickup=False,
        has_anchors=True,
    )


def _candidate(
    candidate_id: str,
    *,
    rank: int,
    x: float,
    y: float,
    score: float,
) -> patches.CandidatePatch:
    return patches.CandidatePatch(
        measure=1,
        id=candidate_id,
        rank=rank,
        center_x=x,
        center_y=y,
        bbox=(round(x) - 4, round(y) - 3, round(x) + 5, round(y) + 4),
        detector_score=score,
        patches={},
    )


def _measure(
    tmp_path: Path,
    *candidates: patches.CandidatePatch,
) -> patches.UnlabeledMeasure:
    return patches.UnlabeledMeasure(
        measure=1,
        source_image=tmp_path / "measure.png",
        source_sha256="fixture",
        staff_lines=(20, 40, 60, 80, 100),
        staff_spacing=20.0,
        candidates=tuple(candidates),
    )
