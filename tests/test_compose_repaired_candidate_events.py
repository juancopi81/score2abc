from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from scripts.experiments import compose_repaired_candidate_events as compositor
from scripts.experiments import spike_notehead_patch_templates as patches


def test_preserves_two_note_chord_and_materializes_three_four(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, inference, measure, lane = _fixture(tmp_path)
    seen_key_hints = []

    def predict(candidate, received_request, image):
        assert image.size == (140, 125)
        seen_key_hints.append(received_request["allowed_context"]["key_hint"])
        return {"d001": "C4", "d002": "E4", "d003": "G4"}[candidate.id]

    monkeypatch.setattr(compositor.dense, "_prepare_dense_measure", lambda *args, **kwargs: measure)
    result = compositor.compose_repaired_candidate_events(
        request,
        inference,
        lane,
        pitch_predictor=predict,
        out_dir=tmp_path,
        selector_method_id="repaired-v1",
    )

    assert result["status"] == "materialized"
    assert result["truth_used"] is False
    assert result["meter_valid"] is True
    assert result["expected_measure_beats"] == 3.0
    assert result["decoded_extent_beats"] == 3.0
    assert result["groups"][0]["pitches"] == ["C4", "E4"]
    assert len(result["groups"][0]["anchors"]) == 2
    assert result["notes"][0]["onset_beats"] == result["notes"][1]["onset_beats"]
    assert result["notes"][0]["duration_beats"] == result["notes"][1]["duration_beats"]
    assert len(result["onsets"]) == 2
    assert len(result["durations"]) == 2
    assert seen_key_hints == ["one flat: Bb"] * 3
    assert result["automatic_key_context"]["candidate_key_hints"] == [
        {"candidate_id": "d001", "key_hint": "one flat: Bb"},
        {"candidate_id": "d002", "key_hint": "one flat: Bb"},
        {"candidate_id": "d003", "key_hint": "one flat: Bb"},
    ]
    assert result["prediction"]["inference_provenance"]["truth_used"] is False


def test_missing_meter_fails_closed_after_pitch_and_grouping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, inference, measure, lane = _fixture(tmp_path)
    request["allowed_context"]["expected_measure_beats"] = None
    inference["allowed_context"]["expected_measure_beats"] = None
    monkeypatch.setattr(compositor.dense, "_prepare_dense_measure", lambda *args, **kwargs: measure)

    result = compositor.compose_repaired_candidate_events(
        request,
        inference,
        lane,
        pitch_predictor=lambda candidate, request, image: "C4",
        out_dir=tmp_path,
        selector_method_id="repaired-v1",
    )

    assert result["status"] == "not_materialized_missing_expected_measure_beats"
    assert result["prediction"] is None
    assert result["onsets"] is None
    assert result["durations"] is None
    assert result["rests"] is None
    assert len(result["notes"]) == 3
    assert all(note["onset_beats"] is None for note in result["notes"])
    assert result["anchor_features"] == []
    assert result["inference_provenance"]["rhythm_decoding_applied"] is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda lane: lane[0].update(candidate_id="unknown"), "unknown candidate"),
        (lambda lane: lane[0]["center"].update(x=99.0), "center drift"),
        (lambda lane: lane.append(dict(lane[0])), "duplicate candidate"),
        (lambda lane: lane[0].update(onset_group_index=0), "positive integers"),
        (lambda lane: lane[2].update(onset_group_index=3), "contiguous"),
        (lambda lane: lane.reverse(), "monotonic x"),
    ],
)
def test_rejects_candidate_identity_group_and_order_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate,
    message: str,
) -> None:
    request, inference, measure, lane = _fixture(tmp_path)
    mutate(lane)
    monkeypatch.setattr(compositor.dense, "_prepare_dense_measure", lambda *args, **kwargs: measure)

    with pytest.raises(ValueError, match=message):
        compositor.compose_repaired_candidate_events(
            request,
            inference,
            lane,
            pitch_predictor=lambda candidate, request, image: "C4",
            out_dir=tmp_path,
            selector_method_id="repaired-v1",
        )


@pytest.mark.parametrize(
    "field",
    ["identity", "allowed_context", "allowed_context_provenance", "staff_geometry"],
)
def test_rejects_inference_context_drift(tmp_path: Path, field: str) -> None:
    request, inference, _, lane = _fixture(tmp_path)
    inference[field] = {"drift": True}
    with pytest.raises(ValueError, match=field):
        compositor.compose_repaired_candidate_events(
            request,
            inference,
            lane,
            pitch_predictor=lambda candidate, request, image: "C4",
            out_dir=tmp_path,
            selector_method_id="repaired-v1",
        )


def test_reports_invalid_meter_when_pickup_decoder_exceeds_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, inference, measure, lane = _fixture(tmp_path)
    monkeypatch.setattr(compositor.dense, "_prepare_dense_measure", lambda *args, **kwargs: measure)
    monkeypatch.setattr(
        compositor,
        "_decode_with_meter_fallback",
        lambda *args, **kwargs: (
            [
                {
                    "kind": "note",
                    "x": 20.0,
                    "duration_beats": 4.0,
                    "onset_beats": 0.0,
                    "pitches": ["C4", "E4"],
                    "group_id": "g001",
                }
            ],
            "test:oversized",
            False,
        ),
    )

    result = compositor.compose_repaired_candidate_events(
        request,
        inference,
        lane,
        pitch_predictor=lambda candidate, request, image: "C4",
        out_dir=tmp_path,
        selector_method_id="repaired-v1",
    )

    assert result["meter_valid"] is False
    assert result["decoded_extent_beats"] == 4.0
    assert result["decoder_status"] == "test:oversized"


def test_request_only_meter_fallback_remains_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, inference, measure, lane = _fixture(tmp_path)
    monkeypatch.setattr(compositor.dense, "_prepare_dense_measure", lambda *args, **kwargs: measure)
    monkeypatch.setattr(
        compositor,
        "_decode_with_meter_fallback",
        lambda *args, **kwargs: (
            [
                {
                    "kind": "rest",
                    "x": 20.0,
                    "duration_beats": 3.0,
                    "onset_beats": 0.0,
                    "evidence": "request_meter_fallback",
                }
            ],
            "request_meter_fallback:meter_repaired",
            True,
        ),
    )

    result = compositor.compose_repaired_candidate_events(
        request,
        inference,
        lane,
        pitch_predictor=lambda candidate, request, image: "C4",
        out_dir=tmp_path,
        selector_method_id="repaired-v1",
    )

    assert result["status"] == compositor.STATUS_REQUEST_METER_FALLBACK
    assert result["prediction"] is None
    assert result["onsets"] is None
    assert result["durations"] is None
    assert result["rests"] is None
    assert result["meter_valid"] is False
    assert result["inference_provenance"]["meter_fallback"] == {
        "applied": True,
        "accepted_as_transcription": False,
        "expected_beats": 3.0,
        "truth_used": False,
    }


def _fixture(tmp_path: Path):
    image_path = tmp_path / "fixture" / "measure.png"
    image_path.parent.mkdir(parents=True)
    image = Image.new("RGB", (140, 125), "white")
    image.save(image_path)
    image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    identity = {
        "slug": "fixture",
        "system_index": 1,
        "system_measure_index": 1,
        "automatic_measure_index": 1,
    }
    context = {
        "expected_measure_beats": 3.0,
        "allow_pickup": False,
        "clef": "treble",
        "time_signature": "3/4",
        "key_hint": "one flat: Bb",
    }
    provenance = {"key_hint": "synthetic visual key fixture", "time_signature": "metadata"}
    staff_geometry = {"raw_staff_lines_y_px": [20, 40, 60, 80, 100]}
    request = {
        "identity": identity,
        "images": {
            "raw": {
                "path_relative_to_out": "fixture/measure.png",
                "sha256": image_hash,
            }
        },
        "staff_geometry": staff_geometry,
        "allowed_context": context,
        "allowed_context_provenance": provenance,
    }
    inference = {
        "identity": dict(identity),
        "truth_used": False,
        "source": {"image": str(image_path), "sha256": image_hash},
        "allowed_context": dict(context),
        "allowed_context_provenance": dict(provenance),
        "staff_geometry": dict(staff_geometry),
    }
    candidates = (
        _candidate("d001", 1, 20.0, 40.0),
        _candidate("d002", 2, 21.0, 60.0),
        _candidate("d003", 3, 80.0, 50.0),
    )
    measure = patches.UnlabeledMeasure(
        measure=1,
        source_image=image_path,
        source_sha256=image_hash,
        staff_lines=(20, 40, 60, 80, 100),
        staff_spacing=20.0,
        candidates=candidates,
    )
    lane = [
        {
            "candidate_id": candidate.id,
            "center": {"x": candidate.center_x, "y": candidate.center_y},
            "onset_group_index": 1 if candidate.id in {"d001", "d002"} else 2,
        }
        for candidate in candidates
    ]
    return request, inference, measure, lane


def _candidate(candidate_id: str, rank: int, x: float, y: float) -> patches.CandidatePatch:
    return patches.CandidatePatch(
        measure=1,
        id=candidate_id,
        rank=rank,
        center_x=x,
        center_y=y,
        bbox=(round(x - 4), round(y - 3), round(x + 4), round(y + 3)),
        detector_score=0.9,
        patches={"dense_features": (0.0,)},
    )
