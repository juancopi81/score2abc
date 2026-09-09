import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from scripts.experiments.spike_stem_endpoint_detector import (
    DetectorConfig,
    EndpointCandidate,
    SealedTruthGate,
    apply_validation_gate,
    build_ink_mask,
    dedupe_candidates,
    find_stem_runs,
    infer_endpoint_candidates,
    natural_midi_for_staff_position,
    predict,
    prepare_request,
)

CONFIG = DetectorConfig(0.8, 4.25, 0.2, 0.46, 0.12, 0.3)
STAFF_LINES = (20.0, 30.0, 40.0, 50.0, 60.0)


def _synthetic_note() -> Image.Image:
    image = Image.new("L", (100, 82), 255)
    draw = ImageDraw.Draw(image)
    for y in STAFF_LINES:
        draw.line((0, y, 99, y), fill=0, width=1)
    draw.line((55, 24, 55, 51), fill=0, width=2)
    draw.ellipse((45, 46, 56, 54), outline=0, width=2)
    return image


def test_stem_runs_survive_staff_line_suppression_and_small_gaps() -> None:
    image = _synthetic_note()
    mask = build_ink_mask(image, threshold=180, staff_lines=STAFF_LINES, spacing=10)

    stems = find_stem_runs(mask, staff_lines=STAFF_LINES, spacing=10, config=CONFIG)

    matching = [stem for stem in stems if abs(stem.x - 55) <= 2]
    assert matching
    assert matching[0].top <= 25
    assert matching[0].bottom >= 50


def test_endpoint_inference_prefers_notehead_end_of_stem() -> None:
    image = _synthetic_note()
    mask = build_ink_mask(image, threshold=180, staff_lines=STAFF_LINES, spacing=10)
    stems = find_stem_runs(mask, staff_lines=STAFF_LINES, spacing=10, config=CONFIG)

    candidates = infer_endpoint_candidates(
        mask,
        [min(stems, key=lambda stem: abs(stem.x - 55))],
        staff_lines=STAFF_LINES,
        spacing=10,
        config=CONFIG,
    )

    assert len(candidates) == 1
    assert candidates[0].endpoint == "bottom"
    assert abs(candidates[0].y - 50) <= 5
    assert candidates[0].pitch_midi == natural_midi_for_staff_position(2)


def test_dedupe_is_score_first_then_returns_left_to_right() -> None:
    def candidate(x: float, y: float, score: float) -> EndpointCandidate:
        return EndpointCandidate(x, y, score, x + 2, 10, 40, "bottom", 64, 0)

    result = dedupe_candidates(
        [candidate(40, 50, 0.7), candidate(42, 51, 0.9), candidate(15, 45, 0.6)],
        spacing=10,
        x_tolerance_spaces=0.3,
    )

    assert [(row.x, row.score) for row in result] == [(15, 0.6), (42, 0.9)]


def test_truth_gate_blocks_unsealed_and_failed_gate_heldout_reads(tmp_path: Path) -> None:
    gate = SealedTruthGate()
    validation_truth = tmp_path / "validation.jsonl"
    validation_truth.write_text('{"split":"validation"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="sealed validation predictions"):
        gate.read_truth("validation", validation_truth)

    validation_predictions = tmp_path / "validation.predictions.jsonl"
    validation_predictions.write_text("{}\n", encoding="utf-8")
    seal = gate.seal_predictions("validation", validation_predictions)
    assert gate.read_truth("validation", validation_truth) == [{"split": "validation"}]
    assert gate.access_log[0]["after_prediction_sha256"] == seal["sha256"]

    gate.record_validation_result({"passed": False})
    heldout_predictions = tmp_path / "heldout.predictions.jsonl"
    heldout_predictions.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="passing validation gate"):
        gate.seal_predictions("heldout", heldout_predictions)


def test_validation_gate_requires_both_preregistered_metrics() -> None:
    passing = apply_validation_gate(
        {"summary": {"pitch_only_note_f1": 0.70, "exact_note_count_rate": 0.50}}
    )
    failing = apply_validation_gate(
        {"summary": {"pitch_only_note_f1": 0.90, "exact_note_count_rate": 0.49}}
    )

    assert passing["passed"] is True
    assert failing["passed"] is False
    assert failing["failure_action"] == "skip_heldout_without_reading_heldout_truth"


def test_prediction_is_deterministic(tmp_path: Path) -> None:
    image_path = tmp_path / "note.png"
    _synthetic_note().save(image_path)
    request = {
        "identity": {
            "slug": "synthetic",
            "system_index": 1,
            "system_measure_index": 1,
            "global_measure_index": 0,
        },
        "images": {
            "raw": {
                "path_relative_to_out": image_path.name,
                "sha256": __import__("hashlib").sha256(image_path.read_bytes()).hexdigest(),
            }
        },
        "staff_geometry": {"raw_staff_lines_y_px": list(STAFF_LINES)},
    }
    prepared = prepare_request(request, out_dir=tmp_path)

    first = predict(prepared, CONFIG)
    second = predict(prepared, CONFIG)

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
