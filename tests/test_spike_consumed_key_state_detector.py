from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.experiments import spike_consumed_key_state_detector as detector


def _staff_image(
    tmp_path: Path,
    name: str,
    *,
    sharp: bool = False,
    flat_like: bool = False,
    double_bar: bool = False,
    noisy: bool = False,
) -> Path:
    image = Image.new("L", (240, 150), 255)
    draw = ImageDraw.Draw(image)
    for y in (35, 50, 65, 80, 95):
        draw.line((0, y, 239, y), fill=0, width=1)
    draw.line((8, 28, 8, 102), fill=0, width=2)
    if double_bar:
        draw.line((13, 28, 13, 102), fill=0, width=2)
    if sharp:
        draw.line((25, 28, 22, 102), fill=0, width=2)
        draw.line((43, 28, 46, 102), fill=0, width=2)
        for y in (43, 58, 73, 88):
            draw.line((22, y, 46, y - 2), fill=0, width=2)
    elif flat_like:
        draw.line((28, 25, 28, 102), fill=0, width=2)
        draw.arc((28, 52, 50, 88), 270, 90, fill=0, width=2)
    else:
        draw.line((28, 25, 28, 102), fill=0, width=2)
        draw.line((30, 30, 46, 44), fill=0, width=2)
    if noisy:
        for x, y in ((18, 20), (58, 23), (71, 110), (102, 14), (155, 118)):
            draw.point((x, y), fill=0)
    path = tmp_path / name
    image.save(path)
    return path


def test_synthetic_sharp_is_detected_without_key_mapping(tmp_path: Path) -> None:
    path = _staff_image(tmp_path, "measure_002.png", sharp=True, double_bar=True)

    result = detector.detect_key_state(path)

    assert result["predicted_change"] == detector.PREDICTION_SHARP
    assert result["predicted_signature_family"] == "sharp"
    assert result["fifths"] == 1
    assert result["fifths_status"] == "confirmed_explicit"
    assert len(result["glyph_groups"]) == 1
    assert result["structural_boundary"]["style"] == "double_bar"
    assert result["boundary_gate_passed"] is True
    assert result["truth_used_for_prediction"] is False
    assert result["accepted_candidate_ids"]


def test_flat_like_single_stem_and_noise_fail_closed(tmp_path: Path) -> None:
    path = _staff_image(tmp_path, "measure_001.png", flat_like=True, noisy=True)

    result = detector.detect_key_state(path)

    assert result["predicted_change"] == detector.PREDICTION_UNKNOWN
    assert result["predicted_signature_family"] is None


def test_single_bar_sharp_accidental_and_notehead_fail_closed(tmp_path: Path) -> None:
    path = _staff_image(tmp_path, "measure_004.png", sharp=True)
    image = Image.open(path).convert("L")
    draw = ImageDraw.Draw(image)
    draw.ellipse((70, 78, 84, 90), fill=0)
    image.save(path)

    result = detector.detect_key_state(path)

    assert result["structural_boundary"]["style"] == "single_bar"
    assert result["boundary_gate_passed"] is False
    assert result["predicted_change"] == detector.PREDICTION_UNKNOWN
    assert result["accepted_candidate_ids"] == []
    assert result["fifths"] is None


def test_labels_are_evaluation_only_and_cannot_change_prediction(tmp_path: Path) -> None:
    path = _staff_image(tmp_path, "measure_003.png", sharp=True, double_bar=True)
    without_labels = detector.analyze_inputs([path], tmp_path / "without")
    with_labels = detector.analyze_inputs(
        [path],
        tmp_path / "with",
        {"labels": [{"input": path.name, "expected_change": detector.PREDICTION_UNKNOWN}]},
    )

    assert without_labels["measures"][0]["predicted_change"] == detector.PREDICTION_SHARP
    assert (
        with_labels["measures"][0]["predicted_change"]
        == without_labels["measures"][0]["predicted_change"]
    )
    assert with_labels["evaluation"] == {
        "compared_count": 1,
        "matches": 0,
        "accuracy": 0.0,
        "rows": [
            {
                "input": str(path.resolve()),
                "predicted": detector.PREDICTION_SHARP,
                "expected": detector.PREDICTION_UNKNOWN,
                "match": False,
            }
        ],
        "truth_used_for_prediction": False,
    }


def test_state_sequence_distinguishes_explicit_inherited_and_unknown(tmp_path: Path) -> None:
    unknown_before = _staff_image(tmp_path, "measure_001.png", flat_like=True)
    explicit_change = _staff_image(tmp_path, "measure_002.png", sharp=True, double_bar=True)
    inherited_after = _staff_image(tmp_path, "measure_003.png", flat_like=True)

    report = detector.analyze_inputs(
        [unknown_before, explicit_change, inherited_after], tmp_path / "states"
    )
    states = [row["state"] for row in report["measures"]]

    assert [state["kind"] for state in states] == [
        detector.STATE_UNKNOWN_INITIAL,
        detector.STATE_EXPLICIT_CHANGE,
        detector.STATE_INHERITED,
    ]
    assert states[0]["signature_family"] is None
    assert states[1]["signature_family"] == "sharp"
    assert states[1]["fifths"] == 1
    assert states[1]["pitch_mapping_ready"] is True
    assert states[2]["signature_family"] == "sharp"
    assert states[2]["fifths"] == 1
    assert states[2]["pitch_mapping_ready"] is True
    assert states[2]["source"] == states[1]["source"]
    assert report["state_model"]["default_key_assumption"] is None
    assert report["context_hints"] == {
        "schema_version": detector.SCHEMA_VERSION,
        "source": "automatic_visual_key_state_detector",
        "truth_used": False,
        "events": [
            {
                "start_measure": 2,
                "key_hint": {"fifths": 1},
                "source": {
                    "kind": "automatic_visual_key_change",
                    "image": str(explicit_change.resolve()),
                    "sha256": detector._sha256(explicit_change),
                    "candidate_id": report["measures"][1]["top_candidate_id"],
                },
            }
        ],
    }
    assert (
        json.loads((tmp_path / "states" / "context_hints.json").read_text(encoding="utf-8"))
        == report["context_hints"]
    )
    assert report["state_model"]["pitch_mapping_requires_exact_fifths"] is True


def test_overlapping_pairs_group_but_multiple_groups_keep_fifths_unknown() -> None:
    candidates = [
        {
            "candidate_id": "c001",
            "bbox": {"left": 10, "top": 0, "right": 24, "bottom": 20},
        },
        {
            "candidate_id": "c002",
            "bbox": {"left": 18, "top": 0, "right": 30, "bottom": 20},
        },
        {
            "candidate_id": "c003",
            "bbox": {"left": 45, "top": 0, "right": 58, "bottom": 20},
        },
    ]

    groups = detector._collapse_accepted_glyph_groups(candidates)

    assert [group["candidate_ids"] for group in groups] == [["c001", "c002"], ["c003"]]
    assert detector._conservative_sharp_fifths(groups) is None


def test_cli_writes_report_and_overlay(tmp_path: Path, capsys) -> None:
    path = _staff_image(tmp_path, "measure_002.png", sharp=True, double_bar=True)
    out_dir = tmp_path / "artifacts"

    assert detector.main(["--input", str(path), "--out-dir", str(out_dir)]) == 0
    captured = capsys.readouterr()
    assert '"measure_count": 1' in captured.out
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report["measures"][0]["predicted_change"] == detector.PREDICTION_SHARP
    assert json.loads((out_dir / "context_hints.json").read_text(encoding="utf-8"))["events"][0][
        "key_hint"
    ] == {"fifths": 1}
    overlay = out_dir / "measure_002_key_state_overlay.png"
    assert overlay.is_file()
    with Image.open(overlay) as image:
        assert image.size == (240, 150)


def test_real_crop_can_be_run_when_artifact_exists(tmp_path: Path) -> None:
    path = detector.DEFAULT_CONSUMED_CROP_DIR / "measure_002.png"
    if not path.is_file():
        return

    report = detector.analyze_inputs([path], tmp_path / "real")

    assert report["measures"][0]["truth_used_for_prediction"] is False
    assert (tmp_path / "real" / "measure_002_key_state_overlay.png").is_file()
