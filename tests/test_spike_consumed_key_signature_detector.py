from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.experiments import spike_consumed_key_signature_detector as detector


def _signature_image(
    tmp_path: Path,
    name: str,
    *,
    family: str,
    count: int,
    mode: str,
    prefix_spaces: float = 0.0,
) -> Path:
    image = Image.new("L", (360, 180), 255)
    draw = ImageDraw.Draw(image)
    lines = (45, 65, 85, 105, 125)
    for y in lines:
        draw.line((0, y, image.width - 1, y), fill=0, width=1)
    draw.line((8, 35, 8, 135), fill=0, width=2)
    x = 15
    if mode == detector.MODE_CHANGE:
        draw.line((14, 35, 14, 135), fill=0, width=2)
        x = 27
    else:
        # A bounded clef-like component; only its right edge matters to this spike.
        draw.line((18, 25, 35, 145), fill=0, width=3)
        draw.arc((15, 50, 48, 132), 80, 300, fill=0, width=3)
        x = 55
    x += round(prefix_spaces * 20)
    positions = (
        detector.SHARP_POSITIONS if family == detector.FAMILY_SHARP else detector.FLAT_POSITIONS
    )
    for position in positions[:count]:
        anchor_y = round(lines[0] + position * 20)
        if family == detector.FAMILY_SHARP:
            draw.line((x, anchor_y - 14, x, anchor_y + 14), fill=0, width=2)
            draw.line((x + 10, anchor_y - 14, x + 10, anchor_y + 14), fill=0, width=2)
            draw.line((x - 2, anchor_y - 5, x + 12, anchor_y - 7), fill=0, width=2)
            draw.line((x - 2, anchor_y + 5, x + 12, anchor_y + 3), fill=0, width=2)
        else:
            draw.line((x, anchor_y - 23, x, anchor_y + 8), fill=0, width=2)
            draw.arc((x - 1, anchor_y - 5, x + 13, anchor_y + 11), 270, 90, fill=0, width=2)
        x += 20
    path = tmp_path / name
    image.save(path)
    return path


def test_detects_initial_single_flat(tmp_path: Path) -> None:
    path = _signature_image(
        tmp_path,
        "system_001.png",
        family=detector.FAMILY_FLAT,
        count=1,
        mode=detector.MODE_INITIAL,
    )

    result = detector.detect_signature(path, mode=detector.MODE_INITIAL)

    assert result["gate_passed"] is True
    assert result["predicted_signature_family"] == detector.FAMILY_FLAT
    assert result["accidental_count"] == 1
    assert result["fifths"] == -1
    assert result["truth_used_for_prediction"] is False


def test_detects_two_sharps_after_double_bar(tmp_path: Path) -> None:
    path = _signature_image(
        tmp_path,
        "measure_004.png",
        family=detector.FAMILY_SHARP,
        count=2,
        mode=detector.MODE_CHANGE,
    )

    result = detector.detect_signature(path, mode=detector.MODE_CHANGE)

    assert result["structural_boundary"]["style"] == "double_bar"
    assert result["predicted_signature_family"] == detector.FAMILY_SHARP
    assert result["accidental_count"] == 2
    assert result["fifths"] == 2


def test_detects_two_flats_after_double_bar(tmp_path: Path) -> None:
    path = _signature_image(
        tmp_path,
        "measure_005.png",
        family=detector.FAMILY_FLAT,
        count=2,
        mode=detector.MODE_CHANGE,
    )

    result = detector.detect_signature(path, mode=detector.MODE_CHANGE)

    assert result["predicted_signature_family"] == detector.FAMILY_FLAT
    assert result["accidental_count"] == 2
    assert result["fifths"] == -2


def test_change_mode_fails_closed_without_double_bar(tmp_path: Path) -> None:
    path = _signature_image(
        tmp_path,
        "measure_006.png",
        family=detector.FAMILY_SHARP,
        count=1,
        mode=detector.MODE_INITIAL,
    )

    result = detector.detect_signature(path, mode=detector.MODE_CHANGE)

    assert result["gate_passed"] is False
    assert result["fifths"] is None
    assert result["predicted_change"] == detector.PREDICTION_UNKNOWN


def test_change_mode_rejects_late_note_like_flat(tmp_path: Path) -> None:
    path = _signature_image(
        tmp_path,
        "measure_007.png",
        family=detector.FAMILY_FLAT,
        count=1,
        mode=detector.MODE_CHANGE,
        prefix_spaces=2.5,
    )

    result = detector.detect_signature(path, mode=detector.MODE_CHANGE)

    assert result["gate_passed"] is True
    assert result["fifths"] is None


def test_context_hints_and_expectations_are_separate(tmp_path: Path) -> None:
    sharp = _signature_image(
        tmp_path,
        "measure_002.png",
        family=detector.FAMILY_SHARP,
        count=1,
        mode=detector.MODE_CHANGE,
    )
    events = [detector.EventInput(sharp, detector.MODE_CHANGE, 2, "change")]

    report = detector.analyze_events(
        events,
        tmp_path / "artifacts",
        expected_fifths={"change": -2},
    )

    assert report["predictions"][0]["fifths"] == 1
    assert report["evaluation"]["matches"] == 0
    assert report["evaluation"]["predictions_materialized_before_expectations"] is True
    assert report["context_hints"]["events"][0]["key_hint"] == {"fifths": 1}
    assert (
        json.loads((tmp_path / "artifacts" / "context_hints.json").read_text(encoding="utf-8"))
        == report["context_hints"]
    )


def test_context_event_records_work_slug_from_out_path(tmp_path: Path) -> None:
    work_dir = tmp_path / "out" / "demo-work" / "systems"
    work_dir.mkdir(parents=True)
    sharp = _signature_image(
        work_dir,
        "system_001.png",
        family=detector.FAMILY_SHARP,
        count=1,
        mode=detector.MODE_INITIAL,
    )

    report = detector.analyze_events(
        [detector.EventInput(sharp, detector.MODE_INITIAL, 1, "initial")],
        tmp_path / "artifacts",
    )

    assert report["predictions"][0]["slug"] == "demo-work"
    assert report["context_hints"]["events"][0]["source"]["slug"] == "demo-work"


def test_ordered_sequence_beats_later_single_glyph() -> None:
    def glyph(
        glyph_id: str,
        left: int,
        right: int,
        position: float,
        *,
        score: float = 0.8,
    ) -> dict[str, object]:
        return {
            "glyph_id": glyph_id,
            "bbox": {"left": left, "right": right},
            "width_staff_spaces": 0.8,
            "height_staff_spaces": 4.5,
            "tracks": [{}, {}],
            "cross_rows_y_px": [10, 20],
            "family_scores": {
                detector.FAMILY_SHARP: score,
                detector.FAMILY_FLAT: 0.0,
            },
            "anchor_positions": {
                detector.FAMILY_SHARP: position,
                detector.FAMILY_FLAT: position,
            },
        }

    result = detector._best_signature(
        [
            glyph("first_approximate", 10, 20, 0.2),
            glyph("second", 25, 35, 1.6),
            glyph("first_perfect_but_late", 30, 40, 0.0),
        ],
        detector.FAMILY_SHARP,
        spacing=20.0,
    )

    assert result is not None
    assert result["fifths"] == 2
    assert result["glyph_ids"] == ["first_approximate", "second"]


def test_broad_full_height_sharp_shape_is_eligible() -> None:
    glyph = {
        "tracks": [{}],
        "cross_rows_y_px": [20, 40],
        "width_staff_spaces": detector.MIN_BROAD_SHARP_WIDTH_SPACES,
        "height_staff_spaces": detector.MIN_BROAD_SHARP_HEIGHT_SPACES,
    }

    assert detector._sharp_shape_eligible(glyph) is True
    assert (
        detector._sharp_shape_eligible(
            {
                **glyph,
                "height_staff_spaces": detector.MIN_BROAD_SHARP_HEIGHT_SPACES - 0.1,
            }
        )
        is False
    )


def test_fragmented_clef_stops_before_first_accidental() -> None:
    image = Image.new("L", (160, 150), 255)
    draw = ImageDraw.Draw(image)
    staff_lines = [30, 50, 70, 90, 110]
    for y in staff_lines:
        draw.line((0, y, image.width - 1, y), fill=0, width=1)
    draw.rectangle((30, 20, 41, 120), fill=0)
    draw.rectangle((60, 20, 76, 120), fill=0)
    mask, _, spacing, _ = detector._staff_geometry(image)

    assert detector._initial_clef_right(mask, staff_lines, spacing, boundary=8) == 41


def test_expected_fifths_parser_supports_none() -> None:
    assert detector._expected_fifths("control=none") == ("control", None)
    assert detector._expected_fifths("two_flats=-2") == ("two_flats", -2)


def test_change_sweep_reports_only_gated_hits(tmp_path: Path) -> None:
    changed = _signature_image(
        tmp_path,
        "measure_001_staff.png",
        family=detector.FAMILY_SHARP,
        count=1,
        mode=detector.MODE_CHANGE,
    )
    control = _signature_image(
        tmp_path,
        "measure_002_staff.png",
        family=detector.FAMILY_FLAT,
        count=1,
        mode=detector.MODE_INITIAL,
    )

    sweep = detector.scan_change_inputs([changed, control])

    assert sweep["input_count"] == 2
    assert sweep["hit_count"] == 1
    assert sweep["hits"][0]["fifths"] == 1
    assert sweep["truth_used_for_prediction"] is False


def test_real_consumed_cases_when_artifacts_exist() -> None:
    cases = [
        (
            detector.REPO_ROOT / "out/jaime-llanos_64_la-chata_pasillo_luis-a-calvo/"
            "vlm_melody_third_score_heldout/v2/system_007/crops/measure_002.png",
            detector.MODE_CHANGE,
            1,
        ),
        (
            detector.REPO_ROOT / "out/jaime-llanos_12_aviador_pasillo_fulgencio-garcia/"
            "vlm_melody_inputs/system_002/measure_009_staff.png",
            detector.MODE_CHANGE,
            2,
        ),
        (
            detector.REPO_ROOT / "out/jaime-llanos_49_gatoe-fique_pasillo_emilio-murillo/"
            "systems/system_001.png",
            detector.MODE_INITIAL,
            2,
        ),
    ]
    for path, mode, expected in cases:
        if path.is_file():
            result = detector.detect_signature(path, mode=mode)
            assert result["truth_used_for_prediction"] is False
            if expected is not None:
                assert result["fifths"] == expected
