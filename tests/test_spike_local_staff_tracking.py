import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from scripts.experiments import spike_local_staff_tracking as spike


def test_track_common_staff_shift_follows_curved_parallel_lines() -> None:
    width = 140
    base_lines = [20, 30, 40, 50, 60]
    image = Image.new("L", (width, 90), color=255)
    expected = []
    for x in range(width):
        shift = round(6 * x / (width - 1))
        expected.append(shift)
        for line in base_lines:
            image.putpixel((x, line + shift), 0)

    shifts = spike.track_common_staff_shift(image, base_lines=base_lines, spacing=10, max_shift=8)

    assert max(abs(actual - truth) for actual, truth in zip(shifts, expected, strict=True)) <= 1


def test_track_common_staff_shift_ignores_isolated_vertical_strokes() -> None:
    width = 100
    base_lines = [20, 30, 40, 50, 60]
    image = Image.new("L", (width, 90), color=255)
    draw = ImageDraw.Draw(image)
    for line in base_lines:
        draw.line([(0, line + 4), (width - 1, line + 4)], fill=0, width=1)
    draw.line([(50, 5), (50, 80)], fill=0, width=4)

    shifts = spike.track_common_staff_shift(image, base_lines=base_lines, spacing=10, max_shift=8)

    assert min(shifts) >= 3
    assert max(shifts) <= 5


def test_staff_position_and_treble_pitch_apply_key_signature() -> None:
    assert spike.staff_position(50, bottom_line_y=60, spacing=10) == 2
    assert spike.pitch_for_staff_position(0, key_flat_letters={"B"}) == "E4"
    assert spike.pitch_for_staff_position(-3, key_flat_letters={"B"}) == "Bb3"
    assert spike.pitch_for_staff_position(5, key_flat_letters={"B"}) == "C5"


def test_refine_notehead_center_prefers_compact_oval_over_staff_and_stem() -> None:
    image = Image.new("L", (80, 60), color=255)
    draw = ImageDraw.Draw(image)
    draw.line([(0, 31), (79, 31)], fill=0, width=1)
    draw.line([(44, 8), (44, 34)], fill=0, width=2)
    draw.ellipse((31, 27, 45, 36), fill=0)

    center_x, center_y = spike.refine_notehead_center(
        image,
        bbox={"left": 27, "top": 21, "right": 50, "bottom": 41},
        spacing=20,
    )

    assert abs(center_x - 38) <= 2
    assert abs(center_y - 31.5) <= 2


def test_report_separates_image_only_tracking_from_human_conditioned_refinement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_dir = tmp_path / "reviews"
    fixture_dir.mkdir()
    image_path = tmp_path / "measure.png"
    Image.new("L", (80, 70), color=255).save(image_path)
    fixture = {
        "candidates": [
            {
                "id": "c001",
                "label": "accepted",
                "bbox": {"left": 20, "top": 20, "right": 40, "bottom": 40},
            }
        ],
        "final_noteheads": [
            {
                "source": {"kind": "candidate", "candidate_id": "c001"},
                "center": {"x": 30, "y": 30},
                "pitch": "B4",
                "order": 1,
            }
        ],
    }
    (fixture_dir / "demo_system_001_measure_001.json").write_text(
        json.dumps(fixture), encoding="utf-8"
    )
    monkeypatch.setattr(spike, "REVIEW_FIXTURE_DIR", fixture_dir)
    monkeypatch.setattr(spike, "REPO_ROOT", tmp_path)
    tracked = spike.TrackedMeasure(
        measure=1,
        image_path=image_path,
        context_path=tmp_path / "context.json",
        base_lines=(10.0, 20.0, 30.0, 40.0, 50.0),
        spacing=10.0,
        shifts=(0,) * 80,
    )

    report = spike.evaluate_tracks([tracked], slug="demo", system_index=1, key_flat_letters={"B"})
    markdown_path = tmp_path / "report.md"
    spike._write_markdown(report, markdown_path)

    assert report["schema_version"] == 2
    assert report["provenance"]["staff_trajectory"]["image_only"] is True
    refinement = report["provenance"]["center_refinement"]
    assert refinement["image_only"] is False
    assert refinement["conditional_on_human_confirmed_notehead"] is True
    assert report["results"][0]["center_refinement_search_region"] == "accepted_candidate_bbox"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Staff trajectories are image-only" in markdown
    assert "human-confirmed candidate bbox" in markdown
