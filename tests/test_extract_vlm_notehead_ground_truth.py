from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from scripts.extract_vlm_notehead_ground_truth import (
    extract_blue_annotation_components,
    write_notehead_ground_truth_fixture,
)


def _annotation_image() -> Image.Image:
    image = Image.new("RGBA", (20, 20), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.ellipse((2, 2, 8, 8), outline=(0, 60, 255, 255), width=2)
    draw.ellipse((12, 10, 18, 16), outline=(0, 60, 255, 255), width=2)
    return image


def test_extract_blue_annotation_components_is_left_to_right_and_deterministic() -> None:
    image = _annotation_image()

    first = extract_blue_annotation_components(image)
    second = extract_blue_annotation_components(image)

    assert first == second
    assert len(first) == 2
    assert [component.bbox for component in first] == [(2, 2, 9, 9), (12, 10, 19, 17)]
    assert first[0].center[0] < first[1].center[0]


def test_fixture_writer_maps_centers_and_uses_canonical_pitch_metadata(tmp_path: Path) -> None:
    annotation_path = tmp_path / "measure_001_raw_notehead_gt.png"
    source_path = tmp_path / "measure_001_raw.png"
    context_path = tmp_path / "measure_001_context.json"
    canonical_path = tmp_path / "canonical.json"
    output_path = tmp_path / "fixture.json"
    _annotation_image().save(annotation_path)
    Image.new("RGB", (40, 40), "white").save(source_path)
    context_path.write_text(
        json.dumps(
            {
                "slug": "demo",
                "system_index": 1,
                "system_measure_index": 1,
                "global_measure_index": 0,
            }
        ),
        encoding="utf-8",
    )
    canonical_path.write_text(
        json.dumps(
            {
                "notes": [
                    {"measure": 0, "pitch_midi": 57},
                    {"measure": 0, "pitch_midi": 70, "accidental": -1},
                ]
            }
        ),
        encoding="utf-8",
    )

    write_notehead_ground_truth_fixture(
        annotation_path,
        source_path,
        canonical_path,
        context_path,
        output_path,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["notehead_count"] == 2
    assert [item["pitch"] for item in payload["noteheads"]] == ["A3", "Bb4"]
    assert payload["noteheads"][0]["center"]["x"] == 10.0
    assert payload["noteheads"][0]["center"]["y"] == 10.0
    assert payload["coordinate_provenance"]["mapping"] == "scaled to raw source dimensions"
    assert payload["musical_gt_provenance"]["type"] == "canonical_ground_truth_json"
    assert "human-labeled PNG" in payload["coordinate_provenance"]["description"]


def test_fixture_writer_rejects_canonical_count_mismatch(tmp_path: Path) -> None:
    annotation_path = tmp_path / "measure_001_raw_notehead_gt.png"
    source_path = tmp_path / "measure_001_raw.png"
    context_path = tmp_path / "measure_001_context.json"
    canonical_path = tmp_path / "canonical.json"
    output_path = tmp_path / "fixture.json"
    _annotation_image().save(annotation_path)
    Image.new("RGB", (20, 20), "white").save(source_path)
    context_path.write_text(
        json.dumps(
            {
                "slug": "demo",
                "system_index": 1,
                "system_measure_index": 1,
                "global_measure_index": 0,
            }
        ),
        encoding="utf-8",
    )
    canonical_path.write_text(
        json.dumps({"notes": [{"measure": 0, "pitch_midi": 57}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match canonical note count"):
        write_notehead_ground_truth_fixture(
            annotation_path,
            source_path,
            canonical_path,
            context_path,
            output_path,
        )
    assert not output_path.exists()


def test_fixture_writer_rejects_deprecated_annotation_name(tmp_path: Path) -> None:
    annotation_path = tmp_path / "measure_001_raw_notehead_gt_deprecated.png"
    source_path = tmp_path / "measure_001_raw.png"
    context_path = tmp_path / "measure_001_context.json"
    canonical_path = tmp_path / "canonical.json"
    annotation_path.write_bytes(b"not-used")

    with pytest.raises(ValueError, match="Deprecated annotation image"):
        write_notehead_ground_truth_fixture(
            annotation_path,
            source_path,
            canonical_path,
            context_path,
            tmp_path / "fixture.json",
        )
