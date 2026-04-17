import json
import logging
from pathlib import Path

from PIL import Image, ImageDraw
from PIL.ImageStat import Stat

from score2abc.render import create_system_crops


def test_create_system_crops_detects_staffs_and_candidate_bands(tmp_path: Path) -> None:
    page_path = tmp_path / "page_001.png"
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir()
    _write_synthetic_page(page_path)
    legacy_path = systems_dir / "chord_region_001.png"
    legacy_path.write_bytes(b"legacy")
    legacy_normalized = systems_dir / "system_normalized_001.png"
    legacy_normalized.write_bytes(b"legacy")

    result = create_system_crops([page_path], systems_dir, logging.getLogger("test.render"))

    assert len(result.system_crops) == 2
    assert len(result.chord_crops_above) == 2
    assert len(result.chord_crops_below) == 2
    assert len(result.debug_overlays) == 1
    assert len(result.debug_manifests) == 1
    assert not legacy_path.exists()
    assert not legacy_normalized.exists()

    manifest = json.loads(result.debug_manifests[0].read_text(encoding="utf-8"))
    assert len(manifest["systems"]) == 2
    assert "page_rotation_degrees" in manifest
    assert manifest["source_page"] == str(page_path)
    assert manifest["page"] == str(result.deskewed_pages[0])
    _assert_manifest_replays(manifest, result.system_crops)

    original_width, original_height = Image.open(page_path).size
    above_darkness: list[float] = []
    below_darkness: list[float] = []
    for item, system_path, above_path, below_path in zip(
        manifest["systems"],
        result.system_crops,
        result.chord_crops_above,
        result.chord_crops_below,
        strict=True,
    ):
        system_bbox = item["system_bbox"]
        system_crop_bbox = item["system_crop_bbox"]
        chord_bbox_above = item["chord_bbox_above"]
        chord_crop_bbox_above = item["chord_crop_bbox_above"]
        chord_bbox_below = item["chord_bbox_below"]
        chord_crop_bbox_below = item["chord_crop_bbox_below"]
        # Chord bands now overlap into the staff region so chords touching the
        # outer staff lines are captured.
        assert chord_bbox_above["bottom"] > system_bbox["top"]
        assert chord_bbox_above["bottom"] <= (system_bbox["top"] + system_bbox["bottom"]) // 2
        assert chord_bbox_below["top"] < system_bbox["bottom"]
        assert chord_bbox_below["top"] >= (system_bbox["top"] + system_bbox["bottom"]) // 2
        assert system_crop_bbox["top"] <= system_bbox["top"]
        assert system_crop_bbox["bottom"] >= system_bbox["bottom"]
        assert chord_crop_bbox_above["top"] <= chord_bbox_above["top"]
        assert chord_crop_bbox_above["bottom"] >= chord_bbox_above["bottom"]
        assert chord_crop_bbox_below["top"] <= chord_bbox_below["top"]
        assert chord_crop_bbox_below["bottom"] >= chord_bbox_below["bottom"]

        system_width, system_height = Image.open(system_path).size
        above_width, above_height = Image.open(above_path).size
        below_width, below_height = Image.open(below_path).size
        assert system_width < original_width
        assert system_height < original_height
        assert system_height > system_bbox["bottom"] - system_bbox["top"]
        assert above_width == system_width
        assert below_width == system_width
        assert above_height < system_height
        assert below_height < system_height
        assert above_height > chord_bbox_above["bottom"] - chord_bbox_above["top"]
        assert below_height > chord_bbox_below["bottom"] - chord_bbox_below["top"]

        above_darkness.append(_crop_darkness(above_path))
        below_darkness.append(_crop_darkness(below_path))

    assert above_darkness[0] > below_darkness[0]
    assert below_darkness[1] > above_darkness[1]


def test_create_system_crops_deskews_full_page(tmp_path: Path) -> None:
    page_path = tmp_path / "page_001.png"
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir()
    skew = 2.25
    _write_synthetic_page(page_path, skew_degrees=skew)

    result = create_system_crops([page_path], systems_dir, logging.getLogger("test.render.skew"))

    assert result.system_crops
    manifest = json.loads(result.debug_manifests[0].read_text(encoding="utf-8"))
    page_rotation = float(manifest["page_rotation_degrees"])
    # Detector should recover a rotation magnitude close to the input skew.
    assert abs(abs(page_rotation) - skew) <= 0.5
    assert manifest["source_page"] == str(page_path)
    assert manifest["page"] == str(result.deskewed_pages[0])
    _assert_manifest_replays(manifest, result.system_crops)

    deskewed_score = _row_peakiness(result.system_crops[0])
    # Re-rotating by the input skew should smear staff rows and lower peakiness.
    re_skewed = (
        Image.open(result.system_crops[0])
        .convert("L")
        .rotate(skew, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=255)
    )
    probe_path = tmp_path / "reskewed_probe.png"
    re_skewed.save(probe_path)
    reskewed_score = _row_peakiness(probe_path)
    assert deskewed_score > reskewed_score


def _write_synthetic_page(page_path: Path, skew_degrees: float = 0.0) -> None:
    image = Image.new("RGB", (1800, 2400), "white")
    draw = ImageDraw.Draw(image)
    left = 160
    right = 1640

    for system_index, top in enumerate((360, 1320), start=1):
        if system_index == 1:
            _draw_annotation_blocks(draw, left + 220, top - 96)
        else:
            _draw_annotation_blocks(draw, left + 260, top + 128)

        for row in range(5):
            y = top + row * 18
            draw.line((left, y, right, y), fill="black", width=3)

        for note_index, note_x in enumerate(range(left + 80, right - 100, 220)):
            head_top = top + 24 + (note_index % 3) * 6
            draw.ellipse((note_x, head_top, note_x + 24, head_top + 18), outline="black", width=3)
            draw.line(
                (note_x + 24, head_top - 36, note_x + 24, head_top + 10), fill="black", width=3
            )

        draw.line((left, top - 10, left, top + 96), fill="black", width=4)
        draw.line((right, top - 10, right, top + 96), fill="black", width=4)

    if skew_degrees:
        image = image.rotate(
            skew_degrees,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor="white",
        )

    image.save(page_path)


def _draw_annotation_blocks(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    for offset in (0, 180, 420):
        draw.rectangle((x + offset, y, x + offset + 110, y + 28), fill="black")


def _crop_darkness(path: Path) -> float:
    image = Image.open(path).convert("L")
    return 255 - Stat(image).mean[0]


def _assert_manifest_replays(manifest: dict, system_crops: list[Path]) -> None:
    """Cropping the file named in manifest["page"] must reproduce the saved system crops."""
    page = Image.open(manifest["page"]).convert("RGB")
    for item, system_path in zip(manifest["systems"], system_crops, strict=True):
        bbox = item["system_crop_bbox"]
        replay = page.crop((bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]))
        saved = Image.open(system_path).convert("RGB")
        assert replay.size == saved.size, (replay.size, saved.size, system_path)
        assert replay.tobytes() == saved.tobytes(), f"manifest bbox did not replay {system_path}"


def _row_peakiness(path: Path) -> float:
    image = Image.open(path).convert("L")
    width = image.width
    left = int(width * 0.02)
    right = max(left + 1, int(width * 0.98))
    pixels = image.load()
    rows = []
    for y in range(image.height):
        dark_pixels = 0
        for x in range(left, right):
            if pixels[x, y] < 220:
                dark_pixels += 1
        rows.append(dark_pixels / max(1, right - left))
    mean = sum(rows) / len(rows)
    return sum((value - mean) ** 2 for value in rows)
