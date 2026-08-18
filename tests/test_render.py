import json
import logging
from pathlib import Path

from PIL import Image, ImageDraw
from PIL.ImageStat import Stat

from score2abc.render import _assess_system_candidate, _darken_ink, create_system_crops


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


def test_create_system_crops_splits_connected_staff_systems(tmp_path: Path) -> None:
    page_path = tmp_path / "page_001.png"
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir()
    _write_synthetic_page(page_path)

    page = Image.open(page_path)
    draw = ImageDraw.Draw(page)
    draw.rectangle((850, 440, 930, 1320), fill="black")
    page.save(page_path)

    result = create_system_crops(
        [page_path],
        systems_dir,
        logging.getLogger("test.render.connected-systems"),
    )

    assert len(result.system_crops) == 2
    manifest = json.loads(result.debug_manifests[0].read_text(encoding="utf-8"))
    assert len(manifest["systems"]) == 2
    assert all(
        len(item["staff_line_rows"]) == 5 and len(item["long_horizontal_line_rows"]) == 5
        for item in manifest["candidates"]
    )


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


def test_create_system_crops_rejects_non_staff_band_and_preserves_source_index(
    tmp_path: Path,
) -> None:
    page_path = tmp_path / "page_001.png"
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir()
    _write_synthetic_page(page_path, with_title_band=True)
    stale_rejected = systems_dir / "rejected_candidate_page_001_999.png"
    stale_rejected.write_bytes(b"stale")

    result = create_system_crops(
        [page_path], systems_dir, logging.getLogger("test.render.eligibility")
    )

    assert len(result.system_crops) == 2
    assert len(result.rejected_candidate_crops) == 1
    assert not stale_rejected.exists()
    assert result.rejected_candidate_crops[0].exists()

    manifest = json.loads(result.debug_manifests[0].read_text(encoding="utf-8"))
    assert len(manifest["candidates"]) == 3
    assert [item["source_candidate_index"] for item in manifest["systems"]] == [2, 3]
    assert [item["output_system_index"] for item in manifest["systems"]] == [1, 2]

    rejected = manifest["rejected_candidates"]
    assert len(rejected) == 1
    assert rejected[0]["source_candidate_index"] == 1
    assert rejected[0]["output_system_index"] is None
    assert rejected[0]["reason"] == "insufficient_long_horizontal_lines"
    assert rejected[0]["candidate_crop"] == str(result.rejected_candidate_crops[0])
    assert manifest["systems"][0]["chord_bbox_above"]["top"] > rejected[0]["system_bbox"]["bottom"]
    assert result.candidate_diagnostics == manifest["candidates"]


def test_create_system_crops_preserves_weak_left_preamble(tmp_path: Path) -> None:
    page_path = tmp_path / "page_001.png"
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir()
    _write_weak_left_preamble_page(page_path)

    result = create_system_crops(
        [page_path], systems_dir, logging.getLogger("test.render.left-preamble")
    )

    assert len(result.system_crops) == 1
    manifest = json.loads(result.debug_manifests[0].read_text(encoding="utf-8"))
    system_bbox = manifest["systems"][0]["system_bbox"]
    assert system_bbox["left"] <= 350

    saved = Image.open(result.system_crops[0]).convert("L")
    left_band = saved.crop((0, 0, min(180, saved.width), saved.height))
    assert Stat(left_band).extrema[0][0] < 80


def test_create_system_crops_recovers_sparse_staff_after_dense_preamble(
    tmp_path: Path,
) -> None:
    page_path = tmp_path / "page_001.png"
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir()
    _write_sparse_staff_after_dense_preamble_page(page_path)

    result = create_system_crops(
        [page_path], systems_dir, logging.getLogger("test.render.sparse-staff")
    )

    assert len(result.system_crops) == 1
    manifest = json.loads(result.debug_manifests[0].read_text(encoding="utf-8"))
    system_bbox = manifest["systems"][0]["system_bbox"]
    assert system_bbox["left"] <= 260
    assert system_bbox["right"] >= 1600


def test_create_system_crops_rejects_trailing_blank_staffs(tmp_path: Path) -> None:
    page_path = tmp_path / "page_001.png"
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir()
    _write_page_with_blank_staff_tail(page_path)

    result = create_system_crops(
        [page_path], systems_dir, logging.getLogger("test.render.blank-tail")
    )

    assert len(result.system_crops) == 1
    assert len(result.rejected_candidate_crops) == 2
    manifest = json.loads(result.debug_manifests[0].read_text(encoding="utf-8"))
    assert [item["reason"] for item in manifest["rejected_candidates"]] == [
        "insufficient_musical_ink",
        "insufficient_musical_ink",
    ]
    assert all(
        item["staff_residual_ink_density"] < 0.09 for item in manifest["rejected_candidates"]
    )


def test_darken_ink_crushes_mid_tones_and_preserves_extremes() -> None:
    probe = Image.new("RGB", (3, 1), (255, 255, 255))
    probe.putpixel((0, 0), (0, 0, 0))
    probe.putpixel((1, 0), (128, 128, 128))
    probe.putpixel((2, 0), (255, 255, 255))

    darkened = _darken_ink(probe, gamma=3.5)

    assert darkened.getpixel((0, 0)) == (0, 0, 0)
    assert darkened.getpixel((2, 0)) == (255, 255, 255)
    mid_after = darkened.getpixel((1, 0))[0]
    assert mid_after < 40, mid_after


def test_system_candidate_rejects_five_inconsistently_spaced_lines() -> None:
    image = Image.new("L", (1000, 180), "white")
    draw = ImageDraw.Draw(image)
    for y in (18, 33, 58, 93, 138):
        draw.line((40, y, 940, y), fill="black", width=3)

    assessment = _assess_system_candidate(
        image,
        page_number=1,
        source_candidate_index=1,
        system_bbox=(0, 0, 1000, 180),
        ink_threshold=220,
    )

    assert not assessment.accepted
    assert assessment.reason == "inconsistent_horizontal_line_spacing"
    assert len(assessment.long_horizontal_line_rows) == 5
    assert not assessment.staff_line_rows


def test_darken_ink_preserves_off_white_background() -> None:
    # Yellowed-paper regression: a page whose brightest pixels are ~245
    # must not have its background pulled down to mid-gray. The curve
    # should anchor to the page's own white point.
    probe = Image.new("RGB", (200, 1), (245, 245, 245))
    for x in range(5):
        probe.putpixel((x, 0), (60, 60, 60))

    darkened = _darken_ink(probe, gamma=3.5)

    background = darkened.getpixel((100, 0))[0]
    ink = darkened.getpixel((0, 0))[0]
    assert background >= 245, background
    assert ink < 30, ink


def _write_synthetic_page(
    page_path: Path,
    skew_degrees: float = 0.0,
    *,
    with_title_band: bool = False,
) -> None:
    image = Image.new("RGB", (1800, 2400), "white")
    draw = ImageDraw.Draw(image)
    left = 160
    right = 1640

    if with_title_band:
        # Separated title/author blocks span enough width and height to become
        # a broad proposal, but do not contain five long horizontal lines.
        draw.rectangle((420, 140, 650, 185), fill="black")
        draw.rectangle((1160, 175, 1390, 220), fill="black")

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


def _write_weak_left_preamble_page(page_path: Path) -> None:
    image = Image.new("RGB", (1800, 1000), "white")
    draw = ImageDraw.Draw(image)
    top = 380
    weak_left = 330
    main_left = 500
    right = 1640

    # The scan preserves only three faint/fragmented staff lines behind the
    # clef and meter. They are insufficient for the broad horizontal-bounds
    # threshold but still establish that the preamble belongs to this staff.
    for row in range(5):
        y = top + row * 18
        if row in {0, 2, 4}:
            draw.line((weak_left, y, main_left, y), fill="black", width=2)
        draw.line((main_left, y, right, y), fill="black", width=3)

    draw.ellipse((350, top - 16, 390, top + 86), outline="black", width=5)
    draw.line((405, top - 5, 405, top + 80), fill="black", width=5)
    draw.line((430, top + 8, 470, top + 8), fill="black", width=4)
    draw.line((430, top + 45, 470, top + 45), fill="black", width=4)
    for note_index, note_x in enumerate(range(main_left + 80, right - 100, 150)):
        head_top = top + 24 + (note_index % 3) * 6
        draw.ellipse((note_x, head_top, note_x + 24, head_top + 18), fill="black")
        draw.line((note_x + 24, head_top - 36, note_x + 24, head_top + 10), fill="black", width=4)
    draw.line((right, top - 10, right, top + 96), fill="black", width=4)
    image.save(page_path)


def _write_sparse_staff_after_dense_preamble_page(page_path: Path) -> None:
    image = Image.new("RGB", (1800, 1000), "white")
    draw = ImageDraw.Draw(image)
    top = 380
    left = 180
    right = 1650

    for row in range(5):
        y = top + row * 18
        draw.line((left, y, right, y), fill="black", width=2)

    # A dense clef/meter opening raises the generic column threshold enough
    # that the lightly drawn continuation no longer forms a broad span.
    for x in range(210, 391, 2):
        draw.line((x, top - 35, x, top + 110), fill="black", width=1)
    for note_index, note_x in enumerate(range(470, right - 80, 145)):
        head_top = top + 24 + (note_index % 3) * 6
        draw.ellipse((note_x, head_top, note_x + 22, head_top + 16), fill="black")
        draw.line((note_x + 22, head_top - 35, note_x + 22, head_top + 8), fill="black", width=3)
    image.save(page_path)


def _write_page_with_blank_staff_tail(page_path: Path) -> None:
    image = Image.new("RGB", (1800, 2200), "white")
    draw = ImageDraw.Draw(image)
    left = 160
    right = 1640
    for system_index, top in enumerate((300, 950, 1600)):
        for row in range(5):
            y = top + row * 18
            draw.line((left, y, right, y), fill="black", width=3)
        draw.line((left, top - 10, left, top + 96), fill="black", width=4)
        draw.line((right, top - 10, right, top + 96), fill="black", width=4)
        if system_index:
            # Decorative writing crossing an otherwise blank ruled staff is
            # the real failure pattern: it creates a broad proposal but is
            # not musical content.
            draw.line(
                (
                    (left + 180, top + 62),
                    (left + 330, top - 12),
                    (left + 470, top + 70),
                    (left + 640, top - 8),
                    (left + 820, top + 48),
                ),
                fill="black",
                width=5,
            )
            draw.rectangle((left + 900, top - 10, left + 970, top + 96), fill="black")
            continue
        for note_index, note_x in enumerate(range(left + 80, right - 80, 90)):
            head_top = top + 22 + (note_index % 3) * 7
            draw.ellipse((note_x, head_top, note_x + 25, head_top + 18), fill="black")
            draw.line(
                (note_x + 24, head_top - 38, note_x + 24, head_top + 10),
                fill="black",
                width=4,
            )
    image.save(page_path)


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
