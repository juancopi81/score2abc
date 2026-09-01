import json
from pathlib import Path

from PIL import Image, ImageDraw

from score2abc.chord_ocr import ChordDetection
from score2abc.chord_ocr.alignment import (
    assign_measures,
    detect_barlines,
    measure_boundaries,
    measure_boundaries_for_system,
    measures_in_system,
)
from scripts.experiments.eval_a1_all_systems import _match, _parse_via


def _draw_system(
    path: Path,
    barline_fractions: list[float],
    *,
    width: int = 800,
    height: int = 160,
    staff_top: int = 50,
    staff_bottom: int = 130,
    include_clef_stroke: bool = False,
) -> Path:
    image = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(image)
    # 5 thin staff lines across the full width.
    staff_spacing = (staff_bottom - staff_top) // 4
    for line in range(5):
        y = staff_top + line * staff_spacing
        draw.line([(0, y), (width - 1, y)], fill=0, width=1)

    if include_clef_stroke:
        clef_x = int(width * 0.01)
        draw.line([(clef_x, staff_top), (clef_x, staff_bottom)], fill=0, width=2)

    for fraction in barline_fractions:
        x = int(fraction * width)
        draw.line([(x, staff_top), (x, staff_bottom)], fill=0, width=2)

    image.save(path)
    return path


def test_detect_barlines_finds_peaks_at_expected_fractions(tmp_path: Path) -> None:
    expected = [0.25, 0.5, 0.75]
    path = _draw_system(tmp_path / "system.png", expected)
    detected = sorted(detect_barlines(path))
    assert len(detected) == 3
    for got, exp in zip(detected, sorted(expected), strict=True):
        assert abs(got - exp) < 0.01


def test_detect_barlines_skips_leading_clef_area(tmp_path: Path) -> None:
    real_barlines = [0.4, 0.7]
    path = _draw_system(tmp_path / "system.png", real_barlines, include_clef_stroke=True)
    detected = sorted(detect_barlines(path))
    assert len(detected) == 2
    for got, exp in zip(detected, sorted(real_barlines), strict=True):
        assert abs(got - exp) < 0.01


def test_detect_barlines_returns_empty_when_no_vertical_ink(tmp_path: Path) -> None:
    path = _draw_system(tmp_path / "system.png", barline_fractions=[])
    assert detect_barlines(path) == []


def test_detect_barlines_uses_all_five_staff_lines_for_vertical_span(tmp_path: Path) -> None:
    path = tmp_path / "system.png"
    image = Image.new("L", (800, 180), color=255)
    draw = ImageDraw.Draw(image)
    for y in (50, 70):
        draw.line([(0, y), (799, y)], fill=0, width=3)
    for y in (90, 110, 130):
        draw.line([(160, y), (640, y)], fill=0, width=3)

    draw.line([(250, 45), (250, 78)], fill=0, width=2)
    draw.line([(600, 50), (600, 130)], fill=0, width=2)
    image.save(path)

    detected = detect_barlines(path)

    assert len(detected) == 1
    assert abs(detected[0] - 0.75) < 0.01


def test_detect_barlines_recovers_weak_edge_barline(tmp_path: Path) -> None:
    path = _draw_system(tmp_path / "system.png", barline_fractions=[], width=800)
    image = Image.open(path)
    draw = ImageDraw.Draw(image)
    draw.line([(20, 50), (20, 130)], fill=0, width=2)
    image.save(path)

    detected = detect_barlines(path)

    assert len(detected) == 1
    assert abs(detected[0] - 0.025) < 0.01


def test_detect_barlines_prefers_clean_barline_over_nearby_note_stem(
    tmp_path: Path,
) -> None:
    path = _draw_system(tmp_path / "system.png", barline_fractions=[], width=800)
    image = Image.open(path)
    draw = ImageDraw.Draw(image)
    staff_top, staff_bottom = 50, 130
    # A note-like stem with side ink sits close enough to compete with the real barline.
    draw.line([(160, staff_top), (160, staff_bottom)], fill=0, width=2)
    draw.ellipse([(140, 86), (166, 106)], fill=0)
    draw.line([(180, staff_top), (180, staff_bottom)], fill=0, width=2)
    image.save(path)

    detected = detect_barlines(path)

    assert len(detected) == 1
    assert abs(detected[0] - 0.225) < 0.01


def test_detect_barlines_recovers_consumed_carrizal_boundary() -> None:
    fixture_dir = (
        Path(__file__).parent
        / "fixtures"
        / "barlines"
        / "jaime-llanos_19_carrizal_pasillo_emilio-murillo"
    )
    image_path = fixture_dir / "system_004.png"
    gt_boxes = _parse_via(
        json.loads((fixture_dir / "system_004_ground_truth.json").read_text(encoding="utf-8"))
    )

    with Image.open(image_path) as image:
        detected = detect_barlines(image_path)
        detected_px = [round(fraction * image.width) for fraction in detected]

    tolerance_px = 9.0
    matches = _match(gt_boxes, detected_px, tolerance_px)
    tp = sum(match is not None for match in matches)
    fp = len(detected_px) - tp
    fn = len(gt_boxes) - tp
    assert (tp, fp, fn) == (9, 0, 0)

    boundaries = measure_boundaries_for_system(image_path, detected)
    assert len(boundaries) == 9
    assert [round(boundary * image.width) for boundary in boundaries] == detected_px


def test_measure_boundaries_for_system_keeps_clean_short_final_measure(tmp_path: Path) -> None:
    path = _draw_system(tmp_path / "system.png", barline_fractions=[], width=800)
    image = Image.open(path)
    draw = ImageDraw.Draw(image)
    draw.line([(688, 50), (688, 130)], fill=0, width=2)
    draw.line([(760, 50), (760, 130)], fill=0, width=9)
    image.save(path)

    boundaries = measure_boundaries_for_system(path, [688 / 800, 760 / 800])

    assert boundaries == [0.0, 0.86, 0.95]


def test_measure_boundaries_for_system_rejects_leading_upstem_after_music(
    tmp_path: Path,
) -> None:
    path = _draw_system(tmp_path / "system.png", barline_fractions=[], width=800)
    image = Image.open(path)
    draw = ImageDraw.Draw(image)
    draw.ellipse([(20, 88), (48, 106)], fill=0)
    draw.line([(60, 20), (60, 130)], fill=0, width=2)
    draw.line([(400, 50), (400, 130)], fill=0, width=2)
    draw.line([(760, 50), (760, 130)], fill=0, width=9)
    image.save(path)

    boundaries = measure_boundaries_for_system(path, [0.075, 0.5, 0.95])

    assert boundaries == [0.0, 0.5, 0.95]


def test_measure_boundaries_for_system_rejects_upstem_before_terminal_barline(
    tmp_path: Path,
) -> None:
    path = _draw_system(tmp_path / "system.png", barline_fractions=[], width=800)
    image = Image.open(path)
    draw = ImageDraw.Draw(image)
    draw.line([(688, 20), (688, 130)], fill=0, width=2)
    draw.ellipse([(664, 88), (692, 106)], fill=0)
    draw.line([(760, 50), (760, 130)], fill=0, width=9)
    image.save(path)

    boundaries = measure_boundaries_for_system(path, [0.86, 0.95])

    assert boundaries == [0.0, 0.95]


def test_measure_boundaries_for_system_trims_blank_tail(tmp_path: Path) -> None:
    path = _draw_system(tmp_path / "system.png", barline_fractions=[], width=800)
    image = Image.open(path)
    draw = ImageDraw.Draw(image)
    staff_top, staff_bottom = 50, 130
    draw.line([(180, staff_top), (180, staff_bottom)], fill=0, width=2)
    draw.line([(360, staff_top), (360, staff_bottom)], fill=0, width=2)
    draw.line([(372, staff_top), (372, staff_bottom)], fill=0, width=2)
    draw.ellipse([(230, 88), (250, 104)], fill=0)
    image.save(path)

    boundaries = measure_boundaries_for_system(path, [180 / 800, 360 / 800])

    assert boundaries == [0.0, 0.225, 0.45]


def test_measure_boundaries_for_system_trims_blank_staff_prefix(tmp_path: Path) -> None:
    path = _draw_system(tmp_path / "system.png", barline_fractions=[], width=1000)
    image = Image.open(path)
    draw = ImageDraw.Draw(image)
    staff_top, staff_bottom = 50, 130
    for x in (90, 350, 650, 950):
        draw.line([(x, staff_top), (x, staff_bottom)], fill=0, width=7)
    draw.ellipse([(170, 88), (194, 106)], fill=0)
    draw.ellipse([(450, 88), (474, 106)], fill=0)
    image.save(path)

    boundaries = measure_boundaries_for_system(path, [0.09, 0.35, 0.65, 0.95])

    assert boundaries == [0.09, 0.35, 0.65, 0.95]


def test_measure_boundaries_for_system_keeps_inked_short_first_measure(
    tmp_path: Path,
) -> None:
    path = _draw_system(tmp_path / "system.png", barline_fractions=[], width=1000)
    image = Image.open(path)
    draw = ImageDraw.Draw(image)
    staff_top, staff_bottom = 50, 130
    for x in (90, 350, 650, 950):
        draw.line([(x, staff_top), (x, staff_bottom)], fill=0, width=7)
    draw.ellipse([(24, 88), (52, 106)], fill=0)
    draw.line([(50, 50), (50, 104)], fill=0, width=3)
    image.save(path)

    boundaries = measure_boundaries_for_system(path, [0.09, 0.35, 0.65, 0.95])

    assert boundaries == [0.0, 0.09, 0.35, 0.65, 0.95]


def test_measure_boundaries_for_system_keeps_single_barline_blank_tail(tmp_path: Path) -> None:
    path = _draw_system(tmp_path / "system.png", barline_fractions=[0.5], width=800)

    assert measure_boundaries_for_system(path, [0.5]) == [0.0, 0.5, 1.0]


def test_measure_boundaries_for_system_merges_accidental_only_slices(tmp_path: Path) -> None:
    path = _draw_system(tmp_path / "system.png", barline_fractions=[], width=1000)
    image = Image.open(path)
    draw = ImageDraw.Draw(image)
    staff_top, staff_bottom = 50, 130
    for x in (50, 250, 550, 950):
        draw.line([(x, staff_top), (x, staff_bottom)], fill=0, width=3)
    draw.line([(680, staff_top), (680, staff_bottom)], fill=0, width=2)
    draw.line([(690, staff_top), (690, staff_bottom)], fill=0, width=2)
    draw.arc([(652, 70), (690, 116)], start=90, end=270, fill=0, width=6)
    draw.ellipse([(650, 88), (672, 106)], fill=0)
    draw.ellipse([(700, 88), (726, 106)], fill=0)
    image.save(path)

    boundaries = measure_boundaries_for_system(path, [0.05, 0.25, 0.55, 0.68, 0.95])

    assert boundaries == [0.05, 0.25, 0.55, 0.95]


def test_measure_boundaries_for_system_merges_narrow_note_stem_slice(tmp_path: Path) -> None:
    path = _draw_system(tmp_path / "system.png", barline_fractions=[], width=1000)
    image = Image.open(path)
    draw = ImageDraw.Draw(image)
    staff_top, staff_bottom = 50, 130
    for x in (50, 300, 600, 950):
        draw.line([(x, staff_top), (x, staff_bottom)], fill=0, width=3)
    draw.line([(700, 70), (700, 112)], fill=0, width=2)
    draw.ellipse([(682, 94), (706, 112)], fill=0)
    for x in (780, 840, 900):
        draw.line([(x, 68), (x, 112)], fill=0, width=2)
        draw.ellipse([(x - 18, 94), (x + 6, 112)], fill=0)
    image.save(path)

    boundaries = measure_boundaries_for_system(path, [0.05, 0.3, 0.6, 0.7, 0.95])

    assert boundaries == [0.05, 0.3, 0.6, 0.95]


def test_measure_boundaries_for_system_merges_wide_blank_trailing_slice(
    tmp_path: Path,
) -> None:
    path = _draw_system(tmp_path / "system.png", barline_fractions=[], width=1000)
    image = Image.open(path)
    draw = ImageDraw.Draw(image)
    staff_top, staff_bottom = 50, 130
    for x in (50, 250, 500, 950):
        draw.line([(x, staff_top), (x, staff_bottom)], fill=0, width=3)
    draw.line([(750, 50), (750, 112)], fill=0, width=2)
    draw.ellipse([(728, 94), (754, 112)], fill=0)
    image.save(path)

    boundaries = measure_boundaries_for_system(path, [0.05, 0.25, 0.5, 0.75, 0.95])

    assert boundaries == [0.05, 0.25, 0.5, 0.95]


def test_measure_boundaries_for_system_recovers_all_coqueteos_measures() -> None:
    path = (
        Path(__file__).parent
        / "fixtures"
        / "barlines"
        / "jaime-llanos_22_coqueteos_pasillo_fulgencio-garcia"
        / "system_002.png"
    )
    with Image.open(path) as image:
        width = image.width

    detected = detect_barlines(path)
    boundaries = measure_boundaries_for_system(path, detected)

    assert [round(boundary * width) for boundary in detected] == [
        150,
        541,
        902,
        1091,
        1390,
        1735,
        1854,
        1951,
        2041,
        2126,
    ]
    assert [round(boundary * width) for boundary in boundaries] == [
        0,
        541,
        902,
        1091,
        1390,
        1735,
        2041,
        2126,
    ]


def test_assign_measures_maps_x_fraction_to_measure_index() -> None:
    detections = [
        ChordDetection(symbol_raw="C", symbol="C", x_fraction=0.1, confidence=1.0, band="above"),
        ChordDetection(symbol_raw="G", symbol="G", x_fraction=0.35, confidence=1.0, band="above"),
        ChordDetection(symbol_raw="Am", symbol="Am", x_fraction=0.6, confidence=1.0, band="above"),
        ChordDetection(symbol_raw="F", symbol="F", x_fraction=0.9, confidence=1.0, band="above"),
    ]
    assert assign_measures(detections, [0.25, 0.5, 0.75]) == [1, 2, 3, 4]


def test_assign_measures_puts_all_in_measure_1_when_no_barlines() -> None:
    detections = [
        ChordDetection(symbol_raw="C", symbol="C", x_fraction=0.1, confidence=1.0, band="above"),
        ChordDetection(symbol_raw="G", symbol="G", x_fraction=0.9, confidence=1.0, band="above"),
    ]
    assert assign_measures(detections, []) == [1, 1]


def test_assign_measures_ignores_leading_and_terminal_barlines() -> None:
    detections = [
        ChordDetection(symbol_raw="C", symbol="C", x_fraction=0.1, confidence=1.0, band="above"),
        ChordDetection(symbol_raw="G", symbol="G", x_fraction=0.6, confidence=1.0, band="above"),
        ChordDetection(symbol_raw="F", symbol="F", x_fraction=0.99, confidence=1.0, band="above"),
    ]

    assert measure_boundaries([0.02, 0.5, 0.98]) == [0.02, 0.5, 0.98]
    assert measures_in_system([0.02, 0.5, 0.98]) == 2
    assert assign_measures(detections, [0.02, 0.5, 0.98]) == [1, 2, 2]


def test_measures_in_system_counts_fencepost() -> None:
    assert measures_in_system([]) == 1
    assert measures_in_system([0.3, 0.6]) == 3


def test_measures_in_system_subtracts_leading_barline() -> None:
    # Leftmost barline within the leading 5% is the *start* of measure 1, not
    # a fence between measures: 2 barlines, not 3, fence 2 measures.
    assert measures_in_system([0.02, 0.5]) == 2


def test_measures_in_system_subtracts_terminal_barline() -> None:
    # Rightmost barline past 0.97 is the closing barline of the last measure,
    # not a fence opening another.
    assert measures_in_system([0.5, 0.98]) == 2


def test_measures_in_system_subtracts_both_when_present() -> None:
    assert measures_in_system([0.02, 0.5, 0.98]) == 2
