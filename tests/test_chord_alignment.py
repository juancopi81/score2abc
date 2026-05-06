from pathlib import Path

from PIL import Image, ImageDraw

from score2abc.chord_ocr import ChordDetection
from score2abc.chord_ocr.alignment import (
    assign_measures,
    detect_barlines,
    measures_in_system,
)


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
