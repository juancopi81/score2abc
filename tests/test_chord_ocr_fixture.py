from pathlib import Path

import pytest

from score2abc.chord_ocr import (
    ChordDetection,
    ChordExtractionRequest,
    FixtureChordOCR,
    FixtureNotFoundError,
    fixture_key,
    write_fixture,
)
from score2abc.chord_ocr.prompt import PROMPT_VERSION


def _make_fake_image(path: Path, content: bytes = b"fake-image") -> Path:
    path.write_bytes(content)
    return path


def test_fixture_key_is_deterministic(tmp_path: Path) -> None:
    image = _make_fake_image(tmp_path / "crop.png")
    first = fixture_key(image, prompt_version="v1", model_id="fixture")
    second = fixture_key(image, prompt_version="v1", model_id="fixture")
    assert first == second


def test_fixture_key_changes_with_prompt_version(tmp_path: Path) -> None:
    image = _make_fake_image(tmp_path / "crop.png")
    v1 = fixture_key(image, prompt_version="v1", model_id="fixture")
    v2 = fixture_key(image, prompt_version="v2", model_id="fixture")
    assert v1 != v2


def test_fixture_key_changes_with_model_id(tmp_path: Path) -> None:
    image = _make_fake_image(tmp_path / "crop.png")
    fix = fixture_key(image, prompt_version="v1", model_id="fixture")
    gem = fixture_key(image, prompt_version="v1", model_id="gemini-2.5-flash")
    assert fix != gem


def test_fixture_key_changes_with_image_bytes(tmp_path: Path) -> None:
    a = _make_fake_image(tmp_path / "a.png", b"aaa")
    b = _make_fake_image(tmp_path / "b.png", b"bbb")
    assert fixture_key(a, prompt_version="v1", model_id="fixture") != fixture_key(
        b, prompt_version="v1", model_id="fixture"
    )


def test_fixture_chord_ocr_round_trip(tmp_path: Path) -> None:
    fixtures_dir = tmp_path / "fixtures"
    image = _make_fake_image(tmp_path / "crop.png")
    detections = [
        ChordDetection(symbol_raw="Em", symbol="Em", x_fraction=0.1, confidence=0.9, band="above"),
        ChordDetection(symbol_raw="B7", symbol="B7", x_fraction=0.5, confidence=0.85, band="above"),
    ]
    key = fixture_key(image, prompt_version=PROMPT_VERSION, model_id="fixture")
    write_fixture(
        fixtures_dir / f"{key}.json",
        image_path=image,
        prompt_version=PROMPT_VERSION,
        model_id="fixture",
        detections=detections,
    )

    ocr = FixtureChordOCR(fixtures_dir)
    request = ChordExtractionRequest(image_path=image, band="above", system_index=1)
    result = list(ocr.extract(request))
    assert result == detections


def test_fixture_chord_ocr_missing_raises(tmp_path: Path) -> None:
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    image = _make_fake_image(tmp_path / "crop.png")
    ocr = FixtureChordOCR(fixtures_dir)
    with pytest.raises(FixtureNotFoundError):
        ocr.extract(ChordExtractionRequest(image_path=image, band="above", system_index=1))
