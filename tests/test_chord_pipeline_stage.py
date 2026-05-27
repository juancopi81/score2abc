from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import pytest
from PIL import Image, ImageDraw

from score2abc.chord_ocr import (
    CachedChordOCR,
    ChordDetection,
    ChordExtractionRequest,
    FixtureChordOCR,
    fixture_key,
    write_fixture,
)
from score2abc.chord_ocr.gemini import DEFAULT_MODEL
from score2abc.chord_ocr.prompt import PROMPT_VERSION
from score2abc.chords import build_chord_ocr, extract_chords_for_systems
from score2abc.pipeline import _build_stub_events
from score2abc.schemas import WorkMetadata


class _ScriptedChordOCR:
    """Test double that returns pre-programmed detections per image path."""

    model_id = "scripted"
    prompt_version = "v-test"

    def __init__(self, script: dict[str, list[ChordDetection]]) -> None:
        self._script = script
        self.requests: list[ChordExtractionRequest] = []

    def extract(self, request: ChordExtractionRequest) -> Sequence[ChordDetection]:
        self.requests.append(request)
        key = request.image_path.name
        return list(self._script.get(key, []))


def _draw_system_with_barlines(path: Path, fractions: list[float]) -> Path:
    image = Image.new("L", (800, 160), color=255)
    draw = ImageDraw.Draw(image)
    staff_top, staff_bottom = 50, 130
    spacing = (staff_bottom - staff_top) // 4
    for line in range(5):
        y = staff_top + line * spacing
        draw.line([(0, y), (image.width - 1, y)], fill=0, width=1)
    for fraction in fractions:
        x = int(fraction * image.width)
        draw.line([(x, staff_top), (x, staff_bottom)], fill=0, width=2)
    image.save(path)
    return path


def _blank_crop(path: Path) -> Path:
    Image.new("L", (800, 40), color=255).save(path)
    return path


def _metadata() -> WorkMetadata:
    return WorkMetadata(
        title="Demo",
        composer="Composer",
        rhythm="Pasillo",
        time_signature="3/4",
        key_hint="Em",
    )


def test_build_stub_events_preserves_empty_extracted_chords() -> None:
    item = SimpleNamespace(metadata=_metadata())

    events = _build_stub_events(item, chords=[])

    assert events["chords"] == []


def test_extract_chords_for_systems_offsets_measures_across_systems(
    tmp_path: Path,
) -> None:
    system_one = _draw_system_with_barlines(tmp_path / "system_001.png", [0.5])  # 2 measures
    system_two = _draw_system_with_barlines(tmp_path / "system_002.png", [0.33, 0.66])  # 3 measures
    above_one = _blank_crop(tmp_path / "chord_region_above_001.png")
    below_one = _blank_crop(tmp_path / "chord_region_below_001.png")
    above_two = _blank_crop(tmp_path / "chord_region_above_002.png")
    below_two = _blank_crop(tmp_path / "chord_region_below_002.png")

    script = {
        above_one.name: [
            ChordDetection(
                symbol_raw="Em",
                symbol="Em",
                x_fraction=0.1,
                confidence=0.9,
                band="above",
            ),
            ChordDetection(
                symbol_raw="B7",
                symbol="B7",
                x_fraction=0.8,
                confidence=0.8,
                band="above",
            ),
        ],
        below_one.name: [],
        above_two.name: [
            ChordDetection(
                symbol_raw="C",
                symbol="C",
                x_fraction=0.2,
                confidence=0.95,
                band="above",
            ),
            ChordDetection(
                symbol_raw="G",
                symbol="G",
                x_fraction=0.5,
                confidence=0.95,
                band="above",
            ),
            ChordDetection(
                symbol_raw="Am",
                symbol="Am",
                x_fraction=0.8,
                confidence=0.9,
                band="above",
            ),
        ],
        below_two.name: [
            ChordDetection(
                symbol_raw="F", symbol="F", x_fraction=0.4, confidence=0.6, band="below"
            ),
        ],
    }
    ocr = _ScriptedChordOCR(script)

    payload = extract_chords_for_systems(
        ocr=ocr,
        system_crops=[system_one, system_two],
        chord_crops_above=[above_one, above_two],
        chord_crops_below=[below_one, below_two],
        metadata=_metadata(),
    )

    # 2 measures in system 1 + 3 measures in system 2 = 5 global measures.
    assert payload["total_measures"] == 5
    assert payload["provider"] == "_ScriptedChordOCR"
    assert payload["model_id"] == "scripted"
    assert payload["prompt_version"] == "v-test"

    chords = payload["chords"]
    # Sorted by (measure, source_band, symbol). System 2's F (below) lands in global measure 4
    # (system-local measure 2 under the 0.33/0.66 barlines, plus offset 2 from system 1).
    measures_symbols = [
        (entry["measure"], entry["symbol"], entry["source_band"]) for entry in chords
    ]
    assert measures_symbols == [
        (1, "Em", "above"),
        (2, "B7", "above"),
        (3, "C", "above"),
        (4, "G", "above"),
        (4, "F", "below"),
        (5, "Am", "above"),
    ]
    for entry in chords:
        assert entry["onset_beats"] == 0.0
        assert 0.0 <= entry["confidence"] <= 1.0

    system_one_payload = payload["systems"][0]
    assert system_one_payload["system_index"] == 1
    assert system_one_payload["measure_count"] == 2
    assert system_one_payload["measure_offset"] == 0
    assert len(system_one_payload["barlines"]) == 1

    system_two_payload = payload["systems"][1]
    assert system_two_payload["measure_offset"] == 2
    assert system_two_payload["measure_count"] == 3
    assert len(system_two_payload["barlines"]) == 2


def test_extract_chords_for_systems_assigns_with_normalized_barline_boundaries(
    tmp_path: Path,
) -> None:
    system = _draw_system_with_barlines(tmp_path / "system_001.png", [0.02, 0.5, 0.98])
    above = _blank_crop(tmp_path / "chord_region_above_001.png")
    below = _blank_crop(tmp_path / "chord_region_below_001.png")
    ocr = _ScriptedChordOCR(
        {
            above.name: [
                ChordDetection(
                    symbol_raw="C",
                    symbol="C",
                    x_fraction=0.1,
                    confidence=0.9,
                    band="above",
                ),
                ChordDetection(
                    symbol_raw="G",
                    symbol="G",
                    x_fraction=0.6,
                    confidence=0.9,
                    band="above",
                ),
                ChordDetection(
                    symbol_raw="F",
                    symbol="F",
                    x_fraction=0.99,
                    confidence=0.9,
                    band="above",
                ),
            ],
            below.name: [],
        }
    )

    payload = extract_chords_for_systems(
        ocr=ocr,
        system_crops=[system],
        chord_crops_above=[above],
        chord_crops_below=[below],
        metadata=_metadata(),
    )

    assert payload["total_measures"] == 2
    assert payload["systems"][0]["measure_count"] == 2
    assert [entry["measure"] for entry in payload["chords"]] == [1, 2, 2]
    assert [entry["system_local_measure"] for entry in payload["systems"][0]["detections"]] == [
        1,
        2,
        2,
    ]


def test_extract_chords_for_systems_passes_metadata_hints(tmp_path: Path) -> None:
    system = _draw_system_with_barlines(tmp_path / "system_001.png", [])
    above = _blank_crop(tmp_path / "chord_region_above_001.png")
    below = _blank_crop(tmp_path / "chord_region_below_001.png")
    ocr = _ScriptedChordOCR({})

    extract_chords_for_systems(
        ocr=ocr,
        system_crops=[system],
        chord_crops_above=[above],
        chord_crops_below=[below],
        metadata=_metadata(),
    )

    assert len(ocr.requests) == 2
    above_request, below_request = ocr.requests
    assert above_request.band == "above"
    assert below_request.band == "below"
    for request in ocr.requests:
        assert request.system_index == 1
        assert request.rhythm_hint == "Pasillo"
        assert request.time_signature_hint == "3/4"
        assert request.key_hint == "Em"


def test_extract_chords_for_systems_handles_missing_fixture_as_empty(
    tmp_path: Path,
) -> None:
    system = _draw_system_with_barlines(tmp_path / "system_001.png", [])
    above = _blank_crop(tmp_path / "chord_region_above_001.png")
    below = _blank_crop(tmp_path / "chord_region_below_001.png")
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    ocr = FixtureChordOCR(fixtures_dir)
    caplog_logger = logging.getLogger("test.extract.missing")

    payload = extract_chords_for_systems(
        ocr=ocr,
        system_crops=[system],
        chord_crops_above=[above],
        chord_crops_below=[below],
        metadata=_metadata(),
        logger=caplog_logger,
    )

    assert payload["chords"] == []
    assert payload["systems"][0]["detections"] == []
    assert payload["total_measures"] == 1


def test_extract_chords_for_systems_rejects_mismatched_lengths(tmp_path: Path) -> None:
    system = _draw_system_with_barlines(tmp_path / "system_001.png", [])
    above = _blank_crop(tmp_path / "chord_region_above_001.png")
    ocr = _ScriptedChordOCR({})

    with pytest.raises(ValueError, match="equal length"):
        extract_chords_for_systems(
            ocr=ocr,
            system_crops=[system],
            chord_crops_above=[above],
            chord_crops_below=[],
            metadata=_metadata(),
        )


def test_build_chord_ocr_without_vlm_returns_fixture_backend(tmp_path: Path) -> None:
    ocr = build_chord_ocr(
        use_vlm=False,
        fixtures_dir=tmp_path / "fixtures",
        cache_dir=tmp_path / "cache",
    )
    assert isinstance(ocr, FixtureChordOCR)
    assert ocr.model_id == DEFAULT_MODEL


def test_build_chord_ocr_without_vlm_replays_promoted_live_fixture(
    tmp_path: Path,
) -> None:
    fixtures_dir = tmp_path / "fixtures"
    image = _blank_crop(tmp_path / "chord_region_above_001.png")
    detections = [
        ChordDetection(symbol_raw="Em", symbol="Em", x_fraction=0.1, confidence=0.9, band="above")
    ]
    key = fixture_key(image, prompt_version=PROMPT_VERSION, model_id=DEFAULT_MODEL)
    write_fixture(
        fixtures_dir / f"{key}.json",
        image_path=image,
        prompt_version=PROMPT_VERSION,
        model_id=DEFAULT_MODEL,
        detections=detections,
    )

    ocr = build_chord_ocr(
        use_vlm=False,
        fixtures_dir=fixtures_dir,
        cache_dir=tmp_path / "cache",
    )
    request = ChordExtractionRequest(image_path=image, band="above", system_index=1)
    assert list(ocr.extract(request)) == detections


def test_build_chord_ocr_with_vlm_wraps_gemini_in_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import score2abc.chord_ocr.gemini as gemini_module

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(gemini_module, "_build_default_client", lambda api_key: object())

    ocr = build_chord_ocr(
        use_vlm=True,
        fixtures_dir=tmp_path / "fixtures",
        cache_dir=tmp_path / "cache",
    )
    assert isinstance(ocr, CachedChordOCR)
    assert ocr.model_id
