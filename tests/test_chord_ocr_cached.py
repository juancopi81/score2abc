from pathlib import Path
from typing import Sequence

from score2abc.chord_ocr import (
    CachedChordOCR,
    ChordDetection,
    ChordExtractionRequest,
)
from score2abc.chord_ocr.prompt import PROMPT_VERSION


class _CountingOCR:
    model_id = "fake-model"
    prompt_version = PROMPT_VERSION

    def __init__(self, detections: Sequence[ChordDetection]) -> None:
        self._detections = list(detections)
        self.calls = 0

    def extract(self, request: ChordExtractionRequest) -> Sequence[ChordDetection]:
        self.calls += 1
        return list(self._detections)


def _make_fake_image(path: Path, content: bytes = b"fake-image") -> Path:
    path.write_bytes(content)
    return path


def test_cached_chord_ocr_writes_on_miss_reads_on_hit(tmp_path: Path) -> None:
    image = _make_fake_image(tmp_path / "crop.png")
    detections = [
        ChordDetection(symbol_raw="Em", symbol="Em", x_fraction=0.1, confidence=0.9, band="above")
    ]
    inner = _CountingOCR(detections)
    cached = CachedChordOCR(inner, tmp_path / "cache")
    request = ChordExtractionRequest(image_path=image, band="above", system_index=1)

    first = list(cached.extract(request))
    assert first == detections
    assert inner.calls == 1
    assert (tmp_path / "cache").exists()

    second = list(cached.extract(request))
    assert second == detections
    assert inner.calls == 1


def test_cached_chord_ocr_invalidates_on_model_change(tmp_path: Path) -> None:
    image = _make_fake_image(tmp_path / "crop.png")
    detections_a = [
        ChordDetection(symbol_raw="Em", symbol="Em", x_fraction=0.1, confidence=0.9, band="above")
    ]
    detections_b = [
        ChordDetection(symbol_raw="B7", symbol="B7", x_fraction=0.5, confidence=0.8, band="above")
    ]

    first_inner = _CountingOCR(detections_a)
    first_cache = CachedChordOCR(first_inner, tmp_path / "cache")

    class _OtherOCR(_CountingOCR):
        model_id = "other-model"

    second_inner = _OtherOCR(detections_b)
    second_cache = CachedChordOCR(second_inner, tmp_path / "cache")

    request = ChordExtractionRequest(image_path=image, band="above", system_index=1)
    assert list(first_cache.extract(request)) == detections_a
    assert list(second_cache.extract(request)) == detections_b
    assert first_inner.calls == 1
    assert second_inner.calls == 1
