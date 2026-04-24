from __future__ import annotations

from pathlib import Path
from typing import Sequence

from score2abc.chord_ocr.base import ChordDetection, ChordExtractionRequest, ChordOCR
from score2abc.chord_ocr.fixture import fixture_key, read_fixture, write_fixture


class CachedChordOCR:
    """Disk-cache wrapper around any ChordOCR backend.

    Memoizes responses keyed by (image bytes, prompt_version, model_id). On a
    miss it delegates to the wrapped backend and persists the response as a
    fixture-compatible JSON file so cache entries can be promoted to committed
    test fixtures simply by copying the file into tests/fixtures/vlm/.
    """

    def __init__(self, inner: ChordOCR, cache_dir: Path) -> None:
        self._inner = inner
        self._cache_dir = cache_dir

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def prompt_version(self) -> str:
        return self._inner.prompt_version

    def extract(self, request: ChordExtractionRequest) -> Sequence[ChordDetection]:
        key = fixture_key(
            request.image_path,
            prompt_version=self.prompt_version,
            model_id=self.model_id,
        )
        cache_path = self._cache_dir / f"{key}.json"
        try:
            return read_fixture(cache_path)
        except FileNotFoundError:
            pass

        detections = list(self._inner.extract(request))
        write_fixture(
            cache_path,
            image_path=request.image_path,
            prompt_version=self.prompt_version,
            model_id=self.model_id,
            detections=detections,
        )
        return detections
