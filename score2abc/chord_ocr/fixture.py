from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

from score2abc.chord_ocr.base import ChordDetection, ChordExtractionRequest
from score2abc.chord_ocr.prompt import PROMPT_VERSION


class FixtureNotFoundError(FileNotFoundError):
    """Raised when no committed fixture matches a ChordOCR request."""


class FixtureChordOCR:
    """Replay-only ChordOCR backed by committed JSON fixtures.

    Used in tests and as the default backend when --use-vlm is false so the
    pipeline stays fully hermetic.
    """

    model_id = "fixture"

    def __init__(self, fixtures_dir: Path) -> None:
        self._fixtures_dir = fixtures_dir

    @property
    def prompt_version(self) -> str:
        return PROMPT_VERSION

    def extract(self, request: ChordExtractionRequest) -> Sequence[ChordDetection]:
        key = fixture_key(
            request.image_path,
            prompt_version=self.prompt_version,
            model_id=self.model_id,
        )
        fixture_path = self._fixtures_dir / f"{key}.json"
        if not fixture_path.exists():
            raise FixtureNotFoundError(
                f"No chord-OCR fixture for image {request.image_path} (key={key})"
            )
        return read_fixture(fixture_path, default_band=request.band)


def fixture_key(image_path: Path, *, prompt_version: str, model_id: str) -> str:
    """Stable content-addressed key for a chord-crop / prompt / model tuple."""
    digest = hashlib.sha256()
    digest.update(image_path.read_bytes())
    digest.update(b"\x1f")
    digest.update(prompt_version.encode("utf-8"))
    digest.update(b"\x1f")
    digest.update(model_id.encode("utf-8"))
    return digest.hexdigest()[:16]


def write_fixture(
    path: Path,
    *,
    image_path: Path,
    prompt_version: str,
    model_id: str,
    detections: Sequence[ChordDetection],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prompt_version": prompt_version,
        "model_id": model_id,
        "image_path": str(image_path),
        "detections": [asdict(detection) for detection in detections],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def read_fixture(path: Path, *, default_band: str) -> list[ChordDetection]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _load_detections(payload["detections"], default_band=default_band)


def _load_detections(raw: Iterable[dict], *, default_band: str) -> list[ChordDetection]:
    return [
        ChordDetection(
            symbol_raw=item["symbol_raw"],
            symbol=item["symbol"],
            x_fraction=float(item["x_fraction"]),
            confidence=float(item["confidence"]),
            band=item.get("band", default_band),
        )
        for item in raw
    ]
