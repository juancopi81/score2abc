from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Sequence

Band = Literal["above", "below"]


@dataclass(frozen=True)
class ChordDetection:
    """A single chord symbol detected in a chord-band crop."""

    symbol_raw: str
    symbol: str
    x_fraction: float
    confidence: float
    band: Band


@dataclass(frozen=True)
class ChordExtractionRequest:
    """Input passed to a ChordOCR backend for one chord-band crop."""

    image_path: Path
    band: Band
    system_index: int
    rhythm_hint: str | None = None
    key_hint: str | None = None
    time_signature_hint: str | None = None


class ChordOCR(Protocol):
    """Protocol implemented by every chord-OCR backend."""

    @property
    def model_id(self) -> str: ...

    @property
    def prompt_version(self) -> str: ...

    def extract(self, request: ChordExtractionRequest) -> Sequence[ChordDetection]: ...
