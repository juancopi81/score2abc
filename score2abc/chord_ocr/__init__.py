"""Chord-symbol OCR contracts, normalization, and backends."""

from score2abc.chord_ocr.base import (
    Band,
    ChordDetection,
    ChordExtractionRequest,
    ChordOCR,
)
from score2abc.chord_ocr.cached import CachedChordOCR
from score2abc.chord_ocr.fixture import (
    FixtureChordOCR,
    FixtureNotFoundError,
    fixture_key,
    read_fixture,
    write_fixture,
)
from score2abc.chord_ocr.gemini import GeminiChordOCR
from score2abc.chord_ocr.normalize import normalize_chord_symbol
from score2abc.chord_ocr.prompt import PROMPT_VERSION, RESPONSE_SCHEMA, SYSTEM_PROMPT

__all__ = [
    "Band",
    "ChordDetection",
    "ChordExtractionRequest",
    "ChordOCR",
    "CachedChordOCR",
    "GeminiChordOCR",
    "FixtureChordOCR",
    "FixtureNotFoundError",
    "fixture_key",
    "read_fixture",
    "write_fixture",
    "normalize_chord_symbol",
    "PROMPT_VERSION",
    "RESPONSE_SCHEMA",
    "SYSTEM_PROMPT",
]
