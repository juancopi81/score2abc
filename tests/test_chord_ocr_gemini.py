import base64
import json
from pathlib import Path
from typing import Any, List

import pytest

from score2abc.chord_ocr import ChordExtractionRequest
from score2abc.chord_ocr.gemini import GeminiChordOCR


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _RecordingModels:
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.calls: List[dict[str, Any]] = []

    def generate_content(self, *, model: str, contents: list, config: dict) -> _FakeResponse:
        self.calls.append({"model": model, "contents": contents, "config": config})
        return _FakeResponse(self._response_text)


class _RecordingClient:
    def __init__(self, response_text: str) -> None:
        self.models = _RecordingModels(response_text)


def _make_image(path: Path, content: bytes = b"\x89PNGfake-bytes") -> Path:
    path.write_bytes(content)
    return path


def test_gemini_chord_ocr_parses_structured_response(tmp_path: Path) -> None:
    image = _make_image(tmp_path / "crop.png")
    response_text = json.dumps(
        {
            "detections": [
                {"symbol": "Em", "x_fraction": 0.1, "confidence": 0.95},
                {"symbol": "B 7", "x_fraction": 0.55, "confidence": 0.8},
            ]
        }
    )
    client = _RecordingClient(response_text)
    ocr = GeminiChordOCR(client=client, model="gemini-2.5-flash")

    request = ChordExtractionRequest(
        image_path=image,
        band="above",
        system_index=1,
        rhythm_hint="Pasillo",
        key_hint="Em",
        time_signature_hint="3/4",
    )
    result = list(ocr.extract(request))

    assert [d.symbol for d in result] == ["Em", "B7"]
    assert result[1].symbol_raw == "B 7"
    assert all(d.band == "above" for d in result)

    call = client.models.calls[0]
    assert call["model"] == "gemini-2.5-flash"
    assert call["config"]["response_mime_type"] == "application/json"
    assert call["config"]["temperature"] == 0.0
    assert call["config"]["system_instruction"]

    user_text = call["contents"][0]["parts"][1]["text"]
    assert "Pasillo" in user_text
    assert "3/4" in user_text
    assert "Em" in user_text

    inline = call["contents"][0]["parts"][0]["inline_data"]
    assert inline["mime_type"] == "image/png"
    assert base64.b64decode(inline["data"]) == image.read_bytes()


def test_gemini_chord_ocr_drops_empty_symbols(tmp_path: Path) -> None:
    image = _make_image(tmp_path / "crop.png")
    response_text = json.dumps(
        {
            "detections": [
                {"symbol": "", "x_fraction": 0.1, "confidence": 0.5},
                {"symbol": "  ", "x_fraction": 0.2, "confidence": 0.5},
                {"symbol": "C", "x_fraction": 0.3, "confidence": 0.9},
            ]
        }
    )
    ocr = GeminiChordOCR(client=_RecordingClient(response_text))
    request = ChordExtractionRequest(image_path=image, band="above", system_index=1)
    result = list(ocr.extract(request))
    assert [d.symbol for d in result] == ["C"]


def test_gemini_chord_ocr_clamps_out_of_range_values(tmp_path: Path) -> None:
    image = _make_image(tmp_path / "crop.png")
    response_text = json.dumps(
        {
            "detections": [
                {"symbol": "C", "x_fraction": -0.1, "confidence": 1.5},
                {"symbol": "G", "x_fraction": 2.0, "confidence": -0.2},
            ]
        }
    )
    ocr = GeminiChordOCR(client=_RecordingClient(response_text))
    request = ChordExtractionRequest(image_path=image, band="below", system_index=1)
    result = list(ocr.extract(request))

    assert [(d.x_fraction, d.confidence) for d in result] == [(0.0, 1.0), (1.0, 0.0)]
    assert result[0].band == "below"


def test_gemini_chord_ocr_requires_api_key_when_no_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiChordOCR()
