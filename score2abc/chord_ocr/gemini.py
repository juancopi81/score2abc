from __future__ import annotations

import base64
import json
import mimetypes
import os
from typing import Any, Sequence

from score2abc.chord_ocr.base import Band, ChordDetection, ChordExtractionRequest
from score2abc.chord_ocr.normalize import normalize_chord_symbol
from score2abc.chord_ocr.prompt import PROMPT_VERSION, RESPONSE_SCHEMA, SYSTEM_PROMPT

DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiChordOCR:
    """Gemini-backed ChordOCR. Use a live client in production, a fake in tests.

    The adapter speaks the google-genai dict content API so tests can mock the
    client with a plain object exposing ``models.generate_content(...)``. The
    SDK is imported lazily only when no client is injected.
    """

    def __init__(
        self,
        *,
        client: Any = None,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self._model = model
        if client is not None:
            self._client = client
        else:
            resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
            if not resolved_key:
                raise RuntimeError("GEMINI_API_KEY is not set; provide api_key or inject a client.")
            self._client = _build_default_client(resolved_key)

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def prompt_version(self) -> str:
        return PROMPT_VERSION

    def extract(self, request: ChordExtractionRequest) -> Sequence[ChordDetection]:
        image_bytes = request.image_path.read_bytes()
        mime_type, _ = mimetypes.guess_type(str(request.image_path))
        mime_type = mime_type or "image/png"

        contents = [
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(image_bytes).decode("ascii"),
                        }
                    },
                    {"text": _format_user_message(request)},
                ],
            }
        ]
        config = {
            "system_instruction": SYSTEM_PROMPT,
            "response_mime_type": "application/json",
            "response_schema": RESPONSE_SCHEMA,
            "temperature": 0.0,
        }

        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )
        payload = json.loads(_response_text(response))
        return _parse_detections(payload, band=request.band)


def _format_user_message(request: ChordExtractionRequest) -> str:
    lines = [
        f"Chord band: {request.band}.",
        f"Staff system index: {request.system_index}.",
    ]
    if request.rhythm_hint:
        lines.append(f"Rhythm: {request.rhythm_hint}.")
    if request.time_signature_hint:
        lines.append(f"Time signature: {request.time_signature_hint}.")
    if request.key_hint:
        lines.append(f"Key hint: {request.key_hint}.")
    return "\n".join(lines)


def _build_default_client(api_key: str) -> Any:
    from google import genai  # type: ignore[import-not-found]

    return genai.Client(api_key=api_key)


def _response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text:
        return text
    try:
        parts = response.candidates[0].content.parts
    except (AttributeError, IndexError) as exc:
        raise RuntimeError(f"Gemini response had no parseable text: {exc}") from exc
    joined = "".join(getattr(part, "text", "") or "" for part in parts)
    if not joined:
        raise RuntimeError("Gemini response contained no text parts")
    return joined


def _parse_detections(payload: Any, *, band: Band) -> list[ChordDetection]:
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object from Gemini, got {type(payload).__name__}")
    raw_detections = payload.get("detections") or []
    result: list[ChordDetection] = []
    for item in raw_detections:
        symbol_raw = str(item.get("symbol", "")).strip()
        if not symbol_raw:
            continue
        normalized = normalize_chord_symbol(symbol_raw)
        if not normalized:
            continue
        result.append(
            ChordDetection(
                symbol_raw=symbol_raw,
                symbol=normalized,
                x_fraction=_clamp01(item.get("x_fraction", 0.0)),
                confidence=_clamp01(item.get("confidence", 0.0)),
                band=band,
            )
        )
    return result


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))
