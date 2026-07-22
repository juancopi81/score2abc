from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

InputKind = Literal[
    "raw",
    "staff",
    "staff_overlay",
    "pitch_ruler",
    "pitch_ruler_soft",
    "pitch_ruler_panel",
    "neighbor_context",
]
INPUT_KINDS: tuple[InputKind, ...] = (
    "raw",
    "staff",
    "staff_overlay",
    "pitch_ruler",
    "pitch_ruler_soft",
    "pitch_ruler_panel",
    "neighbor_context",
)
VLMProvider = Literal["gemini", "openai"]
VLM_PROVIDERS: tuple[VLMProvider, ...] = ("gemini", "openai")
TranscriptionMode = Literal["pitch", "staff_position", "notehead_y"]
TRANSCRIPTION_MODES: tuple[TranscriptionMode, ...] = ("pitch", "staff_position", "notehead_y")

DEFAULT_MELODY_VLM_MODEL = "gemini-3.1-flash-lite"
DEFAULT_OPENAI_MELODY_VLM_MODEL = "gpt-5.5"
MELODY_VLM_PROMPT_VERSION = "melody-vlm-v0"
STAFF_POSITION_PROMPT_VERSION = "melody-vlm-staff-position-v0"
NOTEHEAD_Y_PROMPT_VERSION = "melody-vlm-notehead-y-v0"
PROMPT_CONTEXT_KEYS = (
    "title",
    "rhythm",
    "clef_hint",
    "time_signature_hint",
    "key_hint",
    "system_index",
    "system_measure_index",
    "display_measure_number",
    "allow_pickup",
    "expected_measure_beats",
)

PITCH_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "kind": {"type": "string"},
                    "pitch": {"type": "string"},
                    "duration": {"type": "string"},
                    "accidental": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["kind", "pitch", "duration", "accidental", "confidence"],
            },
        },
        "comments": {"type": "string"},
    },
    "required": ["items", "comments"],
}

PITCH_RULER_SOFT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "kind": {"type": "string"},
                    "pitch": {"type": "string"},
                    "duration": {"type": "string"},
                    "accidental": {"type": "string"},
                    "confidence": {"type": "number"},
                    "evidence": {"type": "string"},
                },
                "required": [
                    "kind",
                    "pitch",
                    "duration",
                    "accidental",
                    "confidence",
                    "evidence",
                ],
            },
        },
        "comments": {"type": "string"},
        "overall_confidence": {"type": "number"},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["items", "comments", "overall_confidence", "uncertainties"],
}

STAFF_POSITION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "kind": {"type": "string"},
                    "notehead_x_fraction": {"type": "number"},
                    "staff_position": {"type": "integer"},
                    "duration": {"type": "string"},
                    "accidental": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "kind",
                    "notehead_x_fraction",
                    "staff_position",
                    "duration",
                    "accidental",
                    "confidence",
                ],
            },
        },
        "comments": {"type": "string"},
    },
    "required": ["items", "comments"],
}

NOTEHEAD_Y_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "kind": {"type": "string"},
                    "notehead_x_fraction": {"type": "number"},
                    "notehead_y_fraction": {"type": "number"},
                    "duration": {"type": "string"},
                    "accidental": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "kind",
                    "notehead_x_fraction",
                    "notehead_y_fraction",
                    "duration",
                    "accidental",
                    "confidence",
                ],
            },
        },
        "comments": {"type": "string"},
    },
    "required": ["items", "comments"],
}

SYSTEM_PROMPT = """\
You transcribe handwritten single-staff melody from cropped score images.
Return JSON only. Do not include Markdown or prose outside JSON.
Ignore colored overlay guide lines if present; they are not music.
Transcribe only the melody notes and rests visible in the single measure crop.
"""

PITCH_RULER_SYSTEM_PROMPT = """\
You transcribe handwritten single-staff melody from cropped score images.
Return JSON only. Do not include Markdown or prose outside JSON.
The image includes pitch labels on the left and horizontal pitch guide lines.
The labels and guide lines are visual aids, not music.
For each notehead, choose the nearest labeled pitch guide and return that pitch.
Transcribe only the melody notes and rests visible in the single measure crop.
Include visible or likely rests, including initial silence before the first note.
"""

PITCH_RULER_SOFT_SYSTEM_PROMPT = """\
You transcribe handwritten single-staff melody from a cropped score image.
Return JSON only. Do not include Markdown or prose outside JSON.
The image includes a soft pitch ruler:
- pitch labels on the left name the horizontal pitch levels
- faint gray dotted horizontal guides extend from those labels
- these labels and guides are visual aids, not music
- the black handwritten ink is the only source of musical notes/rests
Do not count printed labels, gray guide marks, red staff lines, or intersections between helpers
and handwriting as noteheads. First find black handwritten noteheads from left to right, then map
each notehead to the nearest pitch label. You must provide your best-effort transcription; do not
refuse because the image is ambiguous. Use confidence and evidence fields to mark uncertainty.
Include visible or likely rests, including initial silence before the first note.
"""

PITCH_RULER_PANEL_SYSTEM_PROMPT = """\
You transcribe handwritten single-staff melody from a cropped score image.
Return JSON only. Do not include Markdown or prose outside JSON.
The image has two visual regions:
- the left gutter is a printed pitch reference panel with labels and short tick marks
- the right side is the clean handwritten music crop
The left pitch reference panel is not music. Use it only as a vertical ruler: noteheads in the
right-side music crop should be mapped to the nearest pitch label at the same height. The black
handwritten ink in the music crop is the only source of musical notes/rests. Do not count pitch
labels, tick marks, the gutter divider, barlines, stems alone, slurs, helper marks, or guide
graphics as noteheads. You must provide your best-effort transcription; do not refuse because
the image is ambiguous. Use confidence and evidence fields to mark uncertainty. Include visible
or likely rests, including initial silence before the first note.
"""

STAFF_POSITION_SYSTEM_PROMPT = """\
You identify handwritten noteheads on single-staff melody crop images.
Return JSON only. Do not include Markdown or prose outside JSON.
Ignore colored overlay guide lines if present; they are not music.
Transcribe only the melody notes and rests visible in the single measure crop.
Use staff positions instead of pitch names:
- 0 is the bottom staff line
- 1 is the space above the bottom line
- 2 is the second staff line
- 3 is the next space
- 4 is the middle staff line
- 5 is the next space
- 6 is the fourth staff line
- 7 is the next space
- 8 is the top staff line
- values below 0 are below the staff; values above 8 are above the staff
"""

NOTEHEAD_Y_SYSTEM_PROMPT = """\
You identify handwritten noteheads on single-staff melody crop images.
Return JSON only. Do not include Markdown or prose outside JSON.
Ignore colored overlay guide lines if present; they are not music.
Transcribe only the melody notes and rests visible in the single measure crop.
Return each notehead center using image fractions:
- notehead_x_fraction is 0.0 at the left edge and 1.0 at the right edge
- notehead_y_fraction is 0.0 at the top edge and 1.0 at the bottom edge
Do not convert notehead height into pitch or staff position.
"""


@dataclass(frozen=True)
class MelodyVLMItem:
    kind: str
    pitch: str | None
    duration: str
    accidental: str | None
    confidence: float
    notehead_x_fraction: float | None = None
    notehead_y_fraction: float | None = None
    staff_position: int | None = None
    evidence: str | None = None


@dataclass(frozen=True)
class MelodyVLMTranscription:
    items: tuple[MelodyVLMItem, ...]
    comments: str
    raw_response: str
    overall_confidence: float | None = None
    uncertainties: tuple[str, ...] = ()


@dataclass(frozen=True)
class MelodyVLMRequest:
    image_path: Path
    context: dict[str, Any]
    input_kind: InputKind
    transcription_mode: TranscriptionMode = "pitch"


class GeminiMelodyVLM:
    """Gemini-backed one-measure melody transcription client for spike tooling."""

    def __init__(
        self,
        *,
        client: Any = None,
        api_key: str | None = None,
        model: str = DEFAULT_MELODY_VLM_MODEL,
        transcription_mode: TranscriptionMode = "pitch",
    ) -> None:
        self._model = model
        self._transcription_mode = transcription_mode
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
        if self._transcription_mode == "notehead_y":
            return NOTEHEAD_Y_PROMPT_VERSION
        if self._transcription_mode == "staff_position":
            return STAFF_POSITION_PROMPT_VERSION
        return MELODY_VLM_PROMPT_VERSION

    def transcribe(self, request: MelodyVLMRequest) -> MelodyVLMTranscription:
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
                            "data": _base64_ascii(image_bytes),
                        }
                    },
                    {"text": _format_user_message(request)},
                ],
            }
        ]
        config = {
            "system_instruction": _system_prompt(
                request.transcription_mode,
                request.input_kind,
            ),
            "response_mime_type": "application/json",
            "response_schema": _gemini_response_schema(
                request.transcription_mode,
                request.input_kind,
            ),
            "temperature": 0.0,
            "max_output_tokens": 2048,
        }

        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )
        raw_response = _response_text(response)
        payload = parse_json_response(raw_response)
        return parse_transcription_payload(payload, raw_response=raw_response)


def melody_fixture_key(
    image_path: Path,
    *,
    prompt_version: str,
    model_id: str,
    input_kind: InputKind,
    context: dict[str, Any] | None = None,
) -> str:
    digest = hashlib.sha256()
    digest.update(image_path.read_bytes())
    digest.update(b"\x1f")
    digest.update(input_kind.encode("utf-8"))
    digest.update(b"\x1f")
    digest.update(prompt_version.encode("utf-8"))
    digest.update(b"\x1f")
    digest.update(model_id.encode("utf-8"))
    if context is not None:
        digest.update(b"\x1f")
        digest.update(_context_fingerprint(context, input_kind).encode("utf-8"))
    return digest.hexdigest()[:16]


def write_melody_fixture(
    path: Path,
    *,
    image_path: Path,
    context_path: Path,
    context: dict[str, Any],
    input_kind: InputKind,
    prompt_version: str,
    model_id: str,
    transcription: MelodyVLMTranscription,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prompt_version": prompt_version,
        "model_id": model_id,
        "input_kind": input_kind,
        "image_path": str(image_path),
        "context_path": str(context_path),
        "context": _fixture_context(context),
        "transcription": {
            "items": [asdict(item) for item in transcription.items],
            "comments": transcription.comments,
            "overall_confidence": transcription.overall_confidence,
            "uncertainties": list(transcription.uncertainties),
            "raw_response": transcription.raw_response,
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def read_melody_fixture(path: Path) -> MelodyVLMTranscription:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload["transcription"]
    return MelodyVLMTranscription(
        items=tuple(
            MelodyVLMItem(
                kind=str(item.get("kind", "")).strip() or "unknown",
                pitch=_none_if_blank(item.get("pitch")),
                duration=str(item.get("duration", "")).strip() or "unknown",
                accidental=_none_if_blank(item.get("accidental")),
                confidence=_clamp01(item.get("confidence", 0.0)),
                notehead_x_fraction=_optional_float(item.get("notehead_x_fraction")),
                notehead_y_fraction=_optional_float(item.get("notehead_y_fraction")),
                staff_position=_optional_int(item.get("staff_position")),
                evidence=_none_if_blank(item.get("evidence")),
            )
            for item in raw.get("items", [])
        ),
        comments=str(raw.get("comments", "")),
        raw_response=str(raw.get("raw_response", "")),
        overall_confidence=_optional_float(raw.get("overall_confidence")),
        uncertainties=tuple(str(item) for item in raw.get("uncertainties", []) if item),
    )


def parse_json_response(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").strip()
        stripped = stripped.removesuffix("```").strip()
    return json.loads(stripped)


def parse_transcription_payload(
    payload: Any,
    *,
    raw_response: str = "",
) -> MelodyVLMTranscription:
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object from melody VLM, got {type(payload).__name__}")
    raw_items = payload.get("items") or []
    if not isinstance(raw_items, list):
        raise ValueError("Expected melody VLM field 'items' to be a list")
    items: list[MelodyVLMItem] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", "")).strip().lower()
        if kind not in {"note", "rest"}:
            kind = "unknown"
        items.append(
            MelodyVLMItem(
                kind=kind,
                pitch=_none_if_blank(item.get("pitch")),
                duration=str(item.get("duration", "")).strip() or "unknown",
                accidental=_normalize_accidental(item.get("accidental")),
                confidence=_clamp01(item.get("confidence", 0.0)),
                notehead_x_fraction=_optional_float(item.get("notehead_x_fraction")),
                notehead_y_fraction=_optional_float(item.get("notehead_y_fraction")),
                staff_position=_optional_int(item.get("staff_position")),
                evidence=_none_if_blank(item.get("evidence")),
            )
        )
    return MelodyVLMTranscription(
        items=tuple(items),
        comments=str(payload.get("comments", "")),
        raw_response=raw_response,
        overall_confidence=_optional_float(payload.get("overall_confidence")),
        uncertainties=tuple(str(item) for item in payload.get("uncertainties", []) if item),
    )


def _format_user_message(request: MelodyVLMRequest) -> str:
    compact_context = _compact_prompt_context(request.context, request.input_kind)
    if request.transcription_mode == "staff_position":
        return (
            "Return this JSON shape exactly:\n"
            '{"items":[{"kind":"note|rest","notehead_x_fraction":0.0,'
            '"staff_position":0,'
            '"duration":"1/8|1/4|1/2|dotted-1/4|unknown",'
            '"accidental":"sharp|flat|natural|none|unknown","confidence":0.0}],'
            '"comments":"short uncertainty note"}\n'
            "For notehead_x_fraction, use 0.0 at the left edge of the crop and 1.0 "
            "at the right edge of the crop. For rests or unclear noteheads, use "
            "notehead_x_fraction 0.0 and staff_position 0 with low confidence. "
            "If duration is unclear, use unknown. "
            f"Context: {json.dumps(compact_context, sort_keys=True)}"
        )
    if request.transcription_mode == "notehead_y":
        return (
            "Return this JSON shape exactly:\n"
            '{"items":[{"kind":"note|rest","notehead_x_fraction":0.0,'
            '"notehead_y_fraction":0.0,'
            '"duration":"1/8|1/4|1/2|dotted-1/4|unknown",'
            '"accidental":"sharp|flat|natural|none|unknown","confidence":0.0}],'
            '"comments":"short uncertainty note"}\n'
            "Use notehead center coordinates relative to the provided crop image. "
            "For rests or unclear noteheads, use notehead_x_fraction 0.0 and "
            "notehead_y_fraction 0.0 with low confidence. "
            "If duration is unclear, use unknown. "
            f"Context: {json.dumps(compact_context, sort_keys=True)}"
        )
    if request.input_kind == "pitch_ruler":
        return (
            "Return this JSON shape exactly:\n"
            '{"items":[{"kind":"note|rest","pitch":"G4 or empty string",'
            '"duration":"1/8|1/4|1/2|dotted-1/4|unknown",'
            '"accidental":"sharp|flat|natural|none|unknown","confidence":0.0}],'
            '"comments":"short uncertainty note"}\n'
            "Use the pitch labels printed on the left side of the image. "
            "The colored horizontal guide lines and printed pitch labels are aids, not music. "
            "For each handwritten notehead, choose the nearest labeled pitch guide. "
            "Use scientific pitch notation. Use empty pitch for rests or unclear pitch. "
            "If the beginning of the measure is visibly empty before the first note, "
            "include a rest. "
            "If duration is unclear, use unknown. "
            f"Context: {json.dumps(compact_context, sort_keys=True)}"
        )
    if request.input_kind == "pitch_ruler_soft":
        return (
            "Return this JSON shape exactly:\n"
            '{"items":[{"kind":"note|rest","pitch":"G4 or empty string",'
            '"duration":"1/8|1/4|1/2|dotted-1/4|unknown",'
            '"accidental":"sharp|flat|natural|none|unknown","confidence":0.0,'
            '"evidence":"visible facts supporting this item"}],'
            '"comments":"short summary of the transcription",'
            '"overall_confidence":0.0,'
            '"uncertainties":["visible uncertainty"]}\n'
            "Use the pitch labels printed on the left side of the image. "
            "The faint gray horizontal guide marks, red staff lines, and printed pitch labels "
            "are helpers, not music. The black handwritten ink is the only musical source. "
            "Do not count a helper-line crossing, printed label, barline, stem alone, slur, "
            "or accidental as a notehead. For each black handwritten notehead, choose the "
            "nearest labeled pitch guide. Include a rest if the beginning of the measure is "
            "visibly empty before the first note. You must provide the best transcription you "
            "can; use confidence/evidence/uncertainties instead of refusing. "
            "For evidence, cite observable details only, such as approximate left/middle/right "
            "location, nearest guide label, visible stem/flag/dot, or visible empty opening. "
            "If duration is unclear, use unknown. "
            f"Context: {json.dumps(compact_context, sort_keys=True)}"
        )
    if request.input_kind == "pitch_ruler_panel":
        return (
            "Return this JSON shape exactly:\n"
            '{"items":[{"kind":"note|rest","pitch":"G4 or empty string",'
            '"duration":"1/8|1/4|1/2|dotted-1/4|unknown",'
            '"accidental":"sharp|flat|natural|none|unknown","confidence":0.0,'
            '"evidence":"visible facts supporting this item"}],'
            '"comments":"short summary of the transcription",'
            '"overall_confidence":0.0,'
            '"uncertainties":["visible uncertainty"]}\n'
            "Use the left pitch reference gutter only as a vertical ruler. "
            "The actual music is the clean handwritten crop on the right. "
            "Do not count pitch labels, tick marks, the gutter divider, barlines, "
            "stems alone, slurs, or helper marks as noteheads. For each black handwritten "
            "notehead in the right-side music crop, choose the nearest pitch label at the "
            "same height in the left gutter. Include a rest if the beginning of the measure "
            "is visibly empty before the first note. You must provide the best transcription "
            "you can; use confidence/evidence/uncertainties instead of refusing. "
            "For evidence, cite observable details only, such as approximate left/middle/right "
            "location, nearest guide label, visible stem/flag/dot, or visible empty opening. "
            "If duration is unclear, use unknown. "
            f"Context: {json.dumps(compact_context, sort_keys=True)}"
        )
    return (
        "Return this JSON shape exactly:\n"
        '{"items":[{"kind":"note|rest","pitch":"G4 or empty string",'
        '"duration":"1/8|1/4|1/2|dotted-1/4|unknown",'
        '"accidental":"sharp|flat|natural|none|unknown","confidence":0.0}],'
        '"comments":"short uncertainty note"}\n'
        "Use scientific pitch notation. Use empty pitch for rests or unclear pitch. "
        "If duration is unclear, use unknown. "
        f"Context: {json.dumps(compact_context, sort_keys=True)}"
    )


def _system_prompt(
    transcription_mode: TranscriptionMode,
    input_kind: InputKind | None = None,
) -> str:
    if transcription_mode == "notehead_y":
        return NOTEHEAD_Y_SYSTEM_PROMPT
    if transcription_mode == "staff_position":
        return STAFF_POSITION_SYSTEM_PROMPT
    if input_kind == "pitch_ruler_panel":
        return PITCH_RULER_PANEL_SYSTEM_PROMPT
    if input_kind == "pitch_ruler_soft":
        return PITCH_RULER_SOFT_SYSTEM_PROMPT
    if input_kind == "pitch_ruler":
        return PITCH_RULER_SYSTEM_PROMPT
    return SYSTEM_PROMPT


def _response_schema(
    transcription_mode: TranscriptionMode,
    input_kind: InputKind | None = None,
) -> dict[str, Any]:
    if transcription_mode == "notehead_y":
        return NOTEHEAD_Y_RESPONSE_SCHEMA
    if transcription_mode == "staff_position":
        return STAFF_POSITION_RESPONSE_SCHEMA
    if input_kind in {"pitch_ruler_soft", "pitch_ruler_panel"}:
        return PITCH_RULER_SOFT_RESPONSE_SCHEMA
    return PITCH_RESPONSE_SCHEMA


def _gemini_response_schema(
    transcription_mode: TranscriptionMode,
    input_kind: InputKind | None = None,
) -> dict[str, Any]:
    return _strip_schema_key(
        _response_schema(transcription_mode, input_kind),
        "additionalProperties",
    )


def _strip_schema_key(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return {
            item_key: _strip_schema_key(item_value, key)
            for item_key, item_value in value.items()
            if item_key != key
        }
    if isinstance(value, list):
        return [_strip_schema_key(item, key) for item in value]
    return value


def _compact_prompt_context(context: dict[str, Any], input_kind: InputKind) -> dict[str, Any]:
    compact_context = {key: context.get(key) for key in PROMPT_CONTEXT_KEYS}
    compact_context["input_kind"] = input_kind
    return compact_context


def _context_fingerprint(context: dict[str, Any], input_kind: InputKind) -> str:
    return json.dumps(
        _compact_prompt_context(context, input_kind),
        sort_keys=True,
        separators=(",", ":"),
    )


def _fixture_context(context: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "slug",
        "system_index",
        "system_measure_index",
        "global_measure_index",
        "display_measure_number",
        "allow_pickup",
        "clef_hint",
        "time_signature_hint",
        "key_hint",
        "expected_measure_beats",
    )
    return {key: context.get(key) for key in keys}


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


def _build_default_client(api_key: str) -> Any:
    from google import genai  # type: ignore[import-not-found]

    return genai.Client(api_key=api_key)


class OpenAIMelodyVLM:
    """OpenAI Responses API one-measure melody transcription client for spike tooling."""

    def __init__(
        self,
        *,
        client: Any = None,
        api_key: str | None = None,
        model: str = DEFAULT_OPENAI_MELODY_VLM_MODEL,
        transcription_mode: TranscriptionMode = "pitch",
        image_detail: str = "original",
        temperature: float | None = None,
        reasoning_effort: str = "none",
    ) -> None:
        self._model = model
        self._transcription_mode = transcription_mode
        self._image_detail = image_detail
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = client
        if client is None and not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is not set; provide api_key or inject a client.")

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def prompt_version(self) -> str:
        if self._transcription_mode == "notehead_y":
            return NOTEHEAD_Y_PROMPT_VERSION
        if self._transcription_mode == "staff_position":
            return STAFF_POSITION_PROMPT_VERSION
        return MELODY_VLM_PROMPT_VERSION

    def transcribe(self, request: MelodyVLMRequest) -> MelodyVLMTranscription:
        payload = self._request_payload(request)
        response_payload = (
            self._client(payload) if self._client is not None else self._post_response(payload)
        )
        raw_response = _openai_response_text(response_payload)
        payload_json = parse_json_response(raw_response)
        return parse_transcription_payload(payload_json, raw_response=raw_response)

    def _request_payload(self, request: MelodyVLMRequest) -> dict[str, Any]:
        image_bytes = request.image_path.read_bytes()
        mime_type, _ = mimetypes.guess_type(str(request.image_path))
        mime_type = mime_type or "image/png"
        payload = {
            "model": self._model,
            "instructions": _system_prompt(
                request.transcription_mode,
                request.input_kind,
            ),
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": _format_user_message(request)},
                        {
                            "type": "input_image",
                            "image_url": (f"data:{mime_type};base64,{_base64_ascii(image_bytes)}"),
                            "detail": self._image_detail,
                        },
                    ],
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "melody_transcription",
                    "schema": _response_schema(request.transcription_mode, request.input_kind),
                    "strict": True,
                }
            },
            "reasoning": {"effort": self._reasoning_effort},
            "max_output_tokens": 4096,
            "store": False,
        }
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        return payload

    def _post_response(self, payload: dict[str, Any]) -> Any:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenAI melody VLM request failed: {exc.code} {error_body}"
            ) from exc


def default_model_for_provider(provider: VLMProvider) -> str:
    if provider == "openai":
        return DEFAULT_OPENAI_MELODY_VLM_MODEL
    return DEFAULT_MELODY_VLM_MODEL


def fixture_model_id(
    provider: VLMProvider,
    model: str,
    *,
    openai_reasoning_effort: str = "none",
) -> str:
    if provider == "gemini":
        return model
    if openai_reasoning_effort != "none":
        return f"{provider}:{model}:reasoning-{openai_reasoning_effort}"
    return f"{provider}:{model}"


def _openai_response_text(response: Any) -> str:
    if isinstance(response, dict):
        output_text = response.get("output_text")
        if isinstance(output_text, str) and output_text:
            return output_text
        chunks: list[str] = []
        for item in response.get("output") or []:
            if not isinstance(item, dict):
                continue
            for content in item.get("content") or []:
                if isinstance(content, dict) and isinstance(content.get("text"), str):
                    chunks.append(content["text"])
        if chunks:
            return "".join(chunks)
        raise RuntimeError(
            "OpenAI response contained no parseable output text"
            f" ({_openai_response_diagnostic(response)})"
        )

    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text:
        return output_text
    raise RuntimeError("OpenAI response contained no parseable output text")


def _openai_response_diagnostic(response: dict[str, Any]) -> str:
    details: list[str] = []
    status = response.get("status")
    if status:
        details.append(f"status={status}")
    incomplete_details = response.get("incomplete_details")
    if incomplete_details:
        details.append(f"incomplete_details={incomplete_details}")
    output_types = [
        str(item.get("type"))
        for item in response.get("output") or []
        if isinstance(item, dict) and item.get("type")
    ]
    if output_types:
        details.append(f"output_types={output_types}")
    content_types: list[str] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and content.get("type"):
                content_types.append(str(content["type"]))
    if content_types:
        details.append(f"content_types={content_types}")
    return "; ".join(details) or "no diagnostics"


def _base64_ascii(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode("ascii")


def _normalize_accidental(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"", "none", "null"}:
        return None
    return text


def _none_if_blank(value: Any) -> str | None:
    text = str(value or "").strip()
    if text.lower() in {"", "none", "null"}:
        return None
    return text


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return max(0.0, min(1.0, number))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
