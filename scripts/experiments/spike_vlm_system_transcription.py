"""Transcribe one benchmark system from a vertical STAFF-image contact sheet.

This is spike-only tooling. The default run freezes a replayable request without
calling a provider. Pass ``--max-calls 1`` intentionally to permit one OpenAI
Responses API call. Benchmark truth is opened only after a complete provider
response and converted prediction have been saved.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shlex
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_vlm_melody_event_benchmark as benchmark  # noqa: E402

DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_DETAIL = "original"
DEFAULT_MAX_OUTPUT_TOKENS = 8192
DEFAULT_TIMEOUT_SECONDS = 180
EXPERIMENT_ROOT = Path("experiments/vlm_system_transcription")
STANDALONE_SPLITS = ("development", "validation")
LABEL_HEIGHT_PX = 28
ROW_GAP_PX = 12
SIDE_PADDING_PX = 12

SYSTEM_PROMPT = (
    "You are an expert optical music transcription system. Transcribe every labeled measure "
    "in the supplied system contact sheet into strict JSON. Work from the score pixels and the "
    "provided musical context. Make a decisive best-effort transcription even when notation is "
    "ambiguous. No refusal or omission is allowed: include every measure and every visible event, "
    "using low confidence when needed. "
    "Represent simultaneous notes as separate ordered events with the same onset_beats. Return "
    "only JSON matching the supplied schema."
)

EVENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "kind": {"type": "string", "enum": ["note", "rest"]},
        "onset_beats": {"type": "number", "minimum": 0},
        "duration_beats": {"type": "number", "exclusiveMinimum": 0},
        "pitch": {"type": ["string", "null"]},
        "accidental": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "string"},
    },
    "required": [
        "kind",
        "onset_beats",
        "duration_beats",
        "pitch",
        "accidental",
        "confidence",
        "evidence",
    ],
}

TRANSCRIPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "measures": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "system_measure_index": {"type": "integer", "minimum": 1},
                    "events": {"type": "array", "items": EVENT_SCHEMA},
                },
                "required": ["system_measure_index", "events"],
            },
        }
    },
    "required": ["measures"],
}

ProviderTransport = Callable[[dict[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class FrozenRequest:
    rows: tuple[dict[str, Any], ...]
    run_dir: Path
    sheet_path: Path
    sheet_sha256: str
    system_prompt: str
    user_prompt: str
    config: dict[str, Any]
    payload: dict[str, Any]
    cache_key: str
    cache_path: Path


class OpenAIResponsesTransport:
    """Minimal Responses API transport; credentials come from the environment only."""

    def __init__(self, *, timeout_seconds: int) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds

    def __call__(self, payload: dict[str, Any]) -> Mapping[str, Any]:
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            raise RuntimeError(
                f"OpenAI Responses request timed out after {self._timeout_seconds}s"
            ) from exc
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI Responses request failed: {exc.code} {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI Responses request failed: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_experiment(
            args.out_dir,
            slug=args.slug,
            split_name=args.split,
            system_index=args.system,
            experiment_id=args.experiment_id,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            detail=args.detail,
            max_output_tokens=args.max_output_tokens,
            timeout_seconds=args.timeout_seconds,
            max_calls=args.max_calls,
            force=args.force,
        )
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(report["paths"]["run_dir"])
    print(f"status: {report['status']}; live calls: {report['live_calls']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", type=Path, default=Path("out"))
    parser.add_argument("--slug", default=benchmark.DEFAULT_SLUG)
    parser.add_argument("--split", choices=STANDALONE_SPLITS, required=True)
    parser.add_argument("--system", type=int, required=True)
    parser.add_argument("--experiment-id")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
        default=DEFAULT_REASONING_EFFORT,
    )
    parser.add_argument(
        "--detail",
        choices=("low", "high", "original", "auto"),
        default=DEFAULT_DETAIL,
        help="Responses image detail level.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--max-calls",
        type=int,
        choices=(0, 1),
        default=0,
        help="Zero freezes a dry run; one explicitly permits a single live call.",
    )
    parser.add_argument("--force", action="store_true", help="Ignore an existing cache entry.")
    return parser


def run_experiment(
    out_dir: Path,
    *,
    slug: str,
    split_name: str,
    system_index: int,
    experiment_id: str | None = None,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    detail: str = DEFAULT_DETAIL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_calls: int = 0,
    force: bool = False,
    transport: ProviderTransport | None = None,
) -> dict[str, Any]:
    if max_calls not in (0, 1):
        raise ValueError("max_calls must be 0 or 1")
    if system_index <= 0:
        raise ValueError("system_index must be positive")
    if max_output_tokens <= 0 or timeout_seconds <= 0:
        raise ValueError("max_output_tokens and timeout_seconds must be positive")
    if split_name not in benchmark.BENCHMARK_SPLITS:
        raise ValueError(f"Unknown benchmark split: {split_name}")
    if split_name not in STANDALONE_SPLITS:
        raise ValueError(
            "Standalone system transcription only permits development and validation splits"
        )

    frozen = freeze_request(
        out_dir,
        slug=slug,
        split_name=split_name,
        system_index=system_index,
        experiment_id=experiment_id,
        model=model,
        reasoning_effort=reasoning_effort,
        detail=detail,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
        max_calls=max_calls,
    )
    response_payload: Mapping[str, Any] | None = None
    status = "dry_run"
    live_calls = 0
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "vlm_system_transcription_spike",
        "status": status,
        "live_calls": live_calls,
        "cache_key": frozen.cache_key,
        "paths": {"run_dir": str(frozen.run_dir)},
    }
    if frozen.cache_path.exists() and not force:
        response_payload = _load_json(frozen.cache_path)
        status = "cached"
    elif max_calls == 1:
        active_transport = transport or OpenAIResponsesTransport(timeout_seconds=timeout_seconds)
        live_calls = 1
        try:
            response_payload = active_transport(frozen.payload)
        except Exception as exc:
            _write_failed_result(
                frozen.run_dir,
                report,
                status="provider_error",
                live_calls=live_calls,
                exc=exc,
            )
            raise
        _write_json(frozen.cache_path, response_payload)
        status = "called"
    report.update({"status": status, "live_calls": live_calls})
    if response_payload is None:
        _write_json(frozen.run_dir / "result.json", report)
        return report

    # Persist the complete response side before parsing or opening benchmark truth.
    _write_json(frozen.run_dir / "provider_response.json", response_payload)
    _write_json(frozen.run_dir / "usage.json", response_payload.get("usage") or {})
    try:
        raw_response = _response_text(response_payload)
    except (KeyError, TypeError, ValueError) as exc:
        _write_failed_result(
            frozen.run_dir,
            report,
            status="response_error",
            live_calls=live_calls,
            exc=exc,
        )
        raise
    (frozen.run_dir / "raw_response.txt").write_text(raw_response + "\n", encoding="utf-8")

    try:
        parsed = parse_transcription(raw_response, expected_rows=frozen.rows)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        _write_failed_result(
            frozen.run_dir,
            report,
            status="parse_error",
            live_calls=live_calls,
            exc=exc,
        )
        raise
    _write_json(frozen.run_dir / "parsed_prediction.json", parsed)
    predictions = transcription_to_benchmark_predictions(parsed, frozen.rows)
    predictions_path = frozen.run_dir / "predictions.jsonl"
    _write_jsonl(predictions_path, predictions)

    # Prediction and provider artifacts now exist; only this block may read truth.
    truth_path = out_dir / slug / "vlm_melody_event_benchmark" / split_name / "truth.jsonl"
    selected_keys = {_identity_key(row["identity"]) for row in frozen.rows}
    truth_rows = [
        row for row in _read_jsonl(truth_path) if _identity_key(row["identity"]) in selected_keys
    ]
    if len(truth_rows) != len(frozen.rows):
        raise ValueError(
            f"Benchmark truth coverage mismatch for selected system: "
            f"{len(truth_rows)} != {len(frozen.rows)}"
        )
    evaluation = benchmark.evaluate_predictions(truth_rows, predictions)
    evaluation_path = frozen.run_dir / "evaluation.json"
    _write_json(evaluation_path, evaluation)
    report.update(
        {
            "prediction_count": len(predictions),
            "evaluation_summary": evaluation["summary"],
            "paths": {
                **report["paths"],
                "predictions": str(predictions_path),
                "evaluation": str(evaluation_path),
            },
        }
    )
    _write_json(frozen.run_dir / "result.json", report)
    return report


def freeze_request(
    out_dir: Path,
    *,
    slug: str,
    split_name: str,
    system_index: int,
    experiment_id: str | None,
    model: str,
    reasoning_effort: str,
    detail: str,
    max_output_tokens: int,
    timeout_seconds: int,
    max_calls: int,
) -> FrozenRequest:
    request_path = out_dir / slug / "vlm_melody_event_benchmark" / split_name / "requests.jsonl"
    rows = tuple(
        sorted(
            (
                row
                for row in _read_jsonl(request_path)
                if int(row["identity"]["system_index"]) == system_index
            ),
            key=lambda row: int(row["identity"]["system_measure_index"]),
        )
    )
    if not rows:
        raise ValueError(f"No requests for split={split_name}, system={system_index}")
    local_indices = [int(row["identity"]["system_measure_index"]) for row in rows]
    if len(local_indices) != len(set(local_indices)):
        raise ValueError(f"Duplicate system-local measure indices: {local_indices}")

    resolved_id = experiment_id or datetime.now(timezone.utc).strftime(
        f"%Y%m%dT%H%M%SZ-{split_name}-s{system_index:03d}"
    )
    run_dir = out_dir / EXPERIMENT_ROOT / _safe_name(resolved_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = run_dir / "contact_sheet.png"
    sheet_manifest = build_contact_sheet(out_dir, rows, sheet_path)
    sheet_sha256 = _sha256(sheet_path)
    sheet_manifest["sha256"] = sheet_sha256

    context = _shared_context(rows)
    user_prompt = _user_prompt(rows, context)
    config = {
        "transport_version": "openai-responses-v1",
        "event_schema_version": "canonical-system-events-v1",
        "provider": "openai",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "detail": detail,
        "max_output_tokens": max_output_tokens,
        "timeout_seconds": timeout_seconds,
        "store": False,
    }
    payload = _response_payload(
        sheet_path,
        model=model,
        reasoning_effort=reasoning_effort,
        detail=detail,
        max_output_tokens=max_output_tokens,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    cache_key = _cache_key(
        sheet_path,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        schema=TRANSCRIPTION_SCHEMA,
        config=config,
    )
    cache_path = out_dir / EXPERIMENT_ROOT / "cache" / f"{cache_key}.json"

    # Everything below is frozen before run_experiment can access truth.
    _write_json(run_dir / "selected_requests.json", list(rows))
    _write_json(run_dir / "contact_sheet.json", sheet_manifest)
    (run_dir / "system_prompt.txt").write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")
    (run_dir / "user_prompt.txt").write_text(user_prompt + "\n", encoding="utf-8")
    _write_json(run_dir / "schema.json", TRANSCRIPTION_SCHEMA)
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "request_payload.json", payload)
    replay = _replay_command(
        out_dir=out_dir,
        slug=slug,
        split_name=split_name,
        system_index=system_index,
        experiment_id=resolved_id,
        config=config,
        max_calls=max_calls,
    )
    (run_dir / "replay.sh").write_text(replay + "\n", encoding="utf-8")
    _write_json(
        run_dir / "request_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": resolved_id,
            "slug": slug,
            "split": split_name,
            "system_index": system_index,
            "system_measure_indices": local_indices,
            "source_requests_path": str(request_path),
            "contact_sheet_sha256": sheet_sha256,
            "system_prompt_sha256": _text_sha256(SYSTEM_PROMPT),
            "user_prompt_sha256": _text_sha256(user_prompt),
            "schema_sha256": _json_sha256(TRANSCRIPTION_SCHEMA),
            "config_sha256": _json_sha256(config),
            "cache_key": cache_key,
            "cache_path": str(cache_path),
            "truth_read_during_request_preparation": False,
        },
    )
    return FrozenRequest(
        rows=rows,
        run_dir=run_dir,
        sheet_path=sheet_path,
        sheet_sha256=sheet_sha256,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        config=config,
        payload=payload,
        cache_key=cache_key,
        cache_path=cache_path,
    )


def build_contact_sheet(
    out_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    destination: Path,
) -> dict[str, Any]:
    sources: list[tuple[int, Path, Image.Image]] = []
    for row in rows:
        local_index = int(row["identity"]["system_measure_index"])
        relative = Path(str(row["images"]["staff"]["path_relative_to_out"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid path_relative_to_out: {relative}")
        path = out_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"Benchmark STAFF image not found: {path}")
        if _sha256(path) != str(row["images"]["staff"]["sha256"]):
            raise ValueError(f"Benchmark STAFF image hash drift: {path}")
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        sources.append((local_index, path, image))

    max_width = max(image.width for _, _, image in sources)
    width = max_width + 2 * SIDE_PADDING_PX
    height = sum(LABEL_HEIGHT_PX + image.height for _, _, image in sources)
    height += ROW_GAP_PX * (len(sources) - 1)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    placements = []
    y = 0
    for local_index, path, image in sources:
        label = f"SYSTEM MEASURE {local_index:03d}"
        draw.text((SIDE_PADDING_PX, y + 8), label, fill="black", font=font)
        image_y = y + LABEL_HEIGHT_PX
        sheet.paste(image, (SIDE_PADDING_PX, image_y))
        placements.append(
            {
                "system_measure_index": local_index,
                "source_path_relative_to_out": str(path.relative_to(out_dir)),
                "source_sha256": _sha256(path),
                "label": label,
                "label_box_px": [0, y, width, image_y],
                "score_box_px": [
                    SIDE_PADDING_PX,
                    image_y,
                    SIDE_PADDING_PX + image.width,
                    image_y + image.height,
                ],
            }
        )
        y = image_y + image.height + ROW_GAP_PX
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)
    return {
        "kind": "vertical_staff_contact_sheet",
        "width_px": width,
        "height_px": height,
        "background": "white",
        "labels_outside_score_pixels": True,
        "placements": placements,
    }


def parse_transcription(
    raw_response: str,
    *,
    expected_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = json.loads(raw_response)
    if not isinstance(payload, dict) or set(payload) != {"measures"}:
        raise ValueError("Response must contain only the measures field")
    measures = payload["measures"]
    if not isinstance(measures, list):
        raise ValueError("measures must be an array")
    expected = [int(row["identity"]["system_measure_index"]) for row in expected_rows]
    actual = []
    for measure in measures:
        if not isinstance(measure, dict) or set(measure) != {"system_measure_index", "events"}:
            raise ValueError("Each measure must contain system_measure_index and events")
        local_index = _strict_int(measure["system_measure_index"], "system_measure_index")
        actual.append(local_index)
        if not isinstance(measure["events"], list):
            raise ValueError("events must be an array")
        for event in measure["events"]:
            _validate_event(event)
    if actual != expected:
        raise ValueError(f"Response measure order/coverage mismatch: {actual} != {expected}")
    return payload


def transcription_to_benchmark_predictions(
    transcription: Mapping[str, Any],
    request_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_local = {int(row["identity"]["system_measure_index"]): row for row in request_rows}
    predictions = []
    for measure in transcription["measures"]:
        local_index = int(measure["system_measure_index"])
        row = rows_by_local[local_index]
        notes = []
        rests = []
        for event in measure["events"]:
            converted: dict[str, Any] = {
                "onset_beats": event["onset_beats"],
                "duration_beats": event["duration_beats"],
            }
            if event["kind"] == "rest":
                rests.append(converted)
                continue
            converted["pitch_midi"] = pitch_name_to_midi(str(event["pitch"]))
            accidental = _accidental_number(event.get("accidental"))
            if accidental is not None:
                converted["accidental"] = accidental
            notes.append(converted)
        predictions.append(
            {
                "schema_version": 1,
                "identity": dict(row["identity"]),
                "notes": notes,
                "rests": rests,
            }
        )
    return predictions


def pitch_name_to_midi(value: str) -> int:
    match = re.fullmatch(r"\s*([A-Ga-g])([#b]{0,2})(-?\d+)\s*", value)
    if not match:
        raise ValueError(f"Invalid scientific pitch name: {value!r}")
    letter, accidental, octave_text = match.groups()
    semitones = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    offset = accidental.count("#") - accidental.count("b")
    midi = 12 * (int(octave_text) + 1) + semitones[letter.upper()] + offset
    if not 0 <= midi <= 127:
        raise ValueError(f"Pitch outside MIDI range: {value!r}")
    return midi


def _validate_event(event: Any) -> None:
    required = {
        "kind",
        "onset_beats",
        "duration_beats",
        "pitch",
        "accidental",
        "confidence",
        "evidence",
    }
    if not isinstance(event, dict) or set(event) != required:
        raise ValueError(f"Event fields must be exactly {sorted(required)}")
    if event["kind"] not in ("note", "rest"):
        raise ValueError(f"Invalid event kind: {event['kind']!r}")
    if not _is_number(event["onset_beats"]) or event["onset_beats"] < 0:
        raise ValueError("onset_beats must be a non-negative number")
    if not _is_number(event["duration_beats"]) or event["duration_beats"] <= 0:
        raise ValueError("duration_beats must be a positive number")
    if not _is_number(event["confidence"]) or not 0 <= event["confidence"] <= 1:
        raise ValueError("confidence must be a number between 0 and 1")
    if not isinstance(event["evidence"], str):
        raise ValueError("evidence must be a string")
    if event["accidental"] is not None and not isinstance(event["accidental"], str):
        raise ValueError("accidental must be a string or null")
    if event["kind"] == "rest":
        if event["pitch"] is not None:
            raise ValueError("Rest pitch must be null")
    elif not isinstance(event["pitch"], str):
        raise ValueError("Note pitch must be a scientific pitch-name string")
    else:
        pitch_name_to_midi(event["pitch"])


def _response_payload(
    sheet_path: Path,
    *,
    model: str,
    reasoning_effort: str,
    detail: str,
    max_output_tokens: int,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    encoded = base64.b64encode(sheet_path.read_bytes()).decode("ascii")
    return {
        "model": model,
        "instructions": system_prompt,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{encoded}",
                        "detail": detail,
                    },
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "system_melody_transcription",
                "schema": TRANSCRIPTION_SCHEMA,
                "strict": True,
            }
        },
        "reasoning": {"effort": reasoning_effort},
        "max_output_tokens": max_output_tokens,
        "store": False,
    }


def _user_prompt(rows: Sequence[Mapping[str, Any]], context: Mapping[str, Any]) -> str:
    indices = [int(row["identity"]["system_measure_index"]) for row in rows]
    return (
        f"Transcribe system-local measures {indices} in top-to-bottom label order. "
        f"Context: clef={context['clef']}; time_signature={context['time_signature']}; "
        f"key_hint={context['key_hint']!r}; expected_measure_beats="
        f"{context['expected_measure_beats']}. Include one measure object for every label, even "
        "when uncertain or when the measure contains only rests. Within each measure, order "
        "events by onset and then visual low-to-high pitch for simultaneous notes. Use "
        "scientific pitch names such as C4, F#4, or Bb3. For rests, set pitch to null. Describe "
        "concise visual evidence and lower confidence instead of refusing or leaving an event out."
    )


def _shared_context(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    contexts = [dict(row["allowed_context"]) for row in rows]
    keys = ("clef", "time_signature", "key_hint", "expected_measure_beats")
    shared = {key: contexts[0].get(key) for key in keys}
    for context in contexts[1:]:
        if any(context.get(key) != shared[key] for key in keys):
            raise ValueError("Selected benchmark requests do not share musical context")
    return shared


def _cache_key(
    image_path: Path,
    *,
    system_prompt: str,
    user_prompt: str,
    schema: Mapping[str, Any],
    config: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(image_path.read_bytes())
    for value in (
        system_prompt.encode("utf-8"),
        user_prompt.encode("utf-8"),
        _canonical_json(schema),
        _canonical_json(config),
    ):
        digest.update(b"\x1f")
        digest.update(value)
    return digest.hexdigest()


def _response_text(payload: Mapping[str, Any]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    texts = []
    for item in payload.get("output") or []:
        if not isinstance(item, Mapping):
            continue
        for content in item.get("content") or []:
            if isinstance(content, Mapping) and isinstance(content.get("text"), str):
                texts.append(content["text"])
    if not texts:
        raise ValueError("OpenAI response did not contain output text")
    return "\n".join(texts).strip()


def _accidental_number(value: Any) -> int | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    values = {
        "": None,
        "none": None,
        "implicit": None,
        "double_flat": -2,
        "flat": -1,
        "natural": 0,
        "sharp": 1,
        "double_sharp": 2,
        "bb": -2,
        "b": -1,
        "#": 1,
        "##": 2,
    }
    if normalized not in values:
        raise ValueError(f"Unsupported accidental: {value!r}")
    return values[normalized]


def _replay_command(
    *,
    out_dir: Path,
    slug: str,
    split_name: str,
    system_index: int,
    experiment_id: str,
    config: Mapping[str, Any],
    max_calls: int,
) -> str:
    arguments = [
        "uv",
        "run",
        "python",
        "scripts/experiments/spike_vlm_system_transcription.py",
        str(out_dir),
        "--slug",
        slug,
        "--split",
        split_name,
        "--system",
        str(system_index),
        "--experiment-id",
        experiment_id,
        "--model",
        str(config["model"]),
        "--reasoning-effort",
        str(config["reasoning_effort"]),
        "--detail",
        str(config["detail"]),
        "--max-output-tokens",
        str(config["max_output_tokens"]),
        "--timeout-seconds",
        str(config["timeout_seconds"]),
        "--max-calls",
        str(max_calls),
    ]
    return shlex.join(arguments)


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _identity_key(identity: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(identity["slug"]),
        int(identity["system_index"]),
        int(identity["system_measure_index"]),
        int(identity["global_measure_index"]),
    )


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    if not safe:
        raise ValueError("experiment_id has no safe filename characters")
    return safe


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_failed_result(
    run_dir: Path,
    report: Mapping[str, Any],
    *,
    status: str,
    live_calls: int,
    exc: Exception,
) -> None:
    failed = {
        **report,
        "status": status,
        "live_calls": live_calls,
        "error": {"type": type(exc).__name__, "message": str(exc)},
    }
    _write_json(run_dir / "result.json", failed)


if __name__ == "__main__":
    raise SystemExit(main())
