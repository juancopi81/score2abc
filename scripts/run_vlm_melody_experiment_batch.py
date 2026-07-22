"""Run manifest-driven VLM melody prompt/image experiment batches.

This is spike tooling only. It reads prompt variant records created by
`scripts/build_vlm_melody_prompt_variants.py`, sends the exact prompt files and
image to a live provider when `--max-calls` allows it, writes replayable
fixtures under `.cache/vlm_melody/`, evaluates structured fixtures against GT,
and snapshots batch/journal artifacts for later comparison.

Example:
    uv run python scripts/run_vlm_melody_experiment_batch.py out \\
        --prompt-variant-manifest out/vlm_melody_prompt_variants_manifest.jsonl \\
        --provider openai --model gpt-5.5 \\
        --openai-reasoning-effort medium \\
        --max-calls 10 --journal
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from score2abc.melody.vlm import (  # noqa: E402
    DEFAULT_OPENAI_MELODY_VLM_MODEL,
    VLM_PROVIDERS,
    MelodyVLMTranscription,
    VLMProvider,
    _openai_response_text,
    parse_json_response,
    parse_transcription_payload,
)
from score2abc.utils import get_logger  # noqa: E402
from scripts.eval_vlm_melody_fixtures import (  # noqa: E402
    _compare_fixture,
    _load_truth_by_measure,
    _safe_name,
)

DEFAULT_FIXTURES_DIR = REPO_ROOT / ".cache" / "vlm_melody"
DEFAULT_GROUND_TRUTH_DIR = Path("dataset/ground_truth")
DEFAULT_BATCH_ROOT_NAME = "vlm_melody_batches"
DEFAULT_OPENAI_REASONING_EFFORT = "none"
GEMINI_DEFAULT_MODEL = "gemini-3.1-flash-lite"


@dataclass(frozen=True)
class PromptVariantVLMRequest:
    record: dict[str, Any]
    image_path: Path
    context_path: Path
    system_prompt_path: Path
    user_prompt_path: Path
    schema_path: Path | None
    context: dict[str, Any]
    system_prompt: str
    user_prompt: str
    schema: dict[str, Any] | None
    provider: VLMProvider
    model: str
    model_id: str
    openai_reasoning_effort: str
    fixture_key: str

    @property
    def prompt_variant_id(self) -> str:
        return str(self.record["prompt_variant_id"])

    @property
    def prompt_id(self) -> str:
        return str(self.record["prompt_id"])

    @property
    def variant_id(self) -> str:
        return str(self.record["variant_id"])

    @property
    def input_kind(self) -> str:
        return str(self.record["input_kind"])

    @property
    def output_mode(self) -> str:
        return str(self.record.get("output_mode", "json_schema"))

    @property
    def transcription_mode(self) -> str:
        return str(self.record.get("transcription_mode", "pitch"))

    @property
    def is_structured(self) -> bool:
        return self.schema is not None and self.output_mode != "free_response"


@dataclass(frozen=True)
class PromptVariantVLMResponse:
    raw_response: str


class PromptVariantTranscriber(Protocol):
    def transcribe(self, request: PromptVariantVLMRequest) -> PromptVariantVLMResponse: ...


class OpenAIPromptVariantTranscriber:
    """Responses API client that uses prompt variant files verbatim."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        reasoning_effort: str = DEFAULT_OPENAI_REASONING_EFFORT,
        max_output_tokens: int = 4096,
        request_timeout_seconds: int = 120,
        image_detail: str = "original",
        client: Any = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._reasoning_effort = reasoning_effort
        self._max_output_tokens = max_output_tokens
        self._request_timeout_seconds = request_timeout_seconds
        self._image_detail = image_detail
        self._client = client
        if client is None and not api_key:
            import os

            self._api_key = os.environ.get("OPENAI_API_KEY")
        if client is None and not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is not set; provide api_key or inject a client.")

    def transcribe(self, request: PromptVariantVLMRequest) -> PromptVariantVLMResponse:
        payload = self._request_payload(request)
        response_payload = (
            self._client(payload) if self._client is not None else self._post_response(payload)
        )
        return PromptVariantVLMResponse(raw_response=_openai_response_text(response_payload))

    def _request_payload(self, request: PromptVariantVLMRequest) -> dict[str, Any]:
        image_bytes = request.image_path.read_bytes()
        mime_type, _ = mimetypes.guess_type(str(request.image_path))
        mime_type = mime_type or "image/png"
        payload: dict[str, Any] = {
            "model": self._model,
            "instructions": request.system_prompt,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": request.user_prompt},
                        {
                            "type": "input_image",
                            "image_url": (f"data:{mime_type};base64,{_base64_ascii(image_bytes)}"),
                            "detail": self._image_detail,
                        },
                    ],
                }
            ],
            "text": {"format": {"type": "text"}},
            "reasoning": {"effort": self._reasoning_effort},
            "max_output_tokens": self._max_output_tokens,
            "store": False,
        }
        if request.schema is not None:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "melody_transcription",
                    "schema": request.schema,
                    "strict": True,
                }
            }
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
            with urllib.request.urlopen(request, timeout=self._request_timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            raise RuntimeError(
                "OpenAI prompt-variant VLM request timed out after "
                f"{self._request_timeout_seconds}s"
            ) from exc
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenAI prompt-variant VLM request failed: {exc.code} {error_body}"
            ) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise RuntimeError(
                    "OpenAI prompt-variant VLM request timed out after "
                    f"{self._request_timeout_seconds}s"
                ) from exc
            raise RuntimeError(f"OpenAI prompt-variant VLM request failed: {exc}") from exc


class GeminiPromptVariantTranscriber:
    """Gemini client that uses prompt variant files verbatim."""

    def __init__(self, *, model: str, api_key: str | None = None, client: Any = None) -> None:
        self._model = model
        if client is not None:
            self._client = client
            return

        import os

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise RuntimeError("GEMINI_API_KEY is not set; provide api_key or inject a client.")
        from google import genai  # type: ignore[import-not-found]

        self._client = genai.Client(api_key=resolved_key)

    def transcribe(self, request: PromptVariantVLMRequest) -> PromptVariantVLMResponse:
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
                    {"text": request.user_prompt},
                ],
            }
        ]
        config: dict[str, Any] = {
            "system_instruction": request.system_prompt,
            "temperature": 0.0,
            "max_output_tokens": 4096,
        }
        if request.schema is not None:
            config.update(
                {
                    "response_mime_type": "application/json",
                    "response_schema": _strip_schema_key(request.schema, "additionalProperties"),
                }
            )
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )
        return PromptVariantVLMResponse(raw_response=_gemini_response_text(response))


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.max_calls < 0:
        parser.error("--max-calls must be >= 0")

    logger = get_logger("score2abc.run_vlm_melody_experiment_batch")
    model = args.model or default_model_for_provider(args.provider)

    try:
        summary = run_batch(
            out_dir=args.out_dir,
            prompt_variant_manifest=args.prompt_variant_manifest,
            provider=args.provider,
            model=model,
            openai_reasoning_effort=args.openai_reasoning_effort,
            fixtures_dir=args.fixtures_dir,
            ground_truth_dir=args.ground_truth,
            max_calls=args.max_calls,
            max_output_tokens=args.max_output_tokens,
            request_timeout_seconds=args.request_timeout_seconds,
            force=args.force,
            journal=args.journal,
            run_id=args.run_id,
            selected_prompt_variant_ids=(
                set(args.prompt_variant_id) if args.prompt_variant_id else None
            ),
            selected_variant_ids=set(args.variant_id) if args.variant_id else None,
            selected_prompt_ids=set(args.prompt_id) if args.prompt_id else None,
            selected_slugs=set(args.slug) if args.slug else None,
            selected_systems=set(args.system) if args.system else None,
            selected_measures=set(args.measure) if args.measure else None,
            logger=logger,
        )
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Wrote VLM melody batch: %s", summary["paths"]["batch_dir"])
    logger.info("Summary: %s", json.dumps(summary["counts"], sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="Pipeline output directory.")
    parser.add_argument(
        "--prompt-variant-manifest",
        type=Path,
        default=None,
        help=(
            "Prompt variant JSONL manifest. Defaults to "
            "<out_dir>/vlm_melody_prompt_variants_manifest.jsonl."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=VLM_PROVIDERS,
        default="openai",
        help="Live VLM provider to call. Defaults to openai.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Provider model id. Defaults to gpt-5.5 for OpenAI and "
            "gemini-3.1-flash-lite for Gemini."
        ),
    )
    parser.add_argument(
        "--openai-reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh"),
        default=DEFAULT_OPENAI_REASONING_EFFORT,
        help="OpenAI reasoning effort. Included in fixture keys. Defaults to none.",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=0,
        help="Maximum live VLM calls. Defaults to 0 for dry-run/safety.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=4096,
        help=(
            "OpenAI Responses max_output_tokens. Increase this for high reasoning effort. "
            "Defaults to 4096."
        ),
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
        default=120,
        help="OpenAI HTTP request timeout in seconds. Defaults to 120.",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=DEFAULT_FIXTURES_DIR,
        help="Fixture/cache destination. Defaults to .cache/vlm_melody.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_DIR,
        help="Ground-truth directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing prompt-variant fixtures.",
    )
    parser.add_argument(
        "--journal",
        action="store_true",
        help="Write one journal folder per completed/evaluated/non-structured result.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional batch folder name. Defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--prompt-variant-id",
        action="append",
        default=None,
        help="Limit to a prompt_variant_id. Repeat for multiple.",
    )
    parser.add_argument(
        "--variant-id",
        action="append",
        default=None,
        help="Limit to an image variant_id. Repeat for multiple.",
    )
    parser.add_argument(
        "--prompt-id",
        action="append",
        default=None,
        help="Limit to a prompt_id. Repeat for multiple.",
    )
    parser.add_argument("--slug", action="append", default=None, help="Limit to a work slug.")
    parser.add_argument(
        "--system",
        action="append",
        type=int,
        default=None,
        help="Limit to a 1-based system index.",
    )
    parser.add_argument(
        "--measure",
        action="append",
        type=int,
        default=None,
        help="Limit to a system-local measure index.",
    )
    return parser


def run_batch(
    *,
    out_dir: Path,
    provider: VLMProvider,
    model: str,
    prompt_variant_manifest: Path | None = None,
    openai_reasoning_effort: str = DEFAULT_OPENAI_REASONING_EFFORT,
    fixtures_dir: Path = DEFAULT_FIXTURES_DIR,
    ground_truth_dir: Path = DEFAULT_GROUND_TRUTH_DIR,
    max_calls: int = 0,
    max_output_tokens: int = 4096,
    request_timeout_seconds: int = 120,
    force: bool = False,
    journal: bool = False,
    run_id: str | None = None,
    selected_prompt_variant_ids: set[str] | None = None,
    selected_variant_ids: set[str] | None = None,
    selected_prompt_ids: set[str] | None = None,
    selected_slugs: set[str] | None = None,
    selected_systems: set[int] | None = None,
    selected_measures: set[int] | None = None,
    transcriber: PromptVariantTranscriber | None = None,
    logger: Any = None,
) -> dict[str, Any]:
    logger = logger or get_logger("score2abc.run_vlm_melody_experiment_batch")
    if max_calls < 0:
        raise ValueError("max_calls must be >= 0")
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be > 0")
    if request_timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds must be > 0")
    manifest_path = prompt_variant_manifest or out_dir / "vlm_melody_prompt_variants_manifest.jsonl"
    records = list(
        _selected_prompt_variant_records(
            _read_jsonl(manifest_path),
            selected_prompt_variant_ids=selected_prompt_variant_ids,
            selected_variant_ids=selected_variant_ids,
            selected_prompt_ids=selected_prompt_ids,
            selected_slugs=selected_slugs,
            selected_systems=selected_systems,
            selected_measures=selected_measures,
        )
    )
    if not records:
        raise ValueError("No prompt variant records matched the selected filters.")

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_dir = out_dir / DEFAULT_BATCH_ROOT_NAME / _safe_name(run_id)
    reports_dir = batch_dir / "reports"
    journals_dir = batch_dir / "journals"
    batch_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    if journal:
        journals_dir.mkdir(exist_ok=True)
    _write_jsonl(batch_dir / "selected_records.jsonl", records)

    model_id = prompt_variant_model_id(
        provider,
        model,
        openai_reasoning_effort=openai_reasoning_effort,
    )
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    calls = 0
    active_transcriber = transcriber
    for record in records:
        request = _build_request(
            out_dir=out_dir,
            record=record,
            provider=provider,
            model=model,
            model_id=model_id,
            openai_reasoning_effort=openai_reasoning_effort,
        )
        fixture_path = fixtures_dir / f"{request.fixture_key}.json"
        base_result = _base_result(request, fixture_path)

        if max_calls == 0:
            status = "dry_run_existing" if fixture_path.exists() and not force else "dry_run"
            result = {
                **base_result,
                "status": status,
                "action": "dry_run",
                "would_call": status == "dry_run",
            }
            logger.info(
                "%s: %s %s -> %s",
                "Would skip existing" if status == "dry_run_existing" else "Would call",
                request.prompt_variant_id,
                request.image_path.name,
                fixture_path.name,
            )
            results.append(result)
            continue

        if fixture_path.exists() and not force:
            logger.info("Skip existing: %s -> %s", request.prompt_variant_id, fixture_path.name)
            result = _evaluate_result(
                request,
                fixture_path=fixture_path,
                ground_truth_dir=ground_truth_dir,
                base_result={**base_result, "action": "skipped_existing"},
            )
            result = _finalize_result_artifacts(
                request,
                result,
                batch_dir=batch_dir,
                reports_dir=reports_dir,
                journals_dir=journals_dir,
                journal=journal,
                fixture_path=fixture_path,
                manifest_path=manifest_path,
                out_dir=out_dir,
                max_calls=max_calls,
                max_output_tokens=max_output_tokens,
                request_timeout_seconds=request_timeout_seconds,
                force=force,
            )
            results.append(result)
            continue

        if calls >= max_calls:
            logger.info("Call cap reached: %s -> %s", request.prompt_variant_id, fixture_path.name)
            results.append(
                {
                    **base_result,
                    "status": "call_cap_reached",
                    "action": "not_called",
                    "would_call": True,
                }
            )
            continue

        if active_transcriber is None:
            active_transcriber = _build_transcriber(
                provider=provider,
                model=model,
                openai_reasoning_effort=openai_reasoning_effort,
                max_output_tokens=max_output_tokens,
                request_timeout_seconds=request_timeout_seconds,
            )

        response = active_transcriber.transcribe(request)
        calls += 1
        parse_error = _write_prompt_variant_fixture(
            fixture_path,
            request=request,
            raw_response=response.raw_response,
        )
        result = _evaluate_result(
            request,
            fixture_path=fixture_path,
            ground_truth_dir=ground_truth_dir,
            parse_error=parse_error,
            base_result={**base_result, "action": "written"},
        )
        result = _finalize_result_artifacts(
            request,
            result,
            batch_dir=batch_dir,
            reports_dir=reports_dir,
            journals_dir=journals_dir,
            journal=journal,
            fixture_path=fixture_path,
            manifest_path=manifest_path,
            out_dir=out_dir,
            max_calls=max_calls,
            max_output_tokens=max_output_tokens,
            request_timeout_seconds=request_timeout_seconds,
            force=force,
        )
        logger.info(
            "Wrote %s result: %s -> %s",
            result["status"],
            request.prompt_variant_id,
            fixture_path.name,
        )
        results.append(result)

    _write_jsonl(batch_dir / "batch_manifest.jsonl", results)
    summary = _batch_summary(
        run_id=run_id,
        out_dir=out_dir,
        batch_dir=batch_dir,
        manifest_path=manifest_path,
        fixtures_dir=fixtures_dir,
        ground_truth_dir=ground_truth_dir,
        provider=provider,
        model=model,
        model_id=model_id,
        openai_reasoning_effort=openai_reasoning_effort,
        max_calls=max_calls,
        max_output_tokens=max_output_tokens,
        request_timeout_seconds=request_timeout_seconds,
        force=force,
        journal=journal,
        live_calls=calls,
        results=results,
    )
    (batch_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (batch_dir / "README.md").write_text(
        _batch_readme(summary, results),
        encoding="utf-8",
    )
    return summary


def _build_transcriber(
    *,
    provider: VLMProvider,
    model: str,
    openai_reasoning_effort: str,
    max_output_tokens: int = 4096,
    request_timeout_seconds: int = 120,
) -> PromptVariantTranscriber:
    if provider == "openai":
        return OpenAIPromptVariantTranscriber(
            model=model,
            reasoning_effort=openai_reasoning_effort,
            max_output_tokens=max_output_tokens,
            request_timeout_seconds=request_timeout_seconds,
        )
    if provider == "gemini":
        return GeminiPromptVariantTranscriber(model=model)
    raise ValueError(f"Unsupported provider: {provider}")


def _build_request(
    *,
    out_dir: Path,
    record: dict[str, Any],
    provider: VLMProvider,
    model: str,
    model_id: str,
    openai_reasoning_effort: str,
) -> PromptVariantVLMRequest:
    image_path = _resolve_path(out_dir, _require_str(record, "image_path"))
    context_path = _resolve_path(out_dir, _require_str(record, "context_path"))
    system_prompt_path = _resolve_path(out_dir, _require_str(record, "system_prompt_path"))
    user_prompt_path = _resolve_path(out_dir, _require_str(record, "user_prompt_path"))
    schema_path = _optional_path(out_dir, record.get("schema_path"))

    context = json.loads(context_path.read_text(encoding="utf-8"))
    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    user_prompt = user_prompt_path.read_text(encoding="utf-8")
    schema = json.loads(schema_path.read_text(encoding="utf-8")) if schema_path else None
    fixture_key = prompt_variant_fixture_key(
        image_path=image_path,
        record=record,
        context=context,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=schema,
        model_id=model_id,
    )
    return PromptVariantVLMRequest(
        record=record,
        image_path=image_path,
        context_path=context_path,
        system_prompt_path=system_prompt_path,
        user_prompt_path=user_prompt_path,
        schema_path=schema_path,
        context=context,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=schema,
        provider=provider,
        model=model,
        model_id=model_id,
        openai_reasoning_effort=openai_reasoning_effort,
        fixture_key=fixture_key,
    )


def prompt_variant_model_id(
    provider: VLMProvider,
    model: str,
    *,
    openai_reasoning_effort: str = DEFAULT_OPENAI_REASONING_EFFORT,
) -> str:
    if provider == "openai" and openai_reasoning_effort != "none":
        return f"{provider}:{model}:reasoning-{openai_reasoning_effort}"
    return f"{provider}:{model}"


def prompt_variant_fixture_key(
    *,
    image_path: Path,
    record: dict[str, Any],
    context: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    schema: dict[str, Any] | None,
    model_id: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(image_path.read_bytes())
    for value in (
        record.get("input_kind"),
        record.get("output_mode"),
        record.get("transcription_mode"),
        record.get("prompt_variant_id"),
        record.get("prompt_id"),
        record.get("variant_id"),
        model_id,
        _hash_text(system_prompt),
        _hash_text(user_prompt),
        _hash_json(schema),
        _hash_json(context),
    ):
        digest.update(b"\x1f")
        digest.update(str(value).encode("utf-8"))
    return digest.hexdigest()[:16]


def _write_prompt_variant_fixture(
    path: Path,
    *,
    request: PromptVariantVLMRequest,
    raw_response: str,
) -> str | None:
    parse_error = None
    transcription = MelodyVLMTranscription(
        items=(),
        comments="",
        raw_response=raw_response,
    )
    parsed_payload = None
    if request.is_structured:
        try:
            parsed_payload = parse_json_response(raw_response)
            transcription = parse_transcription_payload(
                parsed_payload,
                raw_response=raw_response,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            parse_error = str(exc)

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "prompt_version": f"prompt-variant:{request.prompt_id}",
        "model_id": request.model_id,
        "provider": request.provider,
        "model": request.model,
        "openai_reasoning_effort": request.openai_reasoning_effort,
        "input_kind": request.input_kind,
        "transcription_mode": request.transcription_mode,
        "output_mode": request.output_mode,
        "prompt_variant_id": request.prompt_variant_id,
        "prompt_id": request.prompt_id,
        "variant_id": request.variant_id,
        "image_path": str(request.image_path),
        "context_path": str(request.context_path),
        "system_prompt_path": str(request.system_prompt_path),
        "user_prompt_path": str(request.user_prompt_path),
        "schema_path": str(request.schema_path) if request.schema_path else None,
        "prompt_hashes": {
            "system_sha256": _hash_text(request.system_prompt),
            "user_sha256": _hash_text(request.user_prompt),
            "schema_sha256": _hash_json(request.schema),
        },
        "context": request.context,
        "transcription": {
            "items": [asdict(item) for item in transcription.items],
            "comments": transcription.comments,
            "overall_confidence": transcription.overall_confidence,
            "uncertainties": list(transcription.uncertainties),
            "raw_response": transcription.raw_response,
        },
        "parsed_payload": parsed_payload,
        "parse_error": parse_error,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return parse_error


def _evaluate_result(
    request: PromptVariantVLMRequest,
    *,
    fixture_path: Path,
    ground_truth_dir: Path,
    base_result: dict[str, Any],
    parse_error: str | None = None,
) -> dict[str, Any]:
    if parse_error is not None:
        return {
            **base_result,
            "status": "parse_error",
            "parse_error": parse_error,
        }
    if not request.is_structured:
        return {
            **base_result,
            "status": "not_structured",
            "eval_status": "skipped",
        }

    truth_path = ground_truth_dir / f"{request.record['slug']}.json"
    if not truth_path.exists():
        return {
            **base_result,
            "status": "missing_ground_truth",
            "ground_truth": str(truth_path),
        }

    truth_by_measure = _load_truth_by_measure(truth_path)
    truth_notes = truth_by_measure.get(int(request.record["global_measure_index"]), [])
    from score2abc.melody.vlm import read_melody_fixture

    transcription = read_melody_fixture(fixture_path)
    eval_result = _compare_fixture(
        request.record,
        request.input_kind,
        request.image_path,
        fixture_path,
        transcription,
        truth_notes,
        request.transcription_mode,
    )
    return {
        **base_result,
        **eval_result,
        "prompt_variant_id": request.prompt_variant_id,
        "prompt_id": request.prompt_id,
        "variant_id": request.variant_id,
        "output_mode": request.output_mode,
        "provider": request.provider,
        "model": request.model,
        "model_id": request.model_id,
    }


def _finalize_result_artifacts(
    request: PromptVariantVLMRequest,
    result: dict[str, Any],
    *,
    batch_dir: Path,
    reports_dir: Path,
    journals_dir: Path,
    journal: bool,
    fixture_path: Path,
    manifest_path: Path,
    out_dir: Path,
    max_calls: int,
    max_output_tokens: int,
    request_timeout_seconds: int,
    force: bool,
) -> dict[str, Any]:
    report_path = (
        reports_dir / f"{_safe_name(request.prompt_variant_id)}__{request.fixture_key}.json"
    )
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result = {**result, "report": str(report_path)}

    if journal and result["status"] in {
        "evaluated",
        "not_structured",
        "parse_error",
        "missing_ground_truth",
    }:
        journal_path = _write_journal(
            request,
            result=result,
            journal_root=journals_dir,
            fixture_path=fixture_path,
            report_path=report_path,
            manifest_path=manifest_path,
            out_dir=out_dir,
            batch_dir=batch_dir,
            max_calls=max_calls,
            max_output_tokens=max_output_tokens,
            request_timeout_seconds=request_timeout_seconds,
            force=force,
        )
        result = {**result, "journal": str(journal_path)}
    return result


def _write_journal(
    request: PromptVariantVLMRequest,
    *,
    result: dict[str, Any],
    journal_root: Path,
    fixture_path: Path,
    report_path: Path,
    manifest_path: Path,
    out_dir: Path,
    batch_dir: Path,
    max_calls: int,
    max_output_tokens: int,
    request_timeout_seconds: int,
    force: bool,
) -> Path:
    journal_path = journal_root / f"{_safe_name(request.prompt_variant_id)}__{request.fixture_key}"
    artifacts_dir = journal_path / "artifacts"
    prompts_dir = journal_path / "prompts"
    journal_path.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(exist_ok=True)
    prompts_dir.mkdir(exist_ok=True)

    image_copy = artifacts_dir / f"input{request.image_path.suffix or '.png'}"
    context_copy = artifacts_dir / "context.json"
    fixture_copy = artifacts_dir / "fixture.json"
    report_copy = journal_path / "eval_result.json"
    shutil.copy2(request.image_path, image_copy)
    shutil.copy2(request.context_path, context_copy)
    if fixture_path.exists():
        shutil.copy2(fixture_path, fixture_copy)
    shutil.copy2(request.system_prompt_path, prompts_dir / "system.txt")
    shutil.copy2(request.user_prompt_path, prompts_dir / "user.txt")
    if request.schema_path is not None:
        shutil.copy2(request.schema_path, prompts_dir / "schema.json")
    shutil.copy2(report_path, report_copy)

    experiment = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "identity": _identity(request),
        "config": {
            "provider": request.provider,
            "model": request.model,
            "model_id": request.model_id,
            "openai_reasoning_effort": request.openai_reasoning_effort,
            "max_calls": max_calls,
            "max_output_tokens": max_output_tokens,
            "request_timeout_seconds": request_timeout_seconds,
            "force": force,
            "fixture_key": request.fixture_key,
        },
        "paths": {
            "out_dir": str(out_dir),
            "batch_dir": str(batch_dir),
            "prompt_variant_manifest": str(manifest_path),
            "source_image": str(request.image_path),
            "source_context": str(request.context_path),
            "source_fixture": str(fixture_path),
            "journal_input_image": str(image_copy),
            "journal_context": str(context_copy),
            "journal_fixture": str(fixture_copy) if fixture_path.exists() else None,
            "system_prompt": str(prompts_dir / "system.txt"),
            "user_prompt": str(prompts_dir / "user.txt"),
            "schema": str(prompts_dir / "schema.json") if request.schema_path else None,
            "eval_result": str(report_copy),
        },
        "eval_result": result,
        "git": _git_snapshot(),
        "replay_command": _replay_command(
            out_dir=out_dir,
            manifest_path=manifest_path,
            request=request,
            max_calls=1,
            max_output_tokens=max_output_tokens,
            request_timeout_seconds=request_timeout_seconds,
            force=True,
            journal=True,
        ),
    }
    (journal_path / "experiment.json").write_text(
        json.dumps(experiment, indent=2) + "\n",
        encoding="utf-8",
    )
    (journal_path / "README.md").write_text(
        _journal_readme(experiment, result),
        encoding="utf-8",
    )
    return journal_path


def _base_result(request: PromptVariantVLMRequest, fixture_path: Path) -> dict[str, Any]:
    return {
        **_identity(request),
        "provider": request.provider,
        "model": request.model,
        "model_id": request.model_id,
        "openai_reasoning_effort": request.openai_reasoning_effort,
        "fixture": str(fixture_path),
        "fixture_key": request.fixture_key,
        "image_path": str(request.image_path),
        "context_path": str(request.context_path),
        "system_prompt_path": str(request.system_prompt_path),
        "user_prompt_path": str(request.user_prompt_path),
        "schema_path": str(request.schema_path) if request.schema_path else None,
        "structured": request.is_structured,
    }


def _identity(request: PromptVariantVLMRequest) -> dict[str, Any]:
    return {
        "prompt_variant_id": request.prompt_variant_id,
        "prompt_id": request.prompt_id,
        "variant_id": request.variant_id,
        "slug": request.record["slug"],
        "system_index": int(request.record["system_index"]),
        "system_measure_index": int(request.record["system_measure_index"]),
        "global_measure_index": int(request.record["global_measure_index"]),
        "display_measure_number": request.record.get("display_measure_number"),
        "input_kind": request.input_kind,
        "transcription_mode": request.transcription_mode,
        "output_mode": request.output_mode,
    }


def _batch_summary(
    *,
    run_id: str,
    out_dir: Path,
    batch_dir: Path,
    manifest_path: Path,
    fixtures_dir: Path,
    ground_truth_dir: Path,
    provider: VLMProvider,
    model: str,
    model_id: str,
    openai_reasoning_effort: str,
    max_calls: int,
    max_output_tokens: int,
    request_timeout_seconds: int,
    force: bool,
    journal: bool,
    live_calls: int,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    status_counts = Counter(str(result["status"]) for result in results)
    action_counts = Counter(str(result.get("action", "")) for result in results)
    evaluated = [result for result in results if result["status"] == "evaluated"]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": max_calls == 0,
        "config": {
            "provider": provider,
            "model": model,
            "model_id": model_id,
            "openai_reasoning_effort": openai_reasoning_effort,
            "max_calls": max_calls,
            "max_output_tokens": max_output_tokens,
            "request_timeout_seconds": request_timeout_seconds,
            "force": force,
            "journal": journal,
        },
        "paths": {
            "out_dir": str(out_dir),
            "batch_dir": str(batch_dir),
            "prompt_variant_manifest": str(manifest_path),
            "fixtures_dir": str(fixtures_dir),
            "ground_truth_dir": str(ground_truth_dir),
            "batch_manifest": str(batch_dir / "batch_manifest.jsonl"),
            "selected_records": str(batch_dir / "selected_records.jsonl"),
            "summary": str(batch_dir / "summary.json"),
            "readme": str(batch_dir / "README.md"),
        },
        "counts": {
            "selected": len(results),
            "live_calls": live_calls,
            "statuses": dict(sorted(status_counts.items())),
            "actions": {key: value for key, value in sorted(action_counts.items()) if key},
        },
    }
    if evaluated:
        summary["metrics"] = {
            "evaluated": len(evaluated),
            "note_count_match_rate": round(
                sum(1 for result in evaluated if result.get("note_count_match")) / len(evaluated),
                6,
            ),
            "pitch_order_accuracy_avg": round(
                sum(float(result["pitch_order_accuracy"]) for result in evaluated) / len(evaluated),
                6,
            ),
            "duration_order_accuracy_avg": round(
                sum(float(result["duration_order_accuracy"]) for result in evaluated)
                / len(evaluated),
                6,
            ),
        }
    return summary


def _batch_readme(summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    lines = [
        "# VLM Melody Batch",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Provider/model: `{summary['config']['provider']}` / `{summary['config']['model']}`",
        f"- Dry run: `{summary['dry_run']}`",
        f"- Live calls: `{summary['counts']['live_calls']}`",
        "",
        "## Results",
        "",
        "| Status | Prompt variant | Variant | Prompt | Fixture | Journal | Report |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            "| "
            f"{result.get('status')} | "
            f"`{result.get('prompt_variant_id')}` | "
            f"`{result.get('variant_id')}` | "
            f"`{result.get('prompt_id')}` | "
            f"`{_relative_or_str(result.get('fixture'))}` | "
            f"`{_relative_or_str(result.get('journal'))}` | "
            f"`{_relative_or_str(result.get('report'))}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _journal_readme(experiment: dict[str, Any], result: dict[str, Any]) -> str:
    return (
        "# VLM Melody Prompt Variant Experiment\n\n"
        f"- Prompt variant: `{experiment['identity']['prompt_variant_id']}`\n"
        f"- Provider/model: `{experiment['config']['provider']}` / "
        f"`{experiment['config']['model']}`\n"
        f"- Status: `{result.get('status')}`\n"
        f"- Fixture: `{experiment['paths']['source_fixture']}`\n\n"
        "## Files\n\n"
        "- `artifacts/input.png`: exact image sent to the model.\n"
        "- `prompts/system.txt`: exact system prompt from the prompt variant manifest.\n"
        "- `prompts/user.txt`: exact user prompt from the prompt variant manifest.\n"
        "- `prompts/schema.json`: exact schema, when the prompt is structured.\n"
        "- `artifacts/fixture.json`: raw model response fixture.\n"
        "- `eval_result.json`: local structured eval or skip status.\n"
        "- `experiment.json`: machine-readable config, paths, git snapshot, and replay command.\n"
    )


def _selected_prompt_variant_records(
    records: list[dict[str, Any]],
    *,
    selected_prompt_variant_ids: set[str] | None,
    selected_variant_ids: set[str] | None,
    selected_prompt_ids: set[str] | None,
    selected_slugs: set[str] | None,
    selected_systems: set[int] | None,
    selected_measures: set[int] | None,
) -> list[dict[str, Any]]:
    selected = []
    for record in records:
        if (
            selected_prompt_variant_ids is not None
            and record.get("prompt_variant_id") not in selected_prompt_variant_ids
        ):
            continue
        if (
            selected_variant_ids is not None
            and record.get("variant_id") not in selected_variant_ids
        ):
            continue
        if selected_prompt_ids is not None and record.get("prompt_id") not in selected_prompt_ids:
            continue
        if selected_slugs is not None and record.get("slug") not in selected_slugs:
            continue
        if selected_systems is not None and int(record["system_index"]) not in selected_systems:
            continue
        if (
            selected_measures is not None
            and int(record["system_measure_index"]) not in selected_measures
        ):
            continue
        selected.append(record)
    return selected


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt variant manifest not found: {path}")
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _resolve_path(out_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return out_dir / path


def _optional_path(out_dir: Path, value: Any) -> Path | None:
    if value in {None, ""}:
        return None
    return _resolve_path(out_dir, str(value))


def _require_str(record: dict[str, Any], key: str) -> str:
    value = record[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"Prompt variant record {record.get('prompt_variant_id')} lacks {key}")
    return value


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _base64_ascii(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


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


def _gemini_response_text(response: Any) -> str:
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


def _git_snapshot() -> dict[str, Any]:
    return {
        "branch": _git_output("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": _git_output("rev-parse", "HEAD"),
        "dirty": bool(_git_output("status", "--porcelain")),
    }


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _replay_command(
    *,
    out_dir: Path,
    manifest_path: Path,
    request: PromptVariantVLMRequest,
    max_calls: int,
    max_output_tokens: int,
    request_timeout_seconds: int,
    force: bool,
    journal: bool,
) -> str:
    parts = [
        "uv",
        "run",
        "python",
        "scripts/run_vlm_melody_experiment_batch.py",
        str(out_dir),
        "--prompt-variant-manifest",
        str(manifest_path),
        "--provider",
        request.provider,
        "--model",
        request.model,
        "--prompt-variant-id",
        request.prompt_variant_id,
        "--max-calls",
        str(max_calls),
        "--max-output-tokens",
        str(max_output_tokens),
        "--request-timeout-seconds",
        str(request_timeout_seconds),
    ]
    if request.provider == "openai":
        parts.extend(["--openai-reasoning-effort", request.openai_reasoning_effort])
    if force:
        parts.append("--force")
    if journal:
        parts.append("--journal")
    return " ".join(_shell_quote(part) for part in parts)


def _shell_quote(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:=+-]+", value):
        return value
    return json.dumps(value)


def _relative_or_str(value: Any) -> str:
    if not value:
        return ""
    path = Path(str(value))
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def default_model_for_provider(provider: VLMProvider) -> str:
    if provider == "openai":
        return DEFAULT_OPENAI_MELODY_VLM_MODEL
    return GEMINI_DEFAULT_MODEL


if __name__ == "__main__":
    raise SystemExit(main())
