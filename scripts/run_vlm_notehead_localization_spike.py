"""Run manifest-driven VLM notehead-localization spike batches.

This runner is intentionally separate from the melody-transcription spike. It localizes
notehead centers only, journals every attempted or evaluated record, and never exposes
coordinate or canonical ground truth to a provider.

Example dry run (the default, requiring no API key):
    uv run python scripts/run_vlm_notehead_localization_spike.py out \
        --task-kind direct-localization --slug example --system 1 --measure 2
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import mimetypes
import os
import re
import shutil
import sys
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from score2abc.utils import get_logger  # noqa: E402

TaskKind = Literal["direct-localization", "candidate-assisted-localization"]
Provider = Literal["openai", "gemini"]

TASK_KINDS: tuple[TaskKind, ...] = (
    "direct-localization",
    "candidate-assisted-localization",
)
PROVIDERS: tuple[Provider, ...] = ("openai", "gemini")
OPENAI_REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
DEFAULT_OPENAI_MODEL = "gpt-5.6-sol"
DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"
DEFAULT_REASONING_EFFORT = "none"
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_IMAGE_DETAIL = "original"
ANNOTATION_ELLIPSE_MARGIN = 1.15
DEFAULT_BATCH_ROOT_NAME = "vlm_notehead_localization_batches"
DEFAULT_FIXTURES_DIR = REPO_ROOT / ".cache" / "vlm_notehead_localization"


@dataclass(frozen=True)
class RoleImage:
    role: str
    path: Path


@dataclass(frozen=True)
class LocalizationRequest:
    record: dict[str, Any]
    images: tuple[RoleImage, ...]
    context_path: Path
    context: dict[str, Any]
    candidate_artifact_path: Path | None
    candidate_artifact: dict[str, Any] | None
    candidate_ids: tuple[str, ...]
    coordinate_ground_truth_path: Path | None
    canonical_ground_truth_path: Path
    system_prompt: str
    user_prompt: str
    schema: dict[str, Any]
    provider: Provider
    model: str
    reasoning_effort: str
    image_detail: str
    max_output_tokens: int
    fixture_key: str

    @property
    def experiment_id(self) -> str:
        return str(self.record["experiment_id"])

    @property
    def task_kind(self) -> TaskKind:
        return self.record["task_kind"]


@dataclass(frozen=True)
class LocalizationResponse:
    raw_response: str
    response_id: str | None = None
    usage: dict[str, Any] | None = None
    provider_response: Any = None


class LocalizationTransport(Protocol):
    def localize(self, request: LocalizationRequest) -> LocalizationResponse: ...


class OpenAILocalizationTransport:
    """OpenAI Responses API transport with an injectable client for tests."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        image_detail: str = DEFAULT_IMAGE_DETAIL,
        client: Any = None,
    ) -> None:
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._max_output_tokens = max_output_tokens
        self._timeout_seconds = timeout_seconds
        self._image_detail = image_detail
        self._client = client
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if client is None and not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is not set; inject a client or set the key.")

    def localize(self, request: LocalizationRequest) -> LocalizationResponse:
        payload = self._request_payload(request)
        if self._client is None:
            response = self._post(payload)
        elif callable(self._client):
            response = self._client(payload)
        elif hasattr(self._client, "responses"):
            response = self._client.responses.create(**payload)
        else:
            raise TypeError("Injected OpenAI client must be callable or expose responses.create().")
        return LocalizationResponse(
            raw_response=_openai_response_text(response),
            response_id=_response_id(response),
            usage=_response_usage(response),
            provider_response=_jsonable(response),
        )

    def _request_payload(self, request: LocalizationRequest) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": request.user_prompt}]
        for image in request.images:
            content.append(
                {
                    "type": "input_text",
                    "text": f"Image role: {image.role}",
                }
            )
            content.append(
                {
                    "type": "input_image",
                    "image_url": _data_url(image.path),
                    "detail": self._image_detail,
                }
            )
        return {
            "model": self._model,
            "instructions": request.system_prompt,
            "input": [{"role": "user", "content": content}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "notehead_localization",
                    "schema": request.schema,
                    "strict": True,
                }
            },
            "reasoning": {"effort": self._reasoning_effort},
            "max_output_tokens": self._max_output_tokens,
            "store": False,
        }

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
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
            raise RuntimeError(f"OpenAI request timed out after {self._timeout_seconds}s") from exc
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI request failed: HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc


class GeminiLocalizationTransport:
    """google-genai transport with structured output and an injectable client."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        client: Any = None,
    ) -> None:
        self._model = model
        self._max_output_tokens = max_output_tokens
        if client is not None:
            self._client = client
            return
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise RuntimeError("GEMINI_API_KEY is not set; inject a client or set the key.")
        from google import genai  # type: ignore[import-not-found]
        from google.genai import types  # type: ignore[import-not-found]

        self._client = genai.Client(
            api_key=resolved_key,
            http_options=types.HttpOptions(timeout=timeout_seconds * 1000),
        )

    def localize(self, request: LocalizationRequest) -> LocalizationResponse:
        parts: list[dict[str, Any]] = [{"text": request.user_prompt}]
        for image in request.images:
            parts.append({"text": f"Image role: {image.role}"})
            parts.append(
                {
                    "inline_data": {
                        "mime_type": _mime_type(image.path),
                        "data": base64.b64encode(image.path.read_bytes()).decode("ascii"),
                    }
                }
            )
        config = {
            "system_instruction": request.system_prompt,
            "temperature": 0.0,
            "max_output_tokens": self._max_output_tokens,
            "response_mime_type": "application/json",
            "response_schema": _strip_schema_key(request.schema, "additionalProperties"),
        }
        response = self._client.models.generate_content(
            model=self._model,
            contents=[{"role": "user", "parts": parts}],
            config=config,
        )
        return LocalizationResponse(
            raw_response=_gemini_response_text(response),
            response_id=_response_id(response),
            usage=_response_usage(response),
            provider_response=_jsonable(response),
        )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.max_calls < 0:
        parser.error("--max-calls must be >= 0")
    if args.max_output_tokens <= 0:
        parser.error("--max-output-tokens must be > 0")
    if args.timeout <= 0:
        parser.error("--timeout must be > 0")
    model = args.model or default_model_for_provider(args.provider)
    logger = get_logger("score2abc.run_vlm_notehead_localization_spike")
    try:
        summary = run_batch(
            out_dir=args.out_dir,
            manifest_path=args.manifest,
            provider=args.provider,
            model=model,
            openai_reasoning_effort=args.openai_reasoning_effort,
            max_calls=args.max_calls,
            max_output_tokens=args.max_output_tokens,
            timeout_seconds=args.timeout,
            run_id=args.run_id,
            force=args.force,
            fixtures_dir=args.fixtures_dir,
            selected_task_kinds=set(args.task_kind) if args.task_kind else None,
            selected_slugs=set(args.slug) if args.slug else None,
            selected_systems=set(args.system) if args.system else None,
            selected_measures=set(args.measure) if args.measure else None,
            selected_experiment_ids=(set(args.experiment_id) if args.experiment_id else None),
            logger=logger,
        )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    logger.info("Wrote notehead localization batch: %s", summary["paths"]["batch_dir"])
    logger.info("Summary: %s", json.dumps(summary["overall"], sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="Pipeline output directory.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Defaults to <out_dir>/vlm_notehead_localization_manifest.jsonl.",
    )
    parser.add_argument("--task-kind", action="append", choices=TASK_KINDS)
    parser.add_argument("--slug", action="append")
    parser.add_argument("--system", action="append", type=int)
    parser.add_argument("--measure", action="append", type=int)
    parser.add_argument(
        "--experiment-id",
        action="append",
        help="Select an exact experiment id; used by journal replay commands.",
    )
    parser.add_argument("--provider", choices=PROVIDERS, default="openai")
    parser.add_argument(
        "--model",
        help=(
            f"Defaults to {DEFAULT_OPENAI_MODEL} for OpenAI and "
            f"{DEFAULT_GEMINI_MODEL} for Gemini."
        ),
    )
    parser.add_argument(
        "--openai-reasoning-effort",
        choices=OPENAI_REASONING_EFFORTS,
        default=DEFAULT_REASONING_EFFORT,
    )
    parser.add_argument("--max-calls", type=int, default=0)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument(
        "--timeout",
        "--request-timeout-seconds",
        dest="timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
    )
    parser.add_argument("--run-id")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fixtures-dir", type=Path, default=DEFAULT_FIXTURES_DIR)
    return parser


def run_batch(
    *,
    out_dir: Path,
    provider: Provider,
    model: str,
    manifest_path: Path | None = None,
    openai_reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    max_calls: int = 0,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    run_id: str | None = None,
    force: bool = False,
    fixtures_dir: Path = DEFAULT_FIXTURES_DIR,
    selected_task_kinds: set[str] | None = None,
    selected_slugs: set[str] | None = None,
    selected_systems: set[int] | None = None,
    selected_measures: set[int] | None = None,
    selected_experiment_ids: set[str] | None = None,
    transport: LocalizationTransport | None = None,
    provider_client: Any = None,
    logger: Any = None,
) -> dict[str, Any]:
    """Run selected manifest records without allowing one record to abort the batch."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    if openai_reasoning_effort not in OPENAI_REASONING_EFFORTS:
        raise ValueError(f"Unsupported OpenAI reasoning effort: {openai_reasoning_effort}")
    if max_calls < 0:
        raise ValueError("max_calls must be >= 0")
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be > 0")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")
    logger = logger or get_logger("score2abc.run_vlm_notehead_localization_spike")
    manifest = manifest_path or out_dir / "vlm_notehead_localization_manifest.jsonl"
    records = _select_records(
        _read_jsonl(manifest),
        task_kinds=selected_task_kinds,
        slugs=selected_slugs,
        systems=selected_systems,
        measures=selected_measures,
        experiment_ids=selected_experiment_ids,
    )
    if not records:
        raise ValueError("No notehead-localization manifest records matched the filters.")
    experiment_ids = [str(record.get("experiment_id", "")) for record in records]
    duplicates = sorted(item for item, count in Counter(experiment_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Selected experiment_id values must be unique: {duplicates}")

    resolved_run_id = _safe_name(run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    batch_dir = out_dir / DEFAULT_BATCH_ROOT_NAME / resolved_run_id
    if batch_dir.exists() and not force:
        raise FileExistsError(f"Batch directory already exists; use --force: {batch_dir}")
    if batch_dir.exists():
        shutil.rmtree(batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)
    journals_dir = batch_dir / "journals"
    journals_dir.mkdir(exist_ok=True)
    _write_jsonl(batch_dir / "selected_records.jsonl", records)

    active_transport = transport
    results: list[dict[str, Any]] = []
    live_calls = 0
    for ordinal, record in enumerate(records, start=1):
        request: LocalizationRequest | None = None
        response: LocalizationResponse | None = None
        fixture_path: Path | None = None
        try:
            request = build_request(
                out_dir=out_dir,
                manifest_path=manifest,
                record=record,
                provider=provider,
                model=model,
                reasoning_effort=openai_reasoning_effort,
                image_detail=DEFAULT_IMAGE_DETAIL,
                max_output_tokens=max_output_tokens,
            )
            fixture_path = fixtures_dir / f"{request.fixture_key}.json"
            base = _base_result(request, fixture_path)
            if max_calls == 0:
                result = {
                    **base,
                    "status": "dry_run",
                    "gt_status": "not_evaluated",
                    "action": (
                        "would_evaluate_existing"
                        if fixture_path.exists() and not force
                        else "would_call"
                    ),
                }
                results.append(result)
                continue

            if fixture_path.exists() and not force:
                fixture = _load_json(fixture_path)
                raw_response = str(fixture.get("raw_response", ""))
                response = LocalizationResponse(
                    raw_response=raw_response,
                    response_id=fixture.get("response_id"),
                    usage=fixture.get("usage"),
                    provider_response=fixture.get("provider_response"),
                )
                try:
                    parsed = parse_and_validate_payload(raw_response, request)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    result = {
                        **base,
                        "status": "failed",
                        "gt_status": "not_evaluated",
                        "action": "evaluated_existing",
                        "failure_stage": "parse",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                else:
                    try:
                        result = _evaluate_request(
                            request,
                            parsed,
                            base={**base, "action": "evaluated_existing"},
                        )
                    except Exception as exc:
                        result = {
                            **base,
                            "status": "failed",
                            "gt_status": "failed",
                            "action": "evaluated_existing",
                            "failure_stage": "evaluation",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
            elif live_calls >= max_calls:
                result = {
                    **base,
                    "status": "call_cap_reached",
                    "gt_status": "not_evaluated",
                    "action": "not_called",
                }
                results.append(result)
                continue
            else:
                if active_transport is None:
                    active_transport = _build_transport(
                        provider=provider,
                        model=model,
                        reasoning_effort=openai_reasoning_effort,
                        max_output_tokens=max_output_tokens,
                        timeout_seconds=timeout_seconds,
                        provider_client=provider_client,
                    )
                live_calls += 1
                try:
                    response = active_transport.localize(request)
                except Exception as exc:  # provider failures must not stop the batch
                    result = {
                        **base,
                        "status": "failed",
                        "gt_status": "not_evaluated",
                        "action": "called",
                        "failure_stage": "api",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                else:
                    try:
                        parsed = parse_and_validate_payload(response.raw_response, request)
                    except (json.JSONDecodeError, TypeError, ValueError) as exc:
                        result = {
                            **base,
                            "status": "failed",
                            "gt_status": "not_evaluated",
                            "action": "called",
                            "failure_stage": "parse",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        _write_fixture(
                            fixture_path,
                            request=request,
                            response=response,
                            parsed_payload=None,
                            parse_error=result["error"],
                        )
                    else:
                        _write_fixture(
                            fixture_path,
                            request=request,
                            response=response,
                            parsed_payload=parsed,
                            parse_error=None,
                        )
                        try:
                            result = _evaluate_request(
                                request,
                                parsed,
                                base={**base, "action": "called"},
                            )
                        except Exception as exc:  # evaluation is record-local too
                            result = {
                                **base,
                                "status": "failed",
                                "gt_status": "failed",
                                "action": "called",
                                "failure_stage": "evaluation",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
            journal = _write_journal(
                batch_dir=batch_dir,
                ordinal=ordinal,
                manifest_path=manifest,
                out_dir=out_dir,
                request=request,
                record=record,
                result=result,
                response=response,
                fixture_path=fixture_path,
                max_output_tokens=max_output_tokens,
                timeout_seconds=timeout_seconds,
                force=force,
            )
            results.append({**result, "journal": str(journal)})
        except Exception as exc:  # malformed/missing input is isolated as preparation failure
            result = {
                "experiment_id": str(record.get("experiment_id", f"record-{ordinal}")),
                "task_kind": record.get("task_kind"),
                "slug": record.get("slug"),
                "system_index": record.get("system_index"),
                "system_measure_index": record.get("system_measure_index"),
                "status": "failed",
                "gt_status": "not_evaluated",
                "action": "not_called",
                "failure_stage": "preparation",
                "error": f"{type(exc).__name__}: {exc}",
            }
            journal = _write_journal(
                batch_dir=batch_dir,
                ordinal=ordinal,
                manifest_path=manifest,
                out_dir=out_dir,
                request=None,
                record=record,
                result=result,
                response=response,
                fixture_path=fixture_path,
                max_output_tokens=max_output_tokens,
                timeout_seconds=timeout_seconds,
                force=force,
            )
            results.append({**result, "journal": str(journal)})
            logger.warning("Record %s failed: %s", result["experiment_id"], exc)

    _write_jsonl(batch_dir / "batch_manifest.jsonl", results)
    summary = _batch_summary(
        run_id=resolved_run_id,
        batch_dir=batch_dir,
        manifest_path=manifest,
        provider=provider,
        model=model,
        reasoning_effort=openai_reasoning_effort,
        max_calls=max_calls,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
        force=force,
        live_calls=live_calls,
        results=results,
    )
    _write_json(batch_dir / "summary.json", summary)
    (batch_dir / "README.md").write_text(_batch_readme(summary), encoding="utf-8")
    return summary


def build_request(
    *,
    out_dir: Path,
    manifest_path: Path,
    record: dict[str, Any],
    provider: Provider,
    model: str,
    reasoning_effort: str,
    image_detail: str,
    max_output_tokens: int,
) -> LocalizationRequest:
    _validate_manifest_record(record)
    images = tuple(
        RoleImage(
            role=str(item["role"]),
            path=_resolve_path(out_dir, manifest_path, str(item["path"])),
        )
        for item in record["images"]
    )
    for image in images:
        if not image.path.is_file():
            raise FileNotFoundError(f"Input image not found for role {image.role}: {image.path}")
    context_path = _resolve_path(out_dir, manifest_path, str(record["context_path"]))
    context = _load_json(context_path)
    candidate_path = _optional_resolved_path(
        out_dir, manifest_path, record.get("candidate_artifact_path")
    )
    candidate_artifact = _load_json(candidate_path) if candidate_path else None
    candidate_ids = _candidate_ids(candidate_artifact)
    if record["task_kind"] == "candidate-assisted-localization" and candidate_artifact is None:
        raise ValueError("candidate-assisted-localization requires candidate_artifact_path")
    coordinate_path = _optional_resolved_path(
        out_dir, manifest_path, record.get("coordinate_ground_truth_path")
    )
    canonical_path = _resolve_path(
        out_dir, manifest_path, str(record["canonical_ground_truth_path"])
    )
    schema = localization_schema(record["task_kind"], candidate_ids=candidate_ids)
    system_prompt, user_prompt = localization_prompts(
        record["task_kind"],
        image_roles=[image.role for image in images],
        candidate_ids=candidate_ids,
        candidate_points=_candidate_prompt_points(
            candidate_artifact,
            raw_image_size=_raw_image_size(record["source_context"]),
        ),
    )
    fixture_context = {
        "manifest": {
            key: record[key]
            for key in (
                "schema_version",
                "experiment_id",
                "task_kind",
                "slug",
                "system_index",
                "system_measure_index",
                "global_measure_index",
                "source_context",
            )
        },
        "context": context,
        "candidate_artifact": candidate_artifact,
    }
    key = localization_fixture_key(
        images=images,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=schema,
        context=fixture_context,
        provider=provider,
        model=model,
        reasoning_effort=reasoning_effort,
        image_detail=image_detail,
        max_output_tokens=max_output_tokens,
    )
    return LocalizationRequest(
        record=record,
        images=images,
        context_path=context_path,
        context=context,
        candidate_artifact_path=candidate_path,
        candidate_artifact=candidate_artifact,
        candidate_ids=candidate_ids,
        coordinate_ground_truth_path=coordinate_path,
        canonical_ground_truth_path=canonical_path,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=schema,
        provider=provider,
        model=model,
        reasoning_effort=reasoning_effort,
        image_detail=image_detail,
        max_output_tokens=max_output_tokens,
        fixture_key=key,
    )


def localization_schema(
    task_kind: TaskKind,
    *,
    candidate_ids: Sequence[str] = (),
) -> dict[str, Any]:
    point = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "x_fraction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "y_fraction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "evidence": {"type": "string"},
        },
        "required": ["x_fraction", "y_fraction", "confidence", "evidence"],
    }
    common = {
        "rest_symbols": {"type": "array", "items": point},
        "overall_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "comments": {"type": "string"},
    }
    if task_kind == "direct-localization":
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {"noteheads": {"type": "array", "items": point}, **common},
            "required": ["noteheads", "rest_symbols", "overall_confidence", "comments"],
        }
    if task_kind == "candidate-assisted-localization":
        candidate_item: dict[str, Any] = {"type": "string"}
        if candidate_ids:
            candidate_item["enum"] = list(candidate_ids)
        selected_candidate = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "candidate_id": candidate_item,
                **point["properties"],
            },
            "required": [
                "candidate_id",
                "x_fraction",
                "y_fraction",
                "confidence",
                "evidence",
            ],
        }
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "selected_candidates": {
                    "type": "array",
                    "items": selected_candidate,
                },
                "missing_noteheads": {"type": "array", "items": point},
                **common,
            },
            "required": [
                "selected_candidates",
                "missing_noteheads",
                "rest_symbols",
                "overall_confidence",
                "comments",
            ],
        }
    raise ValueError(f"Unsupported task_kind: {task_kind}")


def localization_prompts(
    task_kind: TaskKind,
    *,
    image_roles: Sequence[str],
    candidate_ids: Sequence[str] = (),
    candidate_points: Sequence[tuple[str, float, float]] = (),
) -> tuple[str, str]:
    system_prompt = (
        "You are a precise music-notation visual localizer. Perform notehead localization only. "
        "The source is a single-line handwritten melody, not polyphonic keyboard notation: "
        "normally one stem or onset has one notehead, and apparent vertical stacks are usually "
        "staff intersections, ink joins, or noise rather than chords. Trace the musical symbols "
        "left-to-right and use stem, flag, beam, and rest grammar to test each proposed head. "
        "A notehead is the filled or hollow oval head of a pitched note. Never classify stems, "
        "beams, flags, accidentals, barlines, staff lines, clefs, text, or rests as noteheads. "
        "Do not infer or report pitches, rhythms, durations, or an expected note count. "
        "Return only the requested strict JSON, and make a best-effort localization even when "
        "the image is hard."
    )
    role_lines = [f"- {role}: {_role_explanation(role)}" for role in image_roles]
    shared = (
        "First identify every visible rest in the target measure and return its visual center in "
        "rest_symbols; return an empty array when there is no rest. Never return the same symbol "
        "as both a rest and a notehead. Then localize the visual centers of noteheads. Coordinates "
        "are fractions "
        "of the raw target image width and height, measured from its top-left corner. Explain each "
        "choice briefly in its evidence field. The images arrive in this exact order and have "
        "these roles:\n" + "\n".join(role_lines)
    )
    if task_kind == "direct-localization":
        user_prompt = (
            shared + "\nReturn noteheads in strict left-to-right order. Include every visible "
            "notehead you can support, but do not include non-notehead symbols. Do not guess "
            "a target count."
        )
    elif task_kind == "candidate-assisted-localization":
        ids = ", ".join(candidate_ids) if candidate_ids else "none"
        references = "; ".join(
            f"{candidate_id}=({x_fraction:.6f}, {y_fraction:.6f})"
            for candidate_id, x_fraction, y_fraction in candidate_points
        )
        user_prompt = (
            shared
            + "\nChoose only candidate labels whose marked center is correctly centered on a true "
            "notehead, then refine its x_fraction and y_fraction to the visual center of the "
            "notehead rather than copying the raw marker center. Available candidate IDs: "
            + ids
            + ". Blind candidate reference centers (x_fraction, y_fraction; starting points only, "
            "not presumed correct): "
            + (references or "none")
            + ". If a true notehead has no correctly centered candidate, add its center to "
            "missing_noteheads. Do not force a poor candidate match, and do not guess a target "
            "count."
        )
    else:
        raise ValueError(f"Unsupported task_kind: {task_kind}")
    return system_prompt, user_prompt


def localization_fixture_key(
    *,
    images: Sequence[RoleImage],
    system_prompt: str,
    user_prompt: str,
    schema: dict[str, Any],
    context: dict[str, Any],
    provider: Provider,
    model: str,
    reasoning_effort: str,
    image_detail: str,
    max_output_tokens: int,
) -> str:
    digest = hashlib.sha256()
    for image in images:
        _hash_component(digest, {"role": image.role, "bytes_sha256": _hash_bytes(image.path)})
    for value in (
        system_prompt,
        user_prompt,
        schema,
        context,
        provider,
        model,
        reasoning_effort,
        image_detail,
        max_output_tokens,
    ):
        _hash_component(digest, value)
    return digest.hexdigest()[:24]


def parse_and_validate_payload(
    raw_response: str,
    request: LocalizationRequest,
) -> dict[str, Any]:
    payload = _parse_json_object(raw_response)
    if request.task_kind == "direct-localization":
        _require_exact_keys(
            payload,
            {"noteheads", "rest_symbols", "overall_confidence", "comments"},
            "payload",
        )
        points = _validate_points(payload["noteheads"], "noteheads", require_sorted=True)
        rests = _validate_points(payload["rest_symbols"], "rest_symbols", require_sorted=True)
        payload = {**payload, "noteheads": points, "rest_symbols": rests}
    else:
        _require_exact_keys(
            payload,
            {
                "selected_candidates",
                "missing_noteheads",
                "rest_symbols",
                "overall_confidence",
                "comments",
            },
            "payload",
        )
        selected = payload["selected_candidates"]
        if not isinstance(selected, list):
            raise ValueError("selected_candidates must be an array")
        selected_points: list[dict[str, Any]] = []
        selected_ids: list[str] = []
        expected = {
            "candidate_id",
            "x_fraction",
            "y_fraction",
            "confidence",
            "evidence",
        }
        for index, item in enumerate(selected):
            if not isinstance(item, dict):
                raise ValueError(f"selected_candidates[{index}] must be an object")
            _require_exact_keys(item, expected, f"selected_candidates[{index}]")
            if not isinstance(item["candidate_id"], str):
                raise ValueError(f"selected_candidates[{index}].candidate_id must be a string")
            _validate_fraction(item["x_fraction"], f"selected_candidates[{index}].x_fraction")
            _validate_fraction(item["y_fraction"], f"selected_candidates[{index}].y_fraction")
            _validate_fraction(item["confidence"], f"selected_candidates[{index}].confidence")
            if not isinstance(item["evidence"], str):
                raise ValueError(f"selected_candidates[{index}].evidence must be a string")
            selected_ids.append(item["candidate_id"])
            selected_points.append(item)
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("selected_candidates must not contain duplicate candidate IDs")
        invalid = sorted(set(selected_ids) - set(request.candidate_ids))
        if invalid:
            raise ValueError(f"Unknown selected candidate IDs: {invalid}")
        if any(
            float(left["x_fraction"]) > float(right["x_fraction"])
            for left, right in zip(selected_points, selected_points[1:], strict=False)
        ):
            raise ValueError("selected_candidates must be ordered left-to-right by x_fraction")
        missing = _validate_points(
            payload["missing_noteheads"], "missing_noteheads", require_sorted=True
        )
        rests = _validate_points(payload["rest_symbols"], "rest_symbols", require_sorted=True)
        payload = {
            **payload,
            "selected_candidates": selected_points,
            "missing_noteheads": missing,
            "rest_symbols": rests,
        }
    _validate_fraction(payload["overall_confidence"], "overall_confidence")
    if not isinstance(payload["comments"], str):
        raise ValueError("comments must be a string")
    return payload


def predicted_points(
    request: LocalizationRequest,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    if request.task_kind == "direct-localization":
        return list(payload["noteheads"])
    points = [
        {
            **item,
            "candidate_id": item["candidate_id"],
        }
        for item in payload["selected_candidates"]
    ]
    points.extend(payload["missing_noteheads"])
    return sorted(points, key=lambda item: (item["x_fraction"], item["y_fraction"]))


def evaluate_coordinate_localization(
    predicted: Sequence[dict[str, Any]],
    ground_truth: dict[str, Any],
    *,
    raw_image_size: tuple[float, float],
    staff_lines_y_px: Sequence[float],
) -> dict[str, Any]:
    width, height = raw_image_size
    spacing = _staff_spacing(staff_lines_y_px)
    x_tolerance = 0.75 * spacing
    y_tolerance = 0.25 * spacing
    predicted_px = [
        {
            "index": index,
            "x": float(point["x_fraction"]) * width,
            "y": float(point["y_fraction"]) * height,
            "candidate_id": point.get("candidate_id"),
        }
        for index, point in enumerate(predicted)
    ]
    truth_px = [
        {
            "index": index,
            "id": item.get("id", f"n{index + 1:03d}"),
            "x": float(item["center"]["x"]),
            "y": float(item["center"]["y"]),
        }
        for index, item in enumerate(ground_truth.get("noteheads", []))
    ]
    region_matching = _match_annotation_regions(predicted_px, truth_px, ground_truth)
    center_matching = _match_points(
        predicted_px,
        truth_px,
        x_tolerance=x_tolerance,
        y_tolerance=y_tolerance,
    )
    tp = len(region_matching)
    fp = len(predicted_px) - tp
    fn = len(truth_px) - tp
    return {
        "kind": "coordinate",
        "localization": {
            "matching": "prediction inside authoritative human annotation ellipse",
            "annotation_ellipse_margin": ANNOTATION_ELLIPSE_MARGIN,
            "predicted_count": len(predicted_px),
            "ground_truth_count": len(truth_px),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn),
            "f1": _f1(tp, fp, fn),
            "exact_count": len(predicted_px) == len(truth_px),
            "assignments": region_matching,
        },
        "pitch": _coordinate_pitch_metrics(
            predicted_px,
            ground_truth,
            staff_lines_y_px=staff_lines_y_px,
        ),
        "strict_center_diagnostic": {
            "matching": "x/y center tolerance; diagnostic only",
            "staff_spacing_px": round(spacing, 6),
            "x_tolerance_px": round(x_tolerance, 6),
            "y_tolerance_px": round(y_tolerance, 6),
            "tp": len(center_matching),
            "fp": len(predicted_px) - len(center_matching),
            "fn": len(truth_px) - len(center_matching),
            "assignments": center_matching,
        },
    }


def evaluate_candidate_coverage(
    candidate_artifact: dict[str, Any] | None,
    ground_truth: dict[str, Any],
    *,
    staff_lines_y_px: Sequence[float],
) -> dict[str, Any] | None:
    if candidate_artifact is None:
        return None
    _staff_spacing(staff_lines_y_px)
    candidates = [
        {
            "index": index,
            "id": item.get("id", f"c{index + 1:03d}"),
            "x": float(item["center"]["x"]),
            "y": float(item["center"]["y"]),
        }
        for index, item in enumerate(candidate_artifact.get("candidates", []))
        if isinstance(item, dict) and isinstance(item.get("center"), dict)
    ]
    truth = [
        {
            "index": index,
            "id": item.get("id", f"n{index + 1:03d}"),
            "x": float(item["center"]["x"]),
            "y": float(item["center"]["y"]),
        }
        for index, item in enumerate(ground_truth.get("noteheads", []))
    ]
    matches = _match_annotation_regions(candidates, truth, ground_truth)
    pitch_correct = 0
    pitch_assignments = []
    noteheads = ground_truth.get("noteheads", [])
    for assignment in matches:
        candidate = candidates[assignment["predicted_index"]]
        notehead = noteheads[assignment["ground_truth_index"]]
        predicted_pitch = treble_pitch_for_y(candidate["y"], staff_lines_y_px)
        expected_pitch = _naturalize_pitch_name(str(notehead["pitch"]))
        correct = predicted_pitch == expected_pitch
        pitch_correct += int(correct)
        pitch_assignments.append(
            {
                "candidate_id": candidate["id"],
                "ground_truth_id": notehead.get("id"),
                "predicted_natural_pitch": predicted_pitch,
                "ground_truth_natural_pitch": expected_pitch,
                "correct": correct,
            }
        )
    return {
        "candidate_count": len(candidates),
        "ground_truth_count": len(truth),
        "covered_ground_truth_count": len(matches),
        "coverage": _ratio(len(matches), len(truth)),
        "annotation_ellipse_margin": ANNOTATION_ELLIPSE_MARGIN,
        "assignments": matches,
        "matched_pitch_correct_count": pitch_correct,
        "matched_pitch_accuracy": _ratio(pitch_correct, len(matches)),
        "pitch_assignments": pitch_assignments,
    }


def treble_pitch_for_y(y_px: float, staff_lines_y_px: Sequence[float]) -> str:
    """Map a y coordinate to the nearest treble-clef diatonic staff position."""
    spacing = _staff_spacing(staff_lines_y_px)
    half_steps_down = _round_half_away_from_zero(
        (float(y_px) - float(staff_lines_y_px[0])) / (spacing / 2.0)
    )
    top_line_diatonic = _diatonic_number("F", 5)
    return _pitch_from_diatonic_number(top_line_diatonic - half_steps_down)


def _coordinate_pitch_metrics(
    predicted_px: Sequence[dict[str, Any]],
    ground_truth: dict[str, Any],
    *,
    staff_lines_y_px: Sequence[float],
) -> dict[str, Any]:
    ordered_predicted = sorted(predicted_px, key=lambda item: (item["x"], item["y"]))
    predicted_pitches = [
        treble_pitch_for_y(float(item["y"]), staff_lines_y_px) for item in ordered_predicted
    ]
    expected_pitches = [
        _naturalize_pitch_name(str(item["pitch"])) for item in ground_truth.get("noteheads", [])
    ]
    correct = sum(
        left == right for left, right in zip(predicted_pitches, expected_pitches, strict=False)
    )
    denominator = max(len(predicted_pitches), len(expected_pitches))
    return {
        "mapping": "treble-clef nearest diatonic staff position; accidentals ignored",
        "predicted_natural_pitches": predicted_pitches,
        "ground_truth_natural_pitches": expected_pitches,
        "correct_count": correct,
        "accuracy": _ratio(correct, denominator),
        "exact_count": len(predicted_pitches) == len(expected_pitches),
        "exact_ordered_pitches": predicted_pitches == expected_pitches,
    }


def evaluate_event_pitch_localization(
    predicted: Sequence[dict[str, Any]],
    canonical: dict[str, Any],
    *,
    global_measure_index: int,
    raw_image_size: tuple[float, float],
    staff_lines_y_px: Sequence[float],
) -> dict[str, Any]:
    _, height = raw_image_size
    ordered = sorted(predicted, key=lambda item: (item["x_fraction"], item["y_fraction"]))
    predicted_pitches = [
        treble_pitch_for_y(float(item["y_fraction"]) * height, staff_lines_y_px) for item in ordered
    ]
    notes = [
        item
        for item in canonical.get("notes", [])
        if isinstance(item, dict) and int(item.get("measure", -1)) == global_measure_index
    ]
    expected_pitches = [_canonical_natural_pitch(item) for item in notes]
    positional_matches = sum(
        left == right for left, right in zip(predicted_pitches, expected_pitches, strict=False)
    )
    denominator = max(len(predicted_pitches), len(expected_pitches))
    return {
        "kind": "event_pitch",
        "accidentals_ignored": True,
        "predicted_count": len(predicted_pitches),
        "ground_truth_count": len(expected_pitches),
        "predicted_natural_pitches": predicted_pitches,
        "ground_truth_natural_pitches": expected_pitches,
        "exact_count": len(predicted_pitches) == len(expected_pitches),
        "ordered_pitch_accuracy": _ratio(positional_matches, denominator),
        "exact_ordered_pitches": predicted_pitches == expected_pitches,
    }


def default_model_for_provider(provider: Provider) -> str:
    return DEFAULT_OPENAI_MODEL if provider == "openai" else DEFAULT_GEMINI_MODEL


def _evaluate_request(
    request: LocalizationRequest,
    payload: dict[str, Any],
    *,
    base: dict[str, Any],
) -> dict[str, Any]:
    points = predicted_points(request, payload)
    source_context = request.record["source_context"]
    size = _raw_image_size(source_context)
    staff_lines = source_context["staff_lines_y_px"]
    if request.coordinate_ground_truth_path is not None:
        if not request.coordinate_ground_truth_path.exists():
            return {
                **base,
                "status": "no_gt",
                "gt_status": "missing_coordinate_gt",
                "parsed_payload": payload,
                "evaluation": None,
                "candidate_coverage": None,
            }
        ground_truth = _load_json(request.coordinate_ground_truth_path)
        evaluation = evaluate_coordinate_localization(
            points,
            ground_truth,
            raw_image_size=size,
            staff_lines_y_px=staff_lines,
        )
        coverage = evaluate_candidate_coverage(
            request.candidate_artifact,
            ground_truth,
            staff_lines_y_px=staff_lines,
        )
        return {
            **base,
            "status": "evaluated",
            "gt_status": "coordinate",
            "parsed_payload": payload,
            "evaluation": evaluation,
            "candidate_coverage": coverage,
        }
    if not request.canonical_ground_truth_path.exists():
        return {
            **base,
            "status": "no_gt",
            "gt_status": "missing_canonical_gt",
            "parsed_payload": payload,
            "evaluation": None,
            "candidate_coverage": None,
        }
    canonical = _load_json(request.canonical_ground_truth_path)
    notes = [
        note
        for note in canonical.get("notes", [])
        if isinstance(note, dict)
        and int(note.get("measure", -1)) == int(request.record["global_measure_index"])
    ]
    if not notes:
        return {
            **base,
            "status": "no_gt",
            "gt_status": "no_canonical_notes_for_measure",
            "parsed_payload": payload,
            "evaluation": None,
            "candidate_coverage": None,
        }
    evaluation = evaluate_event_pitch_localization(
        points,
        canonical,
        global_measure_index=int(request.record["global_measure_index"]),
        raw_image_size=size,
        staff_lines_y_px=staff_lines,
    )
    return {
        **base,
        "status": "evaluated",
        "gt_status": "canonical_event_pitch",
        "parsed_payload": payload,
        "evaluation": evaluation,
        "candidate_coverage": None,
    }


def _build_transport(
    *,
    provider: Provider,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    timeout_seconds: int,
    provider_client: Any,
) -> LocalizationTransport:
    if provider == "openai":
        return OpenAILocalizationTransport(
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
            client=provider_client,
        )
    return GeminiLocalizationTransport(
        model=model,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
        client=provider_client,
    )


def _write_fixture(
    path: Path,
    *,
    request: LocalizationRequest,
    response: LocalizationResponse,
    parsed_payload: dict[str, Any] | None,
    parse_error: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        path,
        {
            "schema_version": 1,
            "fixture_key": request.fixture_key,
            "experiment_id": request.experiment_id,
            "task_kind": request.task_kind,
            "provider": request.provider,
            "model": request.model,
            "openai_reasoning_effort": request.reasoning_effort,
            "image_detail": request.image_detail,
            "max_output_tokens": request.max_output_tokens,
            "response_id": response.response_id,
            "usage": response.usage,
            "raw_response": response.raw_response,
            "provider_response": response.provider_response,
            "parsed_payload": parsed_payload,
            "parse_error": parse_error,
        },
    )


def _write_journal(
    *,
    batch_dir: Path,
    ordinal: int,
    manifest_path: Path,
    out_dir: Path,
    request: LocalizationRequest | None,
    record: dict[str, Any],
    result: dict[str, Any],
    response: LocalizationResponse | None,
    fixture_path: Path | None,
    max_output_tokens: int,
    timeout_seconds: int,
    force: bool,
) -> Path:
    experiment_id = str(record.get("experiment_id", f"record-{ordinal}"))
    suffix = request.fixture_key if request else _hash_json(record)[:12]
    journal = batch_dir / "journals" / f"{ordinal:04d}__{_safe_name(experiment_id)}__{suffix}"
    images_dir = journal / "images"
    prompts_dir = journal / "prompts"
    inputs_dir = journal / "inputs"
    images_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(exist_ok=True)
    inputs_dir.mkdir(exist_ok=True)
    _write_json(inputs_dir / "manifest_record.json", record)
    if request is not None:
        for index, image in enumerate(request.images, start=1):
            target = images_dir / (
                f"{index:02d}__{_safe_name(image.role)}{image.path.suffix or '.img'}"
            )
            shutil.copy2(image.path, target)
        shutil.copy2(request.context_path, inputs_dir / "context_source.json")
        if request.candidate_artifact_path and request.candidate_artifact_path.exists():
            shutil.copy2(request.candidate_artifact_path, inputs_dir / "candidate_artifact.json")
        if request.coordinate_ground_truth_path and request.coordinate_ground_truth_path.exists():
            shutil.copy2(
                request.coordinate_ground_truth_path,
                inputs_dir / "coordinate_ground_truth.json",
            )
        if request.canonical_ground_truth_path.exists():
            shutil.copy2(
                request.canonical_ground_truth_path,
                inputs_dir / "canonical_ground_truth.json",
            )
        (prompts_dir / "system.txt").write_text(request.system_prompt, encoding="utf-8")
        (prompts_dir / "user.txt").write_text(request.user_prompt, encoding="utf-8")
        _write_json(prompts_dir / "schema.json", request.schema)
        _write_json(
            inputs_dir / "context.json",
            {
                "manifest_source_context": record["source_context"],
                "context": request.context,
                "candidate_artifact": request.candidate_artifact,
            },
        )
    raw_response = response.raw_response if response else ""
    (journal / "raw_response.txt").write_text(raw_response, encoding="utf-8")
    parsed = result.get("parsed_payload")
    _write_json(journal / "parsed_payload.json", parsed)
    _write_json(
        journal / "response_metadata.json",
        {
            "response_id": response.response_id if response else None,
            "usage": response.usage if response else None,
            "provider_response": response.provider_response if response else None,
        },
    )
    _write_json(journal / "evaluation.json", result.get("evaluation"))
    _write_json(journal / "candidate_coverage.json", result.get("candidate_coverage"))
    _write_json(journal / "result.json", result)
    if fixture_path and fixture_path.exists():
        shutil.copy2(fixture_path, journal / "fixture.json")
    config = {
        "provider": request.provider if request else None,
        "model": request.model if request else None,
        "openai_reasoning_effort": request.reasoning_effort if request else None,
        "image_detail": request.image_detail if request else DEFAULT_IMAGE_DETAIL,
        "max_output_tokens": max_output_tokens,
        "timeout_seconds": timeout_seconds,
        "force": force,
        "fixture_key": request.fixture_key if request else None,
    }
    _write_json(journal / "config.json", config)
    replay = (
        _replay_command(
            out_dir=out_dir,
            manifest_path=manifest_path,
            request=request,
            experiment_id=experiment_id,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
        )
        if request is not None
        else None
    )
    _write_json(
        journal / "experiment.json",
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_id": experiment_id,
            "fixture_key": request.fixture_key if request else None,
            "result": result,
            "config": config,
            "replay_command": replay,
        },
    )
    (journal / "README.md").write_text(
        f"# {experiment_id}\n\nStatus: `{result['status']}`\n\n"
        + (f"Replay:\n\n```sh\n{replay}\n```\n" if replay else "Replay unavailable.\n"),
        encoding="utf-8",
    )
    return journal


def _batch_summary(
    *,
    run_id: str,
    batch_dir: Path,
    manifest_path: Path,
    provider: Provider,
    model: str,
    reasoning_effort: str,
    max_calls: int,
    max_output_tokens: int,
    timeout_seconds: int,
    force: bool,
    live_calls: int,
    results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    by_task = {
        task: _aggregate_results([item for item in results if item.get("task_kind") == task])
        for task in TASK_KINDS
        if any(item.get("task_kind") == task for item in results)
    }
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": max_calls == 0,
        "config": {
            "provider": provider,
            "model": model,
            "openai_reasoning_effort": reasoning_effort,
            "image_detail": DEFAULT_IMAGE_DETAIL,
            "max_calls": max_calls,
            "max_output_tokens": max_output_tokens,
            "timeout_seconds": timeout_seconds,
            "force": force,
        },
        "overall": {**_aggregate_results(results), "live_calls": live_calls},
        "by_task": by_task,
        "paths": {
            "batch_dir": str(batch_dir),
            "selected_records": str(batch_dir / "selected_records.jsonl"),
            "batch_manifest": str(batch_dir / "batch_manifest.jsonl"),
            "manifest": str(manifest_path),
        },
    }


def _aggregate_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(item.get("status")) for item in results)
    gt_statuses = Counter(str(item.get("gt_status")) for item in results)
    coordinate = [
        item["evaluation"]
        for item in results
        if isinstance(item.get("evaluation"), dict)
        and item["evaluation"].get("kind") == "coordinate"
    ]
    event_pitch = [
        item["evaluation"]
        for item in results
        if isinstance(item.get("evaluation"), dict)
        and item["evaluation"].get("kind") == "event_pitch"
    ]
    coverages = [
        item["candidate_coverage"]
        for item in results
        if isinstance(item.get("candidate_coverage"), dict)
    ]
    tp = sum(int(item["localization"]["tp"]) for item in coordinate)
    fp = sum(int(item["localization"]["fp"]) for item in coordinate)
    fn = sum(int(item["localization"]["fn"]) for item in coordinate)
    covered = sum(int(item["covered_ground_truth_count"]) for item in coverages)
    coverage_gt = sum(int(item["ground_truth_count"]) for item in coverages)
    raw_candidate_pitch_correct = sum(
        int(item["matched_pitch_correct_count"]) for item in coverages
    )
    raw_candidate_pitch_total = sum(int(item["covered_ground_truth_count"]) for item in coverages)
    return {
        "records": len(results),
        "statuses": dict(sorted(statuses.items())),
        "gt_statuses": dict(sorted(gt_statuses.items())),
        "coordinate_metrics": {
            "records": len(coordinate),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn),
            "f1": _f1(tp, fp, fn),
            "exact_count_rate": _mean(
                float(item["localization"]["exact_count"]) for item in coordinate
            ),
            "pitch_accuracy": _mean(float(item["pitch"]["accuracy"]) for item in coordinate),
        },
        "candidate_coverage": {
            "records": len(coverages),
            "covered_ground_truth_count": covered,
            "ground_truth_count": coverage_gt,
            "coverage": _ratio(covered, coverage_gt),
            "matched_pitch_correct_count": raw_candidate_pitch_correct,
            "matched_pitch_total": raw_candidate_pitch_total,
            "matched_pitch_accuracy": _ratio(
                raw_candidate_pitch_correct, raw_candidate_pitch_total
            ),
        },
        "event_pitch_metrics": {
            "records": len(event_pitch),
            "exact_count_rate": _mean(float(item["exact_count"]) for item in event_pitch),
            "ordered_pitch_accuracy": _mean(
                float(item["ordered_pitch_accuracy"]) for item in event_pitch
            ),
        },
    }


def _batch_readme(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    return (
        f"# Notehead localization batch {summary['run_id']}\n\n"
        f"Provider/model: `{summary['config']['provider']}` / `{summary['config']['model']}`\n\n"
        f"Records: {overall['records']}; live calls: {overall['live_calls']}; "
        f"statuses: `{json.dumps(overall['statuses'], sort_keys=True)}`.\n\n"
        "Ground truth is used only after provider output parsing. See `journals/` for exact "
        "inputs, prompts, schemas, responses, parsed payloads, and evaluations.\n"
    )


def _base_result(request: LocalizationRequest, fixture_path: Path) -> dict[str, Any]:
    return {
        "experiment_id": request.experiment_id,
        "fixture_key": request.fixture_key,
        "task_kind": request.task_kind,
        "slug": request.record["slug"],
        "system_index": int(request.record["system_index"]),
        "system_measure_index": int(request.record["system_measure_index"]),
        "global_measure_index": int(request.record["global_measure_index"]),
        "provider": request.provider,
        "model": request.model,
        "openai_reasoning_effort": request.reasoning_effort,
        "fixture": str(fixture_path),
    }


def _validate_manifest_record(record: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "experiment_id",
        "task_kind",
        "slug",
        "system_index",
        "system_measure_index",
        "global_measure_index",
        "context_path",
        "images",
        "candidate_artifact_path",
        "coordinate_ground_truth_path",
        "canonical_ground_truth_path",
        "source_context",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"Manifest record missing fields: {missing}")
    if record["task_kind"] not in TASK_KINDS:
        raise ValueError(f"Invalid task_kind: {record['task_kind']}")
    if not isinstance(record["images"], list) or not record["images"]:
        raise ValueError("Manifest images must be a non-empty array")
    for index, image in enumerate(record["images"]):
        if not isinstance(image, dict) or set(image) != {"role", "path"}:
            raise ValueError(f"images[{index}] must contain exactly role and path")
        if not all(isinstance(image[key], str) and image[key] for key in ("role", "path")):
            raise ValueError(f"images[{index}] role/path must be non-empty strings")
    source_context = record["source_context"]
    if not isinstance(source_context, dict):
        raise ValueError("source_context must be an object")
    _raw_image_size(source_context)
    _staff_spacing(source_context.get("staff_lines_y_px", []))


def _validate_points(
    value: Any,
    field: str,
    *,
    require_sorted: bool,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    points: list[dict[str, Any]] = []
    expected = {"x_fraction", "y_fraction", "confidence", "evidence"}
    for index, point in enumerate(value):
        if not isinstance(point, dict):
            raise ValueError(f"{field}[{index}] must be an object")
        _require_exact_keys(point, expected, f"{field}[{index}]")
        for key in ("x_fraction", "y_fraction", "confidence"):
            _validate_fraction(point[key], f"{field}[{index}].{key}")
        if not isinstance(point["evidence"], str):
            raise ValueError(f"{field}[{index}].evidence must be a string")
        points.append(point)
    if require_sorted and any(
        float(left["x_fraction"]) > float(right["x_fraction"])
        for left, right in zip(points, points[1:], strict=False)
    ):
        raise ValueError(f"{field} must be ordered left-to-right by x_fraction")
    return points


def _validate_fraction(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1")


def _require_exact_keys(value: dict[str, Any], expected: set[str], field: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(f"{field} keys must be exactly {sorted(expected)}; got {sorted(actual)}")


def _match_points(
    predicted: Sequence[dict[str, Any]],
    truth: Sequence[dict[str, Any]],
    *,
    x_tolerance: float,
    y_tolerance: float,
) -> list[dict[str, Any]]:
    pairs = []
    for predicted_index, point in enumerate(predicted):
        for truth_index, target in enumerate(truth):
            dx = abs(float(point["x"]) - float(target["x"]))
            dy = abs(float(point["y"]) - float(target["y"]))
            if dx <= x_tolerance and dy <= y_tolerance:
                distance = math.hypot(dx / x_tolerance, dy / y_tolerance)
                pairs.append((distance, dx, dy, predicted_index, truth_index))
    used_predicted: set[int] = set()
    used_truth: set[int] = set()
    matches = []
    for distance, dx, dy, predicted_index, truth_index in sorted(pairs):
        if predicted_index in used_predicted or truth_index in used_truth:
            continue
        used_predicted.add(predicted_index)
        used_truth.add(truth_index)
        predicted_point = predicted[predicted_index]
        truth_point = truth[truth_index]
        matches.append(
            {
                "predicted_index": predicted_index,
                "candidate_id": predicted_point.get("candidate_id") or predicted_point.get("id"),
                "ground_truth_index": truth_index,
                "ground_truth_id": truth_point.get("id"),
                "dx_px": round(dx, 6),
                "dy_px": round(dy, 6),
                "normalized_distance": round(distance, 6),
            }
        )
    return sorted(matches, key=lambda item: item["predicted_index"])


def _match_annotation_regions(
    predicted: Sequence[dict[str, Any]],
    truth: Sequence[dict[str, Any]],
    ground_truth: dict[str, Any],
) -> list[dict[str, Any]]:
    noteheads = ground_truth.get("noteheads")
    if not isinstance(noteheads, list) or len(noteheads) != len(truth):
        raise ValueError("Coordinate GT noteheads do not match the normalized truth points")
    pairs = []
    for predicted_index, point in enumerate(predicted):
        for truth_index, target in enumerate(truth):
            center_x, center_y, radius_x, radius_y = _annotation_ellipse(noteheads[truth_index])
            normalized_x = (float(point["x"]) - center_x) / (radius_x * ANNOTATION_ELLIPSE_MARGIN)
            normalized_y = (float(point["y"]) - center_y) / (radius_y * ANNOTATION_ELLIPSE_MARGIN)
            normalized_distance = math.hypot(normalized_x, normalized_y)
            if normalized_distance <= 1.0:
                pairs.append(
                    (
                        normalized_distance,
                        abs(float(point["x"]) - center_x),
                        abs(float(point["y"]) - center_y),
                        predicted_index,
                        truth_index,
                        target,
                    )
                )
    used_predicted: set[int] = set()
    used_truth: set[int] = set()
    matches = []
    for distance, dx, dy, predicted_index, truth_index, target in sorted(
        pairs, key=lambda item: item[:5]
    ):
        if predicted_index in used_predicted or truth_index in used_truth:
            continue
        used_predicted.add(predicted_index)
        used_truth.add(truth_index)
        point = predicted[predicted_index]
        matches.append(
            {
                "predicted_index": predicted_index,
                "candidate_id": point.get("candidate_id") or point.get("id"),
                "ground_truth_index": truth_index,
                "ground_truth_id": target.get("id"),
                "dx_from_ellipse_center_px": round(dx, 6),
                "dy_from_ellipse_center_px": round(dy, 6),
                "normalized_ellipse_distance": round(distance, 6),
            }
        )
    return sorted(matches, key=lambda item: item["predicted_index"])


def _annotation_ellipse(notehead: dict[str, Any]) -> tuple[float, float, float, float]:
    geometry = notehead.get("annotation_geometry")
    if not isinstance(geometry, dict):
        raise ValueError("Coordinate GT notehead lacks authoritative annotation_geometry")
    bbox = geometry.get("bbox_px")
    if not isinstance(bbox, dict) or not all(
        key in bbox for key in ("left", "top", "right", "bottom")
    ):
        raise ValueError("annotation_geometry.bbox_px is incomplete")
    center_x = (float(bbox["left"]) + float(bbox["right"])) / 2.0
    center_y = (float(bbox["top"]) + float(bbox["bottom"])) / 2.0
    radius_x = float(geometry.get("radius_x_px", (float(bbox["right"]) - float(bbox["left"])) / 2))
    radius_y = float(geometry.get("radius_y_px", (float(bbox["bottom"]) - float(bbox["top"])) / 2))
    if radius_x <= 0 or radius_y <= 0:
        raise ValueError("annotation_geometry ellipse radii must be positive")
    return center_x, center_y, radius_x, radius_y


def _select_records(
    records: Sequence[dict[str, Any]],
    *,
    task_kinds: set[str] | None,
    slugs: set[str] | None,
    systems: set[int] | None,
    measures: set[int] | None,
    experiment_ids: set[str] | None,
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if (task_kinds is None or record.get("task_kind") in task_kinds)
        and (slugs is None or record.get("slug") in slugs)
        and (systems is None or int(record.get("system_index", -1)) in systems)
        and (measures is None or int(record.get("system_measure_index", -1)) in measures)
        and (experiment_ids is None or record.get("experiment_id") in experiment_ids)
    ]


def _candidate_ids(candidate_artifact: dict[str, Any] | None) -> tuple[str, ...]:
    if candidate_artifact is None:
        return ()
    candidates = candidate_artifact.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Candidate artifact candidates must be an array")
    ids = tuple(str(item["id"]) for item in candidates if isinstance(item, dict) and "id" in item)
    if len(ids) != len(candidates) or len(ids) != len(set(ids)):
        raise ValueError("Candidate artifact must contain unique candidate ids")
    for item in candidates:
        center = item.get("center")
        if not isinstance(center, dict) or not all(key in center for key in ("x", "y")):
            raise ValueError("Every candidate must contain center.x and center.y")
    return ids


def _candidate_prompt_points(
    candidate_artifact: dict[str, Any] | None,
    *,
    raw_image_size: tuple[float, float],
) -> tuple[tuple[str, float, float], ...]:
    if candidate_artifact is None:
        return ()
    width, height = raw_image_size
    points = []
    for item in candidate_artifact.get("candidates", []):
        normalized = item.get("normalized_center")
        if isinstance(normalized, dict) and all(key in normalized for key in ("x", "y")):
            x_fraction = float(normalized["x"])
            y_fraction = float(normalized["y"])
        else:
            center = item["center"]
            x_fraction = float(center["x"]) / width
            y_fraction = float(center["y"]) / height
        points.append((str(item["id"]), x_fraction, y_fraction))
    return tuple(points)


def _raw_image_size(source_context: dict[str, Any]) -> tuple[float, float]:
    value = source_context.get("raw_image_size")
    if isinstance(value, dict):
        width = value.get("width", value.get("width_px"))
        height = value.get("height", value.get("height_px"))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        width, height = value
    else:
        raise ValueError("source_context.raw_image_size must be [width, height] or an object")
    if isinstance(width, bool) or isinstance(height, bool):
        raise ValueError("raw_image_size values must be positive numbers")
    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
        raise ValueError("raw_image_size values must be positive numbers")
    if float(width) <= 0 or float(height) <= 0:
        raise ValueError("raw_image_size values must be positive numbers")
    return float(width), float(height)


def _staff_spacing(staff_lines: Sequence[float]) -> float:
    if len(staff_lines) != 5:
        raise ValueError(f"Exactly five staff_lines_y_px are required; got {staff_lines!r}")
    values = [float(value) for value in staff_lines]
    gaps = [right - left for left, right in zip(values, values[1:], strict=False)]
    if any(gap <= 0 for gap in gaps):
        raise ValueError(f"staff_lines_y_px must be strictly increasing: {staff_lines!r}")
    return sum(gaps) / len(gaps)


def _canonical_natural_pitch(note: dict[str, Any]) -> str:
    midi = int(note["pitch_midi"]) - int(note.get("accidental", 0))
    names = {0: "C", 2: "D", 4: "E", 5: "F", 7: "G", 9: "A", 11: "B"}
    name = names.get(midi % 12)
    if name is None:
        raise ValueError(f"Canonical pitch cannot be naturalized: {note}")
    return f"{name}{midi // 12 - 1}"


def _naturalize_pitch_name(value: str) -> str:
    match = re.fullmatch(r"([A-Ga-g])(?:#{1,2}|b{1,2})?(-?\d+)", value.strip())
    if not match:
        raise ValueError(f"Unsupported coordinate-GT pitch name: {value!r}")
    return f"{match.group(1).upper()}{match.group(2)}"


def _diatonic_number(name: str, octave: int) -> int:
    return octave * 7 + ("C", "D", "E", "F", "G", "A", "B").index(name)


def _pitch_from_diatonic_number(value: int) -> str:
    names = ("C", "D", "E", "F", "G", "A", "B")
    octave, index = divmod(value, 7)
    return f"{names[index]}{octave}"


def _round_half_away_from_zero(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def _role_explanation(role: str) -> str:
    normalized = role.lower().replace("_", "-")
    if "candidate" in normalized or "overlay" in normalized or "gallery" in normalized:
        return "candidate labels and enlarged patches; raw centers are aids, not final answers"
    if "detail" in normalized:
        return "enlarged grayscale target measure with the same framing as the raw target"
    if "binary" in normalized:
        return "thresholded enlarged target measure with the same framing as the raw target"
    if "raw" in normalized or "source" in normalized or "measure" in normalized:
        return "unannotated target measure and the coordinate reference image"
    if "staff" in normalized:
        return "staff-focused view of the same target measure"
    if "context" in normalized or "system" in normalized:
        return (
            "wider visual context; red ticks in the white margins mark the target measure "
            "boundaries and do not touch or alter the music"
        )
    return "supporting view of the same target measure"


def _replay_command(
    *,
    out_dir: Path,
    manifest_path: Path,
    request: LocalizationRequest,
    experiment_id: str,
    max_output_tokens: int,
    timeout_seconds: int,
) -> str:
    parts = [
        "uv",
        "run",
        "python",
        "scripts/run_vlm_notehead_localization_spike.py",
        str(out_dir),
        "--manifest",
        str(manifest_path),
        "--experiment-id",
        experiment_id,
        "--provider",
        request.provider,
        "--model",
        request.model,
        "--max-calls",
        "1",
        "--max-output-tokens",
        str(max_output_tokens),
        "--timeout",
        str(timeout_seconds),
        "--run-id",
        f"replay-{_safe_name(experiment_id)}",
        "--force",
    ]
    if request.provider == "openai":
        parts.extend(["--openai-reasoning-effort", request.reasoning_effort])
    return " ".join(_shell_quote(part) for part in parts)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Manifest record at {path}:{line_number} must be an object")
        records.append(value)
    return records


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resolve_path(out_dir: Path, manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == out_dir.name:
        candidates = [out_dir.parent / path, manifest_path.parent / path, out_dir / path, path]
    else:
        candidates = [manifest_path.parent / path, out_dir / path, path, out_dir.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return out_dir / path


def _optional_resolved_path(
    out_dir: Path,
    manifest_path: Path,
    value: Any,
) -> Path | None:
    if value in (None, ""):
        return None
    return _resolve_path(out_dir, manifest_path, str(value))


def _parse_json_object(raw_response: str) -> dict[str, Any]:
    text = raw_response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Provider response must be a JSON object")
    return value


def _openai_response_text(response: Any) -> str:
    if isinstance(response, dict):
        output_text = response.get("output_text")
        if isinstance(output_text, str) and output_text:
            return output_text
        output = response.get("output", [])
    else:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text:
            return output_text
        output = getattr(response, "output", [])
    for item in output or []:
        content = (
            item.get("content", []) if isinstance(item, dict) else getattr(item, "content", [])
        )
        for part in content or []:
            if isinstance(part, dict):
                text = part.get("text")
            else:
                text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                return text
    # Preserve incomplete/refused responses in fixtures and journals. The batch
    # parser will mark the empty text as a record-local parse failure while the
    # raw provider response retains status, usage, and incomplete details.
    return ""


def _gemini_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(response, dict):
        text = response.get("text")
    if isinstance(text, str) and text:
        return text
    try:
        candidates = response["candidates"] if isinstance(response, dict) else response.candidates
        candidate = candidates[0]
        content = candidate["content"] if isinstance(candidate, dict) else candidate.content
        parts = content["parts"] if isinstance(content, dict) else content.parts
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Gemini response contained no parseable text: {exc}") from exc
    joined = "".join(
        (part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")) or ""
        for part in parts
    )
    if not joined:
        raise RuntimeError("Gemini response contained no text parts")
    return joined


def _response_id(response: Any) -> str | None:
    if isinstance(response, dict):
        value = response.get("id") or response.get("response_id")
    else:
        value = getattr(response, "id", None) or getattr(response, "response_id", None)
    return str(value) if value else None


def _response_usage(response: Any) -> dict[str, Any] | None:
    if isinstance(response, dict):
        usage = response.get("usage") or response.get("usage_metadata")
    else:
        usage = getattr(response, "usage", None) or getattr(response, "usage_metadata", None)
    value = _jsonable(usage)
    return value if isinstance(value, dict) else None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "to_json_dict"):
        return _jsonable(value.to_json_dict())
    return repr(value)


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


def _data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{_mime_type(path)};base64,{encoded}"


def _mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or "image/png"


def _hash_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _hash_component(digest: Any, value: Any) -> None:
    digest.update(b"\x1f")
    digest.update(_canonical_json(value).encode("utf-8"))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return round(2 * precision * recall / (precision + recall), 6) if precision + recall else 0.0


def _mean(values: Sequence[float] | Any) -> float:
    items = list(values)
    return round(sum(items) / len(items), 6) if items else 0.0


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return safe or "run"


def _shell_quote(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:=+-]+", value):
        return value
    return json.dumps(value)


if __name__ == "__main__":
    raise SystemExit(main())
