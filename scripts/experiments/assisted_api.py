"""Budgeted Responses API execution of a sealed, source-only assisted packet.

The caller supplies credentials explicitly; this module never reads credentials.
There are no retries. Cost accounting is a conservative estimate, not an invoice.
Only the two versioned, bounded packet SourceTools operations are exposed to the model.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import math
import os
import re
import shutil
import socket
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Callable

from scripts.experiments import astra_source_tools as trusted_tools
from scripts.experiments import run_astra_baseline_reproduction as runner

API_BASE = "https://api.openai.com/v1"
MAX_OUTPUT_TOKENS = 25_000
MAX_INPUT_TOKENS = 272_000
MILLION = Decimal(1_000_000)
Transport = Callable[[str, dict, float], dict]


class ApiFailure(RuntimeError):
    """A sanitized execution failure, with optional private transport evidence."""

    def __init__(self, message: str, *, private_body: bytes | None = None):
        super().__init__(message)
        self.private_body = private_body


class BudgetExhausted(ApiFailure):
    pass


class ProtocolViolation(ApiFailure):
    pass


def decimal(value: object) -> Decimal:
    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise ValueError("Expected a finite nonnegative dollar amount") from exc
    if not result.is_finite() or result < 0:
        raise ValueError("Expected a finite nonnegative dollar amount")
    return result


@dataclass(frozen=True)
class ApiPricing:
    model: str
    input_per_million_usd: str
    cache_write_per_million_usd: str
    output_per_million_usd: str
    source_url: str
    verified_at: str
    max_input_tokens: int = MAX_INPUT_TOKENS
    cached_input_per_million_usd: str = "1"

    def validate(self, model: str) -> None:
        if (
            self.model != model
            or not self.source_url.startswith("https://developers.openai.com/")
            or not self.verified_at
            or decimal(self.cached_input_per_million_usd) > self.input_rate
            or type(self.max_input_tokens) is not int
            or not 0 < self.max_input_tokens <= MAX_INPUT_TOKENS
            or any(
                decimal(rate) <= 0
                for rate in (
                    self.input_per_million_usd,
                    self.cache_write_per_million_usd,
                    self.output_per_million_usd,
                    self.cached_input_per_million_usd,
                )
            )
        ):
            raise ValueError("Verified conservative pricing for the exact model is required")

    @property
    def input_rate(self) -> Decimal:
        return max(decimal(self.input_per_million_usd), decimal(self.cache_write_per_million_usd))

    @property
    def output_rate(self) -> Decimal:
        return decimal(self.output_per_million_usd)


ASTRA_PRICING = ApiPricing(
    model="gpt-6-astra",
    input_per_million_usd="10",
    cache_write_per_million_usd="25",
    output_per_million_usd="50",
    source_url="https://developers.openai.com/api/docs/pricing",
    verified_at="2026-10-29",
)


class Budget:
    def __init__(self, dollars: object, pricing: ApiPricing):
        self.limit = None if dollars is None else decimal(dollars)
        if self.limit is not None and self.limit <= 0:
            raise ValueError("A positive explicit stage budget is required")
        self.pricing = pricing
        self.entries: list[dict] = []

    @property
    def charged(self) -> Decimal:
        return sum((decimal(row["charged_usd"]) for row in self.entries), Decimal(0))

    def reserve(self, input_tokens: int, max_output_tokens: int, min_output_tokens: int) -> dict:
        if type(input_tokens) is not int or not 0 <= input_tokens <= self.pricing.max_input_tokens:
            raise ProtocolViolation("Input count is invalid or exceeds the priced context tier")
        input_cost = Decimal(input_tokens) * self.pricing.input_rate / MILLION
        output_tokens = max_output_tokens
        if self.limit is not None:
            available = self.limit - self.charged - input_cost
            output_tokens = min(
                max_output_tokens,
                int(
                    (available * MILLION / self.pricing.output_rate).to_integral_value(ROUND_FLOOR)
                ),
            )
        if output_tokens < min_output_tokens:
            raise BudgetExhausted("Stage budget cannot reserve the next bounded generation")
        reserved = input_cost + Decimal(output_tokens) * self.pricing.output_rate / MILLION
        row = {
            "request": len(self.entries) + 1,
            "input_tokens_counted": input_tokens,
            "max_output_tokens": output_tokens,
            "reserved_usd": str(reserved),
            "charged_usd": str(reserved),
            "status": "reserved_unknown_outcome",
        }
        self.entries.append(row)
        return row

    def reconcile(self, row: dict, usage: object) -> None:
        if not isinstance(usage, dict):
            raise ProtocolViolation("Missing response usage; full reservation retained")
        actual_input, actual_output = usage.get("input_tokens"), usage.get("output_tokens")
        details = usage.get("input_tokens_details", {})
        cached = details.get("cached_tokens", 0) if isinstance(details, dict) else None
        if (
            type(actual_input) is not int
            or type(actual_output) is not int
            or not 0 <= actual_input <= row["input_tokens_counted"]
            or not 0 <= actual_output <= row["max_output_tokens"]
            or usage.get("total_tokens") != actual_input + actual_output
            or type(cached) is not int
            or not 0 <= cached <= actual_input
        ):
            raise ProtocolViolation("Unexpected response usage; full reservation retained")
        # Honor explicitly reported cache hits, but reserve unknown cache writes at the maximum.
        charged = (
            Decimal(actual_input - cached) * self.pricing.input_rate
            + Decimal(cached) * decimal(self.pricing.cached_input_per_million_usd)
            + Decimal(actual_output) * self.pricing.output_rate
        ) / MILLION
        row.update(
            status="usage_reconciled_conservative",
            charged_usd=str(charged),
            input_tokens=actual_input,
            output_tokens=actual_output,
            cached_input_tokens=cached,
        )

    def snapshot(self) -> dict:
        return {
            "schema_version": 1,
            "cost_basis": "conservative_estimate_not_invoice",
            "cost_scope": "stage",
            "cost_usd": None,
            "cost_estimate_usd": str(self.charged),
            "budget_usd": str(self.limit) if self.limit is not None else None,
            "limit_usd": str(self.limit) if self.limit is not None else None,
            "spending_policy": (
                "fixed_output_no_dollar_cap" if self.limit is None else "dollar_cap"
            ),
            "charged_or_reserved_usd": str(self.charged),
            "remaining_usd": str(self.limit - self.charged) if self.limit is not None else None,
            "pricing": asdict(self.pricing),
            "entries": self.entries,
        }


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def http_transport(api_key: str) -> Transport:
    if not isinstance(api_key, str) or not api_key or "\n" in api_key or "\r" in api_key:
        raise ValueError("An explicit API key is required")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def request(path: str, payload: dict, timeout: float) -> dict:
        if path not in ("/responses", "/responses/input_tokens"):
            raise ValueError("Unsupported API endpoint")
        req = urllib.request.Request(
            API_BASE + path,
            data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
            headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with opener.open(req, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().replace(api_key.encode(), b"[REDACTED_API_KEY]")
            raise ApiFailure(f"API HTTP {exc.code}; no retry", private_body=body) from None
        except (OSError, urllib.error.URLError):
            raise ApiFailure("API transport outcome unknown; no retry") from None
        try:
            result = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            raise ApiFailure("Invalid API JSON; no retry", private_body=raw) from None
        if not isinstance(result, dict):
            raise ApiFailure("Invalid API response object; no retry", private_body=raw)
        return result

    return request


def _private_write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("xb") as stream:
        os.chmod(path, 0o600)
        stream.write(data if isinstance(data, bytes) else runner.encoded(data))


def _redact(value: object) -> object:
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, dict):
        if value.get("type") == "reasoning":
            return {"type": "reasoning", "redacted_sha256": runner.sha(runner.encoded(value))}
        result = {}
        for key, item in value.items():
            if key == "image_url" and isinstance(item, str) and item.startswith("data:"):
                raw = base64.b64decode(item.split(",", 1)[1], validate=True)
                result[key] = {"redacted": True, "sha256": runner.sha(raw), "bytes": len(raw)}
            else:
                result[key] = _redact(item)
        return result
    return value


def function_tools(module=trusted_tools) -> list[dict]:
    return [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["inputSchema"],
            "strict": False,
        }
        for tool in module.TOOLS
    ]


def _verify_source_runtime(metadata: dict) -> None:
    python = Path(metadata.get("python", ""))
    if (
        not python.is_absolute()
        or not python.is_file()
        or runner.sha(python.read_bytes()) != metadata.get("python_sha256")
    ):
        raise ValueError("Pinned Python executable is required for sandboxed source tools")
    sandbox = Path("/usr/bin/sandbox-exec")
    if not sandbox.is_file() or not os.access(sandbox, os.X_OK):
        raise ValueError("sandbox-exec is required for live source-tool execution")


class SocketSource:
    """Fixed local JSON-RPC client for the existing isolated source process."""

    def __init__(self, workspace: Path):
        self.socket_path = workspace / "mcp.sock"
        self.calls = 0

    def call(self, name: str, arguments: dict) -> dict:
        if name not in {"inspect_source", "save_draft"} or not isinstance(arguments, dict):
            raise ValueError("Only bounded source tools are available")
        self.calls += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.calls,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        encoded = json.dumps(request, ensure_ascii=False, allow_nan=False).encode() + b"\n"
        if len(encoded) > 600_000:
            raise ValueError("Source tool request exceeds the bounded protocol size")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(40)
            connection.connect(str(self.socket_path))
            connection.sendall(encoded)
            with connection.makefile("rb") as stream:
                raw = stream.readline(32_000_001)
        if len(raw) > 32_000_000 or not raw.endswith(b"\n"):
            raise ValueError("Invalid bounded source-tool response")
        response = json.loads(raw)
        if (
            not isinstance(response, dict)
            or response.get("jsonrpc") != "2.0"
            or response.get("id") != self.calls
            or not isinstance(response.get("result"), dict)
        ):
            raise ValueError("Invalid bounded source-tool response")
        return response["result"]


def _load_packet(packet: Path, workspace: Path, *, offline: bool = False) -> tuple[dict, object]:
    runner.verify_seal(packet, "astra_reproduction_input")
    metadata = json.loads((packet / "preparation.json").read_text())
    runner.verify_model_selection(metadata)
    if (
        metadata.get("workflow_id") != runner.WORKFLOW_ID
        or metadata.get("stage") not in ("initial", "reviewed")
        or metadata.get("reference_access") is not False
        or metadata.get("independent_reader") != (metadata["stage"] == "reviewed")
        or type(metadata.get("limit_seconds")) is not int
        or not 60 <= metadata["limit_seconds"] <= 3600
    ):
        raise ValueError("Unsupported source-only packet contract")
    bundle = packet / "bundle"
    runner.verify_source_bundle(bundle, metadata)
    if (
        (bundle / "base.txt").read_text() != runner.BASE
        or runner.sha((bundle / "prompt.txt").read_bytes()) != metadata["prompt_sha256"]
        or runner.sha((bundle / "source.pdf").read_bytes()) != metadata["source_pdf_sha256"]
    ):
        raise ValueError("Packet source, prompt, or trusted tools changed")
    config = json.loads((bundle / "tool_config.json").read_text())
    for key in ("node", "pdftoppm"):
        binary = Path(config[key])
        expected = metadata.get("helper_binary_files", {}).get(str(binary))
        if not binary.is_absolute() or not expected or runner.sha(binary.read_bytes()) != expected:
            raise ValueError("Source helper executable is not pinned")
    for name, expected in metadata.get("implementation", {}).items():
        target = (runner.REPO / name).resolve()
        if not target.is_relative_to(runner.REPO) or runner.sha(target.read_bytes()) != expected:
            raise ValueError("Prepared implementation changed")
    if not offline:
        _verify_source_runtime(metadata)
    shutil.copytree(bundle, workspace, dirs_exist_ok=True)
    if not offline:
        return metadata, SocketSource(workspace)
    # The filename is fixed and its bytes match the trusted implementation above.
    spec = importlib.util.spec_from_file_location(
        "_sealed_assisted_source_tools",
        workspace
        / (
            "astra_source_tools_gated.py"
            if metadata.get("completion_policy")
            else "astra_source_tools.py"
        ),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return metadata, module.SourceTools(workspace)


def initial_request(workspace: Path, metadata: dict) -> dict:
    selected_tools = runner.selected_source_tools(metadata.get("completion_policy"))
    return {
        "model": metadata["model"],
        "instructions": (workspace / "base.txt").read_text()
        + "\n\n"
        + selected_tools.SERVER_INSTRUCTIONS,
        "reasoning": {"effort": metadata["reasoning_effort"]},
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "service_tier": "default",
        "parallel_tool_calls": False,
        "tools": function_tools(selected_tools),
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": (workspace / "prompt.txt").read_text()},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,"
                        + base64.b64encode((workspace / "overview.png").read_bytes()).decode(),
                        "detail": "original",
                    },
                ],
            }
        ],
    }


def count_request(request: dict) -> dict:
    return {
        key: request[key]
        for key in ("model", "instructions", "input", "tools", "reasoning", "parallel_tool_calls")
    }


def _tool_result(source, call: dict) -> dict:
    try:
        arguments = json.loads(call["arguments"])
        result = source.call(call["name"], arguments)
    except Exception as exc:
        result = {
            "isError": True,
            "content": [
                {
                    "type": "text",
                    "text": f"Source tool failed: {type(exc).__name__}. Check bounded arguments.",
                }
            ],
        }
    content = []
    for item in result["content"]:
        if item["type"] == "text":
            content.append({"type": "input_text", "text": item["text"]})
        elif item["type"] == "image" and item["mimeType"] == "image/png":
            content.append(
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64," + item["data"],
                    "detail": "original",
                }
            )
        else:
            raise ProtocolViolation("Unsupported local tool output")
    return {
        "type": "function_call_output",
        "call_id": call["call_id"],
        "output": content,
    }


def _validate_output(response: dict, seen_calls: set[str]) -> list[dict]:
    output = response.get("output")
    if not isinstance(output, list):
        raise ProtocolViolation("Response output must be an array")
    for item in output:
        if not isinstance(item, dict) or item.get("type") not in (
            "reasoning",
            "message",
            "function_call",
        ):
            raise ProtocolViolation("Unexpected tool activity: quarantine this run")
        if item["type"] == "reasoning" and not isinstance(item.get("encrypted_content"), str):
            raise ProtocolViolation("Stateless reasoning continuation is missing encrypted content")
        if item["type"] == "function_call":
            call_id = item.get("call_id")
            if (
                item.get("name") not in {"inspect_source", "save_draft"}
                or not isinstance(item.get("arguments"), str)
                or len(item["arguments"].encode()) > 600_000
                or not isinstance(call_id, str)
                or not call_id
                or call_id in seen_calls
            ):
                raise ProtocolViolation("Unexpected or repeated source function call")
            seen_calls.add(call_id)
    return output


def run(
    packet: Path,
    output: Path,
    *,
    api_key: str,
    budget_usd: object,
    pricing: ApiPricing = ASTRA_PRICING,
    transport: Transport | None = None,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    min_output_tokens: int = 1024,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """Run one fresh stage; injected transports explicitly select offline test execution."""
    if (
        type(max_output_tokens) is not int
        or type(min_output_tokens) is not int
        or not 1 <= min_output_tokens <= max_output_tokens <= MAX_OUTPUT_TOKENS
    ):
        raise ValueError("Invalid bounded output token limits")
    packet, output = Path(packet).resolve(), Path(output).resolve()
    runner.verify_seal(packet, "astra_reproduction_input")
    metadata = json.loads((packet / "preparation.json").read_text())
    pricing.validate(metadata["model"])
    budget = Budget(budget_usd, pricing)
    if budget.limit is None and (
        metadata.get("spending_policy") != "fixed_output_no_dollar_cap"
        or "budget_usd" not in metadata
        or metadata["budget_usd"] is not None
        or metadata.get("max_output_tokens_per_response") != max_output_tokens
    ):
        raise ValueError("No-dollar-cap execution requires an explicit matching sealed policy")
    if metadata.get("budget_usd") is not None and (
        budget.limit is None or budget.limit > decimal(metadata["budget_usd"])
    ):
        raise ValueError("Requested budget exceeds the sealed stage budget")
    send = transport if transport is not None else http_transport(api_key)
    # Validate all local packet evidence before creating an attempt or sending a request.
    with (
        tempfile.TemporaryDirectory(prefix="score2abc-assisted-api-") as temporary,
        ExitStack() as source_stack,
    ):
        workspace = Path(temporary)
        metadata, source = _load_packet(packet, workspace, offline=transport is not None)
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
        private = output / "api" / "private"
        private.mkdir(parents=True, mode=0o700)
        started_at, started = datetime.now(timezone.utc).isoformat(), clock()
        execution = {"usage": None, "completed_turn": False, "tool_calls": []}
        error, timed_out, quarantined, resolved_model = None, False, False, None
        returncode, budget_exhausted = 1, False
        request, seen_calls = initial_request(workspace, metadata), set()
        usage_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        cached_total, reasoning_total = 0, 0
        cached_known, reasoning_known = True, True

        def remaining() -> float:
            seconds = metadata["limit_seconds"] - (clock() - started)
            if seconds <= 0 or not math.isfinite(seconds):
                raise TimeoutError("Runner deadline exceeded")
            return seconds

        def ledger() -> None:
            runner.write(output / "api" / "budget_ledger.json", budget.snapshot())

        try:
            if transport is None:
                source_stack.enter_context(runner.mcp_broker(workspace, metadata))
            for turn in range(1, trusted_tools.MAX_TOOL_CALLS + 2):
                count_payload = count_request(request)
                _private_write(
                    output / "api" / f"count_request_{turn:03d}.json", _redact(count_payload)
                )
                counted = send("/responses/input_tokens", count_payload, remaining())
                _private_write(private / f"count_response_{turn:03d}.json", counted)
                row = budget.reserve(
                    counted.get("input_tokens"), max_output_tokens, min_output_tokens
                )
                request["max_output_tokens"] = row["max_output_tokens"]
                _private_write(output / "api" / f"request_{turn:03d}.json", _redact(request))
                ledger()  # Persist full reservation before any potentially billable request.
                response = send("/responses", request, remaining())
                _private_write(private / f"response_{turn:03d}.json", response)
                if response.get("service_tier") not in (None, "default"):
                    raise ProtocolViolation("Response used an unpriced service tier")
                model = response.get("model")
                if not isinstance(model, str) or not re.fullmatch(
                    re.escape(metadata["model"]) + r"(?:-\d{4}-\d{2}-\d{2})?", model
                ):
                    raise ProtocolViolation("Response model does not match the priced model")
                resolved_model = model
                budget.reconcile(row, response.get("usage"))
                ledger()
                for key in usage_totals:
                    usage_totals[key] += response["usage"][key]
                reported_cached = (
                    response["usage"].get("input_tokens_details", {}).get("cached_tokens")
                )
                output_details = response["usage"].get("output_tokens_details")
                reported_reasoning = (
                    output_details.get("reasoning_tokens")
                    if isinstance(output_details, dict)
                    else None
                )
                cached_known &= type(reported_cached) is int
                reasoning_known &= (
                    type(reported_reasoning) is int
                    and 0 <= reported_reasoning <= response["usage"]["output_tokens"]
                )
                if cached_known:
                    cached_total += reported_cached
                if reasoning_known:
                    reasoning_total += reported_reasoning
                execution["usage"] = dict(usage_totals)
                if cached_known:
                    execution["usage"]["input_tokens_details"] = {"cached_tokens": cached_total}
                if reasoning_known:
                    execution["usage"]["output_tokens_details"] = {
                        "reasoning_tokens": reasoning_total
                    }
                remaining()
                response_output = _validate_output(response, seen_calls)
                request["input"].extend(response_output)
                if response.get("status") != "completed":
                    raise ApiFailure("Response did not complete; no continuation retry")
                calls = [item for item in response_output if item["type"] == "function_call"]
                messages = [item for item in response_output if item["type"] == "message"]
                if not calls:
                    final = "\n".join(
                        part["text"]
                        for item in messages
                        for part in item.get("content", [])
                        if part.get("type") == "output_text"
                    )
                    _private_write(output / "final_message.txt", final.encode())
                    execution["completed_turn"], returncode = True, 0
                    break
                for call in calls:
                    remaining()
                    result = _tool_result(source, call)
                    request["input"].append(result)
                    execution["tool_calls"].append({"tool": call["name"], "status": "completed"})
                    runner.mirror(workspace, output)
            else:
                raise ApiFailure("Source tool turn allowance exhausted")
        except BudgetExhausted as exc:
            error, budget_exhausted = str(exc), True
        except TimeoutError:
            error, timed_out = "Runner deadline exceeded", True
        except ApiFailure as exc:
            error, quarantined = str(exc), isinstance(exc, ProtocolViolation)
            if exc.private_body is not None:
                _private_write(private / "transport_error_body.bin", exc.private_body)
        except Exception as exc:
            # Arbitrary transport exceptions may contain URLs, credentials, or API bodies.
            error = f"Execution failed: {type(exc).__name__}; no retry"
        finally:
            try:
                source_stack.close()
            finally:
                runner.mirror(workspace, output)
                ledger()
        finished = clock()
        if finished - started >= metadata["limit_seconds"]:
            timed_out, returncode = True, 1
            error = error or "Runner deadline exceeded"

    checkpoints = sorted(output.glob("artifacts/drafts/draft_*/checkpoint.json"))
    selected = checkpoints[-1].parent if checkpoints else None
    complete, completion_provenance = runner.checkpoint_completion(selected, metadata)
    if selected:
        shutil.copyfile(selected / "score.abc", output / "score.abc")
        shutil.copyfile(selected / "decisions.json", output / "decisions.json")
    provenance = {
        key: metadata[key]
        for key in (
            "workflow_id",
            "stage",
            "model",
            "reasoning_effort",
            "source_pdf_sha256",
            "reference_access",
            "independent_reader",
            "initial_abc_sha256",
        )
    }
    provenance.update(
        {
            "schema_version": 1,
            "auth_mode": "openai_api",
            "source_execution": (
                "sandboxed_source_process" if transport is None else "in_process_offline_test"
            ),
            "catalog_overrides": metadata.get("catalog_overrides"),
            "native_tool_mode": metadata.get("native_tool_mode"),
            "complete": complete,
            "status": (
                "failed"
                if quarantined
                else runner.stage_status(
                    complete and execution["completed_turn"],
                    selected is not None,
                    returncode,
                    timed_out,
                )
            ),
            "returncode": returncode,
            "quarantined": quarantined,
            "timed_out": timed_out,
            "budget_exhausted": budget_exhausted,
            "deadline_seconds": metadata["limit_seconds"],
            "error": error,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": finished - started,
            "selected_checkpoint": str(selected.relative_to(output)) if selected else None,
            "input_seal_sha256": runner.sha((packet / "seal.json").read_bytes()),
            "api_adapter_sha256": runner.sha(Path(__file__).read_bytes()),
            "resolved_model": resolved_model,
            "cost_usd": None,
            "cost_estimate_usd": str(budget.charged),
            "cost_basis": "conservative_estimate_not_invoice",
            "cost_scope": "stage",
            "budget_usd": str(budget.limit) if budget.limit is not None else None,
            "spending_policy": budget.snapshot()["spending_policy"],
            "max_output_tokens_per_response": max_output_tokens,
            "billing": budget.snapshot(),
            "human_review_minutes": None,
            **execution,
            **completion_provenance,
        }
    )
    runner.write(output / "stage_provenance.json", provenance)
    runner.seal(output, "astra_reproduction_stage")
    # Private API responses contain encrypted reasoning and must not become public logs.
    for path in output.rglob("*"):
        os.chmod(path, 0o700 if path.is_dir() else 0o600)
    return provenance
