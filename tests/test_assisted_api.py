"""Hermetic API transport, budget, stateless reasoning, and checkpoint tests."""

import base64
import copy
import io
import json
import shutil
import stat
import subprocess
import sys
import urllib.error
from contextlib import contextmanager
from dataclasses import replace
from decimal import Decimal

import pytest

from scripts.experiments import assisted_api as api
from scripts.experiments import astra_source_tools as source
from scripts.experiments import run_astra_baseline_reproduction as runner

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Zl1sAAAAASUVORK5CYII="
)
ABC = "X:1\nM:3/4\nL:1/4\nK:C\nC D E|]"


@pytest.fixture
def packet(tmp_path, monkeypatch):
    packet = tmp_path / "packet"
    bundle = packet / "bundle"
    bundle.mkdir(parents=True)
    (bundle / "source.pdf").write_bytes(b"%PDF-synthetic")
    (bundle / "overview.png").write_bytes(PNG)
    (bundle / "prompt.txt").write_text("Transcribe only the seven supplied source bars.")
    (bundle / "base.txt").write_text(runner.BASE)
    shutil.copyfile(source.__file__, bundle / "astra_source_tools.py")
    executable = str(api.Path(sys.executable).resolve())
    config = {
        "source_pdf_sha256": runner.sha((bundle / "source.pdf").read_bytes()),
        "pages": 1,
        "node": executable,
        "pdftoppm": executable,
    }
    runner.write(bundle / "tool_config.json", config)
    metadata = {
        "workflow_id": runner.WORKFLOW_ID,
        "stage": "initial",
        "model": "gpt-6-astra",
        "reasoning_effort": "xhigh",
        "limit_seconds": 1800,
        "source_pdf_sha256": config["source_pdf_sha256"],
        "prompt_sha256": runner.sha((bundle / "prompt.txt").read_bytes()),
        "reference_access": False,
        "independent_reader": False,
        "initial_abc_sha256": None,
        "tools_sha256": runner.sha(runner.encoded(source.TOOLS)),
        "server_instructions_sha256": runner.sha(source.SERVER_INSTRUCTIONS.encode()),
        "helper_binary_files": {executable: runner.sha(api.Path(executable).read_bytes())},
        "python": executable,
        "python_sha256": runner.sha(api.Path(executable).read_bytes()),
    }
    runner.write(packet / "preparation.json", metadata)
    runner.seal(packet, "astra_reproduction_input")

    def parser(*args, **kwargs):
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=json.dumps(
                {
                    "valid": True,
                    "errors": [],
                    "measures": [],
                    "counts": {},
                    "route_status": "resolved",
                }
            ),
        )

    monkeypatch.setattr(source.subprocess, "run", parser)
    return packet


def response(output=(), *, input_tokens=100, output_tokens=20, cached_tokens=0, **kwargs):
    return {
        "id": "response-id",
        "model": "gpt-6-astra",
        "status": "completed",
        "service_tier": "default",
        "output": list(output),
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_details": {"cached_tokens": cached_tokens},
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        **kwargs,
    }


def function_call(name="save_draft", arguments=None, call_id="call-1"):
    return {
        "type": "function_call",
        "id": "function-id",
        "name": name,
        "call_id": call_id,
        "arguments": json.dumps(
            arguments if arguments is not None else {"abc": ABC, "decisions": [], "complete": True}
        ),
        "status": "completed",
    }


class FakeTransport:
    def __init__(self, responses, counts=None):
        self.responses = iter(responses)
        self.counts = iter(counts) if counts is not None else None
        self.calls = []

    def __call__(self, path, payload, timeout):
        self.calls.append((path, copy.deepcopy(payload), timeout))
        if path == "/responses/input_tokens":
            return {"input_tokens": next(self.counts) if self.counts else 100}
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return copy.deepcopy(result)


def execute(packet, tmp_path, transport, **kwargs):
    return api.run(
        packet,
        tmp_path / "run",
        api_key="synthetic-secret",
        budget_usd=kwargs.pop("budget_usd", "3"),
        transport=transport,
        **kwargs,
    )


def test_exact_source_tools_preserve_optional_fields():
    tools = api.function_tools()
    assert [tool["name"] for tool in tools] == ["inspect_source", "save_draft"]
    assert all(tool["type"] == "function" and tool["strict"] is False for tool in tools)
    assert tools[0]["parameters"] == source.TOOLS[0]["inputSchema"]
    assert tools[0]["parameters"]["required"] == ["page"]
    assert "note" not in tools[1]["parameters"]["required"]


def test_budget_shrinks_output_before_request_and_accounts_reported_cache():
    budget = api.Budget("0.1", api.ASTRA_PRICING)
    row = budget.reserve(1000, 25_000, 1)
    assert row["max_output_tokens"] == 1500
    assert Decimal(row["reserved_usd"]) == Decimal("0.1")
    budget.reconcile(
        row, response(input_tokens=1000, output_tokens=100, cached_tokens=900)["usage"]
    )
    assert budget.charged == Decimal("0.0084")
    assert row["cached_input_tokens"] == 900
    assert budget.snapshot()["remaining_usd"] == "0.0916"


def test_no_cap_retains_fixed_output_and_accounts_usage_across_calls():
    budget = api.Budget(None, api.ASTRA_PRICING)
    for _ in range(3):
        row = budget.reserve(100_000, 25_000, 1024)
        assert row["max_output_tokens"] == 25_000
        budget.reconcile(row, response(input_tokens=100_000, output_tokens=20_000)["usage"])
    assert budget.charged == Decimal("10.5")
    assert budget.snapshot()["limit_usd"] is None
    assert budget.snapshot()["remaining_usd"] is None
    assert budget.snapshot()["spending_policy"] == "fixed_output_no_dollar_cap"
    with pytest.raises(api.ProtocolViolation):
        budget.reserve(272_001, 25_000, 1024)


def test_missing_budget_cannot_silently_disable_spending_cap(packet, tmp_path):
    transport = FakeTransport([])
    with pytest.raises(ValueError, match="explicit matching sealed policy"):
        execute(packet, tmp_path, transport, budget_usd=None)
    assert not transport.calls
    assert not (tmp_path / "run").exists()


def test_explicit_no_cap_packet_completes_with_fixed_allowance(packet, tmp_path):
    metadata = json.loads((packet / "preparation.json").read_text())
    metadata.update(
        budget_usd=None,
        spending_policy="fixed_output_no_dollar_cap",
        max_output_tokens_per_response=25_000,
    )
    runner.write(packet / "preparation.json", metadata)
    (packet / "seal.json").unlink()
    runner.seal(packet, "astra_reproduction_input")
    transport = FakeTransport([response([function_call()]), response()])
    result = execute(packet, tmp_path, transport, budget_usd=None)
    assert result["status"] == "complete"
    assert result["budget_usd"] is None
    assert result["cost_estimate_usd"] is not None
    generations = [payload for path, payload, _ in transport.calls if path == "/responses"]
    assert [p["max_output_tokens"] for p in generations] == [25_000, 25_000]


@pytest.mark.parametrize(
    "usage", [None, {}, {"input_tokens": -1}, response(input_tokens=101)["usage"]]
)
def test_unrecognized_usage_consumes_full_reservation(usage):
    budget = api.Budget("1", api.ASTRA_PRICING)
    row = budget.reserve(100, 1000, 1)
    before = budget.charged
    with pytest.raises(api.ProtocolViolation):
        budget.reconcile(row, usage)
    assert budget.charged == before
    assert row["status"] == "reserved_unknown_outcome"


@pytest.mark.parametrize("count", [True, -1, 272001, "10", None])
def test_count_and_context_tier_fail_closed(count):
    budget = api.Budget("3", api.ASTRA_PRICING)
    with pytest.raises(api.ProtocolViolation):
        budget.reserve(count, 25_000, 1)
    assert not budget.entries


def test_unknown_rates_rejected_before_transport(packet, tmp_path):
    transport = FakeTransport([])
    with pytest.raises(ValueError, match="pricing"):
        execute(packet, tmp_path, transport, pricing=replace(api.ASTRA_PRICING, model="unknown"))
    assert not transport.calls
    assert not (tmp_path / "run").exists()


def test_complete_draft_replays_reasoning_and_call_outputs_privately(packet, tmp_path):
    reasoning = {"type": "reasoning", "id": "reasoning-id", "encrypted_content": "encrypted-secret"}
    transport = FakeTransport(
        [
            response([reasoning, function_call()]),
            response(
                [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Saved final draft."}],
                    }
                ]
            ),
        ]
    )
    result = execute(packet, tmp_path, transport)
    assert result["status"] == "complete" and result["complete"]
    assert result["completed_turn"] and result["returncode"] == 0
    assert result["auth_mode"] == "openai_api"
    assert result["cost_usd"] is None
    assert Decimal(result["cost_estimate_usd"]) == Decimal("0.007")
    assert result["billing"]["cost_basis"] == "conservative_estimate_not_invoice"
    assert [call[0] for call in transport.calls] == ["/responses/input_tokens", "/responses"] * 2
    first, second = transport.calls[1][1], transport.calls[3][1]
    assert first["store"] is False
    assert first["include"] == ["reasoning.encrypted_content"]
    assert first["service_tier"] == "default" and first["parallel_tool_calls"] is False
    assert first["instructions"] == runner.BASE + "\n\n" + source.SERVER_INSTRUCTIONS
    assert reasoning in second["input"]
    assert second["input"][-1]["type"] == "function_call_output"
    assert second["input"][-1]["call_id"] == "call-1"
    output = tmp_path / "run"
    assert (output / "score.abc").read_text() == ABC
    snapshot = (output / "api/request_002.json").read_text()
    assert "encrypted-secret" not in snapshot and base64.b64encode(PNG).decode() not in snapshot
    assert runner.sha(PNG) in snapshot
    assert "encrypted-secret" in (output / "api/private/response_001.json").read_text()
    assert stat.S_IMODE((output / "api/private/response_001.json").stat().st_mode) == 0o600
    runner.verify_seal(output, "astra_reproduction_stage")


def test_image_tool_output_is_multimodal_and_never_accepts_model_paths():
    class ImageSource:
        def call(self, name, arguments):
            assert name == "inspect_source" and arguments == {"page": 1}
            return {
                "content": [
                    {"type": "text", "text": "bounded crop evidence"},
                    {
                        "type": "image",
                        "mimeType": "image/png",
                        "data": base64.b64encode(PNG).decode(),
                    },
                ]
            }

    result = api._tool_result(ImageSource(), function_call("inspect_source", {"page": 1}))
    assert result["output"][1]["type"] == "input_image"
    assert result["output"][1]["detail"] == "original"
    assert result["output"][1]["image_url"].startswith("data:image/png;base64,")


def test_unknown_network_outcome_retains_full_reservation_and_never_retries(packet, tmp_path):
    transport = FakeTransport([RuntimeError("synthetic-secret should never be logged")])
    result = execute(packet, tmp_path, transport, budget_usd="0.1", min_output_tokens=1)
    assert len(transport.calls) == 2
    assert result["status"] == "failed" and not result["completed_turn"]
    assert Decimal(result["cost_estimate_usd"]) == Decimal("0.1")
    assert "synthetic-secret" not in (tmp_path / "run/stage_provenance.json").read_text()
    assert result["billing"]["entries"][0]["status"] == "reserved_unknown_outcome"
    runner.verify_seal(tmp_path / "run", "astra_reproduction_stage")


def test_unknown_response_usage_preserves_checkpoint_without_retry(packet, tmp_path):
    transport = FakeTransport([response([function_call()]), response(usage=None)])
    result = execute(packet, tmp_path, transport)
    assert result["quarantined"] and result["status"] == "failed"
    assert (tmp_path / "run/score.abc").read_text() == ABC
    assert len(transport.calls) == 4
    assert result["billing"]["entries"][-1]["status"] == "reserved_unknown_outcome"


def test_budget_exhaustion_stops_before_paid_generation(packet, tmp_path):
    transport = FakeTransport([], counts=[100])
    result = execute(packet, tmp_path, transport, budget_usd="0.001")
    assert [call[0] for call in transport.calls] == ["/responses/input_tokens"]
    assert result["budget_exhausted"] and result["cost_estimate_usd"] == "0"


def test_tool_and_source_mutations_fail_closed(packet, tmp_path):
    tools_file = packet / "bundle/astra_source_tools.py"
    tools_file.write_text(tools_file.read_text() + "\nraise RuntimeError('untrusted')\n")
    # Even a newly sealed packet cannot replace the trusted executable tool module.
    (packet / "seal.json").unlink()
    runner.seal(packet, "astra_reproduction_input")
    transport = FakeTransport([])
    with pytest.raises(ValueError, match="trusted tools"):
        execute(packet, tmp_path, transport)
    assert not transport.calls


@pytest.mark.parametrize("name", ["shell", "web_search", "read_file"])
def test_unapproved_functions_quarantine_without_execution(packet, tmp_path, name):
    transport = FakeTransport([response([function_call(name, {"path": "/etc/passwd"})])])
    result = execute(packet, tmp_path, transport)
    assert result["status"] == "failed" and result["quarantined"]
    assert not result["tool_calls"]
    assert not (tmp_path / "run/score.abc").exists()


def test_incomplete_response_is_not_promoted_or_retried(packet, tmp_path):
    transport = FakeTransport([response([function_call()]), response(status="incomplete")])
    result = execute(packet, tmp_path, transport)
    assert result["status"] == "partial" and not result["completed_turn"]
    assert result["complete"] and len(transport.calls) == 4


def test_runtime_deadline_stops_before_network(packet, tmp_path):
    transport = FakeTransport([])
    ticks = iter([0, 1801, 1802])
    result = execute(packet, tmp_path, transport, clock=lambda: next(ticks))
    assert result["timed_out"] and not transport.calls


def test_transport_rejects_redirect_and_arbitrary_endpoint(monkeypatch):
    assert api._NoRedirect().redirect_request(None, None, 302, "", {}, "https://other") is None
    transport = api.http_transport("synthetic-secret")
    with pytest.raises(ValueError, match="endpoint"):
        transport("/files", {}, 1)


def test_http_error_redacts_key_before_private_storage(monkeypatch):
    class Reject:
        def open(self, request, timeout):
            assert request.full_url == "https://api.openai.com/v1/responses"
            raise urllib.error.HTTPError(
                request.full_url,
                401,
                "unauthorized",
                {},
                io.BytesIO(b"invalid credential: synthetic-secret"),
            )

    monkeypatch.setattr(api.urllib.request, "build_opener", lambda *args: Reject())
    transport = api.http_transport("synthetic-secret")
    with pytest.raises(api.ApiFailure) as failure:
        transport("/responses", {}, 1)
    assert "synthetic-secret" not in str(failure.value)
    assert b"synthetic-secret" not in failure.value.private_body
    assert b"REDACTED_API_KEY" in failure.value.private_body


def test_sealed_budget_cannot_be_increased(packet, tmp_path):
    metadata = json.loads((packet / "preparation.json").read_text())
    metadata["budget_usd"] = "0.5"
    runner.write(packet / "preparation.json", metadata)
    (packet / "seal.json").unlink()
    runner.seal(packet, "astra_reproduction_input")
    transport = FakeTransport([])
    with pytest.raises(ValueError, match="sealed stage budget"):
        execute(packet, tmp_path, transport, budget_usd="0.51")
    assert not transport.calls


def test_usage_details_are_summed_without_inventing_missing_counts(packet, tmp_path):
    first = response([function_call()], cached_tokens=40)
    first["usage"]["output_tokens_details"] = {"reasoning_tokens": 12}
    second = response(cached_tokens=90)
    second["usage"]["output_tokens_details"] = {"reasoning_tokens": 8}
    result = execute(packet, tmp_path, FakeTransport([first, second]))
    assert result["usage"]["input_tokens_details"] == {"cached_tokens": 130}
    assert result["usage"]["output_tokens_details"] == {"reasoning_tokens": 20}


def test_missing_cache_detail_uses_full_input_rate(packet, tmp_path):
    item = response()
    item["usage"].pop("input_tokens_details")
    result = execute(packet, tmp_path, FakeTransport([item]))
    assert "input_tokens_details" not in result["usage"]
    assert "output_tokens_details" not in result["usage"]
    assert Decimal(result["cost_estimate_usd"]) == Decimal("0.0035")


@pytest.mark.parametrize("cached", [True, -1, 101, "20", None])
def test_invalid_cache_usage_retains_reservation(cached):
    budget = api.Budget("1", api.ASTRA_PRICING)
    row = budget.reserve(100, 1000, 1)
    reserved = budget.charged
    usage = response(cached_tokens=cached)["usage"]
    with pytest.raises(api.ProtocolViolation):
        budget.reconcile(row, usage)
    assert budget.charged == reserved


@pytest.mark.parametrize(
    "field,value", [("service_tier", "priority"), ("model", "gpt-6-astra-pro")]
)
def test_unpriced_model_or_service_tier_is_not_reconciled(packet, tmp_path, field, value):
    transport = FakeTransport([response(**{field: value})])
    result = execute(packet, tmp_path, transport, budget_usd="0.1", min_output_tokens=1)
    assert result["quarantined"] and result["status"] == "failed"
    assert Decimal(result["cost_estimate_usd"]) == Decimal("0.1")


def test_missing_encrypted_reasoning_cannot_continue(packet, tmp_path):
    transport = FakeTransport([response([{"type": "reasoning", "summary": []}, function_call()])])
    result = execute(packet, tmp_path, transport)
    assert result["quarantined"] and len(transport.calls) == 2
    assert not result["tool_calls"]


def test_count_transport_failure_has_no_generation_reservation(packet, tmp_path):
    calls = []

    def failed_count(path, payload, timeout):
        calls.append(path)
        raise api.ApiFailure("Input count unavailable; no retry")

    result = execute(packet, tmp_path, failed_count)
    assert calls == ["/responses/input_tokens"]
    assert result["billing"]["entries"] == [] and result["cost_estimate_usd"] == "0"


def test_reservation_is_on_disk_before_transport_receives_generation(packet, tmp_path):
    transport = FakeTransport([response()])

    def inspect_reservation(path, payload, timeout):
        if path == "/responses":
            ledger = json.loads((tmp_path / "run/api/budget_ledger.json").read_text())
            assert ledger["entries"][-1]["status"] == "reserved_unknown_outcome"
            assert Decimal(ledger["charged_or_reserved_usd"]) <= Decimal(ledger["limit_usd"])
        return transport(path, payload, timeout)

    execute(packet, tmp_path, inspect_reservation)


def test_socket_source_uses_only_fixed_bounded_jsonrpc(tmp_path, monkeypatch):
    sent = {}

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def settimeout(self, seconds):
            sent["timeout"] = seconds

        def connect(self, path):
            sent["socket_path"] = path

        def sendall(self, data):
            sent["request"] = json.loads(data)

        def makefile(self, mode):
            assert mode == "rb"
            return io.BytesIO(
                json.dumps(
                    {"jsonrpc": "2.0", "id": 1, "result": {"content": [], "isError": False}}
                ).encode()
                + b"\n"
            )

    monkeypatch.setattr(api.socket, "socket", lambda *args: Connection())
    client = api.SocketSource(tmp_path)
    assert client.call("inspect_source", {"page": 1}) == {"content": [], "isError": False}
    assert sent["timeout"] == 40
    assert sent["socket_path"] == str(tmp_path / "mcp.sock")
    assert sent["request"] == {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "inspect_source", "arguments": {"page": 1}},
    }
    with pytest.raises(ValueError, match="bounded source tools"):
        client.call("read_file", {"path": "/etc/passwd"})
    with pytest.raises(ValueError, match="protocol size"):
        client.call("save_draft", {"abc": "x" * 600_000})


def test_live_execution_requires_sandbox_before_network(packet, tmp_path, monkeypatch):
    transport = FakeTransport([])
    monkeypatch.setattr(api, "http_transport", lambda key: transport)
    monkeypatch.setattr(api.os, "access", lambda *args: False)
    with pytest.raises(ValueError, match="sandbox-exec"):
        api.run(packet, tmp_path / "run", api_key="synthetic", budget_usd="3")
    assert not transport.calls
    assert not (tmp_path / "run").exists()


def test_live_execution_requires_matching_python_pin(packet, tmp_path, monkeypatch):
    metadata = json.loads((packet / "preparation.json").read_text())
    metadata["python_sha256"] = "changed"
    runner.write(packet / "preparation.json", metadata)
    (packet / "seal.json").unlink()
    runner.seal(packet, "astra_reproduction_input")
    transport = FakeTransport([])
    monkeypatch.setattr(api, "http_transport", lambda key: transport)
    with pytest.raises(ValueError, match="Pinned Python"):
        api.run(packet, tmp_path / "run", api_key="synthetic", budget_usd="3")
    assert not transport.calls


def test_live_path_closes_broker_then_mirrors_before_workspace_cleanup(
    packet, tmp_path, monkeypatch
):
    events = []

    @contextmanager
    def broker(workspace, metadata):
        events.append("broker_started")
        yield
        assert workspace.is_dir()
        (workspace / "artifacts/after_shutdown.txt").write_text("preserved final source evidence")
        events.append("broker_stopped")

    transport = FakeTransport([response([function_call()]), response()])
    monkeypatch.setattr(api, "http_transport", lambda key: transport)
    monkeypatch.setattr(api, "_verify_source_runtime", lambda metadata: None)
    monkeypatch.setattr(api, "SocketSource", source.SourceTools)
    monkeypatch.setattr(runner, "mcp_broker", broker)
    result = api.run(packet, tmp_path / "run", api_key="synthetic", budget_usd="3")
    assert result["status"] == "complete"
    assert result["source_execution"] == "sandboxed_source_process"
    assert events == ["broker_started", "broker_stopped"]
    assert (tmp_path / "run/artifacts/after_shutdown.txt").read_text() == (
        "preserved final source evidence"
    )
    runner.verify_seal(tmp_path / "run", "astra_reproduction_stage")


def test_broker_failure_never_falls_back_or_calls_api(packet, tmp_path, monkeypatch):
    @contextmanager
    def broken_broker(workspace, metadata):
        raise RuntimeError("Sandbox source startup failed")
        yield

    transport = FakeTransport([])
    monkeypatch.setattr(api, "http_transport", lambda key: transport)
    monkeypatch.setattr(api, "_verify_source_runtime", lambda metadata: None)
    monkeypatch.setattr(runner, "mcp_broker", broken_broker)
    result = api.run(packet, tmp_path / "run", api_key="synthetic", budget_usd="3")
    assert result["status"] == "failed" and not transport.calls
    assert result["cost_estimate_usd"] == "0"
    assert not (tmp_path / "run/score.abc").exists()
