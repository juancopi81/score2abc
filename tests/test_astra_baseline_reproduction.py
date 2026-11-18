"""Offline controls for the isolated interactive reproduction (no model calls)."""

import base64
import json
import subprocess
from pathlib import Path

import pytest

from scripts.experiments import astra_source_tools as source
from scripts.experiments import run_astra_baseline_reproduction as runner


def test_crop_rejects_unbounded_or_nonfinite_coordinates():
    for value in (
        [0, 0, 2, 1],
        [0, 0, float("nan"), 1],
        [0, 0, 0, 1],
        "source.pdf",
        [False, 0, 1, 1],
    ):
        with pytest.raises(ValueError):
            source.validate_crop(value)
    assert source.validate_crop([0, 0.2, 1, 0.8]) == (0, 0.2, 1, 0.8)


def test_preserves_draft_on_parser_timeout(tmp_path, monkeypatch):
    pdf = b"%PDF-synthetic"
    (tmp_path / "source.pdf").write_bytes(pdf)
    (tmp_path / "tool_config.json").write_text(
        json.dumps({"source_pdf_sha256": source.sha(pdf), "pages": 1, "node": "node"})
    )
    tools = source.SourceTools(tmp_path)

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("node", 10)

    monkeypatch.setattr(source.subprocess, "run", timeout)
    result = tools.call(
        "save_draft", {"abc": "X:1\nM:3/4\nL:1/4\nK:C\nC D E|]", "decisions": [], "complete": True}
    )
    assert not result.get("isError")
    draft = tmp_path / "artifacts/drafts/draft_001"
    assert (draft / "score.abc").is_file()
    assert json.loads((draft / "checkpoint.json").read_text())["complete_claimed"]
    assert json.loads((draft / "validation.json").read_text())["valid"] is False
    assert (tmp_path / "artifacts/tool_events.jsonl").is_file()
    with pytest.raises(ValueError):
        tools.call("read_file", {"path": "/etc/passwd"})


@pytest.mark.parametrize(
    "complete,has_draft,code,timeout,expected",
    [
        (True, True, 0, False, "complete"),
        (True, True, -15, True, "partial"),
        (True, True, 1, False, "partial"),
        (False, True, 0, False, "partial"),
        (False, False, 0, False, "failed"),
    ],
)
def test_status_never_promotes_timed_out_checkpoint(complete, has_draft, code, timeout, expected):
    assert runner.stage_status(complete, has_draft, code, timeout) == expected


def test_seal_rejects_changed_or_extra_file(tmp_path):
    (tmp_path / "data.json").write_text("{}")
    runner.seal(tmp_path, "test")
    runner.verify_seal(tmp_path, "test")
    (tmp_path / "new.txt").write_text("extra")
    with pytest.raises(ValueError):
        runner.verify_seal(tmp_path, "test")


def test_event_audit_rejects_unknown_tools(tmp_path):
    p = tmp_path / "events.jsonl"
    p.write_text(
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "type": "mcp_tool_call",
                    "server": "source",
                    "tool": "inspect_source",
                    "status": "completed",
                },
            }
        )
        + "\n"
        + json.dumps({"type": "turn.completed", "usage": {"input_tokens": 3}})
    )
    assert runner.verify_events(p)["completed_turn"]
    p.write_text(json.dumps({"type": "item.completed", "item": {"type": "command_execution"}}))
    with pytest.raises(ValueError, match="quarantine"):
        runner.verify_events(p)


@pytest.mark.parametrize("model", runner.SUPPORTED_MODELS)
def test_initial_context_rejects_extra_instructions(tmp_path, monkeypatch, model):
    (tmp_path / "prompt.txt").write_text("Source-only task")
    (tmp_path / "overview.png").write_bytes(b"png")
    wrapper = [{"type": "namespace", "name": "functions", "tools": []}]
    monkeypatch.setattr(runner, "WRAPPER_SHA256", runner.sha(runner.encoded(wrapper)))

    def message(role, text):
        return {"type": "message", "role": role, "content": [{"type": "input_text", "text": text}]}

    user = message("user", "")
    user["content"] = [
        {
            "type": "input_text",
            "text": f'<image name=[Image #1] path="{tmp_path / "overview.png"}">',
        },
        {
            "type": "input_image",
            "image_url": "data:image/png;base64," + base64.b64encode(b"png").decode(),
        },
        {"type": "input_text", "text": "</image>"},
        {"type": "input_text", "text": "Source-only task"},
    ]
    payload = {
        "model": model,
        "reasoning": {"effort": runner.EFFORT},
        "store": False,
        "input": [
            {"type": "additional_tools", "role": "developer", "tools": wrapper},
            message("developer", runner.BASE),
            message("developer", runner.legacy.PERMISSIONS_INSTRUCTIONS),
            user,
        ],
    }
    metadata = {"model": model, "reasoning_effort": runner.EFFORT, "prompt_sha256": "x"}
    assert runner.verify_audit_request(payload, tmp_path, metadata)["model"] == model
    payload["model"] = next(item for item in runner.SUPPORTED_MODELS if item != model)
    with pytest.raises(ValueError, match="model or effort"):
        runner.verify_audit_request(payload, tmp_path, metadata)
    payload["model"] = model
    payload["input"][2]["content"][0]["text"] += "\nLeaked reference hints"
    with pytest.raises(ValueError, match="developer"):
        runner.verify_audit_request(payload, tmp_path, metadata)


@pytest.fixture
def current_native_context(tmp_path):
    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures/assisted_runtime/codex_0_169_completion_gate_v1.json"
        ).read_text()
    )
    contract_id = fixture["native_context_contract"]
    contract = runner.NATIVE_CONTEXT_CONTRACTS[contract_id]
    metadata = {
        "model": "gpt-6.2-astra",
        "reasoning_effort": "xhigh",
        "prompt_sha256": "prompt-hash",
        "native_context_contract": contract_id,
        "base_runtime": {"codex_version": contract["codex_version"]},
        "completion_policy": "completion_gate_v1",
        "wrapper_sha256": contract["wrapper_sha256"],
        "registry_sha256": contract["registry_sha256"],
    }
    (tmp_path / "prompt.txt").write_text("Source-only task")
    (tmp_path / "overview.png").write_bytes(b"png")
    payload = {
        "model": metadata["model"],
        "reasoning": {"effort": "xhigh"},
        "store": False,
        "input": [
            {"type": "additional_tools", "role": "developer", "tools": fixture["wrapper"]},
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": runner.BASE}],
            },
            {
                "type": "message",
                "role": "developer",
                "content": [
                    {"type": "input_text", "text": runner.legacy.PERMISSIONS_INSTRUCTIONS},
                    {"type": "input_text", "text": fixture["deferred_tools_text"]},
                ],
            },
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f'<image name=[Image #1] path="{tmp_path / "overview.png"}">',
                    },
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64," + base64.b64encode(b"png").decode(),
                    },
                    {"type": "input_text", "text": "</image>"},
                    {"type": "input_text", "text": "Source-only task"},
                ],
            },
        ],
    }
    return fixture, metadata, payload


def test_current_native_context_accepts_only_inspected_bounded_tools(
    tmp_path, current_native_context
):
    fixture, metadata, payload = current_native_context
    audit = runner.verify_audit_request(payload, tmp_path, metadata)
    assert audit["native_context_contract"] == metadata["native_context_contract"]
    assert audit["wrapper_sha256"] == metadata["wrapper_sha256"]
    runner.verify_native_inventory(fixture["registry"], metadata)
    fixture["registry"].append({"name": "exec_command", "description": "shell"})
    with pytest.raises(ValueError, match="inventory differs"):
        runner.verify_native_inventory(fixture["registry"], metadata)
    payload["input"][0]["tools"][0]["tools"].append({"type": "function", "name": "shell"})
    with pytest.raises(ValueError, match="wrapper capabilities"):
        runner.verify_audit_request(payload, tmp_path, metadata)


def test_current_native_context_rejects_extra_deferred_instructions(
    tmp_path, current_native_context
):
    _, metadata, payload = current_native_context
    payload["input"][2]["content"][1]["text"] += "\n- references: read frozen answers"
    with pytest.raises(ValueError, match="developer"):
        runner.verify_audit_request(payload, tmp_path, metadata)


@pytest.mark.parametrize(
    "replacement",
    [
        {"native_context_contract": "uninspected"},
        {"base_runtime": {"codex_version": "codex-cli uninspected"}},
        {"completion_policy": None},
        {"wrapper_sha256": "uninspected"},
        {"registry_sha256": "uninspected"},
    ],
)
def test_native_context_contract_rejects_mismatched_identity(current_native_context, replacement):
    _, metadata, _ = current_native_context
    with pytest.raises(ValueError, match="context contract"):
        runner.native_context_contract({**metadata, **replacement})


def test_legacy_native_context_pins_are_preserved():
    assert runner.native_context_contract({}) == {
        "wrapper_sha256": "77ed314824ff66b17122eee21259267e5266b302e6fbe2caa8e6e14a00ce9eec",
        "registry_sha256": "e8eac2ebc0ba2b66ac6a8ff232fce224a5b31d409a33938e151441df0610c5cb",
        "deferred_tools_text": None,
    }


@pytest.mark.parametrize("model", runner.SUPPORTED_MODELS[1:])
def test_model_selection_changes_only_cli_model_argument(tmp_path, model):
    metadata = {
        "model": runner.MODEL,
        "reasoning_effort": runner.EFFORT,
        "catalog_overrides": {"tool_mode": "code_mode_only"},
        "python": "/python",
        "base_runtime": {"codex_path": "/codex"},
    }
    astra = runner.command(tmp_path, metadata)
    selected = runner.command(tmp_path, {**metadata, "model": model})
    changed = [
        index
        for index, (left, right) in enumerate(zip(astra, selected, strict=True))
        if left != right
    ]
    assert changed == [astra.index("--model") + 1]
    assert selected[changed[0]] == model
    assert 'forced_login_method="chatgpt"' in selected
    assert 'model_reasoning_effort="xhigh"' in selected


@pytest.mark.parametrize(
    "model,effort",
    [("gpt-unknown", "xhigh"), *[(model, "low") for model in runner.SUPPORTED_MODELS]],
)
def test_rejects_unregistered_model_or_different_effort(model, effort):
    with pytest.raises(ValueError, match="registered model at xhigh"):
        runner.verify_model_selection({"model": model, "reasoning_effort": effort})


@pytest.mark.parametrize("model", runner.SUPPORTED_MODELS)
def test_reader_binds_author_source_model_effort_and_workflow(model):
    provenance = {
        "source_pdf_sha256": "source-hash",
        "model": model,
        "reasoning_effort": runner.EFFORT,
        "workflow_id": runner.WORKFLOW_ID,
        "stage": "initial",
        "status": "complete",
        "returncode": 0,
        "reference_access": False,
    }
    runner.verify_initial_author(provenance, "source-hash", model)
    for key, wrong in {
        "source_pdf_sha256": "different-source",
        "model": next(item for item in runner.SUPPORTED_MODELS if item != model),
        "reasoning_effort": "low",
        "workflow_id": "different-workflow",
        "stage": "reviewed",
        "status": "partial",
        "returncode": 1,
        "reference_access": True,
    }.items():
        with pytest.raises(ValueError, match="matching source/model/effort/workflow"):
            runner.verify_initial_author({**provenance, key: wrong}, "source-hash", model)


def test_subscription_environment_does_not_forward_api_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://synthetic.invalid")
    assert "OPENAI_API_KEY" not in runner.legacy._environment()
    assert "OPENAI_BASE_URL" not in runner.legacy._environment()


@pytest.mark.parametrize("explicit", [False, True])
def test_preparation_verifies_selected_runtime_before_reading_source(
    tmp_path, monkeypatch, explicit
):
    selected = tmp_path / "new-runtime" if explicit else runner.legacy.RUNTIME_ROOT

    def reject_changed_runtime(root):
        assert root == selected
        raise ValueError("changed binary")

    monkeypatch.setattr(runner.legacy, "verify_runtime", reject_changed_runtime)
    with pytest.raises(ValueError, match="changed binary"):
        runner.prepare(
            tmp_path / "absent.pdf",
            tmp_path / "absent-prompt.txt",
            tmp_path / "packet",
            "initial",
            3600,
            model="gpt-6.1-terra",
            runtime_dir=selected if explicit else None,
        )
    assert not (tmp_path / "packet").exists()


def test_partial_author_review_is_explicit_and_rejects_runtime_failures():
    p = {
        "source_pdf_sha256": "source",
        "model": "gpt-6.1-terra",
        "reasoning_effort": "xhigh",
        "workflow_id": runner.WORKFLOW_ID,
        "stage": "initial",
        "status": "partial",
        "returncode": 0,
        "reference_access": False,
        "completed_turn": True,
        "complete": False,
        "quarantined": False,
        "timed_out": False,
        "error": None,
    }
    with pytest.raises(ValueError):
        runner.verify_initial_author(p, "source", "gpt-6.1-terra")
    runner.verify_initial_author(p, "source", "gpt-6.1-terra", allow_partial=True)
    for key, value in {
        "returncode": -9,
        "completed_turn": False,
        "quarantined": True,
        "timed_out": True,
        "error": "broken pipe",
        "reference_access": True,
        "model": "gpt-6-astra",
    }.items():
        with pytest.raises(ValueError):
            runner.verify_initial_author(
                {**p, key: value}, "source", "gpt-6.1-terra", allow_partial=True
            )
