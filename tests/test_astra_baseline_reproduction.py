"""Offline controls for the isolated interactive reproduction (no model calls)."""

import base64
import json
import subprocess

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


def test_initial_context_rejects_extra_instructions(tmp_path, monkeypatch):
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
        "model": runner.MODEL,
        "reasoning": {"effort": runner.EFFORT},
        "store": False,
        "input": [
            {"type": "additional_tools", "role": "developer", "tools": wrapper},
            message("developer", runner.BASE),
            message("developer", runner.legacy.PERMISSIONS_INSTRUCTIONS),
            user,
        ],
    }
    runner.verify_audit_request(payload, tmp_path, {"prompt_sha256": "x"})
    payload["input"][2]["content"][0]["text"] += "\nLeaked reference hints"
    with pytest.raises(ValueError, match="developer"):
        runner.verify_audit_request(payload, tmp_path, {"prompt_sha256": "x"})


def test_subscription_environment_does_not_forward_api_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://synthetic.invalid")
    assert "OPENAI_API_KEY" not in runner.legacy._environment()
    assert "OPENAI_BASE_URL" not in runner.legacy._environment()
