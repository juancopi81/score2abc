"""Isolation boundaries are tested without network access or paid model calls."""

import base64
import hashlib

import pytest

from score2abc import score_eval_runner as runner


def request(image=b"fixture"):
    return {
        "model": "test-model",
        "reasoning": {"effort": "xhigh"},
        "input": [
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": runner.BASE_INSTRUCTIONS}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "transcribe"},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64," + base64.b64encode(image).decode(),
                    },
                ],
            },
        ],
    }


def verify(payload):
    return runner.verify_request(
        payload, "transcribe", [hashlib.sha256(b"fixture").hexdigest()], "test-model", "xhigh"
    )


def test_request_accepts_only_supplied_context():
    result = verify(request())
    assert result["status"] == "passed"
    assert result["exposed_tools"] == []
    assert result["input_image_sha256"] == [hashlib.sha256(b"fixture").hexdigest()]


@pytest.mark.parametrize(
    "addition",
    [
        {"type": "additional_tools", "tools": [{"name": "exec"}]},
        {"type": "message", "role": "assistant", "content": []},
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "AGENTS: hidden answer"}],
        },
        {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": "Memory of this song"}],
        },
        {"type": "function_call_output", "output": "answer"},
    ],
)
def test_rejects_incremental_tools_and_hidden_context(addition):
    payload = request()
    payload["input"].append(addition)
    with pytest.raises(ValueError):
        verify(payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(tools=[{"name": "shell"}]),
        lambda p: p.update(model="different-model"),
        lambda p: p.update(reasoning={"effort": "low"}),
        lambda p: p.update(instructions="hidden reference"),
        lambda p: p["input"][1]["content"][1].update(image_url="https://example.com/score.png"),
        lambda p: p["input"][1]["content"][1].update(
            image_url="data:image/png;base64,Y2hhbmdlZA=="
        ),
    ],
)
def test_rejects_tools_changed_settings_or_foreign_images(mutate):
    payload = request()
    mutate(payload)
    with pytest.raises(ValueError):
        verify(payload)


def test_environment_drops_api_auth_and_inherited_session(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://example.invalid")
    monkeypatch.setenv("CODEX_THREAD_ID", "hidden-parent")
    monkeypatch.setenv("CODEX_HOME", "/tmp/existing-user-auth-location")
    env = runner._environment()
    assert "OPENAI_API_KEY" not in env
    assert "OPENAI_BASE_URL" not in env
    assert "CODEX_THREAD_ID" not in env
    assert env["CODEX_HOME"] == "/tmp/existing-user-auth-location"


def test_command_forces_subscription_and_retains_safety(tmp_path):
    runtime = {"codex_path": "/installed/codex"}
    args = runner.command(runtime, tmp_path, "test-model", "xhigh", [])
    assert 'forced_login_method="chatgpt"' in args
    assert "--ignore-user-config" in args
    assert "--ephemeral" in args
    assert "read-only" in args
    assert "--ignore-rules" not in args
    assert not any("bypass" in arg for arg in args)
    assert "--model" in args
    policy = args[2]
    assert str(runner.PACKAGE_ROOT) in policy
    assert "AGENTS.md" in policy
    assert "memories" in policy
    assert "auth.json" not in policy


def test_sandbox_rejects_working_inside_answer_repository():
    with pytest.raises(ValueError):
        runner.sandbox_policy(runner.PACKAGE_ROOT / "out/isolated")


def test_runtime_manifest_and_catalog_tampering_fail(tmp_path):
    runtime = {"catalog_sha256": runner._sha(b"{}")}
    (tmp_path / "model_catalog.json").write_bytes(b"{}")
    runner._write(tmp_path / "runtime.json", runtime)
    (tmp_path / "runtime.sha256").write_text(runner._sha((tmp_path / "runtime.json").read_bytes()))
    assert runner.runtime_identity(tmp_path) == runtime
    (tmp_path / "model_catalog.json").write_bytes(b'{"changed":true}')
    with pytest.raises(ValueError, match="catalog changed"):
        runner.runtime_identity(tmp_path)


def test_empty_request_is_not_evidence_of_isolation():
    payload = request()
    payload["input"] = []
    with pytest.raises(ValueError):
        verify(payload)


def test_permissions_body_is_exact_not_prefix_whitelisted():
    payload = request()
    payload["input"].insert(
        0,
        {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": runner.PERMISSIONS_INSTRUCTIONS}],
        },
    )
    assert verify(payload)["status"] == "passed"
    payload["input"][0]["content"][0]["text"] = runner.PERMISSIONS_INSTRUCTIONS.replace(
        "Network access is restricted.", "Network access is restricted. Hidden note: F sharp."
    )
    with pytest.raises(ValueError, match="unapproved textual"):
        verify(payload)


@pytest.mark.parametrize(
    "kind", ["command_execution", "mcp_tool_call", "future_unknown_tool", "plan"]
)
def test_event_reader_fails_closed_on_all_non_message_items(kind):
    import json

    with pytest.raises(ValueError, match="quarantine"):
        runner._read_events(json.dumps({"type": "item.completed", "item": {"type": kind}}))


def test_structured_alternatives_preserved(tmp_path):
    payload = {
        "abc": "X:1\nK:C\nC|",
        "decisions": [
            {**{key: "text" for key in runner.DECISION_FIELDS}, "alternatives": ["D", "E"]}
        ],
    }
    path = tmp_path / "response.json"
    runner._write(path, payload)
    assert runner._read_response(path) == payload
    payload["decisions"][0]["alternatives"] = "D"
    runner._write(path, payload)
    with pytest.raises(ValueError):
        runner._read_response(path)


def make_runner_packet(tmp_path, monkeypatch):
    from score2abc import score_eval_config as config

    packet = tmp_path / "packet"
    packet.mkdir()
    source = packet / "source.pdf"
    source.write_bytes(b"fake PDF; renderer is mocked")
    (packet / "instructions.txt").write_text("instructions")
    runtime = {"status": "prepared"}
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    runner._write(
        runtime_root / "model_catalog.json",
        {"models": [{"slug": "test-model", "supported_reasoning_levels": [{"effort": "xhigh"}]}]},
    )
    task = {
        "slug": "test",
        "model_id": "model",
        "model": "test-model",
        "reasoning_effort": "xhigh",
        "adapter": "codex_subscription",
        "provider": "openai",
        "settings": {},
        "trial": 1,
        "system_sha256": "a" * 64,
        "generation_sha256": "b" * 64,
        "input_pdf_path": "source.pdf",
        "input_pdf_sha256": runner._sha(source.read_bytes()),
        "instructions_sha256": runner._sha(b"instructions"),
        "max_revision_passes": 2,
    }
    task["task_id"] = config.task_id(task)
    contract = {
        "schema_version": 2,
        "config": {},
        "generation": {"harness": {"manifest": {"runner_runtime": runtime, "implementation": {}}}},
        "tasks": [task],
    }
    runner._write(packet / "tasks.json", contract)
    (packet / "tasks.sha256").write_text(runner._sha((packet / "tasks.json").read_bytes()))
    monkeypatch.setattr(config, "verify_generation", lambda *_: None)
    monkeypatch.setattr(config, "verify_prepared_task", lambda task, _contract: task)
    monkeypatch.setattr(runner, "verify_runtime", lambda _: runtime)
    return packet, task, runtime_root


def test_render_failure_becomes_failed_registry(tmp_path, monkeypatch):
    import json

    packet, task, runtime = make_runner_packet(tmp_path, monkeypatch)

    def broken_render(*_):
        raise RuntimeError("unreadable source")

    monkeypatch.setattr(runner, "render_inputs", broken_render)
    result = runner.run_task(packet, task["task_id"], tmp_path / "run", runtime)
    assert result["status"] == "failed"
    assert "rendering" in result["error"]
    saved = json.loads((tmp_path / "run/registry.json").read_text())
    assert saved["runs"][0]["status"] == "failed"
    trace = json.loads((tmp_path / "run/run_record.json").read_text())
    assert trace["images"] == []
    assert trace["started_at"] <= trace["finished_at"]


@pytest.mark.parametrize("quarantined,expected", [(False, "partial"), (True, "failed")])
def test_repair_failure_retains_good_draft_unless_quarantined(
    tmp_path, monkeypatch, quarantined, expected
):
    import json

    from score2abc import score_eval

    packet, task, runtime = make_runner_packet(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "render_inputs", lambda *_: [])
    monkeypatch.setattr(
        score_eval, "validate_candidate", lambda *_: {"issues": [{"kind": "meter"}]}
    )
    response = {"abc": "X:1\nK:C\nC|", "decisions": []}
    results = iter(
        [
            {"status": "complete", "response": response, "usage": {"input_tokens": 3}},
            {
                "status": "failed",
                "error": "timeout or unapproved event",
                "quarantined": quarantined,
                "usage": None,
            },
        ]
    )

    def fake_call(*args):
        args[-1].mkdir()
        return next(results)

    monkeypatch.setattr(runner, "subscription_call", fake_call)
    output = tmp_path / "run"
    result = runner.run_task(packet, task["task_id"], output, runtime)
    assert result["status"] == expected
    assert (output / "score.abc").exists() is (not quarantined)
    trace = json.loads((output / "run_record.json").read_text())
    assert len(trace["attempts"]) == 2
    assert trace["attempts"][0]["response"] == response
