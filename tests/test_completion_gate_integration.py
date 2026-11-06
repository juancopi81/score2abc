"""Offline transport/packet tests for the opt-in completion policy."""

import copy
import io
import json
import shutil
import subprocess
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from test_assisted_api import ABC, FakeTransport, execute, function_call, response
from test_assisted_api import packet as packet_fixture

from scripts import run_assisted_transcription as entry
from scripts.experiments import assisted_api as api
from scripts.experiments import astra_source_tools as legacy
from scripts.experiments import astra_source_tools_gated as gated
from scripts.experiments import run_astra_baseline_reproduction as runner

POLICY = "completion_gate_v1"
packet = packet_fixture


def reseal(packet, metadata):
    runner.write(packet / "preparation.json", metadata)
    (packet / "seal.json").unlink()
    runner.seal(packet, "astra_reproduction_input")


def opt_in(packet):
    metadata = json.loads((packet / "preparation.json").read_text())
    bundle = packet / "bundle"
    for name, path in runner.source_tool_files(POLICY).items():
        shutil.copyfile(path, bundle / name)
    metadata.update(
        completion_policy=POLICY,
        completion_policy_sha256=runner.sha((bundle / "completion_gate.py").read_bytes()),
        tools_sha256=runner.sha(runner.encoded(gated.TOOLS)),
        server_instructions_sha256=runner.sha(gated.SERVER_INSTRUCTIONS.encode()),
    )
    config = json.loads((bundle / "tool_config.json").read_text())
    config["completion_policy"] = POLICY
    runner.write(bundle / "tool_config.json", config)
    reseal(packet, metadata)
    return metadata


def meter_parser(monkeypatch, valid):
    normalized = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "counts": {"measures": 1},
        "route_status": "resolved",
        "measures": [
            {
                "number": 1,
                "meter_valid": valid,
                "duration_ticks": 1152 if valid else 1344,
                "expected_ticks": 1152,
                "partial_reason": None,
            }
        ],
    }
    monkeypatch.setattr(
        legacy.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout=json.dumps(normalized)),
    )


def test_default_tool_contract_is_unchanged(packet, tmp_path):
    before = copy.deepcopy(legacy.TOOLS)
    assert runner.selected_source_tools() is legacy
    assert runner.source_tool_files() == {"astra_source_tools.py": api.Path(legacy.__file__)}
    metadata, source = api._load_packet(packet, tmp_path / "workspace", offline=True)
    request = api.initial_request(tmp_path / "workspace", metadata)
    assert request["tools"] == api.function_tools()
    assert request["instructions"] == runner.BASE + "\n\n" + legacy.SERVER_INSTRUCTIONS
    assert "completion_policy" not in metadata
    assert "resolutions" not in request["tools"][1]["parameters"]["properties"]
    assert "resolutions" in gated.TOOLS[1]["inputSchema"]["properties"]
    assert legacy.TOOLS == before


@pytest.mark.parametrize("valid", [True, False])
def test_api_final_status_uses_recomputed_gate_and_preserves_draft(
    packet, tmp_path, monkeypatch, valid
):
    opt_in(packet)
    meter_parser(monkeypatch, valid)
    transport = FakeTransport([response([function_call()]), response()])
    result = execute(packet, tmp_path, transport)
    assert result["complete"] is valid
    assert result["status"] == ("complete" if valid else "partial")
    assert result["completion_policy"] == POLICY
    assert result["completion_gate"]["completion_ready"] is valid
    assert result["completed_turn"]
    assert (tmp_path / "run/score.abc").read_text() == ABC
    assert len(transport.calls) == 4  # One save and one final response; no repair retry.
    request = transport.calls[0][1]
    assert request["tools"] == api.function_tools(gated)
    assert gated.SERVER_INSTRUCTIONS in request["instructions"]
    runner.verify_seal(tmp_path / "run", "astra_reproduction_stage")
    selected = tmp_path / "run/artifacts/drafts/draft_001"
    runner.write(selected / "completion_gate.json", {"completion_ready": not valid})
    metadata = json.loads((packet / "preparation.json").read_text())
    assert runner.checkpoint_completion(selected, metadata)[0] is valid


def test_legacy_final_status_still_accepts_meter_warning(packet, tmp_path, monkeypatch):
    meter_parser(monkeypatch, False)
    result = execute(packet, tmp_path, FakeTransport([response([function_call()]), response()]))
    assert result["complete"] and result["status"] == "complete"
    assert "completion_gate" not in result and "completion_policy" not in result


@pytest.mark.parametrize(
    "filename", ["astra_source_tools.py", "astra_source_tools_gated.py", "completion_gate.py"]
)
def test_resealed_gated_dependency_tamper_fails_before_transport(packet, tmp_path, filename):
    metadata = opt_in(packet)
    path = packet / "bundle" / filename
    path.write_text(path.read_text() + "\nraise RuntimeError('untrusted')\n")
    reseal(packet, metadata)
    transport = FakeTransport([])
    with pytest.raises(ValueError, match="trusted tools"):
        execute(packet, tmp_path, transport)
    assert not transport.calls


@pytest.mark.parametrize("change", ["unknown", "downgrade", "config", "hash"])
def test_policy_mismatch_fails_before_transport(packet, tmp_path, change):
    metadata = opt_in(packet)
    if change == "unknown":
        metadata["completion_policy"] = "untrusted"
    elif change == "downgrade":
        del metadata["completion_policy"]
    elif change == "hash":
        metadata["completion_policy_sha256"] = "incorrect"
    else:
        path = packet / "bundle/tool_config.json"
        config = json.loads(path.read_text())
        del config["completion_policy"]
        runner.write(path, config)
    reseal(packet, metadata)
    transport = FakeTransport([])
    with pytest.raises(ValueError):
        execute(packet, tmp_path, transport)
    assert not transport.calls


@pytest.mark.parametrize("policy", [None, POLICY])
def test_real_subscription_workspace_copy_preserves_selected_contract(packet, tmp_path, policy):
    metadata = opt_in(packet) if policy else json.loads((packet / "preparation.json").read_text())
    metadata.update(implementation={}, base_runtime={"binary_files": {}}, host_binary_files={})
    reseal(packet, metadata)
    workspace = tmp_path / "subscription_workspace"
    assert runner.copy_workspace(packet, workspace) == metadata
    assert (workspace / "astra_source_tools_gated.py").exists() is (policy is not None)
    assert (workspace / "astra_source_tools.py").read_bytes() == api.Path(
        legacy.__file__
    ).read_bytes()


@pytest.mark.parametrize("remove_other_markers", [False, True])
def test_real_subscription_workspace_copy_rejects_resealed_policy_downgrade(
    packet, tmp_path, remove_other_markers
):
    metadata = opt_in(packet)
    metadata.update(implementation={}, base_runtime={"binary_files": {}}, host_binary_files={})
    del metadata["completion_policy"]
    if remove_other_markers:
        del metadata["completion_policy_sha256"]
        metadata["tools_sha256"] = runner.sha(runner.encoded(legacy.TOOLS))
        metadata["server_instructions_sha256"] = runner.sha(legacy.SERVER_INSTRUCTIONS.encode())
        config_path = packet / "bundle/tool_config.json"
        config = json.loads(config_path.read_text())
        del config["completion_policy"]
        runner.write(config_path, config)
    reseal(packet, metadata)
    workspace = tmp_path / "subscription_workspace"
    with pytest.raises(ValueError, match="completion policy"):
        runner.copy_workspace(packet, workspace)
    assert not workspace.exists()


def test_reader_cannot_silently_change_author_policy():
    provenance = {
        "source_pdf_sha256": "source",
        "model": runner.MODEL,
        "reasoning_effort": runner.EFFORT,
        "workflow_id": runner.WORKFLOW_ID,
        "stage": "initial",
        "status": "complete",
        "returncode": 0,
        "reference_access": False,
        "completion_policy": POLICY,
        "completion_policy_sha256": runner.sha(
            runner.source_tool_files(POLICY)["completion_gate.py"].read_bytes()
        ),
    }
    with pytest.raises(ValueError, match="completion policy"):
        runner.verify_initial_author(provenance, "source", runner.MODEL)
    runner.verify_initial_author(provenance, "source", runner.MODEL, completion_policy=POLICY)
    provenance["completion_policy_sha256"] = "changed-implementation"
    with pytest.raises(ValueError, match="unchanged author completion policy"):
        runner.verify_initial_author(provenance, "source", runner.MODEL, completion_policy=POLICY)


def test_interrupted_checkpoint_evidence_blocks_without_raising(packet, tmp_path):
    metadata = opt_in(packet)
    selected = tmp_path / "artifacts/drafts/draft_001"
    selected.mkdir(parents=True)
    (selected / "score.abc").write_text(ABC)
    (selected / "checkpoint.json").write_text('{"complete_claimed":')
    ready, provenance = runner.checkpoint_completion(selected, metadata)
    assert not ready
    assert provenance["completion_gate"]["status"] == "invalid_checkpoint_evidence"
    assert (selected / "score.abc").read_text() == ABC


@pytest.mark.parametrize("policy", [None, POLICY])
def test_subscription_completion_changes_only_when_opted_in(packet, tmp_path, monkeypatch, policy):
    metadata = opt_in(packet) if policy else json.loads((packet / "preparation.json").read_text())
    metadata.update(catalog_overrides={}, native_tool_mode="code_mode_only")
    meter_parser(monkeypatch, False)
    audit = tmp_path / "audit"
    audit.mkdir()
    runner.write(
        audit / "audit.json",
        {"status": "passed", "input_seal_sha256": runner.sha((packet / "seal.json").read_bytes())},
    )
    runner.seal(audit, "astra_reproduction_audit")

    def copy_workspace(_packet, workspace):
        shutil.copytree(packet / "bundle", workspace, dirs_exist_ok=True)
        return metadata

    @contextmanager
    def broker(workspace, _metadata):
        runner.selected_source_tools(policy).SourceTools(workspace).call(
            "save_draft", {"abc": ABC, "decisions": [], "complete": True}
        )
        yield

    def popen(*args, **kwargs):
        kwargs["stdout"].write(json.dumps({"type": "turn.completed", "usage": {}}) + "\n")
        return SimpleNamespace(stdin=io.StringIO(), returncode=0, poll=lambda: 0)

    monkeypatch.setattr(runner, "copy_workspace", copy_workspace)
    monkeypatch.setattr(runner, "mcp_broker", broker)
    monkeypatch.setattr(runner, "command", lambda *args: [])
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    result = runner.run(packet, tmp_path / "subscription", audit)
    assert result["status"] == ("partial" if policy else "complete")
    assert result["complete"] is (policy is None)
    assert (tmp_path / "subscription/score.abc").read_text() == ABC


@pytest.mark.parametrize("backend", ["api", "subscription"])
def test_pair_forwards_policy_and_stops_on_blocked_author(tmp_path, monkeypatch, backend):
    preparations = []
    monkeypatch.setattr(entry, "load_api_key", lambda *_args: "test-only")
    monkeypatch.setattr(entry, "subscription_runtime", lambda *_args: tmp_path / "runtime")
    monkeypatch.setattr(entry, "audit_api", lambda *_args: {})
    monkeypatch.setattr(runner, "audit_packet", lambda *_args: {})
    monkeypatch.setattr(entry, "prepare_api", lambda *a, **k: preparations.append(k))
    monkeypatch.setattr(runner, "prepare", lambda *a, **k: preparations.append(k))

    def blocked(*args, **kwargs):
        return {"status": "partial", "returncode": 0, "complete": False}

    monkeypatch.setattr(api, "run", blocked)
    monkeypatch.setattr(runner, "run", blocked)
    args = SimpleNamespace(
        backend=backend,
        completion_policy=POLICY,
        budget_usd="5" if backend == "api" else None,
        no_dollar_cap=False,
        env_file=None,
        runtime_dir=None,
        output=tmp_path / "pair",
        pdf=tmp_path / "source.pdf",
        prompt=tmp_path / "author.txt",
        review_prompt=tmp_path / "review.txt",
        author_seconds=1800,
        review_seconds=1200,
        model=runner.MODEL,
    )
    result = entry.execute_pair(args)
    assert len(preparations) == 1 and preparations[0]["completion_policy"] == POLICY
    assert not result["complete"] and result["completion_policy"] == POLICY


@pytest.mark.parametrize("policy", [None, POLICY])
def test_api_preparation_and_audit_seal_only_selected_contract(tmp_path, monkeypatch, policy):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name in ("pdfinfo", "pdftoppm", "node", "abc2svg", "abc2svg-1.js", "toaudio-1.js"):
        (binaries / name).write_text("synthetic helper")
    pil = tmp_path / "fake_pil"
    pil.mkdir()
    (pil / "__init__.py").write_text("# synthetic PIL copy")
    monkeypatch.setattr(entry.PIL, "__file__", str(pil / "__init__.py"))
    monkeypatch.setattr(entry.shutil, "which", lambda name: str(binaries / name))
    monkeypatch.setattr(
        entry.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="Pages: 1\n"),
    )

    @contextmanager
    def image_open(_path):
        yield SimpleNamespace(
            thumbnail=lambda *a: None,
            save=lambda path: path.write_bytes(b"synthetic-overview"),
        )

    monkeypatch.setattr(entry.Image, "open", image_open)
    pdf, prompt = tmp_path / "source.pdf", tmp_path / "prompt.txt"
    pdf.write_bytes(b"%PDF-synthetic")
    prompt.write_text("Source-only task")
    output = tmp_path / "prepared"
    metadata = entry.prepare_api(
        pdf, prompt, output, "initial", 1800, "3", completion_policy=policy
    )
    assert (output / "bundle/base.txt").read_text() == runner.BASE
    assert (output / "bundle/astra_source_tools.py").read_bytes() == api.Path(
        legacy.__file__
    ).read_bytes()
    config = json.loads((output / "bundle/tool_config.json").read_text())
    assert ("completion_policy" in metadata) is (policy is not None)
    assert ("completion_policy" in config) is (policy is not None)
    assert (output / "bundle/completion_gate.py").exists() is (policy is not None)
    assert (output / "bundle/astra_source_tools_gated.py").exists() is (policy is not None)
    assert entry.audit_api(output, tmp_path / "audit")["status"] == "passed"
    if policy:
        corrupted = output / "bundle/completion_gate.py"
        corrupted.write_text("untrusted policy")
        reseal(output, metadata)
        with pytest.raises(ValueError, match="trusted tools"):
            entry.audit_api(output, tmp_path / "bad_audit")
