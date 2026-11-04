import json
from types import SimpleNamespace

import pytest

from scripts import run_assisted_transcription as entry


def test_key_loading_does_not_execute_env_file(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    marker = tmp_path / "marker"
    env = tmp_path / ".env"
    env.write_text(
        f"touch {marker}\nOTHER_SECRET=ignore\nexport OPENAI_API_KEY='test-only' # local\n"
    )
    assert entry.load_api_key(env) == "test-only"
    assert not marker.exists()
    assert capsys.readouterr().out == ""


def test_explicit_process_key_takes_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "process-test-only")
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=file-test-only\n")
    assert entry.load_api_key(env) == "process-test-only"


def test_missing_and_malformed_credentials_do_not_echo_contents(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env = tmp_path / ".env"
    with pytest.raises(ValueError, match="unavailable"):
        entry.load_api_key(env)
    env.write_text("OPENAI_API_KEY='synthetic-unclosed-secret\n")
    with pytest.raises(ValueError) as error:
        entry.load_api_key(env)
    assert "synthetic-unclosed-secret" not in str(error.value)


def test_symlink_key_file_is_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    target = tmp_path / "target"
    target.write_text("OPENAI_API_KEY=synthetic\n")
    link = tmp_path / ".env"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="unavailable"):
        entry.load_api_key(link)


def test_subscription_run_never_loads_api_key(tmp_path, monkeypatch, capsys):
    def forbidden(*args):
        pytest.fail("Subscription route accessed an API credential")

    calls = []
    monkeypatch.setattr(entry, "load_api_key", forbidden)
    monkeypatch.setattr(entry.shared, "run", lambda *args: calls.append(args) or {"ok": True})
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_assisted_transcription.py",
            "run",
            "--backend",
            "subscription",
            "--packet",
            str(tmp_path / "packet"),
            "--output",
            str(tmp_path / "result"),
            "--audit-dir",
            str(tmp_path / "audit"),
        ],
    )
    entry.main()
    assert calls == [(tmp_path / "packet", tmp_path / "result", tmp_path / "audit")]
    assert json.loads(capsys.readouterr().out) == {"ok": True}


def test_api_budget_required_before_preparation(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_assisted_transcription.py",
            "prepare",
            "--backend",
            "api",
            "--pdf",
            "source.pdf",
            "--prompt",
            "task.txt",
            "--output",
            "packet",
            "--stage",
            "initial",
            "--limit-seconds",
            "1800",
        ],
    )
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 2


def test_subscription_runtime_skips_stale_pins(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    base = tmp_path / "out/eval/runtime"
    stale = base / "codex_subscription_stale"
    current = base / "codex_subscription_current"
    stale.mkdir(parents=True)
    current.mkdir()

    def verify(path):
        if path != current:
            raise ValueError("Old binary hash")
        return {"status": "prepared"}

    monkeypatch.setattr(entry.shared.legacy, "verify_runtime", verify)
    assert entry.subscription_runtime(None) == current
    with pytest.raises(ValueError, match="Old binary"):
        entry.subscription_runtime(stale)


def test_no_current_subscription_runtime_fails_without_refresh(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="No current verified"):
        entry.subscription_runtime(None)
    assert not (tmp_path / "out").exists()


def test_audit_rejects_wrong_backend_before_creating_directory(tmp_path, monkeypatch):
    packet = tmp_path / "packet"
    packet.mkdir()
    (packet / "preparation.json").write_text(json.dumps({"backend": "subscription"}))
    monkeypatch.setattr(entry.shared, "verify_seal", lambda *args: {})
    output = tmp_path / "audit"
    with pytest.raises(ValueError, match="API packet"):
        entry.audit_api(packet, output)
    assert not output.exists()


def test_review_requires_new_api_author_not_subscription_author(tmp_path, monkeypatch):
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-synthetic")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Generic review")
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "stage_provenance.json").write_text(
        json.dumps({"auth_mode": "chatgpt_subscription"})
    )
    monkeypatch.setattr(entry.shared, "verify_seal", lambda *args: {})
    monkeypatch.setattr(entry.shared, "verify_initial_author", lambda *args: None)
    with pytest.raises(ValueError, match="own API"):
        entry.prepare_api(pdf, prompt, tmp_path / "packet", "reviewed", 1200, "2", initial)
    assert not (tmp_path / "packet").exists()


@pytest.mark.parametrize("author_complete", [True, False])
@pytest.mark.parametrize("no_cap", [True, False])
def test_api_pair_keeps_total_cap_and_stops_after_partial_author(
    tmp_path, monkeypatch, author_complete, no_cap
):
    from scripts.experiments import assisted_api

    preparations, runs = [], []
    monkeypatch.setattr(entry, "load_api_key", lambda *args: "synthetic-test-key")
    monkeypatch.setattr(
        entry, "prepare_api", lambda *args, **kwargs: preparations.append((args, kwargs))
    )
    monkeypatch.setattr(entry, "audit_api", lambda *args: {})

    def fake_run(packet, output, **kwargs):
        runs.append((packet, output, kwargs))
        return {
            "status": "complete" if author_complete else "partial",
            "returncode": 0,
            "complete": author_complete,
        }

    monkeypatch.setattr(assisted_api, "run", fake_run)
    args = SimpleNamespace(
        backend="api",
        budget_usd=None if no_cap else "5",
        no_dollar_cap=no_cap,
        env_file=None,
        output=tmp_path / "pair",
        pdf=tmp_path / "source.pdf",
        prompt=tmp_path / "author.txt",
        review_prompt=tmp_path / "review.txt",
        author_seconds=1800,
        review_seconds=1200,
        model="gpt-6-astra",
    )
    result = entry.execute_pair(args)
    assert result["complete"] is author_complete
    assert len(runs) == (2 if author_complete else 1)
    assert runs[0][2]["budget_usd"] == (None if no_cap else "3.0")
    assert preparations[0][0][6] is None
    assert preparations[0][1].get("no_dollar_cap", False) is no_cap
    if author_complete:
        assert runs[1][2]["budget_usd"] == (None if no_cap else "2.0")
        assert preparations[1][0][6] == args.output / "initial"
    assert "synthetic-test-key" not in (args.output / "pair_status.json").read_text()


@pytest.mark.parametrize("backend,budget", [("api", "5"), ("subscription", None)])
def test_no_cap_rejects_conflicting_backend_or_budget_before_credentials(
    monkeypatch, backend, budget
):
    monkeypatch.setattr(entry, "load_api_key", lambda *args: pytest.fail("Credential read"))
    args = SimpleNamespace(backend=backend, budget_usd=budget, no_dollar_cap=True)
    with pytest.raises(ValueError):
        entry.execute_pair(args)


@pytest.fixture
def interrupted_author(tmp_path):
    initial = tmp_path / "initial"
    checkpoint = initial / "artifacts/drafts/draft_002"
    checkpoint.mkdir(parents=True)
    for directory in (initial, checkpoint):
        (directory / "score.abc").write_text("X:1\nM:3/4\nL:1/4\nK:C\nC D E|]")
        entry.shared.write(directory / "decisions.json", [])
    entry.shared.write(
        checkpoint / "checkpoint.json",
        {
            "complete_claimed": False,
            "abc_sha256": entry.shared.sha((checkpoint / "score.abc").read_bytes()),
            "decisions_sha256": entry.shared.sha((checkpoint / "decisions.json").read_bytes()),
        },
    )
    entry.shared.write(checkpoint / "normalized.json", {"valid": True})
    provenance = {
        "workflow_id": entry.shared.WORKFLOW_ID,
        "stage": "initial",
        "model": entry.shared.MODEL,
        "reasoning_effort": entry.shared.EFFORT,
        "source_pdf_sha256": entry.shared.sha(b"%PDF-synthetic"),
        "auth_mode": "openai_api",
        "reference_access": False,
        "status": "partial",
        "complete": False,
        "completed_turn": False,
        "quarantined": False,
        "timed_out": False,
        "returncode": 1,
        "error": "Response did not complete; no continuation retry",
        "selected_checkpoint": "artifacts/drafts/draft_002",
    }
    entry.shared.write(initial / "stage_provenance.json", provenance)
    private = initial / "api/private"
    private.mkdir(parents=True)
    entry.shared.write(
        private / "response_003.json",
        {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
    )
    entry.shared.seal(initial, "astra_reproduction_stage")
    return initial


def reseal_synthetic_author(initial):
    (initial / "seal.json").unlink()
    entry.shared.seal(initial, "astra_reproduction_stage")


def verify_interrupted(initial, source_hash=None):
    return entry.verify_incomplete_checkpoint_author(
        initial, source_hash or entry.shared.sha(b"%PDF-synthetic"), entry.shared.MODEL
    )


def test_explicit_diagnostic_verifier_preserves_author_and_records_evidence(interrupted_author):
    before = entry.shared.pins(interrupted_author)
    metadata = verify_interrupted(interrupted_author)
    assert metadata["diagnostic_review_of_incomplete_author"] is True
    assert metadata["original_author_status"] == "partial"
    assert metadata["original_author_error"] == "Response did not complete; no continuation retry"
    assert metadata["original_author_incomplete_reason"] == "max_output_tokens"
    assert metadata["original_author_checkpoint"] == "artifacts/drafts/draft_002"
    assert metadata["original_author_seal_sha256"] == entry.shared.sha(
        (interrupted_author / "seal.json").read_bytes()
    )
    assert entry.shared.pins(interrupted_author) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("completion_policy", "completion_gate_v1"),
        ("completion_policy_sha256", "prior-policy-hash"),
        ("completion_gate", {"completion_ready": False}),
    ],
)
def test_legacy_diagnostic_rejects_gated_author_provenance(interrupted_author, field, value):
    path = interrupted_author / "stage_provenance.json"
    provenance = json.loads(path.read_text())
    provenance[field] = value
    entry.shared.write(path, provenance)
    reseal_synthetic_author(interrupted_author)
    with pytest.raises(ValueError, match="Legacy diagnostic review"):
        verify_interrupted(interrupted_author)


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "gpt-6.1-terra"),
        ("source_pdf_sha256", "other-source"),
        ("auth_mode", "chatgpt_subscription"),
        ("quarantined", True),
        ("timed_out", True),
        ("workflow_id", "other-workflow"),
        ("reasoning_effort", "low"),
        ("reference_access", True),
        ("stage", "reviewed"),
        ("status", "complete"),
        ("error", "API transport outcome unknown; no retry"),
    ],
)
def test_diagnostic_review_rejects_unmatched_or_unsafe_author(interrupted_author, field, value):
    path = interrupted_author / "stage_provenance.json"
    provenance = json.loads(path.read_text())
    provenance[field] = value
    entry.shared.write(path, provenance)
    reseal_synthetic_author(interrupted_author)
    with pytest.raises(ValueError, match="matching source-only interrupted API author"):
        verify_interrupted(interrupted_author)


@pytest.mark.parametrize("condition", ["missing", "invalid", "changed_export", "complete_claim"])
def test_diagnostic_review_rejects_missing_or_invalid_checkpoint(interrupted_author, condition):
    checkpoint = interrupted_author / "artifacts/drafts/draft_002"
    if condition == "missing":
        (checkpoint / "normalized.json").unlink()
    elif condition == "invalid":
        entry.shared.write(checkpoint / "normalized.json", {"valid": False})
    elif condition == "changed_export":
        (interrupted_author / "score.abc").write_text("different candidate")
    else:
        path = checkpoint / "checkpoint.json"
        saved = json.loads(path.read_text())
        saved["complete_claimed"] = True
        entry.shared.write(path, saved)
    reseal_synthetic_author(interrupted_author)
    with pytest.raises(ValueError, match="checkpoint"):
        verify_interrupted(interrupted_author)


@pytest.mark.parametrize("reason", ["content_filter", "other", None])
def test_diagnostic_review_requires_explicit_final_output_token_reason(interrupted_author, reason):
    entry.shared.write(
        interrupted_author / "api/private/response_004.json",
        {"status": "incomplete", "incomplete_details": {"reason": reason}},
    )
    reseal_synthetic_author(interrupted_author)
    with pytest.raises(ValueError, match="max_output_tokens"):
        verify_interrupted(interrupted_author)


def test_diagnostic_review_rejects_unsealed_mutation(interrupted_author):
    (interrupted_author / "decisions.json").write_text("[]\nchanged")
    with pytest.raises(ValueError, match="changed"):
        verify_interrupted(interrupted_author)


def test_default_api_review_still_rejects_interrupted_author(tmp_path, interrupted_author):
    pdf, prompt = tmp_path / "source.pdf", tmp_path / "prompt.txt"
    pdf.write_bytes(b"%PDF-synthetic")
    prompt.write_text("Generic review")
    with pytest.raises(ValueError, match="successful matching"):
        entry.prepare_api(
            pdf, prompt, tmp_path / "review", "reviewed", 1200, "2", interrupted_author
        )
    assert not (tmp_path / "review").exists()


@pytest.mark.parametrize("backend,stage", [("subscription", "reviewed"), ("api", "initial")])
def test_diagnostic_cli_flag_rejects_wrong_backend_or_stage(monkeypatch, backend, stage):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_assisted_transcription.py",
            "prepare",
            "--backend",
            backend,
            "--pdf",
            "source.pdf",
            "--prompt",
            "review.txt",
            "--output",
            "packet",
            "--stage",
            stage,
            "--limit-seconds",
            "1200",
            "--budget-usd",
            "2",
            "--review-incomplete-checkpoint",
        ],
    )
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 2


def test_diagnostic_cli_flag_is_forwarded_only_when_explicit(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(entry, "prepare_api", lambda *args, **kwargs: calls.append(kwargs) or {})
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_assisted_transcription.py",
            "prepare",
            "--backend",
            "api",
            "--pdf",
            "source.pdf",
            "--prompt",
            "review.txt",
            "--output",
            "packet",
            "--stage",
            "reviewed",
            "--initial-run",
            "initial",
            "--limit-seconds",
            "1200",
            "--budget-usd",
            "2",
            "--review-incomplete-checkpoint",
        ],
    )
    entry.main()
    assert calls == [{"review_incomplete_checkpoint": True, "no_dollar_cap": False}]
    assert json.loads(capsys.readouterr().out) == {}
