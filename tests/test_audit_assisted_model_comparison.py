"""Compatibility checks must use actual evidence and preserve legitimate draft differences."""

import base64
import copy
import json

import pytest

from scripts.experiments import audit_assisted_model_comparison as audit


def candidate_prompt(generic, abc, decisions):
    return (
        generic
        + audit.PREFIX
        + json.dumps({"abc": abc, "decisions": decisions}, ensure_ascii=False)
        + audit.SUFFIX
    )


def test_different_models_own_drafts_do_not_change_generic_prompt_fingerprint():
    generic = "Review the supplied PDF and this candidate.\n"
    left, left_proof = audit.prompt_evidence(
        candidate_prompt(generic, "X:1\nC D E", []), "reviewed", "X:1\nC D E", []
    )
    right, right_proof = audit.prompt_evidence(
        candidate_prompt(generic, "X:1\nD E F", []), "reviewed", "X:1\nD E F", []
    )
    assert audit.digest(left) == audit.digest(right)
    assert left_proof["own_abc_sha256"] != right_proof["own_abc_sha256"]
    assert left_proof["injected_candidate_sha256"] != right_proof["injected_candidate_sha256"]


@pytest.mark.parametrize(
    "mutation", ["other_candidate", "extra_key", "extra_instruction", "template_spacing"]
)
def test_reader_injection_cannot_hide_a_different_candidate_or_added_context(mutation):
    prompt = candidate_prompt("Review.\n", "ABC own", [])
    if mutation == "other_candidate":
        prompt = candidate_prompt("Review.\n", "ABC another model", [])
    elif mutation == "extra_key":
        prompt = prompt.replace('"decisions": []', '"decisions": [], "hint": "answer"')
    elif mutation == "extra_instruction":
        prompt += "An extra instruction after the candidate"
    else:
        prompt = prompt.replace('"abc": "ABC own"', '"abc":"ABC own"')
    with pytest.raises(ValueError, match="template|own sealed initial"):
        audit.prompt_evidence(prompt, "reviewed", "ABC own", [])


def test_generic_reviewer_changes_are_visible_even_when_candidates_bind():
    left, _ = audit.prompt_evidence(
        candidate_prompt("Review everything", "ABC", []), "reviewed", "ABC", []
    )
    right, _ = audit.prompt_evidence(
        candidate_prompt("Review only easy bars", "ABC", []), "reviewed", "ABC", []
    )
    assert audit.digest(left) != audit.digest(right)


@pytest.mark.parametrize("dimension", audit.DIMENSIONS)
def test_mismatches_have_real_different_fingerprints_and_compact_diagnostics(dimension):
    left = {name: {"observed": "same"} for name in audit.DIMENSIONS}
    right = copy.deepcopy(left)
    right[dimension]["observed"] = "changed"
    rows = audit.compare_dimensions("aviador_s8", "initial", left, right)
    assert len(rows) == 5
    changed = [row for row in rows if row["status"] == "mismatched"]
    assert len(changed) == 1
    assert changed[0]["dimension"] == dimension
    assert changed[0]["fingerprints"]["astra"] != changed[0]["fingerprints"]["terra"]
    assert changed[0]["differences"] == ["observed"]
    assert "observed" in changed[0]["basis"]


def test_exact_twenty_check_keys_and_hashes_derive_from_each_lane():
    dimensions = {name: {"evidence": name} for name in audit.DIMENSIONS}
    rows = [
        row
        for case in audit.CASES
        for stage in audit.STAGES
        for row in audit.compare_dimensions(case, stage, dimensions, dimensions)
    ]
    assert len(rows) == 20
    assert len({(row["case_id"], row["stage"], row["dimension"]) for row in rows}) == 20
    assert all(
        row["fingerprints"]["astra"] == audit.digest(dimensions[row["dimension"]]) for row in rows
    )


def test_numerical_ast_ignores_formatting_but_detects_scoring_changes():
    original = "def score(x):\n    return x / 2\n"
    formatting = "def score(x):\n\n    # formatting only\n    return x/2\n"
    changed = original.replace("/ 2", "/ 3")
    assert audit.ast_fingerprints(original, ("score",)) == audit.ast_fingerprints(
        formatting, ("score",)
    )
    assert audit.ast_fingerprints(original, ("score",)) != audit.ast_fingerprints(
        changed, ("score",)
    )
    with pytest.raises(ValueError, match="lacks"):
        audit.ast_fingerprints(original, ("missing",))


def test_effective_cli_normalizes_only_selected_model_not_limits():
    code = """
MODEL = "gpt-6-astra"
EFFORT = "xhigh"
def controls(workspace, metadata):
    return {"effort": EFFORT, "timeout": metadata["timeout"]}
def command(workspace, metadata):
    return ["codex", "--model", metadata["model"], "--sandbox", "read-only"]
"""
    astra = audit.effective_cli(code, {"model": "gpt-6-astra", "timeout": 40})
    terra = audit.effective_cli(code, {"model": "gpt-6.1-terra", "timeout": 40})
    changed = audit.effective_cli(code, {"model": "gpt-6.1-terra", "timeout": 80})
    assert astra == terra
    assert astra != changed
    different_effort = audit.effective_cli(
        code.replace('"xhigh"', '"low"'), {"model": "gpt-6-astra", "timeout": 40}
    )
    assert astra != different_effort


def test_missing_reports_prevent_publication(tmp_path):
    with pytest.raises(ValueError, match="All four finalized reports"):
        audit.audit_comparison(tmp_path / "astra", tmp_path / "terra")
    assert not list(tmp_path.iterdir())


def test_evidence_changes_and_seal_tampering_are_rejected(tmp_path):
    data = tmp_path / "evidence.txt"
    data.write_text("original")
    seal = {"kind": "synthetic", "files": {"evidence.txt": audit.byte_sha(b"original")}}
    (tmp_path / "seal.json").write_text(json.dumps(seal))
    audit.verify_directory(audit.Evidence(), tmp_path, "synthetic")
    tracked = audit.Evidence()
    tracked.read(data)
    data.write_text("changed")
    with pytest.raises(ValueError, match="changed during audit"):
        tracked.recheck()
    with pytest.raises(ValueError, match="Sealed directory differs"):
        audit.verify_directory(audit.Evidence(), tmp_path, "synthetic")


def test_output_is_create_once(tmp_path):
    output = tmp_path / "audit.json"
    audit.write_audit(output, {"status": "first"})
    with pytest.raises(FileExistsError):
        audit.write_audit(output, {"status": "replacement"})
    assert json.loads(output.read_text()) == {"status": "first"}


def test_missing_key_cannot_match_an_explicit_null():
    dimensions = {name: {} for name in audit.DIMENSIONS}
    changed = copy.deepcopy(dimensions)
    changed["tools"]["added_control"] = None
    tools = audit.compare_dimensions("aviador_s8", "initial", dimensions, changed)[2]
    assert tools["status"] == "mismatched"
    assert tools["differences"] == ["added_control"]
    assert tools["fingerprints"]["astra"] != tools["fingerprints"]["terra"]


def test_archived_runner_selection_requires_exact_preparation_hash():
    old, new = "old runner source\n", "new runner source\n"
    assert audit.select_runner_source([new, old], audit.byte_sha(old.encode())) == old
    with pytest.raises(ValueError, match="No supplied runner"):
        audit.select_runner_source([new], audit.byte_sha(old.encode()))


def test_upgraded_or_removed_installed_binary_preserves_sealed_comparison_evidence(tmp_path):
    binary, absent = tmp_path / "codex", tmp_path / "removed"
    binary.write_bytes(b"upgraded runtime")
    recorded = {str(binary): audit.byte_sha(b"old runtime"), str(absent): "historical-sha"}
    original = copy.deepcopy(recorded)
    evidence = audit.Evidence()
    observed = audit.observe_runtime_binaries(evidence, recorded)
    assert recorded == original
    assert observed[str(binary)]["current_status"] == "changed_since_preparation"
    assert observed[str(binary)]["recorded_sha256"] == audit.byte_sha(b"old runtime")
    assert observed[str(binary)]["current_sha256"] == audit.byte_sha(b"upgraded runtime")
    assert observed[str(absent)]["current_status"] == "unavailable"
    assert observed[str(absent)]["current_sha256"] is None
    evidence.recheck()


def _transport():
    def message(role, text):
        return {"type": "message", "role": role, "content": [{"type": "input_text", "text": text}]}

    user = message("user", "")
    user["content"] = [
        {"type": "input_text", "text": '<image name=[Image #1] path="/workspace/overview.png">'},
        {
            "type": "input_image",
            "image_url": "data:image/png;base64," + base64.b64encode(b"pixels").decode(),
        },
        {"type": "input_text", "text": "</image>"},
        {"type": "input_text", "text": "source-only prompt"},
    ]
    return {
        "model": "gpt-6-astra",
        "reasoning": {"effort": "xhigh"},
        "store": False,
        "max_output_tokens": 1000,
        "input": [
            {"type": "additional_tools", "role": "developer", "tools": []},
            message("developer", "base"),
            message("developer", audit.legacy.PERMISSIONS_INSTRUCTIONS),
            user,
        ],
    }


@pytest.mark.parametrize("mutation", ["context", "instructions", "prompt", "pixels", "tool_role"])
def test_transported_context_cannot_hide_added_or_changed_inputs(mutation):
    request = _transport()
    if mutation == "context":
        request["input"].append({"type": "message", "role": "user", "content": []})
    elif mutation == "instructions":
        request["input"][1]["content"][0]["text"] += " extra hint"
    elif mutation == "prompt":
        request["input"][3]["content"][3]["text"] += " extra hint"
    elif mutation == "pixels":
        request["input"][3]["content"][1]["image_url"] = (
            "data:image/png;base64," + base64.b64encode(b"other pixels").decode()
        )
    else:
        request["input"][0]["role"] = "user"
    with pytest.raises(ValueError, match="Transport|context|overview"):
        audit.request_controls(request, "base", "source-only prompt", audit.byte_sha(b"pixels"))


def test_transport_controls_keep_capability_and_token_limit_differences():
    left = _transport()
    right = copy.deepcopy(left)
    right.update(model="gpt-6.1-terra", prompt_cache_key="session-specific")
    args = ("base", "source-only prompt", audit.byte_sha(b"pixels"))
    assert audit.request_controls(left, *args) == audit.request_controls(right, *args)
    right["max_output_tokens"] = 2000
    right["tools"] = [{"name": "extra_tool"}]
    assert audit.differing_paths(
        audit.request_controls(left, *args), audit.request_controls(right, *args)
    ) == ["max_output_tokens", "tools"]


def _json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audit.encoded(value))


def _pin(path):
    return {"path": str(path), "sha256": audit.byte_sha(path.read_bytes())}


def _seal(root, kind):
    _json_file(
        root / "seal.json",
        {
            "kind": kind,
            "files": {
                str(path.relative_to(root)): _pin(path)["sha256"]
                for path in root.rglob("*")
                if path.is_file() and path.name != "seal.json"
            },
        },
    )


def _case_fixture(
    case,
    system,
    case_id,
    plan_path,
    *,
    initial_complete=True,
    reviewed_complete=True,
    allow_partial_author=False,
):
    """Small sealed provenance fixture; contains no model or musical reference data."""
    model = audit.MODELS[system]
    plan = json.loads(plan_path.read_text())
    source_sha = plan["targets"][case_id]["source_pdf"]["sha256"]
    stages, artifacts, pins = {}, {}, [_pin(plan_path)]
    for stage in audit.STAGES:
        packet, audit_dir, run_dir = case / f"{stage}_packet", case / f"{stage}_audit", case / stage
        identity = {
            "model": model,
            "reasoning_effort": "xhigh",
            "workflow_id": audit.WORKFLOW,
            "stage": stage,
            "reference_access": False,
        }
        metadata = {
            **identity,
            "implementation": {audit.RUNNER: audit.byte_sha(b"fixture runner")},
        }
        if stage == "reviewed" and allow_partial_author:
            metadata["allow_partial_author"] = True
        _json_file(packet / "preparation.json", metadata)
        _seal(packet, "astra_reproduction_input")
        input_sha = _pin(packet / "seal.json")["sha256"]
        _json_file(audit_dir / "audit.json", {"status": "passed", "input_seal_sha256": input_sha})
        _seal(audit_dir, "astra_reproduction_audit")
        complete = initial_complete if stage == "initial" else reviewed_complete
        provenance = {
            **identity,
            "source_pdf_sha256": source_sha,
            "input_seal_sha256": input_sha,
            "audit_sha256": _pin(audit_dir / "seal.json")["sha256"],
            "status": "complete" if complete else "partial",
            "complete": complete,
            "returncode": 0,
            "completed_turn": True,
            "quarantined": False,
            "timed_out": False,
            "error": None,
        }
        _json_file(run_dir / "stage_provenance.json", provenance)
        (run_dir / "score.abc").write_text(f"own {system} {stage} draft")
        _json_file(run_dir / "decisions.json", [])
        _seal(run_dir, "astra_reproduction_stage")
        stages[stage] = {"runtime": provenance}
        artifacts[stage] = {
            key: _pin(run_dir / name)
            for key, name in (
                ("abc", "score.abc"),
                ("decisions", "decisions.json"),
                ("provenance", "stage_provenance.json"),
            )
        }
        pins.extend(artifacts[stage].values())
        pins.extend(_pin(folder / "seal.json") for folder in (packet, audit_dir, run_dir))
    submission = {
        "kind": "astra_assisted_workflow_submission_seal",
        "status": "both_stages_frozen_before_reference_scoring",
        "target_id": case_id,
        "source_pdf_sha256": source_sha,
        "plan_sha256": _pin(plan_path)["sha256"],
        "stages": artifacts,
        "reference_feedback_given": False,
    }
    _json_file(case / "submission_seal.json", submission)
    report = {
        "schema_version": 1,
        "kind": "astra_assisted_workflow_reproduction_report",
        "target_id": case_id,
        "model": model,
        "reasoning_effort": "xhigh",
        "workflow_id": audit.WORKFLOW,
        "both_stages_verified_before_reference_content": True,
        "stages": stages,
        "stage_artifacts": artifacts,
        "pins": pins + [_pin(case / "submission_seal.json")],
    }
    _json_file(case / "evaluation/report.json", report)
    return report


def _plan_fixture(path, system):
    plan = {
        "model": audit.MODELS[system],
        "source_policy": "supplied_pdf_only",
        "targets": {case: {"source_pdf": {"sha256": f"{case}-source"}} for case in audit.CASES},
    }
    _json_file(path, plan)
    return plan


@pytest.mark.parametrize(
    "change",
    [
        {"status": "partial"},
        {"returncode": -9},
        {"returncode": False},
        {"complete": False},
        {"completed_turn": False},
        {"quarantined": True},
        {"timed_out": True},
        {"error": "BrokenPipeError"},
    ],
)
def test_failed_or_inconsistent_runtime_is_not_a_finalized_report(tmp_path, change):
    plan_path, case = tmp_path / "evaluation_plan.json", tmp_path / "aviador_s8"
    plan = _plan_fixture(plan_path, "terra")
    report = _case_fixture(case, "terra", "aviador_s8", plan_path)
    report["stages"]["initial"]["runtime"].update(change)
    _json_file(case / "evaluation/report.json", report)
    with pytest.raises(ValueError, match="finalized matching report"):
        audit.load_report(audit.Evidence(), tmp_path, "aviador_s8", "terra", plan, plan_path)


@pytest.mark.parametrize("reviewed_complete", [True, False])
def test_clean_partial_author_is_auditable_only_with_explicit_reader_policy(
    tmp_path, reviewed_complete
):
    plan_path, case = tmp_path / "evaluation_plan.json", tmp_path / "rumichaca_full"
    plan = _plan_fixture(plan_path, "terra")
    _case_fixture(
        case,
        "terra",
        "rumichaca_full",
        plan_path,
        initial_complete=False,
        reviewed_complete=reviewed_complete,
        allow_partial_author=True,
    )
    report, stages = audit.load_report(
        audit.Evidence(), tmp_path, "rumichaca_full", "terra", plan, plan_path
    )
    assert report["stages"]["initial"]["runtime"]["complete"] is False
    assert stages["initial"].provenance["status"] == "partial"
    assert audit.clean_runtime_turn(stages["initial"].provenance) is True
    assert stages["reviewed"].provenance["complete"] is reviewed_complete


def test_clean_partial_author_does_not_silently_enter_original_review_protocol(tmp_path):
    plan_path, case = tmp_path / "evaluation_plan.json", tmp_path / "rumichaca_full"
    plan = _plan_fixture(plan_path, "terra")
    _case_fixture(case, "terra", "rumichaca_full", plan_path, initial_complete=False)
    with pytest.raises(ValueError, match="explicit sealed policy exception"):
        audit.load_report(audit.Evidence(), tmp_path, "rumichaca_full", "terra", plan, plan_path)


@pytest.mark.parametrize(
    "metadata,stage",
    [({"allow_partial_author": "true"}, "reviewed"), ({"allow_partial_author": True}, "initial")],
)
def test_partial_author_exception_is_only_an_explicit_reviewer_boolean(metadata, stage):
    with pytest.raises(ValueError, match="explicit reviewer boolean"):
        audit.author_review_policy(metadata, stage)


def test_partial_author_exception_is_a_visible_reviewed_limits_mismatch():
    baseline = {name: {} for name in audit.DIMENSIONS}
    baseline["limits"]["author_review_policy"] = audit.author_review_policy({}, "reviewed")
    exploratory = copy.deepcopy(baseline)
    exploratory["limits"]["author_review_policy"] = audit.author_review_policy(
        {"allow_partial_author": True}, "reviewed"
    )
    differences = audit.compare_dimensions("rumichaca_full", "reviewed", baseline, exploratory)
    assert [
        (row["dimension"], row["differences"])
        for row in differences
        if row["status"] == "mismatched"
    ] == [("limits", ["author_review_policy.allow_partial_author"])]
    assert audit.author_review_policy({}, "initial") is None


def test_submission_cannot_substitute_another_sealed_draft(tmp_path):
    plan_path, case = tmp_path / "evaluation_plan.json", tmp_path / "aviador_s8"
    plan = _plan_fixture(plan_path, "terra")
    report = _case_fixture(case, "terra", "aviador_s8", plan_path)
    seal_path = case / "submission_seal.json"
    seal = json.loads(seal_path.read_text())
    other = tmp_path / "another-draft.abc"
    other.write_text("another draft")
    seal["stages"]["initial"]["abc"] = _pin(other)
    _json_file(seal_path, seal)
    report["stage_artifacts"] = seal["stages"]
    report["pins"] = [p for p in report["pins"] if p["path"] != str(seal_path)]
    report["pins"].extend([_pin(seal_path), _pin(other)])
    _json_file(case / "evaluation/report.json", report)
    with pytest.raises(ValueError, match="does not bind the audited stage file"):
        audit.load_report(audit.Evidence(), tmp_path, "aviador_s8", "terra", plan, plan_path)


def test_fresh_rumichaca_root_uses_shared_plan_and_all_twenty_checks(tmp_path, monkeypatch):
    astra, terra = tmp_path / "astra", tmp_path / "cohort/terra"
    retry = tmp_path / "cohort/retries/rumichaca_20261029_v1"
    astra_plan, terra_plan = astra / "evaluation_plan.json", terra.parent / "evaluation_plan.json"
    _plan_fixture(astra_plan, "astra")
    _plan_fixture(terra_plan, "terra")
    for case in audit.CASES:
        _case_fixture(astra / case, "astra", case, astra_plan)
    _case_fixture(terra / "aviador_s8", "terra", "aviador_s8", terra_plan)
    _case_fixture(
        retry,
        "terra",
        "rumichaca_full",
        terra_plan,
        initial_complete=False,
        allow_partial_author=True,
    )
    interrupted = terra / "rumichaca_full/initial/seal.json"
    _json_file(interrupted, {"preserve": "interrupted attempt"})
    old_bytes = interrupted.read_bytes()
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    for relative in (audit.RUNNER, audit.ADAPTER):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture runner")
    monkeypatch.setattr(audit, "git_source", lambda *_: "fixture runner")
    visited = []

    def dimensions(evidence, stage, lane, initial, report, plan, case_id, runner, adapter):
        visited.append((lane.run_dir, stage))
        value = {name: {"same": name} for name in audit.DIMENSIONS}
        value["limits"]["author_review_policy"] = audit.author_review_policy(lane.metadata, stage)
        if lane.run_dir.parent == retry:
            value["tools"]["recorded_runtime"] = "new runtime"
        return value, {"runtime_binary_observations": {"recorded": "separate diagnostic"}}

    monkeypatch.setattr(audit, "stage_dimensions", dimensions)
    report = audit.audit_comparison(astra, terra, terra_rumichaca_root=retry)
    assert len(visited) == 8
    assert {row["status"] for row in report["checks"]} == {"matched", "mismatched"}
    mismatches = [row for row in report["checks"] if row["status"] == "mismatched"]
    assert {(row["case_id"], row["stage"], row["dimension"]) for row in mismatches} == {
        ("rumichaca_full", stage, "tools") for stage in audit.STAGES
    } | {("rumichaca_full", "reviewed", "limits")}
    assert len(report["checks"]) == 20
    assert report["case_roots"]["terra"]["rumichaca_full"] == str(retry)
    assert (
        report["report_sha256"]["terra"]["rumichaca_full"]
        == _pin(retry / "evaluation/report.json")["sha256"]
    )
    assert report["runtime_binary_observations"]["terra"]["rumichaca_full"]["initial"]
    assert report["reviewer_injection_proofs"]["terra"]["rumichaca_full"]["initial"] == {}
    outcome = report["execution_outcomes"]["terra"]["rumichaca_full"]["initial"]
    assert outcome == {
        "runtime_turn_completed_cleanly": True,
        "status": "partial",
        "score_complete": False,
        "author_review_policy": None,
    }
    assert interrupted.read_bytes() == old_bytes
    assert str(interrupted) not in {pin["path"] for pin in report["evidence_pins"]}
    with pytest.raises(ValueError, match="All four finalized reports"):
        audit.audit_comparison(astra, terra)
