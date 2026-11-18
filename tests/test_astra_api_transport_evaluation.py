"""Offline transport-contract controls; no real scores, model calls or credentials."""

import copy
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from test_evaluate_astra_reproduction import _fixture, _full_score_fixture, _write

from scripts.experiments import evaluate_astra_reproduction as evaluator
from scripts.experiments import run_astra_baseline_reproduction as runner


def _packet(root, plan, stage, prompt, *, api, initial_hash=None, target="aviador_s8"):
    packet = root / f"{stage}_packet"
    bundle = packet / "bundle"
    bundle.mkdir(parents=True)
    text = prompt.read_text()
    if stage == "reviewed":
        candidate = (
            {
                "abc": (root / "initial/score.abc").read_text(),
                "decisions": evaluator.read_json(root / "initial/decisions.json"),
            }
            if api
            else {}
        )
        text += (
            "\n\n<untrusted_candidate_draft>\n"
            + json.dumps(candidate, ensure_ascii=False)
            + "\n</untrusted_candidate_draft>\n"
        )
    (bundle / "prompt.txt").write_text(text)
    shutil.copyfile(plan["targets"][target]["source_pdf"]["path"], bundle / "source.pdf")
    for name, source in runner.source_tool_files("completion_gate_v1").items():
        shutil.copyfile(source, bundle / name)
    _write(bundle / "tool_config.json", {"completion_policy": "completion_gate_v1"})
    tools = runner.selected_source_tools("completion_gate_v1")
    metadata = {
        "stage": stage,
        "model": plan["model"],
        "reasoning_effort": plan["reasoning_effort"],
        "workflow_id": plan["workflow_id"],
        "source_pdf_sha256": plan["targets"][target]["source_pdf"]["sha256"],
        "prompt_sha256": evaluator.sha(bundle / "prompt.txt"),
        "prompt_template_sha256": evaluator.sha(prompt),
        "auth_mode": "openai_api" if api else "chatgpt_subscription",
        "completion_policy": plan["completion_policy"],
        "completion_policy_sha256": plan["completion_policy_sha256"],
        "reference_access": False,
        "independent_reader": stage == "reviewed",
        "initial_abc_sha256": initial_hash,
        "limit_seconds": plan["stage_protocol"]["deadline_seconds"][target][stage],
        "tools_sha256": runner.sha(runner.encoded(tools.TOOLS)),
        "server_instructions_sha256": runner.sha(tools.SERVER_INSTRUCTIONS.encode()),
    }
    _write(packet / "preparation.json", metadata)
    runner.seal(packet, "astra_reproduction_input")
    return metadata


def _transport_fixture(
    tmp_path, monkeypatch, *, initial="good", reviewed="good", target="aviador_s8"
):
    baseline_root = tmp_path / "baseline"
    baseline_root.mkdir()
    baseline_args = (
        _full_score_fixture(baseline_root, monkeypatch)
        if target == "rumichaca_full"
        else _fixture(baseline_root, monkeypatch)
    )
    report = evaluator.evaluate(*baseline_args)
    baseline_plan = evaluator.read_json(baseline_args[0])
    prompts = {}
    prompt_names = evaluator._transport_prompts(target)
    for name in prompt_names:
        prompts[name] = baseline_args[0].parent / name
        prompts[name].write_text(f"Synthetic instructions: {name}")
    baseline_plan.update(
        model="gpt-6.2-astra",
        kind=evaluator.COMPARISON_PLAN_KIND,
        completion_policy="completion_gate_v1",
        completion_policy_sha256=evaluator.sha(
            Path(runner.__file__).with_name("completion_gate.py")
        ),
        review_instruction_version="assisted_review_v2",
        submission_contract={},
        current_harness={"prompts": {name: evaluator.pin(p) for name, p in prompts.items()}},
    )
    if target == "rumichaca_full":
        expanded = baseline_args[0].parent / "expanded/manifest.json"
        _write(expanded, {"label_set": "reviewed_six_v4"})
        baseline_plan["supplemental_source_labels"] = evaluator.pin(expanded)
        previous_labels = evaluator.verify_labels

        def labels(path, reference):
            value = copy.deepcopy(previous_labels(path, reference))
            if path == expanded.parent:
                value["label_set"] = "reviewed_six_v4"
            return value

        monkeypatch.setattr(evaluator, "verify_labels", labels)
    _write(baseline_args[0], baseline_plan)
    report.update(model="gpt-6.2-astra", pins=[evaluator.pin(baseline_args[0])])
    baseline_case = baseline_args[2].parent
    for stage, name in zip(evaluator.STAGES, prompt_names, strict=True):
        _packet(baseline_case, baseline_plan, stage, prompts[name], api=False, target=target)
        report["stages"][stage]["runtime"].update(
            auth_mode="chatgpt_subscription",
            input_seal_sha256=evaluator.sha(baseline_case / f"{stage}_packet/seal.json"),
        )
    _write(baseline_args[2] / "report.json", report)
    root = tmp_path / "api"
    root.mkdir()
    plan = evaluator.build_api_transport_plan(
        baseline_args[0],
        baseline_args[2] / "report.json",
        experiment_id="synthetic_transport",
        prompts=prompts,
        **({"target_id": target} if target != "aviador_s8" else {}),
    )
    plan_path = root / "evaluation_plan.json"
    _write(plan_path, plan)
    execution = evaluator.read_json(baseline_args[0].parent / "execution_contract.json")
    execution.update(
        plan_sha256=evaluator.sha(plan_path),
        deadline_seconds=plan["stage_protocol"]["deadline_seconds"],
    )
    _write(root / "execution_contract.json", execution)
    case = root / target
    now = datetime.now(timezone.utc) + timedelta(seconds=1)
    author_hash = None
    for i, (stage, name, value) in enumerate(
        zip(evaluator.STAGES, prompt_names, (initial, reviewed), strict=True)
    ):
        metadata = _packet(
            case, plan, stage, prompts[name], api=True, initial_hash=author_hash, target=target
        )
        packet_hash = evaluator.sha(case / f"{stage}_packet/seal.json")
        _write(
            case / f"{stage}_audit/audit.json",
            {
                "backend": "api",
                "status": "passed",
                "input_seal_sha256": packet_hash,
            },
        )
        runner.seal(case / f"{stage}_audit", "astra_reproduction_audit")
        provenance = {
            **metadata,
            "source_execution": "sandboxed_source_process",
            "input_seal_sha256": packet_hash,
            "status": "complete",
            "complete": True,
            "returncode": 0,
            "error": None,
            "elapsed_seconds": 1,
            "deadline_seconds": metadata["limit_seconds"],
            "started_at": (now + timedelta(seconds=i * 2)).isoformat(),
            "finished_at": (now + timedelta(seconds=i * 2 + 1)).isoformat(),
        }
        _write(case / stage / "stage_provenance.json", provenance)
        _write(case / stage / "decisions.json", [])
        (case / stage / "score.abc").write_text(value)
        runner.seal(case / stage, "astra_reproduction_stage")
        if stage == "initial":
            author_hash = evaluator.sha(case / stage / "score.abc")
    return plan_path, case


def _change(path, update, *, sealed_kind=None):
    value = evaluator.read_json(path)
    update(value)
    _write(path, value)
    if sealed_kind:
        (path.parent / "seal.json").unlink()
        runner.seal(path.parent, sealed_kind)


@pytest.mark.parametrize("reviewed,expected", [("good", 7), ("wrong", 6)])
def test_single_target_api_pair_keeps_exact_and_negative_results(
    tmp_path, monkeypatch, reviewed, expected
):
    plan_path, case = _transport_fixture(tmp_path, monkeypatch, reviewed=reviewed)
    seal = evaluator.seal_api_pair(plan_path, case)
    report = evaluator.evaluate(plan_path, seal, case / "evaluation")
    assert report["model"] == "gpt-6.2-astra"
    assert report["stages"]["reviewed"]["metrics"]["summary"]["exact_measures"] == expected
    assert report["transport_scope"]["rumichaca_scheduled"] is False
    assert set(evaluator.read_json(plan_path)["targets"]) == {"aviador_s8"}
    with pytest.raises(FileExistsError):
        evaluator.seal_api_pair(plan_path, case)


@pytest.mark.parametrize(
    "mutation", ["source", "model", "prompt", "grader", "future_baseline", "rumichaca"]
)
def test_transport_plan_rejects_changed_controls_before_grading(tmp_path, monkeypatch, mutation):
    plan_path, case = _transport_fixture(tmp_path, monkeypatch)
    plan = evaluator.read_json(plan_path)
    if mutation == "source":
        plan["targets"]["aviador_s8"]["system_index"] = 7
    elif mutation == "model":
        plan["model"] = "gpt-6.2-sol"
    elif mutation == "prompt":
        prompt = plan["current_harness"]["prompts"]["aviador_s8.txt"]
        prompt["sha256"] = "a" * 64
    elif mutation == "rumichaca":
        plan["targets"]["rumichaca_full"] = copy.deepcopy(plan["targets"]["aviador_s8"])
    else:
        path = Path(plan["transport_predecessor"]["report"]["path"])
        report = evaluator.read_json(path)
        if mutation == "grader":
            report["implementations"]["historical_scorer"]["sha256"] = "a" * 64
        else:
            report["created_at"] = "2099-01-01T00:00:00+00:00"
        _write(path, report)
        plan["transport_predecessor"]["report"] = evaluator.pin(path)
    with pytest.raises(ValueError):
        evaluator._verify_transport_predecessor(plan, plan_path.parent, plan["created_at"])
    assert not (case / "evaluation").exists()


@pytest.mark.parametrize(
    "mutation", ["auth", "binding", "prompt", "audit", "gate", "model", "stage_binding"]
)
@pytest.mark.parametrize("target", ["aviador_s8", "rumichaca_full"])
def test_api_stage_rejects_self_consistent_wrong_execution_before_truth(
    tmp_path, monkeypatch, mutation, target
):
    plan_path, case = _transport_fixture(tmp_path, monkeypatch, target=target)
    if mutation == "prompt":
        _change(
            case / "reviewed_packet/preparation.json",
            lambda d: d.update(prompt_template_sha256="a" * 64),
            sealed_kind="astra_reproduction_input",
        )
    elif mutation == "audit":
        _change(
            case / "reviewed_audit/audit.json",
            lambda d: d.update(backend="subscription"),
            sealed_kind="astra_reproduction_audit",
        )
    else:
        changes = {
            "auth": {"auth_mode": "chatgpt_subscription"},
            "binding": {"initial_abc_sha256": "a" * 64},
            "gate": {"completion_policy_sha256": "a" * 64},
            "model": {"model": "gpt-6.2-sol"},
            "stage_binding": {"stage": "initial"},
        }
        _change(
            case / "reviewed/stage_provenance.json",
            lambda d: d.update(changes[mutation]),
            sealed_kind="astra_reproduction_stage",
        )
    with pytest.raises(ValueError):
        seal = evaluator.seal_api_pair(plan_path, case)
        evaluator.evaluate(
            plan_path,
            seal,
            case / "evaluation",
            truth_loader=lambda _: pytest.fail("truth opened early"),
        )
    assert not (case / "evaluation").exists()


def test_transport_timeout_remains_descriptive_failure(tmp_path, monkeypatch):
    plan_path, case = _transport_fixture(tmp_path, monkeypatch)
    _change(
        case / "reviewed/stage_provenance.json",
        lambda d: d.update(status="partial", complete=False, returncode=124),
        sealed_kind="astra_reproduction_stage",
    )
    seal = evaluator.seal_api_pair(plan_path, case)
    report = evaluator.evaluate(plan_path, seal, case / "evaluation")
    assert report["stages"]["reviewed"]["metrics"]["summary"]["exact_measures"] == 7
    assert report["gate"]["advance_to_rumichaca"] is False


@pytest.mark.parametrize("mutation", ["source", "author_prompt", "reviewer_candidate"])
@pytest.mark.parametrize("target", ["aviador_s8", "rumichaca_full"])
def test_resealed_packet_cannot_change_actual_source_or_rendered_prompt(
    tmp_path, monkeypatch, mutation, target
):
    plan_path, case = _transport_fixture(tmp_path, monkeypatch, target=target)
    stage = "initial" if mutation == "author_prompt" else "reviewed"
    packet = case / f"{stage}_packet"
    if mutation == "source":
        (packet / "bundle/source.pdf").write_bytes(b"changed source")
    else:
        prompt = packet / "bundle/prompt.txt"
        prompt.write_text(prompt.read_text() + "Extra instructions or different candidate")
        _change(
            packet / "preparation.json", lambda d: d.update(prompt_sha256=evaluator.sha(prompt))
        )
    (packet / "seal.json").unlink()
    runner.seal(packet, "astra_reproduction_input")
    packet_hash = evaluator.sha(packet / "seal.json")
    # Recompute every linkage so only the actual prompt/source contract can reject it.
    _change(
        case / f"{stage}_audit/audit.json",
        lambda d: d.update(input_seal_sha256=packet_hash),
        sealed_kind="astra_reproduction_audit",
    )
    _change(
        case / stage / "stage_provenance.json",
        lambda d: d.update(input_seal_sha256=packet_hash),
        sealed_kind="astra_reproduction_stage",
    )
    seal = evaluator.seal_api_pair(plan_path, case)
    with pytest.raises(ValueError, match="API input, prompt, audit or execution linkage"):
        evaluator.evaluate(
            plan_path,
            seal,
            case / "evaluation",
            truth_loader=lambda _: pytest.fail("truth opened early"),
        )
    assert not (case / "evaluation").exists()


def test_api_wrapper_cannot_silently_reuse_historical_plan(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="single-target transport plan"):
        evaluator.seal_api_pair(args[0], args[0].parent)


@pytest.mark.parametrize("initial", ["good", "wrong"])
def test_rumichaca_transport_preserves_complete_score_and_both_source_label_versions(
    tmp_path, monkeypatch, initial
):
    plan_path, case = _transport_fixture(
        tmp_path, monkeypatch, target="rumichaca_full", initial=initial
    )
    plan = evaluator.read_json(plan_path)
    assert set(plan["targets"]) == {"rumichaca_full"}
    assert plan["stage_protocol"]["deadline_seconds"] == {
        "rumichaca_full": {"initial": 3600, "reviewed": 1800}
    }
    assert plan["transport_scope"]["rumichaca_scheduled"] is True
    assert plan["transport_scope"]["target_selection"] == "rumichaca_full_score_only"
    assert set(plan["current_harness"]["prompts"]) == {
        "rumichaca_full.txt",
        "rumichaca_full_review.txt",
    }
    seal = evaluator.seal_api_pair(plan_path, case)
    assert "prerequisite_evaluation" not in evaluator.read_json(seal)
    report = evaluator.evaluate(plan_path, seal, case / "evaluation")
    assert report["target_id"] == "rumichaca_full" and report["gate"] is None
    # Counts derive from this synthetic reference, never the real score's 45/132.
    assert report["reference_summary"] == {
        "written_measures": 7,
        "performed_measures": 7,
        "route_status": "resolved",
    }
    assert report["stages"]["reviewed"]["metrics"]["edition_note_f1"] == 1
    if initial == "wrong":
        assert report["stages"]["initial"]["metrics"]["edition_note_f1"] < 1
    for lane in report["stages"].values():
        assert "source_note_f1" in lane["metrics"]  # Existing V3 stays separate.
        assert lane["supplemental_source"]["label_set"] == "reviewed_six_v4"
        assert lane["supplemental_source"]["label_manifest"] == plan["supplemental_source_labels"]
    assert plan["supplemental_source_labels"] in report["pins"]


@pytest.mark.parametrize(
    "mutation",
    [
        "v3_labels",
        "v4_labels",
        "v4_missing",
        "v4_file",
        "deadline",
        "prompt_names",
        "report_target",
        "second_target",
    ],
)
def test_rumichaca_rejects_changed_frozen_controls_before_reference_content(
    tmp_path, monkeypatch, mutation
):
    plan_path, case = _transport_fixture(tmp_path, monkeypatch, target="rumichaca_full")
    plan = evaluator.read_json(plan_path)
    if mutation == "v3_labels":
        plan["targets"]["rumichaca_full"]["labels"]["sha256"] = "a" * 64
    elif mutation == "v4_labels":
        plan["supplemental_source_labels"]["sha256"] = "a" * 64
    elif mutation == "v4_missing":
        del plan["supplemental_source_labels"]
    elif mutation == "v4_file":
        Path(plan["supplemental_source_labels"]["path"]).write_text("changed labels")
    elif mutation == "deadline":
        plan["stage_protocol"]["deadline_seconds"]["rumichaca_full"]["reviewed"] = 1200
    elif mutation == "prompt_names":
        plan["current_harness"]["prompts"]["aviador_s8.txt"] = plan["current_harness"][
            "prompts"
        ].pop("rumichaca_full.txt")
    elif mutation == "second_target":
        plan["targets"]["aviador_s8"] = copy.deepcopy(plan["targets"]["rumichaca_full"])
    else:
        report_path = Path(plan["transport_predecessor"]["report"]["path"])
        _change(report_path, lambda d: d.update(target_id="aviador_s8"))
        plan["transport_predecessor"]["report"] = evaluator.pin(report_path)
    _write(plan_path, plan)
    _change(
        plan_path.parent / "execution_contract.json",
        lambda d: d.update(plan_sha256=evaluator.sha(plan_path)),
    )
    seal = evaluator.seal_api_pair(plan_path, case)
    monkeypatch.setattr(evaluator, "verify_freeze", lambda _: pytest.fail("reference opened early"))
    with pytest.raises(ValueError):
        evaluator.evaluate(plan_path, seal, case / "evaluation")
    assert not (case / "evaluation").exists()


def test_rumichaca_timeout_retains_checkpoint_without_reporting_success(tmp_path, monkeypatch):
    plan_path, case = _transport_fixture(tmp_path, monkeypatch, target="rumichaca_full")
    _change(
        case / "reviewed/stage_provenance.json",
        lambda d: d.update(status="partial", complete=False, returncode=124),
        sealed_kind="astra_reproduction_stage",
    )
    seal = evaluator.seal_api_pair(plan_path, case)
    report = evaluator.evaluate(plan_path, seal, case / "evaluation")
    lane = report["stages"]["reviewed"]
    assert lane["metrics"]["edition_note_f1"] == 1
    assert lane["status"] == "partial" and lane["runtime"]["returncode"] == 124
