"""Sealing checks runtime/input linkage without reading musical reference answers."""

import json

import pytest

from scripts.experiments import evaluate_astra_reproduction as evaluation
from scripts.experiments import run_astra_baseline_reproduction as runner


def pair_fixture(tmp_path):
    plan_path = tmp_path / "evaluation_plan.json"
    runner.write(
        plan_path,
        {
            "evidence_kind": "synthetic_control",
            "targets": {"aviador_s8": {"source_pdf": {"sha256": "a" * 64}}},
        },
    )
    runner.write(tmp_path / "execution_contract.json", {"synthetic": True})
    case = tmp_path / "aviador_s8"
    for stage in evaluation.STAGES:
        packet, audit, result = (case / f"{stage}_packet", case / f"{stage}_audit", case / stage)
        for path in (packet, audit, result):
            path.mkdir(parents=True)
        runner.write(packet / "preparation.json", {"synthetic": True})
        runner.seal(packet, "astra_reproduction_input")
        runner.write(
            audit / "audit.json",
            {
                "status": "passed",
                "input_seal_sha256": evaluation.sha(packet / "seal.json"),
            },
        )
        runner.seal(audit, "astra_reproduction_audit")
        runner.write(
            result / "stage_provenance.json",
            {
                "auth_mode": "chatgpt_subscription",
                "input_seal_sha256": evaluation.sha(packet / "seal.json"),
                "audit_sha256": evaluation.sha(audit / "seal.json"),
            },
        )
        (result / "score.abc").write_text("synthetic draft")
        runner.write(result / "decisions.json", [])
        runner.seal(result, "astra_reproduction_stage")
    return plan_path, case


def test_pair_seal_is_create_once_and_binds_both_saved_stages(tmp_path):
    args = pair_fixture(tmp_path)
    path = evaluation.seal_subscription_pair(*args)
    seal = json.loads(path.read_text())
    assert seal["reference_feedback_given"] is False
    assert set(seal["stages"]) == {"initial", "reviewed"}
    assert seal["plan_sha256"] == evaluation.sha(args[0])
    assert len(seal["artifacts"]) == 16
    with pytest.raises(FileExistsError):
        evaluation.seal_subscription_pair(*args)


def test_pair_seal_refuses_changed_draft_before_publication(tmp_path):
    args = pair_fixture(tmp_path)
    (args[1] / "reviewed/score.abc").write_text("changed after run seal")
    with pytest.raises(ValueError, match="Sealed reproduction artifact changed"):
        evaluation.seal_subscription_pair(*args)
    assert not (args[1] / "submission_seal.json").exists()


def test_pair_seal_refuses_api_run_even_with_self_consistent_hashes(tmp_path):
    args = pair_fixture(tmp_path)
    result = args[1] / "reviewed"
    provenance = json.loads((result / "stage_provenance.json").read_text())
    provenance["auth_mode"] = "openai_api"
    runner.write(result / "stage_provenance.json", provenance)
    (result / "seal.json").unlink()
    runner.seal(result, "astra_reproduction_stage")
    with pytest.raises(ValueError, match="Subscription input"):
        evaluation.seal_subscription_pair(*args)
    assert not (args[1] / "submission_seal.json").exists()
