"""No-network controls for final gate comparison and stopped-pair reporting."""

import copy
import json

import pytest
from test_score_eval import bar, score

from scripts import evaluate_completion_gate_pilot as pilot


@pytest.fixture
def preservation(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "ROOT", tmp_path)
    reviewed = tmp_path / "out/test/overrides/review.json"
    reviewed.parent.mkdir(parents=True)
    reviewed.write_text('{"review_state":"reviewed"}')
    payload = {
        "status": "snapshotted_before_inference",
        "files": 1,
        "before": {str(reviewed.relative_to(tmp_path)): pilot.grader.sha(reviewed)},
    }
    path = tmp_path / "live_review_preservation.json"
    path.write_text(json.dumps(payload))
    return path, reviewed, payload


def test_wrapped_preservation_snapshot_counts_and_verifies_actual_review_files(preservation):
    path, reviewed, payload = preservation
    assert pilot.verify_live_reviews(path) == payload["before"]
    assert len(pilot.verify_live_reviews(path)) == 1
    reviewed.write_text("changed human draft")
    with pytest.raises(ValueError, match="changed during execution"):
        pilot.verify_live_reviews(path)


@pytest.mark.parametrize(
    "change", ["flat", "wrong_count", "boolean_count", "wrong_status", "escape", "bad_hash"]
)
def test_preservation_schema_and_paths_fail_closed(preservation, change):
    path, _, payload = preservation
    if change == "flat":
        payload = payload["before"]
    elif change == "wrong_count":
        payload["files"] = 3
    elif change == "boolean_count":
        payload["files"] = True
    elif change == "wrong_status":
        payload["status"] = "not_preregistered"
    elif change == "escape":
        payload["before"] = {"../outside.json": "a" * 64}
    elif change == "bad_hash":
        payload["before"] = {next(iter(payload["before"])): "not a digest"}
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="preservation"):
        pilot.verify_live_reviews(path)


def contract():
    plan = {
        "created_at": "2026-10-29T23:00:00+00:00",
        "model": "gpt-6-astra",
        "reasoning_effort": "xhigh",
        "workflow_id": "source-only",
        "target": {"source_pdf": {"sha256": "source"}},
        "deadline_seconds": {"initial": 3600, "reviewed": 1800},
        "completion_policy": "completion_gate_v1",
        "completion_policy_sha256": "policy",
    }
    provenance = {
        "started_at": "2026-10-29T23:01:00+00:00",
        "stage": "reviewed",
        "auth_mode": "openai_api",
        "source_execution": "sandboxed_source_process",
        "model": "gpt-6-astra",
        "reasoning_effort": "xhigh",
        "workflow_id": "source-only",
        "source_pdf_sha256": "source",
        "input_seal_sha256": "packet",
        "deadline_seconds": 1800,
        "reference_access": False,
        "independent_reader": True,
        "initial_abc_sha256": "author",
        "completion_policy": "completion_gate_v1",
        "completion_policy_sha256": "policy",
        "status": "complete",
        "complete": True,
        "returncode": 0,
        "completed_turn": True,
        "quarantined": False,
        "timed_out": False,
        "error": None,
    }
    return plan, provenance


def test_completed_and_partial_stages_have_distinct_meaning():
    plan, provenance = contract()
    pilot.validate_stage(provenance, plan, "reviewed", "packet", "author")
    provenance.update(status="partial", complete=False, completed_turn=False, returncode=1)
    pilot.validate_stage(provenance, plan, "reviewed", "packet", "author")


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "failed"),
        ("backend", "subscription"),
        ("input_seal_sha256", "unrelated"),
        ("completion_policy", None),
        ("completion_policy_sha256", "other"),
    ],
)
def test_audit_must_match_stage_packet_and_gate(field, value):
    plan, _ = contract()
    audit = {
        "status": "passed",
        "backend": "api",
        "input_seal_sha256": "packet",
        "model": plan["model"],
        "reasoning_effort": plan["reasoning_effort"],
        "reference_access": False,
        "tools": ["inspect_source", "save_draft"],
        "completion_policy": plan["completion_policy"],
        "completion_policy_sha256": plan["completion_policy_sha256"],
    }
    pilot.validate_audit(audit, plan, "packet")
    audit[field] = value
    with pytest.raises(ValueError, match="passing audit"):
        pilot.validate_audit(audit, plan, "packet")


def test_changed_renderer_cannot_be_reported_as_identical_grading(monkeypatch):
    baseline = {"implementations": {"renderer": {"files": {"abc2svg": "old"}}}}
    monkeypatch.setattr(pilot.grader, "renderer_identity", lambda: {"files": {"abc2svg": "old"}})
    pilot.validate_renderer(baseline)
    monkeypatch.setattr(pilot.grader, "renderer_identity", lambda: {"files": {"abc2svg": "new"}})
    with pytest.raises(ValueError, match="renderer differs"):
        pilot.validate_renderer(baseline)


@pytest.mark.parametrize(
    "field,value",
    [
        ("budget_usd", "5"),
        ("max_output_tokens_per_response", 1000),
        ("spending_policy", "dollar_cap"),
    ],
)
def test_spending_and_output_allowance_are_matched(field, value):
    actual = {
        "budget_usd": None,
        "max_output_tokens_per_response": 25000,
        "spending_policy": "fixed_output_no_dollar_cap",
    }
    pilot.validate_spending(actual, actual)
    contract = actual.copy()
    actual[field] = value
    with pytest.raises(ValueError, match="allowance differs"):
        pilot.validate_spending(actual, contract)


def test_packet_source_and_prompt_bytes_are_checked(tmp_path, monkeypatch):
    plan, _ = contract()
    (tmp_path / "source.pdf").write_bytes(b"source")
    (tmp_path / "prompt.txt").write_text("task")
    plan["target"]["source_pdf"]["sha256"] = pilot.grader.sha(tmp_path / "source.pdf")
    metadata = {
        "backend": "api",
        "model": plan["model"],
        "reasoning_effort": plan["reasoning_effort"],
        "source_pdf_sha256": plan["target"]["source_pdf"]["sha256"],
        "limit_seconds": 1800,
        "reference_access": False,
        "prompt_sha256": pilot.grader.sha(tmp_path / "prompt.txt"),
    }
    monkeypatch.setattr(pilot.runner, "verify_source_bundle", lambda *args: None)
    pilot.validate_packet(tmp_path, metadata, plan, "reviewed")
    (tmp_path / "source.pdf").write_bytes(b"another PDF")
    with pytest.raises(ValueError, match="source or task bytes"):
        pilot.validate_packet(tmp_path, metadata, plan, "reviewed")


@pytest.mark.parametrize(
    "field,value",
    [
        ("completion_policy", None),
        ("completion_policy_sha256", "old"),
        ("reference_access", True),
        ("deadline_seconds", 1200),
        ("initial_abc_sha256", "other"),
        ("timed_out", True),
        ("quarantined", True),
        ("returncode", 1),
        ("complete", False),
        ("completed_turn", False),
    ],
)
def test_contract_mismatch_and_false_completion_are_rejected(field, value):
    plan, provenance = contract()
    provenance[field] = value
    with pytest.raises(ValueError):
        pilot.validate_stage(provenance, plan, "reviewed", "packet", "author")


def pair(status="complete"):
    author = {"status": status, "complete": status == "complete"}
    return {
        "backend": "api",
        "completion_policy": "completion_gate_v1",
        "automatic_retries": 0,
        "no_dollar_cap": True,
        "budget_usd": None,
        "stages": {
            "initial": author,
            **({"reviewed": copy.deepcopy(author)} if status == "complete" else {}),
        },
        "complete": status == "complete",
    }


def test_terminal_author_stop_does_not_invent_review():
    assert pilot.validate_pair_status(pair()) == ["initial", "reviewed"]
    assert pilot.validate_pair_status(pair("partial")) == ["initial"]
    assert pilot.validate_pair_status(pair("failed")) == ["initial"]
    data = pair("partial")
    data["stages"]["reviewed"] = {"status": "complete", "complete": True}
    with pytest.raises(ValueError, match="Reader must run"):
        pilot.validate_pair_status(data)


def test_missing_review_after_complete_author_is_not_yet_terminal():
    data = pair()
    del data["stages"]["reviewed"]
    data["complete"] = False
    with pytest.raises(ValueError, match="Reader must run"):
        pilot.validate_pair_status(data)


def test_no_reference_read_before_terminal_and_sealed_input_verification(tmp_path, monkeypatch):
    accessed = []

    def refuse(_):
        raise ValueError("missing terminal evidence")

    monkeypatch.setattr(pilot, "verify_inputs", refuse)
    monkeypatch.setattr(pilot, "grade_available", lambda *args: accessed.append("reference"))
    with pytest.raises(ValueError, match="terminal evidence"):
        pilot.evaluate(tmp_path)
    assert accessed == []
    assert not (tmp_path / "evaluation").exists()


@pytest.fixture
def grading(tmp_path, monkeypatch):
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "manifest.json").write_text("{}")
    (frozen / "answer.abc").write_text("reference")
    labels_path = tmp_path / "labels.json"
    labels_path.write_text("{}")
    target = {
        "slug": "test",
        "source_pdf": {"sha256": "source"},
        "frozen_reference": pilot.grader.pin(frozen / "manifest.json"),
        "labels": pilot.grader.pin(labels_path),
    }
    monkeypatch.setattr(
        pilot.grader,
        "verify_freeze",
        lambda path: {
            "works": [
                {"slug": "test", "source_pdf_path": "source.pdf", "answer_abc_path": "answer.abc"}
            ],
            "files": {"source.pdf": {"sha256": "source"}},
        },
    )
    monkeypatch.setattr(
        pilot.grader,
        "verify_labels",
        lambda *args: {
            "label_set": "reviewed_six_v3",
            "works": [{"slug": "test"}],
        },
    )
    monkeypatch.setattr(pilot.grader, "_source_required_edition_bars", lambda *args: {1})
    monkeypatch.setattr(
        pilot.grader,
        "score_source",
        lambda *args: {
            "metrics": {"source_note_f1": 0.75},
            "metric_status": {"source_note_f1": "test_subset"},
            "coverage": {"source_scored_measures": 1},
            "mismatches": [],
            "source_scope": "test",
        },
    )
    reference = score([bar(1, 60), bar(2, 62)])
    candidate = score([bar(1, 60), bar(2, 63)])

    def normalize(abc):
        return reference if abc == "reference" else candidate

    monkeypatch.setattr(pilot, "normalize_abc", normalize)
    monkeypatch.setattr(pilot.grader, "normalize_abc", normalize)
    monkeypatch.setattr(
        pilot.grader,
        "validate_candidate",
        lambda *args: {
            "issues": [],
            "counts": candidate["counts"],
            "route_status": "resolved",
        },
    )
    stages = {}
    for name in pilot.STAGES:
        folder = tmp_path / name
        folder.mkdir()
        (folder / "score.abc").write_text("candidate")
        (folder / "decisions.json").write_text("[]")
        stages[name] = {
            "abc": folder / "score.abc",
            "folder": folder,
            "provenance_data": {"status": "complete", "returncode": 0, "complete": True},
            "normalized": candidate,
            "decisions_data": [],
            "gate": {"completion_ready": True, "status": "ready"},
        }
    return target, stages


def test_numerical_results_match_unchanged_previous_rumichaca_adapter(grading):
    target, stages = grading
    expected = pilot.grader._rumichaca({"targets": {"rumichaca_full": target}}, stages, [])
    actual = pilot.grade_available(target, stages, [])
    for stage in pilot.STAGES:
        without_gate = {
            key: value for key, value in actual[stage].items() if key != "completion_gate"
        }
        assert without_gate == expected["stages"][stage]


def test_stopped_author_is_graded_as_partial_without_synthetic_reader(grading):
    target, stages = grading
    initial = stages["initial"]
    initial["provenance_data"].update(status="partial", complete=False)
    result = pilot.grade_available(target, {"initial": initial}, [])
    assert set(result) == {"initial"}
    assert result["initial"]["status"] == "partial"
    assert result["initial"]["metrics"]["edition_note_f1"] == 0.5


def test_missing_or_quarantined_candidate_is_not_scored(grading):
    target, stages = grading
    initial = stages["initial"]
    initial["abc"] = None
    result = pilot.grade_available(target, {"initial": initial}, [])
    assert result["initial"]["status"] == "not_scored"
    assert result["initial"]["metrics"] == {}


def report():
    lane = {
        "status": "partial",
        "metrics": {
            "edition_note_f1": 0.8,
            "source_note_f1": 0.7,
            "chord_f1": 0.4,
            "meter_valid_rate": 0.9,
            "form_exact_match": False,
        },
        "coverage": {"candidate_measures": 45, "reference_measures": 45},
        "validation": {
            "issues": [{"kind": "meter", "measure": 16}],
            "counts": {"performed_measures": 117},
        },
        "runtime": {"status": "complete", "elapsed_seconds": 60, "usage": None},
        "completion_gate": {
            "status": "ready_with_source_exceptions",
            "completion_ready": True,
            "unresolved_issues": [],
            "accepted_source_exceptions": [
                {
                    "resolution": {
                        "location": "bar 16",
                        "adopted_choice": "<script>uncertain</script>",
                        "reason": "Damaged ink",
                        "uncertainty": "High",
                        "inspection_call": 3,
                    }
                }
            ],
        },
    }
    return {
        "stages": {"initial": copy.deepcopy(lane), "reviewed": lane},
        "subscription_baseline": {"initial": lane, "reviewed": lane},
        "previous_api": {"initial": lane, "reviewed": lane},
        "pair_complete": True,
        "conservative_estimate_usd": "4.50",
        "limitations": ["Consumed"],
        "traces": {},
        "live_reviews_unchanged": 13,
    }


def test_dashboard_keeps_gate_exceptions_separate_from_musical_warnings():
    data = report()
    page = pilot.render(data)
    assert "Gate passed with documented source exceptions" in page
    assert "1 raw validation issues · 1 exceptions" in page
    assert "model decision and inspected crop provenance" in page
    assert "nine pre-existing model-audited bars" in page
    assert "&lt;script&gt;uncertain&lt;/script&gt;" in page
    assert "<script>" not in page
    assert "actual billing unverified" in page


def test_dashboard_does_not_compare_author_as_new_review():
    data = report()
    del data["stages"]["reviewed"]
    data["pair_complete"] = False
    page = pilot.render(data)
    assert "independent review did not run" in page
    assert "No reviewed result" in page
    assert "percentage points" not in page


def test_cost_total_does_not_turn_missing_usage_into_zero():
    assert (
        pilot.cost_total(
            {
                "initial": {"conservative_estimate_usd": "1.25"},
                "reviewed": {"conservative_estimate_usd": "2.50"},
            }
        )
        == "3.75"
    )
    assert pilot.cost_total({"initial": {"conservative_estimate_usd": None}}) is None
