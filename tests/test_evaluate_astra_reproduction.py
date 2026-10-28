"""Synthetic controls for the sealed assisted-workflow adapter; no model calls."""

import copy
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from test_evaluate_astra_music_benchmark import _predictions, _sealed_fixture, _submission

from score2abc.score_eval_source import NOTE_FIELDS, note_events_sha256
from scripts.experiments import evaluate_astra_reproduction as evaluator


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _normalized(text):
    identities = [{"system_measure_index": i} for i in range(1, 8)]
    rows = _predictions(_submission(), identities)
    measures = []
    for i, row in enumerate(rows, 1):
        measures.append(
            {
                "number": i,
                "duration_ticks": 1152,
                "expected_ticks": 1152,
                "meter_valid": True,
                "partial_reason": None,
                "meter": {"numerator": 3, "denominator": 4},
                "notes": [
                    {
                        "pitch_midi": n["pitch_midi"],
                        "onset_ticks": n["onset_beats"] * 384,
                        "duration_ticks": n["duration_beats"] * 384,
                        "tie_start": False,
                        "tie_end": False,
                    }
                    for n in row["notes"]
                ],
                "rests": [
                    {
                        "onset_ticks": r["onset_beats"] * 384,
                        "duration_ticks": r["duration_beats"] * 384,
                        "visible": True,
                    }
                    for r in row["rests"]
                ],
                "chords": [],
            }
        )
    if text == "wrong":
        measures[0]["notes"][0]["pitch_midi"] += 1
    elif text == "missing":
        measures.pop()
    elif text == "extra":
        measures.append({**copy.deepcopy(measures[-1]), "number": 8})
    elif text == "overflow":
        measures[0]["notes"][0]["duration_ticks"] = 1536
    elif text == "invalid":
        measures = []
    return {
        "valid": text != "invalid",
        "errors": [] if text != "invalid" else ["invalid ABC"],
        "warnings": [],
        "measures": measures,
        "ticks_per_quarter": 384,
        "route": list(range(1, len(measures) + 1)),
        "route_status": "resolved",
    }


def _fixture(tmp_path, monkeypatch, initial="good", reviewed="good"):
    old = _sealed_fixture(tmp_path)
    protocol = json.loads((old / "protocol.json").read_text())
    root = tmp_path / "reproduction"
    root.mkdir()
    source = root / "source.pdf"
    source.write_bytes(b"synthetic PDF source")
    _write(root / "references/manifest.json", {})
    _write(root / "labels/manifest.json", {})
    (root / "references/review.abc").write_text("good")
    deadlines = {
        "aviador_s8": {"initial": 1800, "reviewed": 1200},
        "rumichaca_full": {"initial": 3600, "reviewed": 1800},
    }
    plan = {
        "schema_version": 1,
        "kind": "astra_assisted_workflow_reproduction_plan",
        "status": "preregistered_before_new_author_inference",
        "workflow_id": evaluator.WORKFLOW,
        "model": "gpt-6-astra",
        "reasoning_effort": "xhigh",
        "source_policy": "supplied_pdf_only",
        "stages": list(evaluator.STAGES),
        "evidence_kind": "synthetic_control",
        "stage_protocol": {"deadline_seconds": deadlines},
        "targets": {
            "aviador_s8": {
                "slug": "demo",
                "system_index": 8,
                "source_pdf": evaluator.pin(source),
                "grading_only_expected_measure_indices": list(range(1, 8)),
                "historical_reference_pins": protocol["reference_pins"],
                "historical_protocol": evaluator.pin(old / "protocol.json"),
                "advance_gate": {
                    "initial_usable_complete": {
                        "abc_valid": True,
                        "written_measures": 7,
                        "all_declared_meter_bars_valid": True,
                    },
                    "reviewed_exact_measures_min": 6,
                    "reviewed_strict_note_f1_min": 0.95,
                },
                "unsupported_scope": ["ties", "form"],
            },
            "rumichaca_full": {
                "slug": "demo",
                "source_pdf": evaluator.pin(source),
                "frozen_reference": evaluator.pin(root / "references/manifest.json"),
                "labels": evaluator.pin(root / "labels/manifest.json"),
            },
        },
    }
    plan_path = root / "evaluation_plan.json"
    _write(plan_path, plan)
    execution = {
        "kind": "astra_assisted_workflow_execution_contract",
        "status": "preregistered_before_author_inference",
        "plan_sha256": evaluator.sha(plan_path),
        "gate_eligibility": {
            "both_stage_status": "complete",
            "both_stage_returncode": 0,
            "checkpoint_completion_claim_alone_is_insufficient": True,
        },
        "deadline_seconds": deadlines,
    }
    execution_path = root / "execution_contract.json"
    _write(execution_path, execution)
    stages, artifacts = {}, []
    for stage, value in (("initial", initial), ("reviewed", reviewed)):
        path = root / stage
        path.mkdir()
        (path / "score.abc").write_text(value)
        _write(path / "decisions.json", [])
        provenance = {
            "stage": stage,
            "model": "gpt-6-astra",
            "reasoning_effort": "xhigh",
            "workflow_id": evaluator.WORKFLOW,
            "source_pdf_sha256": evaluator.sha(source),
            "reference_access": False,
            "complete": True,
            "status": "complete",
            "returncode": 0,
            "elapsed_seconds": 1,
            "deadline_seconds": deadlines["aviador_s8"][stage],
            "started_at": "2026-10-28T00:00:00+00:00",
            "finished_at": "2026-10-28T00:00:01+00:00",
            "error": None,
        }
        if stage == "reviewed":
            provenance.update(
                independent_reader=True, initial_abc_sha256=stages["initial"]["abc"]["sha256"]
            )
        _write(path / "provenance.json", provenance)
        stages[stage] = {
            kind: evaluator.pin(path / name)
            for kind, name in (
                ("abc", "score.abc"),
                ("decisions", "decisions.json"),
                ("provenance", "provenance.json"),
            )
        }
        artifacts.extend(stages[stage].values())
    seal = {
        "schema_version": 1,
        "kind": evaluator.SEAL_KIND,
        "status": evaluator.SEAL_STATUS,
        "plan_sha256": evaluator.sha(plan_path),
        "target_id": "aviador_s8",
        "source_pdf_sha256": evaluator.sha(source),
        "reference_feedback_given": False,
        "evidence_kind": "synthetic_control",
        "execution_contract": evaluator.pin(execution_path),
        "stages": stages,
        "artifacts": artifacts,
    }
    seal_path = root / "submission_seal.json"
    _write(seal_path, seal)
    monkeypatch.setattr(evaluator, "normalize_abc", _normalized)
    monkeypatch.setattr(evaluator, "renderer_identity", lambda: {"status": "synthetic_mock"})
    return plan_path, seal_path, root / "evaluation"


def test_exact_control_reproduces_historical_scorer_without_becoming_model_result(
    tmp_path, monkeypatch
):
    args = _fixture(tmp_path, monkeypatch)
    report = evaluator.evaluate(*args)
    assert report["evidence_kind"] == "synthetic_control"
    assert report["baseline"]["historical_summary_reproduced"] is True
    assert report["stages"]["initial"]["metrics"]["summary"]["exact_measures"] == 7
    assert report["gate"]["exact_historical_reproduction_initial_only"] is True
    assert report["gate"]["advance_to_rumichaca"] is True
    assert report["gate"]["independent_review_made_gate_pass"] is False


def test_reader_improvement_is_separate_from_initial_reproduction(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch, initial="wrong")
    report = evaluator.evaluate(*args)
    assert report["stages"]["initial"]["metrics"]["summary"]["exact_measures"] == 6
    assert report["stages"]["reviewed"]["metrics"]["summary"]["exact_measures"] == 7
    assert report["gate"]["exact_historical_reproduction_initial_only"] is False
    assert report["gate"]["advance_to_rumichaca"] is True
    assert report["gate"]["independent_review_made_gate_pass"] is True


@pytest.mark.parametrize("initial", ["missing", "extra", "overflow", "invalid"])
def test_incomplete_or_invalid_initial_cannot_advance_despite_perfect_reader(
    tmp_path, monkeypatch, initial
):
    args = _fixture(tmp_path, monkeypatch, initial=initial)
    report = evaluator.evaluate(*args)
    assert report["gate"]["advance_to_rumichaca"] is False
    assert report["stages"]["reviewed"]["metrics"]["summary"]["exact_measures"] == 7


def test_both_stages_and_artifacts_verified_before_truth_is_opened(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    seal = json.loads(args[1].read_text())
    Path(seal["stages"]["reviewed"]["abc"]["path"]).write_text("changed after seal")
    opened = []
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluator.evaluate(*args, truth_loader=lambda p: opened.append(p))
    assert opened == []
    assert not args[2].exists()


def test_source_recheck_detects_changes_during_grading_and_keeps_no_report(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    seal = json.loads(args[1].read_text())
    initial_path = Path(seal["stages"]["initial"]["abc"]["path"])

    def mutate_then_load(path):
        initial_path.write_text("changed during grading")
        return evaluator.historical._read_jsonl(path)

    with pytest.raises(ValueError, match="recheck hash mismatch"):
        evaluator.evaluate(*args, truth_loader=mutate_then_load)
    assert not args[2].exists()


@pytest.mark.parametrize("stage", ["initial", "reviewed"])
def test_timeout_checkpoint_scores_but_cannot_open_advancement_gate(tmp_path, monkeypatch, stage):
    args = _fixture(tmp_path, monkeypatch)
    seal = json.loads(args[1].read_text())
    record = seal["stages"][stage]["provenance"]
    path = Path(record["path"])
    value = json.loads(path.read_text())
    value.update(status="partial", returncode=124, error="deadline exceeded")
    _write(path, value)
    record["sha256"] = evaluator.sha(path)
    for pin in seal["artifacts"]:
        if pin["path"] == str(path):
            pin["sha256"] = evaluator.sha(path)
    _write(args[1], seal)
    report = evaluator.evaluate(*args)
    assert report["stages"][stage]["metrics"]["summary"]["exact_measures"] == 7
    assert report["gate"]["advance_to_rumichaca"] is False
    assert report["gate"][f"{stage}_runtime_success"] is False


def test_existing_report_and_changed_gate_are_rejected(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    evaluator.evaluate(*args)
    with pytest.raises(FileExistsError, match="create-once"):
        evaluator.evaluate(*args)


def test_reviewed_completion_claim_is_required_even_with_successful_process(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    seal = json.loads(args[1].read_text())
    record = seal["stages"]["reviewed"]["provenance"]
    path = Path(record["path"])
    data = json.loads(path.read_text())
    data["complete"] = False
    _write(path, data)
    record["sha256"] = evaluator.sha(path)
    for item in seal["artifacts"]:
        if item["path"] == str(path):
            item["sha256"] = evaluator.sha(path)
    _write(args[1], seal)
    report = evaluator.evaluate(*args)
    assert report["gate"]["reviewed_runtime_success"] is True
    assert report["gate"]["reviewed_met_final_quality_threshold"] is True
    assert report["gate"]["advance_to_rumichaca"] is False
    assert any("did not declare" in reason for reason in report["gate"]["blocking_reasons"])


def test_optional_ledger_is_consistency_diagnostic_not_an_alternate_score(tmp_path):
    path = tmp_path / "ledger.json"
    _write(path, _submission())
    stage = {"normalized": _normalized("good"), "event_ledger": path}
    assert evaluator.ledger_consistency(stage)["status"] == "consistent"
    payload = _submission()
    payload["measures"][0]["events"][0]["pitch"] = "F4"
    _write(path, payload)
    assert evaluator.ledger_consistency(stage)["status"] == "differs_from_authoritative_abc"


def _full_score_fixture(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    prerequisite = evaluator.evaluate(*args)
    seal = json.loads(args[1].read_text())
    root = args[1].parent / "full-score"
    root.mkdir()
    artifacts, stages = [], {}
    start = datetime.fromisoformat(prerequisite["created_at"]) + timedelta(seconds=1)
    for stage, deadline in (("initial", 3600), ("reviewed", 1800)):
        directory = root / stage
        directory.mkdir()
        stages[stage] = {}
        for kind in ("abc", "decisions", "provenance"):
            original = Path(seal["stages"][stage][kind]["path"])
            destination = directory / original.name
            shutil.copyfile(original, destination)
            if kind == "provenance":
                data = json.loads(destination.read_text())
                data.update(
                    deadline_seconds=deadline,
                    started_at=start.isoformat(),
                    finished_at=(start + timedelta(seconds=1)).isoformat(),
                )
                _write(destination, data)
            stages[stage][kind] = evaluator.pin(destination)
            artifacts.append(stages[stage][kind])
    seal.update(
        target_id="rumichaca_full",
        stages=stages,
        artifacts=artifacts,
        prerequisite_evaluation=evaluator.pin(args[2] / "report.json"),
    )
    seal_path = root / "submission_seal.json"
    _write(seal_path, seal)
    reference = _normalized("good")
    monkeypatch.setattr(
        evaluator,
        "verify_freeze",
        lambda _: {
            "works": [
                {"slug": "demo", "answer_abc_path": "review.abc", "source_pdf_path": "source.pdf"}
            ],
            "files": {"source.pdf": {"sha256": seal["source_pdf_sha256"]}},
        },
    )
    monkeypatch.setattr(
        evaluator,
        "verify_labels",
        lambda *_: {
            "label_set": "reviewed_six_v3",
            "works": [
                {
                    "slug": "demo",
                    "source_audit": {
                        "ticks_per_quarter": 384,
                        "total_source_measures": 7,
                        "measures": [
                            {
                                "edition_bar": 1,
                                "source_bar": 1,
                                "dimensions": ["notes"],
                                "note_fields": list(NOTE_FIELDS),
                                "expected_noteheads": 2,
                                "note_events_sha256": note_events_sha256(reference["measures"][0]),
                                "source_location": {"page": 1, "system": 1, "measure": 1},
                            }
                        ],
                    },
                }
            ],
        },
    )
    monkeypatch.setattr(evaluator, "validate_candidate", lambda *_: {"issues": [], "valid": True})
    return args[0], seal_path, root / "evaluation"


def test_full_score_adapter_uses_existing_metrics_under_distinct_workflow_identity(
    tmp_path, monkeypatch
):
    args = _full_score_fixture(tmp_path, monkeypatch)
    report = evaluator.evaluate(*args)
    assert report["target_id"] == "rumichaca_full"
    assert report["workflow_id"] == evaluator.WORKFLOW
    assert "generation_sha256" not in report
    assert report["gate"] is None
    for lane in report["stages"].values():
        assert lane["status"] == "complete"
        assert lane["metrics"]["edition_note_f1"] == 1
        assert lane["metrics"]["source_note_f1"] == 1
        assert lane["metrics"]["editorial_acceptance"] is None


def test_full_score_requires_passing_prior_gate_before_reference_read(tmp_path, monkeypatch):
    args = _full_score_fixture(tmp_path, monkeypatch)
    seal = json.loads(args[1].read_text())
    del seal["prerequisite_evaluation"]
    _write(args[1], seal)
    monkeypatch.setattr(evaluator, "verify_freeze", lambda _: pytest.fail("reference opened early"))
    with pytest.raises(ValueError, match="Aviador gate report"):
        evaluator.evaluate(*args)


@pytest.mark.skipif(
    not (shutil.which("node") and shutil.which("abc2svg")),
    reason="optional ABC renderer unavailable",
)
def test_real_abc_adapter_matches_synthetic_polyphonic_notes_and_rests():
    abc = "X:1\nM:3/4\nL:1/4\nK:C\n[CE] z2 | D2 z | c3 | d3 | e3 | f3 | g3 |]\n"
    parsed = evaluator.normalize_abc(abc)
    identities = [{"system_measure_index": i} for i in range(1, 8)]
    rows, validation = evaluator.abc_to_historical(parsed, identities)
    assert validation["all_declared_meter_bars_valid"] is True
    expected = _predictions(_submission(), identities)
    for row in [*rows, *expected]:
        row["notes"].sort(key=lambda n: (n["onset_beats"], n["pitch_midi"]))
    assert rows == expected
