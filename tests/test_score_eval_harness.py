"""The offline task packet and bounded author feedback never read answer music."""

import hashlib
import json
import shutil

import pytest

from score2abc import score_eval_harness as harness


def sha(data):
    return hashlib.sha256(data).hexdigest()


def frozen_fixture(tmp_path, monkeypatch):
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    works, files = [], {}
    for index in range(6):
        slug = f"work_{index}"
        source = frozen / slug / "source.pdf"
        source.parent.mkdir()
        data = f"synthetic PDF {index}".encode()
        source.write_bytes(data)
        answer = frozen / "answers" / slug / "review.abc"
        answer.parent.mkdir(parents=True)
        answer.write_text(f"answer {index}")
        relative = f"{slug}/source.pdf"
        works.append(
            {
                "slug": slug,
                "source_pdf_path": relative,
                "answer_abc_path": f"answers/{slug}/review.abc",
            }
        )
        files[relative] = {"sha256": sha(data)}
    freeze = {"works": works, "files": files}
    monkeypatch.setattr(harness, "verify_freeze", lambda _: freeze)
    return frozen, freeze


def test_prepare_inputs_copies_only_six_pdfs_and_pins_twelve_model_tasks(tmp_path, monkeypatch):
    frozen, _ = frozen_fixture(tmp_path, monkeypatch)
    destination = tmp_path / "packet"
    contract = harness.prepare_inputs(frozen, destination)
    assert contract["source_policy"] == "pdf_only"
    assert contract["isolation_status"] == "requires_enforced_runner"
    assert contract["execution_status"] == "not_run"
    assert len(contract["tasks"]) == 12
    assert {task["model_id"] for task in contract["tasks"]} == {"astra", "terra"}
    assert {task["max_revision_passes"] for task in contract["tasks"]} == {2}
    assert {task["status"] for task in contract["tasks"]} == {"not_run"}
    assert len({task["slug"] for task in contract["tasks"]}) == 6
    files = {
        str(path.relative_to(destination)) for path in destination.rglob("*") if path.is_file()
    }
    assert files == {"instructions.txt", "tasks.json", "tasks.sha256"} | {
        f"inputs/work_{index}/source.pdf" for index in range(6)
    }
    assert not any("answers" in name or "review.abc" in name for name in files)
    assert all(
        task["input_pdf_sha256"] == sha((destination / task["input_pdf_path"]).read_bytes())
        for task in contract["tasks"]
    )
    assert all(
        task["instructions_sha256"] == sha((destination / "instructions.txt").read_bytes())
        for task in contract["tasks"]
    )
    assert json.loads((destination / "tasks.json").read_text()) == contract
    assert (destination / "tasks.sha256").read_text().strip() == sha(
        (destination / "tasks.json").read_bytes()
    )


def test_prepare_inputs_refuses_existing_or_overlapping_destination(tmp_path, monkeypatch):
    frozen, _ = frozen_fixture(tmp_path, monkeypatch)
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "sentinel"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        harness.prepare_inputs(frozen, existing)
    assert sentinel.read_text() == "preserve"
    for destination in (frozen / "packet", tmp_path):
        with pytest.raises(ValueError, match="separate from frozen references"):
            harness.prepare_inputs(frozen, destination)


def test_prepare_inputs_rechecks_pdf_hash_and_removes_incomplete_packet(tmp_path, monkeypatch):
    frozen, freeze = frozen_fixture(tmp_path, monkeypatch)
    freeze["files"]["work_3/source.pdf"]["sha256"] = "0" * 64
    destination = tmp_path / "packet"
    with pytest.raises(ValueError, match="PDF changed"):
        harness.prepare_inputs(frozen, destination)
    assert not destination.exists()


def feedback(abc, decisions):
    issues = [] if abc == "fixed" else [{"kind": "meter", "measure": 1}]
    return {
        "valid": True,
        "counts": {},
        "issues": issues,
        "warnings": [],
        "route_status": "resolved",
        "musical_accuracy": "not_measured_by_validation",
    }


def test_revision_loop_keeps_initial_and_both_revisions_with_truth_free_feedback(monkeypatch):
    monkeypatch.setattr(harness, "validate_candidate", feedback)
    calls = []

    def revise(draft, decisions, validator_feedback):
        calls.append((draft, decisions, validator_feedback))
        return ("still wrong", []) if len(calls) == 1 else ("fixed", [])

    result = harness.bounded_revision_loop("initial", [], revise)
    assert result["status"] == "validated"
    assert len(calls) == 2
    assert [attempt["abc"] for attempt in result["attempts"]] == ["initial", "still wrong", "fixed"]
    assert [attempt["attempt"] for attempt in result["attempts"]] == [0, 1, 2]
    assert all(
        attempt["abc_sha256"] == sha(attempt["abc"].encode()) for attempt in result["attempts"]
    )
    assert all("reference" not in str(call) for call in calls)
    assert result["musical_accuracy"] == "not_measured_by_validation"


def test_revision_loop_never_invents_a_repair_or_exceeds_two_passes(monkeypatch):
    monkeypatch.setattr(harness, "validate_candidate", feedback)
    result = harness.bounded_revision_loop(
        "wrong", [], lambda draft, decisions, _: (draft, decisions)
    )
    assert result["status"] == "partial"
    assert [attempt["abc"] for attempt in result["attempts"]] == ["wrong"] * 3
    with pytest.raises(ValueError, match="zero to two"):
        harness.bounded_revision_loop("wrong", [], lambda *_: None, max_revisions=3)
    with pytest.raises(ValueError, match="zero to two"):
        harness.bounded_revision_loop("wrong", [], lambda *_: None, max_revisions=True)


def test_revision_attempts_keep_prior_decision_snapshots_when_callback_mutates(monkeypatch):
    monkeypatch.setattr(harness, "validate_candidate", feedback)
    decisions = [{"id": "initial"}]

    def revise(draft, current, _):
        current[0]["id"] = "revised"
        return "fixed", current

    result = harness.bounded_revision_loop("wrong", decisions, revise)
    assert result["attempts"][0]["decisions"] == [{"id": "initial"}]
    assert result["attempts"][1]["decisions"] == [{"id": "revised"}]


def test_revision_loop_stops_on_valid_first_draft_without_callback(monkeypatch):
    monkeypatch.setattr(harness, "validate_candidate", feedback)

    def forbidden(*_):
        raise AssertionError("revision callback should not run")

    result = harness.bounded_revision_loop("fixed", [], forbidden)
    assert result["status"] == "validated"
    assert len(result["attempts"]) == 1


@pytest.mark.parametrize(
    "revise,expected_error",
    [
        (
            lambda *_: (_ for _ in ()).throw(RuntimeError("private token")),
            "Revision adapter failed: RuntimeError",
        ),
        (lambda *_: (123, []), "Invalid revision response"),
        (lambda *_: ("fixed", {}), "Invalid revision response"),
    ],
)
def test_revision_callback_failure_keeps_attempt_and_hides_exception_text(
    monkeypatch, revise, expected_error
):
    monkeypatch.setattr(harness, "validate_candidate", feedback)
    result = harness.bounded_revision_loop("wrong", [], revise)
    assert result["status"] == "partial"
    assert result["error"] == expected_error
    assert len(result["attempts"]) == 1
    assert result["attempts"][0]["abc"] == "wrong"


@pytest.mark.skipif(
    not (shutil.which("abc2svg") and shutil.which("node")),
    reason="optional local ABC renderer unavailable",
)
def test_six_synthetic_grader_controls_detect_equivalence_and_mutations(tmp_path):
    destination = tmp_path / "grader_demo"
    report = harness.build_grader_demo(destination)
    assert report["evidence_kind"] == "synthetic_demo"
    assert report["frozen_manifest_sha256"] is None
    assert report["labels_sha256"] is None
    cases = {case["slug"]: case for case in report["cases"]}
    assert set(cases) == {"equivalent", "pitch", "duration", "chord", "missing", "route"}
    assert cases["equivalent"]["results"]["mutated"]["metrics"]["edition_exact_bar_rate"] == 1
    for kind in ("pitch", "duration", "missing"):
        assert cases[kind]["results"]["mutated"]["metrics"]["edition_note_f1"] < 1
    assert cases["chord"]["results"]["mutated"]["metrics"]["chord_f1"] < 1
    assert cases["route"]["results"]["mutated"]["metrics"]["form_exact_match"] is False
    assert len(list(destination.glob("*.abc"))) == 12
