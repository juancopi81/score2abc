"""Offline structural gate controls; no model calls or musical truth required."""

import copy
import json
import subprocess

import pytest

from scripts.experiments import astra_source_tools as legacy
from scripts.experiments import astra_source_tools_gated as source
from scripts.experiments import completion_gate as gate

ABC = "X:1\nM:3/4\nL:1/4\nK:C\nC D E|]"


def normalized(*, duration=1152, valid=True, route="resolved", partial=None):
    return {
        "valid": valid,
        "errors": [],
        "warnings": [],
        "route_status": route,
        "measures": [
            {
                "number": 1,
                "duration_ticks": duration,
                "expected_ticks": 1152,
                "meter_valid": duration == 1152,
                "partial_reason": partial,
            }
        ],
        "counts": {"measures": 1},
    }


def assess(value, **kwargs):
    return gate.evaluate(
        value,
        [],
        abc_sha256=gate.sha(ABC.encode()),
        complete_claimed=kwargs.pop("complete", True),
        **kwargs,
    )


def resolution(value):
    issue = assess(value)["issues"][0]
    return {
        "issue_id": issue["issue_id"],
        "abc_sha256": gate.sha(ABC.encode()),
        "category": "source_irregular_meter",
        "location": "page 1, row 1, bar 1",
        "adopted_choice": "Retain the visible irregular bar.",
        "reason": "Compared visible heads, flags and rests against the preceding bar.",
        "uncertainty": "Provisional source reading; musician review needed.",
        "inspection_call": 1,
    }


def test_complete_claim_needs_zero_unresolved_issues_but_partial_progress_is_valid():
    assert assess(normalized())["completion_ready"]
    assert assess(normalized(), complete=False)["status"] == "partial"
    assert not assess(normalized(duration=1344))["completion_ready"]
    assert not assess(normalized(route="unsupported"))["completion_ready"]
    assert not assess(normalized(valid=False))["completion_ready"]
    empty = normalized()
    empty["measures"] = []
    assert not assess(empty)["completion_ready"]
    warning = normalized()
    warning["warnings"] = ["Unsupported notation"]
    assert not assess(warning)["completion_ready"]


def test_corrected_draft_passes_without_inventing_resolution():
    bad = assess(normalized(duration=768))
    assert len(bad["unresolved_issues"]) == 1
    assert assess(normalized())["completion_ready"]


def test_source_exception_remains_explicit_and_needs_evidence():
    value = normalized(duration=1344)
    item = resolution(value)
    assert not assess(value, resolutions=[item])["completion_ready"]
    result = assess(value, resolutions=[item], inspections={1: {"sha256": "verified"}})
    assert result["status"] == "ready_with_source_exceptions"
    assert len(result["issues"]) == len(result["accepted_source_exceptions"]) == 1
    assert result["musical_accuracy"] == "not_measured_by_completion_gate"


@pytest.mark.parametrize(
    "field,value",
    [
        ("abc_sha256", "old"),
        ("issue_id", "old"),
        ("reason", " "),
        ("inspection_call", True),
        ("category", "unclear"),
        ("category", "unpaired_pickup"),
    ],
)
def test_invalid_or_stale_exception_blocks_completion(field, value):
    parsed = normalized(duration=768)
    item = {**resolution(parsed), field: value}
    result = assess(parsed, resolutions=[item], inspections={1: {"sha256": "verified"}})
    assert not result["completion_ready"]
    assert result["resolution_errors"]


def test_duplicate_unmatched_and_non_meter_exceptions_do_not_waive_errors():
    parsed = normalized(duration=1344)
    item = resolution(parsed)
    assert not assess(parsed, resolutions=[item, item], inspections={1: {}})["completion_ready"]
    assert not assess(normalized(), resolutions=[item], inspections={1: {}})["completion_ready"]
    value = normalized(route="unsupported")
    assert not assess(value, resolutions=[resolution(value)], inspections={1: {}})[
        "completion_ready"
    ]
    assert not assess(parsed, resolutions="waive everything")["completion_ready"]


def test_pickup_complement_and_overlay_use_parser_duration_not_sum_of_voices():
    value = normalized()
    value["measures"][0].update(duration_ticks=384, partial_reason="pickup")
    value["measures"].append(
        {
            "number": 2,
            "duration_ticks": 768,
            "expected_ticks": 1152,
            "meter_valid": True,
            "partial_reason": "pickup_complement",
        }
    )
    assert assess(value)["completion_ready"]
    overlay = normalized()
    overlay["measures"][0]["notes"] = [
        {"voice": "1", "duration_ticks": 1152},
        {"voice": "2", "duration_ticks": 1152},
    ]
    assert assess(overlay)["completion_ready"]
    pickup = normalized(duration=384, partial="pickup_candidate")
    assert not assess(pickup)["completion_ready"]
    item = {**resolution(pickup), "category": "unpaired_pickup"}
    assert assess(pickup, resolutions=[item], inspections={1: {}})["completion_ready"]


@pytest.fixture
def tools(tmp_path, monkeypatch):
    (tmp_path / "source.pdf").write_bytes(b"%PDF-test")
    legacy.write_json(
        tmp_path / "tool_config.json",
        {"source_pdf_sha256": gate.sha(b"%PDF-test"), "pages": 1, "node": "node"},
    )
    state = {"normalized": normalized(duration=1344)}

    def parse(*args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps(state["normalized"]))

    monkeypatch.setattr(source.legacy.subprocess, "run", parse)
    return source.SourceTools(tmp_path), state


def save(tools, **kwargs):
    result = tools.call("save_draft", {"abc": ABC, "decisions": [], "complete": True, **kwargs})
    return json.loads(result["content"][0]["text"])["completion_gate"]


def test_tool_preserves_blocked_then_corrected_checkpoints_and_legacy_schema(tools):
    tool, state = tools
    schema = copy.deepcopy(legacy.TOOLS)
    assert not save(tool)["completion_ready"]
    first = tool.artifacts / "drafts/draft_001"
    before = {p.name: p.read_bytes() for p in first.iterdir()}
    state["normalized"] = normalized()
    assert save(tool)["completion_ready"]
    assert {p.name: p.read_bytes() for p in first.iterdir()} == before
    assert legacy.TOOLS == schema
    assert "resolutions" not in legacy.TOOLS[1]["inputSchema"]["properties"]


def test_checkpoint_recomputes_readiness_and_rejects_changed_abc(tools):
    tool, _ = tools
    save(tool)
    directory = tool.artifacts / "drafts/draft_001"
    legacy.write_json(directory / "completion_gate.json", {"completion_ready": True})
    assert not gate.evaluate_checkpoint(directory)["completion_ready"]
    (directory / "score.abc").write_text(ABC + "C")
    with pytest.raises(ValueError, match="changed"):
        gate.evaluate_checkpoint(directory)


def test_real_inspection_pin_is_required_and_future_inspection_cannot_waive(tools):
    tool, state = tools
    pixels = b"synthetic crop evidence"
    path = tool.artifacts / "inspections/inspect_001.png"
    path.write_bytes(pixels)
    event = {
        "name": "inspect_source",
        "call": 1,
        "finished_at": "2000-01-01T00:00:00+00:00",
        "evidence": {"path": "inspections/inspect_001.png", "sha256": gate.sha(pixels)},
    }
    event_path = tool.artifacts / "tool_events.jsonl"
    legacy.write_json(event_path, event)
    # A source-tool event is one line, unlike ordinary pretty JSON.
    event_path.write_text(json.dumps(event) + "\n")
    tool.calls = 1
    assert save(tool, resolutions=[resolution(state["normalized"])])["completion_ready"]
    directory = tool.artifacts / "drafts/draft_001"
    path.write_bytes(b"changed")
    assert not gate.evaluate_checkpoint(directory)["completion_ready"]
    path.write_bytes(pixels)
    event["finished_at"] = "2999-01-01T00:00:00+00:00"
    event_path.write_text(json.dumps(event) + "\n")
    assert not gate.evaluate_checkpoint(directory)["completion_ready"]


def test_malformed_resolution_and_parser_failure_still_save_progress(tools):
    tool, state = tools
    assert not save(tool, resolutions=[{"reason": "unclear"}])["completion_ready"]
    state["normalized"] = {"valid": False, "errors": ["Parser timed out"], "measures": []}
    assert not save(tool)["completion_ready"]
    assert (tool.artifacts / "drafts/draft_002/score.abc").read_text() == ABC


@pytest.mark.parametrize("truncated", ["completion_resolutions.json", "tool_events.jsonl"])
def test_interrupted_evidence_write_blocks_without_losing_saved_draft(tools, truncated):
    tool, state = tools
    save(tool, resolutions=[resolution(state["normalized"])])
    directory = tool.artifacts / "drafts/draft_001"
    target = directory / truncated if truncated.endswith(".json") else tool.artifacts / truncated
    target.write_text('{"interrupted":')
    result = gate.evaluate_checkpoint(directory)
    assert not result["completion_ready"]
    assert result["resolution_errors"]
    assert (directory / "score.abc").read_text() == ABC
