"""Replay integrity and parser drift controls without live inference."""

import copy
import json

import pytest

from scripts import replay_completion_gate as replay


@pytest.fixture
def stage(tmp_path, monkeypatch):
    root = tmp_path / "stage"
    draft = root / "artifacts/drafts/draft_001"
    draft.mkdir(parents=True)
    abc, decisions = b"X:1\nM:3/4\nK:C\nC6|]", b"[]"
    (draft / "score.abc").write_bytes(abc)
    (draft / "decisions.json").write_bytes(decisions)
    parsed = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "route_status": "resolved",
        "measures": [
            {"number": 1, "duration_ticks": 1152, "expected_ticks": 1152, "meter_valid": True}
        ],
    }
    replay.shared.write(draft / "normalized.json", parsed)
    replay.shared.write(
        draft / "checkpoint.json",
        {
            "abc_sha256": replay.gate.sha(abc),
            "decisions_sha256": replay.gate.sha(decisions),
            "created_at": "2026-10-29T00:00:00+00:00",
            "complete_claimed": True,
        },
    )
    replay.shared.write(root / "stage_provenance.json", {"status": "complete", "model": "test"})
    replay.shared.seal(root, "astra_reproduction_stage")
    monkeypatch.setattr(replay, "normalize_abc", lambda abc: copy.deepcopy(parsed))
    return root, parsed


def test_replay_preserves_sealed_sources_and_retains_each_checkpoint(stage):
    root, _ = stage
    before = replay.shared.pins(root)
    report = replay.replay_stage(root)
    assert report["sealed_inputs_unchanged"]
    assert report["final"]["completion_ready"]
    assert report["final"]["current_parser_issues_identical"]
    assert len(report["checkpoints"]) == 1
    assert replay.shared.pins(root) == before


def test_replay_refuses_changed_source_or_parser(stage, monkeypatch):
    root, parsed = stage
    changed = copy.deepcopy(parsed)
    changed["measures"][0]["meter_valid"] = False
    monkeypatch.setattr(replay, "normalize_abc", lambda abc: changed)
    with pytest.raises(ValueError, match="parser changes"):
        replay.replay_stage(root)
    path = root / "artifacts/drafts/draft_001/normalized.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="artifact changed"):
        replay.replay_stage(root)


def test_dashboard_escapes_labels_and_derives_summary_counts(stage):
    root, _ = stage
    item = {**replay.replay_stage(root), "piece": "<script>", "backend": "Test", "stage": "initial"}
    page = replay.render(
        {"stages": [item], "summary": {"stages": 1, "drafts": 1, "finals_ready": 1}}
    )
    assert "&lt;script&gt;" in page and "<script>" not in page
    assert "<strong>1 / 1</strong> final drafts ready" in page
    assert "<strong>0 / 1</strong> finals blocked" in page
