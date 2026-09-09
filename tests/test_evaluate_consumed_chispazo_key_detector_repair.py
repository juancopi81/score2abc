from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.experiments import evaluate_consumed_chispazo_key_detector_repair as repair


def test_detector_event_requires_pinned_repaired_chispazo_hit(tmp_path: Path) -> None:
    work = tmp_path / repair.CASE.gate_case.slug / "systems"
    work.mkdir(parents=True)
    image = work / "system_003.png"
    image.write_bytes(b"score")
    source = {
        "path": str(image),
        "sha256": hashlib.sha256(b"score").hexdigest(),
    }
    hit = {
        "fifths": -2,
        "selection_method": repair.EXPECTED_METHOD,
        "selected_glyph_ids": ["g007", "g012"],
        "structural_boundary": {"x_px": repair.EXPECTED_BOUNDARY_X_PX},
    }
    report = {
        "truth_used_for_prediction": False,
        "systems": [{"scan": {"input": source, "hits": [hit]}}],
    }

    assert repair._detector_event(report)["fifths"] == -2

    hit["selection_method"] = "unvalidated"
    with pytest.raises(ValueError, match="does not match"):
        repair._detector_event(report)


def test_replay_match_checks_full_lane_content() -> None:
    row = {
        "identity": {"automatic_measure_index": 1},
        "lanes": {
            "global_no_key": {"notes": [{"candidate_id": "c001", "pitch": "B4"}]},
            "global_automatic_key": {"notes": [{"candidate_id": "c001", "pitch": "Bb4"}]},
        },
    }

    assert repair._replay_matches_frozen_diagnostic([row], [row]) is True
    changed = {
        **row,
        "lanes": {
            **row["lanes"],
            "global_automatic_key": {"notes": [{"candidate_id": "c001", "pitch": "B4"}]},
        },
    }
    assert repair._replay_matches_frozen_diagnostic([changed], [row]) is False
