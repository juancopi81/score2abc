from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.experiments import evaluate_consumed_key_detector_repair as repair


def test_detector_fifths_requires_pinned_truth_free_prediction(tmp_path: Path) -> None:
    image = tmp_path / "system.png"
    image.write_bytes(b"score")
    report = {
        "truth_used_for_prediction": False,
        "predictions": [
            {
                "slug": "work",
                "fifths": 2,
                "input": {
                    "path": str(image),
                    "sha256": hashlib.sha256(b"score").hexdigest(),
                },
            }
        ],
    }

    assert repair._detector_fifths(report, expected_slug="work") == 2

    report["predictions"][0]["input"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="missing or changed"):
        repair._detector_fifths(report, expected_slug="work")


def test_selection_match_checks_ids_and_coordinates() -> None:
    row = {
        "identity": {"automatic_measure_index": 1},
        "lanes": {
            "global_no_key": {"notes": [{"candidate_id": "c001", "center": {"x": 10.0, "y": 20.0}}]}
        },
    }

    assert repair._selection_matches_frozen([row], [row]) is True
    changed = {
        **row,
        "lanes": {
            "global_no_key": {"notes": [{"candidate_id": "c001", "center": {"x": 11.0, "y": 20.0}}]}
        },
    }
    assert repair._selection_matches_frozen([changed], [row]) is False
