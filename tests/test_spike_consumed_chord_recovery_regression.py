from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from scripts.experiments import spike_consumed_chord_recovery_regression as audit


def _candidate(candidate_id: str, x: float, y: float, score: float, rank: int) -> dict:
    return {
        "candidate_id": candidate_id,
        "center": {"x": x, "y": y},
        "score": score,
        "detector_rank": rank,
    }


def _row() -> dict:
    return {
        "identity": {"automatic_measure_index": 1},
        "staff_geometry": {"raw_staff_lines_y_px": [0, 10, 20, 30, 40]},
        "candidate_predictions": [
            _candidate("a", 10, 20, 0.9, 1),
            _candidate("b", 10, 35, 0.6, 2),
            _candidate("c", 40, 20, 0.8, 3),
        ],
        "truth_used": False,
    }


def _selector() -> dict:
    return {
        "threshold": 0.1,
        "nms_x_spaces": 0.85,
        "minimum_selected_count": 0,
        "maximum_selected_count": 5,
    }


def test_aggregation_is_order_independent_and_reports_counts() -> None:
    rows = [
        audit.candidate_metrics(["a"], ["a", "b"]),
        audit.candidate_metrics(["c", "d"], ["c"], recovered_ids=["d"]),
    ]

    aggregate = audit.aggregate_candidate_metrics(rows)
    reversed_aggregate = audit.aggregate_candidate_metrics(list(reversed(rows)))

    assert aggregate == reversed_aggregate
    assert aggregate["tp"] == 2
    assert aggregate["fp"] == 1
    assert aggregate["fn"] == 1
    assert aggregate["recovered_count_total"] == 1
    assert aggregate["recovered_false_positive_count"] == 1
    assert aggregate["exact_set_rate"] == 0.0


def test_recovery_keeps_groups_and_exposes_false_positive() -> None:
    replayed = audit.replay_inference_row(_row(), _selector())

    baseline = replayed["x_only"]
    assert audit._candidate_ids(baseline["selected"]) == ["a", "c"]
    assert baseline["onset_group_count"] == 2

    permissive = replayed["chord_recovery__min_y_1__max_y_3__score_ratio_0.5"]
    assert audit._candidate_ids(permissive["recovered"]) == ["b"]
    assert len(permissive["selected"]) == 3
    assert permissive["onset_group_count"] == permissive["baseline_onset_group_count"] == 2
    assert permissive["recovery_invariants"] == {
        "onset_group_count_unchanged": True,
        "no_new_onset_groups": True,
        "at_most_one_recovered_head_per_group": True,
    }

    metrics = audit.candidate_metrics(
        audit._candidate_ids(permissive["selected"]), {"a", "c"}, recovered_ids=["b"]
    )
    assert metrics["recovered_false_positive_ids"] == ["b"]
    assert metrics["recovered_true_positive_count"] == 0


def test_score_ratio_rejects_lower_scored_recovery_deterministically() -> None:
    replayed = audit.replay_inference_row(_row(), _selector())

    assert replayed["chord_recovery__min_y_1__max_y_3__score_ratio_0.75"]["recovered"] == []
    assert audit._candidate_ids(
        replayed["chord_recovery__min_y_1__max_y_3__score_ratio_0.5"]["recovered"]
    ) == ["b"]


def test_stem_grid_is_source_pinned_and_preserves_recovery_invariants(tmp_path: Path) -> None:
    image_path = tmp_path / "measure.png"
    image = Image.new("L", (70, 70), 255)
    draw = ImageDraw.Draw(image)
    for y in (10, 20, 30, 40, 50):
        draw.line((0, y, 69, y), fill=0, width=1)
    draw.line((19, 8, 19, 30), fill=0, width=2)
    image.save(image_path)
    row = _row()
    row["source"] = {
        "image": str(image_path),
        "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
    }
    for candidate in row["candidate_predictions"]:
        candidate["bbox"] = {
            "left": int(candidate["center"]["x"] - 5),
            "top": int(candidate["center"]["y"] - 5),
            "right": int(candidate["center"]["x"] + 5),
            "bottom": int(candidate["center"]["y"] + 5),
        }

    replayed = audit.replay_inference_row(row, _selector())
    stem_ids = [config["config_id"] for config in audit.targeted_stem_recovery_configs()]

    assert list(replayed) == [
        "x_only",
        *[config["config_id"] for config in audit.targeted_recovery_configs()],
        *stem_ids,
    ]
    for config_id in stem_ids:
        result = replayed[config_id]
        assert result["recovery_invariants"] == {
            "onset_group_count_unchanged": True,
            "no_new_onset_groups": True,
            "at_most_one_recovered_head_per_group": True,
        }
        assert result["stem_feature_metadata"]["source_image"]["sha256"] == row["source"]["sha256"]
        assert set(result["stem_feature_diagnostics"]) == {"a", "b", "c"}


def test_create_once_report_contains_provenance_and_refuses_overwrite(tmp_path) -> None:
    example = audit.candidate_metrics(["a"], ["a"])
    config = {
        "config_id": "x_only",
        "config_family": "x_only_baseline",
        "parameters": None,
        "overall": audit.aggregate_candidate_metrics([example]),
        "by_source_system": {},
        "examples": [],
    }
    output_dir = tmp_path / "audit"
    report = audit.build_report(
        selector=_selector(),
        model_audit={"truth_used": False, "selection_mode": "dense_threshold"},
        configurations={"x_only": config},
        input_pins={"model": {"path": "model.json", "sha256": "abc", "bytes": 3}},
        output_dir=output_dir,
        example_count=1,
    )

    audit.write_create_once(output_dir, report)
    assert (output_dir / "report.json").is_file()
    assert (output_dir / "report.md").is_file()
    loaded = json.loads((output_dir / "report.json").read_text())
    assert loaded["eligibility"] == {
        "accuracy_claim": False,
        "held_out": False,
        "in_sample_consumed_evidence": True,
        "runtime_adoption_eligible": False,
        "winner_selected_from_truth": False,
    }
    assert loaded["provenance"]["existing_evidence_overwritten"] is False
    assert "Configuration selection: `none`" in (output_dir / "report.md").read_text()

    with pytest.raises(FileExistsError, match="create-once"):
        audit.write_create_once(output_dir, report)
