from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from scripts import eval_vlm_notehead_proposal_baselines as evaluator


def _candidate(candidate_id: str, x: float, y: float) -> dict[str, object]:
    return {"id": candidate_id, "center": {"x": x, "y": y}}


def _gt(gt_id: str, x: float, y: float) -> dict[str, object]:
    return {
        "id": gt_id,
        "pitch": "F5",
        "center": {"x": x, "y": y},
        "annotation_geometry": {"radius_x_px": 10.0, "radius_y_px": 10.0},
    }


def test_match_points_is_deterministic_and_one_to_one() -> None:
    candidates = [_candidate("c001", 10, 10), _candidate("c002", 12, 10)]
    ground_truth = [_gt("n001", 11, 10)]

    first = evaluator.match_points(
        candidates,
        ground_truth,
        eligibility=lambda dx, dy: abs(dx) <= 2 and abs(dy) <= 2,
        tolerance_description="test",
    )
    second = evaluator.match_points(
        candidates,
        ground_truth,
        eligibility=lambda dx, dy: abs(dx) <= 2 and abs(dy) <= 2,
        tolerance_description="test",
    )

    assert first == second
    assert first["tp"] == 1
    assert first["assignments"][0]["candidate_id"] == "c001"
    assert first["unmatched_candidate_ids"] == ["c002"]


def test_pitch_safe_y_tolerance_rejects_candidate_legacy_accepts() -> None:
    candidates = [_candidate("c001", 100, 105)]
    ground_truth = [_gt("n001", 100, 100)]

    legacy = evaluator.match_points(
        candidates,
        ground_truth,
        eligibility=lambda dx, dy: (dx * dx + dy * dy) ** 0.5 <= 13,
        tolerance_description="legacy",
    )
    pitch_safe = evaluator.match_points(
        candidates,
        ground_truth,
        eligibility=lambda dx, dy: abs(dx) <= 7.5 and abs(dy) <= 2.5,
        tolerance_description="pitch-safe",
    )

    assert legacy["tp"] == 1
    assert pitch_safe["tp"] == 0
    assert pitch_safe["unmatched_ground_truth_ids"] == ["n001"]


def test_annotation_region_matching_and_natural_pitch_accuracy_are_separate() -> None:
    ground_truth = [_gt("n001", 100, 80), _gt("n002", 140, 100)]
    ground_truth[0]["pitch"] = "F#5"
    ground_truth[1]["pitch"] = "E4"
    candidates = [_candidate("c001", 108, 80), _candidate("c002", 140, 108)]

    result = evaluator.match_region_points(
        candidates,
        ground_truth,
        staff_lines=[80, 90, 100, 110, 120],
    )

    assert result["tp"] == 2
    assert result["top_k_region_recall"] == 1.0
    assert result["pitch_correct_count"] == 1
    assert result["pitch_accuracy"] == 0.5
    assert result["assignments"][0]["ground_truth_natural_pitch"] == "F5"
    assert result["assignments"][0]["pitch_correct"] is True
    assert result["assignments"][1]["pitch_correct"] is False


def test_annotation_region_margin_is_documented_and_deterministic() -> None:
    ground_truth = [_gt("n001", 10, 10)]
    candidates = [_candidate("c001", 21.4, 10)]

    result = evaluator.match_region_points(
        candidates,
        ground_truth,
        staff_lines=[0, 10, 20, 30, 40],
        margin=1.15,
    )

    assert result["ellipse_margin"] == 1.15
    assert result["tp"] == 1


def test_aggregate_calculations_are_micro_metrics() -> None:
    measures = [
        {
            "gt_count": 2,
            "candidate_count": 3,
            "caps": [
                {
                    "cap": 4,
                    "candidate_count": 3,
                    "gt_count": 2,
                    "annotation_region": {
                        "tp": 1,
                        "fp": 2,
                        "fn": 1,
                        "pitch_correct_count": 1,
                        "exact_coverage": False,
                    },
                    "legacy_euclidean": {
                        "tp": 1,
                        "fp": 2,
                        "fn": 1,
                        "precision": 0,
                        "recall": 0,
                        "f1": 0,
                        "exact_coverage": False,
                    },
                    "pitch_safe": {
                        "tp": 1,
                        "fp": 2,
                        "fn": 1,
                        "precision": 0,
                        "recall": 0,
                        "f1": 0,
                        "exact_coverage": False,
                    },
                }
            ],
        },
        {
            "gt_count": 1,
            "candidate_count": 2,
            "caps": [
                {
                    "cap": 4,
                    "candidate_count": 2,
                    "gt_count": 1,
                    "annotation_region": {
                        "tp": 1,
                        "fp": 1,
                        "fn": 0,
                        "pitch_correct_count": 1,
                        "exact_coverage": True,
                    },
                    "legacy_euclidean": {
                        "tp": 1,
                        "fp": 1,
                        "fn": 0,
                        "precision": 0,
                        "recall": 0,
                        "f1": 0,
                        "exact_coverage": True,
                    },
                    "pitch_safe": {
                        "tp": 0,
                        "fp": 2,
                        "fn": 1,
                        "precision": 0,
                        "recall": 0,
                        "f1": 0,
                        "exact_coverage": False,
                    },
                }
            ],
        },
    ]

    report = evaluator._aggregate_by_cap(measures, [4])

    assert report[0]["legacy_euclidean"] == {
        "candidate_count": 5,
        "gt_count": 3,
        "tp": 2,
        "fp": 3,
        "fn": 1,
        "precision": 0.4,
        "recall": evaluator._ratio(2, 3),
        "f1": evaluator._f1(2, 3, 1),
        "exact_coverage": False,
        "exact_coverage_measure_count": 1,
        "measure_count": 2,
    }
    assert report[0]["pitch_safe"]["tp"] == 1
    assert report[0]["annotation_region"]["tp"] == 2
    assert report[0]["annotation_region"]["pitch_accuracy"] == 1.0


def test_generation_happens_before_ground_truth_load(monkeypatch, tmp_path: Path) -> None:
    source_path = tmp_path / "measure.png"
    Image.new("RGB", (20, 20), "white").save(source_path)
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps({"noteheads": [_gt("n001", 5, 5)]}), encoding="utf-8")
    events: list[str] = []

    monkeypatch.setattr(
        evaluator.detector,
        "_load_or_build_base_record",
        lambda *args, **kwargs: {"paths": {"context": "context.json"}, "global_measure_index": 0},
    )
    monkeypatch.setattr(evaluator.detector, "_resolve_path", lambda out, value: tmp_path / value)
    (tmp_path / "context.json").write_text(
        json.dumps({"paths": {}, "staff_lines_y_px_in_system": [2, 4, 6, 8, 10]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluator.detector, "_source_path", lambda *args, **kwargs: source_path)
    monkeypatch.setattr(
        evaluator.detector, "_staff_lines_for_source", lambda *args: [2, 4, 6, 8, 10]
    )
    monkeypatch.setattr(evaluator.detector, "_staff_spacing", lambda lines: 2.0)

    def detect(*args, **kwargs):
        events.append(f"detect:{kwargs['max_candidates']}")
        return [_candidate("c001", 5, 5)]

    def load_gt(path: Path):
        events.append("gt")
        return evaluator._load_json(gt_path)["noteheads"]

    monkeypatch.setattr(evaluator.detector, "detect_staff_grid_density_candidates", detect)
    monkeypatch.setattr(evaluator, "_ground_truth_path", lambda *args, **kwargs: gt_path)
    monkeypatch.setattr(evaluator, "_load_ground_truth_fixture", load_gt)

    evaluator.evaluate_proposals(tmp_path, slug="demo", system_index=1, measures=[1], caps=[4, 8])

    assert events == ["detect:4", "detect:8", "gt"]


def test_cli_writes_json_and_markdown_with_repeatable_options(monkeypatch, tmp_path: Path) -> None:
    output_path = tmp_path / "reports" / "baseline.json"
    fake_report = {
        "slug": "demo",
        "system_index": 1,
        "selection": {
            "best_deterministic_cap_by_f1": 4,
            "best_deterministic_f1": 1.0,
            "smallest_cap_achieving_max_annotation_region_recall": 4,
            "max_annotation_region_recall": 1.0,
            "best_deterministic_cap_by_pitch_safe_f1": 4,
            "best_deterministic_pitch_safe_f1": 1.0,
            "smallest_cap_achieving_max_pitch_safe_recall": 4,
            "max_pitch_safe_recall": 1.0,
        },
        "aggregate_by_cap": [],
        "per_measure": [],
    }
    calls: list[object] = []

    def fake_evaluate(*args, **kwargs):
        calls.append(kwargs)
        return fake_report

    monkeypatch.setattr(evaluator, "evaluate_proposals", fake_evaluate)

    assert (
        evaluator.main(
            [
                str(tmp_path),
                "--slug",
                "demo",
                "--system",
                "1",
                "--measure",
                "1",
                "--measure",
                "2",
                "--cap",
                "4",
                "--cap",
                "8",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    assert output_path.exists()
    assert output_path.with_suffix(".md").exists()
    assert calls == [{"slug": "demo", "system_index": 1, "measures": [1, 2], "caps": [4, 8]}]
