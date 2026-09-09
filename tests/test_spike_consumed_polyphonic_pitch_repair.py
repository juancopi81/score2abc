from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from scripts.experiments import spike_consumed_polyphonic_pitch_repair as spike


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    inference = tmp_path / "inference.jsonl"
    truth = tmp_path / "truth.jsonl"
    evaluation = tmp_path / "report.json"
    model = tmp_path / "model.json"
    hints = tmp_path / "hints.json"
    source_image = tmp_path / "measure_001.png"
    image = Image.new("L", (60, 60), 255)
    draw = ImageDraw.Draw(image)
    for y in (10, 20, 30, 40, 50):
        draw.line((0, y, 59, y), fill=0, width=1)
    draw.line((14, 9, 14, 30), fill=0, width=2)
    image.save(source_image)
    source_sha256 = hashlib.sha256(source_image.read_bytes()).hexdigest()
    row = {
        "identity": {"automatic_measure_index": 1, "system_measure_index": 1},
        "staff_geometry": {"raw_staff_lines_y_px": [0, 10, 20, 30, 40]},
        "source": {"image": str(source_image), "sha256": source_sha256},
        "truth_used": False,
        "candidate_predictions": [
            {
                "candidate_id": "high",
                "center": {"x": 10, "y": 20},
                "score": 0.9,
                "detector_rank": 1,
                "bbox": {"left": 5, "top": 15, "right": 15, "bottom": 25},
            },
            {
                "candidate_id": "low",
                "center": {"x": 10, "y": 0},
                "score": 0.8,
                "detector_rank": 2,
                "bbox": {"left": 5, "top": 0, "right": 15, "bottom": 8},
            },
            {
                "candidate_id": "next",
                "center": {"x": 30, "y": 20},
                "score": 0.7,
                "detector_rank": 3,
                "bbox": {"left": 25, "top": 15, "right": 35, "bottom": 25},
            },
        ],
        "canonical_prediction": {
            "notes": [
                {"candidate_id": "high", "pitch_midi": 71},
                {"candidate_id": "next", "pitch_midi": 71},
            ]
        },
    }
    inference.write_text(json.dumps(row) + "\n", encoding="utf-8")
    truth.write_text(
        json.dumps(
            {
                "automatic_crop_index": 1,
                "notes": [
                    {"pitch_midi": 64, "pitch": "E4", "onset_divisions": 0, "xml_order": 0},
                    {"pitch_midi": 60, "pitch": "C4", "onset_divisions": 0, "xml_order": 1},
                    {"pitch_midi": 67, "pitch": "G4", "onset_divisions": 1, "xml_order": 2},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    evaluation.write_text(json.dumps({"kind": "consumed_evaluation"}) + "\n", encoding="utf-8")
    model.write_text(
        json.dumps(
            {
                "replay": {
                    "selector": {
                        "threshold": 0.5,
                        "nms_x_spaces": 1.0,
                        "minimum_selected_count": 0,
                        "maximum_selected_count": 5,
                    }
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    hints.write_text(
        json.dumps({"source": "human", "measures": {"1": {"key_hint": "one flat: Bb"}}}) + "\n",
        encoding="utf-8",
    )
    return {
        "inference": inference,
        "truth": truth,
        "evaluation": evaluation,
        "model": model,
        "hints": hints,
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_replay_uses_frozen_context_and_anchor_ids_when_canonical_ids_are_absent(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)
    row = json.loads(paths["inference"].read_text(encoding="utf-8"))
    row["allowed_context"] = {"key_hint": "one flat: Bb"}
    row["automatic_anchors"] = [
        {"source": {"candidate_id": "high"}},
        {"source": {"candidate_id": "next"}},
    ]
    row["canonical_prediction"]["notes"] = [{"pitch_midi": 70}, {"pitch_midi": 70}]
    selector = spike.selector_config_from_model(
        json.loads(paths["model"].read_text(encoding="utf-8"))
    )

    predictions = spike._materialize_predictions(
        [row],
        selector,
        [None],
        lane=spike.LANE_AUTOMATIC,
    )

    assert [note["pitch_midi"] for note in predictions["x_only"][1]["notes"]] == [70, 70]
    assert (
        spike._verify_x_only_replay_parity([row], predictions)["status"]
        == "exact_frozen_baseline_replay"
    )


def _feature_row(tmp_path: Path, *, expected_sha256: str | None = None) -> dict:
    image_path = tmp_path / "feature.png"
    image = Image.new("L", (80, 80), 255)
    draw = ImageDraw.Draw(image)
    for y in (20, 30, 40, 50, 60):
        draw.line((0, y, 79, y), fill=0, width=1)
    draw.rectangle((30, 43, 40, 53), outline=0, width=1)
    draw.line((39, 18, 39, 43), fill=0, width=2)
    image.save(image_path)
    return {
        "identity": {"automatic_measure_index": 1},
        "staff_geometry": {"raw_staff_lines_y_px": [20, 30, 40, 50, 60]},
        "source": {
            "image": str(image_path),
            "sha256": expected_sha256 or hashlib.sha256(image_path.read_bytes()).hexdigest(),
        },
        "candidate_predictions": [
            {
                "candidate_id": "head",
                "center": {"x": 35, "y": 48},
                "bbox": {"left": 30, "top": 43, "right": 40, "bottom": 53},
                "score": 0.8,
            }
        ],
    }


def test_staff_lines_alone_are_rejected_and_adjacent_stem_is_accepted(tmp_path: Path) -> None:
    row = _feature_row(tmp_path)
    features, metadata = spike.candidate_local_stem_features(row)

    assert metadata["source_image"]["sha256"] == row["source"]["sha256"]
    assert features["head"]["score"] >= 0.8

    image_path = tmp_path / "staff_only.png"
    image = Image.new("L", (80, 80), 255)
    draw = ImageDraw.Draw(image)
    for y in (20, 30, 40, 50, 60):
        draw.line((0, y, 79, y), fill=0, width=1)
    image.save(image_path)
    staff_only = dict(row)
    staff_only["source"] = {
        "image": str(image_path),
        "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
    }
    staff_features, _ = spike.candidate_local_stem_features(staff_only)
    assert staff_features["head"]["score"] == 0.0


def test_source_image_hash_drift_is_rejected_before_pixel_read(tmp_path: Path) -> None:
    row = _feature_row(tmp_path, expected_sha256="0" * 64)

    with pytest.raises(ValueError, match="Source image hash mismatch"):
        spike.candidate_local_stem_features(row)


def test_x_only_suppression_and_vertical_chord_retention() -> None:
    row = {
        "staff_geometry": {"raw_staff_lines_y_px": [0, 10, 20, 30, 40]},
        "candidate_predictions": [
            {"candidate_id": "top", "center": {"x": 10, "y": 20}, "score": 0.9},
            {"candidate_id": "bottom", "center": {"x": 10, "y": 0}, "score": 0.8},
            {"candidate_id": "next", "center": {"x": 30, "y": 20}, "score": 0.7},
        ],
    }
    selector = {
        "threshold": 0.0,
        "nms_x_spaces": 1.0,
        "minimum_selected_count": 0,
        "maximum_selected_count": 5,
    }

    x_only = spike.select_candidates(row, selector)
    two_dimensional = spike.select_candidates(row, selector, y_separation_staff_spaces=1.0)

    assert [candidate["candidate_id"] for candidate in x_only] == ["top", "next"]
    assert [candidate["candidate_id"] for candidate in two_dimensional] == [
        "top",
        "bottom",
        "next",
    ]


def test_two_dimensional_maximum_caps_groups_not_heads() -> None:
    row = {
        "staff_geometry": {"raw_staff_lines_y_px": [0, 10, 20, 30, 40]},
        "candidate_predictions": [
            {"candidate_id": "top", "center": {"x": 10, "y": 20}, "score": 0.9},
            {"candidate_id": "bottom", "center": {"x": 10, "y": 0}, "score": 0.8},
            {"candidate_id": "next", "center": {"x": 30, "y": 20}, "score": 0.7},
        ],
    }
    selector = {
        "threshold": 0.0,
        "nms_x_spaces": 1.0,
        "minimum_selected_count": 0,
        "maximum_selected_count": 1,
    }

    x_only = spike.select_candidates(row, selector)
    two_dimensional = spike.select_candidates(row, selector, y_separation_staff_spaces=1.0)

    assert [candidate["candidate_id"] for candidate in x_only] == ["top"]
    assert [candidate["candidate_id"] for candidate in two_dimensional] == ["top", "bottom"]


def test_chord_recovery_never_creates_a_new_onset_group() -> None:
    row = {
        "staff_geometry": {"raw_staff_lines_y_px": [0, 10, 20, 30, 40]},
        "candidate_predictions": [
            {
                "candidate_id": "anchor",
                "center": {"x": 10, "y": 20},
                "score": 0.9,
                "detector_rank": 1,
            },
            {
                "candidate_id": "new-group",
                "center": {"x": 25, "y": 0},
                "score": 0.8,
                "detector_rank": 2,
            },
        ],
    }
    selector = {
        "threshold": 0.5,
        "nms_x_spaces": 1.0,
        "minimum_selected_count": 0,
        "maximum_selected_count": 1,
    }
    baseline = spike.select_candidates(row, selector)

    recovered = spike.recover_chord_candidates(
        row,
        selector,
        baseline,
        minimum_y_gap_staff_spaces=1.0,
        maximum_y_gap_staff_spaces=3.0,
        minimum_score_ratio=0.5,
    )

    assert [candidate["candidate_id"] for candidate in baseline] == ["anchor"]
    assert recovered == []


def test_chord_recovery_adds_at_most_one_head_per_group() -> None:
    row = {
        "staff_geometry": {"raw_staff_lines_y_px": [0, 10, 20, 30, 40]},
        "candidate_predictions": [
            {
                "candidate_id": "anchor",
                "center": {"x": 10, "y": 20},
                "score": 0.9,
                "detector_rank": 1,
            },
            {
                "candidate_id": "first",
                "center": {"x": 10, "y": 0},
                "score": 0.8,
                "detector_rank": 2,
            },
            {
                "candidate_id": "second",
                "center": {"x": 11, "y": 40},
                "score": 0.7,
                "detector_rank": 3,
            },
        ],
    }
    selector = {
        "threshold": 0.5,
        "nms_x_spaces": 1.0,
        "minimum_selected_count": 0,
        "maximum_selected_count": 1,
    }
    baseline = spike.select_candidates(row, selector)

    recovered = spike.recover_chord_candidates(
        row,
        selector,
        baseline,
        minimum_y_gap_staff_spaces=1.0,
        maximum_y_gap_staff_spaces=3.0,
        minimum_score_ratio=0.5,
    )

    assert [candidate["candidate_id"] for candidate in recovered] == ["first"]


def test_chord_recovery_rejects_candidate_below_score_ratio() -> None:
    row = {
        "staff_geometry": {"raw_staff_lines_y_px": [0, 10, 20, 30, 40]},
        "candidate_predictions": [
            {
                "candidate_id": "anchor",
                "center": {"x": 10, "y": 20},
                "score": 0.9,
                "detector_rank": 1,
            },
            {
                "candidate_id": "weak",
                "center": {"x": 10, "y": 0},
                "score": 0.6,
                "detector_rank": 2,
            },
        ],
    }
    selector = {
        "threshold": 0.5,
        "nms_x_spaces": 1.0,
        "minimum_selected_count": 0,
        "maximum_selected_count": 1,
    }
    baseline = spike.select_candidates(row, selector)

    recovered = spike.recover_chord_candidates(
        row,
        selector,
        baseline,
        minimum_y_gap_staff_spaces=1.0,
        maximum_y_gap_staff_spaces=3.0,
        minimum_score_ratio=0.75,
    )

    assert recovered == []


def test_chord_recovery_choice_is_deterministic_by_score_rank_and_id() -> None:
    row = {
        "staff_geometry": {"raw_staff_lines_y_px": [0, 10, 20, 30, 40]},
        "candidate_predictions": [
            {
                "candidate_id": "anchor-left",
                "center": {"x": 10, "y": 20},
                "score": 0.95,
                "detector_rank": 1,
            },
            {
                "candidate_id": "anchor-right",
                "center": {"x": 40, "y": 20},
                "score": 0.9,
                "detector_rank": 2,
            },
            {
                "candidate_id": "rank-second",
                "center": {"x": 10, "y": 0},
                "score": 0.8,
                "detector_rank": 5,
            },
            {
                "candidate_id": "rank-first",
                "center": {"x": 11, "y": 0},
                "score": 0.8,
                "detector_rank": 4,
            },
            {
                "candidate_id": "id-z",
                "center": {"x": 40, "y": 0},
                "score": 0.75,
                "detector_rank": 6,
            },
            {
                "candidate_id": "id-a",
                "center": {"x": 41, "y": 0},
                "score": 0.75,
                "detector_rank": 6,
            },
        ],
    }
    selector = {
        "threshold": 0.5,
        "nms_x_spaces": 1.0,
        "minimum_selected_count": 0,
        "maximum_selected_count": 2,
    }
    baseline = spike.select_candidates(row, selector)

    recovered = spike.recover_chord_candidates(
        row,
        selector,
        baseline,
        minimum_y_gap_staff_spaces=1.0,
        maximum_y_gap_staff_spaces=3.0,
        minimum_score_ratio=0.5,
    )

    assert [candidate["candidate_id"] for candidate in recovered] == ["rank-first", "id-a"]


def test_stem_aware_recovery_keeps_groups_and_is_deterministic() -> None:
    row = {
        "staff_geometry": {"raw_staff_lines_y_px": [0, 10, 20, 30, 40]},
        "candidate_predictions": [
            {
                "candidate_id": "anchor",
                "center": {"x": 10, "y": 20},
                "score": 0.9,
                "detector_rank": 1,
            },
            {
                "candidate_id": "good",
                "center": {"x": 10, "y": 0},
                "score": 0.8,
                "detector_rank": 2,
            },
            {
                "candidate_id": "weak",
                "center": {"x": 11, "y": 40},
                "score": 0.79,
                "detector_rank": 3,
            },
        ],
    }
    selector = {
        "threshold": 0.5,
        "nms_x_spaces": 1.0,
        "minimum_selected_count": 0,
        "maximum_selected_count": 1,
    }
    baseline = spike.select_candidates(row, selector)
    features = {"good": {"score": 0.91}, "weak": {"score": 0.79}}
    first = spike.recover_stem_aware_chord_candidates(
        row,
        selector,
        baseline,
        minimum_y_gap_staff_spaces=1.0,
        maximum_y_gap_staff_spaces=3.0,
        minimum_score_ratio=0.5,
        minimum_stem_score=0.8,
        stem_features=features,
    )
    second = spike.recover_stem_aware_chord_candidates(
        row,
        selector,
        baseline,
        minimum_y_gap_staff_spaces=1.0,
        maximum_y_gap_staff_spaces=3.0,
        minimum_score_ratio=0.5,
        minimum_stem_score=0.8,
        stem_features=features,
    )

    assert [candidate["candidate_id"] for candidate in first] == ["good"]
    assert first == second
    assert spike._onset_group_count([*baseline, *first], 10.0) == 1


def test_edge_safe_stem_recovery_rejects_leading_ink_and_weak_stems() -> None:
    row = {
        "staff_geometry": {"raw_staff_lines_y_px": [0, 10, 20, 30, 40]},
        "candidate_predictions": [
            {"candidate_id": "leading", "center": {"x": 5, "y": 20}, "score": 0.95},
            {
                "candidate_id": "leading-extra",
                "center": {"x": 5, "y": 0},
                "score": 0.8,
            },
            {"candidate_id": "anchor", "center": {"x": 30, "y": 20}, "score": 0.9},
            {"candidate_id": "good", "center": {"x": 30, "y": 0}, "score": 0.8},
            {"candidate_id": "weak", "center": {"x": 31, "y": 40}, "score": 0.79},
        ],
    }
    selector = {
        "threshold": 0.5,
        "nms_x_spaces": 1.0,
        "minimum_selected_count": 0,
        "maximum_selected_count": 2,
    }
    baseline = spike.select_candidates(row, selector)

    recovered = spike.recover_edge_safe_stem_aware_chord_candidates(
        row,
        selector,
        baseline,
        minimum_y_gap_staff_spaces=1.0,
        maximum_y_gap_staff_spaces=3.0,
        minimum_score_ratio=0.5,
        minimum_stem_score=0.55,
        minimum_group_x_staff_spaces=1.0,
        stem_features={
            "leading-extra": {"score": 0.95},
            "good": {"score": 0.8},
            "weak": {"score": 0.4},
        },
    )

    assert [candidate["candidate_id"] for candidate in recovered] == ["good"]
    assert recovered[0]["leading_edge_distance_staff_spaces"] == 3.0
    assert spike._onset_group_count([*baseline, *recovered], 10.0) == 2


def test_edge_safe_multihead_recovery_builds_bounded_vertical_chain() -> None:
    row = {
        "staff_geometry": {"raw_staff_lines_y_px": [0, 10, 20, 30, 40]},
        "candidate_predictions": [
            {"candidate_id": "anchor", "center": {"x": 30, "y": 40}, "score": 0.9},
            {"candidate_id": "middle", "center": {"x": 30, "y": 20}, "score": 0.8},
            {"candidate_id": "high", "center": {"x": 30, "y": 0}, "score": 0.7},
            {"candidate_id": "overflow", "center": {"x": 30, "y": -20}, "score": 0.6},
            {"candidate_id": "new-onset", "center": {"x": 55, "y": 20}, "score": 0.85},
        ],
    }
    selector = {
        "threshold": 0.5,
        "nms_x_spaces": 1.0,
        "minimum_selected_count": 0,
        "maximum_selected_count": 1,
    }
    baseline = spike.select_candidates(row, selector)

    recovered = spike.recover_edge_safe_stem_aware_multihead_candidates(
        row,
        selector,
        baseline,
        minimum_y_gap_staff_spaces=1.0,
        maximum_y_gap_staff_spaces=3.0,
        minimum_score_ratio=0.5,
        minimum_stem_score=0.55,
        minimum_group_x_staff_spaces=1.0,
        maximum_recovered_heads_per_group=2,
        stem_features={
            "middle": {"score": 0.9},
            "high": {"score": 0.8},
            "overflow": {"score": 0.9},
            "new-onset": {"score": 0.9},
        },
    )

    assert [candidate["candidate_id"] for candidate in recovered] == ["middle", "high"]
    assert [candidate["recovery_group_index"] for candidate in recovered] == [1, 1]
    assert spike._onset_group_count([*baseline, *recovered], 10.0) == 1


def test_edge_safe_multihead_recovery_fails_closed_at_leading_edge() -> None:
    row = {
        "staff_geometry": {"raw_staff_lines_y_px": [0, 10, 20, 30, 40]},
        "candidate_predictions": [
            {"candidate_id": "anchor", "center": {"x": 5, "y": 40}, "score": 0.9},
            {"candidate_id": "companion", "center": {"x": 5, "y": 20}, "score": 0.8},
        ],
    }
    selector = {
        "threshold": 0.5,
        "nms_x_spaces": 1.0,
        "minimum_selected_count": 0,
        "maximum_selected_count": 1,
    }
    baseline = spike.select_candidates(row, selector)

    recovered = spike.recover_edge_safe_stem_aware_multihead_candidates(
        row,
        selector,
        baseline,
        minimum_y_gap_staff_spaces=1.0,
        maximum_y_gap_staff_spaces=3.0,
        minimum_score_ratio=0.5,
        minimum_stem_score=0.55,
        minimum_group_x_staff_spaces=1.0,
        maximum_recovered_heads_per_group=3,
        stem_features={"companion": {"score": 0.9}},
    )

    assert recovered == []


def test_unfiltered_recovery_behavior_is_unchanged_by_stem_family() -> None:
    row = {
        "staff_geometry": {"raw_staff_lines_y_px": [0, 10, 20, 30, 40]},
        "candidate_predictions": [
            {"candidate_id": "anchor", "center": {"x": 10, "y": 20}, "score": 0.9},
            {"candidate_id": "extra", "center": {"x": 10, "y": 0}, "score": 0.8},
        ],
    }
    selector = {
        "threshold": 0.5,
        "nms_x_spaces": 1.0,
        "minimum_selected_count": 0,
        "maximum_selected_count": 1,
    }
    baseline = spike.select_candidates(row, selector)
    first = spike.recover_chord_candidates(
        row,
        selector,
        baseline,
        minimum_y_gap_staff_spaces=1.0,
        maximum_y_gap_staff_spaces=3.0,
        minimum_score_ratio=0.5,
    )
    second = spike.recover_chord_candidates(
        row,
        selector,
        baseline,
        minimum_y_gap_staff_spaces=1.0,
        maximum_y_gap_staff_spaces=3.0,
        minimum_score_ratio=0.5,
    )
    assert [candidate["candidate_id"] for candidate in first] == ["extra"]
    assert first == second


def test_unordered_chord_scoring_preserves_group_order() -> None:
    predicted = [
        {"onset_divisions": 0, "pitch_midi": 60},
        {"onset_divisions": 0, "pitch_midi": 64},
        {"onset_divisions": 1, "pitch_midi": 67},
    ]
    truth = [
        {"onset_divisions": 0, "pitch_midi": 64, "xml_order": 0},
        {"onset_divisions": 0, "pitch_midi": 60, "xml_order": 1},
        {"onset_divisions": 1, "pitch_midi": 67, "xml_order": 2},
    ]

    result = spike.score_pitch_groups(predicted, truth)

    assert result["pitch_groups"]["truth_groups"] == [
        {"onset_divisions": 0, "pitch_set": [60, 64]},
        {"onset_divisions": 1, "pitch_set": [67]},
    ]
    assert result["pitch_groups"]["sequence_exact"] is True
    assert result["pitch_groups"]["alignment"]["edit_distance"] == 0
    assert result["legacy_ordered_pitch"]["alignment"]["edit_distance"] == 2


def test_run_keeps_inputs_immutable_refuses_overwrite_and_pins_lanes(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    before = {name: _sha(path) for name, path in paths.items()}
    output_dir = tmp_path / "postmortem_polyphonic_v1"

    report = spike.run_experiment(
        paths["inference"],
        paths["truth"],
        paths["evaluation"],
        paths["model"],
        output_dir=output_dir,
        context_hint_path=paths["hints"],
        y_separation_grid=(0.5, 1.0),
    )

    assert (output_dir / "report.json").is_file()
    assert (output_dir / "report.md").is_file()
    assert {name: _sha(path) for name, path in paths.items()} == before
    assert set(report["lanes"]) == {
        "automatic_natural_context",
        "context_hint",
        "diagnostic_oracle",
    }
    assert report["automatic_claims"]["source_lane"] == "automatic_natural_context"
    assert report["lanes"]["automatic_natural_context"]["context"]["key_hint"] is None
    assert report["lanes"]["context_hint"]["context"]["key_hint_source"] == (
        "human_or_externally_supplied_json"
    )
    assert report["lanes"]["context_hint"]["eligibility"]["automatic_claim"] is False
    assert (
        report["lanes"]["automatic_natural_context"]["eligibility"]["runtime_adoption_eligible"]
        is False
    )
    assert report["lanes"]["diagnostic_oracle"]["truth_access"]["truth_derived_context"] is True
    assert report["lanes"]["diagnostic_oracle"]["eligibility"]["runtime_adoption_eligible"] is False
    assert (
        report["provenance"]["access_audit"]["predictions_materialized_before_truth_read"] is True
    )
    assert report["baseline_parity"]["status"] == "exact_frozen_baseline_replay"
    assert set(report["provenance"]["inputs"]) == {
        "inference_jsonl",
        "truth_jsonl",
        "evaluation_report_json",
        "frozen_model_json",
        "context_hint_json",
    }
    assert report["lanes"]["automatic_natural_context"]["sweep"][0]["config_id"] == "x_only"
    two_dimensional = report["lanes"]["automatic_natural_context"]["sweep"][1]
    assert two_dimensional["per_measure"][0]["onset_group_count"] == 2
    assert two_dimensional["per_measure"][0]["total_head_count"] == 3
    assert two_dimensional["metrics"]["selection_count_metrics"]["predicted_onset_group_count"] == 2
    assert two_dimensional["metrics"]["selection_count_metrics"]["predicted_total_head_count"] == 3
    recovery_results = [
        result
        for result in report["lanes"]["automatic_natural_context"]["sweep"]
        if result["config_family"] == "chord_recovery"
    ]
    assert len(recovery_results) == 4
    for recovery in recovery_results:
        assert recovery["config_id"].startswith("chord_recovery__")
        assert recovery["per_measure"][0]["recovered_head_count"] == 1
        assert recovery["per_measure"][0]["onset_group_count"] == 2
        assert recovery["per_measure"][0]["baseline_onset_group_count"] == 2
        assert recovery["per_measure"][0]["onset_group_count_unchanged_from_x_only"] is True
        assert recovery["metrics"]["selection_count_metrics"]["recovered_head_count"] == 1
        assert (
            recovery["metrics"]["selection_count_metrics"][
                "onset_group_count_unchanged_from_x_only"
            ]
            is True
        )
    assert "winner" not in report["protocol"]

    with pytest.raises(FileExistsError):
        spike.run_experiment(
            paths["inference"],
            paths["truth"],
            paths["evaluation"],
            paths["model"],
            output_dir=output_dir,
            context_hint_path=paths["hints"],
        )


def test_context_events_carry_forward_until_next_change(tmp_path: Path) -> None:
    hints = tmp_path / "events.json"
    hints.write_text(
        json.dumps(
            {
                "source": "human_confirmed_visible_score_context",
                "events": [
                    {"start_measure": 2, "key_hint": "one sharp"},
                    {"start_measure": 5, "key_hint": "one flat"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    values, pin = spike._load_context_hints(hints, measure_indices=range(1, 8))

    assert values == {
        2: "one sharp",
        3: "one sharp",
        4: "one sharp",
        5: "one flat",
        6: "one flat",
        7: "one flat",
    }
    assert pin is not None
