import hashlib
import json
from pathlib import Path

import pytest

from scripts.experiments import spike_cross_score_consumed_retraining as spike


def test_selection_rule_uses_c_only_when_f1_and_pitch_do_not_regress() -> None:
    baseline = {
        "notehead_f1": 0.5,
        "conditional_pitch_accuracy_on_matched_noteheads": 0.75,
        "end_to_end_correct_pitch_recall": 0.4,
    }

    assert (
        spike.select_configuration(
            baseline,
            {
                "notehead_f1": 0.5 - 5e-10,
                "conditional_pitch_accuracy_on_matched_noteheads": 0.75,
                "end_to_end_correct_pitch_recall": 0.1,
            },
        )["selected_configuration"]
        == "C"
    )
    assert (
        spike.select_configuration(
            baseline,
            {
                "notehead_f1": 0.51,
                "conditional_pitch_accuracy_on_matched_noteheads": 0.74,
                "end_to_end_correct_pitch_recall": 0.9,
            },
        )["selected_configuration"]
        == "B"
    )
    assert (
        spike.select_configuration(
            baseline,
            {
                "notehead_f1": 0.49,
                "conditional_pitch_accuracy_on_matched_noteheads": 0.80,
                "end_to_end_correct_pitch_recall": 0.8,
            },
        )["selected_configuration"]
        == "B"
    )

    selected = spike.select_configuration(
        baseline,
        {
            "notehead_f1": 0.51,
            "conditional_pitch_accuracy_on_matched_noteheads": 0.75,
            "end_to_end_correct_pitch_recall": 0.0,
        },
    )
    assert selected["selected_configuration"] == "C"
    assert selected["c_conditional_pitch_on_matched_noteheads_non_regression"] is True
    assert "not part of this preregistered selection rule" in selected["rule"]


def test_macro_metrics_separate_conditional_pitch_from_end_to_end_recall() -> None:
    folds = [
        _fold_metrics(pitch_correct=3, matched_noteheads=4, truth_noteheads=6),
        _fold_metrics(pitch_correct=1, matched_noteheads=2, truth_noteheads=4),
    ]

    metrics = spike._macro_score_metrics(folds)

    assert metrics["conditional_pitch_accuracy_on_matched_noteheads"] == pytest.approx(0.625)
    assert metrics["end_to_end_correct_pitch_recall"] == pytest.approx(0.375)
    assert metrics["conditional_pitch_accuracy_on_matched_noteheads_micro"] == pytest.approx(4 / 6)
    assert metrics["end_to_end_correct_pitch_recall_micro"] == pytest.approx(4 / 10)


def test_medium_heads_are_excluded_from_b_and_included_in_c() -> None:
    heads = [
        {"order": 1, "confidence": "high"},
        {"order": 2, "confidence": "medium"},
        {"order": 3, "confidence": "high"},
    ]

    assert [row["order"] for row in spike._select_review_heads(heads, frozenset({"high"}))] == [
        1,
        3,
    ]
    assert [
        row["order"] for row in spike._select_review_heads(heads, frozenset({"high", "medium"}))
    ] == [1, 2, 3]


def test_score_fold_rejects_training_evaluation_overlap() -> None:
    spike._assert_score_disjoint({"aviador"}, {"carrizal"})

    with pytest.raises(ValueError, match="Score leakage"):
        spike._assert_score_disjoint({"aviador", "carrizal"}, {"carrizal"})
    with pytest.raises(ValueError, match="exactly one score"):
        spike._assert_score_disjoint({"aviador"}, {"carrizal", "third"})


def test_bounded_training_search_is_deterministic_and_budgeted(monkeypatch) -> None:
    scores = {(f"measure-{index}", "candidate"): index / 1000 for index in range(1000)}
    thresholds = spike._threshold_candidates(scores)

    assert thresholds == spike._threshold_candidates(scores)
    assert len(thresholds) <= spike.LOCAL_THRESHOLD_BUDGET
    assert thresholds[0] > max(scores.values())

    calls = []

    def fake_threshold_selection(examples, score_map, **kwargs):
        calls.append(kwargs)
        return 0.5, {
            "f1": 0.5,
            "recall": 0.5,
            "precision": 0.5,
            "exact_measures": 0,
            "selected_count": 0,
        }

    monkeypatch.setattr(spike, "_select_bounded_training_threshold", fake_threshold_selection)
    selection = spike._select_bounded_training_configuration([], scores)

    assert len(calls) == spike.LOCAL_CONFIGURATION_COUNT
    assert selection["search_budget"] == {
        "threshold_budget": spike.LOCAL_THRESHOLD_BUDGET,
        "threshold_candidate_count": len(thresholds),
        "threshold_strategy": (
            "sorted unique score values sampled by evenly spaced rank indices, "
            "plus max_score+1e-9 sentinel"
        ),
        "nms_x_spaces_grid": list(spike.LOCAL_NMS_X_SPACES_GRID),
        "minimum_selected_count_grid": list(spike.LOCAL_MIN_SELECTED_COUNT_GRID),
        "maximum_selected_count_grid": list(spike.LOCAL_MAX_SELECTED_COUNT_GRID),
        "configuration_count": spike.LOCAL_CONFIGURATION_COUNT,
        "max_search_evaluations": spike.LOCAL_CONFIGURATION_COUNT * len(thresholds),
        "objective": "training-review out-of-fold notehead F1 only",
    }


def test_review_provenance_record_is_hash_checked(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(spike, "REPO_ROOT", tmp_path)
    source = tmp_path / "review.json"
    source.write_text('{"review": true}\n', encoding="utf-8")
    record = {"path": "review.json", "sha256": _sha256(source)}

    assert spike._validate_file_record(record, label="review") == record
    source.write_text('{"review": false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        spike._validate_file_record(record, label="review")


def test_protected_third_score_truth_paths_are_rejected() -> None:
    with pytest.raises(ValueError, match="Protected third-score truth"):
        spike._reject_protected_truth_paths(
            (Path(f"dataset/musicxml/{spike.LA_CHATA_SLUG}.musicxml"),)
        )
    with pytest.raises(ValueError, match="Protected third-score truth"):
        spike._reject_protected_truth_paths((Path("dataset/ground_truth/score.json"),))


def test_create_once_outputs_are_deterministic_and_refuse_overwrite(tmp_path: Path) -> None:
    report = {
        "configurations": {
            key: {
                "macro": {
                    "notehead_precision": 0.5,
                    "notehead_recall": 0.5,
                    "notehead_f1": 0.5,
                    "conditional_pitch_accuracy_on_matched_noteheads": 0.5,
                    "end_to_end_correct_pitch_recall": 0.25,
                    "proposal_recall": 0.5,
                    "coordinate_measure_exactness": 0.25,
                }
            }
            for key in ("A", "B", "C")
        },
        "selection": {
            "selected_configuration": "B",
            "rule": "fixture rule",
        },
    }
    selection = {"selected": "B"}
    model = {"weights": [0.1, 0.2]}
    first = tmp_path / "first" / spike.OUTPUT_VERSION
    second = tmp_path / "second" / spike.OUTPUT_VERSION

    spike._write_create_once_artifacts(
        first,
        report=report,
        training_selection=selection,
        model=model,
    )
    spike._write_create_once_artifacts(
        second,
        report=report,
        training_selection=selection,
        model=model,
    )

    for name in ("report.json", "report.md", "training_selection.json", "model.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    first_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    for name, record in first_manifest["artifacts"].items():
        assert record["sha256"] == _sha256(first / name)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        spike._write_create_once_artifacts(
            first,
            report=report,
            training_selection=selection,
            model=model,
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fold_metrics(
    *, pitch_correct: int, matched_noteheads: int, truth_noteheads: int
) -> dict[str, object]:
    return {
        "coordinate_noteheads": {
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
        },
        "conditional_pitch_on_matched_noteheads": {
            "correct": pitch_correct,
            "matched_noteheads": matched_noteheads,
            "accuracy": pitch_correct / matched_noteheads,
        },
        "end_to_end_correct_pitch": {
            "correct": pitch_correct,
            "truth_noteheads": truth_noteheads,
            "recall": pitch_correct / truth_noteheads,
        },
        "proposal_recall": {"recall": 0.5},
        "measure_exactness": {
            "coordinate_rate": 0.25,
            "coordinate_and_pitch_rate": 0.125,
        },
    }
