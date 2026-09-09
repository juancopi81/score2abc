import json
from pathlib import Path

import pytest

from scripts.experiments.spike_notehead_sequence_bootstrap import (
    EVALUATION_SPLITS,
    SPLITS,
    CrossSystemTruthGuard,
    OnsetGroup,
    PseudoAcceptanceRule,
    TruthAccessGuard,
    align_onset_groups,
    apply_pseudo_acceptance_rule,
    bootstrap_gate,
    calibrate_pseudo_acceptance_rule,
    evaluate_validation_gate,
    partition_cross_system_validation_rows,
)


def _candidate(
    candidate_id: str,
    *,
    x: float,
    y: float,
    pitch: int,
    score: float = 0.8,
) -> dict:
    return {
        "id": candidate_id,
        "rank": int(candidate_id.removeprefix("c")),
        "center": {"x": x, "y": y},
        "score": score,
        "auto_pitch": "C4",
        "auto_pitch_midi": pitch,
    }


def test_monotonic_alignment_skips_distractors_without_reordering_notes() -> None:
    groups = [
        OnsetGroup(0.0, (60,)),
        OnsetGroup(1.0, (62,)),
        OnsetGroup(2.0, (64,)),
    ]
    candidates = [
        _candidate("c001", x=10, y=20, pitch=60),
        _candidate("c002", x=20, y=10, pitch=67, score=0.99),
        _candidate("c003", x=30, y=18, pitch=62),
        _candidate("c004", x=40, y=16, pitch=64),
    ]

    result = align_onset_groups(
        groups,
        candidates,
        image_width=50,
        staff_spacing=10,
    )

    assert result["aligned_group_count"] == 3
    assert [row["candidate_ids"] for row in result["assignments"]] == [
        ["c001"],
        ["c003"],
        ["c004"],
    ]


def test_monotonic_alignment_supports_same_x_dyads() -> None:
    groups = [OnsetGroup(0.0, (60, 64)), OnsetGroup(1.0, (67,))]
    candidates = [
        _candidate("c001", x=10.0, y=24, pitch=60),
        _candidate("c002", x=11.5, y=16, pitch=64),
        _candidate("c003", x=12.0, y=10, pitch=72, score=0.95),
        _candidate("c004", x=35.0, y=12, pitch=67),
    ]

    result = align_onset_groups(
        groups,
        candidates,
        image_width=45,
        staff_spacing=10,
    )

    assert result["assignments"][0]["candidate_ids"] == ["c001", "c002"]
    assert [pair["canonical_pitch_midi"] for pair in result["assignments"][0]["pairs"]] == [
        60,
        64,
    ]
    assert result["assignments"][1]["candidate_ids"] == ["c004"]


@pytest.mark.parametrize(
    ("precision", "recall", "passed"),
    [
        (0.85, 0.85, True),
        (0.849999, 1.0, False),
        (1.0, 0.849999, False),
    ],
)
def test_bootstrap_gate_requires_both_precision_and_recall(
    precision: float, recall: float, passed: bool
) -> None:
    result = bootstrap_gate({"precision": precision, "recall": recall})

    assert result["passed"] is passed
    assert result["failure_action"] == (None if passed else "stop_without_training_or_evaluation")


def test_truth_guard_requires_all_candidate_and_prediction_hashes_before_reads(
    tmp_path: Path,
) -> None:
    guard = TruthAccessGuard()
    truth_reads = []

    def reader(path: Path) -> list[dict]:
        if path.name in {f"{split}.truth" for split in EVALUATION_SPLITS}:
            assert set(guard.prediction_hashes) == set(EVALUATION_SPLITS)
        truth_reads.append(path.name)
        return [json.loads(path.read_text(encoding="utf-8"))]

    development_truth = tmp_path / "development.truth"
    development_truth.write_text('{"split": "development"}', encoding="utf-8")

    first_candidate = tmp_path / "development.candidates"
    first_candidate.write_text("development", encoding="utf-8")
    guard.register_candidate_file("development", first_candidate)
    with pytest.raises(RuntimeError, match="every candidate file is hashed"):
        guard.read_development_truth(development_truth, reader)

    for split in SPLITS[1:]:
        path = tmp_path / f"{split}.candidates"
        path.write_text(split, encoding="utf-8")
        guard.register_candidate_file(split, path)
    assert guard.read_development_truth(development_truth, reader) == [{"split": "development"}]

    truth_paths = {}
    for split in EVALUATION_SPLITS:
        truth_path = tmp_path / f"{split}.truth"
        truth_path.write_text(json.dumps({"split": split}), encoding="utf-8")
        truth_paths[split] = truth_path
    validation_predictions = tmp_path / "validation.predictions"
    validation_predictions.write_text("validation", encoding="utf-8")
    guard.register_prediction_file("validation", validation_predictions)
    with pytest.raises(RuntimeError, match="every prediction file is hashed"):
        guard.read_evaluation_truth(truth_paths, reader)
    assert truth_reads == ["development.truth"]

    heldout_predictions = tmp_path / "heldout.predictions"
    heldout_predictions.write_text("heldout", encoding="utf-8")
    guard.register_prediction_file("heldout", heldout_predictions)
    rows = guard.read_evaluation_truth(truth_paths, reader)

    assert set(rows) == set(EVALUATION_SPLITS)
    assert truth_reads == ["development.truth", "validation.truth", "heldout.truth"]
    assert all(
        set(entry["after_prediction_hashes"]) == set(EVALUATION_SPLITS)
        for entry in guard.access_log
        if entry["truth"] in EVALUATION_SPLITS
    )


def test_confidence_calibration_selects_strictest_rule_passing_non_pickup_gate() -> None:
    pseudo_rows = []
    exact_rows = []
    accepted_index = 0
    for measure in (2, 3, 4):
        identity = {
            "slug": "work",
            "system_index": 1,
            "system_measure_index": measure,
            "global_measure_index": measure - 1,
        }
        assignments = []
        labels = []
        for note_index in range(3):
            candidate_id = f"c{measure}{note_index}"
            pitch = 60 + accepted_index
            cost = 0.28 if accepted_index < 7 else 0.40
            assignments.append(
                {
                    "status": "aligned",
                    "match_cost": cost,
                    "pairs": [
                        {
                            "candidate_id": candidate_id,
                            "canonical_pitch_midi": pitch,
                            "absolute_pitch_error": 0,
                        }
                    ],
                }
            )
            labels.append({"candidate_id": candidate_id, "label": 1, "pitch_midi": pitch})
            accepted_index += 1
        pseudo_rows.append({"identity": identity, "alignment": {"assignments": assignments}})
        exact_rows.append({"identity": identity, "labels": labels})

    rule, calibration = calibrate_pseudo_acceptance_rule(pseudo_rows, exact_rows)

    assert rule == PseudoAcceptanceRule(0.30, 0)
    assert calibration["passed"] is True
    assert calibration["selected_metrics"]["precision"] == 1.0
    assert calibration["selected_metrics"]["recall"] == 0.777778


def test_pseudo_rule_applies_only_to_unreviewed_non_pickup_development_measures() -> None:
    pseudo_rows = []
    for system, measure_count in ((1, 8), (2, 9)):
        for measure in range(1, measure_count + 1):
            pseudo_rows.append(
                {
                    "identity": {
                        "slug": "work",
                        "system_index": system,
                        "system_measure_index": measure,
                        "global_measure_index": 100 * system + measure,
                    },
                    "alignment": {
                        "assignments": [
                            {
                                "status": "aligned",
                                "group_index": 0,
                                "match_cost": 0.1,
                                "pairs": [
                                    {
                                        "candidate_id": f"s{system}m{measure}",
                                        "canonical_pitch_midi": 60,
                                        "auto_pitch_midi": 60,
                                        "absolute_pitch_error": 0,
                                    }
                                ],
                            }
                        ]
                    },
                }
            )

    accepted = apply_pseudo_acceptance_rule(pseudo_rows, PseudoAcceptanceRule(0.30, 0))

    accepted_identities = {
        (row["identity"]["system_index"], row["identity"]["system_measure_index"])
        for row in accepted
    }
    assert len(accepted) == 13
    assert not accepted_identities & {(1, 1), (1, 2), (1, 3), (1, 4)}
    assert accepted_identities == {
        *((1, measure) for measure in range(5, 9)),
        *((2, measure) for measure in range(1, 10)),
    }


def test_validation_gate_requires_pitch_gain_and_non_regressing_count_error() -> None:
    passing = evaluate_validation_gate(
        {"ordered_pitch_accuracy": 0.180435, "mean_absolute_note_count_error": 1.714286}
    )
    weak_pitch = evaluate_validation_gate(
        {"ordered_pitch_accuracy": 0.18, "mean_absolute_note_count_error": 1.0}
    )
    weak_count = evaluate_validation_gate(
        {"ordered_pitch_accuracy": 0.5, "mean_absolute_note_count_error": 1.8}
    )

    assert passing["passed"] is True
    assert weak_pitch["passed"] is False
    assert weak_count["passed"] is False


def test_truth_guard_seals_heldout_until_validation_gate_and_prediction_hash(
    tmp_path: Path,
) -> None:
    guard = TruthAccessGuard()
    validation_prediction = tmp_path / "validation.predictions"
    validation_prediction.write_text("validation", encoding="utf-8")
    validation_truth = tmp_path / "validation.truth"
    validation_truth.write_text('{"split": "validation"}', encoding="utf-8")
    heldout_truth = tmp_path / "heldout.truth"
    heldout_truth.write_text('{"split": "heldout"}', encoding="utf-8")

    def reader(path: Path) -> list[dict]:
        return [json.loads(path.read_text(encoding="utf-8"))]

    guard.register_prediction_file("validation", validation_prediction)
    assert guard.read_validation_truth(validation_truth, reader) == [{"split": "validation"}]
    with pytest.raises(RuntimeError, match="passing validation gate"):
        guard.read_heldout_truth(heldout_truth, reader)

    guard.register_validation_gate(True)
    with pytest.raises(RuntimeError, match="persisted heldout prediction hash"):
        guard.read_heldout_truth(heldout_truth, reader)

    heldout_prediction = tmp_path / "heldout.predictions"
    heldout_prediction.write_text("heldout", encoding="utf-8")
    guard.register_prediction_file("heldout", heldout_prediction)
    assert guard.read_heldout_truth(heldout_truth, reader) == [{"split": "heldout"}]
    assert guard.access_log[-1]["after_validation_gate"] is True


def test_cross_system_partition_keeps_s7_training_and_s8_evaluation_disjoint() -> None:
    rows = [
        {
            "identity": {
                "slug": "work",
                "system_index": system,
                "system_measure_index": measure,
                "global_measure_index": 100 * system + measure,
            }
        }
        for system in (7, 8)
        for measure in range(1, 8)
    ]

    s7_rows, s8_rows = partition_cross_system_validation_rows(rows)

    assert {row["identity"]["system_index"] for row in s7_rows} == {7}
    assert {row["identity"]["system_index"] for row in s8_rows} == {8}
    s7_keys = {
        (
            row["identity"]["system_index"],
            row["identity"]["system_measure_index"],
            row["identity"]["global_measure_index"],
        )
        for row in s7_rows
    }
    s8_keys = {
        (
            row["identity"]["system_index"],
            row["identity"]["system_measure_index"],
            row["identity"]["global_measure_index"],
        )
        for row in s8_rows
    }
    assert len(s7_keys) == len(s8_keys) == 7
    assert s7_keys.isdisjoint(s8_keys)


def test_cross_system_truth_guard_orders_s7_s8_and_sealed_s3(tmp_path: Path) -> None:
    guard = CrossSystemTruthGuard()
    for split in SPLITS:
        candidate_path = tmp_path / f"{split}.candidates"
        candidate_path.write_text(split, encoding="utf-8")
        guard.register_candidate_file(split, candidate_path)

    validation_rows = [
        {
            "identity": {
                "slug": "work",
                "system_index": system,
                "system_measure_index": measure,
                "global_measure_index": 100 * system + measure,
            }
        }
        for system in (7, 8)
        for measure in range(1, 8)
    ]
    validation_truth = tmp_path / "validation.truth.jsonl"
    validation_truth.write_text(
        "".join(json.dumps(row) + "\n" for row in validation_rows),
        encoding="utf-8",
    )

    def prefix_reader(path: Path, row_count: int) -> list[dict]:
        return [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()[:row_count]
        ]

    def reader(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    s7_training = guard.read_s7_training_truth(
        validation_truth,
        prefix_reader,
        row_count=7,
    )
    assert {row["identity"]["system_index"] for row in s7_training} == {7}
    assert guard.access_log[-1]["system8_truth_parsed"] is False
    with pytest.raises(RuntimeError, match="persisted S8 prediction hash"):
        guard.read_validation_truth_after_s8_prediction(validation_truth, reader)

    s8_predictions = tmp_path / "system8.predictions"
    s8_predictions.write_text("frozen", encoding="utf-8")
    guard.register_s8_prediction_file(s8_predictions)
    s7_rows, s8_rows = guard.read_validation_truth_after_s8_prediction(validation_truth, reader)
    assert len(s7_rows) == len(s8_rows) == 7

    guard.register_s8_gate(False)
    heldout_predictions = tmp_path / "system3.predictions"
    heldout_predictions.write_text("sealed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="passing S8 gate"):
        guard.register_heldout_prediction_file(heldout_predictions)

    guard.register_s8_gate(True)
    guard.register_heldout_prediction_file(heldout_predictions)
    heldout_truth = tmp_path / "heldout.truth.jsonl"
    heldout_truth.write_text(
        json.dumps(
            {
                "identity": {
                    "slug": "work",
                    "system_index": 3,
                    "system_measure_index": 1,
                    "global_measure_index": 1,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert len(guard.read_heldout_truth(heldout_truth, reader)) == 1
    assert guard.access_log[-1]["truth"] == "system3_sealed_final_test"
