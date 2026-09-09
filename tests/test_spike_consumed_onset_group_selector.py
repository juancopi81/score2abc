from __future__ import annotations

from scripts.experiments import spike_consumed_onset_group_selector as spike


def _candidate(candidate_id: str, x: float, y: float, score: float) -> dict:
    return {
        "candidate_id": candidate_id,
        "center": {"x": x, "y": y},
        "score": score,
    }


def _observation(group_id: str, value: float, *, example_key: str = "synthetic"):
    return spike.GroupObservation(
        group_id=group_id,
        example_key=example_key,
        work_id="synthetic",
        x_center=float(value),
        candidate_ids=(group_id,),
        features=tuple(value for _ in spike.GROUP_FEATURES),
    )


def test_clusters_candidates_and_builds_truth_free_spacing_features() -> None:
    selected = [
        _candidate("c1", 10, 20, 0.8),
        _candidate("c2", 13, 30, 0.7),
        _candidate("c3", 50, 22, 0.6),
    ]
    candidate_features = {
        "c1": {
            "stem_score": 0.9,
            "dense_features": {
                "ink_density": 0.5,
                "core_density": 0.4,
                "line_dominance": 0.2,
                "stem_evidence": 0.8,
                "patch_center_density": 0.6,
            },
        },
        "c2": {
            "stem_score": 0.3,
            "dense_features": {"ink_density": 0.3},
        },
    }

    observations = spike.build_group_observations(
        selected,
        x_radius_px=10,
        spacing=10,
        candidate_features=candidate_features,
        image_width=100,
    )

    assert [observation.candidate_ids for observation in observations] == [
        ("c1", "c2"),
        ("c3",),
    ]
    first = observations[0].feature_map()
    assert first["cluster_size"] == 2
    assert first["vertical_spread_spaces"] == 1.0
    assert first["max_stem_score"] == 0.9
    assert first["right_gap_spaces"] == 3.85
    assert "label" not in first


def test_group_metrics_is_one_to_one_and_truth_scoring_is_separate() -> None:
    predicted = [10.0, 30.0]
    truth = [10.0, 20.0]

    metrics = spike.group_metrics(predicted, truth, tolerance_px=5)

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_filter_predictions_do_not_accept_labels_or_expected_counts() -> None:
    positive = _observation("positive", 2.0)
    negative = _observation("negative", -2.0)
    model = spike.fit_group_filter(
        [
            spike.LabeledGroup(positive, 1),
            spike.LabeledGroup(negative, 0),
        ]
    )

    first_prediction = spike.predict_groups(model, [positive, negative])
    changed_truth = spike.group_metrics(
        [item.x_center for item in first_prediction],
        [999.0, 1000.0, 1001.0, 1002.0],
        tolerance_px=1,
    )
    second_prediction = spike.predict_groups(model, [positive, negative])

    assert [item.group_id for item in first_prediction] == ["positive"]
    assert [item.group_id for item in second_prediction] == ["positive"]
    assert changed_truth["truth_group_count"] == 4
    assert model.training_positive_count == 1


def test_fit_threshold_keeps_all_training_positive_groups() -> None:
    groups = [
        spike.LabeledGroup(_observation("p1", 2.0), 1),
        spike.LabeledGroup(_observation("p2", 1.0), 1),
        spike.LabeledGroup(_observation("n1", -1.0), 0),
    ]

    model = spike.fit_group_filter(groups)
    predicted = spike.predict_groups(model, [group.observation for group in groups])

    assert {group.group_id for group in predicted} == {"p1", "p2"}


def test_aggregate_metrics_sums_nested_example_counts_and_counts_rows_without_them() -> None:
    metrics = spike._aggregate_metrics(
        [
            {
                "example_count": 19,
                "tp": 3,
                "fp": 1,
                "fn": 2,
                "predicted_group_count": 4,
                "truth_group_count": 5,
            },
            {"tp": 1, "fp": 0, "fn": 0, "predicted_group_count": 1, "truth_group_count": 1},
        ]
    )

    assert metrics["example_count"] == 20
