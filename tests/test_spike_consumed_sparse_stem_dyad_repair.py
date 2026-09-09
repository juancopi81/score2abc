from scripts.experiments import spike_consumed_sparse_stem_dyad_repair as spike

SELECTOR = {
    "threshold": 0.5,
    "nms_x_spaces": 0.85,
    "minimum_selected_count": 2,
    "maximum_selected_count": 8,
}


def test_sparse_shared_stem_pair_replaces_weak_second_group() -> None:
    row = _row()
    candidates = spike._normalized_candidates(row)
    by_id = {item["candidate_id"]: item for item in candidates}

    decision = spike.propose_sparse_shared_stem_dyad(
        row,
        SELECTOR,
        [by_id["anchor"], by_id["dot"]],
        _stem_features(),
    )

    assert decision["accepted"] is True
    assert decision["proposed_ids"] == ["anchor", "missing_head"]
    assert [item["candidate_id"] for item in decision["displaced"]] == ["dot"]


def test_sparse_rule_rejects_multiple_visual_onset_clusters() -> None:
    row = _row(extra_pair=True)
    candidates = spike._normalized_candidates(row)
    by_id = {item["candidate_id"]: item for item in candidates}
    stems = {
        **_stem_features(),
        "later_a": _stem(84, 0.9),
        "later_b": _stem(84, 0.4),
        "later_dot_upper": _stem(100, 0.1),
        "later_dot_lower": _stem(100, 0.1),
    }

    decision = spike.propose_sparse_shared_stem_dyad(
        row,
        SELECTOR,
        [by_id["anchor"], by_id["dot"]],
        stems,
    )

    assert decision["accepted"] is False
    assert decision["reason"] == "rejected_sparse_pair_cluster_count"
    assert decision["pair_cluster_count"] == 2


def test_sparse_rule_will_not_replace_a_strong_selected_anchor() -> None:
    row = _row()
    candidates = spike._normalized_candidates(row)
    by_id = {item["candidate_id"]: item for item in candidates}
    stems = {**_stem_features(), "other": _stem(70, 0.8)}

    decision = spike.propose_sparse_shared_stem_dyad(
        row,
        SELECTOR,
        [by_id["anchor"], by_id["other"]],
        stems,
    )

    assert decision["accepted"] is False
    assert decision["reason"] == "rejected_strong_displaced_candidate"


def test_sparse_rule_rejects_non_sparse_current_lane() -> None:
    row = _row()
    candidates = spike._normalized_candidates(row)

    decision = spike.propose_sparse_shared_stem_dyad(
        row,
        SELECTOR,
        candidates[:3],
        _stem_features(),
    )

    assert decision == {
        "accepted": False,
        "reason": "rejected_current_candidate_count",
        "current_ids": [item["candidate_id"] for item in candidates[:3]],
    }


def _row(*, extra_pair: bool = False) -> dict:
    candidates = [
        _candidate("anchor", 40, 30, 0.9),
        _candidate("missing_head", 42, 40, 0.7),
        _candidate("dot_upper", 56, 30, -0.1),
        _candidate("dot", 55, 40, -0.05),
        _candidate("other", 70, 50, 0.6),
    ]
    if extra_pair:
        candidates.extend(
            [
                _candidate("later_a", 80, 30, 0.8),
                _candidate("later_b", 82, 40, 0.7),
                _candidate("later_dot_upper", 100, 30, -0.1),
                _candidate("later_dot_lower", 100, 40, -0.1),
            ]
        )
    return {
        "staff_geometry": {"raw_staff_lines_y_px": [20, 30, 40, 50, 60]},
        "candidate_predictions": candidates,
    }


def _candidate(candidate_id: str, x: float, y: float, score: float) -> dict:
    return {
        "candidate_id": candidate_id,
        "center": {"x": x, "y": y},
        "score": score,
        "detector_rank": 1,
        "bbox": {"left": x - 4, "top": y - 4, "right": x + 4, "bottom": y + 4},
    }


def _stem_features() -> dict:
    return {
        "anchor": _stem(44, 0.9),
        "missing_head": _stem(42, 0.4),
        "dot_upper": _stem(56, 0.1),
        "dot": _stem(55, 0.1),
        "other": _stem(70, 0.8),
    }


def _stem(x: float, score: float) -> dict:
    return {"x": x, "score": score}
