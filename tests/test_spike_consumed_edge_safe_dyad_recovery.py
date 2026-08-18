from scripts.experiments import spike_consumed_edge_safe_dyad_recovery as spike


def test_decision_requires_polyphonic_gain_and_no_candidate_regression() -> None:
    alcira = {
        "baseline": {"note_count_f1": 0.6, "exact_pitch_matches": 10},
        "edge_safe_recovery": {"note_count_f1": 0.8, "exact_pitch_matches": 12},
    }
    la_chata = {
        "baseline": {
            "selection_count_metrics": {"predicted_total_head_count": 30},
            "pitch_group_metrics": {"exact_group_count": 10, "group_edit_distance": 20},
        },
        "edge_safe_recovery": {
            "selection_count_metrics": {"predicted_total_head_count": 35},
            "pitch_group_metrics": {"exact_group_count": 14, "group_edit_distance": 16},
        },
    }
    candidate_regression = {
        "baseline": {"f1": 0.79},
        "edge_safe_recovery": {"f1": 0.79, "recovered_false_positive_count": 0},
    }

    decision = spike._decision(alcira, la_chata, candidate_regression)

    assert decision["status"] == "advance_to_independent_gate"
    assert decision["la_chata_consumed_pitch_groups_improved"] is True
    assert decision["runtime_adoption_eligible"] is False


def test_decision_rejects_candidate_regression() -> None:
    alcira = {
        "baseline": {"note_count_f1": 0.6, "exact_pitch_matches": 10},
        "edge_safe_recovery": {"note_count_f1": 0.8, "exact_pitch_matches": 12},
    }
    la_chata = {
        "baseline": {
            "selection_count_metrics": {"predicted_total_head_count": 30},
            "pitch_group_metrics": {"exact_group_count": 10, "group_edit_distance": 20},
        },
        "edge_safe_recovery": {
            "selection_count_metrics": {"predicted_total_head_count": 35},
            "pitch_group_metrics": {"exact_group_count": 14, "group_edit_distance": 16},
        },
    }
    candidate_regression = {
        "baseline": {"f1": 0.79},
        "edge_safe_recovery": {"f1": 0.78, "recovered_false_positive_count": 1},
    }

    decision = spike._decision(alcira, la_chata, candidate_regression)

    assert decision["status"] == "reject_or_revise"
    assert decision["aviador_carrizal_candidate_regression_preserved"] is False
