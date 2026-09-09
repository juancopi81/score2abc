from scripts.experiments import spike_consumed_multihead_chord_recovery as spike


def _result(*, passed: bool, pitch: int, chord: int, structures: int, cap: int = 2):
    return {
        "config_id": f"config-{pitch}-{chord}-{cap}",
        "parameters": {
            "maximum_recovered_heads_per_group": cap,
            "minimum_stem_score": 0.55,
            "minimum_y_gap_staff_spaces": 1.0,
        },
        "gate": {"passed": passed},
        "evaluations": {
            "no_lo_creas": {
                "multihead_recovery": {
                    "chord_size_alignment_accuracy": 0.5,
                    "ordered_diatonic_alignment_accuracy": 0.4,
                },
                "delta_vs_previous_dyad": {
                    "exact_chord_size_matches": chord,
                    "exact_diatonic_staff_position_matches": pitch,
                    "exact_structure_crops": structures,
                },
                "recovery_records": [{"recovered_candidate_ids": ["d1"]}],
            }
        },
    }


def test_config_grid_is_small_and_deterministic() -> None:
    configs = spike._config_grid()

    assert len(configs) == 8
    assert len({config["config_id"] for config in configs}) == 8
    assert {config["parameters"]["maximum_recovered_heads_per_group"] for config in configs} == {
        2,
        3,
    }


def test_select_best_rejects_all_failed_configs() -> None:
    assert spike._select_best([_result(passed=False, pitch=3, chord=4, structures=2)]) is None


def test_select_best_prioritizes_chord_structure_then_pitch() -> None:
    pitch_first = _result(passed=True, pitch=3, chord=1, structures=1)
    chord_first = _result(passed=True, pitch=1, chord=2, structures=1)

    assert spike._select_best([pitch_first, chord_first]) is chord_first


def test_summary_delta_compares_against_previous_rule() -> None:
    baseline = {
        "predicted_note_count": 10,
        "note_count_f1": 0.5,
        "exact_diatonic_staff_position_matches": 4,
        "ordered_diatonic_alignment_accuracy": 0.2,
        "exact_chord_size_matches": 2,
        "chord_size_alignment_accuracy": 0.25,
        "exact_structure_crops": 1,
    }
    recovered = {
        "predicted_note_count": 12,
        "note_count_f1": 0.6,
        "exact_diatonic_staff_position_matches": 6,
        "ordered_diatonic_alignment_accuracy": 0.3,
        "exact_chord_size_matches": 3,
        "chord_size_alignment_accuracy": 0.35,
        "exact_structure_crops": 2,
    }

    assert spike._summary_delta(recovered, baseline) == {
        "predicted_note_count": 2,
        "note_count_f1": 0.1,
        "exact_diatonic_staff_position_matches": 2,
        "ordered_diatonic_alignment_accuracy": 0.1,
        "exact_chord_size_matches": 1,
        "chord_size_alignment_accuracy": 0.1,
        "exact_structure_crops": 1,
    }
