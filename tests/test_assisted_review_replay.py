from copy import deepcopy

import pytest

from scripts.evaluate_assisted_review_replay import delta, render, resources


def lane():
    return {
        "metrics": {
            "results": [
                {"identity": {"system_measure_index": i}, "exact_measure": i != 7}
                for i in range(1, 8)
            ]
        },
        "adapted_events": [
            {
                "identity": {"system_measure_index": i},
                "notes": [{"pitch_midi": p} for p in [74, 69, 65, 71, 69, 65]],
                "rests": [],
            }
            for i in range(1, 8)
        ],
    }


def test_sharp_repair_with_no_other_event_changes():
    before = lane()
    after = deepcopy(before)
    after["metrics"]["results"][-1]["exact_measure"] = True
    for i in (2, 5):
        after["adapted_events"][-1]["notes"][i]["pitch_midi"] = 66
    assert delta(before, after) == {
        "sharp_fixed": True,
        "corrected_bars": [7],
        "regressed_previously_exact_bars": [],
        "changed_bars": [7],
        "all_scored_events_match": True,
    }


def test_sharp_fix_does_not_hide_a_regression_elsewhere():
    before = lane()
    after = deepcopy(before)
    for i in (2, 5):
        after["adapted_events"][-1]["notes"][i]["pitch_midi"] = 66
    after["metrics"]["results"][-1]["exact_measure"] = True
    after["metrics"]["results"][1]["exact_measure"] = False
    after["adapted_events"][1]["notes"][0]["pitch_midi"] = 75
    result = delta(before, after)
    assert result["sharp_fixed"] is True
    assert result["regressed_previously_exact_bars"] == [2]
    assert result["changed_bars"] == [2, 7]
    assert result["all_scored_events_match"] is False


def test_missing_bar_cannot_be_ignored():
    before, after = lane(), lane()
    after["metrics"]["results"].pop()
    with pytest.raises(ValueError, match="seven target bars"):
        delta(before, after)


def test_unknown_cache_and_billing_stay_unknown():
    result = resources({"runtime": {"cost_estimate_usd": "1.2", "usage": None}})
    assert result["cache_fraction"] is None
    assert result["actual_billed_usd"] is None
    assert result["conservative_estimate_usd"] == "1.2"


def test_cache_fraction_uses_input_tokens_only():
    result = resources(
        {
            "runtime": {
                "usage": {
                    "input_tokens": 1000,
                    "input_tokens_details": {"cached_tokens": 900},
                    "output_tokens": 500,
                }
            }
        }
    )
    assert result["cache_fraction"] == 0.9


def test_dashboard_renders_runtime_and_escapes_decisions():
    candidate = lane()
    candidate["runtime"] = {"elapsed_seconds": 325, "usage": {"output_tokens": 12000}}
    candidate["metrics"]["summary"] = {"exact_measures": 6, "note_f1": 0.93}
    candidate["decisions"] = [{"reason": "<script>bad</script>"}]
    report = {
        "before": candidate,
        "after": candidate,
        "delta": delta(candidate, candidate),
        "resources": {"before": resources(candidate), "after": resources(candidate)},
        "review_instructions_version": "v2",
        "review_instructions_sha256": "sha",
        "live_reviews_unchanged": 13,
    }
    html = render(report)
    assert "5m 25s" in html
    assert "6/7" in html
    assert "&lt;script&gt;bad&lt;/script&gt;" in html
    assert "<script>" not in html
