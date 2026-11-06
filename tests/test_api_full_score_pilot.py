import copy

import pytest

from scripts.evaluate_api_full_score_pilot import render, resource_row, validate_stage


def contract():
    plan = {
        "created_at": "2026-10-29T17:00:00+00:00",
        "model": "gpt-6-astra",
        "reasoning_effort": "xhigh",
        "workflow_id": "source-only",
        "target": {"source_pdf": {"sha256": "source"}},
        "deadline_seconds": {"initial": 3600, "reviewed": 1800},
    }
    provenance = {
        "started_at": "2026-10-29T17:01:00+00:00",
        "stage": "reviewed",
        "auth_mode": "openai_api",
        "source_execution": "sandboxed_source_process",
        "model": "gpt-6-astra",
        "reasoning_effort": "xhigh",
        "workflow_id": "source-only",
        "source_pdf_sha256": "source",
        "input_seal_sha256": "packet",
        "deadline_seconds": 1800,
        "reference_access": False,
        "independent_reader": True,
        "initial_abc_sha256": "author",
        "status": "complete",
        "complete": True,
        "returncode": 0,
    }
    return plan, provenance


def test_matching_fresh_review_is_eligible():
    plan, provenance = contract()
    validate_stage(provenance, plan, "reviewed", "packet", "author")


@pytest.mark.parametrize(
    "field,value",
    [
        ("reference_access", True),
        ("initial_abc_sha256", "another-author"),
        ("input_seal_sha256", "another-packet"),
        ("status", "partial"),
        ("complete", False),
        ("returncode", 1),
        ("deadline_seconds", 1200),
    ],
)
def test_mismatched_or_incomplete_review_is_rejected(field, value):
    plan, provenance = contract()
    provenance[field] = value
    with pytest.raises(ValueError, match="completed matching"):
        validate_stage(provenance, plan, "reviewed", "packet", "author")


def test_historical_author_cannot_be_presented_as_fresh():
    plan, provenance = contract()
    provenance["started_at"] = "2026-10-28T17:01:00+00:00"
    with pytest.raises(ValueError, match="fresh"):
        validate_stage(provenance, plan, "reviewed", "packet", "author")


def test_subscription_and_api_cache_accounting_use_the_same_denominator():
    subscription = {"usage": {"input_tokens": 100, "cached_input_tokens": 80}}
    api = {"usage": {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 80}}}
    assert resource_row(subscription)["cache_fraction"] == 0.8
    assert resource_row(api)["cache_fraction"] == 0.8


def test_dashboard_distinguishes_partial_source_coverage_and_edition_form():
    lane = {
        "status": "complete",
        "metrics": {
            "edition_note_f1": 0.9,
            "source_note_f1": 1.0,
            "chord_f1": 0.7,
            "form_exact_match": False,
            "meter_valid_rate": 1.0,
        },
        "coverage": {
            "candidate_measures": 45,
            "reference_measures": 45,
            "source_scored_measures": 9,
            "exact_measures": 20,
        },
        "validation": {"route_status": "resolved", "counts": {"performed_measures": 117}},
        "runtime": {"elapsed_seconds": 60, "usage": None, "cost_estimate_usd": None},
        "decisions": [{"reason": "<script>untrusted</script>"}],
    }
    report = {
        "stages": {"initial": lane, "reviewed": copy.deepcopy(lane)},
        "subscription_baseline": {"initial": lane, "reviewed": copy.deepcopy(lane)},
        "conservative_estimate_usd": "3.5",
        "live_reviews_unchanged": 13,
        "traces": {},
    }
    page = render(report)
    assert "9/45" in page
    assert "resolved; 117 bars; edition differs" in page
    assert "actual billing unverified" in page
    assert "not a pure transport comparison" in page
    assert "&lt;script&gt;untrusted&lt;/script&gt;" in page
    assert "<script>" not in page
