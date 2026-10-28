"""Presentation checks use invented metrics only, never musical reference answers."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from scripts.build_astra_reproduction_dashboard import render_dashboard


def report_fixture(tmp_path, target="aviador_s8"):
    report = {
        "schema_version": 1,
        "kind": "astra_assisted_workflow_reproduction_report",
        "target_id": target,
        "evidence_kind": "synthetic_control",
        "both_stages_verified_before_reference_content": True,
        "workflow_id": "synthetic_assisted_workflow",
        "model": "test-astra",
        "reasoning_effort": "xhigh",
        "created_at": "2026-10-28T20:00:00+00:00",
        "pins": [{"path": str(tmp_path / "evaluation_plan.json"), "sha256": "a" * 64}],
        "stage_artifacts": {},
        "stages": {},
        "unsupported_scope": ["whole-score source accuracy", "automatic editorial acceptance"],
    }
    for stage in ("initial", "reviewed"):
        report["stage_artifacts"][stage] = {}
        for kind, suffix in (("abc", ".abc"), ("decisions", ".json"), ("provenance", ".json")):
            # Intentionally absent files: renderer must not open score/reference artifacts.
            relative = f"{target}/{stage}/score & evidence/{kind}{suffix}"
            digest = hashlib.sha256(relative.encode()).hexdigest()
            report["stage_artifacts"][stage][kind] = {"path": relative, "sha256": digest}
            report["pins"].append({"path": str(tmp_path / relative), "sha256": digest})
        runtime = {
            "status": "complete",
            "returncode": 0,
            "complete": True,
            "elapsed_seconds": 59.7,
            "usage": None,
            "error": None,
        }
        if target == "aviador_s8":
            report["stages"][stage] = {
                "metrics": {
                    "summary": {
                        "note_f1": 0.91 if stage == "initial" else 0.98,
                        "rest_f1": 1,
                        "exact_measures": 5 if stage == "initial" else 6,
                        "targets": 7,
                    }
                },
                "completion_claim": True,
                "submission_validation": {
                    "headline_metrics_valid": True,
                    "abc_valid": True,
                    "submitted_measures": 7,
                    "expected_measures": 7,
                    "all_declared_meter_bars_valid": True,
                },
                "runtime": runtime,
            }
        else:
            report["stages"][stage] = {
                "metrics": {
                    "source_note_f1": 0.97,
                    "edition_note_f1": 0.85,
                    "edition_exact_bar_rate": 32 / 50,
                    "chord_f1": 0.79,
                    "form_exact_match": False,
                },
                "coverage": {
                    "source_eligible_measures": 9,
                    "source_scored_measures": 8,
                    "source_total_measures": 50,
                    "exact_measures": 32,
                    "reference_measures": 50,
                    "paired_measures": 48,
                    "alignment_columns": 50,
                },
                "status": "partial",
                "runtime": runtime,
            }
    report["gate"] = (
        {
            "advance_to_rumichaca": True,
            "exact_historical_reproduction_initial_only": False,
            "independent_review_made_gate_pass": True,
            "threshold": {"exact_measures_min": 6, "strict_note_f1_min": 0.95},
            "blocking_reasons": [],
        }
        if target == "aviador_s8"
        else None
    )
    return report


def save_report(tmp_path, report):
    path = tmp_path / f'{report["target_id"]}_report.json'
    path.write_text(json.dumps(report))
    return path


def test_stages_gate_and_missing_full_score_are_distinct(tmp_path):
    path = save_report(tmp_path, report_fixture(tmp_path))
    html = render_dashboard(path, tmp_path / "index.html")
    assert "SYNTHETIC CONTROL" in html
    assert "After independent AI review" in html
    assert "Initial AI draft" in html
    assert "91.0%" in html and "98.0%" in html
    assert "5 / 7" in html and "6 / 7" in html
    assert "Passed after independent AI review" in html
    cards = html.split('<div class="cards">', 1)[1].split("Protocol checks:", 1)[0]
    assert "5 / 7 exact bars" in cards and "6 / 7 exact bars" in cards
    assert "Result pending" in cards
    assert "Not reproduced" not in cards and "advancement: <strong>Passed" not in cards
    assert "No full-score result yet" in html
    assert "not a new model result" in html
    assert "ties are not scored" in html
    assert "1m 00s" in html and "0m 60s" not in html
    assert "Not recorded" in html
    assert "$0" not in html and "0 input" not in html
    assert "<script" not in html and "https://" not in html


def test_blocked_gate_and_timed_out_checkpoint_remain_visible(tmp_path):
    report = report_fixture(tmp_path)
    report["gate"].update(
        advance_to_rumichaca=False,
        independent_review_made_gate_pass=False,
        blocking_reasons=["Independent reader timed out <not successful>."],
    )
    report["stages"]["reviewed"]["runtime"].update(
        status="partial", timed_out=True, returncode=None
    )
    html = render_dashboard(save_report(tmp_path, report), tmp_path / "index.html")
    assert "Time limit reached" in html
    assert "Author completion claim: yes" in html
    assert "Skipped by this protocol" in html
    assert "timed out &lt;not successful&gt;" in html
    assert "Made the gate pass" not in html


def test_initial_exact_result_is_separate_from_blocked_advancement(tmp_path):
    report = report_fixture(tmp_path)
    report["stages"]["initial"]["metrics"]["summary"].update(exact_measures=7, note_f1=1)
    report["gate"].update(
        exact_historical_reproduction_initial_only=True,
        advance_to_rumichaca=False,
        independent_review_made_gate_pass=False,
    )
    html = render_dashboard(save_report(tmp_path, report), tmp_path / "index.html")
    assert "7 / 7 exact bars" in html
    assert "Blocked" in html
    assert "Exact initial-draft reproduction: <strong>Yes</strong>" in html


def test_full_score_source_subset_and_edition_agreement_are_not_pooled(tmp_path):
    aviador_path = save_report(tmp_path, report_fixture(tmp_path))
    rumichaca = report_fixture(tmp_path, "rumichaca_full")
    rumichaca["pins"].append(
        {
            "path": str(aviador_path),
            "sha256": hashlib.sha256(aviador_path.read_bytes()).hexdigest(),
        }
    )
    rumichaca_path = save_report(tmp_path, rumichaca)
    html = render_dashboard(aviador_path, tmp_path / "index.html", rumichaca_path)
    assert "9 / 50 source bars audited" in html
    assert "97.0%" in html and "85.0%" in html
    assert "Audited source notes" in html
    assert "Exact edition bars" in html
    assert "Edition playback route" in html and "Different" in html
    assert "Score validation: partial" in html
    assert "must not be pooled" in html
    assert "Editorial acceptance is unmeasured" in html
    cards = html.split('<div class="cards">', 1)[1].split("Protocol checks:", 1)[0]
    assert "85.0% edition note F1" in cards
    assert "Full-score result available; score status: partial" in cards
    assert "Initial draft: 85.0%" in cards
    assert "Notes, rests, ties, meter and chord events must all match" in html
    assert "This is not melody-only accuracy" in html


def test_known_usage_is_partial_and_escaped_local_evidence_links_work(tmp_path):
    report = report_fixture(tmp_path)
    report["model"] = '<script>alert("bad")</script>'
    report["stages"]["initial"]["runtime"].update(
        usage=[{"input_tokens": 1000, "output_tokens": 125, "cached_input_tokens": 700}, None],
        error="<img src=x onerror=bad>",
    )
    report["stages"]["reviewed"]["runtime"]["elapsed_seconds"] = None
    html = render_dashboard(save_report(tmp_path, report), tmp_path / "index.html")
    assert "1,000 input · 125 output" in html
    assert "Reported for 1 / 2 usage records" in html
    assert "1,700 input" not in html
    assert "700 cached input (subset, not additional)" in html
    assert "score%20%26%20evidence/abc.abc" in html
    assert "&lt;script&gt;alert(&quot;bad&quot;)&lt;/script&gt;" in html
    assert "&lt;img src=x onerror=bad&gt;" in html
    assert "<script" not in html and "<img " not in html


def test_exact_edition_bar_denominator_penalizes_extra_candidate_bar(tmp_path):
    aviador_path = save_report(tmp_path, report_fixture(tmp_path))
    report = report_fixture(tmp_path, "rumichaca_full")
    report["pins"].append(
        {
            "path": str(aviador_path),
            "sha256": hashlib.sha256(aviador_path.read_bytes()).hexdigest(),
        }
    )
    for lane in report["stages"].values():
        lane["metrics"]["edition_exact_bar_rate"] = 7 / 8
        lane["coverage"].update(
            exact_measures=7, reference_measures=7, paired_measures=7, alignment_columns=8
        )
    html = render_dashboard(aviador_path, tmp_path / "index.html", save_report(tmp_path, report))
    exact_row = html.split("Exact edition bars", 1)[1].split("</tr>", 1)[0]
    assert "87.5%" in exact_row and "7 / 8 alignment columns" in exact_row
    assert "7 / 7" not in exact_row
    coverage_row = html.split("Reference bars aligned", 1)[1].split("</tr>", 1)[0]
    assert "7 / 7" in coverage_row
    report["stages"]["initial"]["metrics"]["edition_exact_bar_rate"] = 1
    with pytest.raises(ValueError, match="alignment-based rate"):
        render_dashboard(aviador_path, tmp_path / "index.html", save_report(tmp_path, report))


def test_unscorable_aviador_does_not_report_zero_as_measured_accuracy(tmp_path):
    report = report_fixture(tmp_path)
    lane = report["stages"]["initial"]
    lane["submission_validation"].update(
        headline_metrics_valid=False, abc_valid=False, all_declared_meter_bars_valid=False
    )
    lane["metrics"]["summary"].update(note_f1=0, rest_f1=0, exact_measures=0)
    report["gate"].update(
        advance_to_rumichaca=False,
        independent_review_made_gate_pass=False,
        blocking_reasons=["Initial submission is unscorable."],
    )
    html = render_dashboard(save_report(tmp_path, report), tmp_path / "index.html")
    assert "Invalid / unscorable" in html
    assert "ABC invalid; 7 / 7 written bars; meter not valid" in html
    assert "No accuracy claim" in html and "Blocked" in html
    for label in ("Strict historical note F1", "Exact event bars", "Rest F1"):
        row = html.split(label, 1)[1].split("</tr>", 1)[0]
        assert "Unscorable" in row
        assert ">0.0%<" not in row and "0 / 7" not in row
    # Successful execution and a completion claim must not imply musical validity.
    execution = html.split('scope="row">Execution', 1)[1].split("</tr>", 1)[0]
    assert "Complete" in execution and "Author completion claim: yes" in execution
    report["gate"]["advance_to_rumichaca"] = True
    with pytest.raises(ValueError, match="unscorable stage cannot pass"):
        render_dashboard(save_report(tmp_path, report), tmp_path / "index.html")


@pytest.mark.parametrize("mutation", ["foreign_workflow", "foreign_plan", "foreign_gate_report"])
def test_full_score_rejects_incompatible_reports(tmp_path, mutation):
    aviador_path = save_report(tmp_path, report_fixture(tmp_path))
    report = report_fixture(tmp_path, "rumichaca_full")
    report["pins"].append(
        {
            "path": str(aviador_path),
            "sha256": hashlib.sha256(aviador_path.read_bytes()).hexdigest(),
        }
    )
    if mutation == "foreign_workflow":
        report["workflow_id"] = "another_workflow"
    elif mutation == "foreign_plan":
        report["pins"][0]["sha256"] = "b" * 64
    else:
        report["pins"][-1]["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="same passing preregistered workflow"):
        render_dashboard(aviador_path, tmp_path / "index.html", save_report(tmp_path, report))


@pytest.mark.parametrize("invalid", ["remote_link", "nan", "exact_claim", "unsealed"])
def test_rejects_misleading_or_unsafe_report_data(tmp_path, invalid):
    report = copy.deepcopy(report_fixture(tmp_path))
    if invalid == "remote_link":
        report["pins"][0]["path"] = "https://example.test/evaluation_plan.json"
    elif invalid == "nan":
        report["stages"]["initial"]["metrics"]["summary"]["note_f1"] = float("nan")
    elif invalid == "exact_claim":
        report["gate"]["exact_historical_reproduction_initial_only"] = True
    else:
        report["both_stages_verified_before_reference_content"] = False
    with pytest.raises(ValueError):
        render_dashboard(save_report(tmp_path, report), tmp_path / "index.html")
