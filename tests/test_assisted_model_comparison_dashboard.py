"""The assisted comparison uses synthetic metrics and metadata, never music."""

from __future__ import annotations

import hashlib
import json

import pytest

from scripts.build_assisted_model_comparison_dashboard import load_comparison, render_dashboard


def write_json(path, value):
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def fixture_report(tmp_path, system, case):
    report = {
        "schema_version": 1,
        "kind": "astra_assisted_workflow_reproduction_report",
        "target_id": case,
        "evidence_kind": "synthetic_control",
        "both_stages_verified_before_reference_content": True,
        "workflow_id": "synthetic_assisted_workflow",
        "model": f"test-{system}",
        "reasoning_effort": "xhigh",
        "created_at": "2026-10-29T00:00:00+00:00",
        "pins": [{"path": str(tmp_path / "evaluation_plan.json"), "sha256": "a" * 64}],
        "stage_artifacts": {},
        "stages": {},
        "unsupported_scope": ["unseen scores", "automatic editorial acceptance"],
    }
    for stage in ("initial", "reviewed"):
        runtime = {
            "status": "complete",
            "returncode": 0,
            "timed_out": False,
            "elapsed_seconds": 100 if system == "astra" else 50,
            "error": None,
            "usage": {
                "input_tokens": 1000 if system == "astra" else 600,
                "cached_input_tokens": 600 if system == "astra" else 300,
                "output_tokens": 100 if system == "astra" else 50,
                "reasoning_output_tokens": 70 if system == "astra" else 20,
            },
        }
        if case == "aviador_s8":
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
                    "edition_exact_bar_rate": 20 / 46,
                    "chord_f1": 0.79,
                    "form_exact_match": False,
                },
                "coverage": {
                    "source_eligible_measures": 9,
                    "source_scored_measures": 9,
                    "source_total_measures": 45,
                    "exact_measures": 20,
                    "reference_measures": 45,
                    "paired_measures": 45,
                    "alignment_columns": 46,
                },
                "status": "complete",
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
        if case == "aviador_s8"
        else None
    )
    return report


def fixture_bundle(tmp_path):
    manifest = {
        "schema_version": 1,
        "kind": "assisted_model_comparison_manifest",
        "comparison_id": "synthetic_trial_v1",
        "trial": 1,
        "baseline_system_id": "astra",
        "systems": [],
    }
    for identifier in ("astra", "terra"):
        system = {
            "id": identifier,
            "label": identifier.title(),
            "requested_model": f"test-{identifier}",
            "reasoning_effort": "xhigh",
            "reports": {},
        }
        for case in ("aviador_s8", "rumichaca_full"):
            system["reports"][case] = write_json(
                tmp_path / f"{identifier}_{case}.json",
                fixture_report(tmp_path, identifier, case),
            )
        manifest["systems"].append(system)
    audit = {
        "schema_version": 1,
        "kind": "assisted_model_comparison_compatibility_audit",
        "report_sha256": {
            s["id"]: {case: pin["sha256"] for case, pin in s["reports"].items()}
            for s in manifest["systems"]
        },
        "checks": [
            {
                "case_id": case,
                "stage": stage,
                "dimension": dimension,
                "status": "matched",
                "fingerprints": {"astra": "a" * 64, "terra": "a" * 64},
                "basis": "Synthetic normalized contract; candidate text is excluded.",
            }
            for case in ("aviador_s8", "rumichaca_full")
            for stage in ("initial", "reviewed")
            for dimension in ("data", "prompts", "tools", "limits", "grader")
        ],
    }
    return manifest, audit


def save_bundle(tmp_path, manifest, audit):
    manifest["compatibility_audit"] = write_json(tmp_path / "audit.json", audit)
    write_json(tmp_path / "manifest.json", manifest)
    return tmp_path / "manifest.json"


def update_report(tmp_path, manifest, audit, system, case, report):
    entry = next(s for s in manifest["systems"] if s["id"] == system)
    pin = write_json(tmp_path / f"{system}_{case}.json", report)
    entry["reports"][case] = pin
    audit["report_sha256"][system][case] = pin["sha256"]


def test_matched_dashboard_separates_models_stages_metrics_and_resource_subsets(tmp_path):
    manifest, audit = fixture_bundle(tmp_path)
    path = save_bundle(tmp_path, manifest, audit)
    html = render_dashboard(path, tmp_path / "index.html")
    assert "SYNTHETIC CONTROL" in html
    assert "test-astra" in html and "test-terra" in html and "xhigh reasoning" in html
    assert "Initial AI draft" in html and "After independent AI review" in html
    cards = html.split('<div class="cards">', 1)[1].split('class="notice', 1)[0]
    assert cards.count('class="summary-metrics"') == 2
    assert '<th scope="col">Initial AI draft</th>' in cards
    assert '<th scope="col">After AI review</th>' in cards
    assert '91.0%</strong></td><td><strong class="metric">98.0%' in cards
    assert "→" not in cards
    assert "Matched conditions verified by the pinned audit" in html
    assert "9 / 45 source bars audited" in html
    assert "20 / 46 alignment columns" in html
    assert "not melody-only accuracy" in html
    assert "excludes ties" in html and "note metrics include" in html
    assert "Different from edition" in html
    assert "Terra / Astra" in html and "0.60×" in html and "0.50×" in html
    assert "0.29×" in html  # Reasoning subset 20 / 70; not added to output.
    assert "Input includes cached input" in html
    assert "1,600" not in html and "170" not in html
    assert "one quality-scored trial per model and case" in html
    assert "no API-dollar" in html
    assert "<script" not in html and "https://" not in html


def test_missing_compatibility_check_prevents_full_matched_claim_and_pair_ratio(tmp_path):
    manifest, audit = fixture_bundle(tmp_path)
    audit["checks"] = audit["checks"][1:]
    path = save_bundle(tmp_path, manifest, audit)
    comparison = load_comparison(path)
    assert not comparison["matched_pairs"][("aviador_s8", "initial")]
    assert comparison["matched_pairs"][("aviador_s8", "reviewed")]
    html = render_dashboard(path, tmp_path / "index.html")
    assert "Matching conditions not fully verified" in html
    assert "Matching conditions unverified" in html
    assert "91.0%" in html  # Descriptive metrics remain visible.
    assert "Direct comparison is supported for: After independent AI review." in html


def test_runtime_mismatch_is_descriptive_only_for_affected_case(tmp_path):
    manifest, audit = fixture_bundle(tmp_path)
    for check in audit["checks"]:
        if check["case_id"] == "rumichaca_full" and check["dimension"] == "tools":
            check["status"] = "mismatched"
            check["fingerprints"]["terra"] = "b" * 64
            check["basis"] = "Same tool implementation; Codex runtime version changed."
    path = save_bundle(tmp_path, manifest, audit)
    comparison = load_comparison(path)
    assert comparison["matched_pairs"][("aviador_s8", "initial")]
    assert comparison["matched_pairs"][("aviador_s8", "reviewed")]
    assert not comparison["matched_pairs"][("rumichaca_full", "initial")]
    assert not comparison["matched_pairs"][("rumichaca_full", "reviewed")]
    html = render_dashboard(path, tmp_path / "index.html")
    header = html.split("</header>", 1)[0]
    assert "Aviador · system 8.</strong> Comparable within this case" in header
    assert "Rumichaca · whole-score task.</strong> Descriptive results only" in header
    assert "conditions differ (tools or runtime)" in header
    assert "cannot be attributed to model choice alone" in header
    assert "91.0%" in header and "85.0%" in header
    ratios = html.split('class="ratio"', 1)[1].split("</table>", 1)[0]
    assert ratios.count("0.50×") == 6  # Time, cached input and output; only Aviador stages.
    assert ratios.count("Conditions differ; descriptive results only") == 2
    assert "0.60×" in ratios  # Matched Aviador input comparisons remain available.


def test_experiment_notes_are_escaped_near_summary_and_do_not_replace_quality(tmp_path):
    manifest, audit = fixture_bundle(tmp_path)
    manifest["experiment_notes"] = [
        "Runtime changed from 0.164.0-alpha.11.1 to 0.165.0-alpha.1.2.",
        "First Rumichaca attempt: interrupted after 123.72s; usage unknown; not quality-scored. "
        "The tables show the fresh attempt.",
        '<script>alert("not executable")</script>',
    ]
    html = render_dashboard(save_bundle(tmp_path, manifest, audit), tmp_path / "index.html")
    header = html.split("</header>", 1)[0]
    assert header.index('summary-metrics"') < header.index('class="notice experiment-notes"')
    assert "0.164.0-alpha.11.1 to 0.165.0-alpha.1.2" in header
    assert "123.72s; usage unknown; not quality-scored" in header
    assert "fresh attempt" in header and "85.0%" in header
    assert "&lt;script&gt;alert(&quot;not executable&quot;)&lt;/script&gt;" in header
    assert "<script>" not in html


@pytest.mark.parametrize("notes", [None, "a single note", {"note": "text"}, [1], ["valid", None]])
def test_experiment_notes_require_a_list_of_strings(tmp_path, notes):
    manifest, audit = fixture_bundle(tmp_path)
    manifest["experiment_notes"] = notes
    with pytest.raises(ValueError, match="experiment_notes must be a list of strings"):
        render_dashboard(save_bundle(tmp_path, manifest, audit), tmp_path / "index.html")


@pytest.mark.parametrize("mutation", ["unequal", "bad_binding", "duplicate", "mixed_evidence"])
def test_rejects_falsely_matched_or_incompatible_audit(tmp_path, mutation):
    manifest, audit = fixture_bundle(tmp_path)
    if mutation == "unequal":
        audit["checks"][0]["fingerprints"]["terra"] = "b" * 64
    elif mutation == "bad_binding":
        audit["report_sha256"]["terra"]["aviador_s8"] = "b" * 64
    elif mutation == "duplicate":
        audit["checks"].append(audit["checks"][0])
    else:
        report = fixture_report(tmp_path, "terra", "aviador_s8")
        report["evidence_kind"] = "consumed_development_workflow_reproduction"
        update_report(tmp_path, manifest, audit, "terra", "aviador_s8", report)
    with pytest.raises(ValueError):
        render_dashboard(save_bundle(tmp_path, manifest, audit), tmp_path / "index.html")


def test_changed_report_bytes_fail_pin_verification(tmp_path):
    manifest, audit = fixture_bundle(tmp_path)
    path = save_bundle(tmp_path, manifest, audit)
    report_path = tmp_path / "astra_aviador_s8.json"
    report_path.write_text(report_path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        render_dashboard(path, tmp_path / "index.html")


def test_failed_runtime_keeps_partial_metrics_usage_and_withholds_ratio(tmp_path):
    manifest, audit = fixture_bundle(tmp_path)
    report = fixture_report(tmp_path, "terra", "rumichaca_full")
    lane = report["stages"]["reviewed"]
    lane["runtime"].update(
        status="partial", returncode=None, timed_out=True, error="deadline <hit>"
    )
    lane["status"] = "partial"
    lane["coverage"]["paired_measures"] = 20
    update_report(tmp_path, manifest, audit, "terra", "rumichaca_full", report)
    html = render_dashboard(save_bundle(tmp_path, manifest, audit), tmp_path / "index.html")
    assert "Execution: partial" in html and "Time limit reached" in html
    assert "deadline &lt;hit&gt;" in html and "20 / 45 reference bars aligned" in html
    assert "Incomplete execution or score submission; workload differs" in html
    assert "85.0%" in html and ">600<" in html


@pytest.mark.parametrize("runtime_status", ["complete", "partial"])
def test_completed_model_turn_with_partial_score_preserves_coverage_and_withholds_ratio(
    tmp_path, runtime_status
):
    manifest, audit = fixture_bundle(tmp_path)
    report = fixture_report(tmp_path, "terra", "rumichaca_full")
    lane = report["stages"]["initial"]
    lane["runtime"].update(status=runtime_status, completed_turn=True, returncode=0, complete=False)
    lane["status"] = "partial"
    lane["coverage"].update(candidate_measures=17, paired_measures=17)
    update_report(tmp_path, manifest, audit, "terra", "rumichaca_full", report)
    html = render_dashboard(save_bundle(tmp_path, manifest, audit), tmp_path / "index.html")
    header = html.split("</header>", 1)[0]
    assert "Partial score" in header and "85.0%" in header
    assert "Model turn completed; exit code 0" in html
    assert "Completion claimed: no" in html and "Score: partial" in html
    assert "17 / 45 written bars" in html and "17 / 45 reference bars aligned" in html
    ratios = html.split('class="ratio"', 1)[1].split("</table>", 1)[0]
    row = next(
        row for row in ratios.split("<tr>") if "Rumichaca" in row and "Initial AI draft" in row
    )
    assert "Incomplete execution or score submission; workload differs" in row
    assert "×" not in row
    assert "Time limit reached" not in html
    assert "A completed model turn alone does not establish a complete score" in html


def test_invalid_notes_and_missing_usage_are_not_zero_accuracy_or_zero_cost(tmp_path):
    manifest, audit = fixture_bundle(tmp_path)
    report = fixture_report(tmp_path, "terra", "aviador_s8")
    for lane in report["stages"].values():
        lane["submission_validation"]["headline_metrics_valid"] = False
        lane["metrics"]["summary"].update(note_f1=0, rest_f1=0, exact_measures=0)
        lane["runtime"]["usage"] = None
    report["gate"].update(advance_to_rumichaca=False, independent_review_made_gate_pass=False)
    update_report(tmp_path, manifest, audit, "terra", "aviador_s8", report)
    html = render_dashboard(save_bundle(tmp_path, manifest, audit), tmp_path / "index.html")
    assert "Unscorable" in html and "No accuracy claim" in html
    assert ">0.0%<" not in html and "0 / 7" not in html
    assert "Unmeasured" in html and "$0" not in html
    assert "Rumichaca · whole-score task" in html  # No inherited Aviador advancement gate.


def test_missing_report_shows_pinned_failure_evidence_without_fabricated_metrics(tmp_path):
    manifest, audit = fixture_bundle(tmp_path)
    entry = manifest["systems"][1]
    entry["reports"]["rumichaca_full"] = None
    evidence = write_json(tmp_path / "failed_stage.json", {"status": "failed"})
    entry["outcomes"] = {
        "rumichaca_full": {"status": "failed", "reason": "No saved draft", "evidence": evidence}
    }
    audit["report_sha256"]["terra"]["rumichaca_full"] = None
    html = render_dashboard(save_bundle(tmp_path, manifest, audit), tmp_path / "index.html")
    assert "No saved draft" in html and "Execution evidence" in html
    assert "No graded report" in html and "Unavailable report" in html
    assert "Execution failed" in html and "No graded report; no accuracy claim" in html
    assert "Missing or interrupted execution is not an accuracy result" in html
    assert "Matching conditions not fully verified" in html


def test_partial_usage_records_have_known_sum_coverage_and_no_numeric_ratio(tmp_path):
    manifest, audit = fixture_bundle(tmp_path)
    report = fixture_report(tmp_path, "terra", "rumichaca_full")
    report["stages"]["initial"]["runtime"]["usage"] = [
        {"input_tokens": 600, "cached_input_tokens": 300, "output_tokens": 50},
        None,
    ]
    update_report(tmp_path, manifest, audit, "terra", "rumichaca_full", report)
    html = render_dashboard(save_bundle(tmp_path, manifest, audit), tmp_path / "index.html")
    assert "Known sum: 1 / 2 records" in html
    ratio = html.split('class="ratio"', 1)[1]
    assert "Unmeasured" in ratio


def test_audit_text_is_escaped_and_model_effort_must_match_report(tmp_path):
    manifest, audit = fixture_bundle(tmp_path)
    audit["checks"][0]["basis"] = "<script>bad</script>"
    path = save_bundle(tmp_path, manifest, audit)
    html = render_dashboard(path, tmp_path / "index.html")
    assert "&lt;script&gt;bad&lt;/script&gt;" in html and "<script>" not in html
    manifest["systems"][1]["reasoning_effort"] = "low"
    with pytest.raises(ValueError, match="model/effort"):
        render_dashboard(save_bundle(tmp_path, manifest, audit), tmp_path / "index.html")


@pytest.mark.parametrize("absolute", [False, True])
def test_historical_dashboard_becomes_a_relative_local_link(tmp_path, absolute):
    manifest, audit = fixture_bundle(tmp_path)
    relative = "archive/tool-free 20261028.html"
    manifest["historical_dashboard"] = str(tmp_path / relative) if absolute else relative
    html = render_dashboard(save_bundle(tmp_path, manifest, audit), tmp_path / "index.html")
    assert 'href="archive/tool-free%2020261028.html"' in html
    assert "Historical tool-free comparison" in html
    assert "Different workflow; results remain separate" in html
    assert "file://" not in html and str(tmp_path) not in html


@pytest.mark.parametrize("path", ["https://example.test/history", "file:///private/history"])
def test_historical_dashboard_rejects_non_filesystem_paths(tmp_path, path):
    manifest, audit = fixture_bundle(tmp_path)
    manifest["historical_dashboard"] = path
    with pytest.raises(ValueError, match="local filesystem paths"):
        render_dashboard(save_bundle(tmp_path, manifest, audit), tmp_path / "index.html")
