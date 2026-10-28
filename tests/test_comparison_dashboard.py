"""Checks for the offline complete-score comparison view."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from score2abc.comparison_dashboard import render_comparison_dashboard


def _report() -> dict:
    return {
        "schema_version": 1,
        "evidence_kind": "actual",
        "created_at": "2026-10-28T12:00:00Z",
        "frozen_manifest_sha256": "a" * 64,
        "labels_sha256": "b" * 64,
        "models": [
            {"id": "astra", "label": "GPT-6 Astra"},
            {"id": "terra", "label": "GPT-6 Terra"},
        ],
        "cases": [
            {
                "slug": f"piece-{i}",
                "title": f"Piece {i}",
                "results": {
                    model: {"status": "not_run", "metrics": {"edition_note_f1": None}}
                    for model in ("astra", "terra")
                },
            }
            for i in range(6)
        ],
    }


def _render(tmp_path: Path, report: dict) -> str:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return render_comparison_dashboard(path, tmp_path / "index.html")


def test_planned_comparison_shows_all_unrun_pairs_without_scores(tmp_path: Path) -> None:
    html = _render(tmp_path, _report())
    assert "0/6" in html
    assert "<h1>Astra vs. Terra</h1>" in html
    assert "28 Oct 2026, 12:00 UTC" in html
    assert "bars aligned and meter validity" in html
    assert "12</strong> not run" in html
    assert "Insufficient measured evidence to choose a model" in html
    assert "— means unmeasured" in html
    assert html.count('class="status not_run"') == 12
    assert '<span class="metric">0%' not in html


def test_partial_failed_and_synthetic_results_cannot_claim_winner(tmp_path: Path) -> None:
    report = _report()
    report["evidence_kind"] = "synthetic_demo"
    report["frozen_manifest_sha256"] = None
    report["labels_sha256"] = None
    report["models"] = [
        {"id": "control", "label": "Unchanged fixture"},
        {"id": "mutated", "label": "Injected errors"},
    ]
    for case in report["cases"]:
        case["results"] = {
            "control": case["results"]["astra"],
            "mutated": case["results"]["terra"],
        }
    report["cases"][0]["results"]["control"] = {
        "status": "complete",
        "metrics": {
            "edition_note_f1": 0.75,
            "source_note_f1": None,
            "bar_coverage": 1.0,
            "meter_valid_rate": 0.5,
            "cost_usd": None,
        },
        "metric_status": {"source_note_f1": "pending_source_labels"},
        "mismatches": [{"measure": 3, "kind": "pitch", "detail": "one note differs"}],
    }
    report["cases"][0]["results"]["mutated"] = {
        "status": "failed",
        "metrics": {},
        "error": "Timed out",
    }
    report["cases"][1]["results"]["control"] = {"status": "partial", "metrics": {}}
    html = _render(tmp_path, report)
    assert "Synthetic grader demonstration; no model comparison result" in html
    assert "<h1>Does the grader catch mistakes?</h1>" in html
    assert "Synthetic demonstration data" in html
    assert "GPT-6 Astra" not in html
    assert "GPT-6 Terra" not in html
    assert "one note differs" in html
    assert "Timed out" in html
    assert "75%" in html
    assert "0.00 USD" not in html
    assert "0/6" in html
    report["summary"] = {"winner": "control"}
    with pytest.raises(ValueError, match="winner requires"):
        _render(tmp_path, report)


def test_unrun_measures_and_invalid_rates_are_rejected(tmp_path: Path) -> None:
    report = _report()
    report["cases"][0]["results"]["astra"]["metrics"]["edition_note_f1"] = 0.6
    with pytest.raises(ValueError, match="unrun model"):
        _render(tmp_path, report)
    report["cases"][0]["results"]["astra"]["status"] = "complete"
    report["cases"][0]["results"]["astra"]["metrics"]["edition_note_f1"] = 1.2
    with pytest.raises(ValueError, match="rate exceeds one"):
        _render(tmp_path, report)


def test_report_identity_and_case_mismatch_are_rejected(tmp_path: Path) -> None:
    report = _report()
    report["labels_sha256"] = "short"
    with pytest.raises(ValueError, match="labels_sha256"):
        _render(tmp_path, report)
    report = _report()
    report["cases"][1]["slug"] = report["cases"][0]["slug"]
    with pytest.raises(ValueError, match="slugs must be distinct"):
        _render(tmp_path, report)
    report = _report()
    report["cases"][0]["results"]["sol"] = report["cases"][0]["results"].pop("terra")
    with pytest.raises(ValueError, match="results must match"):
        _render(tmp_path, report)


def test_escapes_report_text_and_rejects_remote_links(tmp_path: Path) -> None:
    report = _report()
    report["cases"][0]["title"] = '<img src=x onerror="alert(1)">'
    report["cases"][0]["results"]["astra"] = {
        "status": "complete",
        "metrics": {},
        "mismatches": [{"measure": 1, "kind": "pitch", "detail": "<script>alert(1)</script>"}],
    }
    html = _render(tmp_path, report)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<img src=x" not in html
    report["cases"][0]["results"]["astra"]["links"] = {"candidate_abc": "https://example.com/x"}
    with pytest.raises(ValueError, match="local path"):
        _render(tmp_path, report)


def test_absolute_local_evidence_link_is_encoded(tmp_path: Path) -> None:
    report = _report()
    source = tmp_path / "source score.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    report["cases"][0]["results"]["astra"]["links"] = {
        "source_pdf": str(source),
        "candidate_abc": None,
    }
    html = _render(tmp_path, report)
    assert 'href="source%20score.pdf"' in html


def test_actual_report_links_separate_demo_only_when_present(tmp_path: Path) -> None:
    report = _report()
    demo = tmp_path / "grader_demo" / "index.html"
    demo.parent.mkdir()
    demo.write_text("<!doctype html>", encoding="utf-8")
    assert "separate synthetic demonstration" not in _render(tmp_path, report)
    report["grader_demo_path"] = str(demo)
    html = _render(tmp_path, report)
    assert 'href="grader_demo/index.html"' in html
    assert "Open the separate synthetic demonstration" in html
    report["grader_demo_path"] = "https://example.com/demo"
    with pytest.raises(ValueError, match="local path"):
        _render(tmp_path, report)


def test_summary_includes_measured_partial_run(tmp_path: Path) -> None:
    report = _report()
    report["cases"][0]["results"]["astra"] = {
        "status": "complete",
        "metrics": {"edition_note_f1": 1.0},
    }
    report["cases"][1]["results"]["astra"] = {
        "status": "partial",
        "metrics": {"edition_note_f1": 0.0},
    }
    html = _render(tmp_path, report)
    assert "50% <small>(2/6 pieces)</small>" in html
    assert "including partial outcomes; missing metrics are excluded" in html
    assert "Insufficient measured evidence" in html


def test_reference_decisions_are_separate_and_escaped(tmp_path: Path) -> None:
    report = _report()
    report["cases"][0]["reference_decisions"] = [
        {
            "id": "AM-01",
            "category": "editorial_reconstruction",
            "source_location": {"system": 2, "measure": 4},
            "adopted_choice": "<script>alert(1)</script>",
            "status": "supported_by_review_notes",
            "evidence_origin": "human_musical_judgment",
            "evidence": [{"quote": "faint dot confirmed"}],
        }
    ]
    html = _render(tmp_path, report)
    assert "Reference editorial choices (not model output)" in html
    assert "AM-01 · S2 M4 · editorial reconstruction" in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html
    assert "Review evidence: faint dot confirmed" in html
    report["cases"][0]["reference_decisions"][0].update(
        source_location=None, edition_location={"bar": 45}
    )
    assert "AM-01 · Edition bar 45" in _render(tmp_path, report)
    report["cases"][0]["reference_decisions"][0].update(source_location={"bar_range": [47, 62]})
    assert "AM-01 · Source bars 47–62" in _render(tmp_path, report)


def _versioned_report() -> dict:
    report = _report()
    report["schema_version"] = 2
    report["versions"] = {
        name: {"id": f"{name}-v2", "sha256": "a" * 64}
        for name in ("dataset", "labels", "grader", "harness", "config")
    }
    return report


def test_arbitrary_configs_and_piece_counts_show_exact_settings(tmp_path: Path) -> None:
    report = _versioned_report()
    report["cases"] = report["cases"][:2]
    report["models"].append(
        {
            "id": "luna-medium",
            "label": "Luna medium",
            "model": "example-luna",
            "reasoning_effort": "medium",
            "provider": "example-provider",
            "harness_id": "pdf-tools-v2",
        }
    )
    for case in report["cases"]:
        case["results"]["luna-medium"] = {"status": "not_run", "metrics": {}}
    html = _render(tmp_path, report)
    assert "Compare transcription systems" in html
    assert "0/2" in html
    assert "0/6" in html
    assert "6</strong> not run" in html
    assert "example-luna · medium · example-provider · Harness: pdf-tools-v2" in html
    assert "labels-v2" in html
    assert "grader-v2" in html
    assert html.count('class="status not_run"') == 6
    assert "/12" not in html


def test_version_metadata_and_duplicate_configs_fail_closed(tmp_path: Path) -> None:
    report = _versioned_report()
    del report["versions"]["grader"]
    with pytest.raises(ValueError, match="version identities"):
        _render(tmp_path, report)
    report = _versioned_report()
    report["models"][1]["id"] = report["models"][0]["id"]
    with pytest.raises(ValueError, match="ids must be distinct"):
        _render(tmp_path, report)


def test_paired_deltas_exclude_unpaired_values_and_include_partial(tmp_path: Path) -> None:
    report = _versioned_report()
    for model, values in {"astra": [0.5, 1.0, 1.0], "terra": [0.7, 0.6]}.items():
        for index, value in enumerate(values):
            report["cases"][index]["results"][model] = {
                "status": "partial",
                "metrics": {"edition_note_f1": value},
            }
    html = _render(tmp_path, report)
    assert "Edition: -10.0 points (2/6 paired pieces)" in html
    assert "partial outcomes remain included" in html
    assert "Source:" not in html


def test_single_config_cannot_be_declared_winner(tmp_path: Path) -> None:
    report = _report()
    report["models"] = report["models"][:1]
    for case in report["cases"]:
        del case["results"]["terra"]
    report["summary"] = {"winner": "astra", "basis": "Only configuration"}
    with pytest.raises(ValueError, match="winner requires"):
        _render(tmp_path, report)


def test_source_score_displays_partial_audit_coverage(tmp_path: Path) -> None:
    report = _versioned_report()
    report["cases"][0]["results"]["astra"] = {
        "status": "partial",
        "metrics": {"source_note_f1": 0.8},
        "coverage": {"source_scored_measures": 7, "reference_measures": 48},
    }
    html = _render(tmp_path, report)
    assert "7/48 reference bars audited" in html
    assert "model-assisted source audit is not a new musician approval" in html


def test_token_usage_keeps_missing_attempts_and_does_not_double_count_cached(tmp_path: Path):
    report = _versioned_report()
    report["cases"][0]["results"]["astra"] = {
        "status": "partial",
        "metrics": {},
        "usage": [{"input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 80}, None],
    }
    html = _render(tmp_path, report)
    assert "120 <small>(1/2 recorded attempts)</small>" in html
    assert "0.00 USD" not in html
    report["cases"][0]["results"]["astra"]["usage"][0]["input_tokens"] = -1
    with pytest.raises(ValueError, match="invalid input_tokens"):
        _render(tmp_path, report)


def test_fully_paired_source_finding_is_scoped_and_keeps_editorial_pending(tmp_path: Path):
    report = _versioned_report()
    for case in report["cases"]:
        for model, value in (("astra", 0.8), ("terra", 0.6)):
            case["results"][model] = {
                "status": "partial",
                "metrics": {"source_note_f1": value, "editorial_acceptance": None},
                "coverage": {"source_scored_measures": 5, "source_expected_noteheads": 20},
            }
    html = _render(tmp_path, report)
    assert "GPT-6 Astra has the highest agreement on the audited notes" in html
    assert "Mean note F1 across 6 paired pieces: GPT-6 Astra: 80%; GPT-6 Terra: 60%" in html
    assert "Editorial acceptance and run-to-run reliability remain separate questions" in html
    report["cases"][0]["results"]["terra"]["metrics"]["source_note_f1"] = None
    assert "Insufficient measured evidence" in _render(tmp_path, report)
