"""Synthetic presentation checks; no score contents, references or model calls."""

import hashlib
import json

import pytest
from test_astra_reproduction_dashboard import report_fixture

from scripts.build_astra_transport_comparison_dashboard import render_dashboard


def _manifest(tmp_path, *, api=True, change=None):
    systems = {}
    for name in ("subscription", "api"):
        report = report_fixture(tmp_path)
        if change:
            change(name, report)
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(report))
        systems[name] = {
            "report": (
                {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                if name != "api" or api
                else None
            )
        }
    manifest = {
        "schema_version": 1,
        "kind": "astra_transport_comparison_manifest",
        "comparison_id": "synthetic_transport_test",
        "target_id": "aviador_s8",
        "systems": systems,
        "protocol_differences": ["Native functions replace the CLI code-mode wrapper."],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_four_rows_preserve_stage_metrics_without_opening_source_files(tmp_path):
    path = _manifest(tmp_path)
    html = render_dashboard(path, tmp_path / "index.html")
    quality = html.split('<table class="quality">')[1].split("</table>")[0]
    assert quality.count("<tr>") == 5  # Header plus two stages for each transport.
    assert quality.count("91.0%") == 2 and quality.count("98.0%") == 2
    assert quality.count("5 / 7") == 2 and quality.count("6 / 7") == 2
    assert "SYNTHETIC CONTROL" in html
    assert "No quality scores are pooled" in html
    assert "Billed cost unknown" in html and "$0" not in html
    assert "No per-run monetary cost measured" in html
    assert "<script" not in html and "https://" not in html
    # report_fixture supplies nonexistent score/reference paths; rendering still succeeds.
    assert not (tmp_path / "evaluation_plan.json").exists()


def test_missing_api_report_is_pending_with_no_zero_metrics(tmp_path):
    html = render_dashboard(_manifest(tmp_path, api=False), tmp_path / "index.html")
    quality = html.split('<table class="quality">')[1].split("</table>")[0]
    assert quality.count("Pending") == 2
    assert quality.count("No graded report") == 2
    assert quality.count("91.0%") == 1
    assert ">0.0%<" not in quality and "$0" not in html


def test_partial_runtime_and_unscorable_submission_are_not_complete(tmp_path):
    def change(name, report):
        if name == "api":
            lane = report["stages"]["initial"]
            lane["runtime"].update(status="partial", timed_out=True)
            report["stages"]["reviewed"]["runtime"]["returncode"] = 1
            report["stages"]["reviewed"]["submission_validation"]["headline_metrics_valid"] = False
            report["gate"].update(
                advance_to_rumichaca=False, independent_review_made_gate_pass=False
            )

    html = render_dashboard(_manifest(tmp_path, change=change), tmp_path / "index.html")
    quality = html.split('<table class="quality">')[1].split("</table>")[0]
    assert quality.count("Partial") == 2
    assert quality.count("Checkpoint metrics only") == 2
    assert "Time limit reached" in quality and "Unscorable" in quality


def test_api_estimate_is_separate_from_unknown_bill_and_subscription(tmp_path):
    def change(name, report):
        for lane in report["stages"].values():
            lane["runtime"]["billing"] = {
                "cost_estimate_usd": "1.25",
                "cost_usd": None,
                "cost_basis": "conservative_estimate_not_invoice",
                "budget_usd": "5.00",
                "vendor_extra": "preserve <raw> detail",
            }
            lane["runtime"]["usage"] = [
                {
                    "input_tokens": 1000,
                    "cached_input_tokens": 800,
                    "output_tokens": 100,
                    "reasoning_output_tokens": 60,
                },
                None,
            ]

    html = render_dashboard(_manifest(tmp_path, change=change), tmp_path / "index.html")
    resources = html.split('<table class="resources">')[1].split("</table>")[0]
    assert resources.count("Conservative estimated cap: $1.2500") == 2
    assert resources.count("Billed cost unknown") == 2
    assert "Estimate, not an invoice or hard spending limit" in resources
    assert "1,000 input · 100 output" in resources and "1,800 input" not in resources
    assert "60 reasoning output (subset); recorded for 1 / 2" in resources
    assert "preserve &lt;raw&gt; detail" in html


def test_pin_mismatch_and_remote_report_path_fail(tmp_path):
    path = _manifest(tmp_path)
    (tmp_path / "api.json").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        render_dashboard(path, tmp_path / "index.html")
    manifest = json.loads(path.read_text())
    manifest["systems"]["api"]["report"]["path"] = "https://example.com/report.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="local filesystem"):
        render_dashboard(path, tmp_path / "index.html")


def test_responses_usage_details_remain_subsets(tmp_path):
    def change(name, report):
        if name == "api":
            for lane in report["stages"].values():
                lane["runtime"]["usage"] = {
                    "input_tokens": 1000,
                    "input_tokens_details": {"cached_tokens": 800},
                    "output_tokens": 100,
                    "output_tokens_details": {"reasoning_tokens": 60},
                }

    html = render_dashboard(_manifest(tmp_path, change=change), tmp_path / "index.html")
    assert "800 cached input (subset, not additional)" in html
    assert "60 reasoning output (subset)" in html
    assert "1,000 input · 100 output" in html and "1,800 input" not in html


@pytest.mark.parametrize("top_level", [False, True])
def test_cost_estimate_fallback_preserves_unknown_invoice(tmp_path, top_level):
    def change(name, report):
        if name == "api":
            for lane in report["stages"].values():
                runtime = lane["runtime"]
                runtime["billing"] = {
                    "charged_or_reserved_usd": "1.25",
                    "cost_basis": "conservative_estimate_not_invoice",
                }
                if top_level:
                    runtime.update(cost_estimate_usd="1.50", cost_usd=None)

    html = render_dashboard(_manifest(tmp_path, change=change), tmp_path / "index.html")
    assert f'Conservative estimated cap: ${"1.5000" if top_level else "1.2500"}' in html
    assert "Billed cost unknown" in html and "Recorded billed cost:" not in html


def test_transport_identity_mismatch_fails_and_labels_are_escaped(tmp_path):
    def change(name, report):
        if name == "api":
            report["model"] = "different-model"

    path = _manifest(tmp_path, change=change)
    with pytest.raises(ValueError, match="same requested model"):
        render_dashboard(path, tmp_path / "index.html")
    path = _manifest(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["systems"]["api"]["label"] = "<img src=x>"
    manifest["status"] = "Running <pilot>"
    path.write_text(json.dumps(manifest))
    html = render_dashboard(path, tmp_path / "index.html")
    assert "&lt;img src=x&gt;" in html and "<img " not in html
    assert "Running &lt;pilot&gt;" in html
