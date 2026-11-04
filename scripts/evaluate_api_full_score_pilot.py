"""Verify and grade the single Rumichaca API pair; render its local dashboard.

This offline adapter uses the existing complete-score and partial-source graders.
It does not invoke models or change the historical advancement protocol.
"""

# ruff: noqa: E501  # Standalone HTML is kept beside its renderer.

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score2abc.score_eval_abc import normalize_abc  # noqa: E402
from scripts.build_astra_reproduction_dashboard import CSS, _elapsed, _percent, text  # noqa: E402
from scripts.experiments import evaluate_astra_reproduction as grader  # noqa: E402
from scripts.experiments import run_astra_baseline_reproduction as runner  # noqa: E402


def validate_stage(
    provenance: dict, plan: dict, stage: str, packet_hash: str, author_hash: str | None
) -> None:
    expected = {
        "stage": stage,
        "auth_mode": "openai_api",
        "source_execution": "sandboxed_source_process",
        "model": plan["model"],
        "reasoning_effort": plan["reasoning_effort"],
        "workflow_id": plan["workflow_id"],
        "source_pdf_sha256": plan["target"]["source_pdf"]["sha256"],
        "input_seal_sha256": packet_hash,
        "deadline_seconds": plan["deadline_seconds"][stage],
        "reference_access": False,
        "independent_reader": stage == "reviewed",
        "initial_abc_sha256": author_hash,
        "status": "complete",
        "complete": True,
        "returncode": 0,
    }
    if any(provenance.get(k) != v for k, v in expected.items()):
        raise ValueError(f"{stage} is not a completed matching source-only stage")
    if datetime.fromisoformat(provenance["started_at"]) < datetime.fromisoformat(
        plan["created_at"]
    ):
        raise ValueError("Author and reviewer must be fresh after preregistration")


def resource_row(runtime: dict) -> dict:
    usage = runtime.get("usage") or {}
    input_tokens = usage.get("input_tokens")
    cached = usage.get("input_tokens_details", {}).get("cached_tokens")
    if cached is None:
        cached = usage.get("cached_input_tokens")  # Subscription receipt format.
    return {
        "seconds": runtime.get("elapsed_seconds"),
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "cache_fraction": cached / input_tokens if input_tokens and cached is not None else None,
        "output_tokens": usage.get("output_tokens"),
        "conservative_estimate_usd": runtime.get("cost_estimate_usd"),
        "actual_billed_usd": runtime.get("cost_usd"),
    }


def evaluate(root: Path) -> dict:
    root = root.resolve()
    output = root / "evaluation"
    if output.exists():
        raise FileExistsError("Evaluation is create-once")
    plan = grader.read_json(root / "evaluation_plan.json")
    contract = grader.read_json(root / "execution_contract.json")
    if (
        plan.get("kind") != "astra_api_full_score_pilot_plan"
        or plan.get("status") != "preregistered_before_new_author_inference"
        or plan.get("target_id") != "rumichaca_full"
        or plan.get("source_policy") != "supplied_pdf_only"
        or plan.get("review_instructions_version") != "assisted_review_v2"
        or plan.get("deadline_seconds") != {"initial": 3600, "reviewed": 1800}
        or plan.get("reference_feedback_given") is not False
        or contract.get("plan_sha256") != grader.sha(root / "evaluation_plan.json")
    ):
        raise ValueError("Unexpected full-score pilot contract")
    records = [
        grader.pin(root / "evaluation_plan.json"),
        grader.pin(root / "execution_contract.json"),
    ]
    for key in [
        "author_task",
        "review_task",
        "review_instructions",
        "subscription_baseline_report",
        "subscription_baseline_plan",
        "aviador_regression_report",
    ]:
        records.append(grader.pin(grader.verify_pin(ROOT, plan[key], key)))
    for entry in plan["implementation_snapshots"]:
        path = grader.verify_pin(ROOT, entry, "implementation snapshot")
        relative = path.relative_to(root / "setup/implementation")
        if grader.sha(ROOT / relative) != entry["sha256"]:
            raise ValueError(f"Implementation changed during execution: {relative}")
        records.append(grader.pin(path))
    target = root / "rumichaca_full"
    stages, artifacts = {}, []
    author_hash = None
    for stage, task_key in [("initial", "author_task"), ("reviewed", "review_task")]:
        folder = target / stage
        packet = target / f"{stage}_packet"
        for suffix, kind in [("", "stage"), ("_packet", "input"), ("_audit", "audit")]:
            sealed = target / (stage + suffix)
            runner.verify_seal(sealed, f"astra_reproduction_{kind}")
            artifacts.extend(grader.pin(p) for p in sorted(sealed.rglob("*")) if p.is_file())
        provenance = grader.read_json(folder / "stage_provenance.json")
        validate_stage(provenance, plan, stage, grader.sha(packet / "seal.json"), author_hash)
        metadata = grader.read_json(packet / "preparation.json")
        if metadata["prompt_template_sha256"] != plan[task_key]["sha256"]:
            raise ValueError("Prepared task does not match the versioned prompt")
        if stage == "initial":
            author_hash = grader.sha(folder / "score.abc")
        stages[stage] = {
            "abc": folder / "score.abc",
            "provenance_data": provenance,
            "decisions_data": grader.read_json(folder / "decisions.json"),
            "normalized": normalize_abc((folder / "score.abc").read_text()),
        }
    protected = grader.read_json(root / "live_review_preservation.json")
    if any(grader.sha(ROOT / path) != expected for path, expected in protected.items()):
        raise ValueError("Human review files changed during the run")
    # Seal both candidates and trusted input provenance before opening any reference ABC.
    seal_path = target / "pilot_submission_seal.json"
    if seal_path.exists():
        raise FileExistsError("Pilot submission seal is create-once")
    runner.write(
        seal_path,
        {
            "kind": "astra_api_full_score_pilot_submission_seal",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "plan_sha256": grader.sha(root / "evaluation_plan.json"),
            "both_stages_verified_before_reference_content": True,
            "reference_feedback_given": False,
            "artifacts": artifacts,
        },
    )
    records.extend([grader.pin(seal_path), *artifacts])
    measured = grader._rumichaca({"targets": {"rumichaca_full": plan["target"]}}, stages, records)
    baseline = grader.read_json(Path(plan["subscription_baseline_report"]["path"]))
    baseline_plan = grader.read_json(Path(plan["subscription_baseline_plan"]["path"]))
    for key in ["source_pdf", "frozen_reference", "labels"]:
        if plan["target"][key] != baseline_plan["targets"]["rumichaca_full"][key]:
            raise ValueError("Baseline source or grading target differs")
    for key in ["abc_normalizer", "abc_parser", "complete_score_grader", "source_grader"]:
        grader.verify_pin(ROOT, baseline["implementations"][key], "baseline grader")
    traces = {}
    for stage in stages:
        path = target / stage
        traces[stage] = {
            "generation_requests": len(list((path / "api").glob("request_*.json"))),
            "source_crops": len(list((path / "artifacts/inspections").glob("*.png"))),
            "draft_checkpoints": len(
                list((path / "artifacts/drafts").glob("draft_*/checkpoint.json"))
            ),
        }
    report = {
        "schema_version": 1,
        "kind": "astra_api_full_score_pilot_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": plan["experiment_id"],
        "model": plan["model"],
        "reasoning_effort": plan["reasoning_effort"],
        "review_instructions_version": plan["review_instructions_version"],
        "both_stages_verified_before_reference_content": True,
        **measured,
        "subscription_baseline": baseline["stages"],
        "resources": {
            stage: resource_row(lane["runtime"]) for stage, lane in measured["stages"].items()
        },
        "conservative_estimate_usd": str(
            sum(
                (Decimal(s["provenance_data"]["cost_estimate_usd"]) for s in stages.values()),
                Decimal(0),
            )
        ),
        "actual_billed_usd": None,
        "traces": traces,
        "live_reviews_unchanged": len(protected),
        "limitations": plan["limitations"],
        "pins": [*records, grader.pin(Path(__file__))],
    }
    for entry in report["pins"]:
        grader.verify_pin(ROOT, entry, "final artifact check")
    html = render(report)  # Validate presentation before publishing the report.
    output.mkdir()
    runner.write(output / "report.json", report)
    (root / "index.html").write_text(html)
    return report


def table(headers: list[str], rows: list[list[object]]) -> str:
    head = "".join(f"<th>{text(v)}</th>" for v in headers)
    body = "".join("<tr>" + "".join(f"<td>{text(v)}</td>" for v in row) + "</tr>" for row in rows)
    return f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def render(report: dict) -> str:
    quality, costs = [], []
    for label, lanes in [
        ("Subscription · review v1", report["subscription_baseline"]),
        ("API · review v2", report["stages"]),
    ]:
        for stage, lane in lanes.items():
            metric, coverage = lane["metrics"], lane["coverage"]
            route = lane["validation"].get("route_status", "unknown")
            performed = lane["validation"].get("counts", {}).get("performed_measures", "?")
            quality.append(
                [
                    label,
                    "Author" if stage == "initial" else "Independent review",
                    f'{coverage["candidate_measures"]}/{coverage["reference_measures"]}',
                    _percent(metric.get("edition_note_f1")),
                    _percent(metric.get("chord_f1")),
                    _percent(metric.get("source_note_f1")),
                    f'{coverage["source_scored_measures"]}/45',
                    f"{route}; {performed} bars; "
                    + ("edition match" if metric.get("form_exact_match") else "edition differs"),
                ]
            )
            resource = resource_row(lane["runtime"])
            cache = (
                "Unknown"
                if resource["cache_fraction"] is None
                else f'{resource["cache_fraction"]:.1%}'
            )
            estimate = resource["conservative_estimate_usd"]
            cost = "Not measured" if estimate is None else f"${float(estimate):.2f} estimate"
            costs.append(
                [
                    label,
                    stage,
                    _elapsed(lane["runtime"]),
                    resource["input_tokens"],
                    resource["output_tokens"],
                    cache,
                    cost,
                ]
            )
    final = report["stages"]["reviewed"]
    coverage, metric = final["coverage"], final["metrics"]
    body = table(
        [
            "Execution",
            "Stage",
            "Written bars",
            "Edition note F1",
            "Chord F1",
            "Audited-source note F1",
            "Source coverage",
            "Playback form",
        ],
        quality,
    )
    resource_table = table(
        ["Execution", "Stage", "Time", "Input tokens", "Output tokens", "Cached input", "Cost"],
        costs,
    )
    decisions = "".join(
        f'<li><strong>{text(d.get("location", ""))}</strong>: {text(d.get("adopted_choice", ""))}<p>{text(d.get("reason", ""))}</p><small>{text(d.get("uncertainty", ""))}</small></li>'
        for d in final["decisions"]
    )
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rumichaca · Astra API pilot</title><style>{CSS}.wrap{{max-width:1400px}}.table-scroll table{{min-width:1100px}}thead th:first-child{{width:auto}}</style><body>
<header class="mast"><div class="wrap"><p class="eyebrow">Rumichaca · complete-score API pilot · Astra xhigh</p><h1>Fresh author, independent v2 review</h1><p class="lede">Same supplied PDF, source tools and 60/30-minute limits. Both stages sealed before grading. The subscription baseline used v1 review instructions, so this is not a pure transport comparison.</p></div></header>
<main class="wrap"><div class="cards"><div class="card"><strong>{coverage['candidate_measures']}/{coverage['reference_measures']} written bars</strong><small>Review submission: {text(final['status'])}</small></div><div class="card"><strong>{_percent(metric.get('edition_note_f1'))} edition note F1</strong><small>Agreement with the complete musician-reviewed edition</small></div><div class="card"><strong>${float(report['conservative_estimate_usd']):.2f} estimated total</strong><small>Author + review; actual billing unverified</small></div></div>
<section class="section"><h2>Music: API versus the saved subscription baseline</h2>{body}<p class="scope">Note F1 compares pitch, onset and duration. Source labels cover nine pre-existing model-audited bars, not the entire PDF. The edition includes human editorial choices; a different route is not automatically a source-reading error. All headline values are separate metrics, not one combined accuracy score.</p></section>
<section class="section"><h2>Time, tokens and cost</h2>{resource_table}<p class="scope">Cached input is a subset of input tokens. Subscription USD cost is unknown. API dollar figures are conservative estimates, not invoices. One author and one reader; no automatic retries or additional scores.</p></section>
<section class="section"><h2>What still needs judgment</h2><p>Compare the complete written score and executed form separately. A resolved, finite playback route can still differ from the adopted edition. No model result grants human musical approval.</p><p>Meter-valid rate: {_percent(metric.get('meter_valid_rate'))}. Exact edition bars (including meter, ties, rests and chords): {coverage['exact_measures']}/{coverage['reference_measures']}.</p><details class="details"><summary>Reader decisions and remaining uncertainty</summary><ul>{decisions}</ul></details></section>
<section class="section"><h2>Preserved traces</h2><pre>{text(json.dumps(report['traces'], indent=2))}</pre><p>Prompts, crop/tool events, draft checkpoints, validation feedback, response receipts and usage are preserved locally. {report['live_reviews_unchanged']} live human review files remain unchanged.</p></section>
<div class="notice">One consumed development score. No claim of unseen-score accuracy or general API equivalence. Form, chords and incomplete coverage must be considered before scaling.</div>
<p class="links"><a href="evaluation/report.json">Full report and version pins</a><a href="rumichaca_full/initial/score.abc">Author ABC</a><a href="rumichaca_full/reviewed/score.abc">Reviewed ABC</a><a href="../astra_api_review_context_20261029_v3/index.html">Aviador reviewer-v2 regression</a><a href="../astra_workflow_20261028_v1/index.html">Original subscription baseline</a></p></main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = evaluate(args.root)
    print(
        json.dumps(
            {
                "stages": {
                    stage: {
                        "status": lane["status"],
                        "metrics": lane["metrics"],
                        "coverage": lane["coverage"],
                    }
                    for stage, lane in report["stages"].items()
                },
                "resources": report["resources"],
                "traces": report["traces"],
                "estimate_usd": report["conservative_estimate_usd"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
