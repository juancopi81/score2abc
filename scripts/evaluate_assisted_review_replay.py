"""Grade one sealed Aviador reviewer replay and render a local comparison.

No inference. Reuses the historical scorer, with the old author explicitly reused.
"""

# ruff: noqa: E501  # Keep the standalone HTML beside its presentation code.

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score2abc.score_eval_abc import normalize_abc  # noqa: E402
from scripts.build_astra_reproduction_dashboard import CSS, _elapsed, text  # noqa: E402
from scripts.experiments import evaluate_astra_reproduction as grader  # noqa: E402
from scripts.experiments import run_astra_baseline_reproduction as runner  # noqa: E402


def delta(before: dict, after: dict) -> dict:
    def index(lane):
        return {r["identity"]["system_measure_index"]: r for r in lane["metrics"]["results"]}

    old, new = index(before), index(after)
    if set(old) != set(range(1, 8)) or set(new) != set(old):
        raise ValueError("Both readers must be scored over the seven target bars")
    corrected = [i for i in old if not old[i]["exact_measure"] and new[i]["exact_measure"]]
    regressed = [i for i in old if old[i]["exact_measure"] and not new[i]["exact_measure"]]
    old_events = {r["identity"]["system_measure_index"]: r for r in before["adapted_events"]}
    new_events = {r["identity"]["system_measure_index"]: r for r in after["adapted_events"]}
    changed = [i for i in old if old_events.get(i) != new_events.get(i)]
    notes = new_events.get(7, {}).get("notes", [])
    sharp_fixed = len(notes) == 6 and notes[2]["pitch_midi"] == notes[5]["pitch_midi"] == 66
    return {
        "sharp_fixed": sharp_fixed,
        "corrected_bars": corrected,
        "regressed_previously_exact_bars": regressed,
        "changed_bars": changed,
        "all_scored_events_match": all(r["exact_measure"] for r in new.values()),
    }


def resources(lane: dict) -> dict:
    runtime = lane["runtime"]
    usage = runtime.get("usage") or {}
    total = usage.get("input_tokens")
    cached = usage.get("input_tokens_details", {}).get("cached_tokens")
    return {
        "seconds": runtime.get("elapsed_seconds"),
        "input_tokens": total,
        "cached_input_tokens": cached,
        "cache_fraction": cached / total if total and cached is not None else None,
        "output_tokens": usage.get("output_tokens"),
        "conservative_estimate_usd": runtime.get("cost_estimate_usd"),
        "actual_billed_usd": runtime.get("cost_usd"),
    }


def evaluate(root: Path) -> dict:
    root = root.resolve()
    destination = root / "review_evaluation.json"
    if destination.exists():
        raise FileExistsError("Reviewer evaluation is create-once")
    proposal = grader.read_json(root / "proposed_experiment.json")
    contract = grader.read_json(root / "execution_contract.json")
    amendment = grader.read_json(root / "launcher_permission_amendment.json")
    before_path = ROOT / proposal["baseline_reviewed_run"]
    author_path = ROOT / proposal["initial_run"]
    after_path = root / amendment["new_output"]
    packet = root / "aviador_s8/reviewed_packet"
    for path, kind in [(before_path, "stage"), (author_path, "stage"), (after_path, "stage")]:
        runner.verify_seal(path, f"astra_reproduction_{kind}")
    runner.verify_seal(packet, "astra_reproduction_input")
    for path, expected in [
        (before_path / "seal.json", proposal["baseline_reviewed_seal_sha256"]),
        (author_path / "seal.json", proposal["initial_seal_sha256"]),
        (packet / "seal.json", contract["input_seal_sha256"]),
        (root / proposal["task"], proposal["task_sha256"]),
    ]:
        if grader.sha(path) != expected:
            raise ValueError("Replay input or predecessor pin changed")
    provenance = grader.read_json(after_path / "stage_provenance.json")
    if any(
        provenance.get(k) != v
        for k, v in {
            "input_seal_sha256": contract["input_seal_sha256"],
            "initial_abc_sha256": proposal["initial_abc_sha256"],
            "source_pdf_sha256": proposal["source_pdf_sha256"],
            "stage": "reviewed",
            "model": proposal["model"],
            "reasoning_effort": proposal["reasoning_effort"],
            "deadline_seconds": 1200,
            "reference_access": False,
            "independent_reader": True,
            "status": "complete",
            "complete": True,
            "returncode": 0,
        }.items()
    ):
        raise ValueError("No completed, matching independent review to grade")
    baseline_root = author_path.parent.parent
    plan, seal, stages, pins = grader.verify_submission(
        baseline_root / "evaluation_plan.json", author_path.parent / "submission_seal.json"
    )
    baseline_report = grader.read_json(author_path.parent / "evaluation/report.json")
    for record in baseline_report["implementations"].values():
        if isinstance(record, dict) and "path" in record:
            grader.verify_pin(ROOT, record, "unchanged grading implementation")
    stages["reviewed"] = {
        "abc": after_path / "score.abc",
        "decisions_data": grader.read_json(after_path / "decisions.json"),
        "provenance_data": provenance,
    }
    for stage in stages.values():
        stage["normalized"] = normalize_abc(stage["abc"].read_text())
    measured = grader._aviador(plan, seal, stages, pins, None)
    if measured["stages"]["initial"]["metrics"] != baseline_report["stages"]["initial"]["metrics"]:
        raise ValueError("Reused author no longer reproduces original scoring")
    before, after = baseline_report["stages"]["reviewed"], measured["stages"]["reviewed"]
    protected = grader.read_json(root / "live_review_preservation.json")
    if any(grader.sha(ROOT / path) != expected for path, expected in protected.items()):
        raise ValueError("Live human review changed during the experiment")
    pins.extend(
        grader.pin(p)
        for p in [
            root / "proposed_experiment.json",
            root / "execution_contract.json",
            root / "launcher_permission_amendment.json",
            after_path / "seal.json",
            after_path / "score.abc",
            after_path / "decisions.json",
            after_path / "stage_provenance.json",
            author_path.parent / "evaluation/report.json",
            Path(__file__),
        ]
    )
    for pin in pins:
        grader.verify_pin(ROOT, pin, "final evidence check")
    report = {
        "schema_version": 1,
        "kind": "assisted_reviewer_only_regression",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": proposal["model"],
        "reasoning_effort": proposal["reasoning_effort"],
        "review_instructions_version": proposal["review_instructions_version"],
        "review_instructions_sha256": proposal["review_instructions_sha256"],
        "source_pdf_sha256": proposal["source_pdf_sha256"],
        "author_reused": True,
        "paid_model_attempts": 1,
        "pre_request_launcher_failures": 1,
        "before": before,
        "after": after,
        "delta": delta(before, after),
        "resources": {"before": resources(before), "after": resources(after)},
        "unsupported_scope": measured["unsupported_scope"],
        "metric_notes": measured["metric_notes"],
        "live_reviews_unchanged": len(protected),
        "pins": pins,
        "limitations": [
            "Consumed reviewer-only regression; author cost is excluded from this new run.",
            "One stochastic attempt does not establish causal or general improvement.",
            "No automatic advancement or additional score is authorized by these metrics.",
        ],
    }
    runner.write(destination, report)
    return report


def render(report: dict) -> str:
    change = report["delta"]
    verdict = (
        "Sharp corrected; all seven bars match."
        if change["sharp_fixed"] and change["all_scored_events_match"]
        else "The replay did not resolve the passage without errors."
    )
    rows = []
    for label, key in [
        ("API · review instructions v1", "before"),
        ("API · review instructions v2", "after"),
    ]:
        lane, resource = report[key], report["resources"][key]
        summary = lane["metrics"]["summary"]
        cache = resource["cache_fraction"]
        cache_text = "Unknown" if cache is None else f"{cache:.1%}"
        estimate = resource["conservative_estimate_usd"]
        estimate_text = "Unknown" if estimate is None else f"${float(estimate):.2f} estimate"
        cells = [
            label,
            f'{summary["exact_measures"]}/7',
            f'{summary["note_f1"]:.2%}',
            _elapsed(lane["runtime"]),
            f'{resource["output_tokens"]:,}',
            cache_text,
            estimate_text,
        ]
        rows.append("<tr>" + "".join(f"<td>{text(v)}</td>" for v in cells) + "</tr>")
    bars = []
    for old, new in zip(
        report["before"]["metrics"]["results"], report["after"]["metrics"]["results"], strict=True
    ):
        i = old["identity"]["system_measure_index"]
        cells = [
            i,
            "Exact" if old["exact_measure"] else "Mismatch",
            "Exact" if new["exact_measure"] else "Mismatch",
            "Changed" if i in change["changed_bars"] else "Unchanged",
        ]
        bars.append("<tr>" + "".join(f"<td>{text(v)}</td>" for v in cells) + "</tr>")
    decisions = "".join(
        f"<li>{text(d.get('location', ''))}: {text(d.get('adopted_choice', ''))}<p>{text(d.get('reason', ''))}</p><small>Uncertainty: {text(d.get('uncertainty', 'Not stated'))}</small></li>"
        for d in report["after"]["decisions"]
    )
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Aviador · review instructions v2</title><style>{CSS}</style><body>
<header class="mast"><div class="wrap"><p class="eyebrow">Aviador · 7 bars · Astra xhigh · API reviewer only</p><h1>{text(verdict)}</h1><p class="lede">Same saved author draft, source PDF, tools and 20-minute limit. Only the review instructions changed. No new author run.</p></div></header>
<main class="wrap"><div class="cards"><div class="card"><strong>{report['after']['metrics']['summary']['exact_measures']}/7 exact bars</strong><small>Notes, timing and rests</small></div><div class="card"><strong>{len(change['regressed_previously_exact_bars'])} previously correct bars regressed</strong><small>All seven bars checked</small></div><div class="card"><strong>{'F-sharp confirmed' if change['sharp_fixed'] else 'Sharp unresolved'}</strong><small>Third and sixth notes of the last bar</small></div></div>
<section class="section"><h2>Did the new review help?</h2><div class="table-scroll"><table><thead><tr><th>Reviewer</th><th>Exact bars</th><th>Strict note F1</th><th>Time</th><th>Output tokens</th><th>Cached input</th><th>Cost</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p class="scope">Review stage only. Dollar figures are conservative estimates, not observed charges. Actual billing for this new run is unverified. The earlier subscription review reached 7/7 under v1 instructions; see the preserved comparison below.</p></section>
<section class="section"><h2>Check every bar</h2><div class="table-scroll"><table><thead><tr><th>Bar in system 8</th><th>Old review</th><th>New review</th><th>Scored events</th></tr></thead><tbody>{''.join(bars)}</tbody></table></div></section>
<section class="section"><h2>Caching and future scores</h2><p>Repeated identical prefixes can reuse cached input. Changed instructions and different PDF content limit reuse; output generation is still charged. For future batches, keep shared instructions and tools stable, followed by each score's source and task, and measure actual cache use. No cache settings changed in this experiment.</p><p><a href="https://developers.openai.com/api/docs/guides/prompt-caching">Official caching guidance</a></p></section>
<div class="notice">One known development passage, one fresh review. This does not establish improved accuracy on unseen scores. Ties, slurs, chord symbols, form and engraving remain outside these metrics. The historical 6/7 API result is preserved.</div>
<details class="details"><summary>Reviewer decisions and uncertainty</summary><ul>{decisions}</ul></details>
<details class="details"><summary>Version and execution evidence</summary><p>Instructions: {text(report['review_instructions_version'])}<br>SHA256: {text(report['review_instructions_sha256'])}</p><p>One local startup failed before any API request; preserved separately. One paid model review, no automatic retries. {report['live_reviews_unchanged']} live human review files are unchanged.</p></details>
<p class="links"><a href="review_evaluation.json">Full scored report</a><a href="../astra_api_fixed_output_20261029_v2/index.html">Previous API vs subscription dashboard</a><a href="aviador_s8/reviewed_api/score.abc">New reviewer ABC</a></p></main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = evaluate(args.root)
    (args.root / "index.html").write_text(render(report))
    print(json.dumps({"delta": report["delta"], "resources": report["resources"]}, indent=2))


if __name__ == "__main__":
    main()
