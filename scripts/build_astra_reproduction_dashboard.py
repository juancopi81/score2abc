"""Render sealed Astra reproduction reports as a small, offline HTML dashboard.

This presentation layer reads reports only. It never loads reference answers,
grades submissions, invokes models, or changes a report's advancement decision.
"""

# ruff: noqa: E501  # Keep the standalone stylesheet and HTML next to the renderer.

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from html import escape
from pathlib import Path
from urllib.parse import quote, urlsplit

STAGES = {"initial": "Initial AI draft", "reviewed": "After independent AI review"}
KIND = "astra_assisted_workflow_reproduction_report"
EVIDENCE_KINDS = {"consumed_development_workflow_reproduction", "synthetic_control"}

CSS = """
:root{color-scheme:light;--ink:#153F61;--paper:#FAFCFF;--teal:#227C78;--gold:#9B681F;--line:#C9D8E1;--muted:#4E6271;--soft:#EDF4F6;--bad:#9B3535}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:#183547;font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}
a{color:var(--ink);text-underline-offset:3px}a:hover{color:var(--teal)}a:focus-visible,summary:focus-visible{outline:3px solid var(--gold);outline-offset:3px}
.wrap{max-width:1120px;margin:auto;padding:24px 28px}.mast{background:#F3F8FB;border-bottom:1px solid var(--line)}
h1,h2{font-family:Georgia,"Times New Roman",serif;font-weight:400;color:var(--ink);line-height:1.15}h1{font-size:clamp(2rem,4vw,2.7rem);margin:10px 0}h2{font-size:1.7rem;margin:0 0 8px}h3{font-size:1rem;color:var(--ink)}p{margin:0 0 12px}.eyebrow{font-size:.82rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}.lede{max-width:82ch;color:var(--muted)}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:20px 0 10px}.card{background:white;border:1px solid var(--line);border-top:4px solid var(--teal);padding:13px 16px}.card.pending{border-top-color:var(--gold)}.card strong{display:block;font-size:1.25rem;color:var(--ink)}.card small{display:block;color:var(--muted);margin-top:5px}
.notice{border-left:4px solid var(--gold);background:#FBF3E2;padding:12px 16px;margin:14px 0}.demo{font-weight:700}.verdict{font-size:1.1rem;color:var(--ink);margin:18px 0 8px}.quiet,small{color:var(--muted)}small{display:block}.section{margin:8px 0 32px}.section-head{border-bottom:2px solid var(--ink);padding-bottom:5px;margin-bottom:12px}.table-scroll{overflow-x:auto;border:1px solid var(--line);background:white}table{border-collapse:collapse;width:100%;min-width:620px;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:11px 15px;border-bottom:1px solid var(--line);vertical-align:top}thead th{background:#E6F0F4;color:var(--ink)}thead th:first-child{width:34%}tbody th{font-weight:500}tbody tr:last-child>*{border-bottom:0}tbody tr:nth-child(even){background:#F7FAFB}td strong{color:var(--ink)}.complete{color:var(--teal);font-weight:700}.partial{color:var(--gold);font-weight:700}.failed{color:var(--bad);font-weight:700}
.scope{font-size:.88rem;color:var(--muted);margin-top:10px;max-width:95ch}.links{display:flex;flex-wrap:wrap;gap:8px 18px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:25px}.details{border-top:1px solid var(--line);margin-top:14px;padding-top:10px}.details summary{cursor:pointer;color:var(--ink);font-weight:700}.details ul{padding-left:20px}.details li{margin:5px 0;overflow-wrap:anywhere}.details code{overflow-wrap:anywhere;font-size:.8rem}.pill{display:inline-block;border:1px solid var(--line);padding:3px 8px;margin:0 7px 6px 0;background:white}.metric{font-size:1.1rem}footer{border-top:1px solid var(--line);font-size:.84rem;color:var(--muted)}
@media(max-width:680px){.wrap{padding:20px 16px}.cards,.pair{grid-template-columns:1fr}.cards{gap:8px}.card strong{font-size:1.1rem}}
"""


def text(value: object) -> str:
    return escape(str(value), quote=True)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return float(value)


def _rate(value: object, label: str) -> float | None:
    if value is None:
        return None
    result = _number(value, label)
    if result > 1:
        raise ValueError(f"{label} must be between zero and one")
    return result


def _local_path(value: str) -> Path:
    if not isinstance(value, str) or not value or urlsplit(value).scheme or value.startswith("//"):
        raise ValueError("Evidence links must be local filesystem paths")
    return Path(value)


def _validate(report: dict, target: str) -> dict:
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != 1
        or report.get("kind") != KIND
        or report.get("target_id") != target
        or report.get("evidence_kind") not in EVIDENCE_KINDS
        or report.get("both_stages_verified_before_reference_content") is not True
        or set(report.get("stages", {})) != set(STAGES)
    ):
        raise ValueError(f"Expected a sealed {target} reproduction report")
    for name in ("workflow_id", "model", "reasoning_effort", "created_at"):
        if not isinstance(report.get(name), str) or not report[name]:
            raise ValueError(f"Report requires {name}")
    for stage, lane in report["stages"].items():
        runtime = lane.get("runtime", {})
        if runtime.get("status") not in {"complete", "partial", "failed"}:
            raise ValueError(f"{stage} requires an explicit runtime status")
        if runtime.get("elapsed_seconds") is not None:
            _number(runtime["elapsed_seconds"], "elapsed_seconds")
        metrics = lane["metrics"].get("summary", {}) if target == "aviador_s8" else lane["metrics"]
        for name in (
            "note_f1",
            "rest_f1",
            "source_note_f1",
            "edition_note_f1",
            "edition_exact_bar_rate",
            "chord_f1",
        ):
            _rate(metrics.get(name), name)
        if (
            target == "aviador_s8"
            and type(lane.get("submission_validation", {}).get("headline_metrics_valid"))
            is not bool
        ):
            raise ValueError("Aviador requires an explicit headline metric validity status")
    if target == "aviador_s8":
        gate = report.get("gate", {})
        for name in (
            "advance_to_rumichaca",
            "exact_historical_reproduction_initial_only",
            "independent_review_made_gate_pass",
        ):
            if type(gate.get(name)) is not bool:
                raise ValueError(f"Gate requires explicit {name}")
        if gate.get("threshold") != {"exact_measures_min": 6, "strict_note_f1_min": 0.95}:
            raise ValueError("Unexpected preregistered advancement threshold")
        if gate["independent_review_made_gate_pass"] and not gate["advance_to_rumichaca"]:
            raise ValueError("Review cannot make a blocked gate pass")
        if gate["advance_to_rumichaca"] and any(
            not lane["submission_validation"]["headline_metrics_valid"]
            for lane in report["stages"].values()
        ):
            raise ValueError("An unscorable stage cannot pass the advancement gate")
        initial = report["stages"]["initial"]["metrics"]["summary"]
        if gate["exact_historical_reproduction_initial_only"] and (
            initial.get("targets") != 7
            or initial.get("exact_measures") != 7
            or initial.get("note_f1") != 1
        ):
            raise ValueError("Exact reproduction flag contradicts initial draft metrics")
    for pin in report.get("pins", []):
        _local_path(pin["path"])
    return report


def _link(path: Path, output: Path, label: str) -> str:
    href = quote(os.path.relpath(path.resolve(), output.parent.resolve()), safe="/")
    return f'<a href="{text(href)}">{text(label)}</a>'


def _artifact_path(report: dict, record: dict) -> Path:
    raw = _local_path(record["path"])
    candidates = {
        str(_local_path(pin["path"]))
        for pin in report.get("pins", [])
        if pin.get("sha256") == record.get("sha256")
        and (Path(pin["path"]) == raw or Path(pin["path"]).parts[-len(raw.parts) :] == raw.parts)
    }
    if len(candidates) != 1:
        raise ValueError("Stage evidence must resolve to one pinned artifact path")
    resolved = Path(candidates.pop())
    if not resolved.is_absolute():
        raise ValueError("Report evidence pins must contain absolute paths")
    return resolved


def _plan_hash(report: dict) -> str:
    plans = {
        (_local_path(pin["path"]), pin["sha256"])
        for pin in report.get("pins", [])
        if Path(pin["path"]).name == "evaluation_plan.json"
    }
    predecessor = report.get("baseline_predecessor", {}).get("plan")
    if predecessor is not None:
        identity = (_artifact_path(report, predecessor), predecessor["sha256"])
        if identity not in plans:
            raise ValueError("Baseline predecessor must identify a pinned evaluation plan")
        plans.remove(identity)
    if len(plans) != 1:
        raise ValueError("Report must identify one evaluation plan")
    return plans.pop()[1]


def _percent(value: object) -> str:
    rate = _rate(value, "metric")
    return (
        '<span class="quiet">— <small>Unmeasured</small></span>'
        if rate is None
        else f'<strong class="metric">{rate:.1%}</strong>'
    )


def _count(value: object) -> str:
    if value is None:
        return "—"
    if type(value) is not int or value < 0:
        raise ValueError("Metric counts must be nonnegative integers")
    return f"{value:,}"


def _fraction(numerator: object, denominator: object) -> str:
    return f"{_count(numerator)} / {_count(denominator)}"


def _elapsed(runtime: dict) -> str:
    if runtime.get("elapsed_seconds") is None:
        return "Not recorded"
    seconds = round(_number(runtime["elapsed_seconds"], "elapsed_seconds"))
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes}m {seconds:02d}s"


def _usage(runtime: dict) -> str:
    raw = runtime.get("usage")
    if raw is None:
        return "Not recorded"
    entries = raw if isinstance(raw, list) else [raw]
    totals, known = {"input_tokens": 0, "output_tokens": 0}, 0
    cached, cached_known = 0, 0
    for entry in entries:
        if entry is None:
            continue
        if not isinstance(entry, dict):
            raise ValueError("Usage must contain reported token objects or null")
        if any(entry.get(key) is None for key in totals):
            continue
        for key in totals:
            _count(entry[key])
            totals[key] += entry[key]
        if entry.get("cached_input_tokens") is not None:
            _count(entry["cached_input_tokens"])
            if entry["cached_input_tokens"] > entry["input_tokens"]:
                raise ValueError("Cached input must be a subset of reported input tokens")
            cached += entry["cached_input_tokens"]
            cached_known += 1
        known += 1
    if not known:
        return "Not recorded"
    suffix = f"Reported for {known} / {len(entries)} usage records"
    cache = (
        f"<small>{cached:,} cached input (subset, not additional); recorded for {cached_known} / {len(entries)} usage records</small>"
        if cached_known
        else "<small>Cached input not recorded separately</small>"
    )
    return f'{totals["input_tokens"]:,} input · {totals["output_tokens"]:,} output<small>{suffix}</small>{cache}'


def _status(lane: dict) -> str:
    runtime = lane["runtime"]
    status = runtime["status"]
    label = "Time limit reached" if runtime.get("timed_out") is True else status.title()
    result = f'<span class="{status}">{label}</span>'
    if lane.get("status") and lane["status"] != status:
        result += f'<small>Score validation: {text(lane["status"])}</small>'
    if "completion_claim" in lane:
        result += (
            f'<small>Author completion claim: {"yes" if lane["completion_claim"] else "no"}</small>'
        )
    return result


def _row(label: str, values: list[str], hint: str = "") -> str:
    detail = f"<small>{text(hint)}</small>" if hint else ""
    return (
        f'<tr><th scope="row">{text(label)}{detail}</th>'
        + "".join(f"<td>{value}</td>" for value in values)
        + "</tr>"
    )


def _table(rows: list[str]) -> str:
    headers = '<th scope="col">Measure</th>' + "".join(
        f'<th scope="col">{label}</th>' for label in STAGES.values()
    )
    return f'<div class="table-scroll"><table><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _evidence(report: dict, report_path: Path, output: Path) -> str:
    groups = []
    for stage, title in STAGES.items():
        links = []
        for kind, label in (
            ("abc", "ABC score"),
            ("decisions", "Decision log"),
            ("provenance", "Run record"),
        ):
            record = report.get("stage_artifacts", {}).get(stage, {}).get(kind)
            if record:
                links.append(_link(_artifact_path(report, record), output, label))
        runtime = report["stages"][stage]["runtime"]
        error = (
            f'<p class="quiet">Execution note: {text(runtime["error"])}</p>'
            if runtime.get("error")
            else ""
        )
        groups.append(
            f'<div><h3>{title}</h3><p class="links">{"".join(links) or "No artifact links recorded"}</p>{error}</div>'
        )
    unsupported = "; ".join(str(value) for value in report.get("unsupported_scope", []))
    plan = _plan_hash(report)
    return f'<details class="details"><summary>Evidence and limits</summary><div class="pair">{"".join(groups)}</div><p>{_link(report_path, output, "Evaluation report (JSON)")}</p><p class="quiet">Outside this score: {text(unsupported or "See report")}</p><p><small>Plan SHA256: <code>{text(plan)}</code><br>Evaluated: {text(report["created_at"])}</small></p></details>'


def _stage_rows(report: dict) -> list[str]:
    lanes = list(report["stages"][stage] for stage in STAGES)
    return [
        _row("Execution", [_status(lane) for lane in lanes]),
        _row(
            "Elapsed time",
            [_elapsed(lane["runtime"]) for lane in lanes],
            "Raw time per stage; the reader is additional work",
        ),
        _row(
            "Reported tokens",
            [_usage(lane["runtime"]) for lane in lanes],
            "Usage evidence, not a price estimate",
        ),
    ]


def _aviador_section(report: dict, path: Path, output: Path) -> str:
    lanes = [report["stages"][stage] for stage in STAGES]
    summaries = [lane["metrics"]["summary"] for lane in lanes]
    valid = [lane["submission_validation"]["headline_metrics_valid"] for lane in lanes]
    unscorable = '<span class="failed">Unscorable<small>No accuracy claim</small></span>'
    validation = []
    for lane, scorable in zip(lanes, valid, strict=True):
        checks = lane["submission_validation"]
        parse = "ABC valid" if checks.get("abc_valid") else "ABC invalid"
        bars = _fraction(checks.get("submitted_measures"), checks.get("expected_measures"))
        meter = "meter valid" if checks.get("all_declared_meter_bars_valid") else "meter not valid"
        label = (
            '<span class="complete">Scorable</span>'
            if scorable
            else '<span class="failed">Invalid / unscorable</span>'
        )
        validation.append(f"{label}<small>{parse}; {bars} written bars; {meter}</small>")
    rows = [
        _row("Score validation", validation, "Separate from the model’s completion claim"),
        _row(
            "Strict historical note F1",
            [
                _percent(s.get("note_f1")) if ok else unscorable
                for s, ok in zip(summaries, valid, strict=True)
            ],
            "Pitch + onset + duration; ties are not scored",
        ),
        _row(
            "Exact event bars",
            [
                _fraction(s.get("exact_measures"), s.get("targets")) if ok else unscorable
                for s, ok in zip(summaries, valid, strict=True)
            ],
            "Every note and rest must match",
        ),
        _row(
            "Rest F1",
            [
                _percent(s.get("rest_f1")) if ok else unscorable
                for s, ok in zip(summaries, valid, strict=True)
            ],
        ),
        *_stage_rows(report),
    ]
    baseline = report.get("baseline", {}).get("metrics", {}).get("summary")
    control = ""
    if baseline:
        control = f'<p class="scope">Frozen older pipeline control: {_fraction(baseline.get("exact_measures"), baseline.get("targets"))} exact event bars; {_percent(baseline.get("note_f1"))} strict note F1. This is the earlier automatic pipeline, not an additional Astra trial.</p>'
    return f'<section class="section" id="aviador"><div class="section-head"><h2>1. Aviador · system 8</h2><p class="quiet">Seven bars · same Astra model · one author, then one independent source reader</p></div>{_table(rows)}<p class="scope">Advance only with a usable complete initial draft, successful completion of both stages, and a final AI-reviewed result of at least 6 / 7 exact bars and 95% strict note F1. The initial draft alone must match 7 / 7 bars for exact historical reproduction.</p>{control}{_evidence(report, path, output)}</section>'


def _edition_exact_bars(lane: dict) -> str:
    coverage = lane.get("coverage", {})
    exact, columns = coverage.get("exact_measures"), coverage.get("alignment_columns")
    rate = _rate(lane["metrics"].get("edition_exact_bar_rate"), "edition_exact_bar_rate")
    fraction = _fraction(exact, columns)
    if exact is not None and columns is not None and rate is not None:
        expected_rate = exact / columns if columns else 0
        if exact > columns or not math.isclose(rate, expected_rate, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Exact edition bars disagree with the reported alignment-based rate")
    return f"{_percent(rate)}<small>{fraction} alignment columns</small>"


def _rumichaca_section(report: dict | None, path: Path | None, output: Path, passed: bool) -> str:
    heading = '<div class="section-head"><h2>2. Rumichaca · complete score</h2></div>'
    if report is None:
        state = (
            "No full-score result yet. The excerpt gate passed; this dashboard has no Rumichaca report."
            if passed
            else "Skipped by this protocol. The excerpt gate did not pass, so the full-score stage is not eligible to start."
        )
        return f'<section class="section" id="rumichaca">{heading}<div class="notice">{state}</div></section>'
    lanes = [report["stages"][stage] for stage in STAGES]
    metrics = [lane["metrics"] for lane in lanes]
    coverage = [lane.get("coverage", {}) for lane in lanes]
    source = []
    for metric, covered in zip(metrics, coverage, strict=True):
        eligible = covered.get("source_eligible_measures", covered.get("source_scored_measures"))
        source.append(
            _percent(metric.get("source_note_f1"))
            + f'<small>{_fraction(eligible, covered.get("source_total_measures"))} source bars audited</small>'
        )
    rows = [
        _row(
            "Audited source notes",
            source,
            "F1 includes pitch, onset, duration and ties; partial source coverage",
        ),
        _row(
            "Edition notes",
            [_percent(m.get("edition_note_f1")) for m in metrics],
            "Agreement with the chosen human performing edition",
        ),
        _row(
            "Exact edition bars",
            [_edition_exact_bars(lane) for lane in lanes],
            "Notes, rests, ties, meter and chord events must all match; extra and missing bars count. This is not melody-only accuracy.",
        ),
        _row("Edition chord symbols", [_percent(m.get("chord_f1")) for m in metrics]),
        _row(
            "Edition playback route",
            [
                (
                    "Exact match"
                    if m.get("form_exact_match") is True
                    else "Different" if m.get("form_exact_match") is False else "Unmeasured"
                )
                for m in metrics
            ],
        ),
        _row(
            "Reference bars aligned",
            [_fraction(c.get("paired_measures"), c.get("reference_measures")) for c in coverage],
            "Alignment coverage is not musical accuracy",
        ),
        *_stage_rows(report),
    ]
    return f'<section class="section" id="rumichaca">{heading}{_table(rows)}<p class="scope">The source metric covers the nine pre-existing model-audited bars only. Edition agreement also reflects human reconstructions and performance choices. Neither metric measures whole-score source accuracy or approves an editorial choice. These metrics include ties and must not be pooled with Aviador’s historical metric.</p>{_evidence(report, path, output)}</section>'


def render_dashboard(
    aviador_report: Path, output: Path, rumichaca_report: Path | None = None
) -> str:
    """Render report metadata without opening candidate or reference artifacts."""
    aviador_report, output = Path(aviador_report).resolve(), Path(output).resolve()
    aviador = _validate(json.loads(aviador_report.read_text()), "aviador_s8")
    rumichaca = None
    if rumichaca_report is not None:
        rumichaca_report = Path(rumichaca_report).resolve()
        rumichaca = _validate(json.loads(rumichaca_report.read_text()), "rumichaca_full")
        if (
            any(
                aviador[key] != rumichaca[key]
                for key in ("workflow_id", "model", "reasoning_effort", "evidence_kind")
            )
            or _plan_hash(aviador) != _plan_hash(rumichaca)
            or not aviador["gate"]["advance_to_rumichaca"]
            or not any(
                pin.get("sha256") == hashlib.sha256(aviador_report.read_bytes()).hexdigest()
                and Path(pin["path"]).resolve() == aviador_report
                for pin in rumichaca.get("pins", [])
            )
        ):
            raise ValueError("Rumichaca must follow the same passing preregistered workflow")
    gate = aviador["gate"]
    passed, exact, review_needed = (
        gate[key]
        for key in (
            "advance_to_rumichaca",
            "exact_historical_reproduction_initial_only",
            "independent_review_made_gate_pass",
        )
    )
    reasons = gate.get("blocking_reasons", [])
    blockers = (
        '<div class="notice"><strong>What blocked advancement</strong><ul>'
        + "".join(f"<li>{text(reason)}</li>" for reason in reasons)
        + "</ul></div>"
        if reasons
        else ""
    )
    demo = (
        '<p class="notice demo">SYNTHETIC CONTROL — layout/test data, not a new model result.</p>'
        if aviador["evidence_kind"] == "synthetic_control"
        else ""
    )
    cards = []
    for stage, label in STAGES.items():
        lane = aviador["stages"][stage]
        measured = lane["submission_validation"]["headline_metrics_valid"]
        metrics = lane["metrics"]["summary"]
        value = (
            f'{_fraction(metrics.get("exact_measures"), metrics.get("targets"))} exact bars'
            if measured
            else "Unscorable"
        )
        cards.append(
            (
                f"Aviador · {label}",
                value,
                (
                    "An exact bar matches every written note and rest; historical metric excludes ties"
                    if measured
                    else "No accuracy claim; see score validation below"
                ),
                measured,
            )
        )
    if rumichaca is not None:
        final = rumichaca["stages"]["reviewed"]
        rate = _rate(final["metrics"].get("edition_note_f1"), "edition_note_f1")
        initial_rate = _rate(
            rumichaca["stages"]["initial"]["metrics"].get("edition_note_f1"), "edition_note_f1"
        )
        change = f"Initial draft: {initial_rate:.1%}. " if initial_rate is not None else ""
        cards.append(
            (
                "Rumichaca · After independent AI review",
                f"{rate:.1%} edition note F1" if rate is not None else "Edition notes unmeasured",
                f'Full-score result available; score status: {text(final.get("status", "unknown"))}. {change}Agreement with the human performing edition.',
                rate is not None,
            )
        )
    else:
        cards.append(
            (
                "Rumichaca · complete score",
                "Result pending" if passed else "Skipped by protocol",
                (
                    "No full-score report supplied"
                    if passed
                    else "The excerpt advancement gate is blocked"
                ),
                False,
            )
        )
    card_html = "".join(
        f'<article class="card{"" if ok else " pending"}"><span>{title}</span><strong>{value}</strong><small>{hint}</small></article>'
        for title, value, hint, ok in cards
    )
    gate_result = (
        "Passed after independent AI review; the initial draft missed the quality threshold."
        if review_needed
        else "Passed; the initial draft also met the quality threshold." if passed else "Blocked."
    )
    protocol = (
        '<p class="scope"><strong>Protocol checks:</strong> '
        f'Exact initial-draft reproduction: <strong>{"Yes" if exact else "No"}</strong>. '
        "This requires 7 / 7 exact bars in the initial draft. "
        f"Full-score advancement: <strong>{gate_result}</strong> "
        "Independent AI review is a separate stage, not human approval.</p>"
    )
    subtitle = f'{text(aviador["model"])} · {text(aviador["reasoning_effort"])} reasoning · supplied PDF only'
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Astra workflow reproduction</title><style>{CSS}</style></head>
<body><header class="mast"><div class="wrap"><div class="eyebrow">Score2ABC · bounded workflow pilot</div><h1>Astra transcription results</h1><p class="lede">{subtitle}. An author draft is followed by independent AI source review. Both are sealed before scoring; this is a different workflow from the earlier tool-free comparison.</p>{demo}<div class="cards">{card_html}</div>{protocol}{blockers}</div></header>
<main class="wrap">{_aviador_section(aviador, aviador_report, output)}{_rumichaca_section(rumichaca, rumichaca_report, output, passed)}
<section class="section"><h2>What this tells us</h2><div class="pair"><div><h3>A bounded workflow test</h3><p>These are consumed development examples, not unseen scores. This pilot cannot establish accuracy across all 100 manuscripts or name a winning model. Initial and AI-reviewed drafts are separate stages, not independent trials.</p></div><div><h3>Human review remains separate</h3><p>AI review does not mark a live score human-reviewed. Editorial acceptance is unmeasured here. Subscription usage is shown when recorded; no dollar cost or claim of free inference is inferred.</p></div></div></section></main>
<footer><div class="wrap">Offline report · no scripts, remote resources, or model calls. Follow the evidence links for exact artifacts and execution records.</div></footer></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aviador_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rumichaca-report", type=Path)
    args = parser.parse_args()
    rendered = render_dashboard(args.aviador_report, args.output, args.rumichaca_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
