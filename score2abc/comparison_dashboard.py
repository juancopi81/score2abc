"""Render a versioned complete-score model comparison as offline HTML."""

# ruff: noqa: E501  # The standalone page keeps its CSS and markup together.

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path, PurePosixPath
from urllib.parse import quote

RATE_METRICS = {
    "source_note_f1": ("Source notes", "Note F1 against source-supported labels"),
    "edition_note_f1": ("Edition notes", "Note F1 against the reviewed performing edition"),
    "edition_exact_bar_rate": (
        "Exact bars",
        "Exact notation bar agreement with the reviewed edition",
    ),
    "chord_f1": ("Chords", "Chord-symbol agreement with the reviewed edition"),
    "meter_valid_rate": ("Meter valid", "Share of candidate bars passing meter validation"),
    "bar_coverage": (
        "Bars aligned",
        "Share of reviewed edition bars aligned; not content accuracy",
    ),
    "editorial_acceptance": (
        "Editorial",
        "Share of proposed editorial choices accepted after review",
    ),
}
METRIC_KEYS = set(RATE_METRICS) | {
    "form_exact_match",
    "cost_usd",
    "elapsed_seconds",
    "human_review_minutes",
}
STATUSES = {"not_run", "complete", "partial", "failed"}
LINK_LABELS = {
    "source_pdf": "Source PDF",
    "candidate_abc": "Candidate ABC",
    "decision_log": "Decision log",
    "measurement_record": "Measurement evidence",
    "run_record": "Runner record",
}

_CSS = """
:root{color-scheme:light;--ink:#153F61;--paper:#FAFCFF;--teal:#227C78;--gold:#9B681F;--line:#C9D8E1;--muted:#4E6271;--soft:#EDF4F6;--bad:#9B3535}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:#183547;font:15px/1.48 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}
a{color:var(--ink);text-decoration-thickness:1px;text-underline-offset:3px}a:hover{color:var(--teal)}
a:focus-visible,summary:focus-visible{outline:3px solid var(--gold);outline-offset:3px;border-radius:2px}
.wrap{max-width:1320px;margin:auto;padding:0 30px}.mast{background:#F3F8FB;border-bottom:1px solid var(--line)}
.mast .wrap{padding-top:16px;padding-bottom:14px}.folio{display:flex;justify-content:space-between;gap:12px;color:var(--muted);font-size:.86rem}.folio strong{color:var(--ink)}
h1,h2,h3,p{margin-top:0}h1,h2{font-family:Georgia,"Times New Roman",serif;font-weight:400;color:var(--ink)}h1{font-size:clamp(2rem,4vw,2.5rem);line-height:1.1;letter-spacing:-.025em;max-width:790px;margin:10px 0 5px}
.lede{max-width:72ch;color:var(--muted);font-size:1rem;margin:0}.demo{background:#FBF0D9;color:#663F0D;padding:7px 12px;border-left:4px solid var(--gold);font-weight:700;margin-top:10px}
.demo-nav{margin:6px 0 0;font-size:.88rem;color:var(--muted)}
.verdict{display:grid;grid-template-columns:210px 1fr;gap:16px;border-top:4px solid var(--teal);padding:10px 0 0;margin-top:12px}.verdict b{color:var(--teal)}.verdict strong{display:block;color:var(--ink);font:normal 1.3rem/1.2 Georgia,"Times New Roman",serif;margin-bottom:3px}.verdict p{max-width:70ch;color:var(--muted);margin-bottom:0}
main{padding-top:17px!important;padding-bottom:50px!important}h2{font-size:1.5rem;margin:0 0 6px}.section-head{border-bottom:2px solid var(--ink);padding-bottom:6px;margin-bottom:8px}.section-head p{color:var(--muted);margin:0}
.coverage{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 18px}.coverage span{border:1px solid var(--line);padding:6px 10px;background:white}.coverage strong{color:var(--ink)}
.table-scroll{overflow-x:auto;border:1px solid var(--line);background:white}table{border-collapse:collapse;width:100%;min-width:1040px;font-variant-numeric:tabular-nums}th,td{border-bottom:1px solid var(--line);padding:11px 10px;text-align:left;vertical-align:middle}thead th{background:#E6F0F4;color:var(--ink);font-size:.86rem;white-space:nowrap}tbody tr:last-child td{border-bottom:0}tbody tr:nth-child(4n + 3),tbody tr:nth-child(4n + 4){background:#F7FAFB}.work{font-weight:700;color:var(--ink)}.work a{font-weight:400;font-size:.79rem;display:block;margin-top:3px}.model{font-weight:700}.status{font-weight:700;white-space:nowrap}.not_run{color:var(--gold)}.failed{color:var(--bad)}.partial{color:#74541B}.complete{color:var(--teal)}.unknown{color:var(--muted)}
.metric{min-width:90px}.metric small{display:block;color:var(--muted);font-size:.74rem}.rail{display:block;height:4px;background:#DFE9EB;margin-top:4px;min-width:66px}.rail i{display:block;background:var(--teal);height:100%}
.unknown small,td .quiet{display:block;font-size:.74rem;font-weight:400}
.below{color:var(--muted);font-size:.89rem;max-width:88ch;margin:10px 0 36px}.summary-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px 28px;margin:15px 0 35px}.summary-grid div{border-top:1px solid var(--line);padding-top:10px}.summary-grid strong{color:var(--ink)}.summary-grid p{color:var(--muted);font-size:.86rem;margin:4px 0 0}
.model-summary{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin:12px 0 35px}.model-summary article{background:#F3F8FB;border-top:3px solid var(--teal);padding:13px 17px}.model-summary h3{color:var(--ink);font-size:1.13rem;margin:0 0 8px}.model-summary dl{display:grid;grid-template-columns:1fr auto;gap:5px 12px;margin:0;font-variant-numeric:tabular-nums}.model-summary dt{color:var(--muted)}.model-summary dd{margin:0;color:var(--ink);font-weight:700}.model-summary p{font-size:.8rem;color:var(--muted);margin:9px 0 0}
.details{margin:25px 0 35px}.piece{border-bottom:1px solid var(--line)}.piece summary{cursor:pointer;padding:14px 8px;color:var(--ink);font-weight:700;font-size:1.05rem}.piece summary:hover,.piece[open] summary{background:var(--soft)}.piece-body{padding:2px 12px 20px;border-left:3px solid var(--teal);margin-left:8px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:25px}.pair h3{font-size:1rem;color:var(--ink);margin:8px 0}.pair p{margin-bottom:9px}.pair ul{margin:5px 0 17px;padding-left:20px}.pair li{margin-bottom:7px;overflow-wrap:anywhere}.links{display:flex;flex-wrap:wrap;gap:9px 17px}.quiet{color:var(--muted)}.callout{background:#F3F8FB;border-left:3px solid var(--ink);padding:13px 17px;max-width:90ch}.callout strong{color:var(--ink)}
.reference-decisions{border-top:1px solid var(--line);margin-top:15px;padding-top:8px}.reference-decisions summary{font-size:.92rem;padding:6px 3px}.reference-decisions ul{padding-left:22px}.reference-decisions li{margin-bottom:10px;max-width:88ch}.reference-decisions small{display:block;color:var(--muted)}
footer{border-top:1px solid var(--line);padding:17px 0 24px;color:var(--muted);font-size:.83rem}
@media(max-width:760px){.wrap{padding:0 17px}.folio{display:block}.verdict,.pair,.summary-grid,.model-summary{grid-template-columns:1fr}.verdict{gap:6px}.coverage{gap:6px}}
"""


def _text(value: object) -> str:
    return escape(str(value), quote=True)


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return float(value)


def _validate(report: object) -> dict:
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError("Expected comparison report schema_version 1")
    if report.get("evidence_kind") not in {"actual", "synthetic_demo"}:
        raise ValueError("Report must declare actual or synthetic_demo evidence")
    synthetic = report["evidence_kind"] == "synthetic_demo"
    for name in ("frozen_manifest_sha256", "labels_sha256"):
        value = report.get(name)
        if synthetic and value in (None, "not_applicable"):
            continue
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"Report needs a valid {name}")
    if report.get("fixture_sha256") is not None and not re.fullmatch(
        r"[0-9a-f]{64}", str(report["fixture_sha256"])
    ):
        raise ValueError("Report has an invalid fixture_sha256")
    try:
        created_at = datetime.fromisoformat(str(report["created_at"]).replace("Z", "+00:00"))
    except (KeyError, ValueError) as exc:
        raise ValueError("Report needs an ISO created_at") from exc
    if created_at.tzinfo is None:
        raise ValueError("Report created_at needs a timezone")
    models = report.get("models")
    cases = report.get("cases")
    if not isinstance(models, list) or len(models) != 2:
        raise ValueError("Comparison requires two model records")
    if any(not isinstance(m, dict) or not m.get("id") or not m.get("label") for m in models):
        raise ValueError("Each model needs an id and label")
    ids = [m["id"] for m in models]
    expected_ids = {"control", "mutated"} if synthetic else {"astra", "terra"}
    if set(ids) != expected_ids:
        raise ValueError(
            f"Model ids must be {'control and mutated' if synthetic else 'astra and terra'}"
        )
    if not isinstance(cases, list) or len(cases) != 6:
        raise ValueError("Comparison requires six piece records")
    slugs = []
    for case in cases:
        if not isinstance(case, dict) or not case.get("slug") or not case.get("title"):
            raise ValueError("Each piece needs a slug and title")
        slugs.append(case["slug"])
        results = case.get("results")
        if not isinstance(results, dict) or set(results) != set(ids):
            raise ValueError(f"{case['slug']}: results must match the model ids")
        reference_decisions = case.get("reference_decisions", [])
        if not isinstance(reference_decisions, list) or any(
            not isinstance(item, dict) for item in reference_decisions
        ):
            raise ValueError(f"{case['slug']}: reference_decisions must be records")
        for model_id, result in results.items():
            _validate_result(result, f"{case['slug']}/{model_id}")
    if len(set(slugs)) != len(slugs):
        raise ValueError("Piece slugs must be distinct")
    return report


def _validate_result(result: object, context: str) -> None:
    if not isinstance(result, dict) or result.get("status") not in STATUSES:
        raise ValueError(f"{context}: invalid result status")
    metrics = result.get("metrics", {})
    if not isinstance(metrics, dict) or set(metrics) - METRIC_KEYS:
        raise ValueError(f"{context}: unknown metrics")
    if result["status"] == "not_run" and any(value is not None for value in metrics.values()):
        raise ValueError(f"{context}: unrun model cannot contain measured values")
    for name, metric in metrics.items():
        if metric is None:
            continue
        value = _number(metric, name) if name != "form_exact_match" else metric
        if name == "form_exact_match":
            if not isinstance(value, bool):
                raise ValueError(f"{context}/{name}: expected boolean or null")
        elif name in RATE_METRICS and value > 1:
            raise ValueError(f"{context}/{name}: rate exceeds one")
    metric_status = result.get("metric_status", {})
    if (
        not isinstance(metric_status, dict)
        or set(metric_status) - METRIC_KEYS
        or any(not isinstance(v, str) for v in metric_status.values())
    ):
        raise ValueError(f"{context}: invalid metric status")
    coverage = result.get("coverage", {})
    if not isinstance(coverage, dict):
        raise ValueError(f"{context}: invalid coverage")
    for key in ("reference_measures", "candidate_measures", "source_scored_measures"):
        if coverage.get(key) is not None:
            value = _number(coverage[key], key)
            if not value.is_integer():
                raise ValueError(f"{context}/{key}: expected whole measure count")
    for name in ("mismatches", "decisions"):
        items = result.get(name, [])
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise ValueError(f"{context}: {name} must be records")
    links = result.get("links", {})
    if not isinstance(links, dict) or set(links) - set(LINK_LABELS):
        raise ValueError(f"{context}: unsupported link key")
    if any(path is not None and not isinstance(path, str) for path in links.values()):
        raise ValueError(f"{context}: links must be local paths or null")
    if result["status"] == "not_run" and (result.get("mismatches") or result.get("decisions")):
        raise ValueError(f"{context}: unrun model cannot have musical findings")


def _metric_cell(metric: float | None, status: str = "") -> str:
    if metric is None:
        note = f'<small>{_text(status.replace("_", " "))}</small>' if status else ""
        return f'<span class="unknown" aria-label="Unmeasured">—{note}</span>'
    pct = 100 * metric
    status_markup = f'<small>{_text(status.replace("_", " "))}</small>' if status else ""
    return f'<span class="metric">{pct:.0f}%{status_markup}<span class="rail" aria-hidden="true"><i style="width:{pct:.2f}%"></i></span></span>'


def _amount(value: object, unit: str, precision: int = 1) -> str:
    return f"{value:.{precision}f}{unit}" if value is not None else '<span class="unknown">—</span>'


def _safe_link(report_dir: Path, output: Path, path: str, label: str) -> str:
    if not isinstance(path, str) or "://" in path or path.startswith("file:"):
        raise ValueError("File link must be a local path")
    posix = PurePosixPath(path)
    if not posix.parts or any(part in {".", ".."} for part in posix.parts):
        raise ValueError(f"Unsafe report file path: {path}")
    target = (Path(path) if posix.is_absolute() else report_dir / Path(*posix.parts)).resolve()
    if not target.is_file():
        raise ValueError(f"Missing report file: {path}")
    href = quote(Path(os.path.relpath(target, output.parent)).as_posix(), safe="/.-_~")
    return f'<a href="{_text(href)}">{_text(label)}</a>'


def _result_detail(result: dict, model: dict, report_dir: Path, output: Path) -> str:
    status = result["status"].replace("_", " ")
    links = "".join(
        _safe_link(report_dir, output, path, LINK_LABELS[name])
        for name, path in result.get("links", {}).items()
        if path
    )
    mismatches = result.get("mismatches", [])
    decisions = result.get("decisions", [])
    mismatch_html = "".join(
        f'<li><strong>Measure {_text(item.get("measure", "?"))} · {_text(item.get("kind", "Mismatch"))}:</strong> {_text(item.get("detail", "Detail unrecorded"))}</li>'
        for item in mismatches
    )
    decision_html = "".join(
        f'<li><strong>{_text(item.get("id", "Decision"))} · {_text(item.get("category", "Musical choice"))}:</strong> {_text(item.get("adopted_choice", "Choice unrecorded"))}. <span class="quiet">{_text(item.get("status", "Adjudication unrecorded"))}</span></li>'
        for item in decisions
    )
    coverage = result.get("coverage", {})
    coverage_text = " / ".join(
        f"{_text(coverage[key])} {label}"
        for key, label in (
            ("reference_measures", "reference bars"),
            ("candidate_measures", "candidate bars"),
            ("source_scored_measures", "source-scored bars"),
        )
        if coverage.get(key) is not None
    )
    metric_status = result.get("metric_status", {})
    statuses = "".join(
        f'<li>{_text(RATE_METRICS.get(name, (name,))[0])}: {_text(value.replace("_", " "))}</li>'
        for name, value in metric_status.items()
    )
    exact_bars = _metric_cell(result.get("metrics", {}).get("edition_exact_bar_rate"))
    return f"""<section><h3>{_text(model['label'])} · {status}</h3>
      <p class="quiet">{_text(result.get('note') or ('No run evidence recorded.' if status == 'not run' else 'No run note recorded.'))}</p>
      <p>Exact edition bars: {exact_bars} <span class="quiet">{coverage_text}</span></p>
      {f'<p class="failed">Run error: {_text(result["error"])}</p>' if result.get('error') else ''}
      <div class="links">{links}</div>
      {f'<h3>Metric qualifications</h3><ul>{statuses}</ul>' if statuses else ''}
      <h3>Where it differed</h3>{f'<ul>{mismatch_html}</ul>' if mismatch_html else '<p class="quiet">No mismatch detail recorded.</p>'}
      <h3>Musical choices and adjudication</h3>{f'<ul>{decision_html}</ul>' if decision_html else '<p class="quiet">No decision detail recorded.</p>'}
    </section>"""


def _model_summary(model: dict, cases: list[dict]) -> str:
    results = [case["results"][model["id"]] for case in cases]
    attempted = [result for result in results if result["status"] != "not_run"]

    def mean(name: str) -> str:
        values = [
            result.get("metrics", {}).get(name)
            for result in attempted
            if result.get("metrics", {}).get(name) is not None
        ]
        return (
            f"{100 * sum(values) / len(values):.0f}% <small>({len(values)}/6 pieces)</small>"
            if values
            else "—"
        )

    def total(name: str, unit: str, places: int) -> str:
        values = [
            result.get("metrics", {}).get(name)
            for result in results
            if result.get("metrics", {}).get(name) is not None
        ]
        return (
            f"{sum(values):.{places}f} {unit} <small>({len(values)}/6 runs)</small>"
            if values
            else "—"
        )

    return f"""<article><h3>{_text(model['label'])}</h3><dl>
      <dt>Source note F1, mean</dt><dd>{mean('source_note_f1')}</dd>
      <dt>Edition note F1, mean</dt><dd>{mean('edition_note_f1')}</dd>
      <dt>Exact edition form routes</dt><dd>{mean('form_exact_match')}</dd>
      <dt>Bars aligned, mean</dt><dd>{mean('bar_coverage')}</dd>
      <dt>Known cost, total</dt><dd>{total('cost_usd', 'USD', 2)}</dd>
      <dt>Known runtime, total</dt><dd>{total('elapsed_seconds', 's', 0)}</dd>
    </dl><p>Means include measured attempted runs, including partial outcomes; missing metrics are excluded. Known totals include attempted runs with usage data.</p></article>"""


def _reference_decisions(case: dict) -> str:
    decisions = case.get("reference_decisions", [])
    if not decisions:
        return ""
    items = []
    for decision in decisions:
        location = decision.get("source_location") or {}
        if isinstance(location, dict):
            location_text = f'S{location.get("system", "?")} M{location.get("measure", "?")}'
        else:
            location_text = "Location unrecorded"
        evidence = decision.get("evidence", [])
        quotes = "".join(
            f'<small>Review evidence: {_text(record["quote"])}</small>'
            for record in evidence
            if isinstance(record, dict) and record.get("quote")
        )
        items.append(
            f'<li><strong>{_text(decision.get("id", "Decision"))} · {_text(location_text)} · {_text(decision.get("category", "uncategorized").replace("_", " "))}</strong>: '
            f'{_text(decision.get("adopted_choice", "Choice unrecorded"))}'
            f'<small>{_text(decision.get("status", "Status unrecorded").replace("_", " "))} · Evidence origin: {_text(decision.get("evidence_origin", "unrecorded").replace("_", " "))}</small>{quotes}</li>'
        )
    return (
        '<details class="reference-decisions"><summary>Reference editorial choices (not model output)'
        f' · {len(decisions)}</summary><ul>{"".join(items)}</ul></details>'
    )


def render_comparison_dashboard(report_path: Path, output: Path) -> str:
    """Validate and render one six-piece, two-model report; return HTML for the caller to save."""
    report_path = Path(report_path).resolve()
    output = Path(output).resolve()
    if output == report_path:
        raise ValueError("Dashboard cannot overwrite its source report")
    report = _validate(json.loads(report_path.read_text(encoding="utf-8")))
    models, cases = report["models"], report["cases"]
    results = [(case, model, case["results"][model["id"]]) for case in cases for model in models]
    complete = sum(result["status"] == "complete" for _, _, result in results)
    partial = sum(result["status"] == "partial" for _, _, result in results)
    failed = sum(result["status"] == "failed" for _, _, result in results)
    paired = sum(
        all(case["results"][m["id"]]["status"] == "complete" for m in models) for case in cases
    )
    synthetic = report["evidence_kind"] == "synthetic_demo"
    full_metrics = all(
        all(
            result.get("metrics", {}).get(name) is not None
            for name in (
                "source_note_f1",
                "edition_note_f1",
                "bar_coverage",
                "editorial_acceptance",
                "form_exact_match",
            )
        )
        for _, _, result in results
        if result["status"] == "complete"
    )
    summary = report.get("summary", {})
    if not isinstance(summary, dict):
        raise ValueError("summary must be an object")
    winner = summary.get("winner")
    if winner is not None and (
        winner not in {m["id"] for m in models} or synthetic or paired != 6 or not full_metrics
    ):
        raise ValueError("A winner requires six actual, complete, fully scored pairs")
    if winner and not summary.get("basis"):
        raise ValueError("A winner needs a stated evidence basis")
    if winner:
        verdict = f"{_text(next(m['label'] for m in models if m['id'] == winner))} leads in this initial paired comparison."
        basis = _text(
            summary.get(
                "basis", "See the per-piece metrics; this is one attempt per model and work."
            )
        )
    else:
        verdict = "Insufficient measured evidence to choose a model."
        basis = (
            "All six paired works need scored source, edition, form, completeness and editorial evidence before a choice is supported."
            if paired < 6 or not full_metrics
            else "The measured dimensions do not support a single choice. Review the per-piece differences and unresolved musical decisions."
        )
    if synthetic:
        verdict = "Synthetic grader demonstration; no model comparison result."
        basis = "The values below exercise the report pipeline. They are not Astra or Terra score-reading results."
    rows = []
    for case, model, result in results:
        metrics = result.get("metrics", {})
        statuses = result.get("metric_status", {})
        cells = "".join(
            f"<td>{_metric_cell(metrics.get(name), statuses.get(name, ''))}</td>"
            for name in ("source_note_f1", "edition_note_f1", "chord_f1")
        )
        form = metrics.get("form_exact_match")
        form_cell = (
            '<span class="unknown">—</span>' if form is None else ("Exact" if form else "Different")
        )
        if statuses.get("form_exact_match"):
            form_cell += f'<small class="quiet">{_text(statuses["form_exact_match"].replace("_", " "))}</small>'
        cells += f"<td>{form_cell}</td>"
        cells += (
            f'<td>{_metric_cell(metrics.get("bar_coverage"))}<small class="quiet">bars aligned</small>'
            f'{_metric_cell(metrics.get("meter_valid_rate"))}<small class="quiet">meter valid</small></td>'
        )
        cells += f'<td>{_metric_cell(metrics.get("editorial_acceptance"))}</td>'
        anchor = quote(str(case["slug"]), safe="-_")
        rows.append(
            f'<tr><td class="work">{_text(case["title"])}<a href="#{_text(anchor)}">Inspect evidence</a></td>'
            f'<td class="model">{_text(model["label"])}</td><td class="status {result["status"]}">{_text(result["status"].replace("_", " "))}</td>'
            f'{cells}<td>{_amount(metrics.get("human_review_minutes"), " min")}</td>'
            f'<td>{_amount(metrics.get("cost_usd"), " USD", 2)}</td><td>{_amount(metrics.get("elapsed_seconds"), " s", 0)}</td></tr>'
        )
    details = "".join(
        f'<details class="piece" id="{_text(quote(str(case["slug"]), safe="-_"))}"><summary>{_text(case["title"])}</summary><div class="piece-body"><div class="pair">'
        + "".join(
            _result_detail(case["results"][model["id"]], model, report_path.parent, output)
            for model in models
        )
        + f"</div>{_reference_decisions(case)}</div></details>"
        for case in cases
    )
    metric_guide = "".join(
        f"<div><strong>{_text(label)}</strong><p>{_text(definition)}. Missing values remain unmeasured.</p></div>"
        for label, definition in RATE_METRICS.values()
    )
    demo_banner = (
        '<div class="demo" role="note">Synthetic demonstration data. Do not use these values to compare model quality.</div>'
        if synthetic
        else ""
    )
    demo_path = report.get("grader_demo_path")
    if demo_path is not None and synthetic:
        raise ValueError("Synthetic report cannot link itself as an actual comparison")
    demo_nav = (
        f'<p class="demo-nav">Grader validation: {_safe_link(report_path.parent, output, demo_path, "Open the separate synthetic demonstration")}</p>'
        if demo_path is not None
        else ""
    )
    columns = '<th scope="col">Source notes</th><th scope="col">Edition notes</th><th scope="col">Chords</th><th scope="col">Form</th><th scope="col">Completeness</th><th scope="col">Editorial</th>'
    has_measures = any(
        value is not None
        for _, _, result in results
        for value in result.get("metrics", {}).values()
    )
    model_summaries = "".join(_model_summary(model, cases) for model in models)
    overview = (
        f'<section aria-labelledby="overview-heading"><div class="section-head"><h2 id="overview-heading">At a glance</h2><p>Means are per piece; known cost and time totals show coverage.</p></div><div class="model-summary">{model_summaries}</div></section>'
        if has_measures
        else ""
    )
    headline = "Does the grader catch mistakes?" if synthetic else "Astra vs. Terra"
    lede = (
        "Known changes test the grader. These values are fixtures, not model results."
        if synthetic
        else "Per-score musical results, coverage and measured effort."
    )
    item_label = "Case" if synthetic else "Piece"
    model_label = "Fixture" if synthetic else "Model"
    created_at = datetime.fromisoformat(report["created_at"].replace("Z", "+00:00"))
    created_label = created_at.astimezone(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Model comparison | score2abc</title><style>{_CSS}</style></head><body><header class="mast"><div class="wrap">
<div class="folio"><strong>score2abc · evaluation desk</strong><span>Complete-score comparison · {created_label}</span></div>
<h1>{headline}</h1><p class="lede">{lede}</p>{demo_banner}{demo_nav}
<div class="verdict" role="status"><b>Current finding</b><div><strong>{verdict}</strong><p>{basis}</p></div></div></div></header>
<main class="wrap"><div class="coverage"><span><strong>{paired}/6</strong> complete pairs</span><span><strong>{complete}/12</strong> complete runs</span><span><strong>{partial}</strong> partial</span><span><strong>{failed}</strong> failed</span><span><strong>{12 - complete - partial - failed}</strong> not run</span></div>
{overview}
<section aria-labelledby="matrix-heading"><div class="section-head"><h2 id="matrix-heading">{item_label}-by-{item_label.lower()} comparison</h2><p>— means unmeasured. Completeness shows bars aligned and meter validity separately.</p></div>
<div class="table-scroll" role="region" aria-label="{model_label} comparison table" tabindex="0"><table><thead><tr><th scope="col">{item_label}</th><th scope="col">{model_label}</th><th scope="col">Run</th>{columns}<th scope="col">Review</th><th scope="col">Cost</th><th scope="col">Time</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="below">Source notes measure only source-supported notation. Edition notes measure similarity to a reviewed performing edition, which may contain reconstruction. F1 is the balance of precision and recall. Review time is recorded human effort; cost and runtime appear only when measured. Failed and partial attempts stay visible.</p></section>
<section aria-labelledby="definitions-heading"><div class="section-head"><h2 id="definitions-heading">How to read the measures</h2></div><div class="summary-grid">{metric_guide}</div></section>
<section class="details" aria-labelledby="details-heading"><div class="section-head"><h2 id="details-heading">Evidence by {item_label.lower()}</h2><p>Open a {item_label.lower()} for mismatches, editorial decisions and run files.</p></div>{details}</section>
<aside class="callout"><strong>Interpretation limit</strong><p>One initial attempt per model and piece describes these attempts, not run-to-run reliability. Editorial choices that still need a musician remain pending; an accepted edition alone does not prove what the PDF uniquely supports.</p></aside>
</main><footer><div class="wrap">Built from a local versioned report. Evidence links resolve only to existing local files.</div></footer></body></html>"""
