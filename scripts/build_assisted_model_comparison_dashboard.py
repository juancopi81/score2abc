"""Render an offline comparison of pinned assisted-workflow model reports.

The comparison manifest and compatibility audit are authored separately. This
renderer verifies their report bindings and fingerprint equality; it does not
infer matched conditions from model names or grade musical submissions.
"""

# ruff: noqa: E501  # Keep the standalone HTML with its presentation code.

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_astra_reproduction_dashboard import (  # noqa: E402
    CSS,
    STAGES,
    _count,
    _edition_exact_bars,
    _elapsed,
    _evidence,
    _fraction,
    _link,
    _local_path,
    _percent,
    _validate,
    text,
)

CASES = {"aviador_s8": "Aviador · system 8", "rumichaca_full": "Rumichaca · whole-score task"}
DIMENSIONS = ("data", "prompts", "tools", "limits", "grader")
TOKEN_KEYS = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens")
TOKEN_LABELS = ("Input", "Cached input (subset)", "Output", "Reasoning output (subset)")
EXTRA_CSS = """
.wrap{max-width:1400px}.cards{grid-template-columns:repeat(auto-fit,minmax(270px,1fr))}.card strong{font-size:1.08rem}.card h2{font-family:inherit;font-size:1.2rem;font-weight:700}.summary-metrics{min-width:0;table-layout:fixed;margin:12px 0}.summary-metrics th,.summary-metrics td{padding:9px 7px;font-size:.85rem;vertical-align:middle}.summary-metrics thead th:first-child{width:42%}.summary-metrics thead th:not(:first-child),.summary-metrics td{text-align:center}.summary-metrics .metric{display:inline}.comparison{min-width:1050px}.comparison thead th:first-child{width:auto}.stage{max-width:165px}.resource{min-width:1120px}.resource td,.resource th{font-size:.85rem;padding:9px}.ratio{min-width:800px}.ratio td,.ratio th{font-size:.87rem}.model-id{max-width:210px;overflow-wrap:anywhere}.state{font-weight:700}.audit-passed{border-left-color:var(--teal);background:#EDF6F4}.audit-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}.audit-grid article{border:1px solid var(--line);padding:12px}.audit-grid h3{margin:0 0 8px}.audit-grid ul{font-size:.85rem}.case-summary{display:flex;gap:18px;flex-wrap:wrap}.footnote{font-size:.84rem;color:var(--muted)}
"""


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pinned(root: Path, record: dict, label: str) -> Path:
    if not isinstance(record, dict) or not re.fullmatch("[a-f0-9]{64}", str(record.get("sha256"))):
        raise ValueError(f"{label} requires a SHA256 pin")
    path = _local_path(record.get("path"))
    path = path if path.is_absolute() else root / path
    if not path.is_file() or _hash(path) != record["sha256"]:
        raise ValueError(f"{label} hash mismatch")
    return path.resolve()


def load_comparison(manifest_path: Path) -> dict:
    """Read only the manifest, audit and reports; hash outcome evidence if supplied."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "assisted_model_comparison_manifest"
    ):
        raise ValueError("Expected assisted model comparison manifest v1")
    if type(manifest.get("trial")) is not int or manifest["trial"] < 1:
        raise ValueError("Manifest requires a positive trial number")
    if not isinstance(manifest.get("comparison_id"), str) or not manifest["comparison_id"]:
        raise ValueError("Manifest requires comparison_id")
    notes = manifest.get("experiment_notes", [])
    if not isinstance(notes, list) or any(not isinstance(note, str) for note in notes):
        raise ValueError("experiment_notes must be a list of strings")
    systems = manifest.get("systems")
    if not isinstance(systems, list) or len(systems) < 2:
        raise ValueError("Comparison requires at least two named systems")
    ids, evidence_kinds, bound, reports, outcomes = [], set(), {}, {}, {}
    for system in systems:
        identifier = system.get("id")
        if (
            not isinstance(identifier, str)
            or not re.fullmatch(r"[a-zA-Z0-9_-]+", identifier)
            or identifier in ids
        ):
            raise ValueError("Systems require unique simple identifiers")
        ids.append(identifier)
        if any(
            not isinstance(system.get(key), str) or not system[key]
            for key in ("label", "requested_model", "reasoning_effort")
        ):
            raise ValueError("System requires label, requested_model and reasoning_effort")
        if set(system.get("reports", {})) != set(CASES):
            raise ValueError("Each system must explicitly include both case report pins or null")
        bound[identifier], reports[identifier], outcomes[identifier] = {}, {}, {}
        for case in CASES:
            record = system["reports"][case]
            bound[identifier][case] = None if record is None else record.get("sha256")
            if record is not None:
                path = _pinned(manifest_path.parent, record, f"{identifier} {case} report")
                report = _validate(json.loads(path.read_text()), case)
                if (
                    report["model"] != system["requested_model"]
                    or report["reasoning_effort"] != system["reasoning_effort"]
                ):
                    raise ValueError("Report model/effort differs from configured system")
                evidence_kinds.add(report["evidence_kind"])
                reports[identifier][case] = {"path": path, "report": report}
            else:
                reports[identifier][case] = None
            outcome = system.get("outcomes", {}).get(case)
            if outcome is not None:
                if record is not None or outcome.get("status") not in {
                    "pending",
                    "failed",
                    "skipped",
                }:
                    raise ValueError("Outcome metadata applies only to an unavailable report")
                outcome = dict(outcome)
                if outcome.get("evidence"):
                    outcome["evidence_path"] = _pinned(
                        manifest_path.parent, outcome["evidence"], "case outcome evidence"
                    )
                outcomes[identifier][case] = outcome
    if len(evidence_kinds) > 1:
        raise ValueError("Synthetic and actual reports must not share a comparison")
    baseline = manifest.get("baseline_system_id", ids[0])
    if baseline not in ids:
        raise ValueError("Baseline system is not in this comparison")
    audit_path, checks = None, {}
    if manifest.get("compatibility_audit") is not None:
        audit_path = _pinned(
            manifest_path.parent, manifest["compatibility_audit"], "compatibility audit"
        )
        audit = json.loads(audit_path.read_text())
        if (
            audit.get("schema_version") != 1
            or audit.get("kind") != "assisted_model_comparison_compatibility_audit"
            or audit.get("report_sha256") != bound
        ):
            raise ValueError("Compatibility audit must bind exactly the compared report pins")
        for check in audit.get("checks", []):
            key = (check.get("case_id"), check.get("stage"), check.get("dimension"))
            if (
                key in checks
                or key[0] not in CASES
                or key[1] not in STAGES
                or key[2] not in DIMENSIONS
            ):
                raise ValueError("Unexpected or duplicate compatibility check")
            if check.get("status") not in {"matched", "mismatched", "pending"} or not isinstance(
                check.get("basis"), str
            ):
                raise ValueError("Compatibility check requires status and normalization basis")
            fingerprints = check.get("fingerprints", {})
            if set(fingerprints) != set(ids):
                raise ValueError("Compatibility fingerprints must identify every system")
            for value in fingerprints.values():
                if value is not None and not re.fullmatch("[a-f0-9]{64}", str(value)):
                    raise ValueError("Invalid compatibility fingerprint")
            if check["status"] == "matched" and (
                None in fingerprints.values() or len(set(fingerprints.values())) != 1
            ):
                raise ValueError("Matched compatibility check has unequal or missing fingerprints")
            checks[key] = check
    matched_pairs = {
        (case, stage): all(
            checks.get((case, stage, dimension), {}).get("status") == "matched"
            for dimension in DIMENSIONS
        )
        and all(reports[identifier][case] is not None for identifier in ids)
        for case in CASES
        for stage in STAGES
    }
    historical_path = None
    if manifest.get("historical_dashboard") is not None:
        historical_path = _local_path(manifest["historical_dashboard"])
        if not historical_path.is_absolute():
            historical_path = manifest_path.parent / historical_path
        historical_path = historical_path.resolve()
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "systems": systems,
        "reports": reports,
        "outcomes": outcomes,
        "audit_path": audit_path,
        "checks": checks,
        "matched_pairs": matched_pairs,
        "baseline": baseline,
        "synthetic": evidence_kinds == {"synthetic_control"},
        "historical_path": historical_path,
    }


def _lane(comparison: dict, system: dict, case: str, stage: str) -> dict | None:
    bundle = comparison["reports"][system["id"]][case]
    return bundle["report"]["stages"][stage] if bundle else None


def _scorable(lane: dict, case: str) -> bool:
    return (
        lane["submission_validation"]["headline_metrics_valid"]
        if case == "aviador_s8"
        else lane.get("status") != "failed"
    )


def _metric(lane: dict | None, case: str, metric: str) -> str:
    if lane is None:
        return '<span class="quiet">—<small>No report</small></span>'
    if not _scorable(lane, case):
        return '<span class="failed">Unscorable<small>No accuracy claim</small></span>'
    metrics = lane["metrics"]["summary"] if case == "aviador_s8" else lane["metrics"]
    return _percent(metrics.get(metric))


def _model(system: dict) -> str:
    return f'<strong>{text(system["label"])}</strong><small class="model-id">{text(system["requested_model"])}<br>{text(system["reasoning_effort"])} reasoning</small>'


def _status(comparison: dict, system: dict, case: str, lane: dict | None, output: Path) -> str:
    if lane is None:
        outcome = comparison["outcomes"][system["id"]].get(case, {})
        status = outcome.get("status", "pending")
        label = "Execution failed" if status == "failed" else status.title()
        result = f'<span class="{ "failed" if status == "failed" else "partial"}">{text(label)}</span><small>No graded report; no accuracy claim</small>'
        if outcome.get("reason"):
            result += f'<small>{text(outcome["reason"])}</small>'
        if outcome.get("evidence_path"):
            result += _link(outcome["evidence_path"], output, "Execution evidence")
        return result
    runtime = lane["runtime"]
    result = f'<span class="{text(runtime["status"])}">Execution: {text(runtime["status"])}</span>'
    if runtime.get("completed_turn") is True and runtime.get("returncode") == 0:
        result += "<small>Model turn completed; exit code 0</small>"
    claimed = lane.get("completion_claim", runtime.get("complete"))
    if isinstance(claimed, bool):
        result += f'<small>Completion claimed: {"yes" if claimed else "no"}</small>'
    if runtime.get("timed_out"):
        result += '<small class="failed">Time limit reached</small>'
    if runtime.get("error"):
        result += f'<small>{text(runtime["error"])}</small>'
    if case == "aviador_s8":
        validation = lane["submission_validation"]
        result += f'<small>{_fraction(validation.get("submitted_measures"), validation.get("expected_measures"))} written bars; {"scorable" if _scorable(lane, case) else "unscorable"}</small>'
    else:
        coverage = lane.get("coverage", {})
        if coverage.get("candidate_measures") is not None:
            result += f'<small>{_fraction(coverage["candidate_measures"], coverage.get("reference_measures"))} written bars</small>'
        result += f'<small>Score: {text(lane.get("status", "unknown"))}; {_fraction(coverage.get("paired_measures"), coverage.get("reference_measures"))} reference bars aligned</small>'
    return result


def _table(headings: list[str], rows: list[list[str]], css: str = "comparison") -> str:
    head = "".join(f'<th scope="col">{text(heading)}</th>' for heading in headings)
    body = "".join("<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>" for row in rows)
    return f'<div class="table-scroll"><table class="{css}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _case_conditions(comparison: dict, case: str) -> str:
    matched_stages = [
        label for stage, label in STAGES.items() if comparison["matched_pairs"][(case, stage)]
    ]
    if len(matched_stages) == len(STAGES):
        return "Comparable within this case: matched conditions verified for both stages."
    if any(comparison["reports"][system["id"]][case] is None for system in comparison["systems"]):
        return "Comparison unavailable: a report is missing. Missing or interrupted execution is not an accuracy result."
    mismatches = [
        {"tools": "tools or runtime", "limits": "limits or review protocol"}.get(
            dimension, dimension
        )
        for dimension in DIMENSIONS
        if any(
            comparison["checks"].get((case, stage, dimension), {}).get("status") == "mismatched"
            for stage in STAGES
        )
    ]
    prefix = "Some stages are descriptive only" if matched_stages else "Descriptive results only"
    reason = (
        f'conditions differ ({", ".join(mismatches)})'
        if mismatches
        else "matching conditions are not fully verified"
    )
    result = f"{prefix}: {reason}. Differences in quality or resource use cannot be attributed to model choice alone."
    if matched_stages:
        result += f' Direct comparison is supported for: {", ".join(matched_stages)}.'
    return result


def _case_section(comparison: dict, case: str, output: Path) -> str:
    rows = []
    for system in comparison["systems"]:
        for stage, label in STAGES.items():
            lane = _lane(comparison, system, case, stage)
            row = [
                _model(system),
                f'<span class="stage">{label}</span>',
                _status(comparison, system, case, lane, output),
            ]
            if case == "aviador_s8":
                counts = lane["metrics"]["summary"] if lane else {}
                exact = (
                    _fraction(counts.get("exact_measures"), counts.get("targets"))
                    if lane and _scorable(lane, case)
                    else "Unscorable" if lane else "—"
                )
                row += [_metric(lane, case, "note_f1"), exact, _metric(lane, case, "rest_f1")]
            else:
                coverage = lane.get("coverage", {}) if lane else {}
                source = _metric(lane, case, "source_note_f1")
                if lane:
                    source += f'<small>{_fraction(coverage.get("source_eligible_measures"), coverage.get("source_total_measures"))} source bars audited</small>'
                route = lane["metrics"].get("form_exact_match") if lane else None
                route_text = (
                    "Exact edition route"
                    if route is True
                    else "Different from edition" if route is False else "Unmeasured"
                )
                if lane and not _scorable(lane, case):
                    route_text = "Unscorable"
                row += [
                    source,
                    _metric(lane, case, "edition_note_f1"),
                    _metric(lane, case, "chord_f1"),
                    route_text,
                    (
                        _edition_exact_bars(lane)
                        if lane and _scorable(lane, case)
                        else "Unscorable" if lane else "—"
                    ),
                ]
            rows.append(row)
    headings = ["Model / effort", "Stage", "Completion / coverage"]
    if case == "aviador_s8":
        headings += ["Strict historical note F1", "Exact event bars", "Rest F1"]
        scope = "Seven-bar excerpt. Historical note F1 requires pitch, onset and duration together; it excludes ties. Exact event bars also require every rest to match. These metrics do not evaluate form, chord symbols, or the whole score."
    else:
        headings += [
            "Audited source note F1",
            "Edition note F1",
            "Edition chord F1",
            "Playback route",
            "Exact edition bars",
        ]
        scope = "Source notes cover the fixed nine-of-45-bar model-audited subset, using edition-based alignment. Whole-score edition agreement includes the chosen human reconstructions. These note metrics include ties and must not be pooled with Aviador’s historical metric. Exact edition bars require notes, rests, ties, meter and chord events all to match; extra/missing bars count. They are not melody-only accuracy. A different edition route is not automatic proof of an invalid PDF-only reading."
    return f'<section class="section" id="{case}"><div class="section-head"><h2>{CASES[case]}</h2><p class="quiet">Initial AI draft and independent AI review remain separate.</p></div><p class="scope">{text(_case_conditions(comparison, case))}</p>{_table(headings, rows)}<p class="scope">{scope}</p></section>'


def _usage(runtime: dict) -> dict:
    """Keep each reported counter separate; nulls remain gaps, never zeros."""
    raw = runtime.get("usage")
    entries = raw if isinstance(raw, list) else [raw]
    values, known = {}, {}
    for key in TOKEN_KEYS:
        counters = []
        for entry in entries:
            if entry is None:
                continue
            if not isinstance(entry, dict):
                raise ValueError("Invalid reported usage")
            value = entry.get(key)
            if value is not None:
                _count(value)
                counters.append(value)
        values[key] = sum(counters) if counters else None
        known[key] = len(counters)
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for subset, total in (
            ("cached_input_tokens", "input_tokens"),
            ("reasoning_output_tokens", "output_tokens"),
        ):
            if (
                entry.get(subset) is not None
                and entry.get(total) is not None
                and entry[subset] > entry[total]
            ):
                raise ValueError("Reported usage subset exceeds its parent counter")
    return {"values": values, "known": known, "records": len(entries)}


def _usage_cell(usage: dict, key: str) -> str:
    count = usage["values"][key]
    if count is None:
        return '<span class="quiet">Unmeasured</span>'
    suffix = (
        f'<small>Known sum: {usage["known"][key]} / {usage["records"]} records</small>'
        if usage["known"][key] != usage["records"]
        else ""
    )
    return _count(count) + suffix


def _execution_complete(lane: dict) -> bool:
    """Require completed work, not only a completed turn or successful process exit."""
    runtime = lane["runtime"]
    return (
        runtime["status"] == "complete"
        and runtime.get("returncode") == 0
        and not runtime.get("timed_out")
        and runtime.get("complete") is not False
        and runtime.get("completed_turn") is not False
        and lane.get("completion_claim") is not False
        and lane.get("status") not in {"partial", "failed"}
    )


def _ratio(numerator: object, denominator: object) -> str:
    if numerator is None or denominator in (None, 0):
        return "Unmeasured"
    return f"{numerator / denominator:.2f}×"


def _resource_section(comparison: dict) -> str:
    rows, ratio_rows = [], []
    baseline = next(
        system for system in comparison["systems"] if system["id"] == comparison["baseline"]
    )
    for case, title in CASES.items():
        for stage, label in STAGES.items():
            base_lane = _lane(comparison, baseline, case, stage)
            for system in comparison["systems"]:
                lane = _lane(comparison, system, case, stage)
                usage = _usage(lane["runtime"]) if lane else _usage({})
                rows.append(
                    [
                        title,
                        text(system["label"]),
                        label,
                        _elapsed(lane["runtime"]) if lane else "No report",
                        *(_usage_cell(usage, key) for key in TOKEN_KEYS),
                    ]
                )
                if system["id"] == baseline["id"]:
                    continue
                reason = None
                if lane is None or base_lane is None:
                    reason = "Unavailable report"
                elif not comparison["matched_pairs"][(case, stage)]:
                    reason = (
                        "Conditions differ; descriptive results only"
                        if any(
                            comparison["checks"].get((case, stage, dimension), {}).get("status")
                            == "mismatched"
                            for dimension in DIMENSIONS
                        )
                        else "Matching conditions unverified"
                    )
                elif not _execution_complete(lane) or not _execution_complete(base_lane):
                    reason = "Incomplete execution or score submission; workload differs"
                label_ratio = f'{text(system["label"])} / {text(baseline["label"])}'
                if reason:
                    ratio_rows.append([title, label, label_ratio, reason, *("—" for _ in range(4))])
                    continue
                base_usage = _usage(base_lane["runtime"])
                tokens = []
                for key in TOKEN_KEYS:
                    measured = (
                        usage["known"][key] == usage["records"]
                        and base_usage["known"][key] == base_usage["records"]
                    )
                    tokens.append(
                        _ratio(usage["values"][key], base_usage["values"][key])
                        if measured
                        else "Unmeasured"
                    )
                ratio_rows.append(
                    [
                        title,
                        label,
                        label_ratio,
                        _ratio(
                            lane["runtime"].get("elapsed_seconds"),
                            base_lane["runtime"].get("elapsed_seconds"),
                        ),
                        *tokens,
                    ]
                )
    return f'<section class="section" id="resources"><div class="section-head"><h2>Time and reported token usage</h2><p class="quiet">Raw per-stage totals. Input includes cached input; output includes reasoning output. Never add the subset columns again.</p></div>{_table(["Piece", "Model", "Stage", "Elapsed", *TOKEN_LABELS], rows, "resource")}<h3>Matched case-and-stage ratios</h3><p class="footnote">A ratio below 1 means fewer reported tokens or less elapsed time for that same stage. It does not establish lower billing cost or better musical quality. Ratios require matching conditions, successful execution and complete submissions on both sides, and complete measurements. A completed model turn alone does not establish a complete score. Zero or missing denominators are unmeasured.</p>{_table(["Piece", "Stage", "Ratio direction", "Time ratio", *TOKEN_LABELS], ratio_rows, "ratio")}<p class="scope">These runs use subscription authentication; no API-dollar or subscription-price estimate is made. Cached and reasoning counts are subsets, not additional tokens. Partial usage is shown as a known sum with measurement coverage.</p></section>'


def _summary_cards(comparison: dict) -> str:
    cards = []
    for system in comparison["systems"]:
        stats = []
        for case, metric, label in (
            ("aviador_s8", "note_f1", "Aviador historical notes"),
            ("rumichaca_full", "edition_note_f1", "Rumichaca edition notes"),
        ):
            values = []
            for stage in STAGES:
                lane = _lane(comparison, system, case, stage)
                value = _metric(lane, case, metric)
                if lane and not _execution_complete(lane):
                    status = (
                        "Partial score" if lane.get("status") == "partial" else "Incomplete stage"
                    )
                    value += f"<small>{status}</small>"
                values.append(value)
            stats.append(
                f'<tr><th scope="row">{label}</th><td>{values[0]}</td><td>{values[1]}</td></tr>'
            )
        headings = '<th scope="col">Note F1 scope</th><th scope="col">Initial AI draft</th><th scope="col">After AI review</th>'
        cards.append(
            f'<article class="card"><h2>{text(system["label"])}</h2><small>{text(system["requested_model"])} · {text(system["reasoning_effort"])} reasoning</small><table class="summary-metrics"><thead><tr>{headings}</tr></thead><tbody>{"".join(stats)}</tbody></table><small>Two different metric scopes; no pooled accuracy score.</small></article>'
        )
    return '<div class="cards">' + "".join(cards) + "</div>"


def _audit_section(comparison: dict, output: Path) -> str:
    panels = []
    for case, title in CASES.items():
        for stage, label in STAGES.items():
            items = []
            for dimension in DIMENSIONS:
                check = comparison["checks"].get((case, stage, dimension))
                status = check["status"] if check else "missing"
                basis = check["basis"] if check else "No compatibility evidence supplied."
                items.append(
                    f"<li><strong>{dimension}: {text(status)}</strong><br>{text(basis)}</li>"
                )
            panels.append(f'<article><h3>{title} · {label}</h3><ul>{"".join(items)}</ul></article>')
    links = [_link(comparison["manifest_path"], output, "Comparison manifest")]
    if comparison["audit_path"]:
        links.append(_link(comparison["audit_path"], output, "Compatibility audit"))
    return f'<details class="details"><summary>Why these conditions are—or are not—matched</summary><p class="links">{"".join(links)}</p><p class="footnote">The manifest and audit pin the exact reports. Matching is asserted only when all five normalized fingerprints match for every case and stage. This page verifies those bindings; the audit records how prompts, tools, data, limits and scorer implementations were normalized.</p><div class="audit-grid">{"".join(panels)}</div></details>'


def render_dashboard(manifest_path: Path, output: Path) -> str:
    comparison = load_comparison(manifest_path)
    output = Path(output).resolve()
    matched = all(comparison["matched_pairs"].values())
    compatibility = (
        "Matched conditions verified by the pinned audit"
        if matched
        else "Matching conditions not fully verified for every case"
    )
    case_conditions = "".join(
        f"<li><strong>{title}.</strong> {text(_case_conditions(comparison, case))}</li>"
        for case, title in CASES.items()
    )
    notes = comparison["manifest"].get("experiment_notes", [])
    experiment_notes = (
        '<div class="notice experiment-notes"><strong>Experiment notes</strong><ul>'
        + "".join(f"<li>{text(note)}</li>" for note in notes)
        + "</ul></div>"
        if notes
        else ""
    )
    demo = (
        '<p class="notice demo">SYNTHETIC CONTROL — invented test metrics, not model results.</p>'
        if comparison["synthetic"]
        else ""
    )
    cases = "".join(_case_section(comparison, case, output) for case in CASES)
    evidence = []
    for system in comparison["systems"]:
        for case, title in CASES.items():
            bundle = comparison["reports"][system["id"]][case]
            if bundle:
                evidence.append(
                    f'<h3>{text(system["label"])} · {title}</h3>{_evidence(bundle["report"], bundle["path"], output)}'
                )
    trial = comparison["manifest"]["trial"]
    history = (
        "<p>Earlier cohort: "
        + _link(comparison["historical_path"], output, "Historical tool-free comparison")
        + ". Different workflow; results remain separate.</p>"
        if comparison["historical_path"]
        else ""
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Assisted transcription model comparison</title><style>{CSS}{EXTRA_CSS}</style></head><body>
<header class="mast"><div class="wrap"><div class="eyebrow">Score2ABC · assisted workflow comparison · trial {trial}</div><h1>Compare transcription quality and effort</h1><p class="lede">Aviador and Rumichaca, with an initial AI draft followed by independent AI source review. Both stages remain visible. These results belong to the assisted workflow, separately from the historical tool-free comparison.</p>{demo}{_summary_cards(comparison)}<div class="notice{" audit-passed" if matched else ""}"><strong>{compatibility}</strong><ul>{case_conditions}</ul><small>Completion alone does not identify the better model. Only one quality-scored trial per model and case; no uncertainty estimate or unseen-score claim.</small></div>{experiment_notes}</div></header>
<main class="wrap">{cases}{_resource_section(comparison)}<section class="section"><h2>What to conclude—and what still needs review</h2><p>Compare musical agreement and completion together with resource usage. Edition differences may reflect editorial alternatives; no automatic editorial acceptance is measured. Independent AI review is not the musician’s approval, and these runs do not change the human-reviewed scores.</p><p class="scope">Aviador’s seven-bar metric excludes ties; Rumichaca’s note metrics include them. Source accuracy covers only its audited subset. The two cases cannot be averaged into a single quality percentage. One trial cannot establish stable model superiority or accuracy across the collection.</p>{_audit_section(comparison, output)}</section><section class="section"><h2>Reports, drafts and execution evidence</h2>{"".join(evidence)}</section></main><footer><div class="wrap">{history}Offline HTML · no remote assets, scripts, model calls or dollar estimates. Requested model IDs and exact reasoning efforts are shown; a missing resolved model ID remains unknown in the linked run records.</div></footer></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rendered = render_dashboard(args.manifest, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
