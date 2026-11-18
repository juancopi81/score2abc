"""Render an offline subscription/API comparison from pinned reports only.

Manifest v1: kind=astra_transport_comparison_manifest, comparison_id,
target_id=aviador_s8|rumichaca_full, systems={subscription: {report: {path, sha256}},
api: {report: {path, sha256} | null}}. Each system may include label, status
(pending/partial/failed) and reason. Optional protocol_differences is a list of
strings; status is a string and budget is an object. No score or reference files
are opened. Rumichaca reference_summary counts come from a report or the manifest,
never a built-in score length. Unknown billing fields remain in evidence details.
"""

# ruff: noqa: E501  # Keep the standalone HTML beside its presentation code.

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_astra_reproduction_dashboard import (  # noqa: E402
    CSS,
    STAGES,
    _count,
    _elapsed,
    _fraction,
    _link,
    _local_path,
    _percent,
    _usage,
    _validate,
    text,
)

SYSTEMS = {"subscription": "Codex subscription", "api": "OpenAI API"}
EXTRA_CSS = """
.wrap{max-width:1280px}.cards{grid-template-columns:repeat(2,1fr)}thead th:first-child{width:auto}.quality{min-width:930px}.resources{min-width:1030px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:var(--soft);padding:12px;font-size:.8rem}.cost{min-width:190px}@media(max-width:680px){.cards{grid-template-columns:1fr}}
"""


def load_comparison(manifest_path: Path) -> dict:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "astra_transport_comparison_manifest"
        or manifest.get("target_id") not in {"aviador_s8", "rumichaca_full"}
        or not isinstance(manifest.get("comparison_id"), str)
        or not manifest["comparison_id"]
        or set(manifest.get("systems", {})) != set(SYSTEMS)
    ):
        raise ValueError("Expected a supported subscription/API comparison manifest v1")
    differences = manifest.get("protocol_differences", [])
    if not isinstance(differences, list) or any(not isinstance(v, str) for v in differences):
        raise ValueError("protocol_differences must be a list of strings")
    if "budget" in manifest and not isinstance(manifest["budget"], dict):
        raise ValueError("budget must be an object")
    if "status" in manifest and not isinstance(manifest["status"], str):
        raise ValueError("status must be a string")
    systems, identities, evidence = {}, set(), set()
    for identifier, default_label in SYSTEMS.items():
        system = manifest["systems"][identifier]
        if not isinstance(system, dict) or "report" not in system:
            raise ValueError("Each transport requires a report pin or explicit null")
        for key in ("label", "reason"):
            if key in system and not isinstance(system[key], str):
                raise ValueError(f"{key} must be a string")
        if system.get("status", "pending") not in {"pending", "partial", "failed"}:
            raise ValueError("Unavailable report status must be pending, partial or failed")
        record, report, path = system["report"], None, None
        if record is not None:
            if not isinstance(record, dict) or not re.fullmatch(
                r"[a-f0-9]{64}", str(record.get("sha256"))
            ):
                raise ValueError("Report requires a SHA256 pin")
            path = _local_path(record.get("path"))
            path = path if path.is_absolute() else manifest_path.parent / path
            if path.is_symlink() or not path.is_file():
                raise ValueError("Pinned report must be a regular file")
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError("Report hash mismatch")
            report = _validate(json.loads(data), manifest["target_id"])
            identities.add(
                tuple(report[key] for key in ("model", "reasoning_effort", "workflow_id"))
            )
            evidence.add(report["evidence_kind"])
        systems[identifier] = {
            **system,
            "label": system.get("label", default_label),
            "report": report,
            "path": path,
        }
    if len(identities) > 1:
        raise ValueError(
            "Transport comparison requires the same requested model, effort and workflow"
        )
    if len(evidence) > 1:
        raise ValueError("Synthetic and actual reports must not share a comparison")
    reference = manifest.get("reference_summary")
    for system in systems.values():
        reported = (system["report"] or {}).get("reference_summary")
        if reported is not None:
            if reference is not None and reference != reported:
                raise ValueError("Reference summaries disagree across the comparison")
            reference = reported
    if reference is not None:
        if not isinstance(reference, dict):
            raise ValueError("reference_summary must be an object")
        for field in ("written_measures", "performed_measures"):
            _count(reference.get(field))
    return {
        "manifest": manifest,
        "path": manifest_path,
        "systems": systems,
        "synthetic": evidence == {"synthetic_control"},
        "reference_summary": reference or {},
    }


def _runtime_status(lane: dict, *, require_completion_claim: bool = True) -> str:
    runtime = lane["runtime"]
    if runtime["status"] != "complete":
        return runtime["status"]
    if (
        runtime.get("timed_out") is True
        or runtime.get("returncode") != 0
        or runtime.get("complete") is not True
        or runtime.get("completed_turn") is False
        or (require_completion_claim and lane.get("completion_claim") is not True)
    ):
        return "partial"
    return "complete"


def _status(system: dict, lane: dict | None) -> str:
    if lane is None:
        state = system.get("status", "pending")
        reason = f'<small>{text(system["reason"])}</small>' if system.get("reason") else ""
        return f'<span class="{state if state != "pending" else "partial"}">{state.title()}</span><small>No graded report</small>{reason}'
    state, runtime = _runtime_status(lane), lane["runtime"]
    result = f'<span class="{state}">{state.title()}</span>'
    if runtime.get("timed_out") is True:
        result += "<small>Time limit reached</small>"
    validation = lane["submission_validation"]
    result += f'<small>Submitted bars: {_fraction(validation.get("submitted_measures"), validation.get("expected_measures"))}</small>'
    if state != "complete":
        result += "<small>Checkpoint metrics only</small>"
    if runtime.get("error"):
        result += f'<small>{text(runtime["error"])}</small>'
    return result


def _billing(runtime: dict, identifier: str) -> str:
    if identifier == "subscription":
        return "Subscription<small>No per-run monetary cost measured</small>"
    billing = runtime.get("billing")
    if not isinstance(billing, dict):
        billing = {}
    billed = billing.get("billed_cost_usd", billing.get("actual_billed_cost_usd"))
    estimate = billing.get(
        "estimated_cost_upper_bound_usd", billing.get("conservative_estimated_cost_usd")
    )
    if (
        estimate is None
        and billing.get("cost_basis", runtime.get("cost_basis"))
        == "conservative_estimate_not_invoice"
    ):
        estimate = billing.get(
            "cost_estimate_usd",
            runtime.get("cost_estimate_usd", billing.get("charged_or_reserved_usd")),
        )
    value = "Billed cost unknown" if billed is None else f"Recorded billed cost: ${_money(billed)}"
    value += (
        "<small>No conservative cost estimate recorded</small>"
        if estimate is None
        else f"<small>Conservative estimate: ${_money(estimate)}<br>Estimate, not an invoice or hard spending limit</small>"
    )
    return value


def _money(value: object) -> str:
    try:
        amount = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("Cost evidence must be a finite nonnegative number") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError("Cost evidence must be a finite nonnegative number")
    return f"{amount:.4f}"


def _tool_counts(runtime: dict) -> str:
    calls = runtime.get("tool_calls")
    if not isinstance(calls, list):
        return "Not recorded"
    if any(
        not isinstance(c, dict) or c.get("status") not in {"completed", "failed"} for c in calls
    ):
        return "Tool completion counts unavailable"
    successful = [c for c in calls if isinstance(c, dict) and c.get("status") == "completed"]
    return f'{sum(c.get("tool", c.get("name")) == "inspect_source" for c in successful)} inspection calls<br>{sum(c.get("tool", c.get("name")) == "save_draft" for c in successful)} draft-save calls<small>Completed calls; tool success is separate</small>'


def _usage_runtime(runtime: dict) -> dict:
    """Read both CLI counters and Responses API detail fields without double counting."""
    raw = runtime.get("usage")
    records = raw if isinstance(raw, list) else [raw]
    normalized = []
    for record in records:
        if not isinstance(record, dict):
            normalized.append(record)
            continue
        record = dict(record)
        for key, details, nested in (
            ("cached_input_tokens", "input_tokens_details", "cached_tokens"),
            ("reasoning_output_tokens", "output_tokens_details", "reasoning_tokens"),
        ):
            if record.get(key) is None and isinstance(record.get(details), dict):
                record[key] = record[details].get(nested)
        normalized.append(record)
    return {**runtime, "usage": normalized if isinstance(raw, list) else normalized[0]}


def _reasoning(runtime: dict) -> str:
    raw = _usage_runtime(runtime).get("usage")
    records = raw if isinstance(raw, list) else [raw]
    values = [
        r["reasoning_output_tokens"]
        for r in records
        if isinstance(r, dict) and r.get("reasoning_output_tokens") is not None
    ]
    for record in records:
        if isinstance(record, dict) and record.get("reasoning_output_tokens") is not None:
            _count(record["reasoning_output_tokens"])
            if record.get("output_tokens") is not None:
                _count(record["output_tokens"])
                if record["reasoning_output_tokens"] > record["output_tokens"]:
                    raise ValueError("Reasoning output must be a subset of output tokens")
    return (
        "Reasoning output not recorded separately"
        if not values
        else f"{sum(values):,} reasoning output (subset); recorded for {len(values)} / {len(records)} usage records"
    )


def _table(headers: list[str], rows: list[list[str]], css: str) -> str:
    headings = "".join(f'<th scope="col">{text(v)}</th>' for v in headers)
    body = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in rows)
    return f'<div class="table-scroll"><table class="{css}"><thead><tr>{headings}</tr></thead><tbody>{body}</tbody></table></div>'


def _full_score_status(system: dict, lane: dict | None, reference: dict) -> str:
    if lane is None:
        return _status(system, None)
    runtime, coverage = lane["runtime"], lane.get("coverage", {})
    state = _runtime_status(lane, require_completion_claim=False)
    result = f'<span class="{text(state)}">{text(state.title())}</span>'
    if runtime.get("timed_out") is True:
        result += "<small>Time limit reached</small>"
    result += f'<small>{_count(coverage.get("candidate_measures"))} written bars; reference {_count(coverage.get("reference_measures", reference.get("written_measures")))}</small>'
    result += f'<small>{_fraction(coverage.get("paired_measures"), coverage.get("reference_measures"))} reference bars aligned</small>'
    result += f'<small>Score validation: {text(lane.get("status", "unknown"))}</small>'
    if state != "complete":
        result += "<small>Checkpoint metrics only</small>"
    if runtime.get("error"):
        result += f'<small>{text(runtime["error"])}</small>'
    return result


def _source_v4(lane: dict | None) -> str:
    source = (lane or {}).get("supplemental_source")
    if not isinstance(source, dict) or source.get("label_set") != "reviewed_six_v4":
        return "Not measured<small>No v4 source audit; v3 is separate evidence</small>"
    coverage = source.get("coverage", {})
    return (
        _percent(source.get("metrics", {}).get("source_note_f1"))
        + f'<small>{_fraction(coverage.get("source_eligible_measures"), coverage.get("source_total_measures"))} source bars audited</small>'
        + f'<small>{_count(coverage.get("source_expected_noteheads"))} audited noteheads · v4</small>'
    )


def _full_form(lane: dict | None, reference: dict) -> str:
    if lane is None:
        return "Not measured"
    exact = lane.get("metrics", {}).get("form_exact_match")
    if exact is not None and type(exact) is not bool:
        raise ValueError("form_exact_match must be a boolean or null")
    validation = lane.get("validation", {})
    return (
        "Exact route" if exact is True else "Different route" if exact is False else "Not measured"
    ) + (
        f'<small>{_count(validation.get("counts", {}).get("performed_measures"))} performed bars; '
        f'reference {_count(reference.get("performed_measures"))}</small>'
        f'<small>Route: {text(validation.get("route_status", "unknown"))}</small>'
    )


def _full_meter(lane: dict | None) -> str:
    if lane is None:
        return "Not measured"
    validation, runtime = lane.get("validation", {}), lane["runtime"]
    warnings = validation.get("warnings")
    warning_count = str(len(warnings)) if isinstance(warnings, list) else "Unknown"
    issues = validation.get("issues")
    meter_count = (
        str(sum(isinstance(issue, dict) and issue.get("kind") == "meter" for issue in issues))
        if isinstance(issues, list)
        else "Unknown"
    )
    gate = runtime.get("completion_gate")
    if isinstance(gate, dict):
        exceptions = gate.get("accepted_source_exceptions")
        exception_count = str(len(exceptions)) if isinstance(exceptions, list) else "Unknown"
        gate_status = {
            "ready": "Ready",
            "ready_with_source_exceptions": "Ready with source exceptions",
            "needs_resolution": "Needs resolution",
            "partial": "Partial",
            "no_draft": "No draft",
            "invalid_checkpoint_evidence": "Invalid checkpoint evidence",
        }.get(gate.get("status"), "Unknown")
        gate_text = f"Gate: {gate_status}<br>{exception_count} source exceptions"
    else:
        gate_text = "Completion gate not recorded"
    return (
        _percent(lane["metrics"].get("meter_valid_rate"))
        + f"<small>Ordinary bars with valid duration</small><small>{warning_count} notation warnings</small>"
        + f"<small>Meter issues: {meter_count}</small>"
        + f"<small>{gate_text}</small>"
    )


def render_dashboard(manifest_path: Path, output: Path) -> str:
    comparison = load_comparison(manifest_path)
    manifest, systems = comparison["manifest"], comparison["systems"]
    full_score = manifest["target_id"] == "rumichaca_full"
    reference = comparison["reference_summary"]
    quality, resources, details, cards = [], [], [], []
    for identifier, system in systems.items():
        report = system["report"]
        identity = (
            f'{report["model"]} · {report["reasoning_effort"]}'
            if report
            else "Graded result pending"
        )
        cards.append(
            f'<div class="card"><strong>{text(system["label"])}</strong><small>{text(identity)}</small><small>{"Codex CLI · code-mode tools" if identifier == "subscription" else "Responses API · native function tools"}</small></div>'
        )
        if full_score and report:
            reviewed = report["stages"]["reviewed"]
            cards[-1] = cards[-1][:-6] + (
                f"<small>After independent review · {text(_runtime_status(reviewed, require_completion_claim=False).title())}</small>{_source_v4(reviewed)}"
                f"<small>{_full_form(reviewed, reference)}</small></div>"
            )
        for stage, label in STAGES.items():
            lane = report["stages"][stage] if report else None
            prefix = [text(system["label"]), text(label)]
            if full_score:
                metrics = lane["metrics"] if lane else {}
                quality.append(
                    prefix
                    + [
                        _full_score_status(system, lane, reference),
                        _source_v4(lane),
                        _percent(metrics.get("edition_note_f1")),
                        _percent(metrics.get("chord_f1")),
                        _full_meter(lane),
                        _full_form(lane, reference),
                    ]
                )
            else:
                metrics = lane["metrics"]["summary"] if lane else {}
                scorable = lane and lane["submission_validation"]["headline_metrics_valid"]
                absent = "Unscorable" if lane else "—"
                quality.append(
                    prefix
                    + [
                        _status(system, lane),
                        (
                            _fraction(metrics.get("exact_measures"), metrics.get("targets"))
                            if scorable
                            else absent
                        ),
                        _percent(metrics.get("note_f1")) if scorable else absent,
                        _percent(metrics.get("rest_f1")) if scorable else absent,
                    ]
                )
            runtime = lane["runtime"] if lane else {}
            elapsed, billing = _elapsed(runtime), _billing(runtime, identifier)
            if full_score:
                deadline = runtime.get("deadline_seconds")
                elapsed += (
                    f'<small>Time allowance: {_elapsed({"elapsed_seconds": deadline})}</small>'
                    if deadline is not None
                    else "<small>Time allowance not recorded</small>"
                )
                if identifier == "api" and runtime.get("budget_usd") is not None:
                    billing += (
                        f'<small>Stage budget setting: ${_money(runtime["budget_usd"])}</small>'
                    )
            resources.append(
                prefix
                + [
                    elapsed,
                    _usage(_usage_runtime(runtime)) + f"<small>{_reasoning(runtime)}</small>",
                    _tool_counts(runtime),
                    billing,
                ]
            )
            if lane:
                fields = {
                    key: runtime.get(key)
                    for key in (
                        "status",
                        "returncode",
                        "complete",
                        "completed_turn",
                        "auth_mode",
                        "transport",
                        "tool_mode",
                        "native_tool_mode",
                        "resolved_model",
                        "usage",
                        "billing",
                        "cost_estimate_usd",
                        "cost_basis",
                        "budget_usd",
                        "cost_usd",
                        "deadline_seconds",
                        "completion_policy",
                        "completion_gate",
                    )
                }
                if full_score:
                    fields.update(
                        written_coverage=lane.get("coverage"),
                        notation_validation=lane.get("validation"),
                        source_v3={
                            "source_note_f1": lane["metrics"].get("source_note_f1"),
                            "scope": "Earlier v3 audit; not the displayed v4 source metric",
                        },
                        supplemental_source=lane.get("supplemental_source"),
                    )
                details.append(
                    f'<h3>{text(system["label"])} · {text(label)}</h3><pre>{text(json.dumps(fields, indent=2, sort_keys=True))}</pre>'
                )
        if report:
            details.append(
                f'<p>{_link(system["path"], output, system["label"] + " evaluation report")}</p>'
            )
    differences = manifest.get("protocol_differences", [])
    difference_html = (
        "<ul>" + "".join(f"<li>{text(v)}</li>" for v in differences) + "</ul>"
        if differences
        else "<p>Detailed protocol differences have not been recorded in this manifest.</p>"
    )
    optional = {key: manifest[key] for key in ("status", "budget") if key in manifest}
    if full_score:
        optional["reference_summary"] = reference
    details.append(f'<p>{_link(comparison["path"], output, "Pinned comparison manifest")}</p>')
    if optional:
        details.append(
            f"<h3>Pilot status and budget record</h3><pre>{text(json.dumps(optional, indent=2, sort_keys=True))}</pre>"
        )
    demo = (
        '<p class="notice demo">SYNTHETIC CONTROL — test data, not a new model result.</p>'
        if comparison["synthetic"]
        else ""
    )
    status = f'<p class="notice">{text(manifest["status"])}</p>' if manifest.get("status") else ""
    quality_table = _table(
        (
            [
                "Transport",
                "Stage",
                "Execution / written coverage",
                "Audited source notes · v4",
                "Edition note agreement",
                "Edition chord agreement",
                "Meter / completion gate",
                "Edition playback form",
            ]
            if full_score
            else [
                "Transport",
                "Stage",
                "Execution / coverage",
                "Exact event bars",
                "Strict note F1",
                "Rest F1",
            ]
        ),
        quality,
        "quality",
    )
    resource_table = _table(
        ["Transport", "Stage", "Elapsed", "Reported tokens", "Source tools", "Cost evidence"],
        resources,
        "resources",
    )
    title = "Rumichaca" if full_score else "Aviador"
    target_label = "Rumichaca · complete score" if full_score else "Aviador system 8"
    scope = (
        "Source v4 measures only the explicitly audited PDF bars and noteheads shown; it includes pitch, onset, duration and ties. The earlier v3 audit stays separate in the evidence details. Edition notes, chords and form measure agreement with the chosen human performing edition, including reconstructions. Neither is whole-score source accuracy. Equal performed-bar counts do not prove the same playback route. Meter validity and a passed completion gate do not prove musical accuracy. No quality scores are pooled."
        if full_score
        else "Seven-bar excerpt. Strict note F1 matches pitch, onset and duration together. Exact event bars also require matching rests. Ties, form, chord symbols and engraving are outside this metric. Partial execution remains partial even when a saved draft claims completion. No quality scores are pooled."
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Astra transport comparison · {title}</title><style>{CSS}{EXTRA_CSS}</style></head><body>
<header class="mast"><div class="wrap"><div class="eyebrow">Score2ABC · {target_label}</div><h1>Subscription and API transcription</h1><p class="lede">One author draft and one independent AI review per transport. Read quality and completion first, then time, tokens and cost evidence. Each stage stays separate.</p>{demo}{status}<div class="cards">{"".join(cards)}</div></div></header>
<main class="wrap"><section class="section"><div class="section-head"><h2>What was transcribed correctly?</h2></div>{quality_table}<p class="scope">{scope}</p></section>
<section class="section"><div class="section-head"><h2>What did each stage use?</h2></div>{resource_table}<p class="scope">Cached input is included in input; reasoning output is included in output. Do not add these subsets again. Missing usage is unknown, never zero. API estimates are distinct from billed charges; no subscription per-run cost is inferred.</p></section>
<section class="section"><div class="section-head"><h2>How comparable are these runs?</h2></div><p>The source-reading workflow is shared, but Codex code-mode orchestration and API native function calls differ. These results describe one consumed example; they do not isolate a transport effect or establish accuracy on unseen scores. Independent AI review is not human approval.</p>{difference_html}<details class="details"><summary>Reports, runtime and billing evidence</summary>{"".join(details)}</details></section></main>
<footer><div class="wrap">Offline report · no source/truth files opened, remote resources or model calls. {text(manifest["comparison_id"])}</div></footer></body></html>"""


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
