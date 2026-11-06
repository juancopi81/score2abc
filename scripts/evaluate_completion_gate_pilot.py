"""Offline Rumichaca completion-gate comparison; no inference or historical edits."""

# ruff: noqa: E501  # Standalone HTML remains beside its report renderer.

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_api_full_score_pilot import (  # noqa: E402
    CSS,
    grader,
    normalize_abc,
    resource_row,
    runner,
    table,
    text,
)
from scripts.experiments import completion_gate  # noqa: E402

STAGES = ("initial", "reviewed")


def verify_live_reviews(snapshot_path: Path) -> dict:
    """Validate the preregistered wrapper and recheck each protected human draft."""
    snapshot = grader.read_json(snapshot_path)
    if (
        not isinstance(snapshot, dict)
        or set(snapshot) != {"before", "files", "status"}
        or snapshot.get("status") != "snapshotted_before_inference"
        or not isinstance(snapshot.get("before"), dict)
        or type(snapshot.get("files")) is not int
        or snapshot["files"] != len(snapshot["before"])
        or snapshot["files"] < 1
    ):
        raise ValueError("Invalid preregistered human-review preservation snapshot")
    for relative, expected in snapshot["before"].items():
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or not isinstance(expected, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected) is None
        ):
            raise ValueError("Invalid human-review preservation path or digest")
        path = ROOT / relative
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Human-review preservation path must stay within the repository")
        if not path.is_file() or grader.sha(path) != expected:
            raise ValueError("Human review files changed during execution")
    return snapshot["before"]


def validate_audit(audit: dict, plan: dict, packet_hash: str) -> None:
    expected = {
        "status": "passed",
        "backend": "api",
        "input_seal_sha256": packet_hash,
        "model": plan["model"],
        "reasoning_effort": plan["reasoning_effort"],
        "reference_access": False,
        "tools": ["inspect_source", "save_draft"],
        "completion_policy": plan["completion_policy"],
        "completion_policy_sha256": plan["completion_policy_sha256"],
    }
    if any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage audit is not a passing audit of its gated input packet")


def validate_renderer(baseline: dict) -> None:
    if baseline.get("implementations", {}).get("renderer") != grader.renderer_identity():
        raise ValueError("Installed renderer differs from the numerical-grading baseline")


def validate_spending(actual: dict, contract: dict) -> None:
    expected = {
        "spending_policy": contract["spending_policy"],
        "max_output_tokens_per_response": contract["max_output_tokens_per_response"],
        "budget_usd": None,
    }
    if any(key not in actual or actual[key] != value for key, value in expected.items()):
        raise ValueError("Actual spending/output allowance differs from the fixed run contract")


def validate_packet(bundle: Path, metadata: dict, plan: dict, stage: str) -> None:
    expected = {
        "backend": "api",
        "model": plan["model"],
        "reasoning_effort": plan["reasoning_effort"],
        "source_pdf_sha256": plan["target"]["source_pdf"]["sha256"],
        "limit_seconds": plan["deadline_seconds"][stage],
        "reference_access": False,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Prepared stage differs from the experiment contract")
    if (
        grader.sha(bundle / "source.pdf") != expected["source_pdf_sha256"]
        or grader.sha(bundle / "prompt.txt") != metadata["prompt_sha256"]
    ):
        raise ValueError("Prepared source or task bytes differ from their sealed declarations")
    runner.verify_source_bundle(bundle, metadata)


def validate_stage(provenance: dict, plan: dict, stage: str, packet_hash: str, author_hash) -> None:
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
        "completion_policy": plan["completion_policy"],
        "completion_policy_sha256": plan["completion_policy_sha256"],
    }
    if any(provenance.get(k) != v for k, v in expected.items()):
        raise ValueError(f"{stage} differs from the source-only gated contract")
    if provenance.get("status") not in {"complete", "partial", "failed"}:
        raise ValueError("Stage must have a terminal execution status")
    if provenance.get("status") == "complete" and (
        provenance.get("complete") is not True
        or provenance.get("returncode") != 0
        or provenance.get("completed_turn") is not True
        or provenance.get("timed_out") is not False
        or provenance.get("quarantined") is not False
        or provenance.get("error") is not None
    ):
        raise ValueError("Completed stage cannot hide an interrupted or quarantined execution")
    if datetime.fromisoformat(provenance["started_at"]) < datetime.fromisoformat(
        plan["created_at"]
    ):
        raise ValueError("Both stages must be fresh after preregistration")


def validate_pair_status(pair: dict) -> list[str]:
    """A final pair record, rather than absent files, establishes that execution stopped."""
    stages = pair.get("stages", {})
    if set(stages) not in ({"initial"}, set(STAGES)):
        raise ValueError("Pair requires an author and an optional independent reader")
    if pair.get("backend") != "api" or pair.get("completion_policy") != completion_gate.POLICY_ID:
        raise ValueError("Terminal pair record uses another backend or policy")
    if pair.get("automatic_retries") != 0:
        raise ValueError("This pilot does not admit automatic retries")
    if (
        pair.get("no_dollar_cap") is not True
        or "budget_usd" not in pair
        or pair["budget_usd"] is not None
    ):
        raise ValueError("Pair spending policy differs from the fixed-output pilot")
    if any(s.get("status") not in {"complete", "partial", "failed"} for s in stages.values()):
        raise ValueError("Pair still has nonterminal stages")
    author_complete = (
        stages["initial"].get("status") == "complete" and stages["initial"].get("complete") is True
    )
    if ("reviewed" in stages) != author_complete:
        raise ValueError("Reader must run exactly when the author completes this pair")
    actual_complete = len(stages) == 2 and all(
        s.get("status") == "complete" and s.get("complete") is True for s in stages.values()
    )
    if pair.get("complete") is not actual_complete:
        raise ValueError("Terminal pair completion differs from stage outcomes")
    return [stage for stage in STAGES if stage in stages]


def verify_inputs(root: Path) -> tuple[dict, dict, dict, list]:
    """Verify terminal sealed outputs before any reference or candidate content is graded."""
    plan = grader.read_json(root / "evaluation_plan.json")
    contract = grader.read_json(root / "execution_contract.json")
    policy_hash = grader.sha(Path(completion_gate.__file__))
    if (
        plan.get("kind") != "astra_api_full_score_pilot_plan"
        or plan.get("status") != "preregistered_before_new_author_inference"
        or plan.get("target_id") != "rumichaca_full"
        or plan.get("source_policy") != "supplied_pdf_only"
        or plan.get("review_instructions_version") != "assisted_review_v2"
        or plan.get("deadline_seconds") != {"initial": 3600, "reviewed": 1800}
        or plan.get("completion_policy") != completion_gate.POLICY_ID
        or plan.get("completion_policy_sha256") != policy_hash
        or plan.get("reference_feedback_given") is not False
        or contract.get("plan_sha256") != grader.sha(root / "evaluation_plan.json")
        or contract.get("completion_policy") != plan["completion_policy"]
        or contract.get("completion_policy_sha256") != policy_hash
        or contract.get("deadline_seconds") != plan["deadline_seconds"]
        or contract.get("automatic_retries") != 0
        or contract.get("review_requires_complete_author") is not True
        or contract.get("spending_policy") != "fixed_output_no_dollar_cap"
        or contract.get("max_output_tokens_per_response") != 25000
        or contract.get("model") != plan.get("model")
        or contract.get("reasoning_effort") != plan.get("reasoning_effort")
        or contract.get("reference_feedback_given") is not False
    ):
        raise ValueError("Unexpected completion-gate pilot contract")
    records = [
        grader.pin(root / name) for name in ("evaluation_plan.json", "execution_contract.json")
    ]
    for key in (
        "author_task",
        "review_task",
        "review_instructions",
        "subscription_baseline_report",
        "subscription_baseline_plan",
        "previous_api_report",
        "previous_api_plan",
        "aviador_regression_report",
        "offline_gate_report",
    ):
        records.append(grader.pin(grader.verify_pin(ROOT, plan[key], key)))
    for entry in plan["implementation_snapshots"]:
        path = grader.verify_pin(ROOT, entry, "implementation snapshot")
        relative = path.relative_to(root / "setup/implementation")
        if grader.sha(ROOT / relative) != entry["sha256"]:
            raise ValueError(f"Implementation changed during execution: {relative}")
        records.append(grader.pin(path))
    target = root / "rumichaca_full"
    pair_path = target / "pair_status.json"
    pair = grader.read_json(pair_path)
    present = validate_pair_status(pair)
    records.append(grader.pin(pair_path))
    stages, author_hash = {}, None
    for stage in STAGES:
        folder, packet = target / stage, target / f"{stage}_packet"
        if stage not in present:
            if any((target / (stage + suffix)).exists() for suffix in ("", "_packet", "_audit")):
                raise ValueError("Unreported reviewer artifacts require investigation")
            continue
        for suffix, kind in (("", "stage"), ("_packet", "input"), ("_audit", "audit")):
            sealed = target / (stage + suffix)
            runner.verify_seal(sealed, f"astra_reproduction_{kind}")
            records.extend(grader.pin(p) for p in sorted(sealed.rglob("*")) if p.is_file())
        provenance = grader.read_json(folder / "stage_provenance.json")
        validate_audit(
            grader.read_json(target / f"{stage}_audit/audit.json"),
            plan,
            grader.sha(packet / "seal.json"),
        )
        if provenance != pair["stages"][stage]:
            raise ValueError("Terminal pair record differs from sealed stage provenance")
        validate_stage(provenance, plan, stage, grader.sha(packet / "seal.json"), author_hash)
        metadata = grader.read_json(packet / "preparation.json")
        validate_packet(packet / "bundle", metadata, plan, stage)
        validate_spending(metadata, contract)
        validate_spending(provenance, contract)
        task = "author_task" if stage == "initial" else "review_task"
        if (
            metadata.get("prompt_template_sha256") != plan[task]["sha256"]
            or metadata.get("completion_policy") != plan["completion_policy"]
            or metadata.get("completion_policy_sha256") != policy_hash
        ):
            raise ValueError("Prepared task or completion policy differs from preregistration")
        selected = provenance.get("selected_checkpoint")
        if selected is not None and not re.fullmatch(r"artifacts/drafts/draft_\d{3}", selected):
            raise ValueError("Selected checkpoint must be a fixed draft path")
        ready, gate_provenance = runner.checkpoint_completion(
            folder / selected if selected else None, metadata
        )
        if ready is not provenance.get("complete") or gate_provenance[
            "completion_gate"
        ] != provenance.get("completion_gate"):
            raise ValueError("Saved gate outcome does not reproduce from checkpoint evidence")
        candidate = folder / "score.abc"
        if candidate.exists():
            if not selected or grader.sha(candidate) != grader.sha(folder / selected / "score.abc"):
                raise ValueError("Exported candidate does not match selected checkpoint")
            if grader.sha(folder / "decisions.json") != grader.sha(
                folder / selected / "decisions.json"
            ):
                raise ValueError("Exported decisions do not match selected checkpoint")
            if stage == "initial":
                author_hash = grader.sha(candidate)
        stages[stage] = {
            "abc": candidate if candidate.exists() else None,
            "provenance_data": provenance,
            "gate": gate_provenance["completion_gate"],
            "folder": folder,
        }
    protected_path = root / "live_review_preservation.json"
    verify_live_reviews(protected_path)
    records.append(grader.pin(protected_path))
    return plan, pair, stages, records


def grade_available(target: dict, stages: dict, records: list) -> dict:
    """Use unchanged numeric graders; preserve a missing review as not run, never a copy."""
    frozen_path = grader.verify_pin(ROOT, target["frozen_reference"], "frozen six reference")
    labels_path = grader.verify_pin(ROOT, target["labels"], "V3 labels")
    frozen = grader.verify_freeze(frozen_path.parent)
    labels = grader.verify_labels(labels_path.parent, frozen_path.parent)
    work = next(w for w in frozen["works"] if w["slug"] == target["slug"])
    work_labels = next(w for w in labels["works"] if w["slug"] == target["slug"])
    if (
        frozen["files"][work["source_pdf_path"]]["sha256"] != target["source_pdf"]["sha256"]
        or labels.get("label_set") != "reviewed_six_v3"
    ):
        raise ValueError("Rumichaca source or labels differ from frozen grading contract")
    reference = normalize_abc((frozen_path.parent / work["answer_abc_path"]).read_text())
    if not reference["valid"]:
        raise ValueError("Frozen reference could not be normalized")
    required = grader._source_required_edition_bars(reference, work_labels)
    measured = {}
    for stage, data in stages.items():
        runtime = data["provenance_data"]
        if data["abc"] is None or runtime.get("quarantined"):
            measured[stage] = {
                "status": "not_scored",
                "metrics": {},
                "coverage": {},
                "validation": {},
                "runtime": runtime,
                "completion_gate": data["gate"],
                "decisions": [],
                "reason": "No saved candidate or execution was quarantined.",
            }
            continue
        abc = data["abc"].read_text()
        decisions = grader.read_json(data["folder"] / "decisions.json")
        candidate = normalize_abc(abc)
        lane = grader.compare_normalized(reference, candidate)
        source = grader.score_source(reference, candidate, lane.get("alignment", []), work_labels)
        for field in ("metrics", "metric_status", "coverage"):
            lane[field].update(source[field])
        lane["mismatches"].extend(source["mismatches"])
        feedback = grader.validate_candidate(abc, decisions)
        paired = {r["reference"] for r in lane.get("alignment", []) if r["candidate"] is not None}
        status = (
            "failed"
            if not candidate["valid"]
            else (
                "complete"
                if not feedback["issues"] and required.issubset(paired) and runtime["complete"]
                else "partial"
            )
        )
        if runtime["status"] != "complete" or runtime["returncode"] != 0:
            status = "partial" if candidate["valid"] else "failed"
        lane.update(
            status=status,
            validation=feedback,
            decisions=decisions,
            source_scope=source["source_scope"],
            runtime=runtime,
            completion_gate=data["gate"],
        )
        measured[stage] = lane
    records.extend([grader.pin(frozen_path), grader.pin(labels_path)])
    return measured


def pct(value) -> str:
    return "Not measured" if value is None else f"{value:.1%}"


def cost_total(resources: dict) -> str | None:
    estimates = [item.get("conservative_estimate_usd") for item in resources.values()]
    return (
        str(sum((Decimal(v) for v in estimates), Decimal(0)))
        if estimates and all(v is not None for v in estimates)
        else None
    )


def render(report: dict) -> str:
    final = report["stages"].get("reviewed")
    chosen = final or report["stages"].get("initial", {})
    gate = chosen.get("completion_gate", {})
    metric = chosen.get("metrics", {})
    issues = chosen.get("validation", {}).get("issues", [])
    exceptions = gate.get("accepted_source_exceptions", [])
    prior = report["previous_api"]["reviewed"]["metrics"]["edition_note_f1"]
    current = metric.get("edition_note_f1")
    delta = (
        "No reviewed result"
        if not final
        else "Not measured" if current is None else f"{(current-prior)*100:+.1f} percentage points"
    )
    if not final:
        headline = "Author stopped; independent review did not run"
        finding = "The saved author result is shown as a partial diagnostic. There is no new reviewed result to compare."
    elif not report["pair_complete"]:
        headline = "Review stopped; preserve the partial result"
        finding = "The independent reviewer did not meet the execution/completion contract. Do not treat this as a successful finished score."
    elif exceptions:
        headline = "Gate passed with documented source exceptions"
        finding = "Every structural issue has a recorded disposition, but the source exceptions still need musical judgment."
    else:
        headline = "Gate passed without source exceptions"
        finding = "The submitted final draft cleared structural checks. Compare its notes and form separately before scaling."
    music_rows, resource_rows = [], []
    groups = (
        ("Subscription · review v1", report["subscription_baseline"]),
        ("API · no gate · review v2", report["previous_api"]),
        ("API · completion gate · review v2", report["stages"]),
    )
    for label, lanes in groups:
        lane = lanes.get("reviewed")
        if lane:
            m, coverage = lane.get("metrics", {}), lane.get("coverage", {})
            count = lane.get("validation", {}).get("counts", {}).get("performed_measures", "?")
            route = (
                "Not measured"
                if m.get("form_exact_match") is None
                else ("Matches edition" if m["form_exact_match"] else "Differs from edition")
            )
            music_rows.append(
                [
                    label,
                    f"{coverage.get('candidate_measures', '?')}/{coverage.get('reference_measures', '?')}",
                    pct(m.get("edition_note_f1")),
                    pct(m.get("source_note_f1")),
                    pct(m.get("chord_f1")),
                    pct(m.get("meter_valid_rate")),
                    f"{route}; {count} performed bars",
                ]
            )
        else:
            music_rows.append([label, "Review not run", "—", "—", "—", "—", "—"])
        for stage, lane in lanes.items():
            resource = resource_row(lane.get("runtime", {}))
            seconds = resource["seconds"]
            duration = "Unknown" if seconds is None else f"{seconds/60:.1f} min"
            estimate = resource["conservative_estimate_usd"]
            money = "Unknown" if estimate is None else f"${Decimal(estimate):.2f} estimate"
            resource_rows.append(
                [
                    label,
                    "Author" if stage == "initial" else "Review",
                    duration,
                    resource["input_tokens"],
                    resource["output_tokens"],
                    pct(resource["cache_fraction"]),
                    money,
                ]
            )
    gate_rows, details = [], []
    for stage, lane in report["stages"].items():
        g = lane["completion_gate"]
        raw = lane.get("validation", {}).get("issues", [])
        accepted = g.get("accepted_source_exceptions", [])
        gate_rows.append(
            [
                "Author" if stage == "initial" else "Review",
                lane["runtime"]["status"],
                g.get("status", "Unknown"),
                len(g.get("unresolved_issues", [])),
                len(accepted),
                len(raw),
                lane["status"],
            ]
        )
        for item in accepted:
            resolution = item["resolution"]
            details.append(
                f"<li><strong>{text(stage)} · {text(resolution['location'])}</strong>: {text(resolution['adopted_choice'])}<p>{text(resolution['reason'])}</p><small>{text(resolution['uncertainty'])}; inspected crop call {text(resolution['inspection_call'])}</small></li>"
            )
    total = report["conservative_estimate_usd"]
    total_text = "Unknown cost" if total is None else f"${Decimal(total):.2f} estimated"
    links = ['<a href="evaluation/report.json">Full report and version pins</a>']
    for stage, lane in report["stages"].items():
        if lane["status"] != "not_scored":
            links.append(f'<a href="rumichaca_full/{stage}/score.abc">{text(stage)} ABC</a>')
    limits = "".join(f"<li>{text(item)}</li>" for item in report["limitations"])
    musical = table(
        [
            "Astra xhigh · reviewed result",
            "Written bars",
            "Edition notes",
            "Source notes · 9 bars",
            "Chords",
            "Meter valid",
            "Playback form",
        ],
        music_rows,
    )
    resources = table(
        ["Workflow", "Stage", "Time", "Input tokens", "Output tokens", "Cached input", "Cost"],
        resource_rows,
    )
    gates = table(
        [
            "Stage",
            "Execution",
            "Completion gate",
            "Unresolved",
            "Source exceptions",
            "Raw validation issues",
            "Music validation",
        ],
        gate_rows,
    )
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>Rumichaca · completion gate comparison</title><style>{CSS}.wrap{{max-width:1450px}}thead th:first-child{{width:auto}}.table-scroll table{{min-width:1000px}}</style><body>
<header class="mast"><div class="wrap"><p class="eyebrow">Rumichaca · Astra xhigh · API completion gate v1</p><h1>{text(headline)}</h1><p class="lede">{text(finding)} Same PDF, author/review task texts and 60/30-minute limits as the earlier API run. The gate changes the save-tool feedback and finish rule.</p></div></header>
<main class="wrap"><div class="cards"><div class="card"><strong>{pct(current)} edition note F1</strong><small>{text(delta)} versus the previous API review{' — author only, not comparable as a reviewed result' if not final else ''}</small></div><div class="card"><strong>{len(issues)} raw validation issues · {len(exceptions)} exceptions</strong><small>Structural readiness and musical correctness are separate</small></div><div class="card"><strong>{text(total_text)}</strong><small>All attempted API stages; actual billing unverified</small></div></div>
<section class="section"><h2>Compare the reviewed music</h2>{musical}<p class="scope">Notes compare pitch, onset and duration jointly. Source-note labels cover nine pre-existing model-audited bars only. Chords and form are compared with the adopted human edition, which contains editorial choices. A single fresh run cannot isolate the gate's causal effect.</p></section>
<section class="section"><h2>Did the completion gate do its job?</h2>{gates}<p class="scope">The gate is recomputed from saved checkpoints. A source exception records a model decision and inspected crop provenance; it does not independently prove that the manuscript has an irregular bar. Raw warnings remain visible even when the gate accepts an exception.</p><details class="details"><summary>Recorded source exceptions</summary><ul>{''.join(details) or '<li>None recorded.</li>'}</ul></details></section>
<section class="section"><h2>Time, tokens and cost</h2>{resources}<p class="scope">Cached input is included within input tokens. API estimates use the pinned rate table, not an invoice. Subscription dollar cost is unknown. No automatic retries or six-score batch were run.</p></section>
<section class="section"><h2>What to review next</h2><p>Inspect any source exceptions, new musical disagreements and the executed form before scaling. Passing meter checks alone cannot establish correct pitches, complete PDF coverage or human approval.</p><details class="details"><summary>Scope and trace evidence</summary><ul>{limits}</ul><pre>{text(json.dumps(report['traces'], indent=2))}</pre><p>{report['live_reviews_unchanged']} live human review files remain unchanged. Requests, response receipts, usage, inspections, decisions and checkpoints are preserved locally. Private response contents are not embedded here.</p></details></section>
<p class="links">{' '.join(links)}<a href="../astra_api_rumichaca_20261029_v1/index.html">Previous API run</a><a href="../astra_workflow_20261028_v1/index.html">Subscription baseline</a></p></main></body></html>"""


def evaluate(root: Path) -> dict:
    root = root.resolve()
    output = root / "evaluation"
    if output.exists():
        raise FileExistsError("Evaluation is create-once")
    plan, pair, stages, records = verify_inputs(root)
    baseline = grader.read_json(
        grader.verify_pin(ROOT, plan["subscription_baseline_report"], "subscription report")
    )
    baseline_plan = grader.read_json(
        grader.verify_pin(ROOT, plan["subscription_baseline_plan"], "subscription plan")
    )
    previous = grader.read_json(
        grader.verify_pin(ROOT, plan["previous_api_report"], "previous API report")
    )
    previous_plan = grader.read_json(
        grader.verify_pin(ROOT, plan["previous_api_plan"], "previous API plan")
    )
    for key in ("source_pdf", "frozen_reference", "labels"):
        if (
            plan["target"][key] != baseline_plan["targets"]["rumichaca_full"][key]
            or plan["target"][key] != previous_plan["target"][key]
        ):
            raise ValueError("Baseline source or numerical grading target differs")
    for key in ("author_task", "review_task", "review_instructions"):
        if plan[key]["sha256"] != previous_plan[key]["sha256"]:
            raise ValueError("Source task/review instructions changed beyond completion gate")
    for key in ("abc_normalizer", "abc_parser", "complete_score_grader", "source_grader"):
        grader.verify_pin(ROOT, baseline["implementations"][key], "unchanged baseline grader")
    validate_renderer(baseline)
    submission = root / "rumichaca_full/pilot_submission_seal.json"
    if submission.exists():
        raise FileExistsError("Submission seal is create-once")
    runner.write(
        submission,
        {
            "kind": "completion_gate_pilot_submission_seal",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "plan_sha256": grader.sha(root / "evaluation_plan.json"),
            "terminal_pair_verified": True,
            "stages_verified_before_reference_content": list(stages),
            "reference_feedback_given": False,
            "artifacts": records,
        },
    )
    records.append(grader.pin(submission))
    measured = grade_available(plan["target"], stages, records)
    resources = {stage: resource_row(data["provenance_data"]) for stage, data in stages.items()}
    traces = {
        stage: {
            "generation_requests": len(list((data["folder"] / "api").glob("request_*.json"))),
            "source_crops": len(list((data["folder"] / "artifacts/inspections").glob("*.png"))),
            "draft_checkpoints": len(
                list((data["folder"] / "artifacts/drafts").glob("draft_*/checkpoint.json"))
            ),
        }
        for stage, data in stages.items()
    }
    report = {
        "schema_version": 1,
        "kind": "astra_api_completion_gate_pilot_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": plan["experiment_id"],
        "model": plan["model"],
        "reasoning_effort": plan["reasoning_effort"],
        "completion_policy": plan["completion_policy"],
        "completion_policy_sha256": plan["completion_policy_sha256"],
        "review_instructions_version": plan["review_instructions_version"],
        "terminal_pair_verified_before_reference_content": True,
        "both_stages_verified_before_reference_content": len(stages) == 2,
        "pair_complete": pair["complete"],
        "review_not_run": "reviewed" not in stages,
        "scorer": "unchanged complete-score-v2 plus reviewed_six_v3 audited source subset",
        "stages": measured,
        "subscription_baseline": baseline["stages"],
        "previous_api": previous["stages"],
        "resources": resources,
        "conservative_estimate_usd": cost_total(resources),
        "actual_billed_usd": None,
        "traces": traces,
        "live_reviews_unchanged": len(verify_live_reviews(root / "live_review_preservation.json")),
        "limitations": plan["limitations"],
        "experimental_change": plan["experimental_change"],
        "pins": [*records, grader.pin(Path(__file__))],
    }
    for entry in report["pins"]:
        grader.verify_pin(ROOT, entry, "final artifact check")
    page = render(report)
    portable = ROOT / "docs/eval" / f"{root.name}.json"
    if portable.exists():
        raise FileExistsError("Portable summary is create-once")
    output.mkdir()
    runner.write(output / "report.json", report)
    (root / "index.html").write_text(page)
    runner.write(
        portable,
        {
            key: value
            for key, value in report.items()
            if key not in ("pins", "subscription_baseline", "previous_api")
        }
        | {"report_sha256": grader.sha(output / "report.json")},
    )
    runner.write(
        root / "dashboard_manifest.json",
        {
            "kind": "completion_gate_pilot_dashboard",
            "report": grader.pin(output / "report.json"),
            "dashboard": grader.pin(root / "index.html"),
            "builder": grader.pin(Path(__file__)),
            "portable_summary": grader.pin(portable),
        },
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = evaluate(args.root)
    print(
        json.dumps(
            {
                "pair_complete": result["pair_complete"],
                "resources": result["resources"],
                "stages": {
                    stage: {
                        "status": lane["status"],
                        "metrics": lane["metrics"],
                        "gate_status": lane["completion_gate"]["status"],
                    }
                    for stage, lane in result["stages"].items()
                },
            },
            indent=2,
        )
    )
