"""Summarize saved API batch results locally as Markdown."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from html import escape
from pathlib import Path

STAGES = ("initial", "reviewed")
STAGE_LABELS = {"initial": "Author", "reviewed": "Reader"}
PAIR_LABELS = {
    "complete": "Both passes finished",
    "incomplete": "Stopped before both passes finished",
    "interrupted": "Interrupted",
    "running": "In progress",
}


def text(value: object) -> str:
    value = escape(" ".join(str(value).split()), quote=False)
    return re.sub(r"([\\`*_\[\]|])", r"\\\1", value)


def read_sealed(stage: Path, seal: dict, filename: str):
    path = stage / filename
    if not path.resolve().is_relative_to(stage.resolve()):
        raise ValueError(f"Artifact outside the stage: {filename}")
    data = path.read_bytes()
    if seal["files"].get(filename) != hashlib.sha256(data).hexdigest():
        raise ValueError(f"Saved stage artifact changed: {filename}")
    return json.loads(data)


def read_stage(stage: Path) -> dict | None:
    seal_path = stage / "seal.json"
    if not seal_path.is_file():
        return None
    seal = json.loads(seal_path.read_bytes())
    if seal.get("kind") != "astra_reproduction_stage" or not isinstance(seal.get("files"), dict):
        raise ValueError(f"Unsupported stage seal: {stage}")
    summary = read_sealed(stage, seal, "stage_provenance.json")
    if summary.get("auth_mode") != "openai_api" or summary.get("stage") != stage.name:
        raise ValueError(f"Unexpected stage provenance: {stage}")
    checkpoint = summary.get("selected_checkpoint")
    if checkpoint:
        if not re.fullmatch(r"artifacts/drafts/draft_[0-9]+", checkpoint):
            raise ValueError(f"Unexpected checkpoint path: {stage}")
        summary["validation"] = read_sealed(stage, seal, f"{checkpoint}/validation.json")
        summary["decisions"] = read_sealed(stage, seal, "decisions.json")
    value = summary.get("cost_estimate_usd")
    try:
        cost = Decimal(value) if value is not None else None
    except InvalidOperation as error:
        raise ValueError(f"Invalid recorded cost: {stage}") from error
    if cost is not None and (not cost.is_finite() or cost < 0):
        raise ValueError(f"Invalid recorded cost: {stage}")
    summary["estimated_cost"] = cost
    return summary


def clock(seconds: float) -> str:
    minutes, remainder = divmod(round(seconds), 60)
    return f"{minutes}m {remainder:02d}s"


def stage_label(stage: dict | None) -> str:
    if stage is None:
        return "No saved result"
    status = stage.get("status", "unknown")
    if stage.get("timed_out"):
        status = "time limit reached"
    elif stage.get("budget_exhausted"):
        status = "budget limit reached"
    gate = stage.get("completion_gate", {}).get("status")
    elapsed = clock(stage["elapsed_seconds"])
    return text(f"{status}, {elapsed}" + (f", gate {gate}" if gate else ""))


def is_complete(stage: dict | None) -> bool:
    return bool(stage and stage.get("status") == "complete" and stage.get("complete") is True)


def review_items(stage: dict) -> list[str]:
    decisions = stage.get("decisions", [])
    questions = [
        decision for decision in decisions if decision.get("category") == "human_review_question"
    ]
    selections = questions or [decision for decision in decisions if decision.get("alternatives")]
    if not selections:
        return []
    label = "Questions for the musician" if questions else "Recorded alternatives to review"
    lines = ["", f"**{label} ({len(selections)})**", ""]
    for decision in selections[:5]:
        location = text(decision.get("location") or "Score")
        detail = text(
            decision.get("uncertainty") or decision.get("adopted_choice") or "See decision"
        )
        lines.append(f"- **{location}** — {detail}")
    if len(selections) > 5:
        lines.append(f"- See `decisions.json` for the remaining {len(selections) - 5} entries.")
    return lines


def score_details(key: str, title: str, stages: dict) -> list[str]:
    lines = ["", f"## {text(title)}", ""]
    for name, stage in stages.items():
        if not stage:
            continue
        gate = stage.get("completion_gate", {})
        for issue in gate.get("unresolved_issues", []):
            lines.append(
                f"- {STAGE_LABELS[name]} needs attention — {text(issue.get('detail', ''))}"
            )
        exceptions = len(gate.get("accepted_source_exceptions", []))
        if exceptions:
            lines.append(f"- {STAGE_LABELS[name]} documented {exceptions} source exceptions.")
    if lines[-1]:
        lines.append("")
    selected = (
        "reviewed" if stages["reviewed"] and stages["reviewed"].get("validation") else "initial"
    )
    stage = stages[selected]
    if not stage or not stage.get("validation"):
        return lines + ["No sealed draft checkpoint available."]
    validation = stage["validation"]
    counts = validation.get("counts", {})
    written = counts.get("measures")
    performed = counts.get("performed_measures")
    validation_status = "passed" if validation.get("valid") else "needs attention"
    lines.append(
        f"{STAGE_LABELS[selected]} draft at `works/{key}/{selected}/score.abc`. "
        f"Notation check {validation_status}. {text(written)} written bars and "
        f"{text(performed)} played bar visits."
    )
    lines.append("")
    for decision in stage.get("decisions", []):
        if isinstance(decision.get("rows"), list):
            lines.append(f"- The draft's source map lists {len(decision['rows'])} manuscript rows.")
        route = decision.get("performed_written_bars")
        if performed is not None and isinstance(route, list) and len(route) != performed:
            lines.append(
                f"- **Route needs checking.** The declared route has {len(route)} visits "
                f"and the playback check counted {text(performed)}."
            )
    return lines + review_items(stage)


def build_report(batch: Path) -> str:
    raw = (batch / "plan.json").read_bytes()
    plan = json.loads(raw)
    status = json.loads((batch / "batch_status.json").read_bytes())
    if (
        plan.get("auth_mode") != "openai_api"
        or status.get("plan_sha256") != hashlib.sha256(raw).hexdigest()
    ):
        raise ValueError("Expected an API batch with a matching saved plan")
    ceiling = plan["budget"]["batch_ceiling_usd"]

    rows, details, saved = [], [], []
    states = {item["key"]: item["state"] for item in status["pairs"]}
    seen = set()
    completed = 0
    for work in plan["works"]:
        key = work["key"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", key) or key in seen:
            raise ValueError("Expected unique batch score keys")
        seen.add(key)
        title = work["title"]
        stages = {name: read_stage(batch / "works" / key / name) for name in STAGES}
        work_results = [stage for stage in stages.values() if stage]
        saved.extend(work_results)
        costs = [
            stage["estimated_cost"] for stage in work_results if stage["estimated_cost"] is not None
        ]
        cost = f"${sum(costs):.2f}" if costs else "—"
        if costs and len(costs) < len(STAGES):
            cost += " (partial)"
        pair_state = states.get(key, "not started")
        done = pair_state == "complete" and all(is_complete(stage) for stage in stages.values())
        completed += done
        outcome = PAIR_LABELS.get(pair_state, "Not started")
        if pair_state == "complete" and not done:
            outcome = "Missing a completed stage record"
        rows.append(
            f"| {text(title)} | {outcome} | {stage_label(stages['initial'])} | "
            f"{stage_label(stages['reviewed'])} | {cost} |"
        )
        if pair_state != "not started" or work_results:
            details.extend(score_details(key, title, stages))

    estimates = [stage["estimated_cost"] for stage in saved if stage["estimated_cost"] is not None]
    cost_summary = (
        f"**Recorded API estimate ${sum(estimates):.2f}**"
        if estimates
        else "**No saved API cost estimate yet**"
    )
    input_tokens = sum(stage.get("usage", {}).get("input_tokens", 0) for stage in saved)
    cached = sum(
        stage.get("usage", {}).get("input_tokens_details", {}).get("cached_tokens", 0)
        for stage in saved
    )
    output_tokens = sum(stage.get("usage", {}).get("output_tokens", 0) for stage in saved)
    duration = sum(stage["elapsed_seconds"] for stage in saved)
    return "\n".join(
        [
            "# API transcription batch",
            "",
            f"**{completed}/{len(rows)} scores finished both passes.** "
            f"Model {text(plan['model'])} at {text(plan['reasoning_effort'])}.",
            f"{cost_summary} against a ${text(ceiling)} batch ceiling. "
            "Costs are estimates from saved stages and may be partial during or after "
            "an interruption.",
            f"Saved model stages took {clock(duration)} in total. They recorded {input_tokens:,} "
            f"input tokens ({cached:,} cached) and {output_tokens:,} output tokens.",
            "",
            "| Score | Result | Author | Independent reader | API estimate |",
            "| --- | --- | --- | --- | ---: |",
            *rows,
            "",
            "Completed drafts still need source, playback and musician review before approval. "
            "This report reads the API outputs before any separate delivery edits.",
            *details,
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path, help="Directory with plan.json and batch_status.json")
    parser.add_argument(
        "--output", type=Path, help="New Markdown file (default: print to terminal)"
    )
    args = parser.parse_args()
    try:
        report = build_report(args.batch.expanduser())
        if args.output:
            with args.output.expanduser().open("x") as stream:
                stream.write(report)
            print(f"Saved report to {args.output}")
        else:
            print(report, end="")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
