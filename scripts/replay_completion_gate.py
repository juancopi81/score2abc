#!/usr/bin/env python3
"""Create an offline gate report from sealed saved stages; never run a model."""

from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score2abc.score_eval_abc import normalize_abc  # noqa: E402
from scripts.experiments import completion_gate as gate  # noqa: E402
from scripts.experiments import run_astra_baseline_reproduction as shared  # noqa: E402

DEFAULT_STAGES = (
    ("astra_workflow_20261028_v1", "aviador_s8", "initial", "Subscription"),
    ("astra_workflow_20261028_v1", "aviador_s8", "reviewed", "Subscription"),
    ("astra_workflow_20261028_v1", "rumichaca_full", "initial", "Subscription"),
    ("astra_workflow_20261028_v1", "rumichaca_full", "reviewed", "Subscription"),
    ("astra_api_fixed_output_20261029_v2", "aviador_s8", "initial", "API v2"),
    ("astra_api_fixed_output_20261029_v2", "aviador_s8", "reviewed", "API v2"),
    ("astra_api_review_context_20261029_v3", "aviador_s8", "reviewed_api", "API review v2"),
    ("astra_api_rumichaca_20261029_v1", "rumichaca_full", "initial", "API"),
    ("astra_api_rumichaca_20261029_v1", "rumichaca_full", "reviewed", "API"),
)


def replay_stage(path: Path) -> dict:
    shared.verify_seal(path, "astra_reproduction_stage")
    checkpoints = sorted(path.glob("artifacts/drafts/draft_*/checkpoint.json"))
    if not checkpoints:
        raise ValueError("No saved drafts in supplied stage")
    rows = []
    for checkpoint in checkpoints:
        directory = checkpoint.parent
        result = gate.evaluate_checkpoint(directory)
        parsed = normalize_abc((directory / "score.abc").read_text())
        saved = json.loads((directory / "normalized.json").read_text())
        decisions = json.loads((directory / "decisions.json").read_text())
        # Recheck the current deterministic parser without changing old artifacts.
        if gate.issues_for(parsed, decisions) != gate.issues_for(saved, decisions):
            raise ValueError(f"Current parser changes completion issues: {checkpoint}")
        rows.append(
            {
                "checkpoint": directory.name,
                "current_parser_issues_identical": True,
                "written_measures": len(saved.get("measures", [])),
                **result,
            }
        )
    # Detect any accidental source writes while checking every saved draft.
    shared.verify_seal(path, "astra_reproduction_stage")
    provenance = json.loads((path / "stage_provenance.json").read_text())
    return {
        "path": str(path.resolve()),
        "original_stage_status": provenance.get("status"),
        "model": provenance.get("model"),
        "input_seal_sha256": gate.sha((path / "seal.json").read_bytes()),
        "checkpoints": rows,
        "final": rows[-1],
        "sealed_inputs_unchanged": True,
    }


def render(report: dict) -> str:
    summary = report["summary"]
    rows = []
    for stage in report["stages"]:
        final = stage["final"]
        problems = final["unresolved_issues"]
        bars = [str(i["measure"]) for i in problems if i["kind"] == "meter"]
        details = ", ".join(bars) if len(bars) < 10 else f"{len(bars)} bars"
        cells = [
            stage["piece"],
            stage["backend"],
            stage["stage"],
            final["written_measures"],
            stage["original_stage_status"],
            "Ready" if final["completion_ready"] else "Blocked",
            len(problems),
            details or "None",
        ]
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in cells) + "</tr>")
    return (
        (
            """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Completion gate: offline replay</title><style>
body{font:17px/1.55 system-ui,sans-serif;background:#f6f7fa;color:#142333;max-width:1150px;
margin:auto;padding:36px}h1{line-height:1.15}p{max-width:900px}strong{color:#113e68}
table{border-collapse:collapse;width:100%;background:white;font-size:15px}
td,th{text-align:left;padding:12px;border-bottom:1px solid #d8dee8}th{background:#e4edf8}
.cards{display:flex;gap:16px;flex-wrap:wrap}.card{background:white;padding:16px 24px;
border-top:4px solid #23699e}details{margin-top:24px}a{color:#145286}
</style><h1>The gate catches unresolved rhythm warnings</h1>
<p>Offline replay of saved Astra drafts. This tests a completion rule, not a new model run.
The original subscription workflow and all sealed results are unchanged.</p>
<div class="cards"><div class="card"><strong>DRAFT_COUNT</strong> saved drafts checked</div>
<div class="card"><strong>READY_COUNT / STAGE_COUNT</strong> final drafts ready</div>
<div class="card"><strong>BLOCKED_COUNT / STAGE_COUNT</strong> finals blocked</div>
<div class="card"><strong>0</strong> live model calls</div></div>
<h2>What changes under completion_gate_v1?</h2><table><thead><tr>
<th>Piece</th><th>Backend</th><th>Stage</th><th>Written bars</th><th>Old status</th>
<th>New gate</th><th>Unresolved</th><th>Bars to resolve</th></tr></thead><tbody>"""
            + "".join(rows)
            + """
</tbody></table><p><strong>Rumichaca API review:</strong> bars 2, 16 and 44 need resolution.
Every draft remains saved. A future model receives the warning and must correct it or provide
a structured source exception tied to the current ABC and an inspected PDF crop.</p>
<p><strong>Ready does not mean musically correct.</strong> A wrong pitch can still fit the meter;
a playable form can still follow the wrong route. This gate does not prove all PDF bars were
transcribed, verify an exception's musical justification, or grant human approval. The older
Aviador API drafts with an accidental error still pass these structural checks.</p>
<p>No exceptions or corrections were invented during replay.
All DRAFT_COUNT drafts were reparsed locally;
their completion issues match the saved parser outputs. No historical metrics were recomputed.</p>
<p><strong>Next:</strong> one bounded API trial of Rumichaca with this opt-in gate, followed by
the unchanged music evaluation. This report does not start that experiment.</p>
<details><summary>Reproducibility</summary><p>Policy, implementation hashes, all checkpoint
decisions and input seal hashes are in <a href="report.json">report.json</a>.
Original stage seals were verified before and after replay.</p></details></html>"""
        )
        .replace("DRAFT_COUNT", str(summary["drafts"]))
        .replace("READY_COUNT", str(summary["finals_ready"]))
        .replace("BLOCKED_COUNT", str(summary["stages"] - summary["finals_ready"]))
        .replace("STAGE_COUNT", str(summary["stages"]))
    )


def build(output: Path, reproduction: Path) -> dict:
    if output.exists():
        raise ValueError("Replay output is create-once; choose a new version")
    stages = []
    for run, piece, stage, backend in DEFAULT_STAGES:
        data = replay_stage(reproduction / run / piece / stage)
        stages.append({"run": run, "piece": piece, "stage": stage, "backend": backend, **data})
    report = {
        "schema_version": 1,
        "policy_version": gate.POLICY_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "consumed_offline_completion_replay_not_new_model_performance",
        "live_model_calls": 0,
        "musical_metrics_recomputed": False,
        "implementation_files": {
            str(p.relative_to(ROOT)): gate.sha(p.read_bytes())
            for p in (
                Path(__file__).resolve(),
                Path(gate.__file__),
                ROOT / "scripts/experiments/astra_source_tools.py",
                ROOT / "scripts/experiments/astra_source_tools_gated.py",
                ROOT / "score2abc/score_eval_parse.js",
                ROOT / "score2abc/review_playback.js",
                ROOT / "scripts/experiments/run_astra_baseline_reproduction.py",
                ROOT / "scripts/experiments/assisted_api.py",
                ROOT / "scripts/run_assisted_transcription.py",
            )
        },
        "stages": stages,
        "summary": {
            "stages": len(stages),
            "drafts": sum(len(s["checkpoints"]) for s in stages),
            "finals_ready": sum(s["final"]["completion_ready"] for s in stages),
        },
    }
    output.mkdir(parents=True)
    for name in report["implementation_files"]:
        snapshot = output / "implementation" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((ROOT / name).read_bytes())
    shared.write(output / "report.json", report)
    (output / "index.html").write_text(render(report))
    shared.seal(output, "completion_gate_offline_replay")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reproduction", type=Path, default=ROOT / "out/eval/reproduction")
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.reproduction)["summary"], indent=2))
