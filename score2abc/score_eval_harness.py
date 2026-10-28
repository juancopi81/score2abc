"""Prepare PDF-only tasks and exercise the offline grader without model calls."""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from score2abc.reference_freeze import verify_freeze
from score2abc.score_eval import MODELS, compare_normalized, sha256, validate_candidate
from score2abc.score_eval_abc import normalize_abc

AUTHOR_INSTRUCTIONS = """Transcribe the supplied handwritten Colombian music PDF into one complete
playable ABC score. Use only this PDF and musical reasoning; do not use recordings,
web search, other editions, reviewed answers, or prior work-specific discussions.
Inspect every musical row and all voices. Preserve readable notes, rhythms, rests,
chord symbols, ties, meter/key changes, repeats and alternate endings (casillas).
Unmarked chord bars do not automatically need invented harmony.

Deliver a complete best-effort performing edition. For damaged material choose
the smallest musically supported completion, retaining readable evidence. Record
uncertainty and alternatives without leaving placeholders for a human to compose.
Common forms such as AA BB CC ABC are possibilities, not requirements. Readable
directions take priority; use phrases, cadences and destination-specific lead-ins
to choose a coherent route. State the route and final stopping point. Encode it
with ordinary repeats/endings or bounded explicit single-letter ABC P: order.
An explicit P: order must contain every intended repeat; printed signs then serve
engraving only in this review player.

Use deterministic validation feedback to inspect and repair inconsistencies.
A short bar may be a pickup, damaged rhythm, missing note/rest or incorrect bar
boundary. Do not automatically pad rests or remove notes just to satisfy meter.
Make at most two revision passes after the initial complete draft.

Return score.abc and decisions.json. The decision log is a JSON array with an
entry for each reconstruction or uncertain interpretation: id, location,
category, adopted_choice, reason, uncertainty, and alternatives when meaningful.
Use an empty array when no such decision is needed. Preserve the source reading
in the reason when the adopted edition departs from it. Also retain a raw run
record with the actual model, reasoning setting, time and provider usage when
available. Never invent cost, reviewer approval or successful validation.
"""


def prepare_inputs(frozen_root: Path, destination: Path) -> dict:
    """Create a source-only task packet. Execution isolation is a separate requirement."""
    frozen_root, destination = Path(frozen_root).resolve(), Path(destination).resolve()
    if destination.is_relative_to(frozen_root) or frozen_root.is_relative_to(destination):
        raise ValueError("Task packet must be separate from frozen references")
    frozen = verify_freeze(frozen_root)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        prompt = AUTHOR_INSTRUCTIONS.encode()
        (destination / "instructions.txt").write_bytes(prompt)
        tasks = []
        for work in frozen["works"]:
            relative = f"inputs/{work['slug']}/source.pdf"
            source = frozen_root / work["source_pdf_path"]
            target = destination / relative
            target.parent.mkdir(parents=True)
            data = source.read_bytes()
            if sha256(data) != frozen["files"][work["source_pdf_path"]]["sha256"]:
                raise ValueError("PDF changed while preparing input packet")
            target.write_bytes(data)
            for model in MODELS:
                tasks.append(
                    {
                        "slug": work["slug"],
                        "model_id": model["id"],
                        "model": model["model"],
                        "reasoning_effort": model["reasoning_effort"],
                        "input_pdf_path": relative,
                        "input_pdf_sha256": sha256(data),
                        "instructions_sha256": sha256(prompt),
                        "max_revision_passes": 2,
                        "status": "not_run",
                    }
                )
        contract = {
            "schema_version": 1,
            "source_policy": "pdf_only",
            "isolation_status": "requires_enforced_runner",
            "execution_status": "not_run",
            "tasks": tasks,
        }
        payload = (json.dumps(contract, indent=2) + "\n").encode()
        (destination / "tasks.json").write_bytes(payload)
        (destination / "tasks.sha256").write_text(sha256(payload) + "\n")
        return contract
    except BaseException:
        shutil.rmtree(destination)
        raise


def bounded_revision_loop(draft: str, decisions: list, revise, max_revisions: int = 2) -> dict:
    """Run truth-free validation/repair through a caller-supplied author adapter.

    No provider call is built in. The callback receives only draft, decisions and
    validator feedback; model identity/isolation/usage belong to its run record.
    A bounded trace preserves every attempt rather than selecting by hidden truth.
    """
    if type(max_revisions) is not int or not 0 <= max_revisions <= 2:
        raise ValueError("Use zero to two revision passes")
    attempts = []
    for attempt in range(max_revisions + 1):
        feedback = validate_candidate(draft, decisions)
        attempts.append(
            {
                "attempt": attempt,
                "abc": draft,
                "abc_sha256": sha256(draft.encode()),
                "decisions": deepcopy(decisions),
                "feedback": deepcopy(feedback),
            }
        )
        if not feedback["issues"] or attempt == max_revisions:
            break
        try:
            next_draft, next_decisions = revise(draft, deepcopy(decisions), deepcopy(feedback))
        except Exception as exc:
            return {
                "status": "partial",
                "attempts": attempts,
                "error": f"Revision adapter failed: {type(exc).__name__}",
            }
        if not isinstance(next_draft, str) or not isinstance(next_decisions, list):
            return {"status": "partial", "attempts": attempts, "error": "Invalid revision response"}
        draft, decisions = next_draft, next_decisions
    return {
        "status": "validated" if not attempts[-1]["feedback"]["issues"] else "partial",
        "attempts": attempts,
        "error": None,
        "musical_accuracy": "not_measured_by_validation",
    }


def build_grader_demo(destination: Path) -> dict:
    """Six synthetic controls show grader behavior; neither model is invoked."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    base = (
        "X:1\nT:Synthetic control\nM:3/4\nL:1/4\nK:C\n"
        '"C"C D E | "F"F G A | "G"B A G | "C"E D C |]\n'
    )
    form = (
        "X:1\nT:Synthetic route\nP:AB\nM:3/4\nL:1/4\nK:C\n"
        '[P:A]"C"C D E | "F"F G A | [P:B]"G"B A G | "C"E D C |]\n'
    )
    examples = [
        (
            "equivalent",
            "Equivalent notation",
            base,
            base.replace("L:1/4", "L:1/8")
            .replace('"C"C D E', '"C"C2 D2 E2')
            .replace('"F"F G A', '"F"F2 G2 A2')
            .replace('"G"B A G', '"G"B2 A2 G2')
            .replace('"C"E D C', '"C"E2 D2 C2'),
        ),
        ("pitch", "One wrong pitch", base, base.replace('"C"C D E', '"C"C D F')),
        ("duration", "One wrong duration", base, base.replace('"F"F G A', '"F"F G A2')),
        ("chord", "One wrong chord", base, base.replace('"F"F G A', '"Dm"F G A')),
        ("missing", "One missing bar", base, base.replace('"F"F G A | ', "")),
        ("route", "Wrong return", form, form.replace("P:AB\n", "P:ABA\n")),
    ]
    cases = []
    for name, title, reference_text, candidate_text in examples:
        ref, candidate = normalize_abc(reference_text), normalize_abc(candidate_text)
        if not ref["valid"] or not candidate["valid"]:
            raise ValueError("Synthetic grader test requires the optional local ABC renderer")
        ref_path, candidate_path = (
            destination / f"{name}.control.abc",
            destination / f"{name}.mutated.abc",
        )
        ref_path.write_text(reference_text)
        candidate_path.write_text(candidate_text)
        results = {}
        for identifier, normalized, path in [
            ("control", ref, ref_path),
            ("mutated", candidate, candidate_path),
        ]:
            result = compare_normalized(ref, normalized)
            feedback = validate_candidate(path.read_text(), [])
            incomplete = result["coverage"]["paired_measures"] != len(ref["measures"])
            result.update(
                {
                    "status": "partial" if feedback["issues"] or incomplete else "complete",
                    "validation": feedback,
                    "decisions": [],
                    "error": None,
                    "links": {"candidate_abc": str(path)},
                }
            )
            results[identifier] = result
        metrics = results["mutated"]["metrics"]
        if name == "equivalent":
            assert metrics["edition_note_f1"] == metrics["edition_exact_bar_rate"] == 1
        elif name in ("pitch", "duration", "missing"):
            assert metrics["edition_note_f1"] < 1
        elif name == "chord":
            assert metrics["chord_f1"] < 1
        elif name == "route":
            assert metrics["form_exact_match"] is False
        cases.append({"slug": name, "title": title, "results": results})
    return {
        "schema_version": 1,
        "evidence_kind": "synthetic_demo",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frozen_manifest_sha256": None,
        "labels_sha256": None,
        "models": [
            {"id": "control", "label": "Unchanged fixture"},
            {"id": "mutated", "label": "Injected errors"},
        ],
        "cases": cases,
        "summary": {"recommendation": "Grader controls only; no models ran."},
        "limitations": ["Artificial notation changes test the grader, not model quality."],
    }
