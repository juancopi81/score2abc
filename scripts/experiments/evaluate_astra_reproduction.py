"""Grade sealed assisted Astra drafts without reusing the tool-free run contract.

The Aviador lane uses the unchanged historical event scorer. Rumichaca uses the
existing complete-score grader, with initial and independently reviewed drafts
kept separate. Neither command performs inference or edits historical artifacts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score2abc.eval_labels import verify_labels  # noqa: E402
from score2abc.reference_freeze import verify_freeze  # noqa: E402
from score2abc.score_eval import (  # noqa: E402
    _source_required_edition_bars,
    compare_normalized,
    validate_candidate,
)
from score2abc.score_eval_abc import normalize_abc  # noqa: E402
from score2abc.score_eval_config import renderer_identity  # noqa: E402
from score2abc.score_eval_source import score_source  # noqa: E402
from scripts import build_vlm_melody_event_benchmark as historical  # noqa: E402
from scripts.experiments import evaluate_astra_music_benchmark as old_adapter  # noqa: E402

STAGES = ("initial", "reviewed")
WORKFLOW = "adaptive_source_inspection_author_then_independent_reader_v1"
SEAL_KIND = "astra_assisted_workflow_submission_seal"
SEAL_STATUS = "both_stages_frozen_before_reference_scoring"
LEGACY_PLAN_KIND = "astra_assisted_workflow_reproduction_plan"
COMPARISON_PLAN_KIND = "matched_assisted_workflow_comparison_plan"
TRANSPORT_PLAN_KIND = "single_target_assisted_api_transport_plan"
TRANSPORT_TARGET = "aviador_s8"
TRANSPORT_PROMPTS = ("aviador_s8.txt", "aviador_s8_review.txt")
TRANSPORT_TARGETS = {
    "aviador_s8": {
        "target_selection": "aviador_seven_bars_only",
        "deadline_seconds": {"initial": 1800, "reviewed": 1200},
    },
    "rumichaca_full": {
        "target_selection": "rumichaca_full_score_only",
        "deadline_seconds": {"initial": 3600, "reviewed": 1800},
    },
}
TRANSPORT_TOOL_FILES = (
    "astra_source_tools.py",
    "astra_source_tools_gated.py",
    "completion_gate.py",
)
COMPARISON_POLICY = {
    "target_selection": "both_preregistered_targets_regardless_of_aviador_outcome",
    "rumichaca_prerequisite": "baseline_predecessor_frozen_before_comparison",
}
COMPARISON_MODELS = {"gpt-6-astra", "gpt-6.1-terra", "gpt-6.2-astra", "gpt-6.2-sol"}


def _completed_report(report: dict) -> bool:
    """Execution completion is independent of musical validity or accuracy."""
    return (
        report.get("schema_version") == 1
        and report.get("both_stages_verified_before_reference_content") is True
        and set(report.get("stages", {})) == set(STAGES)
        and all(
            lane.get("runtime", {}).get("status") == "complete"
            and lane["runtime"].get("returncode") == 0
            and lane["runtime"].get("complete") is True
            and lane.get("completion_claim", True) is True
            for lane in report["stages"].values()
        )
    )


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha(path)}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def verify_pin(root: Path, record: dict, label: str) -> Path:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise ValueError(f"Missing {label} pin")
    raw = Path(record["path"])
    path = raw if raw.is_absolute() else root / raw
    if path.is_symlink():
        raise ValueError(f"Symlink is not permitted for {label}")
    path = path.resolve()
    if not path.is_file() or sha(path) != record.get("sha256"):
        raise ValueError(f"{label} hash mismatch")
    return path


def _verify_comparison_predecessor(plan: dict, root: Path, started_at: str) -> list[dict]:
    """Pin the original Astra baseline without opening musical reference content."""
    predecessor = plan.get("baseline_predecessor", {})
    baseline_path = verify_pin(root, predecessor.get("plan"), "Astra baseline plan")
    baseline = read_json(baseline_path)
    if (
        baseline.get("kind") != LEGACY_PLAN_KIND
        or baseline.get("status") != "preregistered_before_new_author_inference"
        or baseline.get("model") != "gpt-6-astra"
        or baseline.get("reasoning_effort") != plan["reasoning_effort"]
        or baseline.get("workflow_id") != WORKFLOW
        or baseline.get("source_policy") != "supplied_pdf_only"
        or baseline.get("stages") != list(STAGES)
        or baseline.get("evidence_kind") != plan.get("evidence_kind")
        or baseline.get("stage_protocol", {}).get("deadline_seconds")
        != plan["stage_protocol"]["deadline_seconds"]
    ):
        raise ValueError("Comparison must preserve the original Astra workflow and allowances")
    contracts = {
        "aviador_s8": (
            "slug",
            "system_index",
            "source_pdf",
            "grading_only_expected_measure_indices",
            "historical_reference_pins",
            "historical_protocol",
            "advance_gate",
            "unsupported_scope",
        ),
        "rumichaca_full": ("slug", "source_pdf", "frozen_reference", "labels"),
    }
    if set(plan.get("targets", {})) != set(contracts):
        raise ValueError("Comparison must preregister both targets")
    for target, keys in contracts.items():
        if any(
            key not in baseline.get("targets", {}).get(target, {})
            or plan["targets"][target].get(key) != baseline["targets"][target][key]
            for key in keys
        ):
            raise ValueError("Comparison source/scoring contracts differ from the Astra baseline")
    reports = predecessor.get("reports", {})
    if set(reports) != set(contracts):
        raise ValueError("Comparison requires both original Astra baseline reports")
    records = [pin(baseline_path)]
    for target, record in reports.items():
        path = verify_pin(root, record, f"Astra {target} baseline report")
        report = read_json(path)
        if (
            report.get("kind") != "astra_assisted_workflow_reproduction_report"
            or report.get("target_id") != target
            or report.get("workflow_id") != WORKFLOW
            or report.get("model") != "gpt-6-astra"
            or report.get("reasoning_effort") != plan["reasoning_effort"]
            or report.get("evidence_kind") != plan.get("evidence_kind")
            or not _completed_report(report)
            or not any(p.get("sha256") == sha(baseline_path) for p in report.get("pins", []))
            or datetime.fromisoformat(report["created_at"]) > datetime.fromisoformat(started_at)
        ):
            raise ValueError("Original completed Astra reports must precede comparison inference")
        implementations = report.get("implementations", {})
        for name in (
            "historical_scorer",
            "historical_adapter",
            "abc_normalizer",
            "abc_parser",
            "complete_score_grader",
            "source_grader",
        ):
            implementation = verify_pin(ROOT, implementations.get(name), f"baseline {name}")
            records.append(pin(implementation))
        if implementations.get("renderer") != renderer_identity():
            raise ValueError("Comparison renderer differs from the original Astra baseline")
        records.append(pin(path))
    return records


def _transport_target(plan: dict) -> str:
    targets = plan.get("targets", {})
    if len(targets) != 1 or next(iter(targets)) not in TRANSPORT_TARGETS:
        raise ValueError("Transport plan requires exactly one supported target")
    return next(iter(targets))


def _transport_prompts(target: str) -> tuple[str, str]:
    return f"{target}.txt", f"{target}_review.txt"


def build_api_transport_plan(
    baseline_plan_path: Path,
    baseline_report_path: Path,
    *,
    experiment_id: str,
    prompts: dict[str, Path],
    target_id: str = TRANSPORT_TARGET,
) -> dict:
    """Build a prospective single-score API comparison without reading answer content."""
    if target_id not in TRANSPORT_TARGETS:
        raise ValueError("Unsupported API transport target")
    baseline = read_json(baseline_plan_path)
    case = Path(baseline_report_path).resolve().parent.parent
    plan = {
        key: copy.deepcopy(baseline[key])
        for key in (
            "schema_version",
            "status",
            "workflow_id",
            "model",
            "reasoning_effort",
            "source_policy",
            "stages",
            "evidence_kind",
            "completion_policy",
            "completion_policy_sha256",
            "review_instruction_version",
            "submission_contract",
        )
    }
    plan.update(
        kind=TRANSPORT_PLAN_KIND,
        experiment_id=experiment_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        auth_mode="openai_api",
        targets={target_id: copy.deepcopy(baseline["targets"][target_id])},
        stage_protocol={
            **copy.deepcopy(baseline["stage_protocol"]),
            "deadline_seconds": {
                target_id: copy.deepcopy(baseline["stage_protocol"]["deadline_seconds"][target_id])
            },
        },
        current_harness={
            "review_version": "assisted_review_v2",
            "completion_policy": "completion_gate_v1",
            "prompts": {name: pin(prompts[name]) for name in _transport_prompts(target_id)},
        },
        transport_predecessor={
            "plan": pin(baseline_plan_path),
            "report": pin(baseline_report_path),
            "packets": {stage: pin(case / f"{stage}_packet" / "seal.json") for stage in STAGES},
        },
        transport_scope={
            "target_selection": TRANSPORT_TARGETS[target_id]["target_selection"],
            "rumichaca_scheduled": target_id == "rumichaca_full",
            "interpretation": (
                "One fresh API pair versus a saved subscription pair; "
                "not a causal transport effect."
            ),
        },
    )
    if target_id == "rumichaca_full":
        plan["supplemental_source_labels"] = copy.deepcopy(baseline["supplemental_source_labels"])
    _verify_transport_predecessor(plan, ROOT, plan["created_at"])
    return plan


def _verify_transport_predecessor(plan: dict, root: Path, started_at: str) -> list[dict]:
    """Check the saved subscription controls, not the historical two-target policy."""
    from scripts.experiments import run_astra_baseline_reproduction as runner

    predecessor = plan.get("transport_predecessor", {})
    baseline_path = verify_pin(root, predecessor.get("plan"), "transport baseline plan")
    report_path = verify_pin(root, predecessor.get("report"), "transport baseline report")
    baseline, report = read_json(baseline_path), read_json(report_path)
    target = _transport_target(plan)
    if (
        baseline.get("kind") != COMPARISON_PLAN_KIND
        or baseline.get("model") != plan.get("model")
        or plan.get("model") != "gpt-6.2-astra"
        or baseline.get("reasoning_effort") != plan.get("reasoning_effort")
        or baseline.get("workflow_id") != plan.get("workflow_id")
        or baseline.get("source_policy") != plan.get("source_policy")
        or baseline.get("evidence_kind") != plan.get("evidence_kind")
        or plan["targets"][target] != baseline.get("targets", {}).get(target)
        or plan.get("stage_protocol", {}).get("deadline_seconds")
        != {target: baseline.get("stage_protocol", {}).get("deadline_seconds", {}).get(target)}
        or plan.get("stage_protocol", {}).get("deadline_seconds", {}).get(target)
        != TRANSPORT_TARGETS[target]["deadline_seconds"]
        or plan.get("auth_mode") != "openai_api"
        or plan.get("completion_policy") != "completion_gate_v1"
        or plan.get("review_instruction_version") != "assisted_review_v2"
        or any(
            plan.get(key) != baseline.get(key)
            for key in (
                "completion_policy",
                "completion_policy_sha256",
                "review_instruction_version",
            )
        )
        or plan.get("transport_scope", {}).get("target_selection")
        != TRANSPORT_TARGETS[target]["target_selection"]
        or plan.get("transport_scope", {}).get("rumichaca_scheduled")
        is not (target == "rumichaca_full")
        or (
            target == "rumichaca_full"
            and (
                not baseline.get("supplemental_source_labels")
                or plan.get("supplemental_source_labels") != baseline["supplemental_source_labels"]
            )
        )
    ):
        raise ValueError("Transport source, identity or harness differs from saved subscription")
    if (
        report.get("kind") != "astra_assisted_workflow_reproduction_report"
        or report.get("target_id") != target
        or any(
            report.get(key) != plan[key]
            for key in ("model", "reasoning_effort", "workflow_id", "evidence_kind")
        )
        or not _completed_report(report)
        or not any(p.get("sha256") == sha(baseline_path) for p in report.get("pins", []))
    ):
        raise ValueError("Transport baseline must be a completed matching subscription report")
    try:
        timestamps = [
            datetime.fromisoformat(value)
            for value in (report["created_at"], plan["created_at"], started_at)
        ]
        if any(value.tzinfo is None for value in timestamps) or not (
            timestamps[0] <= timestamps[1] <= timestamps[2]
        ):
            raise ValueError("invalid ordering")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Transport preregistration must follow baseline and precede inference"
        ) from exc
    records = [pin(baseline_path), pin(report_path)]
    if target == "rumichaca_full":
        records.append(
            pin(verify_pin(root, plan["supplemental_source_labels"], "supplemental labels"))
        )
    harness = plan.get("current_harness", {})
    if (
        harness.get("review_version") != "assisted_review_v2"
        or harness.get("completion_policy") != "completion_gate_v1"
        or set(harness.get("prompts", {})) != set(_transport_prompts(target))
        or set(predecessor.get("packets", {})) != set(STAGES)
    ):
        raise ValueError("Transport requires both pinned prompt and input controls")
    for stage, name in zip(STAGES, _transport_prompts(target), strict=True):
        prompt = verify_pin(root, harness["prompts"][name], "transport prompt")
        original_prompt = baseline.get("current_harness", {}).get("prompts", {}).get(name)
        original = verify_pin(baseline_path.parent, original_prompt, "subscription prompt")
        packet_seal = verify_pin(root, predecessor["packets"][stage], "subscription packet")
        runner.verify_seal(packet_seal.parent, "astra_reproduction_input")
        metadata = read_json(packet_seal.parent / "preparation.json")
        runtime = report["stages"][stage]["runtime"]
        rendered_prompt = packet_seal.parent / "bundle/prompt.txt"
        prefix = original.read_bytes()
        if stage == "reviewed":
            prefix += b"\n\n<untrusted_candidate_draft>\n"
        with rendered_prompt.open("rb") as handle:
            prompt_prefix_matches = handle.read(len(prefix)) == prefix
        if (
            sha(prompt) != sha(original)
            or not prompt_prefix_matches
            or metadata.get("prompt_sha256") != sha(rendered_prompt)
            or (stage == "initial" and sha(rendered_prompt) != sha(original))
            or metadata.get("model") != plan["model"]
            or metadata.get("stage") != stage
            or metadata.get("source_pdf_sha256") != plan["targets"][target]["source_pdf"]["sha256"]
            or runtime.get("auth_mode") != "chatgpt_subscription"
            or runtime.get("input_seal_sha256") != sha(packet_seal)
        ):
            raise ValueError("Saved subscription prompt, packet or authentication differs")
        for filename in TRANSPORT_TOOL_FILES:
            path = packet_seal.parent / "bundle" / filename
            if sha(path) != sha(Path(__file__).with_name(filename)):
                raise ValueError("Shared source tools changed from subscription baseline")
            records.append(pin(path))
        records.extend([pin(prompt), pin(original), pin(packet_seal)])
    for name in (
        "historical_scorer",
        "historical_adapter",
        "abc_normalizer",
        "abc_parser",
        "complete_score_grader",
        "source_grader",
    ):
        records.append(pin(verify_pin(ROOT, report["implementations"].get(name), name)))
    if report["implementations"].get("renderer") != renderer_identity():
        raise ValueError("Transport renderer differs from subscription baseline")
    return records


def verify_submission(plan_path: Path, seal_path: Path) -> tuple[dict, dict, dict, list[dict]]:
    """Verify both drafts and all source/workflow artifacts before any reference read."""
    plan, seal = read_json(plan_path), read_json(seal_path)
    comparison = plan.get("kind") == COMPARISON_PLAN_KIND
    transport = plan.get("kind") == TRANSPORT_PLAN_KIND
    if (
        plan.get("kind") not in {LEGACY_PLAN_KIND, COMPARISON_PLAN_KIND, TRANSPORT_PLAN_KIND}
        or plan.get("status") != "preregistered_before_new_author_inference"
        or plan.get("workflow_id") != WORKFLOW
        or plan.get("model")
        not in (
            COMPARISON_MODELS if comparison else {"gpt-6.2-astra"} if transport else {"gpt-6-astra"}
        )
        or plan.get("reasoning_effort") != "xhigh"
        or plan.get("source_policy") != "supplied_pdf_only"
        or plan.get("stages") != list(STAGES)
        or (comparison and plan.get("comparison_policy") != COMPARISON_POLICY)
        or (not comparison and "comparison_policy" in plan)
    ):
        raise ValueError("Unsupported preregistered workflow plan")
    if (
        seal.get("schema_version") != 1
        or seal.get("kind") != SEAL_KIND
        or seal.get("status") != SEAL_STATUS
        or seal.get("plan_sha256") != sha(plan_path)
        or seal.get("reference_feedback_given") is not False
        or set(seal.get("stages", {})) != set(STAGES)
        or seal.get("target_id") not in {"aviador_s8", "rumichaca_full"}
    ):
        raise ValueError("Both workflow stages must be sealed without reference feedback")
    if transport and seal["target_id"] != _transport_target(plan):
        raise ValueError("Sealed target differs from the single-target transport plan")
    target = plan["targets"][seal["target_id"]]
    if seal.get("evidence_kind", plan.get("evidence_kind")) not in {
        "consumed_development_workflow_reproduction",
        "synthetic_control",
    }:
        raise ValueError("Unrecognized reproduction evidence kind")
    execution_path = verify_pin(
        seal_path.parent, seal.get("execution_contract"), "execution contract"
    )
    execution = read_json(execution_path)
    if (
        execution.get("kind") != "astra_assisted_workflow_execution_contract"
        or execution.get("status") != "preregistered_before_author_inference"
        or execution.get("plan_sha256") != seal["plan_sha256"]
        or execution.get("gate_eligibility")
        != {
            "both_stage_status": "complete",
            "both_stage_returncode": 0,
            "checkpoint_completion_claim_alone_is_insufficient": True,
        }
        or execution.get("deadline_seconds") != plan["stage_protocol"]["deadline_seconds"]
    ):
        raise ValueError("Execution contract differs from preregistered stage validity")
    if seal.get("source_pdf_sha256") != target["source_pdf"]["sha256"]:
        raise ValueError("Sealed input differs from preregistered source PDF")
    source = verify_pin(plan_path.parent, target["source_pdf"], "source PDF")
    records = [pin(plan_path), pin(seal_path), pin(source), pin(execution_path)]
    root, paths = seal_path.parent, {}
    artifacts = seal.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Seal must pin workflow artifacts")
    archived = {}
    for item in artifacts:
        path = verify_pin(root, item, "workflow artifact")
        if path in archived:
            raise ValueError("Duplicate sealed workflow artifact path")
        archived[path] = item["sha256"]
        records.append(pin(path))
    all_stage_paths = set()
    for stage in STAGES:
        entry = seal["stages"][stage]
        if not {"abc", "decisions", "provenance"}.issubset(entry) or set(entry) - {
            "abc",
            "decisions",
            "provenance",
            "event_ledger",
        }:
            raise ValueError("Each stage requires ABC, decisions and provenance pins")
        paths[stage] = {}
        for kind in entry:
            path = verify_pin(root, entry[kind], f"{stage} {kind}")
            if path in all_stage_paths:
                raise ValueError("Initial/reviewed artifacts must use distinct paths")
            all_stage_paths.add(path)
            if archived.get(path) != entry[kind]["sha256"]:
                raise ValueError("Stage artifact is absent from the archive seal")
            paths[stage][kind] = path
        provenance = read_json(paths[stage]["provenance"])
        expected = {
            "stage": stage,
            "model": plan["model"],
            "reasoning_effort": plan["reasoning_effort"],
            "workflow_id": WORKFLOW,
            "source_pdf_sha256": seal["source_pdf_sha256"],
            "reference_access": False,
        }
        if any(provenance.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{stage} provenance differs from the workflow plan")
        if plan.get("completion_policy") is not None and any(
            provenance.get(key) != plan.get(key)
            for key in ("completion_policy", "completion_policy_sha256")
        ):
            raise ValueError("Stage completion policy differs from the preregistered plan")
        if type(provenance.get("complete")) is not bool:
            raise ValueError("Stage provenance needs an explicit completion claim")
        if provenance.get("status") not in {"complete", "partial", "failed"}:
            raise ValueError("Stage provenance requires an explicit runtime status")
        if provenance.get("returncode") is not None and type(provenance["returncode"]) is not int:
            raise ValueError("Stage returncode must be an integer or null")
        elapsed = provenance.get("elapsed_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed < 0
            or provenance.get("deadline_seconds")
            != execution["deadline_seconds"][seal["target_id"]][stage]
            or any(
                key not in provenance
                for key in ("returncode", "started_at", "finished_at", "error")
            )
        ):
            raise ValueError("Stage runtime measurements/deadline are missing or invalid")
        try:
            started = datetime.fromisoformat(provenance["started_at"])
            finished = datetime.fromisoformat(provenance["finished_at"])
            if started.tzinfo is None or finished.tzinfo is None or finished < started:
                raise ValueError("Invalid stage timestamp ordering")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Stage timestamps must be ordered timezone-aware ISO timestamps"
            ) from exc
        if stage == "reviewed" and (
            provenance.get("independent_reader") is not True
            or provenance.get("initial_abc_sha256") != seal["stages"]["initial"]["abc"]["sha256"]
        ):
            raise ValueError("Reviewed stage is not bound to the sealed initial draft")
        decisions = json.loads(paths[stage]["decisions"].read_text())
        if not isinstance(decisions, list) or any(not isinstance(d, dict) for d in decisions):
            raise ValueError("Decisions must be an array of objects")
        paths[stage]["provenance_data"] = provenance
        paths[stage]["decisions_data"] = decisions
    if comparison:
        records.extend(
            _verify_comparison_predecessor(
                plan, plan_path.parent, paths["initial"]["provenance_data"]["started_at"]
            )
        )
    if transport:
        records.extend(
            _verify_transport_predecessor(
                plan, plan_path.parent, paths["initial"]["provenance_data"]["started_at"]
            )
        )
        _verify_api_pair_artifacts(plan, root, paths, archived)
    if seal["target_id"] == "rumichaca_full" and not comparison and not transport:
        prerequisite_path = verify_pin(
            root, seal.get("prerequisite_evaluation"), "Aviador gate report"
        )
        prerequisite = read_json(prerequisite_path)
        if (
            prerequisite.get("kind") != "astra_assisted_workflow_reproduction_report"
            or prerequisite.get("target_id") != "aviador_s8"
            or prerequisite.get("workflow_id") != WORKFLOW
            or prerequisite.get("model") != plan["model"]
            or prerequisite.get("reasoning_effort") != plan["reasoning_effort"]
            or prerequisite.get("evidence_kind")
            != seal.get("evidence_kind", plan.get("evidence_kind"))
            or prerequisite.get("gate", {}).get("advance_to_rumichaca") is not True
            or not any(p.get("sha256") == seal["plan_sha256"] for p in prerequisite.get("pins", []))
            or datetime.fromisoformat(paths["initial"]["provenance_data"]["started_at"])
            < datetime.fromisoformat(prerequisite["created_at"])
        ):
            raise ValueError("Rumichaca requires a passing Aviador report before author inference")
        records.append(pin(prerequisite_path))
    return plan, seal, paths, records


def abc_to_historical(score: dict, identities: list[dict]) -> tuple[list[dict], dict]:
    """Use written ABC note/rest events; do not use edition alignment or fold ties."""
    expected = len(identities)
    measures = score["measures"]
    valid = bool(score.get("valid")) and len(measures) <= expected
    complete = bool(score.get("valid")) and len(measures) == expected
    validation = {
        "abc_valid": bool(score.get("valid")),
        "errors": score.get("errors", []),
        "submitted_measures": len(measures),
        "expected_measures": expected,
        "complete_measure_coverage": complete,
        "headline_metrics_valid": valid,
        "all_declared_meter_bars_valid": complete
        and all(m.get("meter_valid") is True for m in measures),
    }
    if not valid:
        return [], validation
    ticks = score["ticks_per_quarter"]
    if not isinstance(ticks, (int, float)) or isinstance(ticks, bool) or ticks <= 0:
        raise ValueError("Invalid ABC timing unit")
    rows = []
    for index, measure in enumerate(measures):
        timing = lambda event: {  # noqa: E731
            "onset_beats": event["onset_ticks"] / ticks,
            "duration_beats": event["duration_ticks"] / ticks,
        }
        rows.append(
            {
                "identity": dict(identities[index]),
                "notes": [{**timing(n), "pitch_midi": n["pitch_midi"]} for n in measure["notes"]],
                "rests": [timing(r) for r in measure["rests"] if r.get("visible", True)],
            }
        )
    return rows, validation


def ledger_consistency(stage: dict) -> dict:
    """An optional redundant ledger cannot silently replace the authoritative ABC."""
    if "event_ledger" not in stage:
        return {"status": "not_supplied"}
    score = stage["normalized"]
    if not score["valid"]:
        return {"status": "not_checked_invalid_abc"}
    identities = [{"system_measure_index": i} for i in range(1, len(score["measures"]) + 1)]
    expected = list(range(1, len(identities) + 1))
    try:
        ledger, validation = old_adapter._adapt(
            read_json(stage["event_ledger"]), identities, expected
        )
        abc, _ = abc_to_historical(score, identities)

        def events(rows):
            return Counter(
                (
                    row["identity"]["system_measure_index"],
                    kind,
                    event["onset_beats"],
                    event["duration_beats"],
                    event.get("pitch_midi"),
                )
                for row in rows
                for kind in ("notes", "rests")
                for event in row[kind]
            )

        consistent = validation["status"] == "valid" and events(ledger) == events(abc)
        return {
            "status": "consistent" if consistent else "differs_from_authoritative_abc",
            "historical_event_scope_only": True,
            "ledger_validation": validation,
        }
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "invalid_ledger", "error": str(exc)}


def _advance_gate(initial: dict, reviewed: dict, gate: dict) -> dict:
    initial_valid = initial["submission_validation"]
    usable = all(
        initial_valid.get(key) is True
        for key in (
            "abc_valid",
            "complete_measure_coverage",
            "all_declared_meter_bars_valid",
            "headline_metrics_valid",
        )
    )
    initial_runtime_ok = (
        initial["runtime"]["status"] == "complete" and initial["runtime"]["returncode"] == 0
    )
    reviewed_runtime_ok = (
        reviewed["runtime"]["status"] == "complete" and reviewed["runtime"]["returncode"] == 0
    )
    usable = usable and initial["completion_claim"] is True and initial_runtime_ok
    minimum_bars, minimum_f1 = (
        gate["reviewed_exact_measures_min"],
        gate["reviewed_strict_note_f1_min"],
    )

    def quality(lane: dict) -> bool:
        summary, validation = lane["metrics"]["summary"], lane["submission_validation"]
        return (
            validation["headline_metrics_valid"]
            and validation["complete_measure_coverage"]
            and summary["exact_measures"] >= minimum_bars
            and summary["note_f1"] >= minimum_f1
        )

    passed = (
        usable
        and quality(reviewed)
        and reviewed_runtime_ok
        and reviewed["completion_claim"] is True
    )
    exact = (
        usable
        and initial["metrics"]["summary"]["exact_measures"] == 7
        and initial["metrics"]["summary"]["note_f1"] == 1
    )
    reasons = []
    if not initial_runtime_ok:
        reasons.append("Initial author did not finish successfully within the execution contract.")
    if not reviewed_runtime_ok:
        reasons.append(
            "Independent reader did not finish successfully within the execution contract."
        )
    if not usable:
        reasons.append("Initial submission is not a usable complete target under the fixed gate.")
    if reviewed["completion_claim"] is not True:
        reasons.append("Independent reader did not declare the reviewed submission complete.")
    if not quality(reviewed):
        reasons.append("Reviewed output does not meet the predeclared historical event threshold.")
    return {
        "initial_usable_complete": usable,
        "initial_runtime_success": initial_runtime_ok,
        "reviewed_runtime_success": reviewed_runtime_ok,
        "reviewed_completion_claim": reviewed["completion_claim"],
        "initial_met_final_quality_threshold": quality(initial),
        "reviewed_met_final_quality_threshold": quality(reviewed),
        "advance_to_rumichaca": passed,
        "independent_review_made_gate_pass": passed and not quality(initial),
        "exact_historical_reproduction_initial_only": exact,
        "threshold": {"exact_measures_min": minimum_bars, "strict_note_f1_min": minimum_f1},
        "blocking_reasons": reasons,
    }


def _aviador(
    plan: dict,
    seal: dict,
    stages: dict,
    records: list[dict],
    truth_loader: Callable | None,
) -> dict:
    target = plan["targets"]["aviador_s8"]
    expected = target["grading_only_expected_measure_indices"]
    gate = target["advance_gate"]
    if (
        expected != list(range(1, 8))
        or target["system_index"] != 8
        or gate["reviewed_exact_measures_min"] != 6
        or gate["reviewed_strict_note_f1_min"] != 0.95
        or gate["initial_usable_complete"]
        != {
            "abc_valid": True,
            "written_measures": 7,
            "all_declared_meter_bars_valid": True,
        }
    ):
        raise ValueError("Historical reproduction target/gate differs from preregistration")
    pins = target["historical_reference_pins"]
    protocol_path = verify_pin(ROOT, target["historical_protocol"], "historical protocol")
    protocol = read_json(protocol_path)
    if protocol.get("reference_pins") != pins or protocol.get("target") != {
        "slug": target["slug"],
        "system_index": 8,
        "expected_measure_indices": expected,
    }:
        raise ValueError("Preregistered references differ from historical protocol")
    paths = {
        key: verify_pin(ROOT, record, f"historical {key}")
        for key, record in pins.items()
        if key != "truth"
    }
    if paths["evaluator"] != Path(historical.__file__).resolve():
        raise ValueError("Historical scorer pin differs from imported implementation")
    freeze_records = old_adapter._verify_freeze(
        paths["baseline_freeze"], paths["requests"], paths["baseline_predictions"]
    )
    request_target = {
        "slug": target["slug"],
        "system_index": 8,
        "expected_measure_indices": expected,
    }
    requests = old_adapter._target_rows(historical._read_jsonl(paths["requests"]), request_target)
    identities = [row["identity"] for row in requests]
    baseline = historical._read_jsonl(paths["baseline_predictions"])
    old_adapter._same_identities(baseline, identities, "baseline")
    predictions, lanes = {}, {}
    for stage in STAGES:
        predictions[stage], validation = abc_to_historical(stages[stage]["normalized"], identities)
        lanes[stage] = {
            "submission_validation": validation,
            "completion_claim": stages[stage]["provenance_data"]["complete"],
            "decisions": stages[stage]["decisions_data"],
            "runtime": stages[stage]["provenance_data"],
        }
    # No reference event content is parsed until both candidate stages were verified
    # and normalized, and all non-truth historical pins checked above.
    truth_path = verify_pin(ROOT, pins["truth"], "historical truth")
    truth = old_adapter._target_rows(
        (truth_loader or historical._read_jsonl)(truth_path), request_target
    )
    old_adapter._same_identities(truth, identities, "truth")
    baseline_metrics = historical.evaluate_predictions(truth, baseline)
    historical_summary = read_json(paths["baseline_report"])["validation"]["metrics"]["summary"]
    if baseline_metrics["summary"] != historical_summary:
        raise ValueError("Frozen baseline does not reproduce the historical metric summary")
    for stage in STAGES:
        errors = old_adapter._geometry_errors(truth, predictions[stage])
        lanes[stage]["submission_validation"]["event_geometry_errors"] = errors
        if errors:
            lanes[stage]["submission_validation"]["headline_metrics_valid"] = False
        lanes[stage]["metrics"] = historical.evaluate_predictions(
            truth, [] if errors else predictions[stage]
        )
        lanes[stage]["adapted_events"] = predictions[stage]
    records.extend(pin(path) for path in [protocol_path, *paths.values(), truth_path])
    records.extend(freeze_records)
    return {
        "scorer": "scripts.build_vlm_melody_event_benchmark.evaluate_predictions",
        "baseline": {"metrics": baseline_metrics, "historical_summary_reproduced": True},
        "stages": lanes,
        "gate": _advance_gate(lanes["initial"], lanes["reviewed"], gate),
        "unsupported_scope": target["unsupported_scope"],
        "metric_notes": (
            "Strict historical note F1 jointly compares pitch/onset/duration; exact bars "
            "also require matching rests. Written ties/form are not scored here."
        ),
    }


def _rumichaca(plan: dict, stages: dict, records: list[dict]) -> dict:
    target = plan["targets"]["rumichaca_full"]
    freeze_path = verify_pin(ROOT, target["frozen_reference"], "frozen six reference")
    labels_path = verify_pin(ROOT, target["labels"], "V3 labels")
    frozen = verify_freeze(freeze_path.parent)
    labels = verify_labels(labels_path.parent, freeze_path.parent)
    work = next(w for w in frozen["works"] if w["slug"] == target["slug"])
    work_labels = next(w for w in labels["works"] if w["slug"] == target["slug"])
    if (
        frozen["files"][work["source_pdf_path"]]["sha256"] != target["source_pdf"]["sha256"]
        or labels.get("label_set") != "reviewed_six_v3"
    ):
        raise ValueError("Rumichaca source or labels differ from the frozen target contract")
    reference = normalize_abc((freeze_path.parent / work["answer_abc_path"]).read_text())
    if not reference["valid"]:
        raise ValueError("Rumichaca reference failed normalization")
    required = _source_required_edition_bars(reference, work_labels)
    lanes = {}
    for stage in STAGES:
        candidate = stages[stage]["normalized"]
        lane = compare_normalized(reference, candidate)
        source = score_source(reference, candidate, lane.get("alignment", []), work_labels)
        for field in ("metrics", "metric_status", "coverage"):
            lane[field].update(source[field])
        lane["mismatches"].extend(source["mismatches"])
        feedback = validate_candidate(
            stages[stage]["abc"].read_text(), stages[stage]["decisions_data"]
        )
        paired = {r["reference"] for r in lane.get("alignment", []) if r["candidate"] is not None}
        lane["status"] = (
            "failed"
            if not candidate["valid"]
            else (
                "complete"
                if not feedback["issues"]
                and required.issubset(paired)
                and stages[stage]["provenance_data"]["complete"]
                else "partial"
            )
        )
        lane.update(
            validation=feedback,
            decisions=stages[stage]["decisions_data"],
            source_scope=source["source_scope"],
            runtime=stages[stage]["provenance_data"],
        )
        if lane["runtime"]["status"] != "complete" or lane["runtime"]["returncode"] != 0:
            lane["status"] = "partial" if candidate["valid"] else "failed"
        lanes[stage] = lane
    records.extend([pin(freeze_path), pin(labels_path)])
    return {
        "scorer": "complete-score-v2 plus reviewed_six_v3 audited note subset",
        **(
            {
                "reference_summary": {
                    "written_measures": len(reference["measures"]),
                    "performed_measures": (
                        len(reference["route"])
                        if reference.get("route_status") == "resolved"
                        and isinstance(reference.get("route"), list)
                        else None
                    ),
                    "route_status": reference.get("route_status"),
                }
            }
            if plan.get("kind") == TRANSPORT_PLAN_KIND
            else {}
        ),
        "stages": lanes,
        "gate": None,
        "unsupported_scope": ["whole-score source accuracy", "automatic editorial acceptance"],
        "metric_notes": (
            "Edition agreement includes chosen reconstructions/form. Source F1 covers only "
            "the nine pre-existing model-audited bars. This is a different assisted harness, "
            "not a tool-free schema2 run."
        ),
    }


def supplemental_source_scores(
    plan: dict, stages: dict, measured: dict, records: list[dict]
) -> None:
    """Add a separately pinned audit without changing historical V3 measurements."""
    label_pin = plan.get("supplemental_source_labels")
    if label_pin is None:
        return
    target = plan["targets"]["rumichaca_full"]
    freeze_path = verify_pin(ROOT, target["frozen_reference"], "frozen six reference")
    label_path = verify_pin(ROOT, label_pin, "supplemental source labels")
    frozen = verify_freeze(freeze_path.parent)
    labels = verify_labels(label_path.parent, freeze_path.parent)
    if labels.get("label_set") != "reviewed_six_v4":
        raise ValueError("Supplemental source scores require the explicit V4 label set")
    work = next(w for w in frozen["works"] if w["slug"] == target["slug"])
    work_labels = next(w for w in labels["works"] if w["slug"] == target["slug"])
    reference = normalize_abc((freeze_path.parent / work["answer_abc_path"]).read_text())
    for stage in STAGES:
        lane = measured["stages"][stage]
        lane["supplemental_source"] = {
            "label_set": labels["label_set"],
            "label_manifest": pin(label_path),
            **score_source(
                reference, stages[stage]["normalized"], lane.get("alignment", []), work_labels
            ),
        }
    records.append(pin(label_path))


def _verify_api_pair_artifacts(plan: dict, root: Path, stages: dict, archived: dict) -> None:
    """Bind API source-only preparation/audit to each saved runtime, including prompts."""
    from scripts.experiments import run_astra_baseline_reproduction as runner

    target = _transport_target(plan)
    for stage, prompt in zip(STAGES, _transport_prompts(target), strict=True):
        packet, audit, result = root / f"{stage}_packet", root / f"{stage}_audit", root / stage
        for directory, kind in ((packet, "input"), (audit, "audit"), (result, "stage")):
            runner.verify_seal(directory, f"astra_reproduction_{kind}")
            for path in directory.rglob("*"):
                if path.is_file() and archived.get(path.resolve()) != sha(path):
                    raise ValueError("API packet, audit or stage absent from submission archive")
        metadata = read_json(packet / "preparation.json")
        audit_data = read_json(audit / "audit.json")
        provenance = stages[stage]["provenance_data"]
        initial_hash = None if stage == "initial" else sha(stages["initial"]["abc"])
        template = verify_pin(
            root.parent, plan["current_harness"]["prompts"][prompt], "API template"
        )
        expected_prompt = template.read_text()
        if stage == "reviewed":
            expected_prompt += (
                "\n\n<untrusted_candidate_draft>\n"
                + json.dumps(
                    {
                        "abc": stages["initial"]["abc"].read_text(),
                        "decisions": read_json(stages["initial"]["decisions"]),
                    },
                    ensure_ascii=False,
                )
                + "\n</untrusted_candidate_draft>\n"
            )
        expected = {
            "stage": stage,
            "auth_mode": "openai_api",
            "model": plan["model"],
            "reasoning_effort": plan["reasoning_effort"],
            "workflow_id": WORKFLOW,
            "reference_access": False,
            "independent_reader": stage == "reviewed",
            "initial_abc_sha256": initial_hash,
            "completion_policy": plan["completion_policy"],
            "completion_policy_sha256": plan["completion_policy_sha256"],
            "source_pdf_sha256": plan["targets"][target]["source_pdf"]["sha256"],
        }
        if (
            any(
                metadata.get(key) != value or provenance.get(key) != value
                for key, value in expected.items()
            )
            or metadata.get("limit_seconds")
            != plan["stage_protocol"]["deadline_seconds"][target][stage]
            or metadata.get("prompt_template_sha256")
            != plan["current_harness"]["prompts"][prompt]["sha256"]
            or (packet / "bundle/prompt.txt").read_text() != expected_prompt
            or metadata.get("prompt_sha256") != sha(packet / "bundle/prompt.txt")
            or sha(packet / "bundle/source.pdf") != expected["source_pdf_sha256"]
            or provenance.get("source_execution") != "sandboxed_source_process"
            or provenance.get("input_seal_sha256") != sha(packet / "seal.json")
            or audit_data.get("status") != "passed"
            or audit_data.get("backend") != "api"
            or audit_data.get("input_seal_sha256") != sha(packet / "seal.json")
            or stages[stage]["abc"] != result / "score.abc"
            or stages[stage]["decisions"] != result / "decisions.json"
            or stages[stage]["provenance"] != result / "stage_provenance.json"
        ):
            raise ValueError("API input, prompt, audit or execution linkage differs")
        runner.verify_source_bundle(packet / "bundle", metadata)
    if datetime.fromisoformat(
        stages["reviewed"]["provenance_data"]["started_at"]
    ) < datetime.fromisoformat(stages["initial"]["provenance_data"]["finished_at"]):
        raise ValueError("Independent API review must follow the completed author stage")


def seal_subscription_pair(plan_path: Path, case_root: Path) -> Path:
    """Bind both saved subscription stages before any numerical reference scoring."""
    return _seal_pair(plan_path, case_root, auth_mode="chatgpt_subscription")


def seal_api_pair(plan_path: Path, case_root: Path) -> Path:
    """Bind both API stages under the explicit single-target transport contract."""
    if read_json(plan_path).get("kind") != TRANSPORT_PLAN_KIND:
        raise ValueError("API pair requires a single-target transport plan")
    return _seal_pair(plan_path, case_root, auth_mode="openai_api")


def _seal_pair(plan_path: Path, case_root: Path, *, auth_mode: str) -> Path:
    from scripts.experiments import run_astra_baseline_reproduction as runner

    plan_path, case_root = Path(plan_path).resolve(), Path(case_root).resolve()
    plan = read_json(plan_path)
    case = case_root.name
    if case not in {"aviador_s8", "rumichaca_full"}:
        raise ValueError("Expected a preregistered case directory")
    destination = case_root / "submission_seal.json"
    if destination.exists():
        raise FileExistsError("Submission seal is create-once")
    stages, artifacts = {}, []
    for stage in STAGES:
        packet, audit, result = (
            case_root / f"{stage}_packet",
            case_root / f"{stage}_audit",
            case_root / stage,
        )
        for directory, kind in (
            (packet, "astra_reproduction_input"),
            (audit, "astra_reproduction_audit"),
            (result, "astra_reproduction_stage"),
        ):
            runner.verify_seal(directory, kind)
            artifacts.extend(pin(path) for path in sorted(directory.rglob("*")) if path.is_file())
        provenance, audit_record = read_json(result / "stage_provenance.json"), read_json(
            audit / "audit.json"
        )
        if (
            provenance.get("auth_mode") != auth_mode
            or provenance.get("input_seal_sha256") != sha(packet / "seal.json")
            or (
                auth_mode == "chatgpt_subscription"
                and provenance.get("audit_sha256") != sha(audit / "seal.json")
            )
            or (auth_mode == "openai_api" and audit_record.get("backend") != "api")
            or audit_record.get("status") != "passed"
            or audit_record.get("input_seal_sha256") != sha(packet / "seal.json")
        ):
            label = "Subscription" if auth_mode == "chatgpt_subscription" else "API"
            raise ValueError(f"{label} input, audit and execution linkage differs")
        stages[stage] = {
            key: pin(result / filename)
            for key, filename in (
                ("abc", "score.abc"),
                ("decisions", "decisions.json"),
                ("provenance", "stage_provenance.json"),
            )
        }
    seal = {
        "schema_version": 1,
        "kind": SEAL_KIND,
        "status": SEAL_STATUS,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": sha(plan_path),
        "target_id": case,
        "source_pdf_sha256": plan["targets"][case]["source_pdf"]["sha256"],
        "reference_feedback_given": False,
        "evidence_kind": plan["evidence_kind"],
        "execution_contract": pin(plan_path.parent / "execution_contract.json"),
        "stages": stages,
        "artifacts": artifacts,
    }
    with destination.open("x") as handle:
        json.dump(seal, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return destination


def evaluate(
    plan_path: Path, seal_path: Path, output_dir: Path, *, truth_loader: Callable | None = None
) -> dict:
    plan_path, seal_path, output_dir = (
        Path(p).resolve() for p in (plan_path, seal_path, output_dir)
    )
    if output_dir.exists():
        raise FileExistsError("Evaluation output is create-once")
    plan, seal, stages, records = verify_submission(plan_path, seal_path)
    for stage in STAGES:
        stages[stage]["normalized"] = normalize_abc(stages[stage]["abc"].read_text())
        stages[stage]["ledger_check"] = ledger_consistency(stages[stage])
    if seal["target_id"] == "aviador_s8":
        measured = _aviador(plan, seal, stages, records, truth_loader)
    else:
        measured = _rumichaca(plan, stages, records)
        supplemental_source_scores(plan, stages, measured, records)
    for stage in STAGES:
        measured["stages"][stage]["event_ledger_consistency"] = stages[stage]["ledger_check"]
    for record in records:
        verify_pin(ROOT, record, "final source/artifact recheck")
    report = {
        "schema_version": 1,
        "kind": "astra_assisted_workflow_reproduction_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence_kind": seal.get("evidence_kind", plan["evidence_kind"]),
        "target_id": seal["target_id"],
        "workflow_id": WORKFLOW,
        "model": plan["model"],
        "reasoning_effort": plan["reasoning_effort"],
        "both_stages_verified_before_reference_content": True,
        "isolation_basis": (
            "Stage provenance attests source-only access; runtime/context audit artifacts "
            "remain separate evidence. This is not the prior tool-free harness."
        ),
        **measured,
        "stage_artifacts": seal["stages"],
        "pins": records,
        "implementations": {
            "adapter": pin(Path(__file__)),
            "historical_scorer": pin(Path(historical.__file__)),
            "historical_adapter": pin(Path(old_adapter.__file__)),
            "abc_normalizer": pin(ROOT / "score2abc/score_eval_abc.py"),
            "abc_parser": pin(ROOT / "score2abc/score_eval_parse.js"),
            "complete_score_grader": pin(ROOT / "score2abc/score_eval.py"),
            "source_grader": pin(ROOT / "score2abc/score_eval_source.py"),
            "renderer": renderer_identity(),
        },
        "limitations": [
            "Consumed development evidence; no claim about unseen scores or all 100 manuscripts.",
            "Initial and independent-reader outputs are separate stages, not independent trials.",
            "No reference feedback or adaptive retry is permitted after scoring either stage.",
        ],
    }
    if plan["kind"] == COMPARISON_PLAN_KIND:
        report.update(
            comparison_policy=plan["comparison_policy"],
            baseline_predecessor=plan["baseline_predecessor"],
        )
    if plan["kind"] == TRANSPORT_PLAN_KIND:
        report.update(
            transport_predecessor=plan["transport_predecessor"],
            transport_scope=plan["transport_scope"],
        )
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("seal", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.plan, args.seal, args.output)
    print(
        json.dumps(
            {
                "report": str(args.output / "report.json"),
                "target": report["target_id"],
                "gate": report["gate"],
            }
        )
    )


if __name__ == "__main__":
    main()
