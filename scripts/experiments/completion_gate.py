"""Source-only structural completion policy; standalone for isolated tool bundles.

This checks a declared final draft, not musical accuracy or PDF coverage. It
never fills missing beats, reads references, or starts another model call.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

POLICY_ID = "completion_gate_v1"
DECISION_FIELDS = ("id", "location", "category", "adopted_choice", "reason", "uncertainty")
RESOLUTION_FIELDS = (
    "issue_id",
    "abc_sha256",
    "category",
    "location",
    "adopted_choice",
    "reason",
    "uncertainty",
    "inspection_call",
)
EXCEPTION_CATEGORIES = ("source_irregular_meter", "unpaired_pickup")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def issues_for(normalized: dict, decisions: list) -> list[dict]:
    """Rebuild issues from parser output instead of trusting a saved ready flag."""
    issues = [{"kind": "parse", "detail": e} for e in normalized.get("errors", [])]
    if normalized.get("valid") is not True and not issues:
        issues.append({"kind": "parse", "detail": "Notation is not valid."})
    for warning in normalized.get("warnings", []):
        issues.append({"kind": "parser_warning", "detail": warning})
    measures = normalized.get("measures", [])
    if not measures:
        issues.append({"kind": "empty", "detail": "No written measures were parsed."})
    for bar in measures:
        if bar.get("meter_valid") is not True:
            issues.append(
                {
                    "kind": "meter",
                    "measure": bar["number"],
                    "observed_ticks": bar["duration_ticks"],
                    "expected_ticks": bar["expected_ticks"],
                    "partial_reason": bar.get("partial_reason"),
                    "detail": "Inspect the source; do not pad rests automatically.",
                }
            )
    if normalized.get("route_status") != "resolved":
        issues.append({"kind": "form", "detail": "Playback route is unresolved."})
    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict) or any(not decision.get(k) for k in DECISION_FIELDS):
            issues.append(
                {
                    "kind": "decisions",
                    "decision": index + 1,
                    "detail": "Decisions need " + ", ".join(DECISION_FIELDS),
                }
            )
    return [dict(item, issue_id=sha(json.dumps(item, sort_keys=True).encode())) for item in issues]


def evaluate(
    normalized: dict,
    decisions: list,
    *,
    abc_sha256: str,
    complete_claimed: bool,
    resolutions: object = None,
    inspections: dict | None = None,
) -> dict:
    issues = issues_for(normalized, decisions)
    by_id = {issue["issue_id"]: issue for issue in issues}
    accepted, errors, seen = [], [], set()
    inspections = inspections or {}
    if resolutions is None:
        resolutions = []
    if not isinstance(resolutions, list) or len(resolutions) > 128:
        errors.append("Resolutions must be an array of at most 128 entries.")
        resolutions = []
    for index, resolution in enumerate(resolutions):
        prefix = f"Resolution {index + 1}: "
        if not isinstance(resolution, dict) or set(resolution) != set(RESOLUTION_FIELDS):
            errors.append(prefix + "supply exactly the required resolution fields.")
            continue
        text_fields = set(RESOLUTION_FIELDS) - {"inspection_call"}
        if any(
            not isinstance(resolution[k], str)
            or not resolution[k].strip()
            or len(resolution[k]) > 4000
            for k in text_fields
        ):
            errors.append(prefix + "text fields must be nonempty and bounded.")
            continue
        issue_id = resolution["issue_id"]
        if issue_id in seen:
            errors.append(prefix + "duplicate issue resolution.")
            continue
        seen.add(issue_id)
        issue = by_id.get(issue_id)
        if not issue or resolution["abc_sha256"] != abc_sha256:
            errors.append(prefix + "stale or unmatched issue/draft hash; inspect current feedback.")
            continue
        if issue["kind"] != "meter" or not issue.get("expected_ticks"):
            errors.append(prefix + "this issue must be corrected, not waived.")
            continue
        category = resolution["category"]
        if category not in EXCEPTION_CATEGORIES or (
            category == "unpaired_pickup" and issue.get("partial_reason") != "pickup_candidate"
        ):
            errors.append(prefix + "unsupported source exception category.")
            continue
        call = resolution["inspection_call"]
        if type(call) is not int or call not in inspections:
            errors.append(
                prefix + "cite an earlier inspect_source call with verified crop evidence."
            )
            continue
        accepted.append(
            {"issue_id": issue_id, "resolution": resolution, "source_inspection": inspections[call]}
        )
    resolved = {item["issue_id"] for item in accepted}
    unresolved = [issue for issue in issues if issue["issue_id"] not in resolved]
    ready = complete_claimed is True and not unresolved and not errors
    return {
        "policy_version": POLICY_ID,
        "abc_sha256": abc_sha256,
        "complete_claimed": complete_claimed,
        "completion_ready": ready,
        "status": (
            "ready_with_source_exceptions"
            if ready and accepted
            else ("ready" if ready else "needs_resolution" if unresolved or errors else "partial")
        ),
        "issues": issues,
        "unresolved_issues": unresolved,
        "accepted_source_exceptions": accepted,
        "resolution_errors": errors,
        "musical_accuracy": "not_measured_by_completion_gate",
        "source_coverage": "not_measured_by_completion_gate",
        "next_action": (
            "Ready for the next workflow stage; musical review remains necessary."
            if ready
            else "Keep this saved draft. Inspect source/context, correct the ABC and save again. "
            "For a genuinely irregular source bar only, submit a structured source exception "
            "using this draft hash and issue ID. Do not insert rests just to satisfy meter."
        ),
    }


def verified_inspections(artifacts: Path, before: str) -> dict:
    """Resolve only fixed inspection paths produced before this checkpoint."""
    events_path = artifacts / "tool_events.jsonl"
    if not events_path.exists():
        return {}
    result = {}
    for line in events_path.read_text().splitlines():
        event = json.loads(line)
        if event.get("name") != "inspect_source" or event.get("finished_at", "~") > before:
            continue
        evidence = event.get("evidence", {})
        call = event.get("call")
        relative = evidence.get("path", "")
        if type(call) is not int or relative != f"inspections/inspect_{call:03d}.png":
            continue
        image = artifacts / relative
        if image.is_symlink() or image.resolve().parent != (artifacts / "inspections").resolve():
            continue
        if image.is_file() and sha(image.read_bytes()) == evidence.get("sha256"):
            result[call] = evidence
    return result


def evaluate_checkpoint(checkpoint_dir: Path, artifacts_dir: Path | None = None) -> dict:
    """Read immutable evidence; never rewrite a checkpoint or trust its gate report."""

    def read(name):
        return json.loads((checkpoint_dir / name).read_text())

    checkpoint = read("checkpoint.json")
    abc_hash = sha((checkpoint_dir / "score.abc").read_bytes())
    decisions_bytes = (checkpoint_dir / "decisions.json").read_bytes()
    if (
        abc_hash != checkpoint["abc_sha256"]
        or sha(decisions_bytes) != checkpoint["decisions_sha256"]
    ):
        raise ValueError("Checkpoint text or decisions changed")
    artifacts = artifacts_dir if artifacts_dir is not None else checkpoint_dir.parent.parent
    resolution_file = checkpoint_dir / "completion_resolutions.json"
    evidence_error = None
    try:
        resolutions = read("completion_resolutions.json") if resolution_file.exists() else []
        inspections = (
            verified_inspections(artifacts, checkpoint["created_at"]) if resolutions else {}
        )
    except (ValueError, OSError):
        # A deadline may interrupt a sidecar/event write after ABC was saved.
        # Keep the usable draft, but never treat incomplete evidence as a waiver.
        resolutions, inspections = [], {}
        evidence_error = "Resolution or inspection evidence is unreadable or incomplete."
    result = evaluate(
        read("normalized.json"),
        json.loads(decisions_bytes),
        abc_sha256=abc_hash,
        complete_claimed=checkpoint.get("complete_claimed") is True,
        resolutions=resolutions,
        inspections=inspections,
    )
    if evidence_error:
        result["resolution_errors"].append(evidence_error)
        result.update(completion_ready=False, status="needs_resolution")
        result["next_action"] = "Keep the saved ABC; resave complete source-exception evidence."
    return result
