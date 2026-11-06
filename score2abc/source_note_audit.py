"""Add only independently agreed source-note masks; preserve the previous labels.

Readers see the PDF and frozen reference, but not candidate outputs. This is
consumed, model-assisted evidence, not independent human ground truth.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from score2abc.score_eval_source import NOTE_FIELDS, note_events, note_events_sha256

EVIDENCE_PATH = Path(__file__).with_name("source_note_audit_v4_evidence.json")
READERS = {
    "amalia": ("audit4_amalia_aviador", "audit4_coqueteos_rumichaca"),
    "aviador": ("audit4_amalia_aviador", "audit4_coqueteos_rumichaca"),
    "atardecer": ("audit4_atardecer_carrizal", "audit4_amalia_aviador"),
    "carrizal": ("audit4_atardecer_carrizal", "audit4_amalia_aviador"),
    "coqueteos": ("audit4_coqueteos_rumichaca", "audit4_atardecer_carrizal"),
    "rumichaca": ("audit4_coqueteos_rumichaca", "audit4_atardecer_carrizal"),
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def valid_notes(notes: list) -> bool:
    return all(
        set(n) == set(NOTE_FIELDS)
        and all(type(n[f]) is int for f in NOTE_FIELDS[:3])
        and n["onset_ticks"] >= 0
        and n["duration_ticks"] > 0
        and 0 <= n["pitch_midi"] <= 127
        and all(type(n[f]) is bool for f in NOTE_FIELDS[3:])
        for n in notes
    )


def _document(bundle: dict, name: str) -> tuple[dict, str]:
    record = bundle["files"][name]
    data = record["content"].encode("utf-8")
    if sha(data) != record["sha256"]:
        raise ValueError(f"Audit evidence hash mismatch: {name}")
    return json.loads(data), record["sha256"]


def bundle_evidence(root: Path) -> dict:
    plan = json.loads((root / "audit_plan.json").read_text())
    if (root / "audit_plan.sha256").read_text().strip() != sha(
        (root / "audit_plan.json").read_bytes()
    ):
        raise ValueError("Audit plan changed")
    names = ["audit_plan.json"] + [
        f"works/{work['key']}/{name}.json"
        for work in plan["works"]
        for name in ("audit_packet", "audit.author", "audit.review")
    ]
    files = {}
    for name in names:
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Audit evidence escapes root")
        data = path.read_bytes()
        files[name] = {"sha256": sha(data), "content": data.decode("utf-8")}
    return {"schema_version": 1, "reader_assignments": READERS, "files": files}


def expand_inventory(prior: dict, bundle: dict) -> tuple[dict, dict]:
    """Build the new inventory from exact records, never from candidate metrics."""
    plan, plan_sha = _document(bundle, "audit_plan.json")
    if (
        prior["label_set"] != "reviewed_six_v3"
        or plan["reference_manifest_sha256"] != prior["reference_manifest_sha256"]
        or plan["direct_api_calls"] != 0
        or len(plan["works"]) != len(prior["works"])
        or {w["key"] for w in plan["works"]} != set(prior["works"])
    ):
        raise ValueError("Audit plan or previous label identity differs")
    inventory = copy.deepcopy(prior)
    inventory["label_set"] = "reviewed_six_v4"
    summary = {"plan_sha256": plan_sha, "works": {}, "scope": "consumed_model_assisted"}
    for planned in plan["works"]:
        key = planned["key"]
        readers_assigned = bundle["reader_assignments"][key]
        if len(readers_assigned) != 2 or len(set(readers_assigned)) != 2:
            raise ValueError("Audit requires distinct author and reviewer agents")
        packet, packet_sha = _document(bundle, f"works/{key}/audit_packet.json")
        audit = inventory["works"][key]["source_audit"]
        if (
            packet_sha != planned["packet_sha256"]
            or packet["source_pdf_sha256"] != audit["source_pdf_sha256"]
            or packet["ticks_per_quarter"] != audit["ticks_per_quarter"]
            or packet["note_fields"] != list(NOTE_FIELDS)
            or packet["work"] != key
            or packet["slug"] != planned["slug"]
        ):
            raise ValueError(f"Audit packet identity differs: {key}")
        old_bars = {r["edition_bar"] for r in audit["measures"]}
        selected = planned["selected_edition_bars"]
        if (
            len(selected) != 10
            or len(set(selected)) != 10
            or set(selected) & (old_bars | set(planned["excluded_bars"]))
            or len(old_bars) != planned["existing_audited_bars"]
            or [r["edition_bar"] for r in packet["selection"]] != selected
        ):
            raise ValueError(f"Audit selection differs: {key}")
        reports, hashes = [], []
        for name, role in (("author", "author"), ("review", "independent_reviewer")):
            report, report_sha = _document(bundle, f"works/{key}/audit.{name}.json")
            if (
                report["work"] != key
                or report["role"] != role
                or report["packet_sha256"] != packet_sha
                or report["source_pdf_sha256"] != packet["source_pdf_sha256"]
                or report["reference_abc_sha256"] != packet["reference_abc_sha256"]
                or report["candidate_outputs_seen"] is not False
                or report["new_human_approval"] is not False
                or report["model"] != plan["model"]
                or report["reasoning_effort"] != plan["reasoning_effort"]
                or len(report["bars"]) != len(selected)
                or {r["edition_bar"] for r in report["bars"]} != set(selected)
            ):
                raise ValueError(f"Audit reader identity or coverage differs: {key}/{name}")
            reports.append({r["edition_bar"]: r for r in report["bars"]})
            hashes.append(report_sha)
        accepted, rejected = [], []
        for target in packet["selection"]:
            bar = target["edition_bar"]
            if (
                not valid_notes(target["reference_notes"])
                or note_events_sha256({"notes": target["reference_notes"]})
                != target["note_events_sha256"]
                or len(target["reference_notes"]) != target["expected_noteheads"]
            ):
                raise ValueError(f"Invalid reference note pin: {key}/{bar}")
            readers = [r[bar] for r in reports]
            reasons = []
            for name, reader in zip(("author", "reviewer"), readers, strict=True):
                if (
                    reader["source_bar"] != target["source_bar"]
                    or reader["reference_note_events_sha256"] != target["note_events_sha256"]
                    or reader["verdict"] not in {"supported", "ambiguous", "contradicted"}
                    or not reader["evidence"]
                    or not reader["context_checked"]
                ):
                    raise ValueError(f"Invalid reader evidence: {key}/{bar}/{name}")
                if reader["verdict"] != "supported":
                    reasons.append(f"{name}: {reader['verdict']}")
                elif (
                    reader["issues"]
                    or not valid_notes(reader["observed_notes"])
                    or note_events({"notes": reader["observed_notes"]})
                    != note_events({"notes": target["reference_notes"]})
                ):
                    reasons.append(f"{name}: supported verdict lacks exact note-field agreement")
            location = readers[0]["source_location"]
            if readers[1]["source_location"] != location or (
                target.get("source_location") and target["source_location"] != location
            ):
                reasons.append("source-location disagreement")
            if reasons:
                rejected.append({"edition_bar": bar, "reasons": reasons})
                continue
            accepted.append(bar)
            audit["measures"].append(
                {
                    "source_bar": target["source_bar"],
                    "edition_bar": bar,
                    "source_location": location,
                    "dimensions": ["notes"],
                    "note_fields": list(NOTE_FIELDS),
                    "note_events_sha256": target["note_events_sha256"],
                    "expected_noteheads": target["expected_noteheads"],
                    "audit_note": "Two source-only readers agree. Author: "
                    + readers[0]["evidence"]
                    + " Reviewer: "
                    + readers[1]["evidence"],
                }
            )
        audit["measures"].sort(key=lambda r: r["edition_bar"])
        audit["v3_metadata"] = {
            field: audit[field]
            for field in ("selection_policy", "visual_method", "pending")
            if field in audit
        }
        audit["selection_policy"] = (
            "Original v3 masks retained; v4 uses ten preselected additional bars per work. "
            "Both readers must support all note fields. Rejected bars are not replaced."
        )
        audit["visual_method"] = (
            "Original v3 audit plus two separate source-only readers using full PDF context "
            "and local closeups. Reader identity and original reports are hash-bound below."
        )
        audit["pending"] = [
            "Unlisted bars remain unscored. Prior v3 pending notes are historical, not v4 masks."
        ]
        audit["expansion_v4"] = {
            "reader_agents": list(readers_assigned),
            "reference_abc_sha256": packet["reference_abc_sha256"],
            "plan_sha256": plan_sha,
            "packet_sha256": packet_sha,
            "author_sha256": hashes[0],
            "reviewer_sha256": hashes[1],
            "selected_bars": selected,
            "accepted_bars": accepted,
            "rejected": rejected,
            "human_approved_audit": False,
        }
        summary["works"][key] = {
            "previous_bars": len(old_bars),
            "accepted_bars": accepted,
            "rejected": rejected,
            "total_bars": len(audit["measures"]),
        }
    return inventory, summary


def verify_expansion(inventory: dict) -> None:
    """A v4 label file must reproduce exactly from its two-reader evidence."""
    prior_path = Path(__file__).with_name("eval_labels_reviewed_six_v3.json")
    if sha(EVIDENCE_PATH.read_bytes()) != inventory["source_audit_evidence_sha256"]:
        raise ValueError("Source-audit evidence bundle changed")
    rebuilt, _ = expand_inventory(
        json.loads(prior_path.read_text()), json.loads(EVIDENCE_PATH.read_text())
    )
    rebuilt["source_audit_evidence_sha256"] = inventory["source_audit_evidence_sha256"]
    if rebuilt != inventory:
        raise ValueError("Expanded labels differ from independent reader evidence")
