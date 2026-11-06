"""Compile a separate single-edition note mask from two pinned visual audits.

This does not register a new benchmark or alter the reviewed-six defaults.
The source grader can consume the returned work's source_audit directly.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter

from score2abc.score_eval_source import NOTE_FIELDS, note_events, note_events_sha256
from score2abc.source_note_audit import valid_notes


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compile_source_labels(packet_bytes: bytes, normalized: dict, reports: list[dict]) -> dict:
    """Admit only whole bars supported by both readers; never infer missing labels."""
    packet = json.loads(packet_bytes)
    if not normalized.get("valid") or packet.get("ticks_per_quarter") != normalized.get(
        "ticks_per_quarter"
    ):
        raise ValueError("Invalid reference normalization or timing units")
    measures = normalized["measures"]
    selection = packet["selection"]
    expected = list(range(1, len(measures) + 1))
    if [r.get("edition_bar") for r in selection] != expected:
        raise ValueError("Packet must inventory every written bar exactly once in order")
    locations = [(r["system"], r["measure"]) for r in selection]
    if (
        len(set(locations)) != len(locations)
        or any(type(n) is not int or n < 1 for location in locations for n in location)
        or len({r["source_bar"] for r in selection}) != len(selection)
    ):
        raise ValueError("Invalid or duplicated source identity")
    row_counts = Counter(r["system"] for r in selection)
    source_rows = [{"system": n, "bar_count": row_counts[n]} for n in sorted(row_counts)]
    for row, bar in zip(selection, measures, strict=True):
        if (
            type(row["edition_bar"]) is not int
            or not valid_notes(row["reference_notes"])
            or row["edition_bar"] != bar["number"]
            or row["note_events_sha256"] != note_events_sha256(bar)
            or row["expected_noteheads"] != len(bar["notes"])
            or row["reference_notes"] != [{f: n[f] for f in NOTE_FIELDS} for n in bar["notes"]]
        ):
            raise ValueError("Packet reference notes differ from normalized frozen edition")
    readers = [r.get("reader_id") for r in reports]
    if len(reports) != 2 or any(not isinstance(r, str) or not r.strip() for r in readers):
        raise ValueError("Two named visual readers are required")
    if readers[0] == readers[1]:
        raise ValueError("Visual readers must be different")
    input_hashes = {
        "abc_sha256": packet["reference_abc_sha256"],
        "pdf_sha256": packet["reference_pdf_sha256"],
        "audit_packet_sha256": digest(packet_bytes),
    }
    for report in reports:
        if (
            any(report.get("input_hashes", {}).get(k) != v for k, v in input_hashes.items())
            or report.get("all_rows_inspected") is not True
            or report.get("candidate_outputs_seen") is not False
            or report.get("new_human_approval") is not False
            or [
                {k: row.get(k) for k in ("system", "bar_count")}
                for row in report.get("source_rows", [])
            ]
            != source_rows
            or [r.get("edition_bar") for r in report.get("bars", [])] != expected
        ):
            raise ValueError("Reader input pins, visual row inventory or coverage differ")
    accepted, excluded = [], []
    for index, target in enumerate(selection):
        reasons = []
        if target.get("prior_known_exclusion"):
            reasons.append(target["prior_known_exclusion"])
        evidence = []
        for report in reports:
            row = report["bars"][index]
            if any(
                row.get(k) != target[k] for k in ("edition_bar", "source_bar", "system", "measure")
            ):
                raise ValueError("Reader source/edition location differs")
            status = row.get("status")
            if status not in {"supported", "uncertain", "editorial_excluded"}:
                raise ValueError("Unknown reader verdict")
            if not isinstance(row.get("evidence"), str) or not row["evidence"].strip():
                raise ValueError(
                    "Every reader verdict needs visible-source evidence or exclusion rationale"
                )
            evidence.append({"reader_id": report["reader_id"], "evidence": row["evidence"]})
            if status == "supported":
                if (
                    row.get("supported_fields") != list(NOTE_FIELDS)
                    or row.get("note_events_sha256") != target["note_events_sha256"]
                ):
                    raise ValueError("Supported bar must pin every note field and duplicate head")
                if (
                    not row.get("context_checked")
                    or not valid_notes(row.get("observed_notes", []))
                    or note_events({"notes": row.get("observed_notes", [])})
                    != note_events({"notes": target["reference_notes"]})
                ):
                    reasons.append(
                        f"{report['reader_id']}: observed note fields differ or lack context"
                    )
            else:
                reasons.append(f"{report['reader_id']}: {status}: {row['evidence']}")
        location = {"page": 1, "system": target["system"], "measure": target["measure"]}
        if reasons:
            excluded.append(
                {
                    "edition_bar": target["edition_bar"],
                    "source_location": location,
                    "reasons": reasons,
                }
            )
        else:
            accepted.append(
                {
                    "source_bar": target["source_bar"],
                    "edition_bar": target["edition_bar"],
                    "source_location": location,
                    "dimensions": ["notes"],
                    "note_fields": list(NOTE_FIELDS),
                    "note_events_sha256": target["note_events_sha256"],
                    "expected_noteheads": target["expected_noteheads"],
                    "audit_note": " | ".join(e["evidence"] for e in evidence),
                    "reader_evidence": evidence,
                }
            )
    return {
        "source_audit": {
            "status": "partial_model_assisted_visual_audit",
            "human_approved_audit": False,
            "heldout": False,
            "source_pdf_sha256": packet["reference_pdf_sha256"],
            "ticks_per_quarter": packet["ticks_per_quarter"],
            "total_source_measures": len(selection),
            "selection_policy": (
                "All mapped bars inspected; intersection of two readers, "
                "excluding editorial choices"
            ),
            "measures": accepted,
        },
        "source_exclusions": excluded,
        "source_rows": source_rows,
        "readers": readers,
        "coverage": {
            "audited_bars": len(selection),
            "eligible_bars": len(accepted),
            "eligible_noteheads": sum(r["expected_noteheads"] for r in accepted),
            "excluded_bars": len(excluded),
        },
        "unscored_dimensions": [
            "rests",
            "chord_symbols",
            "grace_notes",
            "articulation",
            "dynamics",
            "form",
        ],
    }
