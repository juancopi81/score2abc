"""Grade only explicitly audited source-note regions of a performing edition.

These labels are a partial, model-assisted visual audit of consumed material,
not a claim that every reviewed-edition event appears in the PDF. Alignment is
supplied by the edition grader; uncertain and uninspected regions stay unscored.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter

NOTE_FIELDS = ("onset_ticks", "duration_ticks", "pitch_midi", "tie_start", "tie_end")


def note_events(bar: dict) -> Counter:
    """Ignore voice labels/spelling while retaining simultaneous duplicate heads."""
    return Counter(
        (
            note["onset_ticks"],
            note["duration_ticks"],
            note["pitch_midi"],
            bool(note.get("tie_start")),
            bool(note.get("tie_end")),
        )
        for note in bar.get("notes", [])
    )


def note_events_sha256(bar: dict) -> str:
    """Pin the precise events audited, including multiplicity and timing units."""
    payload = json.dumps(sorted(note_events(bar).elements()), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def score_source(reference: dict, candidate: dict, alignment: list, work_labels: dict) -> dict:
    """Score audited whole-bar notes, charging missing bars as false negatives.

    Candidate-only bars cannot be located in the PDF by this adapter and are not
    charged to an arbitrary audited region. The edition grader reports those
    extras separately. Every candidate note in an aligned audited bar does count,
    so added heads within that bar lower precision. Chords/rests/form are not part
    of this note metric, and a different editorial route is not a source error.
    """
    audit = work_labels.get("source_audit", {})
    rows = audit.get("measures", [])
    result = {
        "metrics": {"source_note_f1": None},
        "metric_status": {"source_note_f1": "pending_source_labels"},
        "coverage": {
            "source_scored_measures": 0,
            "source_eligible_measures": len(rows),
            "source_total_measures": audit.get("total_source_measures"),
            "source_matched_measures": 0,
            "source_expected_noteheads": 0,
            "source_predicted_noteheads": 0,
            "source_matched_noteheads": 0,
        },
        "mismatches": [],
        "source_scope": "partial_model_assisted_visual_audit; edition_based_bar_alignment",
    }
    if not rows:
        return result
    if not reference.get("valid"):
        raise ValueError("Source grading requires a valid reference")
    if audit.get("ticks_per_quarter") != reference.get("ticks_per_quarter"):
        raise ValueError("Source audit timing units differ from normalized reference")
    refs = {bar["number"]: bar for bar in reference["measures"]}
    eligible = set()
    for row in rows:
        number = row["edition_bar"]
        if number in eligible or number not in refs:
            raise ValueError("Duplicate or unknown audited reference bar")
        eligible.add(number)
        if (
            row.get("dimensions") != ["notes"]
            or row.get("note_fields") != list(NOTE_FIELDS)
            or row.get("note_events_sha256") != note_events_sha256(refs[number])
            or row.get("expected_noteheads") != len(refs[number]["notes"])
        ):
            raise ValueError("Audited source-note events differ from reference normalization")
    if not candidate.get("valid"):
        result["metric_status"]["source_note_f1"] = "invalid_candidate"
        return result
    if reference["ticks_per_quarter"] != candidate.get("ticks_per_quarter"):
        raise ValueError("Source grading timing units differ")
    preds = {bar["number"]: bar for bar in candidate["measures"]}
    mapping, seen_candidates = {}, set()
    for item in alignment:
        r, p = item["reference"], item["candidate"]
        if r is not None:
            if r not in refs or r in mapping:
                raise ValueError("Invalid reference bar alignment for source grading")
            mapping[r] = p
        if p is not None:
            if p not in preds or p in seen_candidates:
                raise ValueError("Invalid candidate bar alignment for source grading")
            seen_candidates.add(p)
    if set(mapping) != set(refs) or seen_candidates != set(preds):
        raise ValueError("Source grading requires complete explicit bar alignment")
    matched = predicted = expected = paired = 0
    for row in rows:
        number = row["edition_bar"]
        candidate_number = mapping[number]
        a = note_events(refs[number])
        b = note_events(preds[candidate_number]) if candidate_number is not None else Counter()
        matched += sum((a & b).values())
        expected += sum(a.values())
        predicted += sum(b.values())
        paired += int(candidate_number is not None)
        if a != b:
            result["mismatches"].append(
                {
                    "measure": number,
                    "candidate_measure": candidate_number,
                    "source_bar": row["source_bar"],
                    "source_location": row["source_location"],
                    "kind": "audited_source_notes",
                    "detail": "Notes differ in an explicitly audited PDF region (partial audit).",
                }
            )
    result["metrics"]["source_note_f1"] = (
        2 * matched / (expected + predicted) if expected + predicted else None
    )
    result["metric_status"]["source_note_f1"] = "model_audited_source_subset"
    result["coverage"].update(
        source_scored_measures=len(rows),
        source_matched_measures=paired,
        source_expected_noteheads=expected,
        source_predicted_noteheads=predicted,
        source_matched_noteheads=matched,
    )
    return result
