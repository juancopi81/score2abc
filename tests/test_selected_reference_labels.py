"""A selected reference must not turn uncertain editorial choices into source truth."""

import copy
import json

import pytest

from score2abc.score_eval_source import NOTE_FIELDS, note_events_sha256
from score2abc.selected_reference_labels import compile_source_labels, digest
from scripts import build_selected_reference_labels as builder


def encode(value):
    return (json.dumps(value, ensure_ascii=False) + "\n").encode()


@pytest.fixture
def audit():
    note = dict(zip(NOTE_FIELDS, [0, 384, 60, False, False], strict=True))
    normalized = {
        "valid": True,
        "ticks_per_quarter": 384,
        "measures": [{"number": i, "notes": [dict(note)]} for i in range(1, 4)],
    }
    packet = {
        "reference_abc_sha256": "abc",
        "reference_pdf_sha256": "pdf",
        "ticks_per_quarter": 384,
        "selection": [
            {
                "edition_bar": b["number"],
                "source_bar": str(b["number"]),
                "system": 1,
                "measure": b["number"],
                "reference_notes": copy.deepcopy(b["notes"]),
                "note_events_sha256": note_events_sha256(b),
                "expected_noteheads": 1,
                "prior_known_exclusion": "Accepted reconstruction" if b["number"] == 3 else None,
            }
            for b in normalized["measures"]
        ],
    }
    reports = [
        {
            "reader_id": name,
            "all_rows_inspected": True,
            "candidate_outputs_seen": False,
            "new_human_approval": False,
            "input_hashes": {
                "abc_sha256": "abc",
                "pdf_sha256": "pdf",
                "audit_packet_sha256": digest(encode(packet)),
            },
            "source_rows": [{"system": 1, "bar_count": 3}],
            "bars": [
                {
                    **{k: row[k] for k in ("edition_bar", "source_bar", "system", "measure")},
                    "status": "supported",
                    "evidence": "Visible head, stem, duration and pitch agree.",
                    "context_checked": "Initial key and prior bar.",
                    "supported_fields": list(NOTE_FIELDS),
                    "observed_notes": copy.deepcopy(row["reference_notes"]),
                    "note_events_sha256": row["note_events_sha256"],
                }
                for row in packet["selection"]
            ],
        }
        for name in ("author", "independent_reader")
    ]
    return packet, normalized, reports


def test_editorial_exclusion_wins_even_if_both_readers_support(audit):
    packet, normalized, reports = audit
    result = compile_source_labels(encode(packet), normalized, reports)
    assert result["coverage"] == {
        "audited_bars": 3,
        "eligible_bars": 2,
        "eligible_noteheads": 2,
        "excluded_bars": 1,
    }
    assert result["source_exclusions"][0]["edition_bar"] == 3
    assert result["source_audit"]["human_approved_audit"] is False
    assert result["source_audit"]["heldout"] is False


@pytest.mark.parametrize("change", ["pitch", "tie", "duplicate", "missing", "float", "uncertain"])
def test_independent_disagreement_excludes_whole_bar(audit, change):
    packet, normalized, reports = audit
    row = reports[1]["bars"][0]
    if change == "pitch":
        row["observed_notes"][0]["pitch_midi"] = 61
    elif change == "tie":
        row["observed_notes"][0]["tie_start"] = True
    elif change == "duplicate":
        row["observed_notes"] *= 2
    elif change == "missing":
        row["observed_notes"] = []
    elif change == "float":
        row["observed_notes"][0]["pitch_midi"] = 60.0
    else:
        row["status"] = "uncertain"
    result = compile_source_labels(encode(packet), normalized, reports)
    assert [r["edition_bar"] for r in result["source_audit"]["measures"]] == [2]


@pytest.mark.parametrize(
    "change", ["same_reader", "stale", "candidate", "uninspected", "location", "coverage"]
)
def test_report_identity_and_coverage_fail_closed(audit, change):
    packet, normalized, reports = audit
    second = reports[1]
    if change == "same_reader":
        second["reader_id"] = reports[0]["reader_id"]
    elif change == "stale":
        second["input_hashes"]["abc_sha256"] = "old"
    elif change == "candidate":
        second["candidate_outputs_seen"] = True
    elif change == "uninspected":
        second["all_rows_inspected"] = False
    elif change == "location":
        second["bars"][0]["system"] = 2
    else:
        second["bars"].pop()
    with pytest.raises(ValueError):
        compile_source_labels(encode(packet), normalized, reports)


@pytest.fixture
def publication(tmp_path, monkeypatch, audit):
    packet, normalized, reports = audit
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    abc = b"X:1\nK:C\nC|C|C|\n"
    review = encode({"review_notes": "Final note is an adopted reconstruction."})
    (frozen / "review.abc").write_bytes(abc)
    (frozen / "review.json").write_bytes(review)
    manifest = {
        "works": [
            {
                "slug": "fixture",
                "revision": 14,
                "answer_abc_path": "review.abc",
                "source_pdf_path": "source.pdf",
                "abc_sha256": digest(abc),
            }
        ],
        "files": {"source.pdf": {"sha256": "pdf"}, "review.json": {"sha256": digest(review)}},
    }
    (frozen / "manifest.json").write_bytes(encode(manifest))
    packet.update(reference_abc_sha256=digest(abc), freeze_manifest_sha256=digest(encode(manifest)))
    for report in reports:
        report["input_hashes"].update(
            abc_sha256=digest(abc), audit_packet_sha256=digest(encode(packet))
        )
    evidence = {
        "packet.json": encode(packet),
        "author.json": encode(reports[0]),
        "review.json": encode(reports[1]),
        "decisions.json": encode(
            {
                "decisions": [
                    {
                        "id": "reconstruction",
                        "evidence": [
                            {
                                "path": "review.json",
                                "sha256": digest(review),
                                "field": "review_notes",
                                "quote": "adopted reconstruction",
                            }
                        ],
                    }
                ]
            }
        ),
    }
    monkeypatch.setattr(builder, "verify_freeze", lambda _: manifest)
    monkeypatch.setattr(builder, "normalize_abc", lambda _: normalized)
    return frozen, evidence, tmp_path / "labels"


def test_create_once_and_tamper_detection(publication):
    frozen, evidence, destination = publication
    result = builder.publish(frozen, evidence, destination, "fixture_v1")
    assert builder.verify(destination, frozen) == result
    with pytest.raises(ValueError, match="already exists"):
        builder.publish(frozen, evidence, destination, "fixture_v1")
    with (destination / "review.json").open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="content changed"):
        builder.verify(destination, frozen)


@pytest.mark.parametrize(
    "mutation", ["invented_quote", "outside_evidence", "extra_file", "inside_freeze"]
)
def test_invalid_evidence_never_publishes(publication, mutation):
    frozen, evidence, destination = publication
    decisions = json.loads(evidence["decisions.json"])
    if mutation == "invented_quote":
        decisions["decisions"][0]["evidence"][0]["quote"] = "not in the history"
    elif mutation == "outside_evidence":
        decisions["decisions"][0]["evidence"][0]["path"] = "../outside"
    elif mutation == "extra_file":
        evidence["../outside.json"] = b"{}"
    else:
        destination = frozen / "labels"
    evidence["decisions.json"] = encode(decisions)
    with pytest.raises(ValueError):
        builder.publish(frozen, evidence, destination, "fixture_v1")
    assert not destination.exists()
