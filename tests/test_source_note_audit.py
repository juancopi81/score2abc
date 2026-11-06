"""Independent disagreement, multiplicity and evidence integrity must fail closed."""

from __future__ import annotations

import copy
import json

import pytest

from score2abc.score_eval_source import NOTE_FIELDS, note_events_sha256
from score2abc.source_note_audit import expand_inventory, sha
from scripts.regrade_source_audit import assert_reproduced


def save(bundle, name, value):
    content = json.dumps(value)
    bundle["files"][name] = {"content": content, "sha256": sha(content.encode())}


def edit(bundle, name, change):
    value = json.loads(bundle["files"][name]["content"])
    change(value)
    save(bundle, name, value)


@pytest.fixture
def audit():
    note = dict(zip(NOTE_FIELDS, [0, 384, 60, False, False], strict=True))
    old_row = {"edition_bar": 1, "kept": "byte-equivalent record"}
    prior = {
        "label_set": "reviewed_six_v3",
        "reference_manifest_sha256": "reference",
        "works": {
            "amalia": {
                "decisions": ["human decision"],
                "source_audit": {
                    "source_pdf_sha256": "pdf",
                    "ticks_per_quarter": 384,
                    "measures": [old_row],
                    "selection_policy": "prior selection",
                    "visual_method": "prior method",
                    "pending": ["historical pending note"],
                },
            }
        },
    }
    packet = {
        "work": "amalia",
        "slug": "fixture_amalia_score",
        "source_pdf_sha256": "pdf",
        "reference_abc_sha256": "abc",
        "ticks_per_quarter": 384,
        "note_fields": list(NOTE_FIELDS),
        "selection": [
            {
                "edition_bar": i,
                "source_bar": i,
                "source_location": {"page": 1, "system": 1, "measure": i},
                "reference_notes": [note],
                "note_events_sha256": note_events_sha256({"notes": [note]}),
                "expected_noteheads": 1,
            }
            for i in range(2, 12)
        ],
    }
    bundle = {"reader_assignments": {"amalia": ["reader_a", "reader_b"]}, "files": {}}
    save(bundle, "works/amalia/audit_packet.json", packet)
    packet_sha = bundle["files"]["works/amalia/audit_packet.json"]["sha256"]
    plan = {
        "reference_manifest_sha256": "reference",
        "direct_api_calls": 0,
        "model": "gpt-6-astra",
        "reasoning_effort": "xhigh",
        "works": [
            {
                "key": "amalia",
                "slug": packet["slug"],
                "packet_sha256": packet_sha,
                "selected_edition_bars": list(range(2, 12)),
                "excluded_bars": [],
                "existing_audited_bars": 1,
            }
        ],
    }
    save(bundle, "audit_plan.json", plan)
    for name, role in [("author", "author"), ("review", "independent_reviewer")]:
        report = {
            "work": "amalia",
            "role": role,
            "packet_sha256": packet_sha,
            "source_pdf_sha256": "pdf",
            "reference_abc_sha256": "abc",
            "candidate_outputs_seen": False,
            "new_human_approval": False,
            "model": "gpt-6-astra",
            "reasoning_effort": "xhigh",
            "bars": [
                {
                    "edition_bar": r["edition_bar"],
                    "source_bar": r["source_bar"],
                    "source_location": r["source_location"],
                    "verdict": "supported",
                    "reference_note_events_sha256": r["note_events_sha256"],
                    "observed_notes": [note],
                    "evidence": "Clear source head and stem.",
                    "context_checked": "Prior key and adjacent bars",
                    "issues": [],
                }
                for r in packet["selection"]
            ],
        }
        save(bundle, f"works/amalia/audit.{name}.json", report)
    return prior, bundle


def test_expansion_retains_old_masks_and_human_decisions(audit):
    prior, bundle = audit
    snapshot = copy.deepcopy(prior)
    new, summary = expand_inventory(prior, bundle)
    assert prior == snapshot
    assert new["works"]["amalia"]["decisions"] == prior["works"]["amalia"]["decisions"]
    assert new["works"]["amalia"]["source_audit"]["measures"][0] == {
        "edition_bar": 1,
        "kept": "byte-equivalent record",
    }
    assert summary["works"]["amalia"]["total_bars"] == 11


@pytest.mark.parametrize(
    "mutation",
    [
        "ambiguous",
        "contradicted",
        "pitch",
        "tie",
        "duplicate",
        "missing_field",
        "null_tie",
        "float_pitch",
        "location",
    ],
)
def test_disagreement_excludes_whole_bar_without_replacement(audit, mutation):
    prior, bundle = audit

    def change(report):
        row = report["bars"][0]
        if mutation in {"ambiguous", "contradicted"}:
            row["verdict"] = mutation
        elif mutation == "pitch":
            row["observed_notes"][0]["pitch_midi"] = 61
        elif mutation == "tie":
            row["observed_notes"][0]["tie_start"] = True
        elif mutation == "duplicate":
            row["observed_notes"].append(dict(row["observed_notes"][0]))
        elif mutation == "null_tie":
            row["observed_notes"][0]["tie_end"] = None
        elif mutation == "float_pitch":
            row["observed_notes"][0]["pitch_midi"] = 60.0
        elif mutation == "missing_field":
            del row["observed_notes"][0]["tie_end"]
        else:
            row["source_location"]["measure"] = 1

    edit(bundle, "works/amalia/audit.review.json", change)
    new, summary = expand_inventory(prior, bundle)
    rows = new["works"]["amalia"]["source_audit"]["measures"]
    assert [r["edition_bar"] for r in rows] == [1, *range(3, 12)]
    assert summary["works"]["amalia"]["rejected"][0]["edition_bar"] == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("candidate_outputs_seen", True),
        ("new_human_approval", True),
        ("packet_sha256", "stale"),
        ("role", "author"),
    ],
)
def test_wrong_or_contaminated_reader_identity_rejected(audit, field, value):
    prior, bundle = audit
    edit(bundle, "works/amalia/audit.review.json", lambda r: r.update({field: value}))
    with pytest.raises(ValueError, match="reader identity"):
        expand_inventory(prior, bundle)


def test_tampered_evidence_and_same_reader_rejected(audit):
    prior, bundle = audit
    changed = copy.deepcopy(bundle)
    changed["files"]["audit_plan.json"]["content"] += " "
    with pytest.raises(ValueError, match="hash mismatch"):
        expand_inventory(prior, changed)
    bundle["reader_assignments"]["amalia"] = ["same", "same"]
    with pytest.raises(ValueError, match="distinct"):
        expand_inventory(prior, bundle)


def test_regrading_refuses_silent_historical_metric_drift():
    from scripts.regrade_source_audit import METRICS

    prior = {name: 0.5 for name in (*METRICS, "source_note_f1")}
    result = {"edition": {"metrics": dict(prior)}, "v3": {"metrics": {"source_note_f1": 0.5}}}
    assert_reproduced(result, prior)
    result["v3"]["metrics"]["source_note_f1"] = 0.6
    with pytest.raises(ValueError, match="does not reproduce"):
        assert_reproduced(result, prior)


def test_failed_label_creation_rolls_back_only_new_files(tmp_path, monkeypatch):
    from scripts import regrade_source_audit as script

    evidence, inventory = tmp_path / "evidence.json", tmp_path / "inventory.json"
    labels = tmp_path / "labels"
    labels.mkdir()
    sentinel = labels / "old-v3.json"
    sentinel.write_text("preserve")
    monkeypatch.setattr(script, "EVIDENCE_PATH", evidence)
    monkeypatch.setattr(script, "V4_INVENTORY_PATH", inventory)
    monkeypatch.setattr(script, "LABELS", labels)

    def fail(*args):
        assert evidence.exists() and inventory.exists()
        raise ValueError("invalid frozen reference")

    monkeypatch.setattr(script, "create_labels", fail)
    with pytest.raises(ValueError, match="invalid frozen reference"):
        script.publish_labels(b"{}", {})
    assert not evidence.exists() and not inventory.exists()
    assert list(labels.iterdir()) == [sentinel]
    assert sentinel.read_text() == "preserve"
