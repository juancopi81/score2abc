"""Evidence and immutability checks for reviewed-score label generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from score2abc.eval_labels import (
    INVENTORY_PATH,
    V1_INVENTORY_PATH,
    build_labels,
    create_labels,
    verify_labels,
)
from score2abc.score_eval_abc import normalize_abc

FROZEN = Path(__file__).resolve().parents[1] / "out/eval/references/reviewed_six_20261028_v1"


def _inventory_copy(tmp_path: Path) -> tuple[Path, dict]:
    inventory = json.loads(INVENTORY_PATH.read_text())
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inventory))
    return path, inventory


def test_create_verify_and_no_frozen_mutation(tmp_path: Path) -> None:
    original = (FROZEN / "manifest.json").read_bytes()
    destination = tmp_path / "labels"
    created = create_labels(destination, FROZEN)
    assert len(created["works"]) == 6
    assert created["label_set"] == "reviewed_six_v2"
    assert (destination / "inventory.json").read_bytes() == INVENTORY_PATH.read_bytes()
    assert verify_labels(destination, FROZEN) == created
    assert all(not work["source_metric_eligible_measures"] for work in created["works"])
    assert (FROZEN / "manifest.json").read_bytes() == original
    with pytest.raises(FileExistsError):
        create_labels(destination, FROZEN)


def test_stale_reference_hash_rejected_before_output(tmp_path: Path) -> None:
    path, inventory = _inventory_copy(tmp_path)
    inventory["reference_manifest_sha256"] = "0" * 64
    path.write_text(json.dumps(inventory))
    destination = tmp_path / "labels"
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        create_labels(destination, FROZEN, path)
    assert not destination.exists()


def test_invalid_location_and_missing_evidence_quote_rejected(tmp_path: Path) -> None:
    path, inventory = _inventory_copy(tmp_path)
    inventory["works"]["aviador"]["decisions"][2]["edition_location"] = {"bar": 999}
    path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="invalid bar location"):
        build_labels(FROZEN, path)
    inventory["works"]["aviador"]["decisions"][2]["edition_location"] = {"bar": 29}
    inventory["works"]["aviador"]["decisions"][2]["evidence"][0]["quote"] = "not in note"
    path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="evidence quote absent"):
        build_labels(FROZEN, path)


def test_sidecar_change_rejected_and_exclusions_are_dimension_specific(tmp_path: Path) -> None:
    destination = tmp_path / "labels"
    created = create_labels(destination, FROZEN)
    aviador = next(work for work in created["works"] if work["title"] == "Aviador")
    atardecer = next(work for work in created["works"] if work["title"] == "Atardecer")
    chord_only = next(
        x for x in aviador["source_metric_exclusions"] if x.get("bar_range") == [50, 64]
    )
    assert chord_only["dimensions"] == ["chords"]
    arpeggio = next(
        x for x in atardecer["source_metric_exclusions"] if x.get("decision_ids") == ["AT-02"]
    )
    assert arpeggio["dimensions"] == ["notes", "chords"]
    assert aviador["source_to_edition_mapping"]["source_to_edition_bar"]["44"] == 46
    (destination / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_labels(destination, FROZEN)


def test_v1_sidecar_remains_verifiable_and_v2_inventory_copy_is_pinned(tmp_path: Path) -> None:
    existing_v1 = FROZEN.parents[1] / "labels/reviewed_six_v1"
    assert verify_labels(existing_v1, FROZEN)["label_set"] == "reviewed_six_v1"
    assert build_labels(FROZEN, V1_INVENTORY_PATH)["label_set"] == "reviewed_six_v1"
    destination = tmp_path / "labels"
    create_labels(destination, FROZEN)
    (destination / "inventory.json").write_text("{}")
    with pytest.raises(ValueError, match="sidecar inventory checksum mismatch"):
        verify_labels(destination, FROZEN)


def test_unregistered_inventory_and_sidecar_symlink_rejected(tmp_path: Path) -> None:
    inventory_path, inventory = _inventory_copy(tmp_path)
    inventory["works"]["carrizal"]["decisions"][0]["adopted_choice"] += " "
    inventory_path.write_text(json.dumps(inventory))
    destination = tmp_path / "labels"
    with pytest.raises(ValueError, match="unregistered label inventory"):
        create_labels(destination, FROZEN, inventory_path)
    assert not destination.exists()
    create_labels(destination, FROZEN)
    (destination / "inventory.json").unlink()
    (destination / "inventory.json").symlink_to(INVENTORY_PATH)
    with pytest.raises(ValueError, match="contains a symlink"):
        verify_labels(destination, FROZEN)


def test_aviador_human_revisions_supersede_model_prose() -> None:
    manifest = build_labels(FROZEN)
    aviador = next(work for work in manifest["works"] if work["title"] == "Aviador")
    decisions = {item["id"]: item for item in aviador["decisions"]}
    bar45 = decisions["AV-04"]
    trio = decisions["AV-05"]
    assert "F#5–D5" in bar45["adopted_choice"]
    assert "bar 49 has no chord" in trio["adopted_choice"]
    assert all(x["status"] == "supported_by_current_abc_and_history" for x in (bar45, trio))
    assert all(x["superseded_by_human_revisions"] == [16, 17, 18] for x in (bar45, trio))
    assert any("/history/16-" in source["path"] for source in bar45["evidence"])
    assert any("/history/16-" in source["path"] for source in trio["evidence"])
    frozen_manifest = json.loads((FROZEN / "manifest.json").read_text())
    frozen_work = next(work for work in frozen_manifest["works"] if work["title"] == "Aviador")
    score = normalize_abc((FROZEN / frozen_work["answer_abc_path"]).read_text())
    bar45_events = score["measures"][44]
    assert sorted(note["pitch_midi"] for note in bar45_events["notes"]) == [74, 74, 78, 80]
    assert [chord["symbol"] for chord in bar45_events["chords"]] == ["D", "E7"]
    assert score["measures"][48]["chords"] == []
    assert sum(len(bar["chords"]) for bar in score["measures"][49:64]) == 15
