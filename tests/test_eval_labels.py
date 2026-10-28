"""Hermetic checks of frozen evidence labels; local-score check is optional."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from score2abc import eval_labels
from score2abc.eval_labels import build_labels, create_labels, verify_labels

REAL_FROZEN = Path(__file__).resolve().parents[1] / "out/eval/references/reviewed_six_20261028_v1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()


@pytest.fixture
def sealed_six(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build six minimal reviews with genuine manifest, file and quote checksums."""
    root = tmp_path / "frozen"
    root.mkdir()
    files, works, inventories = {}, [], {}

    def write(relative: str, data: bytes) -> str:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files[relative] = {"sha256": _sha(data), "size": len(data)}
        return files[relative]["sha256"]

    for name in eval_labels.WORK_LIMITS:
        slug = f"fixture_{name}_score"
        abc = f"X:1\nT:{name}\nM:3/4\nL:1/4\nK:C\nC D E |]\n"
        if name == "aviador":
            abc += "% Current bar 45 has D5; first Trio bar has no chord.\n"
        review_path, abc_path = f"answers/{slug}/review.json", f"answers/{slug}/review.abc"
        review = {
            "abc": abc,
            "revision": 2,
            "review_state": "reviewed",
            "review_notes": "Human confirmed the visible note.",
        }
        review_sha = write(review_path, _json(review))
        abc_sha = write(abc_path, abc.encode())
        works.append(
            {
                "slug": slug,
                "title": name.title(),
                "revision": 2,
                "review_state": "reviewed",
                "answer_review_path": review_path,
                "answer_abc_path": abc_path,
                "review_sha256": review_sha,
                "abc_sha256": abc_sha,
            }
        )
        decisions = [
            {
                "id": f"{name.upper()}-01",
                "category": "source_confirmation",
                "source_location": (
                    {"bar": 1} if name == "aviador" else {"system": 1, "measure": 1}
                ),
                "edition_location": (
                    {"bar": 1} if name == "aviador" else {"system": 1, "measure": 1}
                ),
                "adopted_choice": "Visible note retained.",
                "status": "supported_by_review_notes",
                "evidence_origin": "pdf",
                "evidence": [{"kind": "review", "quote": "Human confirmed the visible note."}],
            }
        ]
        exclusions = []
        if name == "atardecer":
            decisions.append(
                {
                    "id": "AT-02",
                    "category": "editorial_reconstruction",
                    "source_location": None,
                    "edition_location": {"bar": 4},
                    "adopted_choice": "Editorial arpeggio and chord.",
                    "status": "supported_by_review_notes",
                    "evidence_origin": "human_musical_judgment",
                    "evidence": [{"kind": "review", "quote": "confirmed the visible note"}],
                }
            )
            exclusions.append(
                {
                    "scope": "decision",
                    "decision_ids": ["AT-02"],
                    "dimensions": ["notes", "chords"],
                    "reason": "Editorial choice.",
                }
            )
        if name == "aviador":
            write(f"answers/{slug}/overrides/history/16-human.json", _json({"abc": "D5 revision"}))
            packet_path = (
                f"answers/{slug}/context/out/collection/"
                "aviador_B_cadence_20261028_v1/decisions.json"
            )
            edition_bars = [n for n in range(1, 65) if n not in (29, 45)]
            write(
                packet_path,
                _json(
                    {"source_to_edition_bar": {str(n): edition_bars[n - 1] for n in range(1, 63)}}
                ),
            )
            decisions.extend(
                [
                    {
                        "id": "AV-04",
                        "category": "editorial_composition",
                        "source_location": None,
                        "edition_location": {"bar": 45},
                        "adopted_choice": "D5 in the current edition.",
                        "status": "supported_by_current_abc_and_history",
                        "evidence_origin": "human_musical_judgment",
                        "evidence": [
                            {"kind": "history", "revision": 16, "quote": "D5 revision"},
                            {"kind": "abc", "quote": "Current bar 45 has D5"},
                        ],
                    },
                    {
                        "id": "AV-05",
                        "category": "editorial_harmonization",
                        "source_location": {"bar_range": [47, 62]},
                        "edition_location": {"bar_range": [49, 64]},
                        "adopted_choice": "First Trio bar has no chord.",
                        "status": "supported_by_current_abc_and_history",
                        "evidence_origin": "human_musical_judgment",
                        "evidence": [
                            {"kind": "abc", "quote": "first Trio bar has no chord"},
                            {
                                "kind": "packet",
                                "packet": "aviador_B_cadence_20261028_v1",
                                "quote": "source_to_edition_bar",
                            },
                        ],
                    },
                ]
            )
            exclusions = [
                {
                    "scope": "chord",
                    "bar_range": [50, 64],
                    "dimensions": ["chords"],
                    "reason": "Editorial harmony.",
                }
            ]
        inventories[name] = {"decisions": decisions, "exclusions": exclusions}

    manifest_bytes = _json({"schema_version": 1, "works": works, "files": files})
    (root / "manifest.json").write_bytes(manifest_bytes)
    (root / "manifest.sha256").write_text(_sha(manifest_bytes) + "\n")
    inventory = {
        "schema_version": 1,
        "label_set": "reviewed_six_v2",
        "reference_name": "hermetic_six",
        "reference_manifest_sha256": _sha(manifest_bytes),
        "works": inventories,
    }
    inventory_path = tmp_path / "inventory_v2.json"
    inventory_path.write_bytes(_json(inventory))
    monkeypatch.setitem(eval_labels.KNOWN_INVENTORIES, "reviewed_six_v2", inventory_path)
    return root, inventory_path, inventory


def test_create_verify_and_no_frozen_mutation(sealed_six, tmp_path: Path) -> None:
    root, inventory_path, _ = sealed_six
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    destination = tmp_path / "labels"
    labels = create_labels(destination, root, inventory_path)
    assert labels["label_set"] == "reviewed_six_v2"
    assert len(labels["works"]) == 6
    assert (destination / "inventory.json").read_bytes() == inventory_path.read_bytes()
    assert verify_labels(destination, root) == labels
    assert all(not work["source_metric_eligible_measures"] for work in labels["works"])
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
    with pytest.raises(FileExistsError):
        create_labels(destination, root, inventory_path)


def test_stale_manifest_and_changed_abc_rejected(sealed_six, tmp_path: Path) -> None:
    root, inventory_path, inventory = sealed_six
    inventory["reference_manifest_sha256"] = "0" * 64
    altered = tmp_path / "altered.json"
    altered.write_bytes(_json(inventory))
    destination = tmp_path / "labels"
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        create_labels(destination, root, altered)
    assert not destination.exists()
    (root / "answers/fixture_amalia_score/review.abc").write_text("changed")
    with pytest.raises(ValueError, match="frozen evidence hash mismatch"):
        build_labels(root, inventory_path)


def test_bad_location_and_quote_rejected(sealed_six, tmp_path: Path) -> None:
    root, _, inventory = sealed_six
    changed = tmp_path / "changed.json"
    target = inventory["works"]["aviador"]["decisions"][1]
    target["edition_location"] = {"bar": 999}
    changed.write_bytes(_json(inventory))
    with pytest.raises(ValueError, match="invalid bar location"):
        build_labels(root, changed)
    target["edition_location"] = {"bar": 45}
    target["evidence"][0]["quote"] = "absent from history"
    changed.write_bytes(_json(inventory))
    with pytest.raises(ValueError, match="evidence quote absent"):
        build_labels(root, changed)


def test_review_revision_mismatch_rejected_even_with_new_seal(sealed_six, tmp_path: Path) -> None:
    root, _, inventory = sealed_six
    relative = "answers/fixture_amalia_score/review.json"
    review_path = root / relative
    review = json.loads(review_path.read_text())
    review["revision"] = 3
    new_review = _json(review)
    review_path.write_bytes(new_review)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files"][relative]["sha256"] = _sha(new_review)
    manifest["works"][0]["review_sha256"] = _sha(new_review)
    new_manifest = _json(manifest)
    (root / "manifest.json").write_bytes(new_manifest)
    (root / "manifest.sha256").write_text(_sha(new_manifest) + "\n")
    inventory["reference_manifest_sha256"] = _sha(new_manifest)
    changed = tmp_path / "changed.json"
    changed.write_bytes(_json(inventory))
    with pytest.raises(ValueError, match="frozen review metadata differs"):
        build_labels(root, changed)


def test_mapping_exclusions_and_historical_evidence_are_pinned(sealed_six) -> None:
    root, inventory_path, _ = sealed_six
    labels = build_labels(root, inventory_path)
    aviador = next(work for work in labels["works"] if work["title"] == "Aviador")
    atardecer = next(work for work in labels["works"] if work["title"] == "Atardecer")
    assert aviador["source_to_edition_mapping"]["source_to_edition_bar"]["44"] == 46
    assert aviador["source_metric_exclusions"][0]["dimensions"] == ["chords"]
    assert atardecer["source_metric_exclusions"][0]["dimensions"] == ["notes", "chords"]
    decisions = {item["id"]: item for item in aviador["decisions"]}
    assert any("/history/16-" in e["path"] for e in decisions["AV-04"]["evidence"])
    assert any(e["path"].endswith("review.abc") for e in decisions["AV-05"]["evidence"])
    assert all(e["sha256"] for item in decisions.values() for e in item["evidence"])


def test_sidecar_manifest_inventory_and_symlink_tampering_rejected(sealed_six, tmp_path) -> None:
    root, inventory_path, _ = sealed_six
    destination = tmp_path / "labels"
    create_labels(destination, root, inventory_path)
    (destination / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_labels(destination, root)
    other = tmp_path / "other"
    create_labels(other, root, inventory_path)
    (other / "inventory.json").write_text("{}")
    with pytest.raises(ValueError, match="sidecar inventory checksum mismatch"):
        verify_labels(other, root)
    third = tmp_path / "third"
    create_labels(third, root, inventory_path)
    (third / "inventory.json").unlink()
    (third / "inventory.json").symlink_to(inventory_path)
    with pytest.raises(ValueError, match="contains a symlink"):
        verify_labels(third, root)


def test_version_specific_inventory_and_unregistered_inventory(sealed_six, tmp_path, monkeypatch):
    root, inventory_path, inventory = sealed_six
    v2_root = tmp_path / "v2"
    create_labels(v2_root, root, inventory_path)
    v1 = dict(inventory)
    v1.pop("label_set")
    v1_path = tmp_path / "inventory_v1.json"
    v1_path.write_bytes(_json(v1))
    monkeypatch.setitem(eval_labels.KNOWN_INVENTORIES, "reviewed_six_v1", v1_path)
    v1_root = tmp_path / "v1"
    create_labels(v1_root, root, v1_path)
    assert verify_labels(v1_root, root)["label_set"] == "reviewed_six_v1"
    assert verify_labels(v2_root, root)["label_set"] == "reviewed_six_v2"
    inventory["works"]["carrizal"]["decisions"][0]["adopted_choice"] += " changed"
    altered = tmp_path / "altered.json"
    altered.write_bytes(_json(inventory))
    with pytest.raises(ValueError, match="unregistered label inventory"):
        create_labels(tmp_path / "unregistered", root, altered)


@pytest.mark.skipif(not REAL_FROZEN.exists(), reason="local frozen six-score fixture absent")
def test_real_aviador_history_when_local_fixture_available() -> None:
    from score2abc.score_eval_abc import normalize_abc

    labels = build_labels(REAL_FROZEN)
    aviador = next(work for work in labels["works"] if work["title"] == "Aviador")
    decisions = {item["id"]: item for item in aviador["decisions"]}
    assert "F#5–D5" in decisions["AV-04"]["adopted_choice"]
    assert "bar 49 has no chord" in decisions["AV-05"]["adopted_choice"]
    reference = json.loads((REAL_FROZEN / "manifest.json").read_text())
    work = next(work for work in reference["works"] if work["title"] == "Aviador")
    score = normalize_abc((REAL_FROZEN / work["answer_abc_path"]).read_text())
    if not score["valid"]:
        pytest.skip("optional local ABC renderer unavailable")
    bar45 = score["measures"][44]
    assert sorted(note["pitch_midi"] for note in bar45["notes"]) == [74, 74, 78, 80]
    assert [chord["symbol"] for chord in bar45["chords"]] == ["D", "E7"]
    assert score["measures"][48]["chords"] == []
    assert sum(len(bar["chords"]) for bar in score["measures"][49:64]) == 15
