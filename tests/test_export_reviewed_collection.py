import json
from pathlib import Path

import pytest
from test_reference_freeze import _fixture

from scripts import export_reviewed_collection as export


def fixture(tmp_path):
    out, _ = _fixture(tmp_path, numbers=(6, 86))
    draft = out / "jaime-llanos_6_test/overrides/review.json"
    data = json.loads(draft.read_text())
    data.update(review_state="draft", review_notes="My unfinished WIP", unresolved=["Not done"])
    draft.write_bytes(export.encoded(data))
    reviewed = out / "jaime-llanos_86_test/overrides/review.json"
    data = json.loads(reviewed.read_text())
    data.update(
        provenance={"model": "gpt-6-astra", "human_review_required": True},
        review_notes="Approved edition; original source reading stays distinct.\n"
        "out/collection/evidence",
    )
    reviewed.write_bytes(export.encoded(data))
    evidence = out / "collection/evidence"
    evidence.mkdir(parents=True)
    (evidence / "source_reading.abc").write_text("X:1\nK:C\nC|\n")
    (evidence / "proposal.abc").write_text("X:1\nK:C\nD|\n")
    (evidence / "decisions.json").write_text('[{"approval": false, "reason": "model proposal"}]')
    (evidence / "private").mkdir()
    (evidence / "private/response.json").write_text("must not be read")
    (evidence / "response_001.json").write_text("must not be read")
    (evidence / ".env").write_text("must not be read")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "ASTRA62_API_RUMICHACA_PILOT.md").write_text("Pending classification; retain approval.")
    protected = tmp_path / "protected.json"
    protected.write_bytes(
        export.encoded(
            {
                "files": {
                    path.relative_to(tmp_path).as_posix(): export.digest(path.read_bytes())
                    for path in out.glob("*/overrides/**/*.json")
                }
            }
        )
    )
    return out, out / "backups/checkpoint_v1", protected


def test_backup_restores_entire_queue_and_exports_only_exact_reviewed_editions(tmp_path):
    out, backup, protected = fixture(tmp_path)
    before = json.loads(protected.read_text())["files"]
    saved = export.backup(out, backup, protected)
    assert len(saved["works"]) == 2
    assert saved["originals_unchanged"] is True
    archive = export.archive_backup(backup)
    receipt = json.loads(backup.with_suffix(".receipt.json").read_text())
    assert receipt["restored_and_hash_verified"] is True
    assert receipt["archive_sha256"] == export.digest(archive.read_bytes())
    assert receipt["off_machine_backup"] is False
    destination = out / "collection/releases/reviewed_v1"
    released = export.export_reviewed(backup, destination)
    assert released["reviewed_count"] == 1
    assert released["first_publication_target_count"] == 10
    assert released["public_release_ready"] is False
    assert released["rights_status"] == "not_cleared_for_public_distribution"
    work = released["works"][0]
    original = out / "jaime-llanos_86_test/overrides/review.json"
    assert (destination / work["review_path"]).read_bytes() == original.read_bytes()
    review = json.loads(original.read_text())
    assert (destination / work["edition_path"]).read_bytes() == review["abc"].encode()
    assert (destination / work["editorial_notes_path"]).read_text() == review["review_notes"]
    assert work["provenance_as_recorded"] == review["provenance"]
    assert work["unresolved_as_recorded"] == []
    assert work["review_state"] == "reviewed" and work["pending_classification"] == []
    assert (destination / "evidence/docs/ASTRA62_API_RUMICHACA_PILOT.md").is_file()
    assert len(work["history"]) == 2 and len(work["history_files"]) == 1
    assert any(path.endswith("source_reading.abc") for path in work["evidence_files"])
    assert any(path.endswith("proposal.abc") for path in work["evidence_files"])
    assert export.verify(destination) == released
    assert all(export.digest((tmp_path / name).read_bytes()) == sha for name, sha in before.items())
    assert all(
        not Path(name).is_absolute() and ".." not in Path(name).parts for name in saved["files"]
    )


def test_private_traces_and_env_are_never_opened(tmp_path, monkeypatch):
    out, backup, protected = fixture(tmp_path)
    for relative in ("overrides/private/response.json", "review_sources/api/response_001.json"):
        path = out / "jaime-llanos_86_test" / relative
        path.parent.mkdir(parents=True)
        path.write_text("must not be read")
    original = export._read_regular

    def guarded(path):
        assert "private" not in path.relative_to(tmp_path).parts
        assert path.name not in {"response_001.json", ".env"}
        return original(path)

    monkeypatch.setattr(export, "_read_regular", guarded)
    saved = export.backup(out, backup, protected)
    assert any("private/response.json" in name for name in saved["omitted_context"])
    assert not any("private" in name or name.endswith(".env") for name in saved["files"])


def test_review_state_change_between_first_read_and_capture_cannot_export_a_draft(
    tmp_path, monkeypatch
):
    out, backup, _ = fixture(tmp_path)
    read = export._read_regular
    review_path = out / "jaime-llanos_86_test/overrides/review.json"
    changed = False

    def save_draft_after_read(path):
        nonlocal changed
        raw = read(path)
        if path == review_path and not changed:
            changed = True
            history = path.parent / "history"
            (history / f"2-{export.digest(raw)}.json").write_bytes(raw)
            payload = json.loads(raw)
            payload.update(
                review_state="draft",
                revision=3,
                save_metadata={"previous_review_sha256": export.digest(raw)},
            )
            path.write_bytes(export.encoded(payload))
        return raw

    monkeypatch.setattr(export, "_read_regular", save_draft_after_read)
    with pytest.raises(ValueError, match="changed while collecting"):
        export.backup(out, backup)
    assert not backup.exists()


def test_protected_before_mismatch_blocks_backup(tmp_path):
    out, backup, protected = fixture(tmp_path)
    review = out / "jaime-llanos_6_test/overrides/review.json"
    review.write_text(review.read_text() + "\n")
    with pytest.raises(ValueError, match="Protected review"):
        export.backup(out, backup, protected)
    assert not backup.exists()


def test_manifest_change_after_parsing_cannot_publish_mixed_source_records(tmp_path, monkeypatch):
    out, snapshot, _ = fixture(tmp_path)
    parse = export._source_records

    def change_manifest(path):
        records = parse(path)
        manifest = path / "manifest.jsonl"
        lines = [json.loads(line) for line in manifest.read_text().splitlines()]
        lines[0]["metadata"]["title"] = "New title saved concurrently"
        manifest.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
        return records

    monkeypatch.setattr(export, "_source_records", change_manifest)
    with pytest.raises(ValueError, match="changed during backup"):
        export.backup(out, snapshot)
    assert not snapshot.exists()


def test_change_during_copy_cannot_publish_mixed_snapshot(tmp_path, monkeypatch):
    out, backup, protected = fixture(tmp_path)
    put = export.put
    changed = False

    def mutate(*args):
        nonlocal changed
        put(*args)
        if not changed:
            changed = True
            review = out / "jaime-llanos_6_test/overrides/review.json"
            review.write_text(review.read_text() + "\n")

    monkeypatch.setattr(export, "put", mutate)
    with pytest.raises(ValueError, match="changed during backup"):
        export.backup(out, backup, protected)
    assert not backup.exists()


def test_create_once_and_tampered_backup_fail_closed(tmp_path):
    out, backup, protected = fixture(tmp_path)
    saved = export.backup(out, backup, protected)
    with pytest.raises(FileExistsError):
        export.backup(out, backup, protected)
    target = backup / "payload" / saved["works"][0]["review_path"]
    target.write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        export.export_reviewed(backup, out / "collection/releases/new")


def test_symlink_input_is_rejected_before_copy(tmp_path):
    out, backup, protected = fixture(tmp_path)
    source = out / "jaime-llanos_86_test/source.pdf"
    source.unlink()
    source.symlink_to(out / "jaime-llanos_6_test/source.pdf")
    with pytest.raises(ValueError, match="Symlink"):
        export.backup(out, backup, protected)
    assert not backup.exists()


def test_changed_rumichaca_edition_does_not_inherit_stale_adjudication(tmp_path, monkeypatch):
    out, snapshot, protected = fixture(tmp_path)
    slug = "jaime-llanos_86_test"
    review_path = out / slug / "overrides/review.json"
    original = json.loads(review_path.read_text())
    monkeypatch.setattr(
        export,
        "RUMICHACA_ADJUDICATION",
        {
            "slug": slug,
            "edition_sha256": export.digest(original["abc"].encode()),
            "diagnosis_date": "2026-11-18",
        },
    )
    assert export.pending_classification(slug, original["abc"])
    assert export.pending_classification("jaime-llanos_86_another_score", original["abc"]) == []
    original_bytes = review_path.read_bytes()
    history = review_path.parent / "history"
    (history / f"2-{export.digest(original_bytes)}.json").write_bytes(original_bytes)
    changed = {
        **original,
        "abc": original["abc"] + "E|\n",
        "revision": 3,
        "save_metadata": {"previous_review_sha256": export.digest(original_bytes)},
    }
    review_path.write_bytes(export.encoded(changed))
    export.backup(out, snapshot)
    result = export.export_reviewed(snapshot, out / "collection/releases/changed")
    work = result["works"][0]
    assert work["revision"] == 3 and work["review_state"] == "reviewed"
    assert work["pending_classification"] == []
    assert work["pending_classification_binding"] is None
    assert work["pending_classification_basis"] is None


@pytest.mark.parametrize("name", ["../outside", "/absolute", "a/../b", "a//b"])
def test_nonportable_restore_paths_are_rejected(name):
    with pytest.raises(ValueError, match="relative path"):
        export.safe_relative(name)
