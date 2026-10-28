import hashlib
import json
from pathlib import Path

import pytest

from score2abc.reference_freeze import FreezeError, freeze_references, verify_freeze


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reseal(frozen: Path, manifest: dict) -> None:
    data = (json.dumps(manifest, ensure_ascii=False) + "\n").encode()
    (frozen / "manifest.json").write_bytes(data)
    (frozen / "manifest.sha256").write_text(_sha(data) + "\n")


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    out = tmp_path / "out"
    out.mkdir()
    lines = []
    for number in (6, 10, 12, 19, 22, 86):
        slug = f"jaime-llanos_{number}_test"
        work = out / slug
        overrides = work / "overrides"
        history = overrides / "history"
        history.mkdir(parents=True)
        pdf = work / "source.pdf"
        pdf.write_bytes(f"%PDF-1.4\n{number}".encode())
        base = "X:1\nT:Base\nM:?\nK:?\n"
        initial = {
            "abc": "X:1\nT:Initial\nK:C\nC|\n",
            "revision": 1,
            "review_state": "draft",
            "unresolved": ["source question"],
            "review_notes": "Keep this exact note.",
            "base_abc": base,
            "base_abc_sha256": _sha(base.encode()),
            "save_metadata": {"previous_review_sha256": None},
        }
        prior = json.dumps(initial, ensure_ascii=False).encode()
        (history / f"1-{_sha(prior)}.json").write_bytes(prior)
        current = {
            **initial,
            "abc": "X:1\nT:Reviewed\nK:C\nC D|\n",
            "revision": 2,
            "review_state": "reviewed",
            "unresolved": [],
            "review_notes": "Literal free text: source ≠ edition.\nDo not classify it.",
            "save_metadata": {"previous_review_sha256": _sha(prior)},
        }
        if number == 6:
            context = out / "collection" / "decision" / "note.txt"
            context.parent.mkdir(parents=True)
            context.write_text("source reading retained", encoding="utf-8")
            current["review_notes"] += "\nout/collection/decision/note.txt"
        (overrides / "review.json").write_text(json.dumps(current, ensure_ascii=False))
        lines.append(
            json.dumps(
                {
                    "slug": slug,
                    "pdf_path": f"{slug}/source.pdf",
                    "metadata": {
                        "title": f"Score {number}",
                        "composer": "Composer",
                        "rhythm": "Pasillo",
                    },
                }
            )
        )
    (out / "manifest.jsonl").write_text("\n".join(lines) + "\n")
    return out, tmp_path / "frozen"


def test_freeze_is_portable_and_create_once(tmp_path: Path) -> None:
    out, frozen = _fixture(tmp_path)
    manifest = freeze_references(out, frozen)
    assert len(manifest["works"]) == 6
    assert manifest["works"][0]["history_revisions"] == [1, 2]
    assert manifest["works"][0]["annotation_status"] == "pending"
    assert (
        frozen / "answers/jaime-llanos_6_test/context/out/collection/decision/note.txt"
    ).exists()
    assert list((frozen / "inputs").rglob("*.json")) == []
    assert verify_freeze(frozen) == manifest
    assert verify_freeze(frozen, out) == manifest
    with pytest.raises(FreezeError, match="already exists"):
        freeze_references(out, frozen)


def test_verify_detects_changed_missing_and_extra_artifacts(tmp_path: Path) -> None:
    out, frozen = _fixture(tmp_path)
    manifest = freeze_references(out, frozen)
    path = frozen / manifest["works"][0]["answer_review_path"]
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    with pytest.raises(FreezeError, match="content changed"):
        verify_freeze(frozen)
    path.write_bytes(original)
    path.unlink()
    with pytest.raises(FreezeError, match="file set changed"):
        verify_freeze(frozen)
    path.write_bytes(original)
    (frozen / "unexpected.txt").write_text("extra")
    with pytest.raises(FreezeError, match="file set changed"):
        verify_freeze(frozen)


def test_rejects_incomplete_review_and_invalid_history(tmp_path: Path) -> None:
    out, frozen = _fixture(tmp_path)
    review_path = out / "jaime-llanos_6_test/overrides/review.json"
    review = json.loads(review_path.read_text())
    review["review_state"] = "draft"
    review_path.write_text(json.dumps(review))
    with pytest.raises(FreezeError, match="completed edition"):
        freeze_references(out, frozen)
    assert not frozen.exists()
    review["review_state"] = "reviewed"
    review_path.write_text(json.dumps(review))
    archive = next((review_path.parent / "history").glob("*.json"))
    archive.write_bytes(archive.read_bytes() + b" ")
    with pytest.raises(FreezeError, match="Invalid review history"):
        freeze_references(out, frozen)
    assert not frozen.exists()


def test_detects_concurrent_review_change_and_cleans_temp(tmp_path: Path, monkeypatch) -> None:
    out, frozen = _fixture(tmp_path)
    import score2abc.reference_freeze as module

    original_put = module._put

    def mutate_after_copy(root, relative, source, files):
        original_put(root, relative, source, files)
        if relative == "answers/jaime-llanos_6_test/review.json":
            source.write_bytes(source.read_bytes() + b" ")

    monkeypatch.setattr(module, "_put", mutate_after_copy)
    with pytest.raises(FreezeError, match="Review (history )?changed while freezing"):
        freeze_references(out, frozen)
    assert not frozen.exists()
    assert not list(tmp_path.glob(".frozen.tmp-*"))


def test_verify_checks_embedded_source_identity_and_optional_live_pdf(tmp_path: Path) -> None:
    out, frozen = _fixture(tmp_path)
    freeze_references(out, frozen)
    pdf = out / "jaime-llanos_6_test/source.pdf"
    pdf.write_bytes(b"changed source")
    assert verify_freeze(frozen)
    with pytest.raises(FreezeError, match="Live source PDF differs"):
        verify_freeze(frozen, out)
    manifest_path = frozen / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["works"][0]["source_manifest_pdf_path"] = "other.pdf"
    _reseal(frozen, manifest)
    with pytest.raises(FreezeError, match="Source manifest identity changed"):
        verify_freeze(frozen)


def test_manifest_seal_and_six_work_status_are_verified(tmp_path: Path) -> None:
    out, frozen = _fixture(tmp_path)
    freeze_references(out, frozen)
    manifest_path = frozen / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(FreezeError, match="manifest seal changed"):
        verify_freeze(frozen)
    manifest["heldout"] = True
    _reseal(frozen, manifest)
    with pytest.raises(FreezeError, match="evidence status"):
        verify_freeze(frozen)
    manifest["heldout"] = False
    manifest["works"].pop()
    _reseal(frozen, manifest)
    with pytest.raises(FreezeError, match="exactly six"):
        verify_freeze(frozen)


def test_publication_never_replaces_concurrently_created_empty_directory(
    tmp_path: Path, monkeypatch
) -> None:
    out, frozen = _fixture(tmp_path)
    import score2abc.reference_freeze as module

    actual_publish = module._publish_no_replace

    def create_destination_then_publish(source, destination):
        destination.mkdir()
        actual_publish(source, destination)

    monkeypatch.setattr(module, "_publish_no_replace", create_destination_then_publish)
    with pytest.raises(FreezeError, match="Freeze destination already exists"):
        freeze_references(out, frozen)
    assert frozen.is_dir() and not list(frozen.iterdir())
    assert not list(tmp_path.glob(".frozen.tmp-*"))


def test_symlinked_override_is_rejected(tmp_path: Path) -> None:
    out, frozen = _fixture(tmp_path)
    overrides = out / "jaime-llanos_6_test" / "overrides"
    (overrides / "linked_dir").symlink_to(out / "collection", target_is_directory=True)
    with pytest.raises(FreezeError, match="Symlink in review overrides"):
        freeze_references(out, frozen)
    assert not frozen.exists()


def test_freeze_cannot_write_into_live_work_or_evidence(tmp_path: Path) -> None:
    out, _ = _fixture(tmp_path)
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    for destination in [
        out / "jaime-llanos_6_test" / "overrides" / "snapshot",
        out / "collection" / "snapshot",
    ]:
        with pytest.raises(FreezeError, match="outside live work and collection"):
            freeze_references(out, destination)
        assert not destination.exists()
    assert before == {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}


def test_nested_context_symlink_is_reported_as_omitted(tmp_path: Path) -> None:
    out, frozen = _fixture(tmp_path)
    context = out / "collection" / "decision"
    (context / "linked_dir").symlink_to(out, target_is_directory=True)
    review_path = out / "jaime-llanos_6_test/overrides/review.json"
    review = json.loads(review_path.read_text())
    review["review_notes"] = "Decision directory: out/collection/decision/"
    review_path.write_text(json.dumps(review))
    manifest = freeze_references(out, frozen)
    omissions = manifest["works"][0]["omitted_context_references"]
    assert "unsafe: out/collection/decision/linked_dir" in omissions
