"""Back up the local review queue and export exact reviewed editions, without publishing.

The backup is restoreable as repository-relative files. The release is a private
edition preview, not a new source-ground-truth freeze or a rights clearance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score2abc.reference_freeze import (  # noqa: E402
    _canonical,
    _read_regular,
    _references,
    _source_records,
)
from score2abc.review import ReviewApp  # noqa: E402

EXCLUDED_PARTS = {
    "api",
    "private",
    "traces",
    "logs",
    "sessions",
    "__pycache__",
    "fontcache",
    "preview_out",
    "verification_out",
    "pipeline_check",
    "preview_desk",
    "new_preview_desk",
    "disposable_desk",
    "browser_qa",
    "node_modules",
    ".git",
}
CONTEXT_SUFFIXES = {
    ".abc",
    ".json",
    ".md",
    ".txt",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".musicxml",
    ".xml",
    ".csv",
    ".diff",
}
MAX_CONTEXT_BYTES = 500_000_000
MAX_CONTEXT_FILES = 5_000
RUMICHACA_ADJUDICATION = {
    "slug": "jaime-llanos_86_rumichaca_bambuco_emilio-murillo",
    # Exact reviewed ABC also used by the November 18 diagnostic's frozen reference.
    "edition_sha256": "c37aaa1e54aab10e8761bee876fabc61879e033ffa5dd05dfeef83e25cdd6b44",
    "diagnosis_date": "2026-11-18",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def safe_relative(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError("Expected a portable relative path")
    return path


def put(root: Path, name: str, data: bytes, files: dict) -> None:
    path = root / safe_relative(name)
    if name in files:
        if files[name]["sha256"] != digest(data):
            raise ValueError("Conflicting snapshot file")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    files[name] = {"sha256": digest(data), "size": len(data)}


def verify(root: Path) -> dict:
    root = _canonical(root)
    raw = _read_regular(root / "manifest.json")
    if _read_regular(root / "manifest.sha256").decode().strip() != digest(raw):
        raise ValueError("Manifest hash mismatch")
    manifest = json.loads(raw)
    if manifest.get("kind") not in {"review_queue_backup", "reviewed_edition_private_preview"}:
        raise ValueError("Unsupported snapshot kind")
    expected = set(manifest["files"]) | {"manifest.json", "manifest.sha256"}
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symlink in snapshot")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != expected:
        raise ValueError("Snapshot inventory mismatch")
    for name, record in manifest["files"].items():
        data = _read_regular(root / safe_relative(name))
        if {"sha256": digest(data), "size": len(data)} != record:
            raise ValueError(f"Snapshot hash mismatch: {name}")
    return manifest


def publish(stage: Path, destination: Path, manifest: dict) -> dict:
    (stage / "manifest.json").write_bytes(encoded(manifest))
    (stage / "manifest.sha256").write_text(digest((stage / "manifest.json").read_bytes()) + "\n")
    verify(stage)
    destination.mkdir(mode=0o700)  # Exclusive, including when another writer won the race.
    try:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                target = destination / path.relative_to(stage)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as handle:
                    handle.write(path.read_bytes())
        return verify(destination)
    except Exception:
        shutil.rmtree(destination)
        raise


def allowed_context(path: Path) -> bool:
    return (
        not any(part.lower() in EXCLUDED_PARTS for part in path.parts)
        and not path.name.startswith(".")
        and not any(
            word in path.stem.lower() for word in ("trace", "request_", "response_", "session")
        )
        and path.suffix.lower() in CONTEXT_SUFFIXES
    )


def backup(out_dir: Path, destination: Path, protected: Path | None = None) -> dict:
    out_dir, destination = _canonical(out_dir), _canonical(destination)
    repo = out_dir.parent
    if destination.exists():
        raise FileExistsError(destination)
    if not destination.is_relative_to(out_dir / "backups"):
        raise ValueError("Queue backups belong in a new out/backups namespace")
    destination.parent.mkdir(parents=True, exist_ok=True)
    originals, selected, works, omitted = {}, set(), [], set()

    def include(path: Path, expected_sha256: str | None = None) -> str:
        path = _canonical(path)
        if not path.is_relative_to(repo):
            raise ValueError("Backup inputs must be inside the repository")
        data = _read_regular(path)
        name = path.relative_to(repo).as_posix()
        current_hash = digest(data)
        if (expected_sha256 is not None and current_hash != expected_sha256) or (
            name in originals and originals[name] != current_hash
        ):
            raise ValueError("Source files changed while collecting backup inputs")
        originals[name] = current_hash
        selected.add(path)
        return name

    include(out_dir / "manifest.jsonl")
    app, source_records = ReviewApp(out_dir), _source_records(out_dir)
    review_paths = sorted(out_dir.glob("*/overrides/review.json"))
    release_notes = []
    for name in (
        "FORM_NOTATION.md",
        "EDITORIAL_TRANSCRIPTION_PROTOCOL.md",
        "ASTRA62_API_RUMICHACA_PILOT.md",
    ):
        note = repo / "docs" / name
        if note.is_file():
            release_notes.append(include(note))
    for path in review_paths:
        slug = path.parent.parent.name
        review_bytes = _read_regular(path)
        review_name = include(path, expected_sha256=digest(review_bytes))
        review = json.loads(review_bytes)
        history = app.history(slug)  # Validates saved revision hashes and predecessor links.
        current = [entry for entry in history if entry["current"]]
        if len(current) != 1 or current[0]["sha256"] != originals[review_name]:
            raise ValueError("Current review changed while reading its history")
        record = json.loads(source_records[slug])
        source = _canonical(out_dir / record["pdf_path"])
        if source.suffix.lower() != ".pdf":
            raise ValueError("Expected source PDF")
        source_name = include(source)
        core = []
        for name in ("overrides", "pages", "review_sources", "source.pdf", "metadata.json"):
            entry = path.parent.parent / name
            for item in sorted(entry.rglob("*")) if entry.is_dir() else [entry]:
                if item.is_symlink():
                    raise ValueError("Symlink in live review files")
                if item.is_file():
                    relative = item.relative_to(repo)
                    if allowed_context(relative):
                        core.append(include(item))
                    else:
                        omitted.add("excluded: " + relative.as_posix())
        contexts = set().union(*(_references(entry["review"], repo) for entry in history))
        context_names = []
        for context in sorted(contexts):
            if context.is_symlink():
                raise ValueError("Symlink in linked evidence")
            if context.is_relative_to(destination) or destination.is_relative_to(context):
                raise ValueError("Backup destination overlaps referenced evidence")
            if not context.exists():
                omitted.add("missing: " + context.relative_to(repo).as_posix())
                continue
            for item in sorted(context.rglob("*")) if context.is_dir() else [context]:
                if item.is_symlink():
                    raise ValueError("Symlink in linked evidence")
                if not item.is_file():
                    continue
                relative = item.relative_to(repo)
                if allowed_context(relative):
                    context_names.append(include(item))
                else:
                    omitted.add("excluded: " + relative.as_posix())
        works.append(
            {
                "slug": slug,
                "review_state": review["review_state"],
                "revision": review["revision"],
                "review_path": path.relative_to(repo).as_posix(),
                "source_path": source_name,
                "source_record": record,
                "core_files": sorted(set(core)),
                "context_files": sorted(set(context_names)),
                "history": [
                    {k: entry[k] for k in ("revision", "sha256", "current")} for entry in history
                ],
            }
        )
    context_names = set().union(*(set(work["context_files"]) for work in works))
    if (
        len(context_names) > MAX_CONTEXT_FILES
        or sum((repo / name).stat().st_size for name in context_names) > MAX_CONTEXT_BYTES
    ):
        raise ValueError("Linked evidence exceeds the explicit backup size limit")
    protected_data = json.loads(_read_regular(protected)) if protected else None
    if protected_data and any(
        originals.get(p) != sha for p, sha in protected_data["files"].items()
    ):
        raise ValueError("Protected review files changed or are absent from backup selection")
    with tempfile.TemporaryDirectory(prefix=".review-backup-", dir=destination.parent) as temp:
        stage, files = Path(temp), {}
        for path in sorted(selected):
            name = path.relative_to(repo).as_posix()
            put(stage, "payload/" + name, _read_regular(path), files)
        # Catch changes and newly saved history, not just mutations to known filenames.
        live_reviews = {
            p.relative_to(repo).as_posix(): digest(_read_regular(p))
            for p in out_dir.glob("*/overrides/**/*.json")
            if allowed_context(p.relative_to(repo))
        }
        snap_reviews = {
            p: sha for p, sha in originals.items() if "/overrides/" in p and p.endswith(".json")
        }
        if live_reviews != snap_reviews or any(
            digest(_read_regular(repo / p)) != sha for p, sha in originals.items()
        ):
            raise ValueError("Source files changed during backup; nothing published")
        manifest = {
            "schema_version": 1,
            "kind": "review_queue_backup",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files": files,
            "works": works,
            "original_sha256": originals,
            "protected_before": protected_data,
            "release_notes": release_notes,
            "originals_unchanged": True,
            "omitted_context": sorted(omitted),
            "scope": (
                "Current saved reviews, all available histories, desk inputs and explicitly "
                "linked proposal/source evidence; no experiment freezes or raw model traces."
            ),
            "off_machine_backup": False,
        }
        return publish(stage, destination, manifest)


def archive_backup(root: Path) -> Path:
    """Create a sibling portable archive, then restore every regular member and verify."""
    root = _canonical(root)
    verify(root)
    archive = root.with_suffix(".tar.gz")
    receipt = root.with_suffix(".receipt.json")
    if archive.exists() or receipt.exists():
        raise FileExistsError("Archive publication is create-once")
    with tarfile.open(archive, "x:gz") as output:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                output.add(path, arcname=path.relative_to(root), recursive=False)
    with tempfile.TemporaryDirectory(prefix=".restore-check-", dir=root.parent) as temporary:
        restored = Path(temporary)
        with tarfile.open(archive, "r:gz") as source:
            for member in source:
                relative = safe_relative(member.name)
                if not member.isfile():
                    raise ValueError("Only regular archive members are allowed")
                target = restored / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as handle:
                    handle.write(source.extractfile(member).read())
        manifest = verify(restored)
    receipt.write_bytes(
        encoded(
            {
                "archive": archive.name,
                "archive_sha256": digest(archive.read_bytes()),
                "archive_bytes": archive.stat().st_size,
                "file_count": len(manifest["files"]),
                "manifest_sha256": digest((root / "manifest.json").read_bytes()),
                "restored_and_hash_verified": True,
                "off_machine_backup": False,
            }
        )
    )
    return archive


def pending_classification(slug: str, abc: str) -> list[dict]:
    """Attach dated findings only to the exact edition that was investigated."""
    if (
        slug != RUMICHACA_ADJUDICATION["slug"]
        or digest(abc.encode()) != RUMICHACA_ADJUDICATION["edition_sha256"]
    ):
        return []
    return [
        {"kind": "source_vs_edition_chords", "written_bars": [1, 6, 30, 31, 32]},
        {
            "kind": "final_dc_route",
            "adopted_performed_bars": 132,
            "api_and_subscription_alternative_bars": 117,
        },
    ]


def export_reviewed(snapshot: Path, destination: Path) -> dict:
    snapshot, destination = _canonical(snapshot), _canonical(destination)
    saved = verify(snapshot)
    if saved["kind"] != "review_queue_backup":
        raise ValueError("Export requires a verified queue backup")
    if destination.exists():
        raise FileExistsError(destination)
    if destination.is_relative_to(snapshot) or snapshot.is_relative_to(destination):
        raise ValueError("Export destination overlaps the backup")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".review-export-", dir=destination.parent) as temp:
        stage, files, works = Path(temp), {}, []
        for name in saved.get("release_notes", []):
            put(stage, "evidence/" + name, (snapshot / "payload" / name).read_bytes(), files)
        for work in saved["works"]:
            if work["review_state"] != "reviewed":
                continue
            slug = work["slug"]
            review = json.loads((snapshot / "payload" / work["review_path"]).read_bytes())
            if (
                review.get("review_state") != "reviewed"
                or review.get("revision") != work["revision"]
            ):
                raise ValueError("Saved edition state/revision differs from backup metadata")
            root = f"works/{slug}"
            put(
                stage,
                root + "/review.json",
                (snapshot / "payload" / work["review_path"]).read_bytes(),
                files,
            )
            put(stage, root + "/edition.abc", review["abc"].encode(), files)
            put(
                stage,
                root + "/source.pdf",
                (snapshot / "payload" / work["source_path"]).read_bytes(),
                files,
            )
            put(
                stage, root + "/editorial_notes.txt", review.get("review_notes", "").encode(), files
            )
            history_files = []
            for name in work["core_files"]:
                if "/overrides/history/" in name:
                    target = root + "/history/" + Path(name).name
                    put(stage, target, (snapshot / "payload" / name).read_bytes(), files)
                    history_files.append(target)
            evidence = []
            for name in work["context_files"]:
                target = "evidence/" + name
                put(stage, target, (snapshot / "payload" / name).read_bytes(), files)
                evidence.append(target)
            pending = pending_classification(slug, review["abc"])
            metadata = {
                "slug": slug,
                "review_state": review["review_state"],
                "revision": review["revision"],
                "metadata": work["source_record"].get("metadata", {}),
                "edition_path": root + "/edition.abc",
                "review_path": root + "/review.json",
                "source_path": root + "/source.pdf",
                "source_sha256": saved["original_sha256"][work["source_path"]],
                "original_source_record": work["source_record"],
                "model_name_as_recorded": review.get("model_name"),
                "draft_source_as_recorded": review.get("draft_source"),
                "provenance_as_recorded": review.get("provenance"),
                "active_review_ms_as_recorded": review.get("active_review_ms"),
                "save_metadata_as_recorded": review.get("save_metadata"),
                "unresolved_as_recorded": review.get("unresolved"),
                "history": work["history"],
                "history_files": history_files,
                "evidence_files": evidence,
                "editorial_notes_path": root + "/editorial_notes.txt",
                "edition_role": "human_reviewed_adopted_edition_not_certified_source_ground_truth",
                "decision_classification": (
                    "Original prose and structured proposals retained without certifying "
                    "their claims or inferring approval per decision."
                ),
                "rights_status": "not_cleared_for_public_distribution",
                "pending_classification": pending,
                "pending_classification_binding": RUMICHACA_ADJUDICATION if pending else None,
                "pending_classification_basis": (
                    "evidence/docs/ASTRA62_API_RUMICHACA_PILOT.md; approval preserved; "
                    "source chords and form alternatives await adjudication."
                    if pending
                    else None
                ),
            }
            put(stage, root + "/metadata.json", encoded(metadata), files)
            works.append(metadata)
        manifest = {
            "schema_version": 1,
            "kind": "reviewed_edition_private_preview",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "release_id": destination.name,
            "visibility": "private_local_preview",
            "public_release_ready": False,
            "rights_status": "not_cleared_for_public_distribution",
            "files": files,
            "works": works,
            "reviewed_count": len(works),
            "first_publication_target_count": 10,
            "source_backup_manifest_sha256": digest((snapshot / "manifest.json").read_bytes()),
            "omitted_context": saved["omitted_context"],
            "scope": (
                "Exact adopted editions and available provenance; no new ground truth labels, "
                "musical approval, publication or model training."
            ),
        }
        verify(snapshot)
        return publish(stage, destination, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    save = commands.add_parser("backup")
    save.add_argument("out_dir", type=Path)
    save.add_argument("destination", type=Path)
    save.add_argument("--protected-manifest", type=Path)
    release = commands.add_parser("export")
    release.add_argument("snapshot", type=Path)
    release.add_argument("destination", type=Path)
    check = commands.add_parser("verify")
    check.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.command == "backup":
        result = backup(args.out_dir, args.destination, args.protected_manifest)
        archive = archive_backup(args.destination)
        print(
            json.dumps(
                {
                    "backup": str(args.destination),
                    "archive": str(archive),
                    "works": len(result["works"]),
                    "files": len(result["files"]),
                }
            )
        )
    else:
        result = (
            export_reviewed(args.snapshot, args.destination)
            if args.command == "export"
            else verify(args.destination)
        )
        print(
            json.dumps(
                {
                    "destination": str(args.destination),
                    "kind": result["kind"],
                    "works": len(result["works"]),
                    "files": len(result["files"]),
                }
            )
        )


if __name__ == "__main__":
    main()
