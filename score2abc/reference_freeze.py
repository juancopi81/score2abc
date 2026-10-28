"""Create-once, portable snapshots of consumed human-reviewed editions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from score2abc.review import ReviewApp, ReviewError

REFERENCE_NUMBERS = (6, 10, 12, 19, 22, 86)
CONTEXT_FILE_LIMIT = 1_000
CONTEXT_BYTE_LIMIT = 150_000_000
CONTEXT_REF = re.compile(r"(?:out/collection|docs)/[A-Za-z0-9_./-]+")
EXCLUDED_CONTEXT_PARTS = {"preview_out", "verification_out", "pipeline_check", "__pycache__"}


class FreezeError(ValueError):
    """A snapshot cannot be safely created or verified."""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _history_id(entries: list[dict]) -> list[tuple[int, str]]:
    return [(entry["revision"], entry["sha256"]) for entry in entries]


def _read_regular(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise FreezeError(f"Missing or unsafe file: {path}")
    return path.read_bytes()


def _inside(path: Path, root: Path) -> bool:
    return path.is_relative_to(root) and not any(p.is_symlink() for p in (path, *path.parents))


def _canonical(path: Path) -> Path:
    """Normalize lexical traversal, then reject symlinks on the actual path used."""
    value = Path(os.path.abspath(path))
    if any(part.is_symlink() for part in (value, *value.parents)):
        raise FreezeError(f"Symlink in freeze path: {path}")
    return value


def _publish_no_replace(source: Path, destination: Path) -> None:
    """Reserve the name exclusively; publish the seal last after staged file copies."""
    try:
        destination.mkdir()
    except FileExistsError as exc:
        raise FreezeError(f"Freeze destination already exists: {destination}") from exc
    try:
        paths = sorted(source.rglob("*"))
        for path in paths:
            if path.is_dir():
                continue
            relative = path.relative_to(source)
            if relative.as_posix() == "manifest.sha256":
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(path.read_bytes())
        (destination / "manifest.sha256").write_bytes((source / "manifest.sha256").read_bytes())
        verify_freeze(destination)
    except Exception:
        shutil.rmtree(destination)
        raise


def _source_records(out_dir: Path) -> dict[str, str]:
    lines = _read_regular(out_dir / "manifest.jsonl").decode("utf-8").splitlines()
    records = {}
    for line in lines:
        if not line.strip():
            continue
        item = json.loads(line)
        slug = item["slug"]
        if slug in records:
            raise FreezeError(f"Duplicate source manifest slug: {slug}")
        records[slug] = line
    return records


def _references(review: dict, repo_root: Path) -> set[Path]:
    """Follow explicit collection/document references, never infer musical labels."""
    text = json.dumps(
        {"provenance": review.get("provenance"), "review_notes": review.get("review_notes")},
        ensure_ascii=False,
    )
    found = set()
    for match in CONTEXT_REF.finditer(text):
        relative = match.group().rstrip("./-")
        if relative.startswith("out/collection/") or relative == "docs/FORM_NOTATION.md":
            found.add(repo_root / relative)
    return found


def _context_paths(reviews: list[dict], repo_root: Path) -> tuple[list[Path], list[str]]:
    candidates = set()
    omitted = []
    references = set().union(*(_references(review, repo_root) for review in reviews))
    for reference in sorted(references):
        relative = reference.relative_to(repo_root).as_posix()
        if not _inside(reference, repo_root) or not reference.exists():
            omitted.append(f"missing_or_unsafe: {relative}")
            continue
        paths = sorted(reference.rglob("*")) if reference.is_dir() else [reference]
        for path in paths:
            rel = path.relative_to(repo_root)
            if any(part in EXCLUDED_CONTEXT_PARTS for part in rel.parts):
                continue
            if not _inside(path, repo_root):
                omitted.append(f"unsafe: {rel.as_posix()}")
                continue
            if path.is_dir():
                continue
            if not path.is_file():
                omitted.append(f"unsafe: {rel.as_posix()}")
                continue
            candidates.add(path)
    # Preserve decisions and provenance before large visual previews consume the cap.
    textual = {".abc", ".json", ".md", ".txt", ".csv", ".musicxml", ".xml"}
    ordered = sorted(candidates, key=lambda p: (p.suffix.lower() not in textual, p.as_posix()))
    selected = []
    total_bytes = 0
    for path in ordered:
        size = path.stat().st_size
        if len(selected) >= CONTEXT_FILE_LIMIT or total_bytes + size > CONTEXT_BYTE_LIMIT:
            omitted.append(f"context_limit: {path.relative_to(repo_root).as_posix()}")
            continue
        selected.append(path)
        total_bytes += size
    return sorted(selected), sorted(set(omitted))


def _put(root: Path, relative: str, source: Path, files: dict) -> None:
    if relative in files or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise FreezeError(f"Unsafe or duplicate freeze path: {relative}")
    before = _read_regular(source)
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(before)
    if _read_regular(source) != before:
        raise FreezeError(f"Source changed while freezing: {source}")
    files[relative] = {"sha256": _digest(before), "size": len(before)}


def _put_bytes(root: Path, relative: str, data: bytes, files: dict) -> None:
    if relative in files:
        raise FreezeError(f"Duplicate freeze path: {relative}")
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    files[relative] = {"sha256": _digest(data), "size": len(data)}


def freeze_references(out_dir: Path, destination: Path) -> dict:
    """Freeze all six editions without writing to the live review tree."""
    out_dir = Path(out_dir).resolve()
    repo_root = out_dir.parent
    destination = _canonical(Path(destination))
    if destination.exists() or destination.is_symlink():
        raise FreezeError(f"Freeze destination already exists: {destination}")
    app = ReviewApp(out_dir)
    if destination.is_relative_to(out_dir / "collection") or any(
        destination.is_relative_to(out_dir / slug) for slug in app.works
    ):
        raise FreezeError("Freeze destination must be outside live work and collection trees")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_records = _source_records(out_dir)
    work_slugs = []
    for number in REFERENCE_NUMBERS:
        candidates = [s for s in app.works if s.startswith(f"jaime-llanos_{number}_")]
        if len(candidates) != 1:
            raise FreezeError(f"Expected one manifest work for catalog number {number}")
        work_slugs.append(candidates[0])

    manifest_before = _read_regular(out_dir / "manifest.jsonl")
    temp = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        files = {}
        works = []
        live_checks = []
        for slug in work_slugs:
            review_source = app.path(slug, "overrides/review.json")
            review_bytes = _read_regular(review_source)
            review = json.loads(review_bytes)
            if (
                review.get("review_state") != "reviewed"
                or type(review.get("revision")) is not int
                or review["revision"] < 1
                or not isinstance(review.get("abc"), str)
                or not review["abc"].strip()
                or review.get("unresolved") != []
                or not isinstance(review.get("review_notes", ""), str)
                or not isinstance(review.get("base_abc"), str)
                or review.get("base_abc_sha256") != _digest(review["base_abc"].encode("utf-8"))
            ):
                raise FreezeError(f"Review is not a consistent completed edition: {slug}")
            try:
                history_before = app.history(slug)
            except ReviewError as exc:
                raise FreezeError(f"Invalid review history for {slug}: {exc}") from exc
            if (
                not history_before
                or not history_before[-1]["current"]
                or history_before[-1]["revision"] != review["revision"]
                or history_before[-1]["sha256"] != _digest(review_bytes)
            ):
                raise FreezeError(f"Current review missing from verified history: {slug}")

            source_line = source_records[slug]
            source_record = json.loads(source_line)
            source_value = source_record["pdf_path"]
            source_path = _canonical(out_dir / source_value)
            if not _inside(source_path, repo_root) or source_path.suffix.lower() != ".pdf":
                raise FreezeError(f"Unsafe source PDF path for {slug}")
            source_relative = f"inputs/{slug}/source.pdf"
            review_relative = f"answers/{slug}/review.json"
            abc_relative = f"answers/{slug}/review.abc"
            _put(temp, source_relative, source_path, files)
            _put(temp, review_relative, review_source, files)
            _put_bytes(temp, abc_relative, review["abc"].encode("utf-8"), files)

            overrides = app.path(slug, "overrides")
            all_override_paths = sorted(overrides.rglob("*"))
            if any(path.is_symlink() for path in all_override_paths):
                raise FreezeError(f"Symlink in review overrides: {slug}")
            override_paths = [path for path in all_override_paths if path.is_file()]
            for path in override_paths:
                if path == review_source:
                    continue
                if not _inside(path, out_dir):
                    raise FreezeError(f"Unsafe override path: {path}")
                rel = path.relative_to(overrides).as_posix()
                _put(temp, f"answers/{slug}/overrides/{rel}", path, files)

            reviews = [entry["review"] for entry in history_before]
            references = set().union(*(_references(entry, repo_root) for entry in reviews))
            if any(destination.is_relative_to(ref) for ref in references):
                raise FreezeError("Freeze destination is inside a referenced context directory")
            context_paths, omitted = _context_paths(reviews, repo_root)
            for path in context_paths:
                rel = path.relative_to(repo_root).as_posix()
                _put(temp, f"answers/{slug}/context/{rel}", path, files)
            history_after = app.history(slug)
            if _history_id(history_after) != _history_id(history_before):
                raise FreezeError(f"Review history changed while freezing: {slug}")
            if _read_regular(review_source) != review_bytes:
                raise FreezeError(f"Review changed while freezing: {slug}")
            if _digest(_read_regular(source_path)) != files[source_relative]["sha256"]:
                raise FreezeError(f"Source PDF changed while freezing: {slug}")
            live_checks.append(
                (
                    slug,
                    review_source,
                    review_bytes,
                    source_path,
                    source_relative,
                    _history_id(history_before),
                )
            )
            works.append(
                {
                    "slug": slug,
                    "title": source_record["metadata"]["title"],
                    "composer": source_record["metadata"]["composer"],
                    "revision": review["revision"],
                    "review_state": "reviewed",
                    "annotation_status": "pending",
                    "answer_review_path": review_relative,
                    "answer_abc_path": abc_relative,
                    "source_pdf_path": source_relative,
                    "source_manifest_pdf_path": source_value,
                    "source_manifest_line": source_line,
                    "source_manifest_line_sha256": _digest(source_line.encode("utf-8")),
                    "review_sha256": files[review_relative]["sha256"],
                    "abc_sha256": files[abc_relative]["sha256"],
                    "history_revisions": [x["revision"] for x in history_before],
                    "omitted_context_references": omitted,
                }
            )
        for (
            slug,
            review_source,
            review_bytes,
            source_path,
            source_relative,
            prior_history,
        ) in live_checks:
            if (
                _read_regular(review_source) != review_bytes
                or _history_id(app.history(slug)) != prior_history
            ):
                raise FreezeError(f"Review changed while freezing: {slug}")
            if _digest(_read_regular(source_path)) != files[source_relative]["sha256"]:
                raise FreezeError(f"Source PDF changed while freezing: {slug}")
        if _read_regular(out_dir / "manifest.jsonl") != manifest_before:
            raise FreezeError("Source manifest changed while freezing")
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "edition_status": "consumed_reviewed_editorial_editions",
            "source_fidelity_ground_truth": False,
            "heldout": False,
            "annotation_status": "pending",
            "source_fidelity_grading_requires_audit": True,
            "history_completeness": (
                "Only surviving archived revisions are included; older overwritten revisions "
                "cannot be reconstructed."
            ),
            "works": works,
            "files": files,
        }
        manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        (temp / "manifest.json").write_bytes(manifest_bytes)
        (temp / "manifest.sha256").write_text(_digest(manifest_bytes) + "\n", encoding="ascii")
        verify_freeze(temp)
        _publish_no_replace(temp, destination)
        return manifest
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def verify_freeze(root: Path, out_dir: Path | None = None) -> dict:
    """Verify every snapshot byte, including the pinned source record, offline."""
    root = _canonical(Path(root))
    manifest_bytes = _read_regular(root / "manifest.json")
    seal = _read_regular(root / "manifest.sha256").decode("ascii")
    if seal != _digest(manifest_bytes) + "\n":
        raise FreezeError("Freeze manifest seal changed")
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), dict):
        raise FreezeError("Unsupported freeze manifest")
    if (
        manifest.get("edition_status") != "consumed_reviewed_editorial_editions"
        or manifest.get("heldout") is not False
        or manifest.get("source_fidelity_ground_truth") is not False
        or manifest.get("annotation_status") != "pending"
        or manifest.get("source_fidelity_grading_requires_audit") is not True
    ):
        raise FreezeError("Invalid freeze evidence status")
    works = manifest.get("works")
    if not isinstance(works, list) or len(works) != len(REFERENCE_NUMBERS):
        raise FreezeError("Freeze must contain exactly six references")
    slugs = [work["slug"] for work in works]

    def has_one(number: int) -> bool:
        pattern = rf"jaime-llanos_{number}_[A-Za-z0-9_-]+"
        return sum(bool(re.fullmatch(pattern, slug)) for slug in slugs) == 1

    if len(set(slugs)) != len(slugs) or not all(has_one(n) for n in REFERENCE_NUMBERS):
        raise FreezeError("Freeze reference catalog identity changed")
    listed = set(manifest["files"])
    actual = {
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink()
    }
    if actual != listed | {"manifest.json", "manifest.sha256"}:
        missing = sorted(listed - actual)
        extra = sorted(actual - listed - {"manifest.json", "manifest.sha256"})
        raise FreezeError(f"Freeze file set changed: missing={missing}, extra={extra}")
    for relative, expected in manifest["files"].items():
        path = root / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts or not _inside(path, root):
            raise FreezeError(f"Unsafe freeze path: {relative}")
        data = _read_regular(path)
        if len(data) != expected["size"] or _digest(data) != expected["sha256"]:
            raise FreezeError(f"Freeze content changed: {relative}")
    for work in works:
        line = work["source_manifest_line"]
        record = json.loads(line)
        if (
            _digest(line.encode("utf-8")) != work["source_manifest_line_sha256"]
            or record["slug"] != work["slug"]
            or record["pdf_path"] != work["source_manifest_pdf_path"]
            or record["metadata"]["title"] != work["title"]
            or record["metadata"]["composer"] != work["composer"]
        ):
            raise FreezeError(f"Source manifest identity changed: {work['slug']}")
        source = work["source_pdf_path"]
        review = work["answer_review_path"]
        abc = work["answer_abc_path"]
        if (
            source != f"inputs/{work['slug']}/source.pdf"
            or review != f"answers/{work['slug']}/review.json"
            or abc != f"answers/{work['slug']}/review.abc"
            or manifest["files"][review]["sha256"] != work["review_sha256"]
            or manifest["files"][abc]["sha256"] != work["abc_sha256"]
            or source not in manifest["files"]
            or work["review_state"] != "reviewed"
        ):
            raise FreezeError(f"Freeze path or digest identity changed: {work['slug']}")
        saved = json.loads(_read_regular(root / review))
        if (
            saved["abc"].encode("utf-8") != _read_regular(root / abc)
            or saved["revision"] != work["revision"]
            or saved["review_state"] != "reviewed"
            or saved.get("unresolved") != []
            or work["annotation_status"] != "pending"
        ):
            raise FreezeError(f"Review identity changed: {work['slug']}")
        archive_root = root / "answers" / work["slug"] / "overrides" / "history"
        revisions = {}
        for archive in sorted(archive_root.glob("*.json")):
            raw = _read_regular(archive)
            entry = json.loads(raw)
            revision = entry.get("revision")
            digest = _digest(raw)
            if type(revision) is not int or archive.name != f"{revision}-{digest}.json":
                raise FreezeError(f"Invalid frozen review history: {work['slug']}")
            if revision in revisions and revisions[revision][0] != digest:
                raise FreezeError(f"Conflicting frozen review history: {work['slug']}")
            revisions[revision] = (digest, entry)
        current_revision = saved["revision"]
        if (
            current_revision in revisions
            and revisions[current_revision][0] != work["review_sha256"]
        ):
            raise FreezeError(f"Conflicting current frozen review: {work['slug']}")
        revisions[current_revision] = (work["review_sha256"], saved)
        if sorted(revisions) != work["history_revisions"]:
            raise FreezeError(f"Frozen history revision list changed: {work['slug']}")
        for revision, (_, entry) in revisions.items():
            prior_hash = (entry.get("save_metadata") or {}).get("previous_review_sha256")
            if prior_hash is not None and revisions.get(revision - 1, (None,))[0] != prior_hash:
                raise FreezeError(f"Broken frozen review history chain: {work['slug']}")
        if out_dir is not None:
            current = (Path(out_dir).resolve() / record["pdf_path"]).resolve()
            if _digest(_read_regular(current)) != manifest["files"][source]["sha256"]:
                raise FreezeError(f"Live source PDF differs from freeze: {work['slug']}")
    return manifest
