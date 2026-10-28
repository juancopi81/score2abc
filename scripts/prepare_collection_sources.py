#!/usr/bin/env python3
"""Split verified collection pages into a create-once, local-only source packet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_collection_inventory import DEFAULT_CATALOG, validate_catalog  # noqa: E402

SAFE_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
MANIFEST_NAME = "packet_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout


def _pdf_page_count(path: Path) -> int:
    output = _run(["pdfinfo", str(path)])
    match = re.search(r"^Pages:\s+(\d+)\s*$", output, re.MULTILINE)
    if not match:
        raise ValueError(f"Could not read PDF page count: {path}")
    return int(match.group(1))


def _selected_entries(catalog: dict, song_numbers: list[int] | None) -> list[dict]:
    validate_catalog(catalog)
    verified = [entry for entry in catalog["entries"] if entry["mapping_status"] == "verified"]

    slugs = [entry["slug"] for entry in verified]
    if any(not isinstance(slug, str) or not SAFE_SLUG.fullmatch(slug) for slug in slugs):
        raise ValueError("Verified entries require safe filename slugs")
    if len(slugs) != len(set(slugs)):
        raise ValueError("Verified entries require unique slugs")

    if song_numbers is None:
        return verified
    if len(song_numbers) != len(set(song_numbers)):
        raise ValueError("Duplicate --song-number values are not allowed")

    by_number = {entry["song_number"]: entry for entry in verified}
    unavailable = [number for number in song_numbers if number not in by_number]
    if unavailable:
        raise ValueError(f"Song numbers are not verified mapped entries: {unavailable}")
    return [by_number[number] for number in song_numbers]


def _write_metadata(entries: list[dict], path: Path) -> None:
    fields = [
        "title",
        "normalized_title",
        "author",
        "genre",
        "time_signature",
        "key_hint",
        "tempo_hint",
        "pdf_file",
    ]
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            literal = entry["literal_source"]
            writer.writerow(
                {
                    "title": literal["title"],
                    "normalized_title": entry["normalized_title"],
                    "author": literal["author"],
                    "genre": literal["genre"],
                    "time_signature": "",
                    "key_hint": "",
                    "tempo_hint": "",
                    "pdf_file": f"pdfs/{entry['slug']}.pdf",
                }
            )


def prepare_packet(
    source: Path,
    output: Path,
    catalog_path: Path = DEFAULT_CATALOG,
    song_numbers: list[int] | None = None,
) -> dict:
    """Verify, stage, and publish a packet without reading linked reference content."""
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")

    catalog_bytes = catalog_path.read_bytes()
    catalog = json.loads(catalog_bytes)
    entries = _selected_entries(catalog, song_numbers)

    source_sha = _sha256(source)
    if source_sha != catalog["source"]["sha256"]:
        raise ValueError("Source PDF hash does not match the inspected collection")

    for executable in ("pdfseparate", "pdfunite", "pdfinfo"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"Required Poppler executable not found: {executable}")
    source_pages = _pdf_page_count(source)
    if source_pages != catalog["source"]["pdf_page_count"]:
        raise ValueError("Source PDF page count differs from the catalog")

    output_parent = output.parent.resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".collection-packet-", dir=output_parent) as temp_name:
        stage = Path(temp_name)
        pdf_dir = stage / "pdfs"
        parts_dir = stage / "pages"
        pdf_dir.mkdir()
        parts_dir.mkdir()

        works = []
        for entry in entries:
            page_parts = []
            for page in entry["pdf_pages"]:
                part = parts_dir / f"page-{page:03d}.pdf"
                if not part.exists():
                    _run(
                        [
                            "pdfseparate",
                            "-f",
                            str(page),
                            "-l",
                            str(page),
                            str(source),
                            str(part),
                        ]
                    )
                page_parts.append(part)

            destination = pdf_dir / f"{entry['slug']}.pdf"
            _run(["pdfunite", *map(str, page_parts), str(destination)])
            output_pages = _pdf_page_count(destination)
            if output_pages != len(entry["pdf_pages"]):
                raise ValueError(f"Extracted page count differs for {entry['slug']}")
            works.append(
                {
                    "song_number": entry["song_number"],
                    "slug": entry["slug"],
                    "status": "source_ready",
                    "transcription_status": "transcription_pending",
                    "original_pdf_pages": entry["pdf_pages"],
                    "original_manuscript_pages": entry["manuscript_pages"],
                    "output_pdf": f"pdfs/{destination.name}",
                    "output_sha256": _sha256(destination),
                    "output_page_count": output_pages,
                }
            )

        _write_metadata(entries, stage / "metadata.csv")
        remaining = [
            {
                "song_number": entry["song_number"],
                "slug": entry["slug"],
                "mapping_status": entry["mapping_status"],
                "manuscript_pages": entry["manuscript_pages"],
                "pdf_pages": entry["pdf_pages"],
            }
            for entry in catalog["entries"]
            if entry["mapping_status"] != "verified"
        ]
        manifest = {
            "schema_version": 1,
            "collection_id": catalog["collection_id"],
            "classification": "local_only_source_packet",
            "source": {
                "path": str(source.resolve()),
                "sha256": source_sha,
                "page_count": source_pages,
            },
            "catalog": {
                "path": str(catalog_path.resolve()),
                "sha256": hashlib.sha256(catalog_bytes).hexdigest(),
            },
            "selection": "all_verified" if song_numbers is None else "explicit_song_numbers",
            "works": works,
            "remaining_missing_or_unresolved_catalog_rows": remaining,
            "content_boundaries": {
                "musicxml_or_reference_content_opened": False,
                "pipeline_run": False,
                "review_drafts_created": False,
            },
        }
        (stage / MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        shutil.rmtree(parts_dir)

        if _sha256(source) != source_sha:
            raise ValueError("Source PDF changed during extraction")
        output.mkdir(exist_ok=False)
        os.replace(stage / "pdfs", output / "pdfs")
        os.replace(stage / "metadata.csv", output / "metadata.csv")
        os.replace(stage / MANIFEST_NAME, output / MANIFEST_NAME)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--song-number", type=int, action="append")
    args = parser.parse_args()
    prepare_packet(args.source, args.output, args.catalog, args.song_number)


if __name__ == "__main__":
    main()
