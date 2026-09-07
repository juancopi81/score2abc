#!/usr/bin/env python3
"""Validate the portable collection catalog and optionally build a local availability report.

This does not OCR, split the PDF, open MusicXML, or change frozen experiment artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "dataset/catalog/jaime_llanos_collection.json"


def validate_catalog(catalog: dict) -> dict:
    """Reject ambiguous ownership, incomplete coverage, and inconsistent status/count claims."""
    entries = catalog["entries"]
    if [entry["song_number"] for entry in entries] != list(range(1, 101)):
        raise ValueError("Expected exactly the ordered 100 source-table rows")
    occupied = [page["pdf_page"] for page in catalog["front_matter"]]
    manuscript = []
    missing = []
    counts = {"verified": 0, "missing": 0, "unresolved": 0}
    for entry in entries:
        status = entry["mapping_status"]
        if status not in counts:
            raise ValueError("Unknown mapping status")
        counts[status] += 1
        physical = entry["pdf_pages"]
        logical = entry["manuscript_pages"]
        literal = entry["literal_source"]
        if literal["song_number"] != str(entry["song_number"]):
            raise ValueError("Literal song number differs from entry identity")
        if status == "verified":
            if not physical or len(physical) != len(logical):
                raise ValueError("Verified mappings require every manuscript page")
            if physical != list(range(physical[0], physical[0] + len(physical))):
                raise ValueError("Multi-page works must preserve consecutive physical pages")
        elif physical:
            raise ValueError("Missing/unresolved entries cannot own verified PDF pages")
        if status == "missing":
            if not logical or literal["title"] == "-":
                raise ValueError("Missing works require named source and manuscript pages")
            missing.extend(logical)
        if status == "unresolved":
            if logical or literal["title"] != "-" or entry["slug"] is not None:
                raise ValueError("Unresolved placeholders must remain unnamed and unmapped")
        occupied.extend(physical)
        manuscript.extend(logical)
        for reference in entry["references"]:
            path = Path(reference["path"])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("References must be portable repository-relative paths")
    if sorted(occupied) != list(range(1, catalog["source"]["pdf_page_count"] + 1)):
        raise ValueError("Physical PDF coverage must be exhaustive and unique")
    if sorted(manuscript) != list(range(1, 100)):
        raise ValueError("Manuscript pages 1-99 must be uniquely accounted for")
    summary = {
        "table_rows": len(entries),
        "named_works": counts["verified"] + counts["missing"],
        "verified_present_works": counts["verified"],
        "missing_named_works": counts["missing"],
        "unresolved_placeholder_rows": counts["unresolved"],
        "mapped_score_pages": sum(len(entry["pdf_pages"]) for entry in entries),
        "front_matter_pages": len(catalog["front_matter"]),
        "missing_manuscript_pages": sorted(missing),
    }
    if summary != catalog["summary"]:
        raise ValueError("Stored summary differs from mapping evidence")
    return summary


def build_report(catalog: dict, source: Path | None, root: Path = ROOT) -> dict:
    """Check file availability only; linked transcription contents are never read."""
    summary = validate_catalog(catalog)
    source_verified = False
    if source is not None:
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != catalog["source"]["sha256"]:
            raise ValueError("Source PDF hash does not match the inspected collection")
        source_verified = True
    return {
        "collection_id": catalog["collection_id"],
        "summary": summary,
        "source_sha256_verified": source_verified,
        "source_path": str(source.resolve()) if source is not None else None,
        "references": [
            {
                "song_number": entry["song_number"],
                "kind": reference["kind"],
                "path": reference["path"],
                "exists": (root / reference["path"]).is_file(),
                "content_opened": False,
            }
            for entry in catalog["entries"]
            for reference in entry["references"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--source", type=Path, help="Local original PDF; verifies exact SHA256")
    parser.add_argument(
        "--output", type=Path, help="Create-once local JSON report; defaults to stdout"
    )
    args = parser.parse_args()
    report = build_report(json.loads(args.catalog.read_text()), args.source)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
