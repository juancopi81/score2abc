from __future__ import annotations

import csv
import json
import os
import unicodedata
from pathlib import Path
from typing import Iterable, List, Optional

from score2abc.schemas import WorkItem, WorkMetadata


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_value.lower()
    cleaned = []
    previous_hyphen = False
    for ch in lowered:
        if ch.isalnum():
            cleaned.append(ch)
            previous_hyphen = False
        elif ch.isspace() or ch in {"-", "_", "/"}:
            if cleaned and not previous_hyphen:
                cleaned.append("-")
                previous_hyphen = True
        else:
            continue
    slug = "".join(cleaned).strip("-")
    return slug or "untitled"


def load_metadata_csv(metadata_csv: Path, input_dir: Path) -> List[WorkItem]:
    """Load dataset metadata into WorkItem entries."""
    items: List[WorkItem] = []
    seen_slugs: set[str] = set()
    with metadata_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Metadata CSV must include a header row")

        for row in reader:
            title = (row.get("title") or "").strip()
            if not title:
                raise ValueError("Metadata CSV row missing title")

            rhythm = (row.get("rhythm") or row.get("genre") or "").strip()
            if not rhythm:
                raise ValueError(f"Metadata CSV row missing rhythm/genre for title: {title}")

            composer = (row.get("composer") or row.get("author") or "").strip()
            if not composer:
                raise ValueError(f"Metadata CSV row missing composer/author for title: {title}")

            pdf_file = (row.get("pdf_file") or "").strip()
            if pdf_file:
                pdf_path = input_dir / pdf_file
                slug = Path(pdf_file).stem
            else:
                slug = _slugify(f"{title}-{rhythm}-{composer}")
                pdf_path = input_dir / f"{slug}.pdf"

            if slug in seen_slugs:
                raise ValueError(f"Duplicate slug detected in metadata CSV: {slug}")
            seen_slugs.add(slug)

            if not pdf_path.exists():
                raise FileNotFoundError(f"Source PDF not found for slug '{slug}': {pdf_path}")

            metadata = WorkMetadata(
                title=title,
                composer=composer,
                rhythm=rhythm,
                time_signature=_optional_field(row, "time_signature"),
                key_hint=_optional_field(row, "key_hint"),
                tempo_hint=_optional_field(row, "tempo_hint"),
            )
            item = WorkItem(slug=slug, pdf_path=pdf_path, metadata=metadata)
            items.append(item)

    return items


def _optional_field(row: dict, key: str) -> Optional[str]:
    value = row.get(key)
    if value is None:
        return None
    value = value.strip()
    return value or None


def write_manifest_jsonl(items: Iterable[WorkItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_dir = path.parent.resolve()
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            pdf_path = item.pdf_path
            if not pdf_path.is_absolute():
                pdf_path = (Path.cwd() / pdf_path).resolve()

            try:
                pdf_path_value = os.path.relpath(pdf_path, start=manifest_dir)
            except ValueError:
                # Fallback for cross-volume paths where relpath is not possible.
                pdf_path_value = str(pdf_path)

            payload = {
                "slug": item.slug,
                "pdf_path": pdf_path_value,
                "metadata": item.metadata.model_dump(mode="json"),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_manifest_jsonl(path: Path) -> List[WorkItem]:
    items: List[WorkItem] = []
    manifest_dir = path.parent.resolve()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            pdf_path = Path(payload["pdf_path"])
            if not pdf_path.is_absolute():
                payload["pdf_path"] = str((manifest_dir / pdf_path).resolve())
            items.append(WorkItem.model_validate(payload))
    return items
