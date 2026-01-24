from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List

from score2abc.abc import events_to_abc
from score2abc.manifest import load_manifest_jsonl, load_metadata_csv, write_manifest_jsonl
from score2abc.render import create_minimal_system_crops, render_abc_preview, render_pdf_to_images
from score2abc.schemas import WorkItem
from score2abc.utils import Timer, get_logger

DEFAULT_DPI = 300


def ingest(input_dir: Path, metadata_csv: Path, out_dir: Path) -> List[WorkItem]:
    logger = get_logger("score2abc.ingest")
    logger.info("Loading metadata")

    with Timer("load metadata", logger=logger):
        work_items = load_metadata_csv(metadata_csv, input_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    for item in work_items:
        work_dir = out_dir / item.slug
        work_dir.mkdir(parents=True, exist_ok=True)

        source_pdf = work_dir / "source.pdf"
        if item.pdf_path.exists():
            shutil.copy(item.pdf_path, source_pdf)
            logger.info("Copied source PDF: %s", source_pdf)
        else:
            logger.warning("Source PDF missing: %s", item.pdf_path)

        metadata_path = work_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(item.metadata.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        logger.info("Wrote metadata: %s", metadata_path)

    manifest_path = out_dir / "manifest.jsonl"
    write_manifest_jsonl(work_items, manifest_path)
    logger.info("Wrote manifest: %s", manifest_path)
    return work_items


def run(out_dir: Path, workers: int = 1, use_vlm: bool = False) -> int:
    logger = get_logger("score2abc.run")
    manifest_path = out_dir / "manifest.jsonl"
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        return 1

    work_items = load_manifest_jsonl(manifest_path)
    logger.info("Running %d work items (workers=%d, use_vlm=%s)", len(work_items), workers, use_vlm)

    for item in work_items:
        work_dir = out_dir / item.slug
        pages_dir = work_dir / "pages"
        systems_dir = work_dir / "systems"
        intermediate_dir = work_dir / "intermediate"
        final_dir = work_dir / "final"

        intermediate_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Processing %s", item.slug)
        source_pdf = work_dir / "source.pdf"
        pdf_path = source_pdf if source_pdf.exists() else item.pdf_path

        with Timer(f"render pages ({item.slug})", logger=logger):
            page_paths = render_pdf_to_images(pdf_path, pages_dir, DEFAULT_DPI, logger)

        with Timer(f"system crops ({item.slug})", logger=logger):
            create_minimal_system_crops(page_paths, systems_dir, logger)

        events = _build_stub_events(item)
        events_path = intermediate_dir / "events.json"
        events_path.write_text(json.dumps(events, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote events: %s", events_path)

        abc_text = events_to_abc(events, item.metadata)
        melody_path = final_dir / "melody.abc"
        melody_path.write_text(abc_text, encoding="utf-8")
        logger.info("Wrote ABC: %s", melody_path)

        melody_chords_path = final_dir / "melody_with_chords.abc"
        melody_chords_path.write_text(abc_text, encoding="utf-8")
        logger.info("Wrote ABC with chords: %s", melody_chords_path)

        preview_path = final_dir / "preview.svg"
        render_abc_preview(melody_chords_path, preview_path, logger)

    return 0


def qa(out_dir: Path, open_ui: bool = False) -> int:
    logger = get_logger("score2abc.qa")
    manifest_path = out_dir / "manifest.jsonl"
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        return 1

    work_items = load_manifest_jsonl(manifest_path)
    for item in work_items:
        preview_path = out_dir / item.slug / "final" / "preview.svg"
        if preview_path.exists():
            logger.info("Preview available: %s", preview_path)
        else:
            logger.warning("Preview missing: %s", preview_path)

    if open_ui:
        logger.info("UI not implemented yet (stub)")

    return 0


def export(out_dir: Path, export_format: str = "index.md") -> int:
    logger = get_logger("score2abc.export")
    manifest_path = out_dir / "manifest.jsonl"
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        return 1

    work_items = load_manifest_jsonl(manifest_path)
    index_path = out_dir / export_format

    lines = ["# score2abc catalog", ""]
    for item in work_items:
        work_dir = out_dir / item.slug
        abc_path = work_dir / "final" / "melody.abc"
        abc_text = abc_path.read_text(encoding="utf-8") if abc_path.exists() else ""
        lines.append(f"## {item.metadata.title}")
        lines.append("")
        lines.append(f"- Composer: {item.metadata.composer}")
        lines.append(f"- Rhythm: {item.metadata.rhythm}")
        lines.append("")
        lines.append("```abc")
        lines.append(abc_text.strip() or "C D E F |")
        lines.append("```")
        lines.append("")

    index_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote catalog: %s", index_path)
    return 0


def _build_stub_events(item: WorkItem) -> dict:
    return {
        "time_signature": item.metadata.time_signature or "4/4",
        "notes": [
            {
                "measure": 1,
                "onset_beats": 0.0,
                "duration_beats": 1.0,
                "pitch_midi": 60,
            },
            {
                "measure": 1,
                "onset_beats": 1.0,
                "duration_beats": 1.0,
                "pitch_midi": 62,
            },
            {
                "measure": 1,
                "onset_beats": 2.0,
                "duration_beats": 1.0,
                "pitch_midi": 64,
            },
            {
                "measure": 1,
                "onset_beats": 3.0,
                "duration_beats": 1.0,
                "pitch_midi": 65,
            },
        ],
        "chords": [
            {
                "measure": 1,
                "onset_beats": 0.0,
                "symbol": item.metadata.key_hint or "C",
            }
        ],
    }
