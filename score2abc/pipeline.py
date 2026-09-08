from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from score2abc.abc import events_to_abc
from score2abc.chords import build_chord_ocr, extract_chords_for_systems
from score2abc.dataset import load_dataset_metadata
from score2abc.evaluation import evaluate as run_evaluation
from score2abc.manifest import load_manifest_jsonl, write_manifest_jsonl
from score2abc.melody import (
    DEFAULT_AUDIVERIS_COMMAND,
    DEFAULT_HOMR_COMMAND,
    DEFAULT_MUSICXML_SOURCE_DIR,
    INTERMEDIATE_MUSICXML_FILENAME,
    MusicXMLBackendError,
    build_musicxml_backend,
    extract_canonical_melody_events,
    extract_melody_events,
)
from score2abc.render import (
    create_system_crops,
    render_abc_preview,
    render_pdf_to_images,
)
from score2abc.schemas import WorkItem
from score2abc.utils import Timer, get_logger

DEFAULT_DPI = 300
DEFAULT_VLM_FIXTURES_DIR = Path("tests/fixtures/vlm")
DEFAULT_VLM_CACHE_DIR = Path(".cache/vlm")


def ingest(input_dir: Path, metadata_csv: Path, out_dir: Path) -> int:
    logger = get_logger("score2abc.ingest")
    logger.info("Loading metadata")

    try:
        with Timer("load metadata", logger=logger):
            work_items = load_dataset_metadata(metadata_csv, input_dir)
    except Exception as exc:
        logger.error("Failed to load metadata: %s", exc)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    per_work_status: List[Dict[str, Any]] = []

    for item in work_items:
        item_status: Dict[str, Any] = {
            "slug": item.slug,
            "status": "success",
            "errors": [],
        }
        work_dir = out_dir / item.slug
        work_dir.mkdir(parents=True, exist_ok=True)
        started_at = _utcnow()

        try:
            source_pdf = work_dir / "source.pdf"
            shutil.copy(item.pdf_path, source_pdf)
            logger.info("Copied source PDF: %s", source_pdf)

            metadata_path = work_dir / "metadata.json"
            metadata_path.write_text(
                json.dumps(item.metadata.model_dump(mode="json"), ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
            logger.info("Wrote metadata: %s", metadata_path)

            ended_at = _utcnow()
            _write_stage_artifact(
                work_dir=work_dir,
                stage="ingest",
                status="success",
                started_at=started_at,
                ended_at=ended_at,
                inputs={
                    "pdf_path": str(item.pdf_path),
                    "metadata_csv": str(metadata_csv),
                },
                outputs={
                    "source_pdf": str(source_pdf),
                    "metadata_json": str(metadata_path),
                },
                params={},
            )
        except Exception as exc:
            ended_at = _utcnow()
            message = f"Failed to ingest work item: {exc}"
            item_status["status"] = "failed"
            item_status["errors"].append(message)
            logger.error("%s (%s)", message, item.slug)
            _write_stage_artifact(
                work_dir=work_dir,
                stage="ingest",
                status="failed",
                started_at=started_at,
                ended_at=ended_at,
                inputs={
                    "pdf_path": str(item.pdf_path),
                    "metadata_csv": str(metadata_csv),
                },
                outputs={},
                params={},
                error=message,
            )

        per_work_status.append(item_status)

    manifest_path = out_dir / "manifest.jsonl"
    if any(status["status"] == "failed" for status in per_work_status):
        logger.error("Ingest failed for at least one work item; manifest not written")
    else:
        write_manifest_jsonl(work_items, manifest_path)
        logger.info("Wrote manifest: %s", manifest_path)

    exit_code = 1 if any(status["status"] == "failed" for status in per_work_status) else 0
    _write_command_status(
        out_dir=out_dir,
        command="ingest",
        per_work=per_work_status,
        extra={"manifest_path": str(manifest_path), "manifest_written": exit_code == 0},
    )
    return exit_code


def run(
    out_dir: Path,
    workers: int = 1,
    use_vlm: bool = False,
    musicxml_backend_name: str = "fixture",
    audiveris_command: str = DEFAULT_AUDIVERIS_COMMAND,
    audiveris_input: str = "page",
    homr_command: str = DEFAULT_HOMR_COMMAND,
    homr_input: str = "page",
    slugs: list[str] | None = None,
) -> int:
    logger = get_logger("score2abc.run")
    manifest_path = out_dir / "manifest.jsonl"
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        return 1

    work_items = load_manifest_jsonl(manifest_path)
    if slugs:
        requested_slugs = set(slugs)
        work_items = [item for item in work_items if item.slug in requested_slugs]
        missing_slugs = sorted(requested_slugs - {item.slug for item in work_items})
        if missing_slugs:
            logger.error("Requested slug(s) not found in manifest: %s", ", ".join(missing_slugs))
            return 1

    logger.info(
        "Running %d work items (workers=%d, use_vlm=%s, musicxml_backend=%s)",
        len(work_items),
        workers,
        use_vlm,
        musicxml_backend_name,
    )
    per_work_status: List[Dict[str, Any]] = []

    for item in work_items:
        item_status: Dict[str, Any] = {
            "slug": item.slug,
            "status": "success",
            "errors": [],
        }
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

        try:
            render_started = _utcnow()
            with Timer(f"render pages ({item.slug})", logger=logger):
                page_paths = render_pdf_to_images(pdf_path, pages_dir, DEFAULT_DPI, logger)
            render_ended = _utcnow()
            render_status = "success" if page_paths else "failed"
            _write_stage_artifact(
                work_dir=work_dir,
                stage="render_pages",
                status=render_status,
                started_at=render_started,
                ended_at=render_ended,
                inputs={"pdf_path": str(pdf_path)},
                outputs={"pages": [str(p) for p in page_paths]},
                params={"dpi": DEFAULT_DPI},
                error=None if page_paths else "No pages rendered from source PDF",
            )
            if not page_paths:
                raise RuntimeError("No pages were rendered from PDF")

            segment_started = _utcnow()
            with Timer(f"system crops ({item.slug})", logger=logger):
                segmentation = create_system_crops(page_paths, systems_dir, logger)
            segment_ended = _utcnow()
            _write_stage_artifact(
                work_dir=work_dir,
                stage="segment_systems",
                status="success" if segmentation.system_crops else "failed",
                started_at=segment_started,
                ended_at=segment_ended,
                inputs={"pages": [str(p) for p in page_paths]},
                outputs={
                    "system_crops": [str(p) for p in segmentation.system_crops],
                    "chord_crops_above": [str(p) for p in segmentation.chord_crops_above],
                    "chord_crops_below": [str(p) for p in segmentation.chord_crops_below],
                    "deskewed_pages": [str(p) for p in segmentation.deskewed_pages],
                    "debug_overlays": [str(p) for p in segmentation.debug_overlays],
                    "debug_manifests": [str(p) for p in segmentation.debug_manifests],
                    "rejected_candidate_crops": [
                        str(p) for p in segmentation.rejected_candidate_crops
                    ],
                    "candidate_diagnostics": segmentation.candidate_diagnostics,
                },
                params={},
                error=(None if segmentation.system_crops else "No system crops generated"),
            )
            if not segmentation.system_crops:
                raise RuntimeError("No system crops were generated")

            chords_started = _utcnow()
            ocr = build_chord_ocr(
                use_vlm,
                fixtures_dir=DEFAULT_VLM_FIXTURES_DIR,
                cache_dir=DEFAULT_VLM_CACHE_DIR,
            )
            chords_payload = extract_chords_for_systems(
                ocr=ocr,
                system_crops=segmentation.system_crops,
                chord_crops_above=segmentation.chord_crops_above,
                chord_crops_below=segmentation.chord_crops_below,
                metadata=item.metadata,
                logger=logger,
            )
            chords_path = intermediate_dir / "chords.json"
            chords_path.write_text(json.dumps(chords_payload, indent=2) + "\n", encoding="utf-8")
            logger.info("Wrote chords: %s", chords_path)
            chords_ended = _utcnow()
            _write_stage_artifact(
                work_dir=work_dir,
                stage="extract_chords",
                status="success",
                started_at=chords_started,
                ended_at=chords_ended,
                inputs={
                    "system_crops": [str(p) for p in segmentation.system_crops],
                    "chord_crops_above": [str(p) for p in segmentation.chord_crops_above],
                    "chord_crops_below": [str(p) for p in segmentation.chord_crops_below],
                },
                outputs={"chords_json": str(chords_path)},
                params={
                    "use_vlm": use_vlm,
                    "provider": chords_payload["provider"],
                    "model_id": chords_payload["model_id"],
                    "prompt_version": chords_payload["prompt_version"],
                },
            )

            musicxml_started = _utcnow()
            musicxml_source_dir = DEFAULT_MUSICXML_SOURCE_DIR
            musicxml_backend = build_musicxml_backend(
                source_dir=musicxml_source_dir,
                backend=musicxml_backend_name,
                audiveris_command=audiveris_command,
                audiveris_input=audiveris_input,
                homr_command=homr_command,
                homr_input=homr_input,
            )
            intermediate_musicxml_path = intermediate_dir / INTERMEDIATE_MUSICXML_FILENAME
            musicxml_inputs: Dict[str, Any] = {
                "source_dir": str(musicxml_source_dir),
                "slug": item.slug,
            }
            if musicxml_backend.name in {"audiveris", "homr"}:
                musicxml_inputs["pages_dir"] = str(pages_dir)
                musicxml_inputs["systems_dir"] = str(systems_dir)
            musicxml_outputs: Dict[str, Any] = {"musicxml": None}
            musicxml_status: str
            musicxml_error: str | None = None
            try:
                produced = musicxml_backend.produce_musicxml(item=item, work_dir=work_dir)
            except MusicXMLBackendError as exc:
                musicxml_status = "failed"
                musicxml_error = str(exc)
                logger.error("%s (%s)", musicxml_error, item.slug)
            else:
                if produced is None:
                    musicxml_status = "skipped"
                    if intermediate_musicxml_path.exists():
                        musicxml_inputs["manual_override"] = str(intermediate_musicxml_path)
                        musicxml_outputs["musicxml"] = str(intermediate_musicxml_path)
                        logger.info(
                            "No MusicXML fixture for %s; using existing %s",
                            item.slug,
                            intermediate_musicxml_path,
                        )
                    else:
                        logger.info(
                            "No MusicXML source for %s; extract_musicxml skipped",
                            item.slug,
                        )
                else:
                    musicxml_status = "success"
                    musicxml_inputs["source"] = str(produced.source_path)
                    musicxml_outputs["musicxml"] = str(produced.output_path)
                    if produced.raw_output_path is not None:
                        musicxml_outputs["raw_musicxml"] = str(produced.raw_output_path)
                    logger.info(
                        "Produced MusicXML for %s from %s",
                        item.slug,
                        produced.source_path,
                    )
            musicxml_ended = _utcnow()
            _write_stage_artifact(
                work_dir=work_dir,
                stage="extract_musicxml",
                status=musicxml_status,
                started_at=musicxml_started,
                ended_at=musicxml_ended,
                inputs=musicxml_inputs,
                outputs=musicxml_outputs,
                params={
                    "backend": musicxml_backend.name,
                    "audiveris_command": (
                        audiveris_command if musicxml_backend.name == "audiveris" else None
                    ),
                    "audiveris_input": (
                        audiveris_input if musicxml_backend.name == "audiveris" else None
                    ),
                    "homr_command": homr_command if musicxml_backend.name == "homr" else None,
                    "homr_input": homr_input if musicxml_backend.name == "homr" else None,
                },
                error=musicxml_error,
            )
            if musicxml_status == "failed":
                raise RuntimeError(musicxml_error or "extract_musicxml failed")

            melody_started = _utcnow()
            musicxml_source = _find_musicxml_source(work_dir)
            melody_json_path = intermediate_dir / "melody.json"
            melody_payload: Dict[str, Any] | None = None
            normalize_melody_payload: Dict[str, Any] | None = None
            melody_status: str
            melody_error: str | None = None
            if musicxml_source is None:
                melody_status = "skipped"
                logger.info(
                    "No MusicXML source found for %s; melody extraction skipped",
                    item.slug,
                )
            else:
                try:
                    melody_payload = extract_melody_events(musicxml_source)
                    normalize_melody_payload = extract_canonical_melody_events(musicxml_source)
                    melody_json_path.write_text(
                        json.dumps(melody_payload, indent=2) + "\n", encoding="utf-8"
                    )
                    logger.info("Wrote melody: %s", melody_json_path)
                    melody_status = "success"
                except Exception as exc:
                    melody_payload = None
                    normalize_melody_payload = None
                    melody_status = "failed"
                    melody_error = f"Failed to parse MusicXML {musicxml_source}: {exc}"
                    logger.error("%s (%s)", melody_error, item.slug)
            melody_ended = _utcnow()
            _write_stage_artifact(
                work_dir=work_dir,
                stage="extract_melody",
                status=melody_status,
                started_at=melody_started,
                ended_at=melody_ended,
                inputs={
                    "musicxml": str(musicxml_source) if musicxml_source else None,
                },
                outputs={
                    "melody_json": (str(melody_json_path) if melody_payload is not None else None),
                },
                params={},
                error=melody_error,
            )
            if melody_status == "failed":
                raise RuntimeError(melody_error or "extract_melody failed")

            events_started = _utcnow()
            if normalize_melody_payload is not None:
                events = _build_events_from_melody(
                    item,
                    melody=normalize_melody_payload,
                    chords=chords_payload["chords"],
                    musicxml_chord_source=(
                        "supplied_musicxml"
                        if musicxml_backend.name == "fixture"
                        or "manual_override" in musicxml_inputs
                        else "recognized_musicxml"
                    ),
                )
                normalize_inputs = {
                    "chords_json": str(chords_path),
                    "melody_json": str(melody_json_path),
                }
            else:
                events = _build_stub_events(item, chords=chords_payload["chords"])
                events["chord_source"] = "automatic_ocr" if events["chords"] else "none"
                normalize_inputs = {"chords_json": str(chords_path)}
            events_path = intermediate_dir / "events.json"
            events_path.write_text(json.dumps(events, indent=2) + "\n", encoding="utf-8")
            logger.info("Wrote events: %s", events_path)
            events_ended = _utcnow()
            _write_stage_artifact(
                work_dir=work_dir,
                stage="normalize_events",
                status="success",
                started_at=events_started,
                ended_at=events_ended,
                inputs=normalize_inputs,
                outputs={"events_json": str(events_path)},
                params={"chord_source": events["chord_source"]},
            )

            abc_started = _utcnow()
            abc_text = events_to_abc(events, item.metadata)
            melody_path = final_dir / "melody.abc"
            melody_path.write_text(abc_text, encoding="utf-8")
            logger.info("Wrote ABC: %s", melody_path)

            melody_chords_path = final_dir / "melody_with_chords.abc"
            melody_chords_path.write_text(abc_text, encoding="utf-8")
            logger.info("Wrote ABC with chords: %s", melody_chords_path)
            abc_ended = _utcnow()
            _write_stage_artifact(
                work_dir=work_dir,
                stage="export_abc",
                status="success",
                started_at=abc_started,
                ended_at=abc_ended,
                inputs={"events_json": str(events_path)},
                outputs={
                    "melody_abc": str(melody_path),
                    "melody_with_chords_abc": str(melody_chords_path),
                },
                params={},
            )

            preview_started = _utcnow()
            preview_path = final_dir / "preview.svg"
            render_abc_preview(melody_chords_path, preview_path, logger)
            preview_ended = _utcnow()
            preview_ok = preview_path.exists()
            _write_stage_artifact(
                work_dir=work_dir,
                stage="render_preview",
                status="success" if preview_ok else "failed",
                started_at=preview_started,
                ended_at=preview_ended,
                inputs={"abc_path": str(melody_chords_path)},
                outputs={"preview_svg": str(preview_path)},
                params={},
                error=None if preview_ok else "Preview SVG was not created",
            )
            if not preview_ok:
                raise RuntimeError("Preview SVG was not created")
        except Exception as exc:
            message = str(exc)
            item_status["status"] = "failed"
            item_status["errors"].append(message)
            logger.error("Work item failed: %s (%s)", item.slug, message)

        per_work_status.append(item_status)

    exit_code = 1 if any(status["status"] == "failed" for status in per_work_status) else 0
    _write_command_status(
        out_dir=out_dir,
        command="run",
        per_work=per_work_status,
        extra={
            "workers": workers,
            "use_vlm": use_vlm,
            "musicxml_backend": musicxml_backend_name,
            "audiveris_command": (
                audiveris_command if musicxml_backend_name == "audiveris" else None
            ),
            "audiveris_input": audiveris_input if musicxml_backend_name == "audiveris" else None,
            "homr_command": homr_command if musicxml_backend_name == "homr" else None,
            "homr_input": homr_input if musicxml_backend_name == "homr" else None,
            "slugs": slugs or None,
        },
    )
    return exit_code


def qa(out_dir: Path, open_ui: bool = False) -> int:
    logger = get_logger("score2abc.qa")
    manifest_path = out_dir / "manifest.jsonl"
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        return 1

    work_items = load_manifest_jsonl(manifest_path)
    per_work_status: List[Dict[str, Any]] = []
    for item in work_items:
        item_status: Dict[str, Any] = {
            "slug": item.slug,
            "status": "success",
            "errors": [],
        }
        work_dir = out_dir / item.slug
        stage_started = _utcnow()
        preview_path = out_dir / item.slug / "final" / "preview.svg"
        if preview_path.exists():
            logger.info("Preview available: %s", preview_path)
            stage_status = "success"
            stage_error = None
        else:
            stage_status = "failed"
            stage_error = f"Preview missing: {preview_path}"
            item_status["status"] = "failed"
            item_status["errors"].append(stage_error)
            logger.error(stage_error)

        _write_stage_artifact(
            work_dir=work_dir,
            stage="qa_preview_check",
            status=stage_status,
            started_at=stage_started,
            ended_at=_utcnow(),
            inputs={"preview_svg": str(preview_path)},
            outputs={"preview_exists": preview_path.exists()},
            params={"open_ui": open_ui},
            error=stage_error,
        )
        per_work_status.append(item_status)

    if open_ui:
        logger.info("UI not implemented yet (stub)")

    exit_code = 1 if any(status["status"] == "failed" for status in per_work_status) else 0
    _write_command_status(
        out_dir=out_dir,
        command="qa",
        per_work=per_work_status,
        extra={"open_ui": open_ui},
    )
    return exit_code


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


def evaluate(out_dir: Path, ground_truth_dir: Path) -> int:
    return run_evaluation(out_dir, ground_truth_dir)


_MUSICXML_SOURCE_CANDIDATES = (
    Path("intermediate") / INTERMEDIATE_MUSICXML_FILENAME,
    Path("intermediate") / "musicxml.musicxml",
)


def _find_musicxml_source(work_dir: Path) -> Path | None:
    """Locate a MusicXML file to feed the extract_melody stage.

    Until OMR is integrated, callers can drop a MusicXML at one of the candidate
    paths under the work directory to exercise the melody pipeline end-to-end.
    """
    for relative in _MUSICXML_SOURCE_CANDIDATES:
        candidate = work_dir / relative
        if candidate.exists():
            return candidate
    return None


def _build_events_from_melody(
    item: WorkItem,
    *,
    melody: Dict[str, Any],
    chords: List[Dict[str, Any]],
    musicxml_chord_source: str = "supplied_musicxml",
) -> dict:
    """Prefer MusicXML harmonies; use OCR only when MusicXML supplies none."""
    xml_chords = melody.get("chords") or []
    selected_chords = xml_chords or chords
    chord_source = musicxml_chord_source if xml_chords else "automatic_ocr" if chords else "none"
    chord_events = [
        {
            "measure": entry["measure"],
            "onset_beats": entry.get("onset_beats", 0.0),
            "symbol": entry["symbol"],
        }
        for entry in selected_chords
    ]
    time_signature = melody.get("time_signature") or item.metadata.time_signature or "4/4"
    return {
        "time_signature": time_signature,
        "notes": list(melody.get("notes") or []),
        "chords": chord_events,
        "chord_source": chord_source,
    }


def _build_stub_events(item: WorkItem, *, chords: List[Dict[str, Any]] | None = None) -> dict:
    if chords is None:
        chord_events = [
            {
                "measure": 1,
                "onset_beats": 0.0,
                "symbol": item.metadata.key_hint or "C",
            }
        ]
    else:
        chord_events = [
            {
                "measure": entry["measure"],
                "onset_beats": entry.get("onset_beats", 0.0),
                "symbol": entry["symbol"],
            }
            for entry in chords
        ]

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
        "chords": chord_events,
    }


def _write_stage_artifact(
    *,
    work_dir: Path,
    stage: str,
    status: str,
    started_at: str,
    ended_at: str,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    params: Dict[str, Any],
    error: str | None = None,
) -> None:
    started_dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    ended_dt = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
    duration_seconds = max(0.0, (ended_dt - started_dt).total_seconds())
    payload: Dict[str, Any] = {
        "stage": stage,
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": round(duration_seconds, 6),
        "inputs": inputs,
        "outputs": outputs,
        "params": params,
    }
    if error:
        payload["error"] = error

    stages_dir = work_dir / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)
    stage_path = stages_dir / f"{stage}.json"
    stage_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_command_status(
    *,
    out_dir: Path,
    command: str,
    per_work: List[Dict[str, Any]],
    extra: Dict[str, Any],
) -> None:
    status_path = out_dir / f"{command}_status.json"
    failures = [item for item in per_work if item["status"] == "failed"]
    payload: Dict[str, Any] = {
        "command": command,
        "timestamp": _utcnow(),
        "summary": {
            "total": len(per_work),
            "succeeded": len(per_work) - len(failures),
            "failed": len(failures),
        },
        "per_work": per_work,
        "extra": extra,
    }
    status_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _utcnow() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
