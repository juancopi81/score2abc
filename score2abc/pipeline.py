from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from score2abc.abc import events_to_abc
from score2abc.dataset import load_dataset_metadata
from score2abc.evaluation import evaluate as run_evaluation
from score2abc.manifest import load_manifest_jsonl, write_manifest_jsonl
from score2abc.render import create_minimal_system_crops, render_abc_preview, render_pdf_to_images
from score2abc.schemas import WorkItem
from score2abc.utils import Timer, get_logger

DEFAULT_DPI = 300


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
        item_status: Dict[str, Any] = {"slug": item.slug, "status": "success", "errors": []}
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
                inputs={"pdf_path": str(item.pdf_path), "metadata_csv": str(metadata_csv)},
                outputs={"source_pdf": str(source_pdf), "metadata_json": str(metadata_path)},
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
                inputs={"pdf_path": str(item.pdf_path), "metadata_csv": str(metadata_csv)},
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


def run(out_dir: Path, workers: int = 1, use_vlm: bool = False) -> int:
    logger = get_logger("score2abc.run")
    manifest_path = out_dir / "manifest.jsonl"
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        return 1

    work_items = load_manifest_jsonl(manifest_path)
    logger.info("Running %d work items (workers=%d, use_vlm=%s)", len(work_items), workers, use_vlm)
    per_work_status: List[Dict[str, Any]] = []

    for item in work_items:
        item_status: Dict[str, Any] = {"slug": item.slug, "status": "success", "errors": []}
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
                system_outputs = create_minimal_system_crops(page_paths, systems_dir, logger)
            segment_ended = _utcnow()
            _write_stage_artifact(
                work_dir=work_dir,
                stage="segment_systems",
                status="success" if system_outputs else "failed",
                started_at=segment_started,
                ended_at=segment_ended,
                inputs={"pages": [str(p) for p in page_paths]},
                outputs={"system_crops": [str(p) for p in system_outputs]},
                params={},
                error=None if system_outputs else "No system crops generated",
            )
            if not system_outputs:
                raise RuntimeError("No system crops were generated")

            events_started = _utcnow()
            events = _build_stub_events(item)
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
                inputs={},
                outputs={"events_json": str(events_path)},
                params={},
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
        extra={"workers": workers, "use_vlm": use_vlm},
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
        item_status: Dict[str, Any] = {"slug": item.slug, "status": "success", "errors": []}
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
