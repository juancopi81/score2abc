"""Record VLM melody-transcription fixtures from measure input crops.

This is spike tooling only. It does not wire VLM transcription into the main
pipeline. It reads records produced by `scripts/build_vlm_melody_inputs.py`,
calls a live Gemini model for selected measure crops, and stores replayable
fixtures under `.cache/vlm_melody/` by default. Gemini and OpenAI are both
available as provider choices; this script is intentionally outside the main
pipeline.

Example:
    uv run python scripts/record_vlm_melody_fixtures.py out \\
        --slug jaime-llanos_12_aviador_pasillo_fulgencio-garcia \\
        --system 1 --input-kind staff --model gemini-3.1-flash-lite --max-calls 8

    uv run python scripts/record_vlm_melody_fixtures.py out \\
        --slug jaime-llanos_12_aviador_pasillo_fulgencio-garcia \\
        --system 1 --measure 1 --input-kind all \\
        --provider openai --model gpt-5.5 --max-calls 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from score2abc.melody.vlm import (
    DEFAULT_MELODY_VLM_MODEL,
    DEFAULT_OPENAI_MELODY_VLM_MODEL,
    INPUT_KINDS,
    MELODY_VLM_PROMPT_VERSION,
    NOTEHEAD_Y_PROMPT_VERSION,
    STAFF_POSITION_PROMPT_VERSION,
    TRANSCRIPTION_MODES,
    VLM_PROVIDERS,
    GeminiMelodyVLM,
    InputKind,
    MelodyVLMRequest,
    OpenAIMelodyVLM,
    TranscriptionMode,
    VLMProvider,
    default_model_for_provider,
    fixture_model_id,
    melody_fixture_key,
    write_melody_fixture,
)
from score2abc.utils import get_logger

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES_DIR = REPO_ROOT / ".cache" / "vlm_melody"
INPUT_KIND_TO_PATH_KEY: dict[InputKind, str] = {
    "raw": "measure_raw",
    "staff": "measure_staff",
    "staff_overlay": "measure_staff_overlay",
    "pitch_ruler": "measure_pitch_ruler",
    "pitch_ruler_soft": "measure_pitch_ruler_soft",
    "pitch_ruler_panel": "measure_pitch_ruler_panel",
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.model = args.model or default_model_for_provider(args.provider)
    logger = get_logger("score2abc.record_vlm_melody_fixtures")

    records = list(
        _selected_records(
            _load_manifest(args.out_dir),
            selected_slugs=set(args.slug) if args.slug else None,
            selected_systems=set(args.system) if args.system else None,
            selected_measures=set(args.measure) if args.measure else None,
        )
    )
    selected_input_kinds = INPUT_KINDS if args.input_kind == "all" else (args.input_kind,)
    planned = len(records) * len(selected_input_kinds)
    logger.info(
        "Selected %d measure records x %d input kind(s) = %d fixture target(s)",
        len(records),
        len(selected_input_kinds),
        planned,
    )

    fixtures_dir: Path = args.fixtures_dir
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    transcriber = None
    if args.max_calls > 0:
        transcriber = _build_transcriber(
            provider=args.provider,
            model=args.model,
            transcription_mode=args.transcription_mode,
            openai_reasoning_effort=args.openai_reasoning_effort,
        )

    written, skipped, would_call = record_fixtures(
        records,
        out_dir=args.out_dir,
        fixtures_dir=fixtures_dir,
        input_kinds=selected_input_kinds,
        model_id=fixture_model_id(
            args.provider,
            args.model,
            openai_reasoning_effort=args.openai_reasoning_effort,
        ),
        transcription_mode=args.transcription_mode,
        transcriber=transcriber,
        max_calls=args.max_calls,
        force=args.force,
        context_overrides={
            "clef_hint": args.clef_hint,
            "time_signature_hint": args.time_signature_hint,
            "key_hint": args.key_hint,
            "expected_measure_beats": args.expected_measure_beats,
        },
        logger=logger,
    )
    logger.info(
        "Done. wrote=%d skipped=%d would_call=%d fixtures_dir=%s",
        written,
        skipped,
        would_call,
        fixtures_dir,
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="Pipeline output directory.")
    parser.add_argument("--slug", action="append", default=None, help="Limit to a work slug.")
    parser.add_argument(
        "--system", action="append", type=int, default=None, help="Limit to a system."
    )
    parser.add_argument(
        "--measure",
        action="append",
        type=int,
        default=None,
        help="Limit to a system-local measure index.",
    )
    parser.add_argument(
        "--input-kind",
        choices=(*INPUT_KINDS, "all"),
        default="staff",
        help="Crop variant to send. Defaults to staff.",
    )
    parser.add_argument(
        "--provider",
        choices=VLM_PROVIDERS,
        default="gemini",
        help="Live VLM provider to call. Defaults to gemini.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Provider model to call. Defaults to "
            f"{DEFAULT_MELODY_VLM_MODEL} for Gemini and {DEFAULT_OPENAI_MELODY_VLM_MODEL} "
            "for OpenAI."
        ),
    )
    parser.add_argument(
        "--transcription-mode",
        choices=TRANSCRIPTION_MODES,
        default="pitch",
        help="Prompt/schema mode to use. Defaults to direct pitch transcription.",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=DEFAULT_FIXTURES_DIR,
        help="Fixture/cache destination. Defaults to .cache/vlm_melody.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing fixtures.")
    parser.add_argument("--clef-hint", default=None, help="Override the context clef hint.")
    parser.add_argument(
        "--time-signature-hint",
        default=None,
        help="Override the context time-signature hint, e.g. 3/4.",
    )
    parser.add_argument("--key-hint", default=None, help="Override the context key hint.")
    parser.add_argument(
        "--expected-measure-beats",
        default=None,
        help="Override the expected measure duration in beats.",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=0,
        help="Maximum live VLM calls. Defaults to 0 for dry-run/safety.",
    )
    parser.add_argument(
        "--openai-reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh"),
        default="none",
        help=(
            "OpenAI reasoning effort. Defaults to none to match prior spike runs; "
            "non-default values are namespaced in fixture keys."
        ),
    )
    parser.set_defaults(model=None)
    return parser


def record_fixtures(
    records: Iterable[dict[str, Any]],
    *,
    out_dir: Path,
    fixtures_dir: Path,
    input_kinds: tuple[InputKind, ...],
    model_id: str,
    transcription_mode: TranscriptionMode,
    transcriber: GeminiMelodyVLM | OpenAIMelodyVLM | None,
    max_calls: int,
    force: bool,
    logger,
    context_overrides: dict[str, Any] | None = None,
) -> tuple[int, int, int]:
    written = 0
    skipped = 0
    would_call = 0
    calls = 0

    for record in records:
        context_path = _resolve_path(out_dir, record["paths"]["context"])
        context = json.loads(context_path.read_text(encoding="utf-8"))
        context = _apply_context_overrides(context, context_overrides)
        for input_kind in input_kinds:
            image_path = _resolve_path(out_dir, record["paths"][INPUT_KIND_TO_PATH_KEY[input_kind]])
            key = melody_fixture_key(
                image_path,
                prompt_version=(
                    transcriber.prompt_version
                    if transcriber
                    else _prompt_version_for_mode(transcription_mode)
                ),
                model_id=model_id,
                input_kind=input_kind,
                context=context,
            )
            target = fixtures_dir / f"{key}.json"
            if target.exists() and not force:
                logger.info("Skip (exists): %s %s -> %s", input_kind, image_path.name, target.name)
                skipped += 1
                continue
            if transcriber is None or calls >= max_calls:
                logger.info("Would call: %s %s -> %s", input_kind, image_path.name, target.name)
                would_call += 1
                continue

            request = MelodyVLMRequest(
                image_path=image_path,
                context=context,
                input_kind=input_kind,
                transcription_mode=transcription_mode,
            )
            transcription = transcriber.transcribe(request)
            write_melody_fixture(
                target,
                image_path=image_path,
                context_path=context_path,
                context=context,
                input_kind=input_kind,
                prompt_version=transcriber.prompt_version,
                model_id=model_id,
                transcription=transcription,
            )
            logger.info(
                "Wrote %d item(s): %s %s -> %s",
                len(transcription.items),
                input_kind,
                image_path.name,
                target.name,
            )
            written += 1
            calls += 1

    return written, skipped, would_call


def _build_transcriber(
    *,
    provider: VLMProvider,
    model: str,
    transcription_mode: TranscriptionMode,
    openai_reasoning_effort: str = "none",
) -> GeminiMelodyVLM | OpenAIMelodyVLM:
    if provider == "openai":
        return OpenAIMelodyVLM(
            model=model,
            transcription_mode=transcription_mode,
            reasoning_effort=openai_reasoning_effort,
        )
    return GeminiMelodyVLM(model=model, transcription_mode=transcription_mode)


def _apply_context_overrides(
    context: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    if not overrides:
        return context
    updated = dict(context)
    for key, value in overrides.items():
        if value is not None:
            updated[key] = value
    return updated


def _prompt_version_for_mode(transcription_mode: TranscriptionMode) -> str:
    if transcription_mode == "notehead_y":
        return NOTEHEAD_Y_PROMPT_VERSION
    if transcription_mode == "staff_position":
        return STAFF_POSITION_PROMPT_VERSION
    return MELODY_VLM_PROMPT_VERSION


def _load_manifest(out_dir: Path) -> list[dict[str, Any]]:
    manifest_path = out_dir / "vlm_melody_inputs_manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"VLM melody input manifest not found: {manifest_path}. "
            "Run scripts/build_vlm_melody_inputs.py first."
        )
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pitch_ruler_manifest_path = out_dir / "vlm_pitch_ruler_inputs_manifest.jsonl"
    if pitch_ruler_manifest_path.exists():
        pitch_ruler_by_key = {
            _manifest_record_key(record): record
            for record in (
                json.loads(line)
                for line in pitch_ruler_manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
        merged_records = []
        for record in records:
            pitch_ruler_record = pitch_ruler_by_key.get(_manifest_record_key(record))
            if pitch_ruler_record is None:
                merged_records.append(record)
                continue
            next_record = dict(record)
            next_paths = dict(record["paths"])
            next_paths.update(pitch_ruler_record.get("paths", {}))
            next_record["paths"] = next_paths
            if "pitch_ruler" in pitch_ruler_record:
                next_record["pitch_ruler"] = pitch_ruler_record["pitch_ruler"]
            merged_records.append(next_record)
        return merged_records
    return records


def _manifest_record_key(record: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(record["slug"]),
        int(record["system_index"]),
        int(record["system_measure_index"]),
        int(record["global_measure_index"]),
    )


def _selected_records(
    records: Iterable[dict[str, Any]],
    *,
    selected_slugs: set[str] | None,
    selected_systems: set[int] | None,
    selected_measures: set[int] | None,
) -> Iterable[dict[str, Any]]:
    for record in records:
        if selected_slugs is not None and record["slug"] not in selected_slugs:
            continue
        if selected_systems is not None and int(record["system_index"]) not in selected_systems:
            continue
        if (
            selected_measures is not None
            and int(record["system_measure_index"]) not in selected_measures
        ):
            continue
        yield record


def _resolve_path(out_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = out_dir / path
    return candidate


if __name__ == "__main__":
    raise SystemExit(main())
