"""Snapshot one VLM melody experiment into a reproducible journal folder.

This is spike tooling. It does not call a model. It records the exact image,
prompt text, model/provider config, fixture, and eval result for a previously
recorded VLM melody fixture.

Example:
    uv run python scripts/journal_vlm_melody_experiment.py out \\
        --slug jaime-llanos_12_aviador_pasillo_fulgencio-garcia \\
        --system 1 --measure 3 \\
        --input-kind pitch_ruler_panel \\
        --provider openai --model gpt-5.5 \\
        --transcription-mode pitch --openai-reasoning-effort medium
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from score2abc.melody.vlm import (  # noqa: E402
    DEFAULT_MELODY_VLM_MODEL,
    DEFAULT_OPENAI_MELODY_VLM_MODEL,
    INPUT_KINDS,
    TRANSCRIPTION_MODES,
    VLM_PROVIDERS,
    InputKind,
    MelodyVLMRequest,
    TranscriptionMode,
    VLMProvider,
    _format_user_message,
    _system_prompt,
    default_model_for_provider,
    fixture_model_id,
)
from score2abc.utils import get_logger  # noqa: E402
from scripts.eval_vlm_melody_fixtures import (  # noqa: E402
    _safe_name,
    evaluate_records,
    melody_fixture_key_for_model,
)
from scripts.record_vlm_melody_fixtures import (  # noqa: E402
    DEFAULT_FIXTURES_DIR,
    INPUT_KIND_TO_PATH_KEY,
    _apply_context_overrides,
    _load_manifest,
    _resolve_path,
    _selected_records,
)

DEFAULT_GROUND_TRUTH_DIR = Path("dataset/ground_truth")
DEFAULT_JOURNAL_DIR_NAME = "vlm_melody_experiments"


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.model = args.model or default_model_for_provider(args.provider)
    logger = get_logger("score2abc.journal_vlm_melody_experiment")

    try:
        journal_path = journal_experiment(
            out_dir=args.out_dir,
            slug=args.slug,
            system_index=args.system,
            system_measure_index=args.measure,
            input_kind=args.input_kind,
            provider=args.provider,
            model=args.model,
            transcription_mode=args.transcription_mode,
            fixtures_dir=args.fixtures_dir,
            ground_truth_dir=args.ground_truth,
            openai_reasoning_effort=args.openai_reasoning_effort,
            journal_root=args.journal_dir,
            run_id=args.run_id,
            notes=args.notes,
            context_overrides={
                "clef_hint": args.clef_hint,
                "time_signature_hint": args.time_signature_hint,
                "key_hint": args.key_hint,
                "expected_measure_beats": args.expected_measure_beats,
            },
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Wrote VLM melody experiment journal: %s", journal_path)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="Pipeline output directory.")
    parser.add_argument("--slug", required=True, help="Work slug.")
    parser.add_argument("--system", required=True, type=int, help="System index.")
    parser.add_argument(
        "--measure",
        required=True,
        type=int,
        help="System-local measure index.",
    )
    parser.add_argument(
        "--input-kind",
        choices=INPUT_KINDS,
        required=True,
        help="Crop variant used for the VLM call.",
    )
    parser.add_argument(
        "--provider",
        choices=VLM_PROVIDERS,
        required=True,
        help="VLM provider used for the fixture.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Provider model id. Defaults to "
            f"{DEFAULT_MELODY_VLM_MODEL} for Gemini and {DEFAULT_OPENAI_MELODY_VLM_MODEL} "
            "for OpenAI."
        ),
    )
    parser.add_argument(
        "--transcription-mode",
        choices=TRANSCRIPTION_MODES,
        default="pitch",
        help="Prompt/schema mode used by the fixture.",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=DEFAULT_FIXTURES_DIR,
        help="Fixture/cache directory. Defaults to .cache/vlm_melody.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_DIR,
        help="Ground-truth directory.",
    )
    parser.add_argument(
        "--journal-dir",
        type=Path,
        default=None,
        help="Destination root. Defaults to <out_dir>/vlm_melody_experiments.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional stable journal folder name. Defaults to timestamp plus experiment identity.",
    )
    parser.add_argument("--notes", default="", help="Optional human notes for this run.")
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
        "--openai-reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh"),
        default="none",
        help="OpenAI reasoning effort used when the fixture was recorded.",
    )
    parser.set_defaults(model=None)
    return parser


def journal_experiment(
    *,
    out_dir: Path,
    slug: str,
    system_index: int,
    system_measure_index: int,
    input_kind: InputKind,
    provider: VLMProvider,
    model: str,
    transcription_mode: TranscriptionMode,
    fixtures_dir: Path,
    ground_truth_dir: Path,
    openai_reasoning_effort: str = "none",
    journal_root: Path | None = None,
    run_id: str | None = None,
    notes: str = "",
    context_overrides: dict[str, Any] | None = None,
) -> Path:
    records = list(
        _selected_records(
            _load_manifest(out_dir),
            selected_slugs={slug},
            selected_systems={system_index},
            selected_measures={system_measure_index},
        )
    )
    if len(records) != 1:
        raise ValueError(
            "Expected exactly one VLM melody record for "
            f"slug={slug!r} system={system_index} measure={system_measure_index}; "
            f"found {len(records)}"
        )
    record = records[0]
    context_path = _resolve_path(out_dir, record["paths"]["context"])
    context = json.loads(context_path.read_text(encoding="utf-8"))
    context = _apply_context_overrides(context, context_overrides)
    image_path = _resolve_path(out_dir, record["paths"][INPUT_KIND_TO_PATH_KEY[input_kind]])
    model_id = fixture_model_id(
        provider,
        model,
        openai_reasoning_effort=openai_reasoning_effort,
    )
    fixture_key = melody_fixture_key_for_model(
        image_path,
        model_id,
        input_kind,
        context,
        transcription_mode,
    )
    fixture_path = fixtures_dir / f"{fixture_key}.json"

    report = evaluate_records(
        [record],
        out_dir=out_dir,
        ground_truth_dir=ground_truth_dir,
        fixtures_dir=fixtures_dir,
        input_kinds=(input_kind,),
        model_id=model_id,
        transcription_mode=transcription_mode,
        context_overrides=context_overrides,
    )
    prompt_request = MelodyVLMRequest(
        image_path=image_path,
        context=context,
        input_kind=input_kind,
        transcription_mode=transcription_mode,
    )
    system_prompt = _system_prompt(transcription_mode, input_kind)
    user_prompt = _format_user_message(prompt_request)

    journal_root = journal_root or out_dir / DEFAULT_JOURNAL_DIR_NAME
    experiment_id = run_id or _default_run_id(
        slug=slug,
        system_index=system_index,
        system_measure_index=system_measure_index,
        provider=provider,
        model=model,
        input_kind=input_kind,
        transcription_mode=transcription_mode,
        openai_reasoning_effort=openai_reasoning_effort,
    )
    journal_path = journal_root / experiment_id
    artifacts_dir = journal_path / "artifacts"
    prompts_dir = journal_path / "prompts"
    journal_path.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(exist_ok=True)
    prompts_dir.mkdir(exist_ok=True)

    copied_image_path = artifacts_dir / f"input{image_path.suffix or '.png'}"
    copied_context_path = artifacts_dir / "context.json"
    shutil.copy2(image_path, copied_image_path)
    shutil.copy2(context_path, copied_context_path)
    copied_fixture_path = None
    if fixture_path.exists():
        copied_fixture_path = artifacts_dir / "fixture.json"
        shutil.copy2(fixture_path, copied_fixture_path)

    (prompts_dir / "system.txt").write_text(system_prompt, encoding="utf-8")
    (prompts_dir / "user.txt").write_text(user_prompt, encoding="utf-8")
    (journal_path / "eval_result.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    experiment = _experiment_payload(
        record=record,
        context=context,
        out_dir=out_dir,
        slug=slug,
        system_index=system_index,
        system_measure_index=system_measure_index,
        input_kind=input_kind,
        provider=provider,
        model=model,
        model_id=model_id,
        transcription_mode=transcription_mode,
        openai_reasoning_effort=openai_reasoning_effort,
        notes=notes,
        image_path=image_path,
        context_path=context_path,
        fixture_path=fixture_path,
        fixture_key=fixture_key,
        copied_image_path=copied_image_path,
        copied_context_path=copied_context_path,
        copied_fixture_path=copied_fixture_path,
        report=report,
        system_prompt_path=prompts_dir / "system.txt",
        user_prompt_path=prompts_dir / "user.txt",
    )
    (journal_path / "experiment.json").write_text(
        json.dumps(experiment, indent=2) + "\n",
        encoding="utf-8",
    )
    (journal_path / "README.md").write_text(
        _readme_text(experiment, report),
        encoding="utf-8",
    )
    return journal_path


def _experiment_payload(
    *,
    record: dict[str, Any],
    context: dict[str, Any],
    out_dir: Path,
    slug: str,
    system_index: int,
    system_measure_index: int,
    input_kind: InputKind,
    provider: VLMProvider,
    model: str,
    model_id: str,
    transcription_mode: TranscriptionMode,
    openai_reasoning_effort: str,
    notes: str,
    image_path: Path,
    context_path: Path,
    fixture_path: Path,
    fixture_key: str,
    copied_image_path: Path,
    copied_context_path: Path,
    copied_fixture_path: Path | None,
    report: dict[str, Any],
    system_prompt_path: Path,
    user_prompt_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
        "identity": {
            "slug": slug,
            "system_index": system_index,
            "system_measure_index": system_measure_index,
            "global_measure_index": record.get("global_measure_index"),
            "display_measure_number": record.get("display_measure_number"),
        },
        "config": {
            "provider": provider,
            "model": model,
            "model_id": model_id,
            "input_kind": input_kind,
            "transcription_mode": transcription_mode,
            "openai_reasoning_effort": openai_reasoning_effort,
            "fixture_key": fixture_key,
        },
        "paths": {
            "out_dir": str(out_dir),
            "source_image": str(image_path),
            "source_context": str(context_path),
            "source_fixture": str(fixture_path),
            "journal_input_image": str(copied_image_path),
            "journal_context": str(copied_context_path),
            "journal_fixture": str(copied_fixture_path) if copied_fixture_path else None,
            "system_prompt": str(system_prompt_path),
            "user_prompt": str(user_prompt_path),
            "eval_result": "eval_result.json",
        },
        "context": context,
        "eval_summary": report.get("summary", {}),
        "eval_result": (report.get("results") or [{}])[0],
        "git": _git_snapshot(),
        "replay_commands": _replay_commands(
            out_dir=out_dir,
            slug=slug,
            system_index=system_index,
            system_measure_index=system_measure_index,
            input_kind=input_kind,
            provider=provider,
            model=model,
            transcription_mode=transcription_mode,
            openai_reasoning_effort=openai_reasoning_effort,
        ),
    }


def _default_run_id(
    *,
    slug: str,
    system_index: int,
    system_measure_index: int,
    provider: VLMProvider,
    model: str,
    input_kind: InputKind,
    transcription_mode: TranscriptionMode,
    openai_reasoning_effort: str,
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    reasoning = ""
    if provider == "openai" and openai_reasoning_effort != "none":
        reasoning = f"_reasoning-{_safe_name(openai_reasoning_effort)}"
    return (
        f"{timestamp}_{_safe_name(slug)}_s{system_index:03d}_m{system_measure_index:03d}_"
        f"{provider}_{_safe_name(model)}{reasoning}_{transcription_mode}_{input_kind}"
    )


def _git_snapshot() -> dict[str, Any]:
    return {
        "branch": _git_output("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": _git_output("rev-parse", "HEAD"),
        "dirty": bool(_git_output("status", "--porcelain")),
    }


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _replay_commands(
    *,
    out_dir: Path,
    slug: str,
    system_index: int,
    system_measure_index: int,
    input_kind: InputKind,
    provider: VLMProvider,
    model: str,
    transcription_mode: TranscriptionMode,
    openai_reasoning_effort: str,
) -> dict[str, str]:
    base_args = (
        f'{out_dir} --slug "{slug}" --system {system_index} --measure {system_measure_index} '
        f"--input-kind {input_kind} --provider {provider} --model {_shell_quote(model)} "
        f"--transcription-mode {transcription_mode}"
    )
    reasoning = ""
    if provider == "openai":
        reasoning = f" --openai-reasoning-effort {openai_reasoning_effort}"
    return {
        "record_fixture": (
            "uv run python scripts/record_vlm_melody_fixtures.py "
            f"{base_args}{reasoning} --max-calls 1 --force"
        ),
        "eval_fixture": (
            "uv run python scripts/eval_vlm_melody_fixtures.py " f"{base_args}{reasoning}"
        ),
        "journal": (
            "uv run python scripts/journal_vlm_melody_experiment.py " f"{base_args}{reasoning}"
        ),
    }


def _shell_quote(value: str) -> str:
    if value.replace("-", "").replace("_", "").replace(".", "").isalnum():
        return value
    return json.dumps(value)


def _readme_text(experiment: dict[str, Any], report: dict[str, Any]) -> str:
    result = (report.get("results") or [{}])[0]
    return (
        f"# VLM Melody Experiment\n\n"
        f"- Slug: `{experiment['identity']['slug']}`\n"
        f"- System / measure: `{experiment['identity']['system_index']}` / "
        f"`{experiment['identity']['system_measure_index']}`\n"
        f"- Provider/model: `{experiment['config']['provider']}` / "
        f"`{experiment['config']['model']}`\n"
        f"- Input kind: `{experiment['config']['input_kind']}`\n"
        f"- Transcription mode: `{experiment['config']['transcription_mode']}`\n"
        f"- Reasoning effort: `{experiment['config']['openai_reasoning_effort']}`\n\n"
        "## Result\n\n"
        f"- Status: `{result.get('status')}`\n"
        f"- Predicted pitches: `{result.get('pred_pitches')}`\n"
        f"- Truth pitches: `{result.get('truth_pitches')}`\n"
        f"- Pitch accuracy: `{result.get('pitch_order_accuracy')}`\n"
        f"- Duration accuracy: `{result.get('duration_order_accuracy')}`\n\n"
        "## Files\n\n"
        "- `artifacts/input.png`: exact image sent to the model.\n"
        "- `prompts/system.txt`: exact system/developer prompt text.\n"
        "- `prompts/user.txt`: exact user prompt text and compact context.\n"
        "- `artifacts/fixture.json`: raw model response fixture, when available.\n"
        "- `eval_result.json`: local eval comparison against GT.\n"
        "- `experiment.json`: machine-readable config and replay commands.\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
