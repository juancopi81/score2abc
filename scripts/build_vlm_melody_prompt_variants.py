"""Build spike-only prompt variants for VLM melody image experiments.

This script does not call a VLM or any network API. It reads image variant
records from `scripts/build_vlm_melody_variants.py` and writes prompt artifacts
that can be reviewed before spending live model calls.

Each generated record associates one exact image variant with one prompt id:

    variant_id + prompt_id -> image_path + system/user prompt + optional schema

If the image variant manifest is missing or does not contain the selected
records, this script can build variants first by importing
`build_vlm_melody_variants(...)`. This remains spike tooling; the main pipeline
does not consume these prompts.

Example:
    uv run python scripts/build_vlm_melody_prompt_variants.py out \\
        --slug jaime-llanos_12_aviador_pasillo_fulgencio-garcia \\
        --system 1 --measure 3 --variant all --prompt all
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from score2abc.melody.vlm import (  # noqa: E402
    INPUT_KINDS,
    PITCH_RESPONSE_SCHEMA,
    PITCH_RULER_SOFT_RESPONSE_SCHEMA,
    InputKind,
    TranscriptionMode,
)
from score2abc.utils import get_logger  # noqa: E402
from scripts.build_vlm_melody_variants import (  # noqa: E402
    VARIANT_IDS,
    VARIANTS_MANIFEST_NAME,
    build_vlm_melody_variants,
)

PROMPT_VARIANTS_ROOT_NAME = "vlm_melody_prompt_variants"
PROMPT_VARIANTS_MANIFEST_NAME = "vlm_melody_prompt_variants_manifest.jsonl"
PER_WORK_PROMPT_VARIANTS_MANIFEST_NAME = "prompt_variants_manifest.jsonl"
OutputMode = Literal["json_schema", "free_response"]

NOTEHEAD_COUNT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "notehead_count": {"type": "integer"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "kind": {"type": "string"},
                    "pitch": {"type": "string"},
                    "duration": {"type": "string"},
                    "accidental": {"type": "string"},
                    "confidence": {"type": "number"},
                    "evidence": {"type": "string"},
                },
                "required": [
                    "kind",
                    "pitch",
                    "duration",
                    "accidental",
                    "confidence",
                    "evidence",
                ],
            },
        },
        "comments": {"type": "string"},
        "overall_confidence": {"type": "number"},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "notehead_count",
        "items",
        "comments",
        "overall_confidence",
        "uncertainties",
    ],
}

COMPACT_CONTEXT_KEYS = (
    "title",
    "rhythm",
    "clef_hint",
    "time_signature_hint",
    "key_hint",
    "system_index",
    "system_measure_index",
    "display_measure_number",
    "allow_pickup",
    "expected_measure_beats",
)


@dataclass(frozen=True)
class PromptSpec:
    prompt_id: str
    compatible_input_kinds: tuple[InputKind, ...]
    transcription_mode: TranscriptionMode
    output_mode: OutputMode
    description: str


PROMPT_REGISTRY: tuple[PromptSpec, ...] = (
    PromptSpec(
        prompt_id="direct_pitch_v0",
        compatible_input_kinds=("raw", "staff"),
        transcription_mode="pitch",
        output_mode="json_schema",
        description="Minimal direct pitch transcription for clean crops.",
    ),
    PromptSpec(
        prompt_id="direct_pitch_best_effort_v1",
        compatible_input_kinds=("raw", "staff"),
        transcription_mode="pitch",
        output_mode="json_schema",
        description="Best-effort direct pitch transcription with evidence and uncertainty.",
    ),
    PromptSpec(
        prompt_id="educated_pitch_v2",
        compatible_input_kinds=INPUT_KINDS,
        transcription_mode="pitch",
        output_mode="json_schema",
        description=(
            "Educated best-effort transcription that prefers low-confidence notes over omission."
        ),
    ),
    PromptSpec(
        prompt_id="staff_overlay_explained_v1",
        compatible_input_kinds=("staff_overlay",),
        transcription_mode="pitch",
        output_mode="json_schema",
        description="Explains red staff-line helpers on staff-overlay crops.",
    ),
    PromptSpec(
        prompt_id="pitch_ruler_standard_explained_v1",
        compatible_input_kinds=("pitch_ruler",),
        transcription_mode="pitch",
        output_mode="json_schema",
        description="Explains strong colored pitch guide lines.",
    ),
    PromptSpec(
        prompt_id="pitch_ruler_soft_explained_v1",
        compatible_input_kinds=("pitch_ruler_soft",),
        transcription_mode="pitch",
        output_mode="json_schema",
        description="Explains faint dotted pitch guide lines.",
    ),
    PromptSpec(
        prompt_id="pitch_ruler_panel_explained_v1",
        compatible_input_kinds=("pitch_ruler_panel",),
        transcription_mode="pitch",
        output_mode="json_schema",
        description="Explains left pitch-reference gutter and clean crop on the right.",
    ),
    PromptSpec(
        prompt_id="neighbor_context_transcribe_v1",
        compatible_input_kinds=("neighbor_context",),
        transcription_mode="pitch",
        output_mode="json_schema",
        description=(
            "Transcribes only the margin-delimited target while using adjacent measures "
            "to learn the handwriting."
        ),
    ),
    PromptSpec(
        prompt_id="notehead_count_then_pitch_v1",
        compatible_input_kinds=INPUT_KINDS,
        transcription_mode="pitch",
        output_mode="json_schema",
        description="Forces a count-first pass before pitch and duration transcription.",
    ),
    PromptSpec(
        prompt_id="free_response_describe_v1",
        compatible_input_kinds=INPUT_KINDS,
        transcription_mode="pitch",
        output_mode="free_response",
        description="Debug prompt asking the model to describe what it sees.",
    ),
    PromptSpec(
        prompt_id="describe_then_guess_v2",
        compatible_input_kinds=INPUT_KINDS,
        transcription_mode="pitch",
        output_mode="free_response",
        description="Debug prompt asking for visual description plus final educated transcription.",
    ),
)
PROMPT_IDS = tuple(spec.prompt_id for spec in PROMPT_REGISTRY)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logger = get_logger("score2abc.build_vlm_melody_prompt_variants")

    try:
        records = build_vlm_melody_prompt_variants(
            args.out_dir,
            selected_slugs=set(args.slug) if args.slug else None,
            selected_systems=set(args.system) if args.system else None,
            selected_measures=set(args.measure) if args.measure else None,
            variant_ids=tuple(args.variant) if args.variant else None,
            prompt_ids=tuple(args.prompt) if args.prompt else None,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Wrote %d VLM melody prompt variant records", len(records))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="Pipeline output directory.")
    parser.add_argument("--slug", action="append", default=None, help="Limit to a work slug.")
    parser.add_argument(
        "--system",
        action="append",
        type=int,
        default=None,
        help="Limit to a 1-based system index.",
    )
    parser.add_argument(
        "--measure",
        action="append",
        type=int,
        default=None,
        help="Limit to a system-local measure index.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=(*VARIANT_IDS, "all"),
        default=None,
        help="Image variant id to pair with prompts. Repeat for subsets; defaults to all.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        choices=(*PROMPT_IDS, "all"),
        default=None,
        help="Prompt id to render. Repeat for subsets; defaults to all compatible prompts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite prompt files. Manifests are always rewritten.",
    )
    return parser


def build_vlm_melody_prompt_variants(
    out_dir: Path,
    *,
    selected_slugs: set[str] | None = None,
    selected_systems: set[int] | None = None,
    selected_measures: set[int] | None = None,
    variant_ids: tuple[str, ...] | None = None,
    prompt_ids: tuple[str, ...] | None = None,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Create prompt files and prompt-image association manifests."""
    prompt_specs = _selected_prompt_specs(prompt_ids)
    variant_records = _load_or_build_variant_records(
        out_dir,
        selected_slugs=selected_slugs,
        selected_systems=selected_systems,
        selected_measures=selected_measures,
        variant_ids=variant_ids,
    )

    records: list[dict[str, Any]] = []
    for variant_record in variant_records:
        for prompt_spec in prompt_specs:
            if variant_record["input_kind"] not in prompt_spec.compatible_input_kinds:
                continue
            records.append(
                _write_prompt_variant_record(
                    out_dir,
                    variant_record,
                    prompt_spec,
                    overwrite=overwrite,
                )
            )

    if not records:
        raise ValueError(
            "No compatible prompt/image variant pairs selected. "
            "Check --variant, --prompt, and input_kind compatibility."
        )
    _write_prompt_variant_manifests(out_dir, records)
    return records


def _selected_prompt_specs(prompt_ids: tuple[str, ...] | None) -> tuple[PromptSpec, ...]:
    if not prompt_ids or "all" in prompt_ids:
        return PROMPT_REGISTRY

    requested = set(prompt_ids)
    unknown = sorted(requested - set(PROMPT_IDS))
    if unknown:
        raise ValueError(f"Unknown prompt id(s): {', '.join(unknown)}")
    return tuple(spec for spec in PROMPT_REGISTRY if spec.prompt_id in requested)


def _load_or_build_variant_records(
    out_dir: Path,
    *,
    selected_slugs: set[str] | None,
    selected_systems: set[int] | None,
    selected_measures: set[int] | None,
    variant_ids: tuple[str, ...] | None,
) -> list[dict[str, Any]]:
    manifest_path = out_dir / VARIANTS_MANIFEST_NAME
    if not manifest_path.exists():
        build_vlm_melody_variants(
            out_dir,
            selected_slugs=selected_slugs,
            selected_systems=selected_systems,
            selected_measures=selected_measures,
            variant_ids=variant_ids,
            overwrite=False,
        )
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"VLM melody variants manifest not found: {manifest_path}. "
            "Run scripts/build_vlm_melody_variants.py first."
        )

    selected = list(
        _selected_variant_records(
            _read_jsonl(manifest_path),
            selected_slugs=selected_slugs,
            selected_systems=selected_systems,
            selected_measures=selected_measures,
            variant_ids=variant_ids,
        )
    )
    if not selected and (out_dir / "manifest.jsonl").exists():
        build_vlm_melody_variants(
            out_dir,
            selected_slugs=selected_slugs,
            selected_systems=selected_systems,
            selected_measures=selected_measures,
            variant_ids=variant_ids,
            overwrite=False,
        )
        selected = list(
            _selected_variant_records(
                _read_jsonl(manifest_path),
                selected_slugs=selected_slugs,
                selected_systems=selected_systems,
                selected_measures=selected_measures,
                variant_ids=variant_ids,
            )
        )
    return selected


def _write_prompt_variant_record(
    out_dir: Path,
    variant_record: dict[str, Any],
    prompt_spec: PromptSpec,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    context_path = _resolve_path(out_dir, variant_record["context_path"])
    context = json.loads(context_path.read_text(encoding="utf-8"))
    prompt_dir = _prompt_output_dir(out_dir, variant_record, prompt_spec)
    prompt_dir.mkdir(parents=True, exist_ok=True)

    system_prompt, user_prompt = _render_prompts(variant_record, context, prompt_spec)
    schema = _schema_for_prompt(variant_record["input_kind"], prompt_spec)
    system_path = prompt_dir / "system.txt"
    user_path = prompt_dir / "user.txt"
    schema_path = prompt_dir / "schema.json"
    config_path = prompt_dir / "config.json"

    if overwrite or not system_path.exists():
        system_path.write_text(system_prompt, encoding="utf-8")
    if overwrite or not user_path.exists():
        user_path.write_text(user_prompt, encoding="utf-8")
    if schema is not None:
        if overwrite or not schema_path.exists():
            schema_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    elif schema_path.exists() and overwrite:
        schema_path.unlink()

    prompt_record = {
        "prompt_variant_id": f"{variant_record['variant_id']}__{prompt_spec.prompt_id}",
        "prompt_id": prompt_spec.prompt_id,
        "variant_id": variant_record["variant_id"],
        "input_kind": variant_record["input_kind"],
        "transcription_mode": prompt_spec.transcription_mode,
        "output_mode": prompt_spec.output_mode,
        "slug": variant_record["slug"],
        "system_index": variant_record["system_index"],
        "system_measure_index": variant_record["system_measure_index"],
        "global_measure_index": variant_record["global_measure_index"],
        "display_measure_number": variant_record["display_measure_number"],
        "image_path": variant_record["image_path"],
        "context_path": variant_record["context_path"],
        "system_prompt_path": str(system_path),
        "user_prompt_path": str(user_path),
        "schema_path": str(schema_path) if schema is not None else None,
        "config_path": str(config_path),
        "prompt": {
            "description": prompt_spec.description,
            "compatible_input_kinds": list(prompt_spec.compatible_input_kinds),
        },
        "image_variant_recipe": variant_record.get("recipe", {}),
    }
    if overwrite or not config_path.exists():
        config_path.write_text(json.dumps(prompt_record, indent=2) + "\n", encoding="utf-8")
    return prompt_record


def _render_prompts(
    variant_record: dict[str, Any],
    context: dict[str, Any],
    prompt_spec: PromptSpec,
) -> tuple[str, str]:
    context_text = json.dumps(_compact_context(context, variant_record), sort_keys=True)
    image_description = _image_description(variant_record)
    response_shape = _response_shape_text(prompt_spec)
    system_prompt = _system_prompt_for(prompt_spec, variant_record)
    user_prompt = (
        f"{response_shape}\n"
        f"Image variant: {variant_record['variant_id']}\n"
        f"{image_description}\n"
        f"{_task_instruction(prompt_spec, variant_record)}\n"
        f"Context: {context_text}"
    )
    return system_prompt, user_prompt


def _system_prompt_for(prompt_spec: PromptSpec, variant_record: dict[str, Any]) -> str:
    input_kind = str(variant_record["input_kind"])
    if prompt_spec.prompt_id == "describe_then_guess_v2":
        return (
            "You inspect handwritten single-staff melody crop images. Describe visible "
            "musical evidence, then make an educated best-effort transcription. The crop "
            "is intended to contain one measure. Be willing to make low-confidence guesses "
            "when black handwritten shapes plausibly indicate noteheads."
        )
    if prompt_spec.prompt_id == "free_response_describe_v1":
        return (
            "You inspect handwritten single-staff melody crop images. "
            "Describe what you see directly and concisely. Do not invent certainty."
        )
    if prompt_spec.prompt_id == "educated_pitch_v2":
        return (
            "You transcribe handwritten single-staff melody from a cropped score image. "
            "Return JSON only. The crop is intended to contain one measure. Make an "
            "educated best-effort transcription and use confidence/evidence to mark "
            "uncertainty. Do not omit plausible notes just because the handwriting is "
            "ambiguous."
        )
    if prompt_spec.prompt_id == "notehead_count_then_pitch_v1":
        return (
            "You transcribe handwritten single-staff melody from cropped score images. "
            "Return JSON only. First decide how many real noteheads are visible, then "
            "map only those noteheads to pitch and duration. Rests are allowed but do "
            "not count as noteheads."
        )
    if input_kind == "staff_overlay":
        return (
            "You transcribe handwritten single-staff melody from a cropped score image. "
            "Return JSON only. Red horizontal staff lines are visual helpers, not music."
        )
    if input_kind == "pitch_ruler":
        return (
            "You transcribe handwritten single-staff melody from a cropped score image. "
            "Return JSON only. Pitch labels and colored guide lines are visual helpers, "
            "not music."
        )
    if input_kind == "pitch_ruler_soft":
        return (
            "You transcribe handwritten single-staff melody from a cropped score image. "
            "Return JSON only. Faint dotted pitch guides are visual helpers, not music."
        )
    if input_kind == "pitch_ruler_panel":
        return (
            "You transcribe handwritten single-staff melody from a cropped score image. "
            "Return JSON only. The left pitch-reference gutter is not music; use it "
            "only as a vertical ruler for the clean music crop on the right."
        )
    if input_kind == "neighbor_context":
        return (
            "You transcribe a target measure from a handwritten single-staff melody. "
            "Return JSON only. Thin red ticks appear only in the white top and bottom "
            "margins and delimit the target measure; they are not music. Adjacent measures "
            "are present only to reveal the writer's notation style."
        )
    return (
        "You transcribe handwritten single-staff melody from cropped score images. "
        "Return JSON only. Transcribe only the melody notes and rests visible in the crop."
    )


def _task_instruction(prompt_spec: PromptSpec, variant_record: dict[str, Any]) -> str:
    if prompt_spec.prompt_id == "direct_pitch_v0":
        return (
            "Use scientific pitch notation. Use empty pitch for rests or unclear pitch. "
            "If duration is unclear, use unknown."
        )
    if prompt_spec.prompt_id == "describe_then_guess_v2":
        return (
            "First describe the visible black handwritten ink: likely noteheads, stems, "
            "flags/beams, rests, barlines, helper graphics, and ambiguous shapes. Then give "
            "a final best-guess transcription in order, using scientific pitch names and "
            "durations. Do not stop at 'too ambiguous'; say what you would transcribe and "
            "where confidence is low."
        )
    if prompt_spec.prompt_id == "free_response_describe_v1":
        return (
            "Describe the visual musical content: likely notes, rests, pitch levels, "
            "durations, helper graphics, and uncertainties. A final transcription is useful "
            "but not required."
        )
    if prompt_spec.prompt_id == "educated_pitch_v2":
        return (
            "Treat this as one intended measure crop. First identify plausible black "
            "handwritten notehead shapes from left to right. If a black shape could be a "
            "notehead attached to a stem/flag, include it as a low-confidence note rather "
            "than dropping it. Do not return zero notes unless the crop is truly blank of "
            "notehead-shaped ink. Do not classify notehead/stem combinations as rests unless "
            "they are clearly rest symbols. If an apparent internal vertical stroke appears "
            "inside the crop, do not assume the image contains a second measure unless it is "
            "clearly a barline; mark uncertainty instead. Use any pitch labels or staff-line "
            "helpers only as visual aids. Provide evidence for each item."
        )
    if prompt_spec.prompt_id == "notehead_count_then_pitch_v1":
        return (
            "Step 1: identify only real black handwritten noteheads from left to right. "
            "Step 2: set notehead_count to that count. Step 3: transcribe those notes and "
            "any visible rests. Do not treat barlines, stems alone, slurs, guide lines, "
            "labels, accidentals, or noise as noteheads."
        )
    if str(variant_record["input_kind"]) == "staff_overlay":
        return (
            "The red horizontal lines mark estimated staff lines. They help locate pitch but "
            "are not music. Do not count red line intersections as noteheads. Provide your "
            "best-effort transcription with evidence and uncertainty."
        )
    if str(variant_record["input_kind"]) == "pitch_ruler":
        return (
            "Use the pitch labels on the left and colored guide lines only to map vertical "
            "position to pitch. Do not count guide-line crossings, labels, barlines, stems "
            "alone, slurs, or accidentals as noteheads. Provide evidence and uncertainty."
        )
    if str(variant_record["input_kind"]) == "pitch_ruler_soft":
        return (
            "Use the left labels and faint dotted pitch guides only to map vertical position "
            "to pitch. The black handwritten ink is the musical source. Do not count helper "
            "dots or intersections as noteheads. Provide evidence and uncertainty."
        )
    if str(variant_record["input_kind"]) == "pitch_ruler_panel":
        return (
            "The left gutter contains pitch labels and short ticks; the right side contains "
            "the clean music crop. Use the gutter only as a vertical pitch ruler. Do not count "
            "labels, ticks, the divider, barlines, stems alone, slurs, or helper marks as "
            "noteheads. Provide evidence and uncertainty."
        )
    if str(variant_record["input_kind"]) == "neighbor_context":
        return (
            "Transcribe only the music horizontally between the two red margin ticks. "
            "Use the neighboring measures to distinguish this writer's long stems from "
            "barlines, but do not copy or transcribe neighboring notes. First identify "
            "rest symbols in the target, then identify each filled notehead from left to "
            "right and attach its stem or flag. A long internal vertical stroke with an "
            "attached notehead is a stem, not a measure boundary. Count a barline only "
            "when it spans the full five-line staff without an attached notehead. Return "
            "every plausible target note once; express ambiguity through confidence and "
            "evidence rather than inventing a repeated phrase."
        )
    return (
        "Provide your best-effort transcription. Include visible or likely rests, including "
        "initial silence before the first note. Use confidence, evidence, and uncertainty "
        "instead of refusing when ambiguous."
    )


def _response_shape_text(prompt_spec: PromptSpec) -> str:
    if prompt_spec.prompt_id == "describe_then_guess_v2":
        return (
            "Return concise prose with these sections: Visible evidence; Ambiguities; "
            "Final best-guess transcription. Do not use JSON for this debug prompt."
        )
    if prompt_spec.prompt_id == "free_response_describe_v1":
        return "Return concise prose. Do not use JSON for this debug prompt."
    if prompt_spec.prompt_id in {"notehead_count_then_pitch_v1", "educated_pitch_v2"}:
        return (
            "Return this JSON shape exactly:\n"
            '{"notehead_count":0,'
            '"items":[{"kind":"note|rest","pitch":"G4 or empty string",'
            '"duration":"1/8|1/4|1/2|dotted-1/4|unknown",'
            '"accidental":"sharp|flat|natural|none|unknown","confidence":0.0,'
            '"evidence":"visible facts supporting this item"}],'
            '"comments":"short summary","overall_confidence":0.0,'
            '"uncertainties":["visible uncertainty"]}'
        )
    if prompt_spec.prompt_id == "direct_pitch_v0":
        return (
            "Return this JSON shape exactly:\n"
            '{"items":[{"kind":"note|rest","pitch":"G4 or empty string",'
            '"duration":"1/8|1/4|1/2|dotted-1/4|unknown",'
            '"accidental":"sharp|flat|natural|none|unknown","confidence":0.0}],'
            '"comments":"short uncertainty note"}'
        )
    return (
        "Return this JSON shape exactly:\n"
        '{"items":[{"kind":"note|rest","pitch":"G4 or empty string",'
        '"duration":"1/8|1/4|1/2|dotted-1/4|unknown",'
        '"accidental":"sharp|flat|natural|none|unknown","confidence":0.0,'
        '"evidence":"visible facts supporting this item"}],'
        '"comments":"short summary","overall_confidence":0.0,'
        '"uncertainties":["visible uncertainty"]}'
    )


def _schema_for_prompt(input_kind: InputKind, prompt_spec: PromptSpec) -> dict[str, Any] | None:
    if prompt_spec.output_mode == "free_response":
        return None
    if prompt_spec.prompt_id in {"notehead_count_then_pitch_v1", "educated_pitch_v2"}:
        return NOTEHEAD_COUNT_RESPONSE_SCHEMA
    if prompt_spec.prompt_id == "direct_pitch_v0":
        return PITCH_RESPONSE_SCHEMA
    return PITCH_RULER_SOFT_RESPONSE_SCHEMA


def _image_description(variant_record: dict[str, Any]) -> str:
    input_kind = str(variant_record["input_kind"])
    recipe = variant_record.get("recipe", {})
    if input_kind == "raw":
        return "Image description: full-height measure crop from the detected system."
    if input_kind == "staff":
        return "Image description: clean staff-region crop with no added helper graphics."
    if input_kind == "staff_overlay":
        return "Image description: staff-region crop with red horizontal staff-line helpers."
    source = recipe.get("source_kind", "unknown source")
    if input_kind == "pitch_ruler":
        return (
            "Image description: pitch labels on the left and strong colored horizontal "
            f"pitch guides drawn over a {source} crop."
        )
    if input_kind == "pitch_ruler_soft":
        return (
            "Image description: pitch labels on the left and faint dotted horizontal "
            f"pitch guides drawn over a {source} crop."
        )
    if input_kind == "pitch_ruler_panel":
        return (
            "Image description: pitch labels and short ticks in a left gutter, with the "
            f"{source} music crop on the right."
        )
    if input_kind == "neighbor_context":
        return (
            "Image description: the target measure is horizontally delimited by thin red "
            "ticks in white margins; one adjacent measure on each side provides handwriting "
            "context."
        )
    return f"Image description: {input_kind}."


def _compact_context(context: dict[str, Any], variant_record: dict[str, Any]) -> dict[str, Any]:
    compact = {key: context.get(key) for key in COMPACT_CONTEXT_KEYS}
    compact["variant_id"] = variant_record["variant_id"]
    compact["input_kind"] = variant_record["input_kind"]
    compact["image_variant_recipe"] = variant_record.get("recipe", {})
    return compact


def _prompt_output_dir(
    out_dir: Path,
    variant_record: dict[str, Any],
    prompt_spec: PromptSpec,
) -> Path:
    return (
        out_dir
        / PROMPT_VARIANTS_ROOT_NAME
        / str(variant_record["slug"])
        / f"system_{int(variant_record['system_index']):03d}"
        / f"measure_{int(variant_record['system_measure_index']):03d}"
        / f"{variant_record['variant_id']}__{prompt_spec.prompt_id}"
    )


def _selected_variant_records(
    records: list[dict[str, Any]],
    *,
    selected_slugs: set[str] | None,
    selected_systems: set[int] | None,
    selected_measures: set[int] | None,
    variant_ids: tuple[str, ...] | None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    requested_variants = None if not variant_ids or "all" in variant_ids else set(variant_ids)
    for record in records:
        if selected_slugs is not None and record.get("slug") not in selected_slugs:
            continue
        if selected_systems is not None and int(record["system_index"]) not in selected_systems:
            continue
        if (
            selected_measures is not None
            and int(record["system_measure_index"]) not in selected_measures
        ):
            continue
        if requested_variants is not None and record.get("variant_id") not in requested_variants:
            continue
        selected.append(record)
    return selected


def _write_prompt_variant_manifests(out_dir: Path, records: list[dict[str, Any]]) -> None:
    _write_jsonl(out_dir / PROMPT_VARIANTS_MANIFEST_NAME, records)

    by_slug: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_slug.setdefault(str(record["slug"]), []).append(record)

    for slug, slug_records in by_slug.items():
        manifest_path = (
            out_dir / slug / "vlm_melody_inputs" / PER_WORK_PROMPT_VARIANTS_MANIFEST_NAME
        )
        _write_jsonl(manifest_path, slug_records)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _resolve_path(out_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return out_dir / path


if __name__ == "__main__":
    raise SystemExit(main())
