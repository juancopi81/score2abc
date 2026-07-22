"""Build pitch-ruler VLM melody inputs from existing measure crops.

This script does not call any VLM. It reads records produced by
`scripts/build_vlm_melody_inputs.py` and creates an additional inspectable image
variant where deterministic treble-clef pitch labels and horizontal guide lines
are drawn beside the measure crop.

Example:
    uv run python scripts/build_vlm_pitch_ruler_inputs.py out \\
        --slug jaime-llanos_12_aviador_pasillo_fulgencio-garcia \\
        --system 1 --measure 3
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Literal

from PIL import Image, ImageDraw, ImageFont

from score2abc.utils import get_logger

SourceKind = Literal["staff", "staff_overlay"]
RulerStyle = Literal["standard", "soft", "panel"]

INPUT_MANIFEST_NAME = "vlm_melody_inputs_manifest.jsonl"
OUTPUT_MANIFEST_NAME = "vlm_pitch_ruler_inputs_manifest.jsonl"
PER_SLUG_MANIFEST_NAME = "pitch_ruler_manifest.jsonl"
SOURCE_PATH_KEYS: dict[SourceKind, str] = {
    "staff": "measure_staff",
    "staff_overlay": "measure_staff_overlay",
}
PITCH_RULER_PATH_KEY = "measure_pitch_ruler"
PITCH_RULER_SOFT_PATH_KEY = "measure_pitch_ruler_soft"
PITCH_RULER_PANEL_PATH_KEY = "measure_pitch_ruler_panel"
DEFAULT_LABEL_WIDTH_PX = 72
GUIDE_LINE_COLOR = (60, 100, 180)
STAFF_LINE_COLOR = (220, 20, 60)
SOFT_GUIDE_LINE_COLOR = (214, 214, 214)
SOFT_STAFF_LINE_COLOR = (190, 190, 190)
SOFT_LABEL_COLOR = (80, 80, 80)
LABEL_COLOR = (20, 20, 20)
PAD_BACKGROUND = (255, 255, 255)
TREBLE_REFERENCE_LETTER_INDEX = 2  # E
TREBLE_REFERENCE_OCTAVE = 4
LETTERS = ("C", "D", "E", "F", "G", "A", "B")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logger = get_logger("score2abc.build_vlm_pitch_ruler_inputs")

    try:
        records = build_vlm_pitch_ruler_inputs(
            args.out_dir,
            selected_slugs=set(args.slug) if args.slug else None,
            selected_systems=set(args.system) if args.system else None,
            selected_measures=set(args.measure) if args.measure else None,
            source_kind=args.source_kind,
            style=args.style,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Wrote %d pitch-ruler VLM input records", len(records))
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
        "--source-kind",
        choices=tuple(SOURCE_PATH_KEYS),
        default="staff_overlay",
        help="Existing measure crop variant to annotate. Defaults to staff_overlay.",
    )
    parser.add_argument(
        "--style",
        choices=("standard", "soft", "panel"),
        default="standard",
        help=(
            "Pitch-ruler visual style. Soft uses faint guides; panel keeps labels/ticks "
            "separate from the music crop."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing images.")
    return parser


def build_vlm_pitch_ruler_inputs(
    out_dir: Path,
    *,
    selected_slugs: set[str] | None = None,
    selected_systems: set[int] | None = None,
    selected_measures: set[int] | None = None,
    source_kind: SourceKind = "staff_overlay",
    style: RulerStyle = "standard",
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Create pitch-ruler image inputs for selected existing measure records."""
    manifest_path = out_dir / INPUT_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"VLM melody input manifest not found: {manifest_path}. "
            "Run scripts/build_vlm_melody_inputs.py first."
        )

    records = list(
        _selected_records(
            _read_manifest(manifest_path),
            selected_slugs=selected_slugs,
            selected_systems=selected_systems,
            selected_measures=selected_measures,
        )
    )
    pitch_ruler_records = [
        _build_for_record(
            out_dir,
            record,
            source_kind=source_kind,
            style=style,
            overwrite=overwrite,
        )
        for record in records
    ]
    _write_manifests(out_dir, pitch_ruler_records)
    return pitch_ruler_records


def _build_for_record(
    out_dir: Path,
    record: dict[str, Any],
    *,
    source_kind: SourceKind,
    style: RulerStyle,
    overwrite: bool,
) -> dict[str, Any]:
    context_path = _resolve_path(out_dir, record["paths"]["context"])
    context = json.loads(context_path.read_text(encoding="utf-8"))
    clef = context.get("clef_hint")
    if clef != "treble":
        raise ValueError(
            "Pitch-ruler inputs currently support only treble clef; "
            f"got {clef!r} for {context_path}"
        )

    source_path = _resolve_path(out_dir, record["paths"][SOURCE_PATH_KEYS[source_kind]])
    output_path = _pitch_ruler_output_path(source_path, source_kind, style)
    if overwrite or not output_path.exists():
        staff_lines = _staff_lines_for_source(context, source_kind)
        source_image = Image.open(source_path).convert("RGB")
        ruler_image = make_pitch_ruler_image(source_image, staff_lines, style=style)
        ruler_image.save(output_path)

    next_record = dict(record)
    next_paths = dict(record["paths"])
    path_key = _path_key_for_style(style)
    next_paths[path_key] = str(output_path)
    next_record["paths"] = next_paths
    next_record["pitch_ruler"] = {
        "source_kind": source_kind,
        "style": style,
        "clef": clef,
        "staff_lines_y_px": _staff_lines_for_source(context, source_kind),
        "label_width_px": DEFAULT_LABEL_WIDTH_PX,
    }
    return next_record


def make_pitch_ruler_image(
    source_image: Image.Image,
    staff_lines_y_px: list[int],
    *,
    label_width_px: int = DEFAULT_LABEL_WIDTH_PX,
    style: RulerStyle = "standard",
) -> Image.Image:
    """Return a source crop with treble pitch labels and optional guide lines."""
    if len(staff_lines_y_px) != 5:
        raise ValueError(f"Expected exactly five staff lines, got {staff_lines_y_px!r}")

    source = source_image.convert("RGB")
    width, height = source.size
    canvas = Image.new("RGB", (width + label_width_px, height), color=PAD_BACKGROUND)
    canvas.paste(source, (label_width_px, 0))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    guide_positions = _visible_staff_positions(staff_lines_y_px, height)
    if style == "panel":
        draw.line(
            [(label_width_px - 1, 0), (label_width_px - 1, canvas.height - 1)],
            fill=(220, 220, 220),
            width=1,
        )
    for position, y in guide_positions:
        label = _treble_pitch_label(position)
        is_staff_line = position % 2 == 0
        if style == "soft":
            color = SOFT_STAFF_LINE_COLOR if is_staff_line else SOFT_GUIDE_LINE_COLOR
            label_color = SOFT_LABEL_COLOR
        else:
            color = STAFF_LINE_COLOR if is_staff_line else GUIDE_LINE_COLOR
            label_color = LABEL_COLOR

        if style == "panel":
            draw.line([(label_width_px - 10, y), (label_width_px - 2, y)], fill=color, width=1)
        elif style == "soft":
            _draw_soft_guide(draw, label_width_px, canvas.width - 1, y, fill=color)
        elif is_staff_line:
            draw.line([(label_width_px, y), (canvas.width - 1, y)], fill=color, width=1)
        else:
            _draw_dashed_line(draw, label_width_px, canvas.width - 1, y, fill=color)
        if style != "panel":
            draw.line([(label_width_px - 8, y), (label_width_px - 1, y)], fill=color, width=1)

        bbox = draw.textbbox((0, 0), label, font=font)
        text_height = bbox[3] - bbox[1]
        draw.text(
            (label_width_px - 12 - (bbox[2] - bbox[0]), y - text_height // 2),
            label,
            fill=label_color,
            font=font,
        )

    return canvas


def _visible_staff_positions(
    staff_lines_y_px: list[int],
    image_height: int,
) -> list[tuple[int, int]]:
    top_line = staff_lines_y_px[0]
    bottom_line = staff_lines_y_px[-1]
    staff_spacing = (bottom_line - top_line) / 4
    if staff_spacing <= 0:
        raise ValueError(f"Invalid staff line geometry: {staff_lines_y_px!r}")
    half_spacing = staff_spacing / 2

    min_position = math.floor((bottom_line - (image_height - 1)) / half_spacing)
    max_position = math.ceil(bottom_line / half_spacing)
    positions: list[tuple[int, int]] = []
    for position in range(min_position, max_position + 1):
        y = round(bottom_line - position * half_spacing)
        if 0 <= y < image_height:
            positions.append((position, y))
    return sorted(positions, key=lambda item: item[1])


def _treble_pitch_label(staff_position: int) -> str:
    diatonic_index = TREBLE_REFERENCE_LETTER_INDEX + staff_position
    letter = LETTERS[diatonic_index % len(LETTERS)]
    octave = TREBLE_REFERENCE_OCTAVE + (diatonic_index // len(LETTERS))
    return f"{letter}{octave}"


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    x0: int,
    x1: int,
    y: int,
    *,
    fill: tuple[int, int, int],
    dash_px: int = 6,
    gap_px: int = 5,
) -> None:
    x = x0
    while x <= x1:
        draw.line([(x, y), (min(x + dash_px, x1), y)], fill=fill, width=1)
        x += dash_px + gap_px


def _draw_soft_guide(
    draw: ImageDraw.ImageDraw,
    x0: int,
    x1: int,
    y: int,
    *,
    fill: tuple[int, int, int],
    dash_px: int = 1,
    gap_px: int = 9,
) -> None:
    x = x0
    while x <= x1:
        draw.point((x, y), fill=fill)
        if dash_px > 1:
            draw.line([(x, y), (min(x + dash_px - 1, x1), y)], fill=fill, width=1)
        x += dash_px + gap_px


def _staff_lines_for_source(context: dict[str, Any], source_kind: SourceKind) -> list[int]:
    if source_kind in ("staff", "staff_overlay"):
        values = context.get("staff_lines_y_px_in_staff_crop")
    else:
        values = None
    if not isinstance(values, list) or len(values) != 5:
        raise ValueError(f"Missing staff-line geometry in context: {context.get('paths', {})}")
    return [int(value) for value in values]


def _pitch_ruler_output_path(source_path: Path, source_kind: SourceKind, style: RulerStyle) -> Path:
    suffix = f"_{source_kind}"
    if not source_path.stem.endswith(suffix):
        raise ValueError(f"Unexpected {source_kind} crop filename: {source_path}")
    base_stem = source_path.stem[: -len(suffix)]
    if style == "panel":
        output_suffix = "pitch_ruler_panel"
    elif style == "soft":
        output_suffix = "pitch_ruler_soft"
    else:
        output_suffix = "pitch_ruler"
    return source_path.with_name(f"{base_stem}_{output_suffix}.png")


def _path_key_for_style(style: RulerStyle) -> str:
    if style == "panel":
        return PITCH_RULER_PANEL_PATH_KEY
    if style == "soft":
        return PITCH_RULER_SOFT_PATH_KEY
    return PITCH_RULER_PATH_KEY


def _selected_records(
    records: Iterable[dict[str, Any]],
    *,
    selected_slugs: set[str] | None,
    selected_systems: set[int] | None,
    selected_measures: set[int] | None,
) -> Iterable[dict[str, Any]]:
    for record in records:
        if selected_slugs is not None and record.get("slug") not in selected_slugs:
            continue
        if selected_systems is not None and record.get("system_index") not in selected_systems:
            continue
        if (
            selected_measures is not None
            and record.get("system_measure_index") not in selected_measures
        ):
            continue
        yield record


def _read_manifest(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def _write_manifests(out_dir: Path, records: list[dict[str, Any]]) -> None:
    top_level_manifest = out_dir / OUTPUT_MANIFEST_NAME
    with top_level_manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    by_slug: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_slug.setdefault(str(record["slug"]), []).append(record)

    for slug, slug_records in by_slug.items():
        manifest_path = out_dir / slug / "vlm_melody_inputs" / PER_SLUG_MANIFEST_NAME
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as handle:
            for record in slug_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _resolve_path(out_dir: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    if path.parts[:1] == (out_dir.name,):
        if out_dir.is_absolute():
            return out_dir.parent / path
        return path
    return out_dir / path


if __name__ == "__main__":
    raise SystemExit(main())
