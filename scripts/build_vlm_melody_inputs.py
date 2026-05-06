"""Build local VLM melody-input crops from existing pipeline system crops.

This script does not call any VLM. It prepares inspectable measure-level image
inputs and JSON context files so melody-recognition prompt experiments can be
run from stable artifacts.

Usage:
    uv run python scripts/build_vlm_melody_inputs.py out \\
        --slug jaime-llanos_12_aviador_pasillo_fulgencio-garcia --system 1
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageOps

from score2abc.chord_ocr.alignment import detect_barlines
from score2abc.events import measure_length_beats
from score2abc.manifest import load_manifest_jsonl
from score2abc.schemas import WorkItem
from score2abc.utils import get_logger
from score2abc.utils.imaging import estimate_ink_threshold

SEPARATOR_PADDING_PX = 12
MIN_MEASURE_WIDTH_PX = 16
LEADING_BARLINE_FRACTION = 0.08
TRAILING_BARLINE_FRACTION = 0.92
SYSTEM_CROP_RE = re.compile(r"^system_(?P<index>\d{3})$")


@dataclass(frozen=True)
class StaffEstimate:
    y0: int
    y1: int
    line_ys: tuple[int, ...]


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logger = get_logger("score2abc.build_vlm_melody_inputs")

    manifest_path = args.out_dir / "manifest.jsonl"
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        return 1

    system_filter = set(args.system) if args.system else None
    selected_slugs = set(args.slug) if args.slug else None
    records = build_vlm_melody_inputs(
        args.out_dir,
        selected_slugs=selected_slugs,
        selected_systems=system_filter,
        overwrite=args.overwrite,
    )
    logger.info("Wrote %d VLM melody input records", len(records))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="Pipeline output directory with manifest.jsonl")
    parser.add_argument(
        "--slug",
        action="append",
        default=None,
        help="Limit to specific work slugs (repeat to pass multiple).",
    )
    parser.add_argument(
        "--system",
        action="append",
        type=int,
        default=None,
        help="Limit to a 1-based system index (repeat to pass multiple).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing crop/context files. The manifest is always rewritten.",
    )
    return parser


def build_vlm_melody_inputs(
    out_dir: Path,
    *,
    selected_slugs: set[str] | None = None,
    selected_systems: set[int] | None = None,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Create measure-level melody VLM input crops for selected work items."""
    manifest_path = out_dir / "manifest.jsonl"
    work_items = load_manifest_jsonl(manifest_path)
    records: list[dict[str, Any]] = []

    for item in work_items:
        if selected_slugs is not None and item.slug not in selected_slugs:
            continue
        records.extend(
            _build_for_work(
                item,
                out_dir=out_dir,
                selected_systems=selected_systems,
                overwrite=overwrite,
            )
        )

    _write_manifest(out_dir, records)
    return records


def _build_for_work(
    item: WorkItem,
    *,
    out_dir: Path,
    selected_systems: set[int] | None,
    overwrite: bool,
) -> list[dict[str, Any]]:
    systems_dir = out_dir / item.slug / "systems"
    if not systems_dir.exists():
        return []

    output_root = out_dir / item.slug / "vlm_melody_inputs"
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    global_measure_index = 0
    for system_path in _iter_system_crops(systems_dir):
        system_index = _parse_system_index(system_path.stem)
        if system_index <= 0:
            continue

        barlines = sorted(detect_barlines(system_path))
        boundaries = _measure_boundaries(barlines)
        measure_count = max(0, len(boundaries) - 1)
        if selected_systems is not None and system_index not in selected_systems:
            global_measure_index += measure_count
            continue

        records_for_system = _build_for_system(
            item,
            system_path=system_path,
            output_root=output_root,
            system_index=system_index,
            start_global_measure_index=global_measure_index,
            barlines=barlines,
            boundaries=boundaries,
            overwrite=overwrite,
        )
        records.extend(records_for_system)
        global_measure_index += measure_count

    return records


def _build_for_system(
    item: WorkItem,
    *,
    system_path: Path,
    output_root: Path,
    system_index: int,
    start_global_measure_index: int,
    barlines: list[float],
    boundaries: list[float],
    overwrite: bool,
) -> list[dict[str, Any]]:
    system_output_dir = output_root / f"system_{system_index:03d}"
    system_output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        _clear_generated_system_outputs(system_output_dir)

    image = Image.open(system_path).convert("RGB")
    width, height = image.size
    staff = _estimate_staff(image)

    records: list[dict[str, Any]] = []
    for measure_offset, (left_fraction, right_fraction) in enumerate(_pairs(boundaries), start=0):
        system_measure_index = measure_offset + 1
        global_measure_index = start_global_measure_index + measure_offset
        x0, x1 = _measure_x_bounds(width, left_fraction, right_fraction)
        bbox_raw = (x0, 0, x1, height)
        bbox_staff = (x0, staff.y0, x1, staff.y1)

        stem = f"measure_{system_measure_index:03d}"
        raw_path = system_output_dir / f"{stem}_raw.png"
        staff_path = system_output_dir / f"{stem}_staff.png"
        overlay_path = system_output_dir / f"{stem}_staff_overlay.png"
        context_path = system_output_dir / f"{stem}_context.json"

        if overwrite or not raw_path.exists():
            image.crop(bbox_raw).save(raw_path)
        if overwrite or not staff_path.exists():
            image.crop(bbox_staff).save(staff_path)
        if overwrite or not overlay_path.exists():
            _write_staff_overlay(image, bbox_staff, staff.line_ys, overlay_path)

        context = _context_payload(
            item,
            system_path=system_path,
            raw_path=raw_path,
            staff_path=staff_path,
            overlay_path=overlay_path,
            context_path=context_path,
            system_index=system_index,
            system_measure_index=system_measure_index,
            global_measure_index=global_measure_index,
            barlines=barlines,
            x_bounds=(x0, x1),
            x_fraction_bounds=(left_fraction, right_fraction),
            staff=staff,
        )
        if overwrite or not context_path.exists():
            _write_json(context_path, context)
        records.append(context)

    return records


def _estimate_staff(image: Image.Image) -> StaffEstimate:
    gray = ImageOps.grayscale(image)
    width, height = gray.size
    threshold = estimate_ink_threshold(gray)
    x0 = int(width * 0.05)
    x1 = max(x0 + 1, int(width * 0.95))
    rows: list[float] = []
    for y in range(height):
        dark = 0
        for x in range(x0, x1):
            if gray.getpixel((x, y)) < threshold:
                dark += 1
        rows.append(dark / (x1 - x0))

    line_ys = _detect_staff_line_rows(rows, height)
    if len(line_ys) >= 2:
        span = max(line_ys) - min(line_ys)
        padding = max(18, int(span * 0.45))
        y0 = max(0, min(line_ys) - padding)
        y1 = min(height, max(line_ys) + padding + 1)
        return StaffEstimate(y0=y0, y1=y1, line_ys=tuple(line_ys))

    padding = max(24, height // 6)
    return StaffEstimate(y0=0, y1=height, line_ys=tuple(_fallback_lines(height, padding)))


def _detect_staff_line_rows(rows: list[float], height: int) -> list[int]:
    if not rows:
        return []
    max_density = max(rows)
    if max_density <= 0:
        return []

    cutoff = max(0.08, max_density * 0.45)
    spans: list[tuple[float, int]] = []
    start: int | None = None
    for y, density in enumerate(rows):
        if density >= cutoff and start is None:
            start = y
        elif density < cutoff and start is not None:
            spans.append(_score_row_span(rows, start, y - 1))
            start = None
    if start is not None:
        spans.append(_score_row_span(rows, start, len(rows) - 1))

    if len(spans) < 5:
        return []

    selected = sorted(spans, reverse=True)[:5]
    line_ys = sorted(center for _, center in selected)
    if line_ys[-1] - line_ys[0] < max(16, height // 20):
        return []
    return line_ys


def _score_row_span(rows: list[float], start: int, end: int) -> tuple[float, int]:
    densities = rows[start : end + 1]
    score = max(densities)
    weighted_sum = sum((start + idx) * density for idx, density in enumerate(densities))
    total_density = sum(densities)
    if total_density <= 0:
        center = (start + end) // 2
    else:
        center = round(weighted_sum / total_density)
    return score, center


def _fallback_lines(height: int, padding: int) -> list[int]:
    top = padding
    bottom = max(top + 4, height - padding)
    if bottom <= top:
        return []
    step = (bottom - top) / 4
    return [round(top + idx * step) for idx in range(5)]


def _write_staff_overlay(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    line_ys: tuple[int, ...],
    output_path: Path,
) -> None:
    crop = image.crop(bbox).convert("RGB")
    draw = ImageDraw.Draw(crop)
    _, y0, _, _ = bbox
    for y in line_ys:
        overlay_y = y - y0
        if 0 <= overlay_y < crop.height:
            draw.line([(0, overlay_y), (crop.width - 1, overlay_y)], fill=(220, 20, 60), width=1)
    crop.save(output_path)


def _context_payload(
    item: WorkItem,
    *,
    system_path: Path,
    raw_path: Path,
    staff_path: Path,
    overlay_path: Path,
    context_path: Path,
    system_index: int,
    system_measure_index: int,
    global_measure_index: int,
    barlines: list[float],
    x_bounds: tuple[int, int],
    x_fraction_bounds: tuple[float, float],
    staff: StaffEstimate,
) -> dict[str, Any]:
    time_signature = item.metadata.time_signature
    expected_beats = _expected_beats(time_signature)
    return {
        "slug": item.slug,
        "title": item.metadata.title,
        "composer": item.metadata.composer,
        "rhythm": item.metadata.rhythm,
        "time_signature_hint": time_signature,
        "key_hint": item.metadata.key_hint,
        "tempo_hint": item.metadata.tempo_hint,
        "clef_hint": "treble",
        "system_index": system_index,
        "system_measure_index": system_measure_index,
        "global_measure_index": global_measure_index,
        "display_measure_number": global_measure_index + 1,
        "allow_pickup": global_measure_index == 0,
        "expected_measure_beats": expected_beats,
        "task": "Transcribe only the melody notes visible in this single measure image.",
        "paths": {
            "source_system": str(system_path),
            "measure_raw": str(raw_path),
            "measure_staff": str(staff_path),
            "measure_staff_overlay": str(overlay_path),
            "context": str(context_path),
        },
        "barlines_x_fraction": barlines,
        "x_bounds_px": {"left": x_bounds[0], "right": x_bounds[1]},
        "x_fraction_bounds": {"left": x_fraction_bounds[0], "right": x_fraction_bounds[1]},
        "staff_crop_y_bounds_px": {"top": staff.y0, "bottom": staff.y1},
        "staff_lines_y_px_in_system": list(staff.line_ys),
        "staff_lines_y_px_in_staff_crop": [y - staff.y0 for y in staff.line_ys],
    }


def _expected_beats(time_signature: str | None) -> str | None:
    if not time_signature:
        return None
    try:
        beats = measure_length_beats(time_signature)
    except ValueError:
        return None
    return _fraction_to_json(beats)


def _fraction_to_json(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _measure_boundaries(barlines: list[float]) -> list[float]:
    """Return measure-edge x-fractions for the system.

    Barlines inside the leading 5% of the crop sit before measure 1 (they're
    a real start-of-system barline plus possibly a clef-area FP or two);
    they collapse to a single left boundary at the rightmost of them.
    Barlines past 97% behave symmetrically as a single right boundary.
    Everything in between fences interior measures normally.
    """
    leading = [b for b in barlines if b <= LEADING_BARLINE_FRACTION]
    trailing = [b for b in barlines if b >= TRAILING_BARLINE_FRACTION]
    middle = [
        b
        for b in barlines
        if LEADING_BARLINE_FRACTION < b < TRAILING_BARLINE_FRACTION
    ]
    left = max(leading) if leading else 0.0
    right = min(trailing) if trailing else 1.0
    return [left, *middle, right]


def _measure_x_bounds(width: int, left_fraction: float, right_fraction: float) -> tuple[int, int]:
    left = round(left_fraction * width)
    right = round(right_fraction * width)
    left = max(0, left - SEPARATOR_PADDING_PX)
    right = min(width, right + SEPARATOR_PADDING_PX)
    if right - left < MIN_MEASURE_WIDTH_PX:
        center = max(0, min(width, (left + right) // 2))
        left = max(0, center - MIN_MEASURE_WIDTH_PX // 2)
        right = min(width, left + MIN_MEASURE_WIDTH_PX)
    return left, right


def _pairs(values: Iterable[float]) -> Iterable[tuple[float, float]]:
    ordered = list(values)
    for idx in range(len(ordered) - 1):
        yield ordered[idx], ordered[idx + 1]


def _parse_system_index(stem: str) -> int:
    match = SYSTEM_CROP_RE.fullmatch(stem)
    if match is None:
        return 0
    return int(match.group("index"))


def _iter_system_crops(systems_dir: Path) -> Iterable[Path]:
    for path in sorted(systems_dir.glob("system_*.png")):
        if SYSTEM_CROP_RE.fullmatch(path.stem):
            yield path


def _write_manifest(out_dir: Path, records: list[dict[str, Any]]) -> None:
    by_slug: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_slug.setdefault(str(record["slug"]), []).append(record)

    for slug, slug_records in by_slug.items():
        manifest_path = out_dir / slug / "vlm_melody_inputs" / "manifest.jsonl"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as handle:
            for record in slug_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    top_level_manifest = out_dir / "vlm_melody_inputs_manifest.jsonl"
    with top_level_manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _clear_generated_system_outputs(system_output_dir: Path) -> None:
    for path in system_output_dir.glob("measure_*"):
        if path.is_file():
            path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
