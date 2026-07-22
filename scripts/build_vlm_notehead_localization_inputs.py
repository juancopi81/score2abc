"""Build reproducible multi-view inputs for notehead-localization spikes.

The builder consumes existing measure records from
``out/vlm_melody_inputs_manifest.jsonl``. It never reads coordinate or
canonical ground truth; discoverable ground-truth paths are copied into the
output manifest only as evaluation metadata.

Example:
    uv run python scripts/build_vlm_notehead_localization_inputs.py out \
        --slug jaime-llanos_12_aviador_pasillo_fulgencio-garcia \
        --system 1 --measure 3 --task-kind all
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from score2abc.utils import get_logger  # noqa: E402
from score2abc.utils.imaging import estimate_ink_threshold  # noqa: E402
from scripts.build_vlm_notehead_candidates import (  # noqa: E402
    GridCandidate,
    detect_staff_grid_density_candidates,
)

INPUT_MANIFEST_NAME = "vlm_melody_inputs_manifest.jsonl"
OUTPUT_MANIFEST_NAME = "vlm_notehead_localization_manifest.jsonl"
DIRECT_LOCALIZATION = "direct-localization"
CANDIDATE_ASSISTED_LOCALIZATION = "candidate-assisted-localization"
TASK_KINDS = (DIRECT_LOCALIZATION, CANDIDATE_ASSISTED_LOCALIZATION)
TaskKind = Literal["direct-localization", "candidate-assisted-localization"]
TaskSelection = Literal["direct-localization", "candidate-assisted-localization", "all"]

DEFAULT_MAX_CANDIDATES = 24
DETAIL_SCALE = 4
GALLERY_SCALE = 3
GALLERY_CROP_STAFF_SPACINGS = 3.0
GALLERY_HEADER_HEIGHT_PX = 24
GALLERY_GAP_PX = 12
GALLERY_MAX_COLUMNS = 4
CONTEXT_MIN_MARGIN_PX = 24
TARGET_BOUNDARY_COLOR = (190, 20, 35)

COORDINATE_GROUND_TRUTH_DIR = REPO_ROOT / "tests/fixtures/vlm_melody/notehead_ground_truth"
CANONICAL_GROUND_TRUTH_DIR = REPO_ROOT / "dataset/ground_truth"


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logger = get_logger("score2abc.build_vlm_notehead_localization_inputs")
    try:
        records = build_vlm_notehead_localization_inputs(
            args.out_dir,
            selected_slugs=set(args.slug) if args.slug else None,
            selected_systems=set(args.system) if args.system else None,
            selected_measures=set(args.measure) if args.measure else None,
            task_kind=args.task_kind,
            max_candidates=args.max_candidates,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Wrote %d notehead-localization task record(s)", len(records))
    logger.info("Manifest: %s", args.out_dir / OUTPUT_MANIFEST_NAME)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="Pipeline output directory.")
    parser.add_argument(
        "--slug",
        action="append",
        default=None,
        help="Limit to a work slug; repeat for multiple slugs.",
    )
    parser.add_argument(
        "--system",
        action="append",
        type=int,
        default=None,
        help="Limit to a 1-based system index; repeat for multiple systems.",
    )
    parser.add_argument(
        "--measure",
        action="append",
        type=int,
        default=None,
        help="Limit to a 1-based system-local measure; repeat for multiple measures.",
    )
    parser.add_argument(
        "--task-kind",
        choices=(*TASK_KINDS, "all"),
        default="all",
        help="Task rows to emit; all emits direct then candidate-assisted rows.",
    )
    parser.add_argument(
        "--max-candidates",
        type=_positive_int,
        default=DEFAULT_MAX_CANDIDATES,
        help=f"Blind candidate cap per measure; defaults to {DEFAULT_MAX_CANDIDATES}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing generated views and candidate artifacts.",
    )
    return parser


def build_vlm_notehead_localization_inputs(
    out_dir: Path,
    *,
    selected_slugs: set[str] | None = None,
    selected_systems: set[int] | None = None,
    selected_measures: set[int] | None = None,
    task_kind: TaskSelection = "all",
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Build localization views and return deterministic task manifest records."""
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    if task_kind not in (*TASK_KINDS, "all"):
        raise ValueError(f"Unsupported task kind: {task_kind!r}")

    input_manifest_path = out_dir / INPUT_MANIFEST_NAME
    if not input_manifest_path.exists():
        raise FileNotFoundError(
            f"VLM melody input manifest not found: {input_manifest_path}. "
            "Run scripts/build_vlm_melody_inputs.py first."
        )

    all_records = sorted(_read_jsonl(input_manifest_path), key=_record_sort_key)
    _validate_unique_measure_records(all_records)
    selected_records = list(
        _selected_records(
            all_records,
            selected_slugs=selected_slugs,
            selected_systems=selected_systems,
            selected_measures=selected_measures,
        )
    )

    output_records: list[dict[str, Any]] = []
    for base_record in selected_records:
        system_records = [
            record
            for record in all_records
            if record["slug"] == base_record["slug"]
            and int(record["system_index"]) == int(base_record["system_index"])
        ]
        artifacts = build_notehead_localization_artifacts_for_record(
            out_dir,
            input_manifest_path=input_manifest_path,
            base_record=base_record,
            system_records=system_records,
            max_candidates=max_candidates,
            overwrite=overwrite,
        )
        for selected_task_kind in _expanded_task_kinds(task_kind):
            output_records.append(
                _manifest_record(
                    base_record,
                    task_kind=selected_task_kind,
                    artifacts=artifacts,
                )
            )

    experiment_ids = [record["experiment_id"] for record in output_records]
    if len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError("Generated duplicate notehead-localization experiment IDs")
    _write_jsonl(out_dir / OUTPUT_MANIFEST_NAME, output_records)
    return output_records


def build_notehead_localization_artifacts_for_record(
    out_dir: Path,
    *,
    input_manifest_path: Path,
    base_record: dict[str, Any],
    system_records: Sequence[dict[str, Any]],
    max_candidates: int,
    overwrite: bool,
    output_root: Path | None = None,
    include_evaluation_metadata: bool = True,
) -> dict[str, Any]:
    """Build GT-blind localization artifacts for one explicit measure record.

    ``output_root`` lets spike orchestration keep derived artifacts isolated.
    Existing callers retain the historical ``vlm_notehead_localization`` path.
    """
    slug = str(base_record["slug"])
    system_index = int(base_record["system_index"])
    measure_index = int(base_record["system_measure_index"])
    paths = base_record.get("paths")
    if not isinstance(paths, dict):
        raise ValueError(f"Missing paths for {slug} system {system_index} measure {measure_index}")

    source_context_path = _required_input_path(out_dir, paths, "context")
    source_raw_path = _required_input_path(out_dir, paths, "measure_raw")
    source_system_path = _required_input_path(out_dir, paths, "source_system")
    source_context = json.loads(source_context_path.read_text(encoding="utf-8"))
    staff_lines = _staff_lines_in_raw_image(source_context)
    staff_spacing = _staff_spacing(staff_lines)

    with Image.open(source_raw_path) as opened_raw:
        raw_image = opened_raw.convert("RGB")
    with Image.open(source_system_path) as opened_system:
        system_image = opened_system.convert("RGB")

    if output_root is None:
        output_root = out_dir / slug / "vlm_notehead_localization"
    output_dir = output_root / f"system_{system_index:03d}" / f"measure_{measure_index:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "context": output_dir / "context.png",
        "detail": output_dir / "detail.png",
        "binary": output_dir / "binary.png",
        "candidate_gallery": output_dir / "candidate_gallery.png",
        "candidates": output_dir / "candidates.json",
    }

    context_image, context_geometry = make_context_image(
        system_image,
        system_records=system_records,
        target_record=base_record,
        staff_spacing=staff_spacing,
    )
    detail_image = make_detail_image(raw_image)
    binary_image, binary_threshold = make_binary_image(raw_image)
    _save_image(output_paths["context"], context_image, overwrite=overwrite)
    _save_image(output_paths["detail"], detail_image, overwrite=overwrite)
    _save_image(output_paths["binary"], binary_image, overwrite=overwrite)

    _write_candidate_outputs(
        raw_image,
        output_paths=output_paths,
        input_manifest_path=input_manifest_path,
        source_context_path=source_context_path,
        source_raw_path=source_raw_path,
        base_record=base_record,
        staff_lines=staff_lines,
        staff_spacing=staff_spacing,
        max_candidates=max_candidates,
        overwrite=overwrite,
    )

    coordinate_ground_truth_path = None
    canonical_ground_truth_path = None
    if include_evaluation_metadata:
        coordinate_ground_truth_path = _discover_coordinate_ground_truth_path(
            slug,
            system_index=system_index,
            measure_index=measure_index,
        )
        canonical_ground_truth_path = CANONICAL_GROUND_TRUTH_DIR / f"{slug}.json"
    manifest_provenance = {
        "builder": "scripts.build_vlm_notehead_localization_inputs",
        "input_manifest_path": _display_path(input_manifest_path),
        "source_system_path": _display_path(source_system_path),
        "source_measure_path": _display_path(source_raw_path),
        "source_context_path": _display_path(source_context_path),
        "ground_truth_usage": (
            "paths are evaluation metadata only; no ground-truth content is read by this builder"
        ),
        "views": {
            "context": {
                "source_crop_x_bounds_px": context_geometry["source_crop_x_bounds_px"],
                "target_bounds_x_px_in_context": context_geometry["target_bounds_x_px_in_context"],
                "white_margin_px": context_geometry["white_margin_px"],
                "target_markers_touch_music_pixels": False,
            },
            "detail": {
                "source": "untouched raw measure geometry",
                "transform": "grayscale, autocontrast, Lanczos 4x",
            },
            "binary": {
                "source": "untouched raw measure geometry",
                "transform": "grayscale, autocontrast, deterministic threshold, nearest 4x",
                "threshold": binary_threshold,
            },
            "candidate_gallery": {
                "source": "blind staff-grid-density v2 candidates on raw measure",
                "labels_touch_patch_pixels": False,
            },
        },
    }
    return {
        "paths": output_paths,
        "source_context_path": source_context_path,
        "coordinate_ground_truth_path": (
            _display_path(coordinate_ground_truth_path)
            if coordinate_ground_truth_path is not None
            else None
        ),
        "canonical_ground_truth_path": (
            _display_path(canonical_ground_truth_path)
            if canonical_ground_truth_path is not None
            else None
        ),
        "source_context": {
            "clef": source_context.get("clef_hint"),
            "time_signature": source_context.get("time_signature_hint"),
            "key": source_context.get("key_hint"),
            "allow_pickup": bool(source_context.get("allow_pickup", False)),
            "expected_measure_beats": source_context.get("expected_measure_beats"),
            "staff_lines_y_px": staff_lines,
            "raw_image_size": {"width": raw_image.width, "height": raw_image.height},
        },
        "provenance": manifest_provenance,
    }


def make_context_image(
    system_image: Image.Image,
    *,
    system_records: Sequence[dict[str, Any]],
    target_record: dict[str, Any],
    staff_spacing: float,
) -> tuple[Image.Image, dict[str, Any]]:
    """Return one-neighbor context with target markers outside musical pixels."""
    ordered = sorted(system_records, key=lambda record: int(record["system_measure_index"]))
    target_key = _measure_key(target_record)
    target_position = next(
        (index for index, record in enumerate(ordered) if _measure_key(record) == target_key),
        None,
    )
    if target_position is None:
        raise ValueError(f"Target measure is absent from its source-system records: {target_key}")

    first_record = ordered[max(0, target_position - 1)]
    last_record = ordered[min(len(ordered) - 1, target_position + 1)]
    context_left, _ = _x_bounds(first_record)
    _, context_right = _x_bounds(last_record)
    target_left, target_right = _x_bounds(target_record)
    context_left = max(0, min(system_image.width - 1, context_left))
    context_right = max(context_left + 1, min(system_image.width, context_right))
    target_left = max(context_left, min(context_right - 1, target_left))
    target_right = max(target_left + 1, min(context_right, target_right))

    source_crop = system_image.convert("RGB").crop(
        (context_left, 0, context_right, system_image.height)
    )
    margin = max(CONTEXT_MIN_MARGIN_PX, round(staff_spacing))
    canvas = Image.new(
        "RGB",
        (source_crop.width, source_crop.height + 2 * margin),
        "white",
    )
    canvas.paste(source_crop, (0, margin))
    marker_left = target_left - context_left
    marker_right = min(source_crop.width - 1, target_right - context_left)
    draw = ImageDraw.Draw(canvas)
    for marker_x in (marker_left, marker_right):
        draw.line((marker_x, 0, marker_x, margin - 1), fill=TARGET_BOUNDARY_COLOR, width=2)
        draw.line(
            (
                marker_x,
                margin + source_crop.height,
                marker_x,
                canvas.height - 1,
            ),
            fill=TARGET_BOUNDARY_COLOR,
            width=2,
        )

    return canvas, {
        "source_crop_x_bounds_px": {"left": context_left, "right": context_right},
        "target_bounds_x_px_in_context": {"left": marker_left, "right": marker_right},
        "white_margin_px": margin,
        "neighbor_measure_indices": [
            int(record["system_measure_index"])
            for record in ordered[max(0, target_position - 1) : target_position + 2]
            if _measure_key(record) != target_key
        ],
    }


def make_detail_image(raw_image: Image.Image) -> Image.Image:
    """Return a grayscale/autocontrast 4x view without annotations."""
    normalized = ImageOps.autocontrast(ImageOps.grayscale(raw_image))
    return normalized.resize(
        (normalized.width * DETAIL_SCALE, normalized.height * DETAIL_SCALE),
        Image.Resampling.LANCZOS,
    )


def make_binary_image(raw_image: Image.Image) -> tuple[Image.Image, int]:
    """Return a geometry-preserving binary 4x view and its threshold."""
    normalized = ImageOps.autocontrast(ImageOps.grayscale(raw_image))
    threshold = estimate_ink_threshold(normalized)
    binary = normalized.point(lambda value: 0 if value < threshold else 255, mode="1")
    upscaled = binary.resize(
        (binary.width * DETAIL_SCALE, binary.height * DETAIL_SCALE),
        Image.Resampling.NEAREST,
    )
    return upscaled, threshold


def make_candidate_gallery(
    raw_image: Image.Image,
    candidates: Sequence[dict[str, Any]],
    *,
    staff_spacing: float,
) -> tuple[Image.Image, dict[str, Any]]:
    """Compose left-to-right candidate patches with external text headers."""
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            float(candidate["center"]["x"]),
            float(candidate["center"]["y"]),
            str(candidate["id"]),
        ),
    )
    if not ordered:
        empty = Image.new("L", (360, 64), 255)
        ImageDraw.Draw(empty).text(
            (12, 24),
            "No blind candidates generated",
            fill=0,
            font=ImageFont.load_default(),
        )
        return empty, {
            "candidate_ids_left_to_right": [],
            "crop_span_staff_spacings": GALLERY_CROP_STAFF_SPACINGS,
            "upscale_factor": GALLERY_SCALE,
            "header_height_px": GALLERY_HEADER_HEIGHT_PX,
            "labels_touch_patch_pixels": False,
            "cells": [],
        }

    normalized = ImageOps.autocontrast(ImageOps.grayscale(raw_image))
    crop_size = max(24, round(staff_spacing * GALLERY_CROP_STAFF_SPACINGS))
    patches: list[tuple[dict[str, Any], Image.Image, tuple[int, int, int, int]]] = []
    for candidate in ordered:
        center_x = float(candidate["center"]["x"])
        center_y = float(candidate["center"]["y"])
        left = round(center_x - crop_size / 2)
        top = round(center_y - crop_size / 2)
        source_bbox = (left, top, left + crop_size, top + crop_size)
        patch = _crop_with_white_padding(normalized, source_bbox).resize(
            (crop_size * GALLERY_SCALE, crop_size * GALLERY_SCALE),
            Image.Resampling.LANCZOS,
        )
        patches.append((candidate, patch, source_bbox))

    columns = min(GALLERY_MAX_COLUMNS, len(patches))
    rows = math.ceil(len(patches) / columns)
    cell_width = patches[0][1].width
    cell_height = GALLERY_HEADER_HEIGHT_PX + patches[0][1].height
    gallery = Image.new(
        "L",
        (
            columns * cell_width + (columns - 1) * GALLERY_GAP_PX,
            rows * cell_height + (rows - 1) * GALLERY_GAP_PX,
        ),
        255,
    )
    cells: list[dict[str, Any]] = []
    for index, (candidate, patch, source_bbox) in enumerate(patches):
        row, column = divmod(index, columns)
        cell_x = column * (cell_width + GALLERY_GAP_PX)
        cell_y = row * (cell_height + GALLERY_GAP_PX)
        normalized_x = float(candidate["normalized_center"]["x"])
        label = f"{candidate['id']}  x={normalized_x:.3f}"
        cell = make_gallery_cell(patch, label)
        gallery.paste(cell, (cell_x, cell_y))
        cells.append(
            {
                "candidate_id": candidate["id"],
                "normalized_x": normalized_x,
                "source_crop_bbox_px": {
                    "left": source_bbox[0],
                    "top": source_bbox[1],
                    "right": source_bbox[2],
                    "bottom": source_bbox[3],
                },
                "gallery_cell": {"row": row, "column": column},
                "header_bbox_px_in_cell": {
                    "left": 0,
                    "top": 0,
                    "right": cell_width,
                    "bottom": GALLERY_HEADER_HEIGHT_PX,
                },
                "patch_bbox_px_in_cell": {
                    "left": 0,
                    "top": GALLERY_HEADER_HEIGHT_PX,
                    "right": cell_width,
                    "bottom": cell_height,
                },
            }
        )

    return gallery, {
        "candidate_ids_left_to_right": [candidate["id"] for candidate in ordered],
        "crop_span_staff_spacings": GALLERY_CROP_STAFF_SPACINGS,
        "crop_size_px_in_source": crop_size,
        "upscale_factor": GALLERY_SCALE,
        "header_height_px": GALLERY_HEADER_HEIGHT_PX,
        "labels_touch_patch_pixels": False,
        "cells": cells,
    }


def make_gallery_cell(patch: Image.Image, label: str) -> Image.Image:
    """Place a label in a header strip, then paste the untouched patch below it."""
    patch_image = patch.convert("L")
    cell = Image.new(
        "L",
        (patch_image.width, patch_image.height + GALLERY_HEADER_HEIGHT_PX),
        255,
    )
    draw = ImageDraw.Draw(cell)
    draw.text((6, 6), label, fill=0, font=ImageFont.load_default())
    cell.paste(patch_image, (0, GALLERY_HEADER_HEIGHT_PX))
    return cell


def _write_candidate_outputs(
    raw_image: Image.Image,
    *,
    output_paths: dict[str, Path],
    input_manifest_path: Path,
    source_context_path: Path,
    source_raw_path: Path,
    base_record: dict[str, Any],
    staff_lines: list[int],
    staff_spacing: float,
    max_candidates: int,
    overwrite: bool,
) -> None:
    candidates_path = output_paths["candidates"]
    gallery_path = output_paths["candidate_gallery"]
    if not overwrite and candidates_path.exists() and gallery_path.exists():
        existing = json.loads(candidates_path.read_text(encoding="utf-8"))
        existing_cap = existing.get("max_candidates")
        if existing_cap != max_candidates:
            raise ValueError(
                f"Existing candidate cap is {existing_cap!r}, requested {max_candidates}; "
                "rerun with --overwrite"
            )
        return

    detected = detect_staff_grid_density_candidates(
        raw_image,
        staff_lines=staff_lines,
        max_candidates=max_candidates,
    )
    candidates = [
        _candidate_to_json(index, candidate, image_size=raw_image.size)
        for index, candidate in enumerate(detected[:max_candidates], start=1)
    ]
    gallery, gallery_metadata = make_candidate_gallery(
        raw_image,
        candidates,
        staff_spacing=staff_spacing,
    )
    payload = {
        "schema_version": 2,
        "kind": "vlm_notehead_candidates",
        "strategy": "staff-grid-density",
        "strategy_version": 2,
        "slug": str(base_record["slug"]),
        "system_index": int(base_record["system_index"]),
        "system_measure_index": int(base_record["system_measure_index"]),
        "global_measure_index": int(base_record["global_measure_index"]),
        "source_image_path": _display_path(source_raw_path),
        "source_image_size_px": {"width": raw_image.width, "height": raw_image.height},
        "coordinate_space": "raw measure pixels, origin at top-left",
        "staff_lines_y_px": staff_lines,
        "staff_spacing_px": round(staff_spacing, 3),
        "max_candidates": max_candidates,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "gallery": {
            "path": _display_path(gallery_path),
            **gallery_metadata,
        },
        "provenance": {
            "input_manifest_path": _display_path(input_manifest_path),
            "source_context_path": _display_path(source_context_path),
            "detector": (
                "scripts.build_vlm_notehead_candidates." "detect_staff_grid_density_candidates"
            ),
            "detector_rank_becomes_candidate_id": True,
            "candidate_generation_is_blind": True,
            "ground_truth_files_read": [],
            "description": (
                "Candidates are generated from raw image pixels and staff geometry only; "
                "ground truth is reserved for later evaluation."
            ),
        },
    }
    _write_json(candidates_path, payload)
    gallery.save(gallery_path)


def _candidate_to_json(
    index: int,
    candidate: GridCandidate,
    *,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    width, height = image_size
    center_x, center_y = candidate.center
    return {
        "id": f"c{index:03d}",
        "rank": index,
        "bbox": {
            "left": candidate.bbox[0],
            "top": candidate.bbox[1],
            "right": candidate.bbox[2] + 1,
            "bottom": candidate.bbox[3] + 1,
        },
        "center": {"x": round(center_x, 2), "y": round(center_y, 2)},
        "normalized_center": {
            "x": round(center_x / width, 6) if width else 0.0,
            "y": round(center_y / height, 6) if height else 0.0,
        },
        "score": candidate.score,
        "features": candidate.features,
    }


def _manifest_record(
    base_record: dict[str, Any],
    *,
    task_kind: TaskKind,
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    slug = str(base_record["slug"])
    system_index = int(base_record["system_index"])
    measure_index = int(base_record["system_measure_index"])
    paths = artifacts["paths"]
    if task_kind == DIRECT_LOCALIZATION:
        image_roles = ("context", "detail", "binary")
        candidate_artifact_path = None
    else:
        image_roles = ("context", "detail", "candidate_gallery")
        candidate_artifact_path = _display_path(paths["candidates"])

    return {
        "schema_version": 1,
        "experiment_id": (
            f"vlm-notehead-localization:{slug}:s{system_index:03d}:"
            f"m{measure_index:03d}:{task_kind}"
        ),
        "task_kind": task_kind,
        "slug": slug,
        "system_index": system_index,
        "system_measure_index": measure_index,
        "global_measure_index": int(base_record["global_measure_index"]),
        "context_path": _display_path(artifacts["source_context_path"]),
        "images": [{"role": role, "path": _display_path(paths[role])} for role in image_roles],
        "candidate_artifact_path": candidate_artifact_path,
        "coordinate_ground_truth_path": artifacts["coordinate_ground_truth_path"],
        "canonical_ground_truth_path": artifacts["canonical_ground_truth_path"],
        "source_context": artifacts["source_context"],
        "provenance": artifacts["provenance"],
    }


def _selected_records(
    records: Iterable[dict[str, Any]],
    *,
    selected_slugs: set[str] | None,
    selected_systems: set[int] | None,
    selected_measures: set[int] | None,
) -> Iterable[dict[str, Any]]:
    for record in records:
        if selected_slugs is not None and str(record["slug"]) not in selected_slugs:
            continue
        if selected_systems is not None and int(record["system_index"]) not in selected_systems:
            continue
        if (
            selected_measures is not None
            and int(record["system_measure_index"]) not in selected_measures
        ):
            continue
        yield record


def _expanded_task_kinds(task_kind: TaskSelection) -> tuple[TaskKind, ...]:
    if task_kind == "all":
        return TASK_KINDS
    return (task_kind,)


def _validate_unique_measure_records(records: Sequence[dict[str, Any]]) -> None:
    seen: set[tuple[str, int, int]] = set()
    for record in records:
        key = _measure_key(record)
        if key in seen:
            raise ValueError(f"Duplicate VLM melody input record: {key}")
        seen.add(key)


def _record_sort_key(record: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(record["slug"]),
        int(record["system_index"]),
        int(record["system_measure_index"]),
        int(record["global_measure_index"]),
    )


def _measure_key(record: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(record["slug"]),
        int(record["system_index"]),
        int(record["system_measure_index"]),
    )


def _x_bounds(record: dict[str, Any]) -> tuple[int, int]:
    bounds = record.get("x_bounds_px")
    if not isinstance(bounds, dict) or "left" not in bounds or "right" not in bounds:
        raise ValueError(f"Missing source-system x bounds for {_measure_key(record)}")
    left, right = int(bounds["left"]), int(bounds["right"])
    if right <= left:
        raise ValueError(f"Invalid source-system x bounds for {_measure_key(record)}: {bounds}")
    return left, right


def _staff_lines_in_raw_image(context: dict[str, Any]) -> list[int]:
    values = context.get("staff_lines_y_px_in_system")
    if not isinstance(values, list) or len(values) != 5:
        raise ValueError(
            "Expected five staff_lines_y_px_in_system values in source context: "
            f"{context.get('paths', {})}"
        )
    lines = [int(value) for value in values]
    if lines != sorted(lines) or len(set(lines)) != 5:
        raise ValueError(f"Invalid staff-line geometry: {lines!r}")
    return lines


def _staff_spacing(staff_lines: Sequence[int]) -> float:
    gaps = [right - left for left, right in zip(staff_lines, staff_lines[1:], strict=False)]
    spacing = sum(gaps) / len(gaps)
    if spacing <= 0:
        raise ValueError(f"Invalid staff-line spacing: {staff_lines!r}")
    return spacing


def _crop_with_white_padding(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
) -> Image.Image:
    left, top, right, bottom = bbox
    patch = Image.new(image.mode, (right - left, bottom - top), 255)
    source_left = max(0, left)
    source_top = max(0, top)
    source_right = min(image.width, right)
    source_bottom = min(image.height, bottom)
    if source_right > source_left and source_bottom > source_top:
        source = image.crop((source_left, source_top, source_right, source_bottom))
        patch.paste(source, (source_left - left, source_top - top))
    return patch


def _discover_coordinate_ground_truth_path(
    slug: str,
    *,
    system_index: int,
    measure_index: int,
) -> Path | None:
    fixture_name = f"{slug}_system_{system_index:03d}_measure_{measure_index:03d}.json"
    fixture_path = COORDINATE_GROUND_TRUTH_DIR / fixture_name
    return fixture_path if fixture_path.is_file() else None


def _required_input_path(out_dir: Path, paths: dict[str, Any], key: str) -> Path:
    value = paths.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Missing required base path {key!r}: {paths}")
    path = _resolve_path(out_dir, value)
    if not path.exists():
        raise FileNotFoundError(f"Required base artifact not found: {path}")
    return path


def _resolve_path(out_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    if path.parts[:1] == (out_dir.name,) and out_dir.is_absolute():
        return out_dir.parent / path
    return out_dir / path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _save_image(path: Path, image: Image.Image, *, overwrite: bool) -> None:
    if overwrite or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
