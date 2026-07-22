"""Build no-API notehead-candidate review artifacts for one VLM melody crop.

This is a spike-only image preprocessing script. It reads an existing measure
crop from `scripts/build_vlm_melody_inputs.py`, finds conservative dark-ink
components that may contain noteheads, and writes human-review artifacts next to
the selected measure image.

Example:
    uv run python scripts/build_vlm_notehead_candidates.py out \\
        --slug jaime-llanos_12_aviador_pasillo_fulgencio-garcia \\
        --system 1 --measure 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from score2abc.utils import get_logger  # noqa: E402
from score2abc.utils.imaging import estimate_ink_threshold  # noqa: E402
from scripts.build_vlm_melody_inputs import build_vlm_melody_inputs  # noqa: E402

BASE_MANIFEST_NAME = "vlm_melody_inputs_manifest.jsonl"
SOURCE_VARIANTS = ("staff", "raw", "staff_overlay")
SourceVariant = Literal["staff", "raw", "staff_overlay"]
STRATEGIES = ("connected-components", "staff-grid-density")
Strategy = Literal["connected-components", "staff-grid-density"]
STAFF_GRID_STRATEGY = "staff-grid-density"
V2_MAX_CANDIDATES = 12


@dataclass(frozen=True)
class Component:
    bbox: tuple[int, int, int, int]
    area: int

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)


@dataclass(frozen=True)
class DetectionStages:
    threshold: int
    threshold_mask: list[list[bool]]
    suppression_rows: list[int]
    suppressed_mask: list[list[bool]]
    raw_components: list[Component]
    candidates: list[Component]


@dataclass(frozen=True)
class GridCandidate:
    bbox: tuple[int, int, int, int]
    score: float
    features: dict[str, float]

    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1


@dataclass(frozen=True)
class StaffGridStages:
    threshold: int
    threshold_mask: list[list[bool]]
    pitch_rows: list[float]
    staff_spacing: float
    window_width: int
    window_height: int
    nms_distance: float
    max_candidates: int
    scored_windows: list[tuple[float, float, float]]
    candidates: list[GridCandidate]


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logger = get_logger("score2abc.build_vlm_notehead_candidates")

    try:
        artifact = build_notehead_candidates_for_measure(
            args.out_dir,
            slug=args.slug,
            system_index=args.system,
            measure_index=args.measure,
            source_variant=args.source_variant,
            strategy=args.strategy,
            ground_truth_path=args.ground_truth,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Wrote notehead candidates JSON: %s", artifact["json_path"])
    logger.info("Wrote notehead candidates overlay: %s", artifact["overlay_path"])
    if artifact.get("heatmap_path"):
        logger.info("Wrote notehead density heatmap: %s", artifact["heatmap_path"])
    if artifact.get("comparison_overlay_path"):
        logger.info("Wrote notehead GT comparison overlay: %s", artifact["comparison_overlay_path"])
    if artifact.get("contact_sheet_path"):
        logger.info("Wrote notehead candidates contact sheet: %s", artifact["contact_sheet_path"])
    for path in artifact.get("diagnostic_paths", {}).values():
        logger.info("Wrote notehead diagnostic: %s", path)
    logger.info("Detected %d candidate(s)", artifact["candidate_count"])
    if artifact.get("evaluation"):
        evaluation = artifact["evaluation"]
        logger.info(
            "Evaluation TP=%d FP=%d FN=%d recall=%.3f tolerance_px=%.2f",
            evaluation["true_positives"],
            evaluation["false_positives"],
            evaluation["false_negatives"],
            evaluation["gt_recall"],
            evaluation["distance_tolerance_px"],
        )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="Pipeline output directory.")
    parser.add_argument("--slug", required=True, help="Work slug.")
    parser.add_argument("--system", required=True, type=int, help="1-based system index.")
    parser.add_argument("--measure", required=True, type=int, help="System-local measure index.")
    parser.add_argument(
        "--source-variant",
        choices=SOURCE_VARIANTS,
        default="staff",
        help="Measure crop variant to inspect; defaults to staff.",
    )
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default="connected-components",
        help="Candidate strategy; the default preserves the v1 component detector.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Optional notehead GT JSON used only after candidate generation for evaluation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing notehead-candidate artifacts.",
    )
    return parser


def build_notehead_candidates_for_measure(
    out_dir: Path,
    *,
    slug: str,
    system_index: int,
    measure_index: int,
    source_variant: SourceVariant = "staff",
    strategy: Strategy = "connected-components",
    ground_truth_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build notehead-candidate JSON and review images for one measure crop."""
    base_record = _load_or_build_base_record(
        out_dir,
        slug=slug,
        system_index=system_index,
        measure_index=measure_index,
    )
    context_path = _resolve_path(out_dir, base_record["paths"]["context"])
    context = json.loads(context_path.read_text(encoding="utf-8"))
    source_path = _source_path(out_dir, base_record, source_variant)
    output_stem = _measure_output_stem(source_path, source_variant)
    if strategy == STAFF_GRID_STRATEGY:
        return _build_staff_grid_density_artifacts(
            base_record,
            context,
            source_path=source_path,
            source_variant=source_variant,
            output_stem=output_stem,
            ground_truth_path=ground_truth_path,
            overwrite=overwrite,
        )
    json_path = source_path.with_name(f"{output_stem}_notehead_candidates.json")
    overlay_path = source_path.with_name(f"{output_stem}_notehead_candidates_overlay.png")
    diagnostic_paths = _diagnostic_paths(source_path, output_stem)
    contact_sheet_path = diagnostic_paths["contact_sheet"]

    if (
        not overwrite
        and json_path.exists()
        and overlay_path.exists()
        and contact_sheet_path.exists()
        and all(path.exists() for path in diagnostic_paths.values())
    ):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        return {
            "json_path": json_path,
            "overlay_path": overlay_path,
            "contact_sheet_path": contact_sheet_path,
            "diagnostic_paths": diagnostic_paths,
            "candidate_count": len(payload.get("candidates", [])),
        }

    image = Image.open(source_path).convert("RGB")
    staff_lines = _staff_lines_for_source(context, source_variant)
    stages = _detect_notehead_stages(image, staff_lines=staff_lines)
    ground_truth_path = _ground_truth_image_path(source_path)
    payload = _candidate_payload(
        base_record,
        context,
        source_path=source_path,
        source_variant=source_variant,
        staff_lines=staff_lines,
        stages=stages,
        diagnostic_paths=diagnostic_paths,
        ground_truth_path=ground_truth_path,
    )

    _write_json(json_path, payload)
    _write_mask_image(stages.threshold_mask, diagnostic_paths["threshold_ink_mask"])
    _write_staff_line_mask(
        image.size,
        stages.suppression_rows,
        diagnostic_paths["staff_line_mask"],
    )
    _write_mask_image(stages.suppressed_mask, diagnostic_paths["staff_suppressed_mask"])
    _write_component_overlay(image, stages.raw_components, diagnostic_paths["raw_components"])
    _write_overlay(image, stages.candidates, overlay_path)
    panels: list[tuple[str, Image.Image]] = [("original / raw", image)]
    if ground_truth_path is not None:
        panels.append(("human GT annotation", Image.open(ground_truth_path).convert("RGB")))
    panels.extend(
        [
            ("thresholded ink mask", Image.open(diagnostic_paths["threshold_ink_mask"])),
            (
                f"staff-line mask / selected rows {stages.suppression_rows}",
                Image.open(diagnostic_paths["staff_line_mask"]),
            ),
            ("after staff-line suppression", Image.open(diagnostic_paths["staff_suppressed_mask"])),
            (
                "connected components before final filtering",
                Image.open(diagnostic_paths["raw_components"]),
            ),
            ("current final candidate overlay", Image.open(overlay_path)),
        ]
    )
    _write_diagnostic_contact_sheet(panels, contact_sheet_path)
    return {
        "json_path": json_path,
        "overlay_path": overlay_path,
        "contact_sheet_path": contact_sheet_path,
        "diagnostic_paths": diagnostic_paths,
        "candidate_count": len(stages.candidates),
    }


def detect_notehead_candidates(
    image: Image.Image,
    *,
    staff_lines: list[int] | None = None,
) -> list[Component]:
    """Return conservative connected-component candidates from dark ink."""
    return _detect_notehead_stages(image, staff_lines=staff_lines or []).candidates


def detect_staff_grid_density_candidates(
    image: Image.Image,
    *,
    staff_lines: list[int],
    max_candidates: int = V2_MAX_CANDIDATES,
) -> list[GridCandidate]:
    """Generate v2 candidates without consulting any annotation or GT file."""
    return _detect_staff_grid_stages(
        image,
        staff_lines=staff_lines,
        max_candidates=max_candidates,
    ).candidates


def evaluate_notehead_candidates(
    candidates: Sequence[GridCandidate | dict[str, Any]],
    ground_truth: Sequence[dict[str, Any]],
    *,
    tolerance_px: float,
) -> dict[str, Any]:
    """Match candidate centers to GT centers using deterministic closest pairs."""
    normalized_candidates = [_candidate_center(candidate) for candidate in candidates]
    normalized_gt = [
        (
            str(item.get("id", f"n{index:03d}")),
            float(item["center"]["x"]),
            float(item["center"]["y"]),
        )
        for index, item in enumerate(ground_truth, start=1)
    ]
    possible_pairs = sorted(
        (
            math.hypot(cx - gx, cy - gy),
            candidate_index,
            gt_index,
        )
        for candidate_index, (cx, cy) in enumerate(normalized_candidates)
        for gt_index, (_, gx, gy) in enumerate(normalized_gt)
    )
    used_candidates: set[int] = set()
    used_gt: set[int] = set()
    assignments: list[dict[str, Any]] = []
    for distance, candidate_index, gt_index in possible_pairs:
        if distance > tolerance_px or candidate_index in used_candidates or gt_index in used_gt:
            continue
        used_candidates.add(candidate_index)
        used_gt.add(gt_index)
        assignments.append(
            {
                "candidate_index": candidate_index,
                "candidate_id": _candidate_id(candidates[candidate_index], candidate_index),
                "ground_truth_index": gt_index,
                "ground_truth_id": normalized_gt[gt_index][0],
                "distance_px": round(distance, 3),
            }
        )
    assignments.sort(key=lambda item: item["candidate_index"])
    true_positives = len(assignments)
    return {
        "distance_tolerance_px": round(tolerance_px, 3),
        "tolerance_derivation": (
            "caller-provided; real-artifact rule is max(4 px, " "0.55 * mean staff spacing)"
        ),
        "matching": "global closest-distance pairs with candidate/GT index tie-breaks",
        "true_positives": true_positives,
        "false_positives": len(candidates) - true_positives,
        "false_negatives": len(ground_truth) - true_positives,
        "gt_count": len(ground_truth),
        "candidate_count": len(candidates),
        "gt_recall": round(true_positives / len(ground_truth), 6) if ground_truth else 1.0,
        "assignments": assignments,
        "unmatched_candidate_indices": [
            index for index in range(len(candidates)) if index not in used_candidates
        ],
        "unmatched_ground_truth_indices": [
            index for index in range(len(ground_truth)) if index not in used_gt
        ],
    }


def _build_staff_grid_density_artifacts(
    base_record: dict[str, Any],
    context: dict[str, Any],
    *,
    source_path: Path,
    source_variant: SourceVariant,
    output_stem: str,
    ground_truth_path: Path | None,
    overwrite: bool,
) -> dict[str, Any]:
    paths = _staff_grid_artifact_paths(source_path, output_stem)
    if not overwrite and all(path.exists() for path in paths.values()):
        payload = json.loads(paths["json"].read_text(encoding="utf-8"))
        cached_ground_truth = payload.get("ground_truth_path")
        requested_ground_truth = str(ground_truth_path) if ground_truth_path else None
        if cached_ground_truth == requested_ground_truth:
            return {
                "json_path": paths["json"],
                "overlay_path": paths["overlay"],
                "heatmap_path": paths["heatmap"],
                "comparison_overlay_path": paths["comparison"],
                "candidate_count": len(payload.get("candidates", [])),
                "evaluation": payload.get("evaluation"),
            }

    image = Image.open(source_path).convert("RGB")
    staff_lines = _staff_lines_for_source(context, source_variant)
    stages = _detect_staff_grid_stages(image, staff_lines=staff_lines)

    # Candidate generation is complete before this optional GT read.
    ground_truth = _load_ground_truth(ground_truth_path) if ground_truth_path else None
    evaluation = None
    if ground_truth is not None:
        tolerance = max(4.0, stages.staff_spacing * 0.55)
        evaluation = evaluate_notehead_candidates(
            stages.candidates,
            ground_truth,
            tolerance_px=tolerance,
        )

    payload = _staff_grid_payload(
        base_record,
        context,
        source_path=source_path,
        source_variant=source_variant,
        staff_lines=staff_lines,
        stages=stages,
        paths=paths,
        ground_truth_path=ground_truth_path,
        evaluation=evaluation,
    )
    _write_json(paths["json"], payload)
    _write_staff_grid_heatmap(image, stages, paths["heatmap"])
    _write_staff_grid_overlay(image, stages.candidates, paths["overlay"])
    _write_staff_grid_comparison(
        image,
        stages.candidates,
        ground_truth or [],
        evaluation,
        paths["comparison"],
    )
    return {
        "json_path": paths["json"],
        "overlay_path": paths["overlay"],
        "heatmap_path": paths["heatmap"],
        "comparison_overlay_path": paths["comparison"],
        "candidate_count": len(stages.candidates),
        "evaluation": evaluation,
    }


def _detect_notehead_stages(
    image: Image.Image,
    *,
    staff_lines: list[int],
) -> DetectionStages:
    gray = ImageOps.grayscale(image)
    threshold = estimate_ink_threshold(gray)
    width, height = gray.size
    mask = [[gray.getpixel((x, y)) < threshold for x in range(width)] for y in range(height)]
    suppressed_mask = [row[:] for row in mask]
    suppression_rows = _suppress_staff_line_rows(suppressed_mask, staff_lines)
    raw_components = _connected_components(suppressed_mask)
    merged = _merge_close_components(raw_components, staff_lines=staff_lines)
    filtered = [
        component
        for component in merged
        if _looks_like_notehead_candidate(
            component,
            image_width=width,
            staff_lines=staff_lines,
        )
    ]
    return DetectionStages(
        threshold=threshold,
        threshold_mask=mask,
        suppression_rows=suppression_rows,
        suppressed_mask=suppressed_mask,
        raw_components=raw_components,
        candidates=sorted(
            filtered,
            key=lambda component: (component.center[0], component.center[1]),
        ),
    )


def _detect_staff_grid_stages(
    image: Image.Image,
    *,
    staff_lines: list[int],
    max_candidates: int = V2_MAX_CANDIDATES,
) -> StaffGridStages:
    if len(staff_lines) != 5:
        raise ValueError(
            "Staff-grid density requires exactly five staff lines, " f"got {staff_lines!r}"
        )
    spacing = _staff_spacing(staff_lines)
    if spacing is None or spacing <= 0:
        raise ValueError(f"Invalid staff-line geometry: {staff_lines!r}")
    gray = ImageOps.grayscale(image)
    threshold = estimate_ink_threshold(gray)
    width, height = gray.size
    mask = [[gray.getpixel((x, y)) < threshold for x in range(width)] for y in range(height)]
    half_space = spacing / 2.0
    pitch_rows = _derived_pitch_rows(staff_lines, height=height, half_space=half_space)
    window_width = max(9, round(spacing * 0.82))
    window_height = max(9, round(spacing * 0.76))
    nms_distance = max(4.0, spacing * 0.9)
    scored_windows: list[tuple[float, float, float]] = []
    scored_candidates: list[GridCandidate] = []
    for row in pitch_rows:
        center_y = round(row)
        half_width = window_width // 2
        if center_y < 0 or center_y >= height:
            continue
        for center_x in range(half_width, width - (window_width - half_width - 1)):
            score, features = _score_grid_window(
                mask,
                center_x=center_x,
                center_y=center_y,
                window_width=window_width,
                window_height=window_height,
                staff_spacing=spacing,
            )
            scored_windows.append((float(center_x), row, score))
            if score < 0.19 or features["vertical_support"] < 0.22:
                continue
            left = center_x - half_width
            top = center_y - window_height // 2
            right = left + window_width - 1
            bottom = top + window_height - 1
            scored_candidates.append(
                GridCandidate(
                    bbox=(left, top, right, bottom),
                    score=score,
                    features=features,
                )
            )
    candidates = _non_max_suppress_grid_candidates(
        scored_candidates,
        distance=nms_distance,
        max_candidates=max_candidates,
    )
    return StaffGridStages(
        threshold=threshold,
        threshold_mask=mask,
        pitch_rows=pitch_rows,
        staff_spacing=spacing,
        window_width=window_width,
        window_height=window_height,
        nms_distance=nms_distance,
        max_candidates=max_candidates,
        scored_windows=scored_windows,
        candidates=candidates,
    )


def _derived_pitch_rows(
    staff_lines: list[int],
    *,
    height: int,
    half_space: float,
) -> list[float]:
    top = float(staff_lines[0])
    first_step = math.ceil((0.0 - top) / half_space)
    last_step = math.floor((height - 1.0 - top) / half_space)
    return [round(top + step * half_space, 3) for step in range(first_step, last_step + 1)]


def _score_grid_window(
    mask: list[list[bool]],
    *,
    center_x: int,
    center_y: int,
    window_width: int,
    window_height: int,
    staff_spacing: float,
) -> tuple[float, dict[str, float]]:
    height = len(mask)
    width = len(mask[0]) if mask else 0
    half_width = window_width // 2
    half_height = window_height // 2
    left = center_x - half_width
    top = center_y - half_height
    pixels = [
        [
            0 <= x < width and 0 <= y < height and mask[y][x]
            for x in range(left, left + window_width)
        ]
        for y in range(top, top + window_height)
    ]
    row_counts = [sum(row) for row in pixels]
    column_counts = [
        sum(pixels[row][column] for row in range(window_height)) for column in range(window_width)
    ]
    ink_count = sum(row_counts)
    window_area = window_width * window_height
    core_radius_x = max(2.0, window_width * 0.31)
    core_radius_y = max(2.0, window_height * 0.34)
    core_pixels = 0
    for row_index, row in enumerate(pixels):
        normalized_y = (row_index - (window_height - 1) / 2) / core_radius_y
        for column_index, value in enumerate(row):
            if not value:
                continue
            normalized_x = (column_index - (window_width - 1) / 2) / core_radius_x
            if normalized_x * normalized_x + normalized_y * normalized_y <= 1.0:
                core_pixels += 1
    row_threshold = max(2, round(window_width * 0.18))
    column_threshold = max(2, round(window_height * 0.22))
    vertical_support = sum(count >= row_threshold for count in row_counts) / window_height
    horizontal_support = sum(count >= column_threshold for count in column_counts) / window_width
    ink_density = ink_count / window_area
    core_area = math.pi * core_radius_x * core_radius_y
    core_density = min(1.0, core_pixels / core_area)
    row_peak_density = max(row_counts, default=0) / window_width
    column_peak_density = max(column_counts, default=0) / window_height
    line_dominance = row_peak_density * (1.0 - vertical_support)

    stem_x = center_x
    stem_top = max(0, center_y - round(staff_spacing * 0.9))
    stem_bottom = min(height, center_y + round(staff_spacing * 0.9) + 1)
    stem_pixels = sum(
        mask[y][x]
        for y in range(stem_top, stem_bottom)
        for x in range(max(0, stem_x - 1), min(width, stem_x + 2))
    )
    stem_area = max(1, (stem_bottom - stem_top) * 3)
    stem_evidence = min(1.0, stem_pixels / stem_area * 2.5)
    score = min(
        1.0,
        max(
            0.0,
            0.46 * core_density
            + 0.24 * vertical_support
            + 0.18 * horizontal_support
            + 0.12 * ink_density
            + 0.04 * stem_evidence
            - 0.16 * line_dominance,
        ),
    )
    features = {
        "ink_count": float(ink_count),
        "ink_density": round(ink_density, 6),
        "core_density": round(core_density, 6),
        "vertical_support": round(vertical_support, 6),
        "horizontal_support": round(horizontal_support, 6),
        "row_peak_density": round(row_peak_density, 6),
        "column_peak_density": round(column_peak_density, 6),
        "line_dominance": round(line_dominance, 6),
        "stem_evidence": round(stem_evidence, 6),
    }
    return round(score, 6), features


def _non_max_suppress_grid_candidates(
    candidates: Sequence[GridCandidate],
    *,
    distance: float,
    max_candidates: int,
) -> list[GridCandidate]:
    ordered = sorted(
        candidates,
        key=lambda candidate: (-candidate.score, candidate.center[0], candidate.center[1]),
    )
    selected: list[GridCandidate] = []
    for candidate in ordered:
        cx, cy = candidate.center
        if any(
            math.hypot(cx - selected_x, cy - selected_y) < distance
            for selected_x, selected_y in (item.center for item in selected)
        ):
            continue
        selected.append(candidate)
        if len(selected) >= max_candidates:
            break
    return selected


def _candidate_center(candidate: GridCandidate | dict[str, Any]) -> tuple[float, float]:
    if isinstance(candidate, GridCandidate):
        return candidate.center
    center = candidate["center"]
    return float(center["x"]), float(center["y"])


def _candidate_id(candidate: GridCandidate | dict[str, Any], index: int) -> str:
    if isinstance(candidate, dict):
        return str(candidate.get("id", f"c{index + 1:03d}"))
    return f"c{index + 1:03d}"


def _staff_grid_artifact_paths(source_path: Path, output_stem: str) -> dict[str, Path]:
    suffix = f"_notehead_candidates_{STAFF_GRID_STRATEGY}_v2"
    return {
        "json": source_path.with_name(f"{output_stem}{suffix}.json"),
        "overlay": source_path.with_name(f"{output_stem}{suffix}_overlay.png"),
        "heatmap": source_path.with_name(f"{output_stem}{suffix}_heatmap.png"),
        "comparison": source_path.with_name(f"{output_stem}{suffix}_gt_compare.png"),
    }


def _load_ground_truth(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    noteheads = payload.get("noteheads") if isinstance(payload, dict) else payload
    if not isinstance(noteheads, list):
        raise ValueError(f"Ground-truth JSON has no noteheads list: {path}")
    return [item for item in noteheads if isinstance(item, dict) and "center" in item]


def _staff_grid_payload(
    base_record: dict[str, Any],
    context: dict[str, Any],
    *,
    source_path: Path,
    source_variant: SourceVariant,
    staff_lines: list[int],
    stages: StaffGridStages,
    paths: dict[str, Path],
    ground_truth_path: Path | None,
    evaluation: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "kind": "vlm_notehead_candidates",
        "strategy": STAFF_GRID_STRATEGY,
        "slug": base_record["slug"],
        "system_index": int(base_record["system_index"]),
        "system_measure_index": int(base_record["system_measure_index"]),
        "global_measure_index": int(base_record["global_measure_index"]),
        "display_measure_number": int(base_record["display_measure_number"]),
        "source_variant": source_variant,
        "source_image_path": str(source_path),
        "context_path": base_record["paths"]["context"],
        "staff_lines_y_px": staff_lines,
        "heuristic": {
            "description": (
                "scan staff-derived half-space rows with notehead-sized local density "
                "windows; score compact ink thicker than a horizontal staff line"
            ),
            "staff_lines_removed_from_detection_image": False,
            "vertical_stem_evidence_required": False,
            "missing_true_noteheads_is_worse_than_false_positives": True,
        },
        "staff_grid": {
            "staff_spacing_px": round(stages.staff_spacing, 3),
            "half_space_px": round(stages.staff_spacing / 2.0, 3),
            "pitch_rows_y_px": stages.pitch_rows,
            "window_width_px": stages.window_width,
            "window_height_px": stages.window_height,
            "nms_distance_px": round(stages.nms_distance, 3),
            "max_candidates": stages.max_candidates,
        },
        "threshold": stages.threshold,
        "scanned_window_count": len(stages.scored_windows),
        "candidate_count": len(stages.candidates),
        "ground_truth_path": str(ground_truth_path) if ground_truth_path else None,
        "evaluation": evaluation,
        "artifacts": {key: str(path) for key, path in paths.items()},
        "diagnostics": {
            "density_heatmap": str(paths["heatmap"]),
            "candidate_overlay": str(paths["overlay"]),
            "ground_truth_comparison": str(paths["comparison"]),
        },
        "candidates": [
            _grid_candidate_to_json(index, candidate, staff_lines=staff_lines)
            for index, candidate in enumerate(stages.candidates, start=1)
        ],
        "source_context": {
            "clef_hint": context.get("clef_hint"),
            "time_signature_hint": context.get("time_signature_hint"),
            "expected_measure_beats": context.get("expected_measure_beats"),
        },
    }


def _grid_candidate_to_json(
    index: int,
    candidate: GridCandidate,
    *,
    staff_lines: list[int],
) -> dict[str, Any]:
    center_x, center_y = candidate.center
    payload: dict[str, Any] = {
        "id": f"c{index:03d}",
        "bbox": {
            "left": candidate.bbox[0],
            "top": candidate.bbox[1],
            "right": candidate.bbox[2] + 1,
            "bottom": candidate.bbox[3] + 1,
        },
        "center": {"x": round(center_x, 2), "y": round(center_y, 2)},
        "score": candidate.score,
        "features": candidate.features,
    }
    helper = _staff_position_helper(center_y, staff_lines)
    if helper is not None:
        payload["staff_position"] = helper
    return payload


def _write_staff_grid_heatmap(
    image: Image.Image,
    stages: StaffGridStages,
    output_path: Path,
) -> None:
    heatmap = image.convert("RGB")
    draw = ImageDraw.Draw(heatmap, "RGBA")
    half_width = stages.window_width // 2
    half_height = stages.window_height // 2
    for center_x, center_y, score in stages.scored_windows:
        intensity = max(0, min(220, round(score * 255)))
        if intensity < 25:
            continue
        left = round(center_x) - half_width
        top = round(center_y) - half_height
        right = left + stages.window_width - 1
        bottom = top + stages.window_height - 1
        draw.rectangle((left, top, right, bottom), fill=(255, 40, 30, intensity // 3 + 15))
    draw = ImageDraw.Draw(heatmap)
    for row in stages.pitch_rows:
        y = round(row)
        if 0 <= y < heatmap.height:
            draw.line((0, y, heatmap.width - 1, y), fill=(30, 100, 220), width=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap.save(output_path)


def _write_staff_grid_overlay(
    image: Image.Image,
    candidates: Sequence[GridCandidate],
    output_path: Path,
) -> None:
    overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    for index, candidate in enumerate(candidates, start=1):
        left, top, right, bottom = candidate.bbox
        draw.rectangle((left, top, right, bottom), outline=(255, 30, 0), width=2)
        cx, cy = candidate.center
        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), outline=(0, 80, 255), width=2)
        label = f"{index}:{candidate.score:.2f}"
        draw.text((max(0, left), max(0, top - 10)), label, fill=(255, 30, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)


def _write_staff_grid_comparison(
    image: Image.Image,
    candidates: Sequence[GridCandidate],
    ground_truth: Sequence[dict[str, Any]],
    evaluation: dict[str, Any] | None,
    output_path: Path,
) -> None:
    comparison = image.convert("RGB")
    draw = ImageDraw.Draw(comparison)
    font = ImageFont.load_default()
    matched_candidates = {
        item["candidate_index"] for item in (evaluation or {}).get("assignments", [])
    }
    for index, candidate in enumerate(candidates):
        left, top, right, bottom = candidate.bbox
        color = (0, 170, 0) if index in matched_candidates else (230, 30, 30)
        draw.rectangle((left, top, right, bottom), outline=color, width=2)
        cx, cy = candidate.center
        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), outline=color, width=2)
        draw.text((max(0, left), max(0, top - 10)), f"c{index + 1:03d}", fill=color, font=font)
    matched_gt = {item["ground_truth_index"] for item in (evaluation or {}).get("assignments", [])}
    for index, item in enumerate(ground_truth):
        cx = float(item["center"]["x"])
        cy = float(item["center"]["y"])
        color = (0, 160, 0) if index in matched_gt else (150, 0, 180)
        draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), outline=color, width=2)
        draw.text((cx + 8, cy - 5), str(item.get("id", f"n{index + 1:03d}")), fill=color, font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(output_path)


def _load_or_build_base_record(
    out_dir: Path,
    *,
    slug: str,
    system_index: int,
    measure_index: int,
) -> dict[str, Any]:
    manifest_path = out_dir / BASE_MANIFEST_NAME
    if not manifest_path.exists():
        build_vlm_melody_inputs(
            out_dir,
            selected_slugs={slug},
            selected_systems={system_index},
            overwrite=False,
        )
    records = _read_jsonl(manifest_path)
    selected = _matching_records(
        records,
        slug=slug,
        system_index=system_index,
        measure_index=measure_index,
    )
    if selected and not _has_missing_base_paths(out_dir, selected[0]):
        return selected[0]

    if (out_dir / "manifest.jsonl").exists():
        build_vlm_melody_inputs(
            out_dir,
            selected_slugs={slug},
            selected_systems={system_index},
            overwrite=False,
        )
        records = _read_jsonl(manifest_path)
        selected = _matching_records(
            records,
            slug=slug,
            system_index=system_index,
            measure_index=measure_index,
        )
    if not selected:
        raise FileNotFoundError(
            f"No VLM melody input record for slug={slug!r}, system={system_index}, "
            f"measure={measure_index}. Run scripts/build_vlm_melody_inputs.py first."
        )
    if _has_missing_base_paths(out_dir, selected[0]):
        raise FileNotFoundError(f"Base measure crop paths are missing for record: {selected[0]}")
    return selected[0]


def _matching_records(
    records: list[dict[str, Any]],
    *,
    slug: str,
    system_index: int,
    measure_index: int,
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("slug") == slug
        and int(record["system_index"]) == system_index
        and int(record["system_measure_index"]) == measure_index
    ]


def _has_missing_base_paths(out_dir: Path, record: dict[str, Any]) -> bool:
    for key in ("context", "measure_raw", "measure_staff", "measure_staff_overlay"):
        value = record.get("paths", {}).get(key)
        if not isinstance(value, str) or not _resolve_path(out_dir, value).exists():
            return True
    return False


def _source_path(out_dir: Path, record: dict[str, Any], source_variant: SourceVariant) -> Path:
    path_key = {
        "raw": "measure_raw",
        "staff": "measure_staff",
        "staff_overlay": "measure_staff_overlay",
    }[source_variant]
    return _resolve_path(out_dir, record["paths"][path_key])


def _staff_lines_for_source(context: dict[str, Any], source_variant: SourceVariant) -> list[int]:
    if source_variant == "raw":
        return [int(value) for value in context.get("staff_lines_y_px_in_system", [])]
    return [int(value) for value in context.get("staff_lines_y_px_in_staff_crop", [])]


def _measure_output_stem(source_path: Path, source_variant: SourceVariant) -> str:
    if source_variant != "staff":
        return source_path.stem
    suffix = f"_{source_variant}"
    if source_path.stem.endswith(suffix):
        return source_path.stem[: -len(suffix)]
    return source_path.stem


def _diagnostic_paths(source_path: Path, output_stem: str) -> dict[str, Path]:
    return {
        "threshold_ink_mask": source_path.with_name(f"{output_stem}_notehead_ink_mask.png"),
        "staff_line_mask": source_path.with_name(f"{output_stem}_notehead_staff_line_mask.png"),
        "staff_suppressed_mask": source_path.with_name(
            f"{output_stem}_notehead_staff_suppressed.png"
        ),
        "raw_components": source_path.with_name(f"{output_stem}_notehead_components.png"),
        "contact_sheet": source_path.with_name(
            f"{output_stem}_notehead_diagnostics_contact_sheet.png"
        ),
    }


def _ground_truth_image_path(source_path: Path) -> Path | None:
    candidate = source_path.with_name(f"{source_path.stem}_notehead_gt{source_path.suffix}")
    return candidate if candidate.exists() else None


def _suppress_staff_line_rows(mask: list[list[bool]], staff_lines: list[int]) -> list[int]:
    if not mask or not mask[0] or not staff_lines:
        return []
    height = len(mask)
    width = len(mask[0])
    spacing = _staff_spacing(staff_lines)
    radius = max(1, round(spacing * 0.08)) if spacing is not None else 1
    rows_to_suppress: set[int] = set()
    for line_y in staff_lines:
        for y in range(max(0, line_y - radius), min(height, line_y + radius + 1)):
            dark_ratio = sum(mask[y]) / width
            if dark_ratio >= 0.18:
                rows_to_suppress.add(y)
    for y in rows_to_suppress:
        mask[y] = [False] * width
    return sorted(rows_to_suppress)


def _connected_components(mask: list[list[bool]]) -> list[Component]:
    if not mask or not mask[0]:
        return []
    height = len(mask)
    width = len(mask[0])
    visited = [[False for _ in range(width)] for _ in range(height)]
    components: list[Component] = []

    for y in range(height):
        for x in range(width):
            if not mask[y][x] or visited[y][x]:
                continue
            components.append(_flood_component(mask, visited, x, y))
    return components


def _flood_component(
    mask: list[list[bool]],
    visited: list[list[bool]],
    start_x: int,
    start_y: int,
) -> Component:
    height = len(mask)
    width = len(mask[0])
    queue: deque[tuple[int, int]] = deque([(start_x, start_y)])
    visited[start_y][start_x] = True
    min_x = max_x = start_x
    min_y = max_y = start_y
    area = 0

    while queue:
        x, y = queue.popleft()
        area += 1
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        for nx in range(max(0, x - 1), min(width, x + 2)):
            for ny in range(max(0, y - 1), min(height, y + 2)):
                if visited[ny][nx] or not mask[ny][nx]:
                    continue
                visited[ny][nx] = True
                queue.append((nx, ny))

    return Component(bbox=(min_x, min_y, max_x, max_y), area=area)


def _merge_close_components(
    components: list[Component],
    *,
    staff_lines: list[int],
) -> list[Component]:
    if not components:
        return []
    spacing = _staff_spacing(staff_lines) or 10.0
    pad_x = max(2, round(spacing * 0.35))
    pad_y = max(2, round(spacing * 0.28))
    remaining = list(components)
    merged: list[Component] = []

    while remaining:
        current = remaining.pop(0)
        changed = True
        while changed:
            changed = False
            keep: list[Component] = []
            for other in remaining:
                if _expanded_boxes_touch(
                    current.bbox,
                    other.bbox,
                    pad_x=pad_x,
                    pad_y=pad_y,
                ) and not _merge_would_attach_stem(current, other, spacing=spacing):
                    current = _union_component(current, other)
                    changed = True
                else:
                    keep.append(other)
            remaining = keep
        merged.append(current)
    return merged


def _looks_like_notehead_candidate(
    component: Component,
    *,
    image_width: int,
    staff_lines: list[int],
) -> bool:
    width = component.width
    height = component.height
    if component.area < 8 or width < 3 or height < 3:
        return False
    if width > image_width * 0.35:
        return False

    spacing = _staff_spacing(staff_lines) or 10.0
    max_width = max(12, round(spacing * 2.4))
    max_height = max(10, round(spacing * 2.2))
    if width > max_width or height > max_height:
        return False

    aspect = width / height
    if aspect < 0.3 or aspect > 3.3:
        return False
    fill_ratio = component.area / (width * height)
    return fill_ratio >= 0.16


def _merge_would_attach_stem(left: Component, right: Component, *, spacing: float) -> bool:
    union = _union_component(left, right)
    max_height = max(10, round(spacing * 2.2))
    if union.height <= max_height:
        return False
    return _is_narrow_stem_fragment(left, spacing=spacing) or _is_narrow_stem_fragment(
        right,
        spacing=spacing,
    )


def _is_narrow_stem_fragment(component: Component, *, spacing: float) -> bool:
    return component.width <= max(2, round(spacing * 0.35)) and component.height >= spacing * 0.45


def _expanded_boxes_touch(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
    *,
    pad_x: int,
    pad_y: int,
) -> bool:
    return not (
        left[2] + pad_x < right[0]
        or right[2] + pad_x < left[0]
        or left[3] + pad_y < right[1]
        or right[3] + pad_y < left[1]
    )


def _union_component(left: Component, right: Component) -> Component:
    return Component(
        bbox=(
            min(left.bbox[0], right.bbox[0]),
            min(left.bbox[1], right.bbox[1]),
            max(left.bbox[2], right.bbox[2]),
            max(left.bbox[3], right.bbox[3]),
        ),
        area=left.area + right.area,
    )


def _candidate_payload(
    base_record: dict[str, Any],
    context: dict[str, Any],
    *,
    source_path: Path,
    source_variant: SourceVariant,
    staff_lines: list[int],
    stages: DetectionStages,
    diagnostic_paths: dict[str, Path],
    ground_truth_path: Path | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "vlm_notehead_candidates",
        "slug": base_record["slug"],
        "system_index": int(base_record["system_index"]),
        "system_measure_index": int(base_record["system_measure_index"]),
        "global_measure_index": int(base_record["global_measure_index"]),
        "display_measure_number": int(base_record["display_measure_number"]),
        "source_variant": source_variant,
        "source_image_path": str(source_path),
        "context_path": base_record["paths"]["context"],
        "staff_lines_y_px": staff_lines,
        "heuristic": {
            "description": "threshold dark ink, suppress staff-line rows, merge nearby components",
            "missing_true_noteheads_is_worse_than_false_positives": True,
        },
        "staff_suppression": {
            "selected_rows_y_px": stages.suppression_rows,
            "configured_row_count": len(staff_lines),
        },
        "threshold": stages.threshold,
        "pre_filter_component_count": len(stages.raw_components),
        "candidate_count": len(stages.candidates),
        "ground_truth_image_path": str(ground_truth_path) if ground_truth_path else None,
        "diagnostics": {key: str(path) for key, path in diagnostic_paths.items()},
        "candidates": [
            _candidate_to_json(index, component, staff_lines=staff_lines)
            for index, component in enumerate(stages.candidates, start=1)
        ],
        "source_context": {
            "clef_hint": context.get("clef_hint"),
            "time_signature_hint": context.get("time_signature_hint"),
            "expected_measure_beats": context.get("expected_measure_beats"),
        },
    }


def _candidate_to_json(
    index: int,
    component: Component,
    *,
    staff_lines: list[int],
) -> dict[str, Any]:
    center_x, center_y = component.center
    payload: dict[str, Any] = {
        "id": f"c{index:03d}",
        "bbox": {
            "left": component.bbox[0],
            "top": component.bbox[1],
            "right": component.bbox[2] + 1,
            "bottom": component.bbox[3] + 1,
        },
        "center": {"x": round(center_x, 2), "y": round(center_y, 2)},
        "area": component.area,
        "width": component.width,
        "height": component.height,
    }
    helper = _staff_position_helper(center_y, staff_lines)
    if helper is not None:
        payload["staff_position"] = helper
    return payload


def _staff_position_helper(center_y: float, staff_lines: list[int]) -> dict[str, Any] | None:
    if len(staff_lines) < 2:
        return None
    spacing = _staff_spacing(staff_lines)
    if spacing is None or spacing <= 0:
        return None
    nearest_index, nearest_y = min(
        enumerate(staff_lines, start=1),
        key=lambda item: abs(center_y - item[1]),
    )
    return {
        "nearest_staff_line_index": nearest_index,
        "nearest_staff_line_offset_px": round(center_y - nearest_y, 2),
        "half_space_steps_from_top_line": round((center_y - staff_lines[0]) / (spacing / 2), 2),
    }


def _staff_spacing(staff_lines: list[int]) -> float | None:
    if len(staff_lines) < 2:
        return None
    gaps = [b - a for a, b in zip(staff_lines, staff_lines[1:], strict=False) if b > a]
    if not gaps:
        return None
    return sum(gaps) / len(gaps)


def _write_mask_image(mask: list[list[bool]], output_path: Path) -> None:
    if not mask or not mask[0]:
        image = Image.new("L", (1, 1), 255)
    else:
        height = len(mask)
        width = len(mask[0])
        image = Image.new("L", (width, height), 255)
        pixels = image.load()
        for y, row in enumerate(mask):
            for x, value in enumerate(row):
                pixels[x, y] = 0 if value else 255
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _write_staff_line_mask(
    size: tuple[int, int],
    rows: list[int],
    output_path: Path,
) -> None:
    width, height = size
    image = Image.new("L", size, 255)
    pixels = image.load()
    for y in rows:
        if 0 <= y < height:
            for x in range(width):
                pixels[x, y] = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _write_component_overlay(
    image: Image.Image,
    components: list[Component],
    output_path: Path,
) -> None:
    overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    for index, component in enumerate(components, start=1):
        left, top, right, bottom = component.bbox
        color = (255, 140, 0)
        draw.rectangle((left, top, right, bottom), outline=color, width=1)
        draw.text((left, max(0, top - 8)), f"r{index:03d}", fill=color, font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)


def _write_overlay(image: Image.Image, candidates: list[Component], output_path: Path) -> None:
    overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    for index, component in enumerate(candidates, start=1):
        label = str(index)
        left, top, right, bottom = component.bbox
        color = (255, 38, 0)
        draw.rectangle((left, top, right, bottom), outline=color, width=2)
        center_x, center_y = component.center
        radius = max(3, min(8, round(max(component.width, component.height) * 0.45)))
        draw.ellipse(
            (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
            outline=(0, 102, 255),
            width=2,
        )
        label_bbox = draw.textbbox((0, 0), label, font=font)
        label_width = label_bbox[2] - label_bbox[0] + 4
        label_height = label_bbox[3] - label_bbox[1] + 4
        label_x = max(0, min(overlay.width - label_width, left))
        label_y = max(0, top - label_height - 1)
        draw.rectangle(
            (label_x, label_y, label_x + label_width, label_y + label_height),
            fill=(255, 255, 255),
            outline=color,
        )
        draw.text((label_x + 2, label_y + 1), label, fill=color, font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)


def _write_contact_sheet(
    image: Image.Image,
    candidates: list[Component],
    output_path: Path,
) -> None:
    tile_size = 72
    label_height = 14
    columns = 6
    rows = max(1, (len(candidates) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * tile_size, rows * (tile_size + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, component in enumerate(candidates, start=1):
        col = (index - 1) % columns
        row = (index - 1) // columns
        x0 = col * tile_size
        y0 = row * (tile_size + label_height)
        crop = _candidate_crop(image, component, padding=10)
        crop.thumbnail((tile_size - 8, tile_size - 8))
        paste_x = x0 + (tile_size - crop.width) // 2
        paste_y = y0 + 2 + (tile_size - 8 - crop.height) // 2
        sheet.paste(crop, (paste_x, paste_y))
        draw.rectangle((x0, y0, x0 + tile_size - 1, y0 + tile_size - 1), outline=(180, 180, 180))
        draw.text((x0 + 3, y0 + tile_size), f"c{index:03d}", fill=(0, 0, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _write_diagnostic_contact_sheet(
    panels: list[tuple[str, Image.Image]],
    output_path: Path,
) -> None:
    tile_width = 320
    tile_height = 310
    label_height = 22
    columns = 2
    rows = max(1, (len(panels) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (label, panel) in enumerate(panels):
        col = index % columns
        row = index // columns
        x0 = col * tile_width
        y0 = row * tile_height
        draw.text((x0 + 6, y0 + 5), label, fill=(0, 0, 0), font=font)
        preview = panel.convert("RGB")
        preview.thumbnail((tile_width - 12, tile_height - label_height - 10))
        paste_x = x0 + (tile_width - preview.width) // 2
        paste_y = y0 + label_height + (tile_height - label_height - preview.height) // 2
        sheet.paste(preview, (paste_x, paste_y))
        draw.rectangle(
            (x0, y0, x0 + tile_width - 1, y0 + tile_height - 1),
            outline=(180, 180, 180),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _candidate_crop(image: Image.Image, component: Component, *, padding: int) -> Image.Image:
    left, top, right, bottom = component.bbox
    return image.crop(
        (
            max(0, left - padding),
            max(0, top - padding),
            min(image.width, right + padding + 1),
            min(image.height, bottom + padding + 1),
        )
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _resolve_path(out_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return out_dir / path


if __name__ == "__main__":
    raise SystemExit(main())
