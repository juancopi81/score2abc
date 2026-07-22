from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import fmean
from typing import List

from PIL import Image, ImageDraw, ImageOps

from score2abc.utils.imaging import estimate_ink_threshold

BBox = tuple[int, int, int, int]

_REQUIRED_STAFF_LINES = 5
_MIN_STAFF_LINE_SUPPORT = 0.18
_MAX_STAFF_SPACING_DEVIATION_RATIO = 0.30


@dataclass(frozen=True)
class DetectedSystem:
    page_number: int
    source_candidate_index: int
    system_bbox: BBox
    system_crop_bbox: BBox
    staff_line_rows: tuple[int, ...]
    staff_line_support: tuple[float, ...]
    chord_bbox_above: BBox
    chord_crop_bbox_above: BBox
    chord_bbox_below: BBox
    chord_crop_bbox_below: BBox


@dataclass(frozen=True)
class SystemCandidateAssessment:
    page_number: int
    source_candidate_index: int
    system_bbox: BBox
    accepted: bool
    reason: str
    long_horizontal_line_rows: tuple[int, ...]
    long_horizontal_line_support: tuple[float, ...]
    staff_line_rows: tuple[int, ...]
    staff_line_support: tuple[float, ...]
    mean_staff_spacing: float | None
    max_spacing_deviation_ratio: float | None


@dataclass(frozen=True)
class SegmentationResult:
    system_crops: List[Path]
    chord_crops_above: List[Path]
    chord_crops_below: List[Path]
    deskewed_pages: List[Path]
    debug_overlays: List[Path]
    debug_manifests: List[Path]
    rejected_candidate_crops: List[Path]
    candidate_diagnostics: List[dict[str, object]]

    @property
    def all_outputs(self) -> List[Path]:
        return [
            *self.system_crops,
            *self.chord_crops_above,
            *self.chord_crops_below,
            *self.deskewed_pages,
            *self.debug_overlays,
            *self.debug_manifests,
            *self.rejected_candidate_crops,
        ]


def render_pdf_to_images(
    pdf_path: Path,
    pages_dir: Path,
    dpi: int,
    logger: logging.Logger,
) -> List[Path]:
    """Render a PDF to PNG images at a fixed DPI."""
    try:
        from pdf2image import convert_from_path
    except ImportError as exc:
        logger.error("pdf2image not installed; install it to render PDFs: %s", exc)
        return []

    pages_dir.mkdir(parents=True, exist_ok=True)
    try:
        images = convert_from_path(str(pdf_path), dpi=dpi)
    except Exception as exc:
        logger.error("Failed to render PDF %s: %s", pdf_path, exc)
        return []
    page_paths: List[Path] = []
    for idx, image in enumerate(images, start=1):
        filename = f"page_{idx:03d}.png"
        output_path = pages_dir / filename
        image.save(output_path)
        page_paths.append(output_path)
        logger.info("Rendered page: %s", output_path)
    return page_paths


def create_system_crops(
    page_paths: List[Path],
    systems_dir: Path,
    logger: logging.Logger,
) -> SegmentationResult:
    """Heuristically segment each page into staff systems and candidate chord bands."""
    systems_dir.mkdir(parents=True, exist_ok=True)
    if not page_paths:
        logger.warning("No pages available for system crops")
        return SegmentationResult([], [], [], [], [], [], [], [])

    _clear_segmentation_outputs(systems_dir)

    system_paths: List[Path] = []
    chord_paths_above: List[Path] = []
    chord_paths_below: List[Path] = []
    deskewed_page_paths: List[Path] = []
    overlay_paths: List[Path] = []
    manifest_paths: List[Path] = []
    rejected_candidate_paths: List[Path] = []
    candidate_diagnostics: List[dict[str, object]] = []
    system_index = 1

    for page_number, page_path in enumerate(page_paths, start=1):
        with Image.open(page_path) as page_image:
            original_rgb = page_image.convert("RGB")
            original_gray = ImageOps.autocontrast(original_rgb.convert("L"), cutoff=1)
            page_rotation_degrees = _estimate_page_skew(original_gray)
            page_rgb = _darken_ink(_deskew_page(original_rgb, page_rotation_degrees))
            page_gray = ImageOps.autocontrast(page_rgb.convert("L"), cutoff=1)

            deskewed_page_path = systems_dir / f"page_{page_number:03d}_deskewed.png"
            page_rgb.save(deskewed_page_path)
            deskewed_page_paths.append(deskewed_page_path)
            logger.info("Wrote deskewed page: %s", deskewed_page_path)

            detected_systems, candidate_assessments = _detect_systems_with_assessments(
                page_gray, page_number
            )

            if not detected_systems:
                logger.warning("No staff systems detected on page %s", page_path)

            output_index_by_source_candidate: dict[int, int] = {}
            for detected in detected_systems:
                output_index_by_source_candidate[detected.source_candidate_index] = system_index
                system_path = systems_dir / f"system_{system_index:03d}.png"
                chord_path_above = systems_dir / f"chord_region_above_{system_index:03d}.png"
                chord_path_below = systems_dir / f"chord_region_below_{system_index:03d}.png"
                page_rgb.crop(detected.system_crop_bbox).save(system_path)
                page_rgb.crop(detected.chord_crop_bbox_above).save(chord_path_above)
                page_rgb.crop(detected.chord_crop_bbox_below).save(chord_path_below)
                system_paths.append(system_path)
                chord_paths_above.append(chord_path_above)
                chord_paths_below.append(chord_path_below)
                logger.info("Created system crop: %s", system_path)
                logger.info("Created chord region crop above: %s", chord_path_above)
                logger.info("Created chord region crop below: %s", chord_path_below)
                system_index += 1

            page_candidate_diagnostics: List[dict[str, object]] = []
            for assessment in candidate_assessments:
                candidate_crop_bbox = _expand_system_crop_bbox(
                    assessment.system_bbox, page_rgb.width, page_rgb.height
                )
                rejected_crop_path: Path | None = None
                if not assessment.accepted:
                    rejected_crop_path = systems_dir / (
                        f"rejected_candidate_page_{page_number:03d}_"
                        f"{assessment.source_candidate_index:03d}.png"
                    )
                    page_rgb.crop(candidate_crop_bbox).save(rejected_crop_path)
                    rejected_candidate_paths.append(rejected_crop_path)
                    logger.info(
                        "Rejected system candidate page=%s candidate=%s reason=%s: %s",
                        page_number,
                        assessment.source_candidate_index,
                        assessment.reason,
                        rejected_crop_path,
                    )

                diagnostic = _candidate_assessment_to_dict(
                    assessment,
                    candidate_crop_bbox=candidate_crop_bbox,
                    output_system_index=output_index_by_source_candidate.get(
                        assessment.source_candidate_index
                    ),
                    rejected_crop_path=rejected_crop_path,
                )
                page_candidate_diagnostics.append(diagnostic)
                candidate_diagnostics.append(diagnostic)

            overlay_path = systems_dir / f"page_{page_number:03d}_overlay.png"
            _write_segmentation_overlay(
                page_rgb,
                detected_systems,
                candidate_assessments,
                overlay_path,
            )
            overlay_paths.append(overlay_path)

            manifest_path = systems_dir / f"page_{page_number:03d}_segments.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "page": str(deskewed_page_path),
                        "source_page": str(page_path),
                        "page_rotation_degrees": round(page_rotation_degrees, 4),
                        "candidates": page_candidate_diagnostics,
                        "rejected_candidates": [
                            item for item in page_candidate_diagnostics if not item["accepted"]
                        ],
                        "systems": [
                            {
                                "page_number": detected.page_number,
                                "output_system_index": output_index_by_source_candidate[
                                    detected.source_candidate_index
                                ],
                                "source_candidate_index": detected.source_candidate_index,
                                "system_bbox": _bbox_to_dict(detected.system_bbox),
                                "system_crop_bbox": _bbox_to_dict(detected.system_crop_bbox),
                                "staff_line_rows": list(detected.staff_line_rows),
                                "staff_line_support": [
                                    round(value, 6) for value in detected.staff_line_support
                                ],
                                "chord_bbox_above": _bbox_to_dict(detected.chord_bbox_above),
                                "chord_crop_bbox_above": _bbox_to_dict(
                                    detected.chord_crop_bbox_above
                                ),
                                "chord_bbox_below": _bbox_to_dict(detected.chord_bbox_below),
                                "chord_crop_bbox_below": _bbox_to_dict(
                                    detected.chord_crop_bbox_below
                                ),
                            }
                            for detected in detected_systems
                        ],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_paths.append(manifest_path)

    return SegmentationResult(
        system_paths,
        chord_paths_above,
        chord_paths_below,
        deskewed_page_paths,
        overlay_paths,
        manifest_paths,
        rejected_candidate_paths,
        candidate_diagnostics,
    )


def render_abc_preview(
    abc_path: Path,
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Render ABC to an SVG preview if a renderer is installed, else stub."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    abc2svg = shutil.which("abc2svg")
    if abc2svg:
        result = subprocess.run(
            [abc2svg, str(abc_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            svg = _extract_svg(result.stdout)
            if svg:
                output_path.write_text(svg, encoding="utf-8")
                logger.info("Rendered preview via abc2svg: %s", output_path)
                return
            logger.warning("abc2svg produced HTML without SVG; falling back to placeholder")
        else:
            logger.warning("abc2svg failed; falling back to placeholder: %s", result.stderr)

    abcm2ps = shutil.which("abcm2ps")
    if abcm2ps:
        prefix = output_path.with_suffix("")
        result = subprocess.run(
            [abcm2ps, "-g", "-O", str(prefix), str(abc_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            rendered = _find_abcm2ps_output(prefix)
            if rendered:
                if rendered != output_path:
                    rendered.replace(output_path)
                logger.info("Rendered preview via abcm2ps: %s", output_path)
                return
            logger.warning("abcm2ps did not produce an SVG output for prefix: %s", prefix)
        else:
            logger.warning("abcm2ps failed; falling back to placeholder: %s", result.stderr)

    _write_placeholder_svg(output_path)
    logger.info("Wrote placeholder preview: %s", output_path)


def _write_placeholder_svg(output_path: Path) -> None:
    svg = """<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"800\" height=\"200\">
  <rect width=\"100%\" height=\"100%\" fill=\"white\" />
  <text x=\"20\" y=\"40\" font-family=\"sans-serif\" font-size=\"16\">Preview not rendered</text>
  <text x=\"20\" y=\"70\" font-family=\"sans-serif\" font-size=\"12\">
    Install abc2svg or abcm2ps to render.
  </text>
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def _extract_svg(html_text: str) -> str | None:
    start = html_text.find("<svg")
    end = html_text.rfind("</svg>")
    if start == -1 or end == -1:
        return None
    end += len("</svg>")
    return html_text[start:end]


def _find_abcm2ps_output(prefix: Path) -> Path | None:
    candidates = sorted(prefix.parent.glob(f"{prefix.name}*.svg"))
    if not candidates:
        return None
    return candidates[0]


def _detect_systems(page_gray: Image.Image, page_number: int) -> List[DetectedSystem]:
    detected, _ = _detect_systems_with_assessments(page_gray, page_number)
    return detected


def _detect_systems_with_assessments(
    page_gray: Image.Image, page_number: int
) -> tuple[List[DetectedSystem], List[SystemCandidateAssessment]]:
    width, height = page_gray.size
    if width == 0 or height == 0:
        return [], []

    left_margin = int(width * 0.05)
    right_margin = int(width * 0.95)
    ink_threshold = estimate_ink_threshold(page_gray)
    row_profile = _ink_density_by_row(page_gray, left_margin, right_margin, ink_threshold)
    row_smooth = _moving_average(row_profile, _odd(max(15, int(height * 0.007))))
    peak_density = max(row_smooth, default=0.0)
    if peak_density <= 0:
        return [], []

    activity_threshold = max(peak_density * 0.18, 0.015)
    broad_spans = _find_active_spans(
        row_smooth,
        threshold=activity_threshold,
        min_length=max(50, int(height * 0.015)),
        gap=max(12, int(height * 0.004)),
    )

    system_bboxes: List[BBox] = []
    for band_top, band_bottom in broad_spans:
        band_profile = row_profile[band_top : band_bottom + 1]
        band_peak = max(band_profile, default=0.0)
        if band_peak <= 0:
            continue

        band_smooth = _moving_average(band_profile, _odd(max(7, int(height * 0.003))))
        core_peak = max(band_smooth, default=0.0)
        core_spans = _find_active_spans(
            band_smooth,
            threshold=max(core_peak * 0.55, activity_threshold * 2.5),
            min_length=max(4, int(height * 0.0015)),
            gap=max(8, int(height * 0.003)),
        )
        if not core_spans:
            continue

        staff_top = band_top + core_spans[0][0]
        staff_bottom = band_top + core_spans[-1][1]
        top_padding = max(12, int(height * 0.004))
        bottom_padding = max(18, int(height * 0.006))
        system_top = max(0, min(band_top, staff_top - top_padding) - 6)
        system_bottom = min(height, max(band_bottom, staff_bottom + bottom_padding) + 8)
        system_left, system_right = _detect_horizontal_bounds(
            page_gray,
            ink_threshold=ink_threshold,
            top=system_top,
            bottom=system_bottom,
        )
        if system_right - system_left < int(width * 0.35):
            continue

        system_bboxes.append((system_left, system_top, system_right, system_bottom))

    candidate_assessments = [
        _assess_system_candidate(
            page_gray,
            page_number=page_number,
            source_candidate_index=index,
            system_bbox=system_bbox,
            ink_threshold=ink_threshold,
        )
        for index, system_bbox in enumerate(system_bboxes, start=1)
    ]
    accepted_candidates = [
        assessment for assessment in candidate_assessments if assessment.accepted
    ]
    candidate_position_by_source_index = {
        assessment.source_candidate_index: index
        for index, assessment in enumerate(candidate_assessments)
    }

    detected: List[DetectedSystem] = []
    for assessment in accepted_candidates:
        system_bbox = assessment.system_bbox
        _, system_top, _, system_bottom = system_bbox
        system_height = system_bottom - system_top
        desired_chord_height = max(32, min(96, system_height // 2))
        staff_overlap = max(24, int(system_height * 0.40))
        candidate_position = candidate_position_by_source_index[assessment.source_candidate_index]
        previous_system_bottom = (
            0
            if candidate_position == 0
            else candidate_assessments[candidate_position - 1].system_bbox[3]
        )
        next_system_top = (
            height
            if candidate_position + 1 == len(candidate_assessments)
            else candidate_assessments[candidate_position + 1].system_bbox[1]
        )
        system_crop_bbox = _expand_system_crop_bbox(system_bbox, width, height)

        chord_bbox_above = _build_annotation_band_above(
            system_bbox=system_bbox,
            previous_system_bottom=previous_system_bottom,
            page_height=height,
            desired_height=desired_chord_height,
            staff_overlap=staff_overlap,
        )
        chord_crop_bbox_above = _expand_annotation_crop_bbox(
            chord_bbox_above,
            page_width=width,
            page_height=height,
            direction="above",
        )
        chord_bbox_below = _build_annotation_band_below(
            system_bbox=system_bbox,
            next_system_top=next_system_top,
            page_height=height,
            desired_height=desired_chord_height,
            staff_overlap=staff_overlap,
        )
        chord_crop_bbox_below = _expand_annotation_crop_bbox(
            chord_bbox_below,
            page_width=width,
            page_height=height,
            direction="below",
        )

        detected.append(
            DetectedSystem(
                page_number=page_number,
                source_candidate_index=assessment.source_candidate_index,
                system_bbox=system_bbox,
                system_crop_bbox=system_crop_bbox,
                staff_line_rows=assessment.staff_line_rows,
                staff_line_support=assessment.staff_line_support,
                chord_bbox_above=chord_bbox_above,
                chord_crop_bbox_above=chord_crop_bbox_above,
                chord_bbox_below=chord_bbox_below,
                chord_crop_bbox_below=chord_crop_bbox_below,
            )
        )

    return detected, candidate_assessments


def _assess_system_candidate(
    page_gray: Image.Image,
    *,
    page_number: int,
    source_candidate_index: int,
    system_bbox: BBox,
    ink_threshold: int,
) -> SystemCandidateAssessment:
    left, top, right, bottom = system_bbox
    pixels = page_gray.load()
    width = max(1, right - left)
    row_support = [
        sum(pixels[x, y] < ink_threshold for x in range(left, right)) / width
        for y in range(top, bottom)
    ]
    support_spans = _find_active_spans(
        row_support,
        threshold=_MIN_STAFF_LINE_SUPPORT,
        min_length=1,
        gap=0,
    )

    long_line_rows: List[int] = []
    long_line_support: List[float] = []
    for span_top, span_bottom in support_spans:
        strongest_offset = max(
            range(span_top, span_bottom + 1),
            key=lambda offset: row_support[offset],
        )
        long_line_rows.append(top + strongest_offset)
        long_line_support.append(row_support[strongest_offset])

    if len(long_line_rows) < _REQUIRED_STAFF_LINES:
        return SystemCandidateAssessment(
            page_number=page_number,
            source_candidate_index=source_candidate_index,
            system_bbox=system_bbox,
            accepted=False,
            reason="insufficient_long_horizontal_lines",
            long_horizontal_line_rows=tuple(long_line_rows),
            long_horizontal_line_support=tuple(long_line_support),
            staff_line_rows=(),
            staff_line_support=(),
            mean_staff_spacing=None,
            max_spacing_deviation_ratio=None,
        )

    best_sequence: tuple[float, float, tuple[int, ...]] | None = None
    for indices in combinations(range(len(long_line_rows)), _REQUIRED_STAFF_LINES):
        rows = tuple(long_line_rows[index] for index in indices)
        spacings = [rows[index + 1] - rows[index] for index in range(len(rows) - 1)]
        mean_spacing = fmean(spacings)
        if mean_spacing < 6 or mean_spacing > 80:
            continue
        spacing_deviation = max(abs(spacing - mean_spacing) for spacing in spacings)
        spacing_deviation_ratio = spacing_deviation / mean_spacing
        if spacing_deviation_ratio > _MAX_STAFF_SPACING_DEVIATION_RATIO:
            continue
        mean_support = fmean(long_line_support[index] for index in indices)
        score = (spacing_deviation_ratio, -mean_support, indices)
        if best_sequence is None or score < best_sequence:
            best_sequence = score

    if best_sequence is None:
        return SystemCandidateAssessment(
            page_number=page_number,
            source_candidate_index=source_candidate_index,
            system_bbox=system_bbox,
            accepted=False,
            reason="inconsistent_horizontal_line_spacing",
            long_horizontal_line_rows=tuple(long_line_rows),
            long_horizontal_line_support=tuple(long_line_support),
            staff_line_rows=(),
            staff_line_support=(),
            mean_staff_spacing=None,
            max_spacing_deviation_ratio=None,
        )

    spacing_deviation_ratio, _, selected_indices = best_sequence
    staff_line_rows = tuple(long_line_rows[index] for index in selected_indices)
    staff_line_support = tuple(long_line_support[index] for index in selected_indices)
    mean_staff_spacing = fmean(
        staff_line_rows[index + 1] - staff_line_rows[index]
        for index in range(len(staff_line_rows) - 1)
    )
    return SystemCandidateAssessment(
        page_number=page_number,
        source_candidate_index=source_candidate_index,
        system_bbox=system_bbox,
        accepted=True,
        reason="five_consistently_spaced_staff_lines",
        long_horizontal_line_rows=tuple(long_line_rows),
        long_horizontal_line_support=tuple(long_line_support),
        staff_line_rows=staff_line_rows,
        staff_line_support=staff_line_support,
        mean_staff_spacing=mean_staff_spacing,
        max_spacing_deviation_ratio=spacing_deviation_ratio,
    )


def _ink_density_by_row(
    page_gray: Image.Image, left: int, right: int, ink_threshold: int
) -> List[float]:
    pixels = page_gray.load()
    active_width = max(1, right - left)
    density: List[float] = []
    for y in range(page_gray.height):
        dark_pixels = 0
        for x in range(left, right):
            if pixels[x, y] < ink_threshold:
                dark_pixels += 1
        density.append(dark_pixels / active_width)
    return density


def _detect_horizontal_bounds(
    page_gray: Image.Image,
    ink_threshold: int,
    top: int,
    bottom: int,
) -> tuple[int, int]:
    pixels = page_gray.load()
    active_height = max(1, bottom - top)
    column_density: List[float] = []
    for x in range(page_gray.width):
        dark_pixels = 0
        for y in range(top, bottom):
            if pixels[x, y] < ink_threshold:
                dark_pixels += 1
        column_density.append(dark_pixels / active_height)

    column_smooth = _moving_average(column_density, _odd(max(21, int(page_gray.width * 0.012))))
    peak_density = max(column_smooth, default=0.0)
    if peak_density <= 0:
        return 0, page_gray.width

    spans = _find_active_spans(
        column_smooth,
        threshold=max(peak_density * 0.2, 0.02),
        min_length=max(80, int(page_gray.width * 0.08)),
        gap=max(20, int(page_gray.width * 0.01)),
    )
    if not spans:
        return 0, page_gray.width

    left = max(0, spans[0][0] - 24)
    right = min(page_gray.width, spans[-1][1] + 24)
    return left, right


def _build_annotation_band_above(
    system_bbox: BBox,
    previous_system_bottom: int,
    page_height: int,
    desired_height: int,
    staff_overlap: int,
) -> BBox:
    left, system_top, right, system_bottom = system_bbox
    staff_height = max(1, system_bottom - system_top)
    invasion = min(max(0, staff_overlap), staff_height // 2)
    band_bottom = min(page_height, system_top + invasion)
    gap_start = max(0, previous_system_bottom + 6)
    band_top = max(gap_start, band_bottom - desired_height - invasion)
    if band_bottom - band_top < 12:
        band_top = max(0, band_bottom - 12)
    return (left, band_top, right, band_bottom)


def _build_annotation_band_below(
    system_bbox: BBox,
    next_system_top: int,
    page_height: int,
    desired_height: int,
    staff_overlap: int,
) -> BBox:
    left, system_top, right, system_bottom = system_bbox
    staff_height = max(1, system_bottom - system_top)
    invasion = min(max(0, staff_overlap), staff_height // 2)
    band_top = max(0, system_bottom - invasion)
    gap_end = min(page_height, max(band_top + 12, next_system_top - 6))
    band_bottom = min(gap_end, band_top + desired_height + invasion)
    if band_bottom - band_top < 12:
        band_bottom = min(page_height, band_top + 12)
    return (left, band_top, right, band_bottom)


def _expand_system_crop_bbox(system_bbox: BBox, page_width: int, page_height: int) -> BBox:
    left, top, right, bottom = system_bbox
    height = bottom - top
    return _expand_bbox(
        bbox=system_bbox,
        page_width=page_width,
        page_height=page_height,
        pad_left=max(12, int(page_width * 0.004)),
        pad_right=max(12, int(page_width * 0.004)),
        pad_top=max(18, height // 8),
        pad_bottom=max(24, height // 6),
    )


def _expand_annotation_crop_bbox(
    bbox: BBox,
    *,
    page_width: int,
    page_height: int,
    direction: str,
) -> BBox:
    _, top, _, bottom = bbox
    height = max(1, bottom - top)
    side_padding = max(12, int(page_width * 0.004))
    primary_padding = max(18, height // 4)
    secondary_padding = max(6, height // 12)

    if direction == "above":
        return _expand_bbox(
            bbox=bbox,
            page_width=page_width,
            page_height=page_height,
            pad_left=side_padding,
            pad_right=side_padding,
            pad_top=primary_padding,
            pad_bottom=secondary_padding,
        )
    if direction == "below":
        return _expand_bbox(
            bbox=bbox,
            page_width=page_width,
            page_height=page_height,
            pad_left=side_padding,
            pad_right=side_padding,
            pad_top=secondary_padding,
            pad_bottom=primary_padding,
        )

    raise ValueError(f"Unsupported annotation crop direction: {direction}")


def _expand_bbox(
    *,
    bbox: BBox,
    page_width: int,
    page_height: int,
    pad_left: int,
    pad_right: int,
    pad_top: int,
    pad_bottom: int,
) -> BBox:
    left, top, right, bottom = bbox
    return (
        max(0, left - pad_left),
        max(0, top - pad_top),
        min(page_width, right + pad_right),
        min(page_height, bottom + pad_bottom),
    )


def _estimate_page_skew(page_gray: Image.Image) -> float:
    width, height = page_gray.size
    if width < 64 or height < 64:
        return 0.0
    probe = _downsample_for_probe(page_gray, max_dimension=800)
    candidate_angles = [step / 4 for step in range(-16, 17)]
    best_angle = 0.0
    best_score = float("-inf")
    for angle in candidate_angles:
        rotated = probe.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=255,
        )
        score = _row_peakiness_score(rotated)
        if score > best_score:
            best_score = score
            best_angle = angle
    return best_angle


def _deskew_page(page_rgb: Image.Image, angle_degrees: float) -> Image.Image:
    if abs(angle_degrees) < 0.125:
        return page_rgb
    return page_rgb.rotate(
        angle_degrees,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor="white",
    )


def _darken_ink(page: Image.Image, gamma: float = 3.5) -> Image.Image:
    """Push faded ink toward pure black via a gamma curve.

    Anchors the curve to the page's own p99 luma (not a hardcoded 255) so
    scans with slightly yellowed or shadowed backgrounds get their paper
    normalized to white *before* gamma bites — otherwise a 245 paper pixel
    would be pulled to 222 (mid-gray) instead of staying near white.
    """
    histogram = page.convert("L").histogram()
    total = sum(histogram)
    if total == 0:
        return page
    cumulative = 0
    white_point = 255
    target = total * 0.99
    for value, count in enumerate(histogram):
        cumulative += count
        if cumulative >= target:
            white_point = max(value, 1)
            break
    scale = 255.0 / white_point
    table = [int(round((min(1.0, value * scale / 255.0) ** gamma) * 255.0)) for value in range(256)]
    bands = page.split()
    return Image.merge(page.mode, [band.point(table) for band in bands])


def _downsample_for_probe(image: Image.Image, max_dimension: int) -> Image.Image:
    width, height = image.size
    largest = max(width, height)
    if largest <= max_dimension:
        return image
    scale = max_dimension / largest
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.BILINEAR)


def _row_peakiness_score(crop_gray: Image.Image) -> float:
    width = crop_gray.width
    left_margin = int(width * 0.02)
    right_margin = max(left_margin + 1, int(width * 0.98))
    threshold = estimate_ink_threshold(crop_gray)
    row_profile = _ink_density_by_row(crop_gray, left_margin, right_margin, threshold)
    if not row_profile:
        return 0.0

    mean = sum(row_profile) / len(row_profile)
    return sum((value - mean) ** 2 for value in row_profile)


def _moving_average(values: List[float], window: int) -> List[float]:
    if not values:
        return []

    prefix_sums = [0.0]
    for value in values:
        prefix_sums.append(prefix_sums[-1] + value)

    half_window = window // 2
    smoothed: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx - half_window)
        end = min(len(values), idx + half_window + 1)
        smoothed.append((prefix_sums[end] - prefix_sums[start]) / (end - start))
    return smoothed


def _find_active_spans(
    values: List[float],
    threshold: float,
    min_length: int,
    gap: int,
) -> List[tuple[int, int]]:
    raw_spans: List[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if value >= threshold and start is None:
            start = index
        elif value < threshold and start is not None:
            if index - start >= min_length:
                raw_spans.append((start, index - 1))
            start = None
    if start is not None and len(values) - start >= min_length:
        raw_spans.append((start, len(values) - 1))

    if not raw_spans:
        return []

    merged: List[tuple[int, int]] = [raw_spans[0]]
    for span_start, span_end in raw_spans[1:]:
        last_start, last_end = merged[-1]
        if span_start - last_end <= gap:
            merged[-1] = (last_start, span_end)
        else:
            merged.append((span_start, span_end))
    return merged


def _odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def _write_segmentation_overlay(
    page_rgb: Image.Image,
    detected_systems: List[DetectedSystem],
    candidate_assessments: List[SystemCandidateAssessment],
    output_path: Path,
) -> None:
    overlay = page_rgb.copy()
    draw = ImageDraw.Draw(overlay)
    for assessment in candidate_assessments:
        if assessment.accepted:
            continue
        draw.rectangle(assessment.system_bbox, outline="orange", width=4)
        label_x = assessment.system_bbox[0] + 8
        label_y = max(0, assessment.system_bbox[1] - 18)
        draw.text(
            (label_x, label_y),
            f"rejected candidate {assessment.source_candidate_index}",
            fill="orange",
        )
    for index, detected in enumerate(detected_systems, start=1):
        draw.rectangle(detected.chord_crop_bbox_above, outline="blue", width=4)
        draw.rectangle(detected.chord_crop_bbox_below, outline="green", width=4)
        draw.rectangle(detected.system_crop_bbox, outline="red", width=4)
        label_x = detected.system_crop_bbox[0] + 8
        label_y = max(0, detected.chord_crop_bbox_above[1] - 18)
        draw.text((label_x, label_y), f"{index}", fill="red")
    overlay.save(output_path)


def _candidate_assessment_to_dict(
    assessment: SystemCandidateAssessment,
    *,
    candidate_crop_bbox: BBox,
    output_system_index: int | None,
    rejected_crop_path: Path | None,
) -> dict[str, object]:
    return {
        "page_number": assessment.page_number,
        "source_candidate_index": assessment.source_candidate_index,
        "output_system_index": output_system_index,
        "accepted": assessment.accepted,
        "reason": assessment.reason,
        "system_bbox": _bbox_to_dict(assessment.system_bbox),
        "candidate_crop_bbox": _bbox_to_dict(candidate_crop_bbox),
        "candidate_crop": str(rejected_crop_path) if rejected_crop_path else None,
        "required_staff_line_count": _REQUIRED_STAFF_LINES,
        "minimum_line_support": _MIN_STAFF_LINE_SUPPORT,
        "long_horizontal_line_rows": list(assessment.long_horizontal_line_rows),
        "long_horizontal_line_support": [
            round(value, 6) for value in assessment.long_horizontal_line_support
        ],
        "staff_line_rows": list(assessment.staff_line_rows),
        "staff_line_support": [round(value, 6) for value in assessment.staff_line_support],
        "mean_staff_spacing": (
            round(assessment.mean_staff_spacing, 6)
            if assessment.mean_staff_spacing is not None
            else None
        ),
        "max_spacing_deviation_ratio": (
            round(assessment.max_spacing_deviation_ratio, 6)
            if assessment.max_spacing_deviation_ratio is not None
            else None
        ),
    }


def _bbox_to_dict(bbox: BBox) -> dict[str, int]:
    left, top, right, bottom = bbox
    return {"left": left, "top": top, "right": right, "bottom": bottom}


def _clear_segmentation_outputs(systems_dir: Path) -> None:
    patterns = (
        "system_*.png",
        "system_normalized_*.png",
        "chord_region_*.png",
        "chord_region_above_*.png",
        "chord_region_above_normalized_*.png",
        "chord_region_below_*.png",
        "chord_region_below_normalized_*.png",
        "page_*_deskewed.png",
        "page_*_overlay.png",
        "page_*_segments.json",
        "rejected_candidate_page_*.png",
    )
    for pattern in patterns:
        for path in systems_dir.glob(pattern):
            path.unlink()
