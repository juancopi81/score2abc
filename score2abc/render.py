from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageOps, ImageStat

BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class DetectedSystem:
    page_number: int
    rotation_degrees: float
    system_bbox: BBox
    system_crop_bbox: BBox
    chord_bbox_above: BBox
    chord_crop_bbox_above: BBox
    chord_bbox_below: BBox
    chord_crop_bbox_below: BBox


@dataclass(frozen=True)
class SegmentationResult:
    system_crops: List[Path]
    system_crops_normalized: List[Path]
    chord_crops_above: List[Path]
    chord_crops_above_normalized: List[Path]
    chord_crops_below: List[Path]
    chord_crops_below_normalized: List[Path]
    debug_overlays: List[Path]
    debug_manifests: List[Path]

    @property
    def all_outputs(self) -> List[Path]:
        return [
            *self.system_crops,
            *self.system_crops_normalized,
            *self.chord_crops_above,
            *self.chord_crops_above_normalized,
            *self.chord_crops_below,
            *self.chord_crops_below_normalized,
            *self.debug_overlays,
            *self.debug_manifests,
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
    system_normalized_paths: List[Path] = []
    chord_paths_above: List[Path] = []
    chord_paths_above_normalized: List[Path] = []
    chord_paths_below: List[Path] = []
    chord_paths_below_normalized: List[Path] = []
    overlay_paths: List[Path] = []
    manifest_paths: List[Path] = []
    system_index = 1

    for page_number, page_path in enumerate(page_paths, start=1):
        with Image.open(page_path) as page_image:
            page_rgb = page_image.convert("RGB")
            page_gray = ImageOps.autocontrast(page_rgb.convert("L"), cutoff=1)
            detected_systems = _detect_systems(page_gray, page_number)

            if not detected_systems:
                logger.warning("No staff systems detected on page %s", page_path)
                continue

            for detected in detected_systems:
                system_path = systems_dir / f"system_{system_index:03d}.png"
                system_normalized_path = systems_dir / f"system_normalized_{system_index:03d}.png"
                chord_path_above = systems_dir / f"chord_region_above_{system_index:03d}.png"
                chord_path_above_normalized = (
                    systems_dir / f"chord_region_above_normalized_{system_index:03d}.png"
                )
                chord_path_below = systems_dir / f"chord_region_below_{system_index:03d}.png"
                chord_path_below_normalized = (
                    systems_dir / f"chord_region_below_normalized_{system_index:03d}.png"
                )
                page_rgb.crop(detected.system_crop_bbox).save(system_path)
                page_rgb.crop(detected.chord_crop_bbox_above).save(chord_path_above)
                page_rgb.crop(detected.chord_crop_bbox_below).save(chord_path_below)
                _write_rotated_crop(
                    page_rgb=page_rgb,
                    crop_bbox=detected.system_crop_bbox,
                    angle_degrees=detected.rotation_degrees,
                    output_path=system_normalized_path,
                )
                _write_rotated_crop(
                    page_rgb=page_rgb,
                    crop_bbox=detected.chord_crop_bbox_above,
                    angle_degrees=detected.rotation_degrees,
                    output_path=chord_path_above_normalized,
                )
                _write_rotated_crop(
                    page_rgb=page_rgb,
                    crop_bbox=detected.chord_crop_bbox_below,
                    angle_degrees=detected.rotation_degrees,
                    output_path=chord_path_below_normalized,
                )
                system_paths.append(system_path)
                system_normalized_paths.append(system_normalized_path)
                chord_paths_above.append(chord_path_above)
                chord_paths_above_normalized.append(chord_path_above_normalized)
                chord_paths_below.append(chord_path_below)
                chord_paths_below_normalized.append(chord_path_below_normalized)
                logger.info("Created system crop: %s", system_path)
                logger.info("Created normalized system crop: %s", system_normalized_path)
                logger.info("Created chord region crop above: %s", chord_path_above)
                logger.info(
                    "Created normalized chord region crop above: %s", chord_path_above_normalized
                )
                logger.info("Created chord region crop below: %s", chord_path_below)
                logger.info(
                    "Created normalized chord region crop below: %s", chord_path_below_normalized
                )
                system_index += 1

            overlay_path = systems_dir / f"page_{page_number:03d}_overlay.png"
            _write_segmentation_overlay(page_rgb, detected_systems, overlay_path)
            overlay_paths.append(overlay_path)

            manifest_path = systems_dir / f"page_{page_number:03d}_segments.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "page": str(page_path),
                        "systems": [
                            {
                                "page_number": detected.page_number,
                                "rotation_degrees": round(detected.rotation_degrees, 4),
                                "system_bbox": _bbox_to_dict(detected.system_bbox),
                                "system_crop_bbox": _bbox_to_dict(detected.system_crop_bbox),
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
        system_normalized_paths,
        chord_paths_above,
        chord_paths_above_normalized,
        chord_paths_below,
        chord_paths_below_normalized,
        overlay_paths,
        manifest_paths,
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
    width, height = page_gray.size
    if width == 0 or height == 0:
        return []

    left_margin = int(width * 0.05)
    right_margin = int(width * 0.95)
    ink_threshold = _estimate_ink_threshold(page_gray)
    row_profile = _ink_density_by_row(page_gray, left_margin, right_margin, ink_threshold)
    row_smooth = _moving_average(row_profile, _odd(max(15, int(height * 0.007))))
    peak_density = max(row_smooth, default=0.0)
    if peak_density <= 0:
        return []

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

    detected: List[DetectedSystem] = []
    for index, system_bbox in enumerate(system_bboxes):
        rotation_degrees = _estimate_rotation_degrees(page_gray.crop(system_bbox))
        _, system_top, _, system_bottom = system_bbox
        system_height = system_bottom - system_top
        desired_chord_height = max(32, min(96, system_height // 2))
        previous_system_bottom = 0 if index == 0 else system_bboxes[index - 1][3]
        next_system_top = height if index + 1 == len(system_bboxes) else system_bboxes[index + 1][1]
        system_crop_bbox = _expand_system_crop_bbox(system_bbox, width, height)

        chord_bbox_above = _build_annotation_band_above(
            system_bbox=system_bbox,
            previous_system_bottom=previous_system_bottom,
            page_height=height,
            desired_height=desired_chord_height,
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
                rotation_degrees=rotation_degrees,
                system_bbox=system_bbox,
                system_crop_bbox=system_crop_bbox,
                chord_bbox_above=chord_bbox_above,
                chord_crop_bbox_above=chord_crop_bbox_above,
                chord_bbox_below=chord_bbox_below,
                chord_crop_bbox_below=chord_crop_bbox_below,
            )
        )

    return detected


def _estimate_ink_threshold(page_gray: Image.Image) -> int:
    mean_luma = ImageStat.Stat(page_gray).mean[0]
    return max(150, min(220, int(mean_luma - 28)))


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
) -> BBox:
    left, system_top, right, _ = system_bbox
    band_top = previous_system_bottom + 6
    band_bottom = system_top - 8
    return _normalize_annotation_band(
        left=left,
        right=right,
        preferred_top=band_bottom - desired_height,
        preferred_bottom=band_bottom,
        fallback_anchor=max(band_top, band_bottom - 12),
        lower_bound=band_top,
        upper_bound=band_bottom,
        page_height=page_height,
    )


def _build_annotation_band_below(
    system_bbox: BBox,
    next_system_top: int,
    page_height: int,
    desired_height: int,
) -> BBox:
    left, _, right, system_bottom = system_bbox
    band_top = system_bottom + 8
    band_bottom = next_system_top - 6
    return _normalize_annotation_band(
        left=left,
        right=right,
        preferred_top=band_top,
        preferred_bottom=band_top + desired_height,
        fallback_anchor=band_top,
        lower_bound=band_top,
        upper_bound=band_bottom,
        page_height=page_height,
    )


def _normalize_annotation_band(
    *,
    left: int,
    right: int,
    preferred_top: int,
    preferred_bottom: int,
    fallback_anchor: int,
    lower_bound: int,
    upper_bound: int,
    page_height: int,
) -> BBox:
    minimum_height = 12
    top = max(lower_bound, preferred_top)
    bottom = min(upper_bound, preferred_bottom)

    if bottom - top < minimum_height:
        top = max(0, min(page_height - 1, fallback_anchor))
        bottom = min(page_height, top + minimum_height)

    if bottom <= top:
        bottom = min(page_height, top + 1)

    return (left, top, right, bottom)


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


def _estimate_rotation_degrees(crop_gray: Image.Image) -> float:
    candidate_angles = [step / 4 for step in range(-16, 17)]
    best_angle = 0.0
    best_score = float("-inf")

    for angle in candidate_angles:
        rotated = crop_gray.rotate(
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


def _row_peakiness_score(crop_gray: Image.Image) -> float:
    width = crop_gray.width
    left_margin = int(width * 0.02)
    right_margin = max(left_margin + 1, int(width * 0.98))
    threshold = _estimate_ink_threshold(crop_gray)
    row_profile = _ink_density_by_row(crop_gray, left_margin, right_margin, threshold)
    if not row_profile:
        return 0.0

    mean = sum(row_profile) / len(row_profile)
    return sum((value - mean) ** 2 for value in row_profile)


def _write_rotated_crop(
    *,
    page_rgb: Image.Image,
    crop_bbox: BBox,
    angle_degrees: float,
    output_path: Path,
) -> None:
    crop = page_rgb.crop(crop_bbox)
    rotated = crop.rotate(
        angle_degrees,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor="white",
    )
    rotated.save(output_path)


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
    output_path: Path,
) -> None:
    overlay = page_rgb.copy()
    draw = ImageDraw.Draw(overlay)
    for index, detected in enumerate(detected_systems, start=1):
        draw.rectangle(detected.chord_crop_bbox_above, outline="blue", width=4)
        draw.rectangle(detected.chord_crop_bbox_below, outline="green", width=4)
        draw.rectangle(detected.system_crop_bbox, outline="red", width=4)
        label_x = detected.system_crop_bbox[0] + 8
        label_y = max(0, detected.chord_crop_bbox_above[1] - 18)
        draw.text((label_x, label_y), f"{index}", fill="red")
    overlay.save(output_path)


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
        "page_*_overlay.png",
        "page_*_segments.json",
    )
    for pattern in patterns:
        for path in systems_dir.glob(pattern):
            path.unlink()
