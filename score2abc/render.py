from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import List


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


def create_minimal_system_crops(
    page_paths: List[Path],
    systems_dir: Path,
    logger: logging.Logger,
) -> List[Path]:
    """Stub system crops: copy first page to system and chord region."""
    systems_dir.mkdir(parents=True, exist_ok=True)
    if not page_paths:
        logger.warning("No pages available for system crops")
        return []

    system_path = systems_dir / "system_001.png"
    chord_path = systems_dir / "chord_region_001.png"
    shutil.copy(page_paths[0], system_path)
    shutil.copy(page_paths[0], chord_path)
    logger.info("Created stub system crop: %s", system_path)
    logger.info("Created stub chord region: %s", chord_path)
    return [system_path, chord_path]


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
