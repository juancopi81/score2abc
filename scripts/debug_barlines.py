"""Render detected chord-alignment barlines over system crops.

Usage:
    uv run python scripts/debug_barlines.py out --slug <slug>

Reads `out/<slug>/intermediate/chords.json`, draws each system's detected
barline x-fractions as red vertical lines over `systems/system_XXX.png`, and
writes overlays to `out/<slug>/debug_barlines/system_XXX_barlines.png`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from score2abc.utils import get_logger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="Pipeline output directory.")
    parser.add_argument("--slug", required=True, help="Work slug under the output directory.")
    args = parser.parse_args(argv)

    logger = get_logger("score2abc.debug_barlines")
    work_dir = args.out_dir / args.slug
    chords_path = work_dir / "intermediate" / "chords.json"
    if not chords_path.exists():
        logger.error("chords.json not found: %s", chords_path)
        return 1

    payload = json.loads(chords_path.read_text(encoding="utf-8"))
    debug_dir = work_dir / "debug_barlines"
    debug_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for system in payload.get("systems", []):
        out_path = _draw_system_barlines(work_dir, debug_dir, system, logger=logger)
        if out_path is not None:
            logger.info("Wrote %s", out_path)
            written += 1

    logger.info("Done. wrote=%d debug_dir=%s", written, debug_dir)
    return 0


def _draw_system_barlines(
    work_dir: Path,
    debug_dir: Path,
    system: dict[str, Any],
    *,
    logger,
) -> Path | None:
    system_index = int(system["system_index"])
    image_path = work_dir / "systems" / f"system_{system_index:03d}.png"
    if not image_path.exists():
        logger.warning("System image not found: %s", image_path)
        return None

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for x_fraction in system.get("barlines", []):
        x = min(max(round(float(x_fraction) * width), 0), width - 1)
        draw.line([(x, 0), (x, height - 1)], fill=(255, 0, 0), width=3)

    out_path = debug_dir / f"system_{system_index:03d}_barlines.png"
    image.save(out_path)
    return out_path


if __name__ == "__main__":
    raise SystemExit(main())
