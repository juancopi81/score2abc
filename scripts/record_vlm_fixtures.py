"""Record chord-OCR fixtures from committed work crops.

Usage:
    uv sync --extra vlm
    export GEMINI_API_KEY=...  # free-tier key is fine
    uv run python scripts/record_vlm_fixtures.py out [--slug aviador] \\
        [--fixtures-dir tests/fixtures/vlm] [--force]

For every `chord_region_above_*.png` and `chord_region_below_*.png` under
`out/<slug>/systems/`, calls the live Gemini backend once, normalizes the
detections, and writes a fixture to `tests/fixtures/vlm/<key>.json`. Fixture
keys are content-addressed (SHA256 of the image bytes + prompt_version +
model_id), so the same crop/prompt/model always resolves to the same filename.

This is a one-off dev step: run it to capture fixtures the first time you
label a work (or whenever the prompt/model changes). CI and the hermetic
pipeline path (`use_vlm=False`) replay fixtures from disk — they never call
Gemini.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from score2abc.chord_ocr import (
    BANDS,
    Band,
    ChordExtractionRequest,
    GeminiChordOCR,
    fixture_key,
    write_fixture,
)
from score2abc.manifest import load_manifest_jsonl
from score2abc.schemas import WorkItem
from score2abc.utils import get_logger

REPO_ROOT = Path(__file__).resolve().parents[1]
BAND_GLOB: dict[Band, str] = {
    "above": "chord_region_above_*.png",
    "below": "chord_region_below_*.png",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="Pipeline output directory with manifest.jsonl")
    parser.add_argument(
        "--slug",
        action="append",
        default=None,
        help="Limit to specific work slugs (repeat to pass multiple).",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "vlm",
        help="Destination directory for fixture JSON files.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the Gemini model (defaults to GeminiChordOCR default).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing fixtures instead of skipping them.",
    )
    args = parser.parse_args(argv)

    logger = get_logger("score2abc.record_vlm_fixtures")
    out_dir: Path = args.out_dir
    manifest_path = out_dir / "manifest.jsonl"
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        return 1

    fixtures_dir: Path = args.fixtures_dir
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    ocr = GeminiChordOCR(model=args.model) if args.model else GeminiChordOCR()
    logger.info(
        "Recording fixtures with model=%s prompt_version=%s -> %s",
        ocr.model_id,
        ocr.prompt_version,
        fixtures_dir,
    )

    selected_slugs = set(args.slug) if args.slug else None
    work_items = load_manifest_jsonl(manifest_path)

    total_written = 0
    total_skipped = 0
    for item in work_items:
        if selected_slugs is not None and item.slug not in selected_slugs:
            continue
        written, skipped = _record_for_work(
            item,
            out_dir=out_dir,
            fixtures_dir=fixtures_dir,
            ocr=ocr,
            force=args.force,
            logger=logger,
        )
        total_written += written
        total_skipped += skipped

    logger.info(
        "Done. wrote=%d skipped=%d fixtures_dir=%s", total_written, total_skipped, fixtures_dir
    )
    return 0


def _record_for_work(
    item: WorkItem,
    *,
    out_dir: Path,
    fixtures_dir: Path,
    ocr: GeminiChordOCR,
    force: bool,
    logger,
) -> tuple[int, int]:
    systems_dir = out_dir / item.slug / "systems"
    if not systems_dir.exists():
        logger.warning(
            "No systems dir for %s (did you run `score2abc run`?): %s",
            item.slug,
            systems_dir,
        )
        return 0, 0

    written = 0
    skipped = 0
    for band in BANDS:
        for image_path in sorted(systems_dir.glob(BAND_GLOB[band])):
            system_index = _parse_system_index(image_path.stem)
            key = fixture_key(
                image_path,
                prompt_version=ocr.prompt_version,
                model_id=ocr.model_id,
            )
            target = fixtures_dir / f"{key}.json"
            if target.exists() and not force:
                logger.info("Skip (exists): %s -> %s", image_path.name, target.name)
                skipped += 1
                continue

            request = ChordExtractionRequest(
                image_path=image_path,
                band=band,
                system_index=system_index,
                rhythm_hint=item.metadata.rhythm,
                key_hint=item.metadata.key_hint,
                time_signature_hint=item.metadata.time_signature,
            )
            detections = list(ocr.extract(request))
            write_fixture(
                target,
                image_path=image_path,
                prompt_version=ocr.prompt_version,
                model_id=ocr.model_id,
                detections=detections,
            )
            logger.info(
                "Wrote %d detections: %s -> %s", len(detections), image_path.name, target.name
            )
            written += 1

    return written, skipped


def _parse_system_index(stem: str) -> int:
    tail = stem.rsplit("_", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
