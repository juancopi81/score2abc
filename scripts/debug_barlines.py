"""Render detected chord-alignment barlines over system crops.

Usage:
    uv run python scripts/debug_barlines.py out --slug <slug>
    uv run python scripts/debug_barlines.py out --slug <slug> \\
        --ground-truth out/<slug>/debug_barlines/ground_truth/system_001_ground_truth.json

Reads `out/<slug>/intermediate/chords.json`, draws each system's detected
barline x-fractions as red vertical lines over `systems/system_XXX.png`, and
writes overlays to `out/<slug>/debug_barlines/system_XXX_barlines.png`.

When `--ground-truth` is provided, the script also parses a VIA-format JSON
file with rectangle annotations for a single system, draws GT bboxes in green
and detected bboxes in red on `system_XXX_compare.png`, and writes a
`system_XXX_compare.csv` summary with TP/FP/FN rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from score2abc.utils import get_logger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="Pipeline output directory.")
    parser.add_argument("--slug", required=True, help="Work slug under the output directory.")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help=(
            "Path to a VIA-format ground-truth JSON for a single system. "
            "When set, runs the GT comparison overlay+CSV instead of the "
            "default all-systems red-line overlay."
        ),
    )
    parser.add_argument(
        "--system",
        type=int,
        default=None,
        help=(
            "Restrict to a single system index (1-based). Required when "
            "--ground-truth is set unless the VIA filename encodes it as "
            "system_NNN.png."
        ),
    )
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

    if args.ground_truth is not None:
        return _run_compare(
            work_dir=work_dir,
            debug_dir=debug_dir,
            payload=payload,
            ground_truth_path=args.ground_truth,
            system_override=args.system,
            logger=logger,
        )

    written = 0
    for system in payload.get("systems", []):
        if args.system is not None and int(system["system_index"]) != args.system:
            continue
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


def _run_compare(
    *,
    work_dir: Path,
    debug_dir: Path,
    payload: dict[str, Any],
    ground_truth_path: Path,
    system_override: int | None,
    logger,
) -> int:
    if not ground_truth_path.exists():
        logger.error("Ground-truth JSON not found: %s", ground_truth_path)
        return 1

    via = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    gt_filename, gt_boxes = _parse_via(via)
    system_index = system_override or _system_index_from_filename(gt_filename)
    if system_index is None:
        logger.error(
            "Could not infer system index from VIA filename %r; pass --system N.",
            gt_filename,
        )
        return 1

    image_path = work_dir / "systems" / f"system_{system_index:03d}.png"
    if not image_path.exists():
        logger.error("System image not found: %s", image_path)
        return 1

    system = next(
        (s for s in payload.get("systems", []) if int(s["system_index"]) == system_index),
        None,
    )
    if system is None:
        logger.error("System %d not found in chords.json", system_index)
        return 1

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    detections_px = [
        max(0, min(width - 1, round(float(xf) * width)))
        for xf in system.get("barlines", [])
    ]
    detection_fractions = [float(xf) for xf in system.get("barlines", [])]

    median_gt_width = (
        statistics.median(box["width"] for box in gt_boxes) if gt_boxes else 16
    )
    tolerance_px = max(1.0, median_gt_width / 2.0)
    det_box_half_w = max(2, int(round(median_gt_width / 2.0)))

    matches = _match_detections(gt_boxes, detections_px, tolerance_px=tolerance_px)

    overlay_path = debug_dir / f"system_{system_index:03d}_compare.png"
    csv_path = debug_dir / f"system_{system_index:03d}_compare.csv"

    _draw_compare_overlay(
        image=image,
        gt_boxes=gt_boxes,
        detections_px=detections_px,
        det_box_half_w=det_box_half_w,
        out_path=overlay_path,
    )
    _write_compare_csv(
        csv_path=csv_path,
        gt_boxes=gt_boxes,
        detections_px=detections_px,
        detection_fractions=detection_fractions,
        matches=matches,
        image_width=width,
    )

    tp = sum(1 for m in matches if m is not None)
    fp = len(detections_px) - tp
    fn = len(gt_boxes) - tp
    logger.info(
        "system=%d gt=%d det=%d TP=%d FP=%d FN=%d tolerance_px=%.1f",
        system_index,
        len(gt_boxes),
        len(detections_px),
        tp,
        fp,
        fn,
        tolerance_px,
    )
    logger.info("Wrote %s", overlay_path)
    logger.info("Wrote %s", csv_path)
    return 0


def _parse_via(via: dict[str, Any]) -> tuple[str, list[dict[str, int]]]:
    image_meta = via.get("_via_img_metadata")
    if image_meta is None:
        image_meta = {
            k: v for k, v in via.items()
            if isinstance(v, dict) and "filename" in v
        }
    if not image_meta:
        raise ValueError("VIA JSON has no recognizable image metadata")
    # Use the first (typically only) entry.
    entry = next(iter(image_meta.values()))
    filename = str(entry.get("filename", ""))
    boxes: list[dict[str, int]] = []
    for region in entry.get("regions", []):
        shape = region.get("shape_attributes", {}) or {}
        if shape.get("name") != "rect":
            continue
        x = int(shape["x"])
        y = int(shape["y"])
        w = int(shape["width"])
        h = int(shape["height"])
        boxes.append(
            {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "x_center": x + w / 2.0,
            }
        )
    boxes.sort(key=lambda b: b["x_center"])
    return filename, boxes


_SYSTEM_FILENAME_RE = re.compile(r"system_(\d+)\.png", re.IGNORECASE)


def _system_index_from_filename(filename: str) -> int | None:
    match = _SYSTEM_FILENAME_RE.search(filename)
    if not match:
        return None
    return int(match.group(1))


def _match_detections(
    gt_boxes: list[dict[str, int]],
    detections_px: list[int],
    *,
    tolerance_px: float,
) -> list[int | None]:
    """Greedy nearest-GT matching for each detection.

    Returns a list aligned with `detections_px`; each entry is the matched
    `gt_boxes` index or `None` if the detection is a false positive.
    """
    used: set[int] = set()
    result: list[int | None] = []
    for det_x in detections_px:
        best_idx: int | None = None
        best_dist = tolerance_px + 1.0
        for idx, box in enumerate(gt_boxes):
            if idx in used:
                continue
            dist = abs(det_x - box["x_center"])
            if dist <= tolerance_px and dist < best_dist:
                best_idx = idx
                best_dist = dist
        if best_idx is not None:
            used.add(best_idx)
        result.append(best_idx)
    return result


def _draw_compare_overlay(
    *,
    image: Image.Image,
    gt_boxes: list[dict[str, int]],
    detections_px: list[int],
    det_box_half_w: int,
    out_path: Path,
) -> None:
    width, height = image.size
    draw = ImageDraw.Draw(image)
    for box in gt_boxes:
        x0 = box["x"]
        y0 = box["y"]
        x1 = x0 + box["width"]
        y1 = y0 + box["height"]
        draw.rectangle([(x0, y0), (x1, y1)], outline=(0, 200, 0), width=3)
    for det_x in detections_px:
        x0 = max(0, det_x - det_box_half_w)
        x1 = min(width - 1, det_x + det_box_half_w)
        draw.rectangle([(x0, 0), (x1, height - 1)], outline=(255, 0, 0), width=3)
    image.save(out_path)


def _write_compare_csv(
    *,
    csv_path: Path,
    gt_boxes: list[dict[str, int]],
    detections_px: list[int],
    detection_fractions: list[float],
    matches: list[int | None],
    image_width: int,
) -> None:
    matched_gt = {idx for idx in matches if idx is not None}
    rows: list[dict[str, Any]] = []
    for det_idx, gt_idx in enumerate(matches):
        det_x = detections_px[det_idx]
        det_frac = detection_fractions[det_idx]
        if gt_idx is not None:
            box = gt_boxes[gt_idx]
            rows.append(
                {
                    "kind": "TP",
                    "gt_index": gt_idx,
                    "gt_x_left_px": box["x"],
                    "gt_x_right_px": box["x"] + box["width"],
                    "gt_x_center_px": round(box["x_center"], 2),
                    "det_x_px": det_x,
                    "det_x_fraction": round(det_frac, 6),
                    "error_px": round(det_x - box["x_center"], 2),
                }
            )
        else:
            rows.append(
                {
                    "kind": "FP",
                    "gt_index": "",
                    "gt_x_left_px": "",
                    "gt_x_right_px": "",
                    "gt_x_center_px": "",
                    "det_x_px": det_x,
                    "det_x_fraction": round(det_frac, 6),
                    "error_px": "",
                }
            )
    for gt_idx, box in enumerate(gt_boxes):
        if gt_idx in matched_gt:
            continue
        rows.append(
            {
                "kind": "FN",
                "gt_index": gt_idx,
                "gt_x_left_px": box["x"],
                "gt_x_right_px": box["x"] + box["width"],
                "gt_x_center_px": round(box["x_center"], 2),
                "det_x_px": "",
                "det_x_fraction": "",
                "error_px": "",
            }
        )

    rows.sort(key=_row_sort_key)

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "kind",
                "gt_index",
                "gt_x_left_px",
                "gt_x_right_px",
                "gt_x_center_px",
                "det_x_px",
                "det_x_fraction",
                "error_px",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _row_sort_key(row: dict[str, Any]) -> tuple[float, int]:
    kind_order = {"TP": 0, "FN": 1, "FP": 2}[row["kind"]]
    if row["kind"] == "FP":
        anchor = float(row["det_x_px"])
    else:
        anchor = float(row["gt_x_center_px"])
    return (anchor, kind_order)


if __name__ == "__main__":
    raise SystemExit(main())
