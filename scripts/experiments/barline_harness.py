"""Score one or more candidate barline detectors against a labeled GT system.

Each detector is a separate Python script that, when invoked as

    uv run python <detector_script> <image_path>

prints a single JSON list of barline x-fractions in [0, 1] to stdout.

Usage:
    uv run python scripts/experiments/barline_harness.py \\
        --image out/<slug>/systems/system_001.png \\
        --ground-truth out/<slug>/debug_barlines/ground_truth/system_001_ground_truth.json \\
        --detector scripts/experiments/barline_<name>.py [--detector ...]

Prints a per-detector summary table and writes a JSON report alongside.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument(
        "--detector",
        type=Path,
        action="append",
        required=True,
        help="Detector script. Repeat to evaluate multiple detectors.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("scripts/experiments/results.json"),
        help="Path to write the JSON results.",
    )
    args = parser.parse_args(argv)

    from PIL import Image

    with Image.open(args.image) as im:
        image_width = im.size[0]

    gt_boxes = _parse_via(json.loads(args.ground_truth.read_text(encoding="utf-8")))
    median_gt_width = statistics.median(b["width"] for b in gt_boxes) if gt_boxes else 16
    tolerance_px = max(1.0, median_gt_width / 2.0)

    results: list[dict[str, Any]] = []
    for detector_script in args.detector:
        result = _evaluate_detector(
            detector_script=detector_script,
            image_path=args.image,
            image_width=image_width,
            gt_boxes=gt_boxes,
            tolerance_px=tolerance_px,
        )
        results.append(result)

    _print_table(results, gt_count=len(gt_boxes), tolerance_px=tolerance_px)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "image": str(args.image),
                "ground_truth": str(args.ground_truth),
                "gt_count": len(gt_boxes),
                "tolerance_px": tolerance_px,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {args.report}")
    return 0


def _evaluate_detector(
    *,
    detector_script: Path,
    image_path: Path,
    image_width: int,
    gt_boxes: list[dict[str, Any]],
    tolerance_px: float,
) -> dict[str, Any]:
    name = detector_script.stem
    try:
        proc = subprocess.run(
            ["uv", "run", "python", str(detector_script), str(image_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        x_fractions = json.loads(proc.stdout.strip())
    except subprocess.CalledProcessError as exc:
        return {
            "detector": name,
            "error": f"detector failed: {exc.stderr.strip()[:400]}",
        }
    except json.JSONDecodeError as exc:
        return {
            "detector": name,
            "error": f"detector did not print valid JSON: {exc}",
        }

    detections_px = [
        max(0, min(image_width - 1, round(float(xf) * image_width)))
        for xf in x_fractions
    ]
    matches = _match(gt_boxes, detections_px, tolerance_px=tolerance_px)
    tp = sum(1 for m in matches if m is not None)
    fp = len(detections_px) - tp
    fn = len(gt_boxes) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    errors = [
        abs(detections_px[i] - gt_boxes[gt_idx]["x_center"])
        for i, gt_idx in enumerate(matches)
        if gt_idx is not None
    ]
    median_error = statistics.median(errors) if errors else None
    return {
        "detector": name,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "median_error_px": round(median_error, 2) if median_error is not None else None,
        "detections_px": detections_px,
        "x_fractions": x_fractions,
    }


def _match(
    gt_boxes: list[dict[str, Any]],
    detections_px: list[int],
    *,
    tolerance_px: float,
) -> list[int | None]:
    used: set[int] = set()
    out: list[int | None] = []
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
        out.append(best_idx)
    return out


def _parse_via(via: dict[str, Any]) -> list[dict[str, Any]]:
    image_meta = via.get("_via_img_metadata")
    if image_meta is None:
        image_meta = {
            k: v for k, v in via.items()
            if isinstance(v, dict) and "filename" in v
        }
    if not image_meta:
        raise ValueError("VIA JSON has no recognizable image metadata")
    entry = next(iter(image_meta.values()))
    boxes: list[dict[str, Any]] = []
    for region in entry.get("regions", []):
        shape = region.get("shape_attributes", {}) or {}
        if shape.get("name") != "rect":
            continue
        x = int(shape["x"])
        w = int(shape["width"])
        boxes.append(
            {
                "x": x,
                "y": int(shape["y"]),
                "width": w,
                "height": int(shape["height"]),
                "x_center": x + w / 2.0,
            }
        )
    boxes.sort(key=lambda b: b["x_center"])
    return boxes


def _print_table(
    results: list[dict[str, Any]],
    *,
    gt_count: int,
    tolerance_px: float,
) -> None:
    header = f"{'detector':<28} {'TP':>3} {'FP':>3} {'FN':>3} {'P':>6} {'R':>6} {'F1':>6} {'med_err_px':>11}"
    print(f"GT={gt_count} tolerance_px={tolerance_px:.1f}")
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{r['detector']:<28} ERROR: {r['error']}")
            continue
        med = "" if r["median_error_px"] is None else f"{r['median_error_px']:.1f}"
        print(
            f"{r['detector']:<28} {r['tp']:>3} {r['fp']:>3} {r['fn']:>3} "
            f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} {med:>11}"
        )


if __name__ == "__main__":
    sys.exit(main())
