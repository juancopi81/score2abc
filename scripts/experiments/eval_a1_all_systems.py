"""Evaluate A1 (staff_runlength) against every labeled system in a slug.

For each `system_NNN_ground_truth.json` it finds, runs A1 on the matching
`system_NNN.png`, then writes:

  - `system_NNN_a1_vs_gt.png`  green GT bboxes + red detected rects
  - `system_NNN_a1_compare.csv`  TP/FP/FN rows

Also prints a per-system + aggregate summary table and writes
`out/<slug>/debug_barlines/a1_summary.json`.

Usage:
    uv run python scripts/experiments/eval_a1_all_systems.py out --slug <slug>
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

_DETECTOR_SCRIPT = Path("scripts/experiments/barline_a1_staff_runlength.py")
_DETECTOR_COLOR = (255, 0, 0)
_DEFAULT_GT_ROOT = Path("tests/fixtures/barlines")
_GT_RE = re.compile(r"^system_(\d{3})_ground_truth\.json$")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--slug", required=True)
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=None,
        help=(
            "Directory with system_NNN_ground_truth.json files. Defaults to "
            "tests/fixtures/barlines/<slug>, then falls back to "
            "out/<slug>/debug_barlines/ground_truth."
        ),
    )
    args = parser.parse_args(argv)

    work_dir = args.out_dir / args.slug
    debug_dir = work_dir / "debug_barlines"
    debug_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = _resolve_gt_dir(args.slug, debug_dir, args.ground_truth_dir)
    if not gt_dir.is_dir():
        print(f"No ground_truth/ found at {gt_dir}")
        return 1

    gt_paths = sorted(p for p in gt_dir.iterdir() if _GT_RE.match(p.name))
    if not gt_paths:
        print(f"No system_NNN_ground_truth.json files in {gt_dir}")
        return 1

    print(f"Found {len(gt_paths)} labeled system(s).")
    per_system_rows: list[dict[str, Any]] = []

    for gt_path in gt_paths:
        sys_idx = int(_GT_RE.match(gt_path.name).group(1))
        image_path = work_dir / "systems" / f"system_{sys_idx:03d}.png"
        if not image_path.exists():
            print(f"  WARN: system image missing for {gt_path.name}: {image_path}")
            continue

        gt_boxes = _parse_via(json.loads(gt_path.read_text(encoding="utf-8")))
        median_gt_w = statistics.median(b["width"] for b in gt_boxes) if gt_boxes else 16
        tol = max(1.0, median_gt_w / 2.0)
        det_box_half_w = max(2, int(round(median_gt_w / 2.0)))

        with Image.open(image_path) as im:
            image_width = im.size[0]

        x_fractions = _run_detector(_DETECTOR_SCRIPT, image_path)
        detections_px = [
            max(0, min(image_width - 1, round(float(xf) * image_width))) for xf in x_fractions
        ]
        matches = _match(gt_boxes, detections_px, tol)
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
        med_err = statistics.median(errors) if errors else None

        overlay_path = debug_dir / f"system_{sys_idx:03d}_a1_vs_gt.png"
        csv_path = debug_dir / f"system_{sys_idx:03d}_a1_compare.csv"

        _draw_overlay(
            image_path=image_path,
            gt_boxes=gt_boxes,
            detections_px=detections_px,
            detector_color=_DETECTOR_COLOR,
            det_box_half_w=det_box_half_w,
            out_path=overlay_path,
        )
        _write_csv(
            csv_path=csv_path,
            gt_boxes=gt_boxes,
            detections_px=detections_px,
            detection_fractions=[float(xf) for xf in x_fractions],
            matches=matches,
        )

        row = {
            "system": sys_idx,
            "gt_count": len(gt_boxes),
            "tolerance_px": round(tol, 2),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "median_error_px": round(med_err, 2) if med_err is not None else None,
            "detected": len(detections_px),
        }
        per_system_rows.append(row)
        _print_system_row(row)

    summary = _aggregate(per_system_rows)
    _print_aggregate(summary)
    summary_path = debug_dir / "a1_summary.json"
    summary_path.write_text(
        json.dumps(
            {"per_system": per_system_rows, "aggregate": summary},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {summary_path}")
    return 0


def _run_detector(script: Path, image_path: Path) -> list[float]:
    proc = subprocess.run(
        [sys.executable, str(script), str(image_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return list(json.loads(proc.stdout.strip()))


def _parse_via(via: dict[str, Any]) -> list[dict[str, Any]]:
    image_meta = via.get("_via_img_metadata")
    if image_meta is None:
        image_meta = {k: v for k, v in via.items() if isinstance(v, dict) and "filename" in v}
    if not image_meta:
        return []
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


def _match(
    gt_boxes: list[dict[str, Any]],
    detections_px: list[int],
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


def _draw_overlay(
    *,
    image_path: Path,
    gt_boxes: list[dict[str, Any]],
    detections_px: list[int],
    detector_color: tuple[int, int, int],
    det_box_half_w: int,
    out_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for box in gt_boxes:
        x0 = box["x"]
        y0 = box["y"]
        x1 = x0 + box["width"]
        y1 = y0 + box["height"]
        draw.rectangle([(x0, y0), (x1, y1)], outline=(0, 200, 0), width=3)
    for det_x in detections_px:
        x0 = max(0, det_x - det_box_half_w)
        x1 = min(width - 1, det_x + det_box_half_w)
        draw.rectangle([(x0, 0), (x1, height - 1)], outline=detector_color, width=3)
    image.save(out_path)


def _write_csv(
    *,
    csv_path: Path,
    gt_boxes: list[dict[str, Any]],
    detections_px: list[int],
    detection_fractions: list[float],
    matches: list[int | None],
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

    def _key(row: dict[str, Any]) -> tuple[float, int]:
        kind_order = {"TP": 0, "FN": 1, "FP": 2}[row["kind"]]
        anchor = float(row["det_x_px"]) if row["kind"] == "FP" else float(row["gt_x_center_px"])
        return (anchor, kind_order)

    rows.sort(key=_key)

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


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    scored_rows = [r for r in rows if r["gt_count"] or r["detected"]]
    f1s = [r["f1"] for r in scored_rows]
    empty_systems = len(rows) - len(scored_rows)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_per_system_f1": round(statistics.mean(f1s), 4) if f1s else 0.0,
        "min_per_system_f1": round(min(f1s), 4) if f1s else 0.0,
        "empty_systems": empty_systems,
    }


def _resolve_gt_dir(slug: str, debug_dir: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    fixture_dir = _DEFAULT_GT_ROOT / slug
    if fixture_dir.is_dir():
        return fixture_dir
    return debug_dir / "ground_truth"


def _print_system_row(row: dict[str, Any]) -> None:
    med = "" if row["median_error_px"] is None else f"{row['median_error_px']:.1f}"
    print(
        f"  system_{row['system']:03d}  gt={row['gt_count']:>2}  "
        f"tp={row['tp']:>2} fp={row['fp']:>2} fn={row['fn']:>2}  "
        f"P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1']:.3f}  "
        f"med_err={med:>4}"
    )


def _print_aggregate(summary: dict[str, Any]) -> None:
    print("\nAggregate:")
    print(
        f"  TP={summary['tp']}  FP={summary['fp']}  FN={summary['fn']}  "
        f"P={summary['precision']:.3f}  R={summary['recall']:.3f}  "
        f"F1={summary['f1']:.3f}"
    )
    print(
        f"  mean F1/system = {summary['mean_per_system_f1']:.3f}  "
        f"min F1/system = {summary['min_per_system_f1']:.3f}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
