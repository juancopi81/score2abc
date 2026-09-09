"""Evaluate staff-grid-density v2 notehead proposals against coordinate GT.

Candidate generation is deliberately blind: all requested caps are generated
before the annotated notehead fixture for a measure is read. The report is a
development-measure baseline and does not claim performance generalization.

Example:
    uv run python scripts/eval_vlm_notehead_proposal_baselines.py out \
        --slug jaime-llanos_12_aviador_pasillo_fulgencio-garcia \
        --system 1 --measure 1 --measure 2 --measure 3 --measure 4
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_vlm_notehead_candidates as detector  # noqa: E402

DEFAULT_CAPS = (4, 8, 12, 16, 24)
DEFAULT_GT_DIR = REPO_ROOT / "tests/fixtures/vlm_melody/notehead_ground_truth"
LEGACY_MIN_TOLERANCE_PX = 4.0
LEGACY_SPACING_FACTOR = 0.55
PITCH_SAFE_X_FACTOR = 0.75
PITCH_SAFE_Y_FACTOR = 0.25
ANNOTATION_REGION_MARGIN = 1.15


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    measures = _positive_unique(args.measure or [1, 2, 3, 4], "measure")
    caps = _positive_unique(args.cap or DEFAULT_CAPS, "cap")
    output_path = args.output or _default_output_path(args.out_dir, args.slug, args.system)

    try:
        report = evaluate_proposals(
            args.out_dir,
            slug=args.slug,
            system_index=args.system,
            measures=measures,
            caps=caps,
        )
        write_report(report, output_path)
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(output_path)
    print(output_path.with_suffix(".md"))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="Pipeline output directory.")
    parser.add_argument("--slug", required=True, help="Work slug.")
    parser.add_argument("--system", required=True, type=int, help="1-based system index.")
    parser.add_argument(
        "--measure",
        action="append",
        type=int,
        help="1-based system-local measure index; repeat to select measures.",
    )
    parser.add_argument(
        "--cap",
        "--caps",
        dest="cap",
        action="append",
        type=int,
        help="Candidate cap; repeat to select caps. Defaults to 4, 8, 12, 16, 24.",
    )
    parser.add_argument(
        "--output",
        "--output-path",
        dest="output",
        type=Path,
        default=None,
        help="JSON report path. Markdown is written beside it with a .md suffix.",
    )
    return parser


def evaluate_proposals(
    out_dir: Path,
    *,
    slug: str,
    system_index: int,
    measures: Sequence[int],
    caps: Sequence[int] = DEFAULT_CAPS,
    ground_truth_dir: Path = DEFAULT_GT_DIR,
) -> dict[str, Any]:
    """Evaluate all caps for selected measures, generating proposals before GT reads."""
    selected_measures = _positive_unique(measures, "measure")
    selected_caps = _positive_unique(caps, "cap")
    per_measure: list[dict[str, Any]] = []

    for measure in selected_measures:
        base_record = detector._load_or_build_base_record(
            out_dir,
            slug=slug,
            system_index=system_index,
            measure_index=measure,
        )
        context_path = detector._resolve_path(out_dir, base_record["paths"]["context"])
        context = _load_json(context_path)
        source_path = detector._source_path(out_dir, base_record, "raw")
        staff_lines = detector._staff_lines_for_source(context, "raw")
        spacing = detector._staff_spacing(staff_lines)
        if spacing is None or spacing <= 0:
            raise ValueError(f"Invalid staff-line geometry for measure {measure}: {staff_lines!r}")

        with Image.open(source_path) as image:
            image_rgb = image.convert("RGB")
            generated_by_cap = {
                cap: detector.detect_staff_grid_density_candidates(
                    image_rgb,
                    staff_lines=staff_lines,
                    max_candidates=cap,
                )
                for cap in selected_caps
            }

        # This read intentionally follows every detector call for this measure.
        ground_truth_path = _ground_truth_path(
            ground_truth_dir,
            slug=slug,
            system_index=system_index,
            measure=measure,
        )
        ground_truth = _load_ground_truth_fixture(ground_truth_path)
        per_measure.append(
            _measure_report(
                measure=measure,
                base_record=base_record,
                source_path=source_path,
                staff_lines=staff_lines,
                spacing=spacing,
                ground_truth_path=ground_truth_path,
                ground_truth=ground_truth,
                generated_by_cap=generated_by_cap,
                caps=selected_caps,
            )
        )

    return {
        "schema_version": 1,
        "kind": "vlm_notehead_proposal_baseline_evaluation",
        "slug": slug,
        "system_index": system_index,
        "detector": {
            "strategy": "staff-grid-density",
            "version": 2,
            "source_variant": "raw",
            "candidate_generation_uses_ground_truth": False,
            "candidate_generation": (
                "existing detector internals called once per requested cap before GT load"
            ),
        },
        "metrics": {
            "annotation_region": {
                "primary": True,
                "ellipse_margin": ANNOTATION_REGION_MARGIN,
                "matching": "global closest-normalized-ellipse one-to-one pairs",
                "pitch": "treble-staff natural pitch from candidate y; GT accidentals ignored",
            },
            "legacy_euclidean": {
                "tolerance_rule": "max(4 px, 0.55 * mean staff spacing)",
                "matching": "global closest-distance one-to-one pairs",
            },
            "pitch_safe": {
                "threshold_rule": (
                    "abs(dx) <= 0.75 * mean staff spacing and "
                    "abs(dy) <= 0.25 * mean staff spacing"
                ),
                "matching": "global closest-distance eligible one-to-one pairs",
            },
        },
        "ground_truth": {
            "directory": str(ground_truth_dir),
            "coordinate_space": "raw measure image pixels, origin at top-left",
            "source": "annotated development measures only",
        },
        "per_measure": per_measure,
        "aggregate_by_cap": _aggregate_by_cap(per_measure, selected_caps),
        "selection": _selection_summary(_aggregate_by_cap(per_measure, selected_caps)),
        "scope_note": (
            "These are reproducible baselines on the annotated development measures selected "
            "above; they do not establish generalization beyond those measures."
        ),
    }


def _measure_report(
    *,
    measure: int,
    base_record: dict[str, Any],
    source_path: Path,
    staff_lines: Sequence[int],
    spacing: float,
    ground_truth_path: Path,
    ground_truth: Sequence[dict[str, Any]],
    generated_by_cap: dict[int, Sequence[Any]],
    caps: Sequence[int],
) -> dict[str, Any]:
    legacy_tolerance = max(LEGACY_MIN_TOLERANCE_PX, LEGACY_SPACING_FACTOR * spacing)
    pitch_safe_thresholds = {
        "x_px": PITCH_SAFE_X_FACTOR * spacing,
        "y_px": PITCH_SAFE_Y_FACTOR * spacing,
    }
    cap_reports = []
    for cap in caps:
        candidates = generated_by_cap[cap]
        candidate_points = _candidate_points(candidates)
        legacy = match_points(
            candidate_points,
            ground_truth,
            eligibility=lambda dx, dy: math.hypot(dx, dy) <= legacy_tolerance,
            tolerance_description=f"Euclidean distance <= {legacy_tolerance:.3f} px",
        )
        region = match_region_points(
            candidate_points,
            ground_truth,
            staff_lines=staff_lines,
            margin=ANNOTATION_REGION_MARGIN,
        )
        pitch_safe = match_points(
            candidate_points,
            ground_truth,
            eligibility=lambda dx, dy: (
                abs(dx) <= pitch_safe_thresholds["x_px"]
                and abs(dy) <= pitch_safe_thresholds["y_px"]
            ),
            tolerance_description=(
                f"abs(dx) <= {pitch_safe_thresholds['x_px']:.3f} px and "
                f"abs(dy) <= {pitch_safe_thresholds['y_px']:.3f} px"
            ),
        )
        cap_reports.append(
            {
                "cap": cap,
                "candidate_count": len(candidate_points),
                "gt_count": len(ground_truth),
                "candidates": candidate_points,
                "annotation_region": region,
                "legacy_euclidean": legacy,
                "pitch_safe": pitch_safe,
            }
        )

    return {
        "measure": measure,
        "global_measure_index": int(base_record["global_measure_index"]),
        "source_image": str(source_path),
        "ground_truth_path": str(ground_truth_path),
        "staff_lines_y_px": list(staff_lines),
        "mean_staff_spacing_px": round(spacing, 6),
        "gt_count": len(ground_truth),
        "caps": cap_reports,
    }


def match_points(
    candidates: Sequence[dict[str, Any]],
    ground_truth: Sequence[dict[str, Any]],
    *,
    eligibility: Callable[[float, float], bool],
    tolerance_description: str,
) -> dict[str, Any]:
    """Match eligible points by distance with stable index tie-breaks."""
    gt_points = [
        (
            str(item.get("id", f"n{index + 1:03d}")),
            float(item["center"]["x"]),
            float(item["center"]["y"]),
        )
        for index, item in enumerate(ground_truth)
    ]
    possible_pairs = []
    for candidate_index, candidate in enumerate(candidates):
        cx, cy = float(candidate["center"]["x"]), float(candidate["center"]["y"])
        for gt_index, (_, gx, gy) in enumerate(gt_points):
            dx = cx - gx
            dy = cy - gy
            distance = math.hypot(dx, dy)
            if eligibility(dx, dy):
                possible_pairs.append((distance, candidate_index, gt_index, dx, dy))

    used_candidates: set[int] = set()
    used_gt: set[int] = set()
    assignments = []
    for distance, candidate_index, gt_index, dx, dy in sorted(possible_pairs):
        if candidate_index in used_candidates or gt_index in used_gt:
            continue
        used_candidates.add(candidate_index)
        used_gt.add(gt_index)
        assignments.append(
            {
                "candidate_index": candidate_index,
                "candidate_id": candidates[candidate_index]["id"],
                "ground_truth_index": gt_index,
                "ground_truth_id": gt_points[gt_index][0],
                "dx_px": round(dx, 3),
                "dy_px": round(dy, 3),
                "distance_px": round(distance, 3),
            }
        )
    assignments.sort(key=lambda item: item["candidate_index"])
    tp = len(assignments)
    fp = len(candidates) - tp
    fn = len(ground_truth) - tp
    return {
        "tolerance": tolerance_description,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _f1(tp, fp, fn),
        "exact_coverage": tp == len(ground_truth),
        "assignments": assignments,
        "unmatched_candidate_ids": [
            candidate["id"]
            for index, candidate in enumerate(candidates)
            if index not in used_candidates
        ],
        "unmatched_ground_truth_ids": [
            gt_points[index][0] for index in range(len(gt_points)) if index not in used_gt
        ],
    }


def match_region_points(
    candidates: Sequence[dict[str, Any]],
    ground_truth: Sequence[dict[str, Any]],
    *,
    staff_lines: Sequence[int],
    margin: float = ANNOTATION_REGION_MARGIN,
) -> dict[str, Any]:
    """Match candidates inside expanded GT annotation ellipses."""
    if margin <= 0:
        raise ValueError("Annotation-region margin must be positive")
    possible_pairs = []
    for candidate_index, candidate in enumerate(candidates):
        cx, cy = float(candidate["center"]["x"]), float(candidate["center"]["y"])
        for gt_index, item in enumerate(ground_truth):
            geometry = item.get("annotation_geometry", {})
            radius_x = float(geometry.get("radius_x_px", 0.0)) * margin
            radius_y = float(geometry.get("radius_y_px", 0.0)) * margin
            if radius_x <= 0 or radius_y <= 0:
                raise ValueError(f"GT annotation ellipse has invalid radii at index {gt_index}")
            ellipse_center_x, ellipse_center_y = _annotation_ellipse_center(item)
            dx = cx - ellipse_center_x
            dy = cy - ellipse_center_y
            normalized_distance = math.sqrt((dx / radius_x) ** 2 + (dy / radius_y) ** 2)
            if normalized_distance <= 1.0:
                possible_pairs.append((normalized_distance, candidate_index, gt_index, dx, dy))

    used_candidates: set[int] = set()
    used_gt: set[int] = set()
    assignments = []
    for normalized_distance, candidate_index, gt_index, dx, dy in sorted(possible_pairs):
        if candidate_index in used_candidates or gt_index in used_gt:
            continue
        used_candidates.add(candidate_index)
        used_gt.add(gt_index)
        predicted_pitch = _natural_pitch_from_y(
            float(candidates[candidate_index]["center"]["y"]), staff_lines
        )
        gt_pitch = _natural_pitch_name(str(ground_truth[gt_index].get("pitch", "")))
        assignments.append(
            {
                "candidate_index": candidate_index,
                "candidate_id": candidates[candidate_index]["id"],
                "ground_truth_index": gt_index,
                "ground_truth_id": str(ground_truth[gt_index].get("id", f"n{gt_index + 1:03d}")),
                "dx_px": round(dx, 3),
                "dy_px": round(dy, 3),
                "distance_px": round(math.hypot(dx, dy), 3),
                "normalized_ellipse_distance": round(normalized_distance, 6),
                "predicted_natural_pitch": predicted_pitch,
                "ground_truth_natural_pitch": gt_pitch,
                "pitch_correct": predicted_pitch == gt_pitch,
            }
        )
    assignments.sort(key=lambda item: item["candidate_index"])
    tp = len(assignments)
    fp = len(candidates) - tp
    fn = len(ground_truth) - tp
    pitch_correct_count = sum(item["pitch_correct"] for item in assignments)
    return {
        "ellipse_margin": margin,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "top_k_region_recall": _ratio(tp, tp + fn),
        "f1": _f1(tp, fp, fn),
        "exact_coverage": tp == len(ground_truth),
        "pitch_correct_count": pitch_correct_count,
        "pitch_accuracy": _ratio(pitch_correct_count, tp),
        "assignments": assignments,
        "unmatched_candidate_ids": [
            candidate["id"]
            for index, candidate in enumerate(candidates)
            if index not in used_candidates
        ],
        "unmatched_ground_truth_ids": [
            str(item.get("id", f"n{index + 1:03d}"))
            for index, item in enumerate(ground_truth)
            if index not in used_gt
        ],
    }


def write_report(report: dict[str, Any], output_path: Path) -> tuple[Path, Path]:
    """Write JSON and concise Markdown reports beside one another."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path = output_path.with_suffix(".md")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return output_path, markdown_path


def _aggregate_by_cap(
    per_measure: Sequence[dict[str, Any]], caps: Sequence[int]
) -> list[dict[str, Any]]:
    aggregates = []
    for cap in caps:
        rows = [
            next(row for row in measure["caps"] if row["cap"] == cap) for measure in per_measure
        ]
        aggregates.append(
            {
                "cap": cap,
                "annotation_region": _aggregate_metric(rows, "annotation_region"),
                "legacy_euclidean": _aggregate_metric(rows, "legacy_euclidean"),
                "pitch_safe": _aggregate_metric(rows, "pitch_safe"),
            }
        )
    return aggregates


def _aggregate_metric(rows: Sequence[dict[str, Any]], metric: str) -> dict[str, Any]:
    values = [row[metric] for row in rows]
    tp = sum(value["tp"] for value in values)
    fp = sum(value["fp"] for value in values)
    fn = sum(value["fn"] for value in values)
    exact_count = sum(value["exact_coverage"] for value in values)
    result = {
        "candidate_count": sum(row["candidate_count"] for row in rows),
        "gt_count": sum(row["gt_count"] for row in rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _f1(tp, fp, fn),
        "exact_coverage": exact_count == len(rows),
        "exact_coverage_measure_count": exact_count,
        "measure_count": len(rows),
    }
    if metric == "annotation_region":
        pitch_correct_count = sum(value["pitch_correct_count"] for value in values)
        result["pitch_correct_count"] = pitch_correct_count
        result["pitch_accuracy"] = _ratio(pitch_correct_count, tp)
        result["top_k_region_recall"] = result["recall"]
    return result


def _selection_summary(aggregates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    best_region_f1 = max(aggregates, key=lambda row: (row["annotation_region"]["f1"], -row["cap"]))
    best_pitch_f1 = max(aggregates, key=lambda row: (row["pitch_safe"]["f1"], -row["cap"]))
    best_legacy_f1 = max(aggregates, key=lambda row: (row["legacy_euclidean"]["f1"], -row["cap"]))
    max_recall = max(row["pitch_safe"]["recall"] for row in aggregates)
    smallest_max_recall = min(
        row["cap"] for row in aggregates if row["pitch_safe"]["recall"] == max_recall
    )
    max_region_recall = max(row["annotation_region"]["recall"] for row in aggregates)
    smallest_max_region_recall = min(
        row["cap"] for row in aggregates if row["annotation_region"]["recall"] == max_region_recall
    )
    return {
        "best_deterministic_cap_by_f1": best_region_f1["cap"],
        "best_deterministic_f1": best_region_f1["annotation_region"]["f1"],
        "best_deterministic_cap_by_annotation_region_f1": best_region_f1["cap"],
        "best_deterministic_annotation_region_f1": best_region_f1["annotation_region"]["f1"],
        "best_deterministic_cap_by_pitch_safe_f1": best_pitch_f1["cap"],
        "best_deterministic_pitch_safe_f1": best_pitch_f1["pitch_safe"]["f1"],
        "best_deterministic_cap_by_legacy_f1": best_legacy_f1["cap"],
        "best_deterministic_legacy_f1": best_legacy_f1["legacy_euclidean"]["f1"],
        "max_pitch_safe_recall": max_recall,
        "smallest_cap_achieving_max_pitch_safe_recall": smallest_max_recall,
        "max_annotation_region_recall": max_region_recall,
        "smallest_cap_achieving_max_annotation_region_recall": smallest_max_region_recall,
        "tie_break": (
            "smallest cap wins when F1 ties; recall selection explicitly chooses smallest cap"
        ),
    }


def _markdown_report(report: dict[str, Any]) -> str:
    selection = report["selection"]
    lines = [
        f"# Notehead Proposal Baselines: `{report['slug']}` S{report['system_index']}",
        "",
        "Development-measure baseline only; this does not establish generalization beyond "
        "the annotated measures.",
        "",
        "- Primary region match: annotation ellipse from GT geometry expanded by `1.15x`; "
        "pitch accuracy is computed only on region-matched assignments.",
        "- Region recall in the table is top-K recall for the detector's ranked candidate list.",
        f"- Best primary annotation-region F1 cap: `{selection['best_deterministic_cap_by_f1']}` "
        f"(F1 `{selection['best_deterministic_f1']:.6f}`)",
        "- Smallest cap at maximum annotation-region recall: "
        f"`{selection['smallest_cap_achieving_max_annotation_region_recall']}` "
        f"(recall `{selection['max_annotation_region_recall']:.6f}`)",
        "- Smallest cap at maximum pitch-safe recall: "
        f"`{selection['smallest_cap_achieving_max_pitch_safe_recall']}` "
        f"(recall `{selection['max_pitch_safe_recall']:.6f}`)",
        "",
        "## Aggregate Micro Metrics",
        "",
        "| Cap | Region P/R/F1 | Region exact | Region pitch acc. | "
        "Legacy P/R/F1 | Pitch-safe P/R/F1 |",
        "| ---: | --- | :---: | :---: | --- | --- |",
    ]
    for row in report["aggregate_by_cap"]:
        legacy = row["legacy_euclidean"]
        pitch = row["pitch_safe"]
        region = row["annotation_region"]
        lines.append(
            f"| {row['cap']} | {region['precision']:.3f} / "
            f"{region['recall']:.3f} / {region['f1']:.3f} | "
            f"{'yes' if region['exact_coverage'] else 'no'} | {region['pitch_accuracy']:.3f} | "
            f"{legacy['precision']:.3f} / {legacy['recall']:.3f} / {legacy['f1']:.3f} | "
            f"{pitch['precision']:.3f} / {pitch['recall']:.3f} / {pitch['f1']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Per Measure",
            "",
            "| Measure | Cap | Candidates | GT | Region TP/FP/FN | Region recall | "
            "Pitch accuracy | Legacy TP/FP/FN | Pitch-safe TP/FP/FN |",
            "| ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |",
        ]
    )
    for measure in report["per_measure"]:
        for row in measure["caps"]:
            legacy = row["legacy_euclidean"]
            pitch = row["pitch_safe"]
            region = row["annotation_region"]
            lines.append(
                f"| {measure['measure']} | {row['cap']} | {row['candidate_count']} | "
                f"{measure['gt_count']} | "
                f"{region['tp']}/{region['fp']}/{region['fn']} | {region['recall']:.3f} | "
                f"{region['pitch_accuracy']:.3f} | {legacy['tp']}/{legacy['fp']}/{legacy['fn']} | "
                f"{pitch['tp']}/{pitch['fp']}/{pitch['fn']} |"
            )
    lines.append("")
    return "\n".join(lines)


def _candidate_points(candidates: Sequence[Any]) -> list[dict[str, Any]]:
    points = []
    for index, candidate in enumerate(candidates, start=1):
        if isinstance(candidate, dict):
            center = candidate["center"]
            score = candidate.get("score")
        else:
            center = {"x": candidate.center[0], "y": candidate.center[1]}
            score = getattr(candidate, "score", None)
        point = {
            "id": f"c{index:03d}",
            "center": {"x": round(float(center["x"]), 3), "y": round(float(center["y"]), 3)},
        }
        if score is not None:
            point["score"] = round(float(score), 6)
        points.append(point)
    return points


def _load_ground_truth_fixture(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    noteheads = payload.get("noteheads") if isinstance(payload, dict) else payload
    if not isinstance(noteheads, list):
        raise ValueError(f"Ground-truth JSON has no noteheads list: {path}")
    return [item for item in noteheads if isinstance(item, dict) and "center" in item]


def _annotation_ellipse_center(item: dict[str, Any]) -> tuple[float, float]:
    geometry = item.get("annotation_geometry", {})
    bbox = geometry.get("bbox_px") if isinstance(geometry, dict) else None
    if isinstance(bbox, dict) and all(key in bbox for key in ("left", "top", "right", "bottom")):
        return (
            (float(bbox["left"]) + float(bbox["right"])) / 2.0,
            (float(bbox["top"]) + float(bbox["bottom"])) / 2.0,
        )
    center = item["center"]
    return float(center["x"]), float(center["y"])


_PITCH_RE = re.compile(r"^\s*([A-Ga-g])(?:[#b]*)(-?\d+)\s*$")


def _natural_pitch_name(pitch: str) -> str | None:
    match = _PITCH_RE.match(pitch)
    return f"{match.group(1).upper()}{match.group(2)}" if match else None


def _natural_pitch_from_y(y: float, staff_lines: Sequence[int]) -> str | None:
    spacing = detector._staff_spacing(list(staff_lines))
    if spacing is None or spacing <= 0 or not staff_lines:
        return None
    step = math.floor((y - float(staff_lines[0])) / (spacing / 2.0) + 0.5)
    letters = ("C", "D", "E", "F", "G", "A", "B")
    top_letter_index = letters.index("F")
    letter_index = (top_letter_index - step) % len(letters)
    octave = 5 + (top_letter_index - step) // len(letters)
    return f"{letters[letter_index]}{octave}"


def _ground_truth_path(
    ground_truth_dir: Path, *, slug: str, system_index: int, measure: int
) -> Path:
    return ground_truth_dir / f"{slug}_system_{system_index:03d}_measure_{measure:03d}.json"


def _default_output_path(out_dir: Path, slug: str, system_index: int) -> Path:
    return (
        out_dir
        / slug
        / "vlm_melody_inputs"
        / f"system_{system_index:03d}"
        / "notehead_proposal_baselines.json"
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _positive_unique(values: Sequence[int], label: str) -> list[int]:
    result = sorted(set(values))
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"{label} values must be positive")
    return result


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return round(2 * precision * recall / (precision + recall), 6) if precision + recall else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
