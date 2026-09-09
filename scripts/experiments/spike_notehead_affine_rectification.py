"""Test GT-blind affine staff rectification on the fixed Aviador development set.

The experiment fits one shared straight-line staff slope from the complete system
image, rectifies each raw measure crop, and reruns the existing staff-grid-density
detector unchanged. All image preparation and candidate generation finish before
the coordinate ground truth is loaded.

This is a bounded spike, not production pipeline code. Its fixed development
gates come from ``docs/VLM_MELODY_SPIKE.md`` and must not be tuned against the
four annotated measures.

Example:
    ./.venv/bin/python scripts/experiments/spike_notehead_affine_rectification.py
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from score2abc.utils.imaging import estimate_ink_threshold  # noqa: E402
from scripts import build_vlm_notehead_candidates as detector  # noqa: E402
from scripts import eval_vlm_notehead_proposal_baselines as evaluator  # noqa: E402

DEFAULT_OUT_DIR = REPO_ROOT / "out"
DEFAULT_OUTPUT_DIR = DEFAULT_OUT_DIR / "experiments/notehead_affine_rectification"
DEFAULT_SLUG = "jaime-llanos_12_aviador_pasillo_fulgencio-garcia"
SYSTEM_INDEX = 1
MEASURES = (1, 2, 3, 4)
CAPS = (4, 8, 24)
GT_DIR = REPO_ROOT / "tests/fixtures/vlm_melody/notehead_ground_truth"

# Fixed before evaluation. These values are intentionally not CLI-tunable.
MAX_ABS_SLOPE = 0.02
SLOPE_STEP = 0.00025
MAX_INTERCEPT_OFFSET_PX = 6
LINE_HALF_BAND_PX = 1
VERTICAL_COLUMN_INK_LIMIT = 0.45


@dataclass(frozen=True)
class ParallelStaffModel:
    slope: float
    center_x: float
    flat_lines: tuple[int, ...]
    support: float
    zero_slope_support: float
    sampled_column_count: int
    threshold: int


@dataclass
class PreparedMeasure:
    measure: int
    source_path: Path
    context_path: Path
    image: Image.Image
    baseline_staff_lines: list[int]
    x_left_in_system: int
    baseline_by_cap: dict[int, list[Any]]
    rectified_image: Image.Image
    rectified_path: Path
    rectified_by_cap: dict[int, list[Any]]
    mapped_rectified_by_cap: dict[int, list[dict[str, Any]]]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = run_experiment(
            args.out_dir,
            slug=args.slug,
            output_dir=args.output_dir,
        )
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(args.output_dir / "report.json")
    print(args.output_dir / "report.md")
    print(f"gate: {report['decision']['status']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def run_experiment(out_dir: Path, *, slug: str, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = {
        measure: detector._load_or_build_base_record(
            out_dir,
            slug=slug,
            system_index=SYSTEM_INDEX,
            measure_index=measure,
        )
        for measure in MEASURES
    }
    first_context_path = detector._resolve_path(out_dir, records[MEASURES[0]]["paths"]["context"])
    first_context = _load_json(first_context_path)
    system_path = detector._resolve_path(out_dir, first_context["paths"]["source_system"])
    nominal_lines = [int(value) for value in first_context["staff_lines_y_px_in_system"]]
    with Image.open(system_path) as opened:
        system_image = opened.convert("RGB")
    model = fit_parallel_staff_model(system_image, nominal_lines)
    _write_staff_model_overlay(system_image, model, output_dir / "system_staff_model.png")

    # This entire pass is GT-blind. Do not move a GT read above its completion.
    prepared = {
        measure: _prepare_measure(
            out_dir,
            record=records[measure],
            measure=measure,
            model=model,
            output_dir=output_dir,
        )
        for measure in MEASURES
    }

    ground_truth = {
        measure: evaluator._load_ground_truth_fixture(
            evaluator._ground_truth_path(
                GT_DIR,
                slug=slug,
                system_index=SYSTEM_INDEX,
                measure=measure,
            )
        )
        for measure in MEASURES
    }
    per_measure = [
        _evaluate_measure(prepared[measure], ground_truth[measure], model=model)
        for measure in MEASURES
    ]
    baseline_aggregate = evaluator._aggregate_by_cap(
        [{"caps": row["baseline"]["caps"]} for row in per_measure], CAPS
    )
    rectified_aggregate = evaluator._aggregate_by_cap(
        [{"caps": row["rectified"]["caps"]} for row in per_measure], CAPS
    )
    gates = _fixed_gates(per_measure, baseline_aggregate, rectified_aggregate)
    status = "pass" if all(gate["passed"] for gate in gates) else "fail"
    report = {
        "schema_version": 1,
        "kind": "notehead_affine_rectification_spike",
        "slug": slug,
        "system_index": SYSTEM_INDEX,
        "measures": list(MEASURES),
        "caps": list(CAPS),
        "candidate_generation_uses_ground_truth": False,
        "audit": {
            "order": (
                "fit system model; rectify all measures; generate all baseline and rectified "
                "candidate lists; then load coordinate GT"
            ),
            "parameter_source": "fixed constants declared in this script before evaluation",
            "production_code_changed": False,
        },
        "staff_model": {
            "source_image": str(system_path),
            "slope_y_per_x": round(model.slope, 8),
            "angle_degrees": round(math.degrees(math.atan(model.slope)), 6),
            "center_x": round(model.center_x, 3),
            "flat_lines_y": list(model.flat_lines),
            "support": round(model.support, 6),
            "zero_slope_support": round(model.zero_slope_support, 6),
            "support_delta": round(model.support - model.zero_slope_support, 6),
            "sampled_column_count": model.sampled_column_count,
            "ink_threshold": model.threshold,
            "search": {
                "max_abs_slope": MAX_ABS_SLOPE,
                "slope_step": SLOPE_STEP,
                "max_intercept_offset_px": MAX_INTERCEPT_OFFSET_PX,
                "line_half_band_px": LINE_HALF_BAND_PX,
                "vertical_column_ink_limit": VERTICAL_COLUMN_INK_LIMIT,
            },
        },
        "per_measure": per_measure,
        "aggregate": {
            "baseline": baseline_aggregate,
            "rectified": rectified_aggregate,
        },
        "decision": {
            "status": status,
            "gates": gates,
            "recommendation": (
                "Annotate system 2 measures 1-4 for blind affine validation."
                if status == "pass"
                else "Stop affine tuning and prototype candidate-confirming human review."
            ),
        },
    }
    _write_diagnostics(prepared, ground_truth, per_measure, model=model, output_dir=output_dir)
    _write_report(report, output_dir)
    return report


def fit_parallel_staff_model(
    image: Image.Image, nominal_lines: Sequence[int]
) -> ParallelStaffModel:
    """Fit five parallel lines from image ink without annotations."""
    if len(nominal_lines) != 5:
        raise ValueError(f"Expected five nominal staff lines, got {nominal_lines!r}")
    gray = ImageOps.grayscale(image)
    threshold = estimate_ink_threshold(gray)
    width, height = gray.size
    pixels = gray.load()
    ink = [[pixels[x, y] < threshold for x in range(width)] for y in range(height)]
    spacing = statistics.mean(
        second - first for first, second in zip(nominal_lines, nominal_lines[1:], strict=False)
    )
    band_top = max(0, round(nominal_lines[0] - spacing))
    band_bottom = min(height, round(nominal_lines[-1] + spacing) + 1)
    sample_step = max(2, round(width / 900))
    start = max(0, round(width * 0.03))
    stop = min(width, round(width * 0.97))
    sample_xs = []
    for x in range(start, stop, sample_step):
        column_ink = sum(ink[y][x] for y in range(band_top, band_bottom))
        if column_ink / max(1, band_bottom - band_top) <= VERTICAL_COLUMN_INK_LIMIT:
            sample_xs.append(x)
    if len(sample_xs) < 20:
        raise ValueError("Too few usable columns to estimate a staff slope")

    center_x = (width - 1) / 2.0
    offsets = range(-MAX_INTERCEPT_OFFSET_PX, MAX_INTERCEPT_OFFSET_PX + 1)

    def evaluate_slope(slope: float) -> tuple[float, tuple[int, ...]]:
        supports = []
        chosen_lines = []
        for nominal_y in nominal_lines:
            choices = []
            for offset in offsets:
                center_y = nominal_y + offset
                hits = 0
                valid = 0
                for x in sample_xs:
                    y = round(center_y + slope * (x - center_x))
                    if y < 0 or y >= height:
                        continue
                    valid += 1
                    if any(
                        ink[band_y][x]
                        for band_y in range(
                            max(0, y - LINE_HALF_BAND_PX),
                            min(height, y + LINE_HALF_BAND_PX + 1),
                        )
                    ):
                        hits += 1
                support = hits / valid if valid else 0.0
                choices.append((support, -abs(offset), -offset, center_y))
            best = max(choices)
            supports.append(best[0])
            chosen_lines.append(int(best[3]))
        return statistics.mean(supports), tuple(chosen_lines)

    slope_steps = round(MAX_ABS_SLOPE / SLOPE_STEP)
    tested = []
    for step in range(-slope_steps, slope_steps + 1):
        slope = step * SLOPE_STEP
        support, flat_lines = evaluate_slope(slope)
        tested.append((support, -abs(slope), -slope, slope, flat_lines))
    best = max(tested)
    zero = next(row for row in tested if row[3] == 0.0)
    return ParallelStaffModel(
        slope=float(best[3]),
        center_x=center_x,
        flat_lines=tuple(best[4]),
        support=float(best[0]),
        zero_slope_support=float(zero[0]),
        sampled_column_count=len(sample_xs),
        threshold=threshold,
    )


def rectify_measure_image(
    image: Image.Image, *, model: ParallelStaffModel, x_left_in_system: int
) -> Image.Image:
    """Flatten the fitted staff; Pillow coefficients map output back to source."""
    y_offset = model.slope * (x_left_in_system - model.center_x)
    return image.transform(
        image.size,
        Image.Transform.AFFINE,
        (1.0, 0.0, 0.0, model.slope, 1.0, y_offset),
        resample=Image.Resampling.BICUBIC,
        fillcolor=(255, 255, 255),
    )


def map_rectified_point_to_source(
    x: float,
    y: float,
    *,
    model: ParallelStaffModel,
    x_left_in_system: int,
) -> tuple[float, float]:
    source_y = y + model.slope * (x_left_in_system + x - model.center_x)
    return x, source_y


def _prepare_measure(
    out_dir: Path,
    *,
    record: dict[str, Any],
    measure: int,
    model: ParallelStaffModel,
    output_dir: Path,
) -> PreparedMeasure:
    context_path = detector._resolve_path(out_dir, record["paths"]["context"])
    context = _load_json(context_path)
    source_path = detector._source_path(out_dir, record, "raw")
    baseline_staff_lines = detector._staff_lines_for_source(context, "raw")
    x_left = int(context["x_bounds_px"]["left"])
    with Image.open(source_path) as opened:
        image = opened.convert("RGB")
    baseline_by_cap = {
        cap: detector.detect_staff_grid_density_candidates(
            image,
            staff_lines=baseline_staff_lines,
            max_candidates=cap,
        )
        for cap in CAPS
    }
    rectified = rectify_measure_image(image, model=model, x_left_in_system=x_left)
    rectified_path = output_dir / f"measure_{measure:03d}_rectified.png"
    rectified.save(rectified_path)
    rectified_by_cap = {
        cap: detector.detect_staff_grid_density_candidates(
            rectified,
            staff_lines=list(model.flat_lines),
            max_candidates=cap,
        )
        for cap in CAPS
    }
    mapped_by_cap = {
        cap: _mapped_candidate_points(
            candidates,
            model=model,
            x_left_in_system=x_left,
        )
        for cap, candidates in rectified_by_cap.items()
    }
    return PreparedMeasure(
        measure=measure,
        source_path=source_path,
        context_path=context_path,
        image=image,
        baseline_staff_lines=baseline_staff_lines,
        x_left_in_system=x_left,
        baseline_by_cap=baseline_by_cap,
        rectified_image=rectified,
        rectified_path=rectified_path,
        rectified_by_cap=rectified_by_cap,
        mapped_rectified_by_cap=mapped_by_cap,
    )


def _mapped_candidate_points(
    candidates: Sequence[Any], *, model: ParallelStaffModel, x_left_in_system: int
) -> list[dict[str, Any]]:
    points = []
    for index, candidate in enumerate(candidates, start=1):
        rectified_x, rectified_y = candidate.center
        source_x, source_y = map_rectified_point_to_source(
            float(rectified_x),
            float(rectified_y),
            model=model,
            x_left_in_system=x_left_in_system,
        )
        points.append(
            {
                "id": f"c{index:03d}",
                "center": {"x": round(source_x, 3), "y": round(source_y, 3)},
                "rectified_center": {
                    "x": round(float(rectified_x), 3),
                    "y": round(float(rectified_y), 3),
                },
                "score": round(float(candidate.score), 6),
            }
        )
    return points


def _evaluate_measure(
    prepared: PreparedMeasure,
    ground_truth: Sequence[dict[str, Any]],
    *,
    model: ParallelStaffModel,
) -> dict[str, Any]:
    spacing = detector._staff_spacing(prepared.baseline_staff_lines)
    if spacing is None or spacing <= 0:
        raise ValueError(f"Invalid staff spacing for measure {prepared.measure}")
    baseline_caps = []
    rectified_caps = []
    for cap in CAPS:
        baseline_points = evaluator._candidate_points(prepared.baseline_by_cap[cap])
        baseline_caps.append(
            _cap_metrics(
                cap,
                baseline_points,
                ground_truth,
                spacing=spacing,
                pitch_staff_lines=prepared.baseline_staff_lines,
            )
        )
        rectified_points = prepared.mapped_rectified_by_cap[cap]
        rectified_caps.append(
            _cap_metrics(
                cap,
                rectified_points,
                ground_truth,
                spacing=spacing,
                pitch_staff_lines=list(model.flat_lines),
                use_rectified_pitch_y=True,
            )
        )
    return {
        "measure": prepared.measure,
        "source_image": str(prepared.source_path),
        "rectified_image": str(prepared.rectified_path),
        "gt_count": len(ground_truth),
        "baseline": {"staff_lines_y": prepared.baseline_staff_lines, "caps": baseline_caps},
        "rectified": {"staff_lines_y": list(model.flat_lines), "caps": rectified_caps},
    }


def _cap_metrics(
    cap: int,
    candidates: Sequence[dict[str, Any]],
    ground_truth: Sequence[dict[str, Any]],
    *,
    spacing: float,
    pitch_staff_lines: Sequence[int],
    use_rectified_pitch_y: bool = False,
) -> dict[str, Any]:
    legacy_tolerance = max(
        evaluator.LEGACY_MIN_TOLERANCE_PX,
        evaluator.LEGACY_SPACING_FACTOR * spacing,
    )
    pitch_x = evaluator.PITCH_SAFE_X_FACTOR * spacing
    pitch_y = evaluator.PITCH_SAFE_Y_FACTOR * spacing
    legacy = evaluator.match_points(
        candidates,
        ground_truth,
        eligibility=lambda dx, dy: math.hypot(dx, dy) <= legacy_tolerance,
        tolerance_description=f"Euclidean distance <= {legacy_tolerance:.3f} px",
    )
    region = evaluator.match_region_points(
        candidates,
        ground_truth,
        staff_lines=pitch_staff_lines,
        margin=evaluator.ANNOTATION_REGION_MARGIN,
    )
    if use_rectified_pitch_y:
        by_id = {candidate["id"]: candidate for candidate in candidates}
        for assignment in region["assignments"]:
            candidate = by_id[assignment["candidate_id"]]
            rectified_y = float(candidate["rectified_center"]["y"])
            gt_item = ground_truth[assignment["ground_truth_index"]]
            predicted = evaluator._natural_pitch_from_y(rectified_y, pitch_staff_lines)
            expected = evaluator._natural_pitch_name(str(gt_item.get("pitch", "")))
            assignment["predicted_natural_pitch"] = predicted
            assignment["ground_truth_natural_pitch"] = expected
            assignment["pitch_correct"] = predicted == expected
        pitch_correct = sum(item["pitch_correct"] for item in region["assignments"])
        region["pitch_correct_count"] = pitch_correct
        region["pitch_accuracy"] = evaluator._ratio(pitch_correct, region["tp"])
    pitch_safe = evaluator.match_points(
        candidates,
        ground_truth,
        eligibility=lambda dx, dy: abs(dx) <= pitch_x and abs(dy) <= pitch_y,
        tolerance_description=f"abs(dx) <= {pitch_x:.3f} px and abs(dy) <= {pitch_y:.3f} px",
    )
    return {
        "cap": cap,
        "candidate_count": len(candidates),
        "gt_count": len(ground_truth),
        "candidates": list(candidates),
        "annotation_region": region,
        "legacy_euclidean": legacy,
        "pitch_safe": pitch_safe,
    }


def _fixed_gates(
    per_measure: Sequence[dict[str, Any]],
    baseline_aggregate: Sequence[dict[str, Any]],
    rectified_aggregate: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    m1 = next(row for row in per_measure if row["measure"] == 1)
    baseline_m1_cap8 = _cap_row(m1["baseline"]["caps"], 8)["annotation_region"]["tp"]
    rectified_m1_cap8 = _cap_row(m1["rectified"]["caps"], 8)["annotation_region"]["tp"]
    baseline_cap8 = _cap_row(baseline_aggregate, 8)
    rectified_cap8 = _cap_row(rectified_aggregate, 8)
    baseline_cap4 = _cap_row(baseline_aggregate, 4)
    rectified_cap4 = _cap_row(rectified_aggregate, 4)
    return [
        {
            "id": "s1_m1_cap8_region_tp",
            "baseline": baseline_m1_cap8,
            "observed": rectified_m1_cap8,
            "required": 4,
            "comparison": ">=",
            "passed": rectified_m1_cap8 >= 4,
        },
        {
            "id": "aggregate_cap8_region_tp",
            "baseline": baseline_cap8["annotation_region"]["tp"],
            "observed": rectified_cap8["annotation_region"]["tp"],
            "required": 12,
            "comparison": ">=",
            "passed": rectified_cap8["annotation_region"]["tp"] >= 12,
        },
        {
            "id": "aggregate_cap8_pitch_safe_tp",
            "baseline": baseline_cap8["pitch_safe"]["tp"],
            "observed": rectified_cap8["pitch_safe"]["tp"],
            "required": 11,
            "comparison": ">=",
            "passed": rectified_cap8["pitch_safe"]["tp"] >= 11,
        },
        {
            "id": "aggregate_cap4_region_f1",
            "baseline": baseline_cap4["annotation_region"]["f1"],
            "observed": rectified_cap4["annotation_region"]["f1"],
            "required": 0.533,
            "comparison": ">=",
            "passed": rectified_cap4["annotation_region"]["f1"] >= 0.533,
        },
    ]


def _cap_row(rows: Sequence[dict[str, Any]], cap: int) -> dict[str, Any]:
    return next(row for row in rows if row["cap"] == cap)


def _write_staff_model_overlay(
    image: Image.Image, model: ParallelStaffModel, output_path: Path
) -> None:
    overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for flat_y in model.flat_lines:
        left_y = flat_y + model.slope * (0 - model.center_x)
        right_y = flat_y + model.slope * (overlay.width - 1 - model.center_x)
        draw.line((0, left_y, overlay.width - 1, right_y), fill=(0, 180, 70), width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)


def _write_diagnostics(
    prepared: dict[int, PreparedMeasure],
    ground_truth: dict[int, Sequence[dict[str, Any]]],
    per_measure: Sequence[dict[str, Any]],
    *,
    model: ParallelStaffModel,
    output_dir: Path,
) -> None:
    for measure in MEASURES:
        row = next(item for item in per_measure if item["measure"] == measure)
        baseline_cap8 = _cap_row(row["baseline"]["caps"], 8)
        rectified_cap8 = _cap_row(row["rectified"]["caps"], 8)
        baseline = _comparison_overlay(
            prepared[measure].image,
            baseline_cap8["candidates"],
            ground_truth[measure],
            baseline_cap8["annotation_region"],
            title="baseline cap 8",
        )
        rectified = _comparison_overlay(
            prepared[measure].image,
            rectified_cap8["candidates"],
            ground_truth[measure],
            rectified_cap8["annotation_region"],
            title="rectified candidates mapped to source",
        )
        _draw_staff_trace(rectified, model, prepared[measure].x_left_in_system)
        baseline.save(output_dir / f"measure_{measure:03d}_baseline_cap8_overlay.png")
        rectified.save(output_dir / f"measure_{measure:03d}_rectified_cap8_overlay.png")
        _write_pair(
            baseline,
            rectified,
            output_dir / f"measure_{measure:03d}_baseline_vs_rectified.png",
        )


def _comparison_overlay(
    image: Image.Image,
    candidates: Sequence[dict[str, Any]],
    ground_truth: Sequence[dict[str, Any]],
    metric: dict[str, Any],
    *,
    title: str,
) -> Image.Image:
    overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    matched_candidates = {item["candidate_id"] for item in metric["assignments"]}
    matched_gt = {item["ground_truth_id"] for item in metric["assignments"]}
    for candidate in candidates:
        x = float(candidate["center"]["x"])
        y = float(candidate["center"]["y"])
        color = (0, 165, 0) if candidate["id"] in matched_candidates else (225, 35, 20)
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=color, width=2)
        draw.text((x + 8, y - 6), candidate["id"], fill=color, font=font)
    for item in ground_truth:
        geometry = item["annotation_geometry"]
        bbox = geometry["bbox_px"]
        gt_id = str(item["id"])
        color = (0, 145, 0) if gt_id in matched_gt else (150, 0, 190)
        draw.ellipse(
            (bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]),
            outline=color,
            width=2,
        )
    draw.rectangle((0, 0, min(230, overlay.width - 1), 14), fill="white")
    draw.text((3, 2), title, fill="black", font=font)
    return overlay


def _draw_staff_trace(image: Image.Image, model: ParallelStaffModel, x_left_in_system: int) -> None:
    draw = ImageDraw.Draw(image)
    for flat_y in model.flat_lines:
        left_y = flat_y + model.slope * (x_left_in_system - model.center_x)
        right_y = flat_y + model.slope * (x_left_in_system + image.width - 1 - model.center_x)
        draw.line((0, left_y, image.width - 1, right_y), fill=(20, 130, 230), width=1)


def _write_pair(left: Image.Image, right: Image.Image, output_path: Path) -> None:
    gap = 12
    paired = Image.new(
        "RGB",
        (left.width + right.width + gap, max(left.height, right.height)),
        "white",
    )
    paired.paste(left, (0, 0))
    paired.paste(right, (left.width + gap, 0))
    paired.save(output_path)


def _write_report(report: dict[str, Any], output_dir: Path) -> None:
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Affine Staff Rectification Spike",
        "",
        f"Decision: **{report['decision']['status'].upper()}**",
        "",
        "The fit and all candidate lists were generated before coordinate GT was loaded.",
        "",
        "## Staff Model",
        "",
        f"- Slope: `{report['staff_model']['slope_y_per_x']:.8f}` y/x "
        f"(`{report['staff_model']['angle_degrees']:.6f}` degrees)",
        f"- Image support: `{report['staff_model']['support']:.6f}` "
        f"(zero-slope `{report['staff_model']['zero_slope_support']:.6f}`)",
        f"- Flat lines: `{report['staff_model']['flat_lines_y']}`",
        "",
        "## Fixed Gates",
        "",
        "| Gate | Baseline | Rectified | Required | Pass |",
        "| --- | ---: | ---: | ---: | :---: |",
    ]
    for gate in report["decision"]["gates"]:
        lines.append(
            f"| `{gate['id']}` | {gate['baseline']} | {gate['observed']} | "
            f"{gate['comparison']} {gate['required']} | "
            f"{'yes' if gate['passed'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "| Method | Cap | Region TP/FP/FN | Region P/R/F1 | Pitch-safe TP/FP/FN |",
            "| --- | ---: | --- | --- | --- |",
        ]
    )
    for method in ("baseline", "rectified"):
        for row in report["aggregate"][method]:
            region = row["annotation_region"]
            pitch = row["pitch_safe"]
            lines.append(
                f"| {method} | {row['cap']} | {region['tp']}/{region['fp']}/{region['fn']} | "
                f"{region['precision']:.3f}/{region['recall']:.3f}/{region['f1']:.3f} | "
                f"{pitch['tp']}/{pitch['fp']}/{pitch['fn']} |"
            )
    lines.extend(["", f"Next: {report['decision']['recommendation']}", ""])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
