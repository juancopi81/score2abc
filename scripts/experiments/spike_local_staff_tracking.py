"""Track locally curved staff lines and evaluate pitch mapping on reviewed heads.

This spike addresses a specific failure of constant-y staff geometry. It computes
all staff trajectories from image ink before opening any human review fixture.
Center refinement and pitch evaluation are then conditioned on human-confirmed
notehead identities and search regions from the review fixtures.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SLUG = "jaime-llanos_12_aviador_pasillo_fulgencio-garcia"
DEFAULT_MEASURES = (1, 2, 3, 4)
REVIEW_FIXTURE_DIR = REPO_ROOT / "tests/fixtures/vlm_melody/notehead_reviews"
OUTPUT_SUBDIR = Path("experiments/local_staff_tracking")
PITCH_RE = re.compile(r"^([A-G])([#b]?)(-?\d+)$")
DIATONIC_LETTERS = ("C", "D", "E", "F", "G", "A", "B")


@dataclass(frozen=True)
class TrackedMeasure:
    measure: int
    image_path: Path
    context_path: Path
    base_lines: tuple[float, ...]
    spacing: float
    shifts: tuple[int, ...]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        measures = tuple(args.measure or DEFAULT_MEASURES)
        tracked = [
            track_measure(
                args.out_dir,
                slug=args.slug,
                system_index=args.system,
                measure=measure,
            )
            for measure in measures
        ]
        report = evaluate_tracks(
            tracked,
            slug=args.slug,
            system_index=args.system,
            key_flat_letters={"B"},
        )
        report_path = write_report(report, tracked, args.out_dir / OUTPUT_SUBDIR)
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(report_path)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", type=Path, default=Path("out"))
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--system", type=int, default=1)
    parser.add_argument("--measure", action="append", type=int)
    return parser


def track_measure(
    out_dir: Path,
    *,
    slug: str,
    system_index: int,
    measure: int,
) -> TrackedMeasure:
    measure_dir = out_dir / slug / "vlm_melody_inputs" / f"system_{system_index:03d}"
    image_path = measure_dir / f"measure_{measure:03d}_raw.png"
    context_path = measure_dir / f"measure_{measure:03d}_context.json"
    if not image_path.exists() or not context_path.exists():
        raise FileNotFoundError(
            f"Missing measure input for system {system_index}, measure {measure}"
        )
    context = json.loads(context_path.read_text(encoding="utf-8"))
    base_lines = tuple(float(value) for value in context["staff_lines_y_px_in_system"])
    spacing = _median_spacing(base_lines)
    with Image.open(image_path) as opened:
        image = opened.convert("L")
    shifts = track_common_staff_shift(image, base_lines=base_lines, spacing=spacing)
    return TrackedMeasure(
        measure=measure,
        image_path=image_path.resolve(),
        context_path=context_path.resolve(),
        base_lines=base_lines,
        spacing=spacing,
        shifts=tuple(shifts),
    )


def track_common_staff_shift(
    image: Image.Image,
    *,
    base_lines: Sequence[float],
    spacing: float,
    max_shift: int | None = None,
    horizontal_radius: int = 3,
    vertical_radius: int = 1,
) -> list[int]:
    """Use dynamic programming to follow one smooth offset shared by five lines."""
    gray = image.convert("L")
    width, height = gray.size
    if width <= 0 or height <= 0:
        raise ValueError("Cannot track an empty image")
    if len(base_lines) != 5 or spacing <= 0:
        raise ValueError("Staff tracking requires five ordered lines and positive spacing")
    shift_limit = max_shift if max_shift is not None else max(3, round(spacing * 0.75))
    states = tuple(range(-shift_limit, shift_limit + 1))
    ink_threshold = _ink_threshold(gray)

    emissions = [
        [
            _horizontal_line_support(
                gray,
                x=x,
                shifted_lines=[line + shift for line in base_lines],
                ink_threshold=ink_threshold,
                horizontal_radius=horizontal_radius,
                vertical_radius=vertical_radius,
            )
            for shift in states
        ]
        for x in range(width)
    ]

    transition_penalty = 1.8
    scores = [float(value) for value in emissions[0]]
    backpointers: list[list[int]] = []
    for x in range(1, width):
        next_scores: list[float] = []
        pointers: list[int] = []
        for state_index, shift in enumerate(states):
            best_index = max(
                range(len(states)),
                key=lambda previous: (
                    scores[previous]
                    - transition_penalty * abs(states[previous] - shift)
                    - 0.04 * abs(shift)
                ),
            )
            next_scores.append(
                scores[best_index]
                - transition_penalty * abs(states[best_index] - shift)
                - 0.04 * abs(shift)
                + emissions[x][state_index]
            )
            pointers.append(best_index)
        scores = next_scores
        backpointers.append(pointers)

    state_index = max(range(len(states)), key=scores.__getitem__)
    path = [states[state_index]]
    for pointers in reversed(backpointers):
        state_index = pointers[state_index]
        path.append(states[state_index])
    path.reverse()
    return path


def _horizontal_line_support(
    image: Image.Image,
    *,
    x: int,
    shifted_lines: Sequence[float],
    ink_threshold: int,
    horizontal_radius: int,
    vertical_radius: int,
) -> float:
    width, height = image.size
    support = 0.0
    for line_y in shifted_lines:
        center_y = round(line_y)
        for source_x in range(max(0, x - horizontal_radius), min(width, x + horizontal_radius + 1)):
            if any(
                image.getpixel((source_x, source_y)) <= ink_threshold
                for source_y in range(
                    max(0, center_y - vertical_radius),
                    min(height, center_y + vertical_radius + 1),
                )
            ):
                support += 1.0
    return support


def evaluate_tracks(
    tracked: Sequence[TrackedMeasure],
    *,
    slug: str,
    system_index: int,
    key_flat_letters: set[str],
) -> dict[str, Any]:
    """Load review labels only after every image trajectory has been computed."""
    results: list[dict[str, Any]] = []
    for measure in tracked:
        fixture_path = REVIEW_FIXTURE_DIR / (
            f"{slug}_system_{system_index:03d}_measure_{measure.measure:03d}.json"
        )
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        candidate_by_id = {candidate["id"]: candidate for candidate in fixture["candidates"]}
        with Image.open(measure.image_path) as opened:
            image = opened.convert("L")
        heads = sorted(fixture["final_noteheads"], key=lambda head: int(head["order"]))
        for head in heads:
            x = min(max(round(float(head["center"]["x"])), 0), len(measure.shifts) - 1)
            y = float(head["center"]["y"])
            constant_position = staff_position(
                y,
                bottom_line_y=measure.base_lines[-1],
                spacing=measure.spacing,
            )
            tracked_position = staff_position(
                y,
                bottom_line_y=measure.base_lines[-1] + measure.shifts[x],
                spacing=measure.spacing,
            )
            source = head.get("source", {})
            source_kind = source.get("kind")
            candidate_id = source.get("candidate_id")
            candidate = candidate_by_id.get(candidate_id)
            if source_kind == "candidate":
                if candidate is None:
                    raise ValueError(
                        f"Final notehead references unknown candidate {candidate_id!r}"
                    )
                if candidate.get("label") != "accepted":
                    raise ValueError(
                        f"Final notehead references non-accepted candidate {candidate_id!r}"
                    )
                bbox = candidate["bbox"]
                search_region_source = "accepted_candidate_bbox"
            elif source_kind == "manual":
                bbox = {
                    "left": x - round(measure.spacing * 0.45),
                    "top": y - round(measure.spacing * 0.4),
                    "right": x + round(measure.spacing * 0.45),
                    "bottom": y + round(measure.spacing * 0.4),
                }
                search_region_source = "reviewed_manual_head_center"
            else:
                raise ValueError(f"Unsupported final notehead source kind: {source_kind!r}")
            refined_x, refined_y = refine_notehead_center(
                image,
                bbox=bbox,
                spacing=measure.spacing,
            )
            refined_index = min(max(round(refined_x), 0), len(measure.shifts) - 1)
            refined_position = staff_position(
                refined_y,
                bottom_line_y=measure.base_lines[-1] + measure.shifts[refined_index],
                spacing=measure.spacing,
            )
            truth_pitch = str(head["pitch"])
            constant_pitch = pitch_for_staff_position(
                constant_position, key_flat_letters=key_flat_letters
            )
            tracked_pitch = pitch_for_staff_position(
                tracked_position, key_flat_letters=key_flat_letters
            )
            refined_pitch = pitch_for_staff_position(
                refined_position, key_flat_letters=key_flat_letters
            )
            results.append(
                {
                    "measure": measure.measure,
                    "order": int(head["order"]),
                    "x": float(head["center"]["x"]),
                    "y": y,
                    "local_shift_px": measure.shifts[x],
                    "refined_x": refined_x,
                    "refined_y": refined_y,
                    "center_refinement_search_region": search_region_source,
                    "truth_pitch": truth_pitch,
                    "constant_pitch": constant_pitch,
                    "tracked_pitch": tracked_pitch,
                    "refined_pitch": refined_pitch,
                    "constant_diatonic_match": _diatonic_pitch(constant_pitch)
                    == _diatonic_pitch(truth_pitch),
                    "tracked_diatonic_match": _diatonic_pitch(tracked_pitch)
                    == _diatonic_pitch(truth_pitch),
                    "constant_key_aware_match": constant_pitch == truth_pitch,
                    "tracked_key_aware_match": tracked_pitch == truth_pitch,
                    "refined_diatonic_match": _diatonic_pitch(refined_pitch)
                    == _diatonic_pitch(truth_pitch),
                    "refined_key_aware_match": refined_pitch == truth_pitch,
                }
            )

    count = len(results)
    summary = {
        "heads": count,
        "constant_diatonic_accuracy": _accuracy(results, "constant_diatonic_match"),
        "tracked_diatonic_accuracy": _accuracy(results, "tracked_diatonic_match"),
        "constant_key_aware_accuracy": _accuracy(results, "constant_key_aware_match"),
        "tracked_key_aware_accuracy": _accuracy(results, "tracked_key_aware_match"),
        "refined_diatonic_accuracy": _accuracy(results, "refined_diatonic_match"),
        "refined_key_aware_accuracy": _accuracy(results, "refined_key_aware_match"),
        "tracked_diatonic_matches": sum(
            1 for result in results if result["tracked_diatonic_match"]
        ),
        "gate": {
            "required_refined_diatonic_accuracy": 0.95,
            "passed": _accuracy(results, "refined_diatonic_match") >= 0.95,
        },
    }
    return {
        "schema_version": 2,
        "kind": "local_staff_tracking_pitch_spike",
        "provenance": {
            "staff_trajectory": {
                "image_only": True,
                "computed_before_review_fixture_load": True,
            },
            "center_refinement": {
                "image_only": False,
                "conditional_on_human_confirmed_notehead": True,
                "candidate_search_region": "human-confirmed candidate bbox",
                "manual_search_region": "box centered on human-reviewed manual head",
                "uses_ground_truth_pitch": False,
            },
            "pitch_evaluation": {
                "uses_human_reviewed_upstream_heads_and_pitches": True,
            },
        },
        "evaluation_fixture_dir": str(REVIEW_FIXTURE_DIR.relative_to(REPO_ROOT)),
        "summary": summary,
        "results": results,
    }


def staff_position(y: float, *, bottom_line_y: float, spacing: float) -> int:
    return round((bottom_line_y - y) / (spacing / 2))


def refine_notehead_center(
    image: Image.Image,
    *,
    bbox: Mapping[str, Any],
    spacing: float,
) -> tuple[float, float]:
    """Find the most compact oval-like ink center inside a proposal box."""
    gray = image.convert("L")
    width, height = gray.size
    left = max(0, math.floor(float(bbox["left"])))
    top = max(0, math.floor(float(bbox["top"])))
    right = min(width - 1, math.ceil(float(bbox["right"])))
    bottom = min(height - 1, math.ceil(float(bbox["bottom"])))
    if left > right or top > bottom:
        raise ValueError(f"Invalid candidate bbox: {bbox}")
    threshold = _ink_threshold(gray)
    radius_x = max(3, round(spacing * 0.34))
    radius_y = max(2, round(spacing * 0.22))
    original_x = (left + right) / 2
    original_y = (top + bottom) / 2
    best: tuple[float, float, float] | None = None
    for center_y in range(top, bottom + 1):
        for center_x in range(left, right + 1):
            score = _oval_ink_score(
                gray,
                center_x=center_x,
                center_y=center_y,
                radius_x=radius_x,
                radius_y=radius_y,
                threshold=threshold,
            )
            distance = math.hypot(center_x - original_x, center_y - original_y)
            candidate = (score - 0.015 * distance, float(center_x), float(center_y))
            if best is None or candidate > best:
                best = candidate
    assert best is not None
    return best[1], best[2]


def _oval_ink_score(
    image: Image.Image,
    *,
    center_x: int,
    center_y: int,
    radius_x: int,
    radius_y: int,
    threshold: int,
) -> float:
    row_counts: dict[int, int] = {}
    column_counts: dict[int, int] = {}
    total = 0
    width, height = image.size
    for y in range(max(0, center_y - radius_y), min(height, center_y + radius_y + 1)):
        for x in range(max(0, center_x - radius_x), min(width, center_x + radius_x + 1)):
            normalized = ((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2
            if normalized > 1.0 or image.getpixel((x, y)) > threshold:
                continue
            total += 1
            row_counts[y] = row_counts.get(y, 0) + 1
            column_counts[x] = column_counts.get(x, 0) + 1
    if not total:
        return 0.0
    max_row = max(row_counts.values(), default=0)
    max_column = max(column_counts.values(), default=0)
    occupied_rows = len(row_counts)
    occupied_columns = len(column_counts)
    return total - 0.75 * max_row - 0.75 * max_column + 0.15 * min(occupied_rows, occupied_columns)


def pitch_for_staff_position(position: int, *, key_flat_letters: set[str]) -> str:
    e4_index = 4 * 7 + DIATONIC_LETTERS.index("E")
    absolute = e4_index + position
    octave, letter_index = divmod(absolute, 7)
    letter = DIATONIC_LETTERS[letter_index]
    accidental = "b" if letter in key_flat_letters else ""
    return f"{letter}{accidental}{octave}"


def write_report(
    report: dict[str, Any], tracked: Sequence[TrackedMeasure], output_dir: Path
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_markdown(report, output_dir / "report.md")
    for measure in tracked:
        _write_overlay(measure, output_dir / f"measure_{measure.measure:03d}_staff_track.png")
    return json_path


def _write_overlay(measure: TrackedMeasure, path: Path) -> None:
    with Image.open(measure.image_path) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = ("#1677ff", "#00a65a", "#9c27b0", "#e67e22", "#d62728")
    for line_index, (base_line, color) in enumerate(zip(measure.base_lines, colors, strict=True)):
        points = [
            (x, base_line + measure.shifts[x])
            for x in range(0, image.width, max(1, image.width // 300))
        ]
        draw.line(points, fill=color, width=1)
        draw.line([(0, base_line), (image.width - 1, base_line)], fill="#999999", width=1)
        if line_index == 4:
            draw.text((3, max(0, round(base_line) - 14)), "gray=constant color=tracked", fill=color)
    image.save(path)


def _write_markdown(report: Mapping[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# Local Staff Tracking Spike",
        "",
        ("Staff trajectories are image-only and generated before any review label is " "loaded."),
        (
            "Center refinement is human-conditioned: candidate-backed heads use the "
            "human-confirmed candidate bbox, while manual heads use a search box centered "
            "on the reviewed head."
        ),
        "",
        "| Metric | Constant | Tracked | Human-conditioned refined + tracked |",
        "| --- | ---: | ---: | ---: |",
        (
            "| Diatonic pitch accuracy | "
            f"{summary['constant_diatonic_accuracy']:.3f} | "
            f"{summary['tracked_diatonic_accuracy']:.3f} | "
            f"{summary['refined_diatonic_accuracy']:.3f} |"
        ),
        (
            "| Key-aware exact pitch accuracy | "
            f"{summary['constant_key_aware_accuracy']:.3f} | "
            f"{summary['tracked_key_aware_accuracy']:.3f} | "
            f"{summary['refined_key_aware_accuracy']:.3f} |"
        ),
        "",
        f"Gate passed: **{summary['gate']['passed']}**",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _median_spacing(lines: Sequence[float]) -> float:
    gaps = sorted(second - first for first, second in zip(lines, lines[1:], strict=False))
    if len(gaps) != 4 or gaps[0] <= 0:
        raise ValueError(f"Invalid staff lines: {lines}")
    return (gaps[1] + gaps[2]) / 2


def _ink_threshold(image: Image.Image) -> int:
    histogram = image.histogram()
    total = sum(histogram)
    target = max(1, round(total * 0.18))
    cumulative = 0
    for value, count in enumerate(histogram):
        cumulative += count
        if cumulative >= target:
            return min(210, max(80, value + 25))
    return 180


def _diatonic_pitch(pitch: str) -> tuple[str, int]:
    match = PITCH_RE.match(pitch)
    if not match:
        raise ValueError(f"Invalid pitch: {pitch!r}")
    letter, _, octave = match.groups()
    return letter, int(octave)


def _accuracy(rows: Iterable[Mapping[str, Any]], key: str) -> float:
    values = list(rows)
    return round(sum(1 for row in values if row[key]) / len(values), 6) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
