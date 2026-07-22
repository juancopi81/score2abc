"""Extract durable notehead GT fixtures from blue-circle annotation PNGs.

The blue annotations are the sole source of notehead centers. Canonical musical
GT is read only after extraction to validate the note count and attach ordered
pitch labels.

Examples:
    ./.venv/bin/python scripts/extract_vlm_notehead_ground_truth.py out \
        --slug jaime-llanos_12_aviador_pasillo_fulgencio-garcia \
        --system 1 --measure 1 --measure 2 --measure 4

    ./.venv/bin/python scripts/extract_vlm_notehead_ground_truth.py out \
        --slug jaime-llanos_12_aviador_pasillo_fulgencio-garcia \
        --system 1 --summary --measure 1 --measure 2 --measure 3 --measure 4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE_DIR = REPO_ROOT / "tests/fixtures/vlm_melody/notehead_ground_truth"
BLUE_MIN = 120
BLUE_RED_GAP = 40
BLUE_GREEN_GAP = 25
MIN_BLUE_COMPONENT_PIXELS = 20


@dataclass(frozen=True)
class BlueComponent:
    """One connected blue annotation mark in annotation-image coordinates."""

    bbox: tuple[int, int, int, int]
    pixel_count: int
    center: tuple[float, float]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    measures = args.measure or [1, 2, 3, 4]
    try:
        if args.summary:
            output_path = args.summary_output or _default_summary_path(
                args.out_dir, args.slug, args.system
            )
            summary = write_notehead_evaluation_summary(
                _artifact_paths(args.out_dir, args.slug, args.system, measures),
                output_path,
                slug=args.slug,
                system_index=args.system,
            )
            print(json.dumps(summary, indent=2))
        else:
            canonical_path = args.canonical_ground_truth or (
                REPO_ROOT / "dataset/ground_truth" / f"{args.slug}.json"
            )
            written = []
            for measure in measures:
                output_path = args.fixture_dir / _fixture_name(args.slug, args.system, measure)
                written.append(
                    write_notehead_ground_truth_fixture(
                        _annotation_path(args.out_dir, args.slug, args.system, measure),
                        _source_path(args.out_dir, args.slug, args.system, measure),
                        canonical_path,
                        _context_path(args.out_dir, args.slug, args.system, measure),
                        output_path,
                    )
                )
            for path in written:
                print(path)
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
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
        help="1-based system-local measure index; defaults to 1-4.",
    )
    parser.add_argument(
        "--canonical-ground-truth",
        type=Path,
        default=None,
        help="Canonical events JSON; defaults to dataset/ground_truth/<slug>.json.",
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=DEFAULT_FIXTURE_DIR,
        help="Directory for notehead GT fixture JSON files.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Aggregate existing staff-grid-density v2 evaluation artifacts instead of extracting.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Output path for --summary; defaults beside the per-measure artifacts.",
    )
    return parser


def extract_blue_annotation_components(
    image: Image.Image,
    *,
    min_pixels: int = MIN_BLUE_COMPONENT_PIXELS,
) -> list[BlueComponent]:
    """Return blue connected components in deterministic left-to-right order."""
    rgba = image.convert("RGBA")
    blue_pixels = {
        (x, y)
        for y in range(rgba.height)
        for x in range(rgba.width)
        if _is_blue(rgba.getpixel((x, y)))
    }
    components: list[BlueComponent] = []
    remaining = set(blue_pixels)
    while remaining:
        seed = min(remaining, key=lambda point: (point[1], point[0]))
        queue = deque([seed])
        remaining.remove(seed)
        points: list[tuple[int, int]] = []
        while queue:
            x, y = queue.popleft()
            points.append((x, y))
            for neighbor in _neighbors(x, y):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        if len(points) < min_pixels:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        components.append(
            BlueComponent(
                bbox=(min(xs), min(ys), max(xs) + 1, max(ys) + 1),
                pixel_count=len(points),
                center=(sum(xs) / len(points), sum(ys) / len(points)),
            )
        )
    return sorted(components, key=lambda component: (component.center[0], component.center[1]))


def write_notehead_ground_truth_fixture(
    annotation_path: Path,
    source_path: Path,
    canonical_ground_truth_path: Path,
    context_path: Path,
    output_path: Path,
) -> Path:
    """Extract, validate, and write one notehead human-GT fixture."""
    if annotation_path.name.endswith("_deprecated.png"):
        raise ValueError(f"Deprecated annotation image is not accepted: {annotation_path}")
    context = _load_json(context_path)
    canonical = _load_json(canonical_ground_truth_path)
    with Image.open(annotation_path) as annotation_image, Image.open(source_path) as source_image:
        components = extract_blue_annotation_components(annotation_image)
        expected_notes = _canonical_notes_for_measure(
            canonical,
            global_measure_index=int(context["global_measure_index"]),
        )
        if len(components) != len(expected_notes):
            raise ValueError(
                f"Blue annotation count {len(components)} does not match canonical note count "
                f"{len(expected_notes)} for {annotation_path}"
            )
        payload = _fixture_payload(
            annotation_path,
            source_path,
            canonical_ground_truth_path,
            context_path,
            context,
            source_image.size,
            annotation_image.size,
            components,
            expected_notes,
        )
    validate_notehead_ground_truth_fixture(payload, expected_count=len(expected_notes))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_path


def validate_notehead_ground_truth_fixture(
    payload: dict[str, Any],
    *,
    expected_count: int | None = None,
) -> None:
    """Validate the compact fixture contract used by notehead evaluation."""
    if payload.get("kind") != "vlm_melody_notehead_human_ground_truth":
        raise ValueError(f"Unexpected fixture kind: {payload.get('kind')!r}")
    noteheads = payload.get("noteheads")
    if not isinstance(noteheads, list):
        raise ValueError("Fixture noteheads must be a list")
    if payload.get("notehead_count") != len(noteheads):
        raise ValueError("Fixture notehead_count does not match noteheads length")
    if expected_count is not None and len(noteheads) != expected_count:
        raise ValueError(
            f"Fixture note count {len(noteheads)} does not match expected count {expected_count}"
        )
    for order, notehead in enumerate(noteheads, start=1):
        if notehead.get("order") != order or not notehead.get("id"):
            raise ValueError("Fixture noteheads must have contiguous ids and order")
        center = notehead.get("center")
        if not isinstance(center, dict) or not all(key in center for key in ("x", "y")):
            raise ValueError("Fixture notehead center must contain x and y")
        if not isinstance(notehead.get("pitch"), str) or not notehead["pitch"]:
            raise ValueError("Fixture notehead pitch must be a non-empty string")


def aggregate_notehead_evaluations(
    artifact_paths: Sequence[Path],
    *,
    slug: str,
    system_index: int,
) -> dict[str, Any]:
    """Build a concise micro/macro summary from evaluated v2 artifacts."""
    per_measure: list[dict[str, Any]] = []
    for path in artifact_paths:
        payload = _load_json(path)
        evaluation = payload.get("evaluation")
        if not isinstance(evaluation, dict):
            raise ValueError(f"Artifact has no evaluation: {path}")
        tp = int(evaluation["true_positives"])
        fp = int(evaluation["false_positives"])
        fn = int(evaluation["false_negatives"])
        per_measure.append(
            {
                "measure": int(payload["system_measure_index"]),
                "artifact": str(path),
                "candidate_count": int(evaluation["candidate_count"]),
                "gt_count": int(evaluation["gt_count"]),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": _ratio(tp, tp + fp),
                "recall": _ratio(tp, tp + fn),
                "f1": _f1(tp, fp, fn),
                "tolerance_px": evaluation["distance_tolerance_px"],
            }
        )
    per_measure.sort(key=lambda row: row["measure"])
    tp = sum(row["tp"] for row in per_measure)
    fp = sum(row["fp"] for row in per_measure)
    fn = sum(row["fn"] for row in per_measure)
    return {
        "schema_version": 1,
        "kind": "vlm_melody_notehead_human_ground_truth_evaluation",
        "slug": slug,
        "system_index": system_index,
        "detector": {
            "strategy": "staff-grid-density",
            "version": 2,
            "candidate_generation_uses_ground_truth": False,
        },
        "per_measure": per_measure,
        "aggregate": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "candidate_count": sum(row["candidate_count"] for row in per_measure),
            "gt_count": sum(row["gt_count"] for row in per_measure),
            "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn),
            "f1": _f1(tp, fp, fn),
            "mean_per_measure_recall": _mean(row["recall"] for row in per_measure),
            "mean_per_measure_f1": _mean(row["f1"] for row in per_measure),
        },
    }


def write_notehead_evaluation_summary(
    artifact_paths: Sequence[Path],
    output_path: Path,
    *,
    slug: str,
    system_index: int,
) -> dict[str, Any]:
    summary = aggregate_notehead_evaluations(
        artifact_paths,
        slug=slug,
        system_index=system_index,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _fixture_payload(
    annotation_path: Path,
    source_path: Path,
    canonical_ground_truth_path: Path,
    context_path: Path,
    context: dict[str, Any],
    source_size: tuple[int, int],
    annotation_size: tuple[int, int],
    components: Sequence[BlueComponent],
    canonical_notes: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    scale_x = source_size[0] / annotation_size[0]
    scale_y = source_size[1] / annotation_size[1]
    noteheads = []
    for index, (component, canonical_note) in enumerate(
        zip(components, canonical_notes, strict=True), start=1
    ):
        left, top, right, bottom = component.bbox
        mapped_bbox = {
            "left": round(left * scale_x),
            "top": round(top * scale_y),
            "right": round(right * scale_x),
            "bottom": round(bottom * scale_y),
        }
        width = mapped_bbox["right"] - mapped_bbox["left"]
        height = mapped_bbox["bottom"] - mapped_bbox["top"]
        radius_x = width / 2
        radius_y = height / 2
        noteheads.append(
            {
                "id": f"n{index:03d}",
                "order": index,
                "pitch": _pitch_name(
                    int(canonical_note["pitch_midi"]),
                    int(canonical_note.get("accidental", 0)),
                ),
                "center": {
                    "x": round(component.center[0] * scale_x, 2),
                    "y": round(component.center[1] * scale_y, 2),
                },
                "annotation_geometry": {
                    "bbox_px": mapped_bbox,
                    "bbox_area_px": width * height,
                    "radius_x_px": round(radius_x, 2),
                    "radius_y_px": round(radius_y, 2),
                    "approximate_radius_px": round((radius_x + radius_y) / 2, 2),
                    "approximate_circle_area_px": round(math.pi * radius_x * radius_y, 1),
                },
            }
        )
    pitches = [note["pitch"] for note in noteheads]
    mapping = (
        "same pixel coordinates"
        if source_size == annotation_size
        else "scaled to raw source dimensions"
    )
    return {
        "schema_version": 1,
        "kind": "vlm_melody_notehead_human_ground_truth",
        "slug": context["slug"],
        "system_index": int(context["system_index"]),
        "system_measure_index": int(context["system_measure_index"]),
        "source_image_path": _display_path(source_path),
        "annotation_image_path": _display_path(annotation_path),
        "coordinate_provenance": {
            "type": "human_annotation_png",
            "description": (
                "Centers come from connected blue-circle components in the human-labeled PNG; "
                "they are not generated from canonical musical GT."
            ),
            "coordinate_space": "source_image_path pixels, origin at top-left",
            "mapping": mapping,
            "notehead_order": f"left-to-right: {', '.join(pitches)}",
        },
        "musical_gt_provenance": {
            "type": "canonical_ground_truth_json",
            "source_path": _display_path(canonical_ground_truth_path),
            "source_measure_index": int(context["global_measure_index"]),
            "fields_used": ["measure", "pitch_midi", "accidental"],
            "description": (
                "Canonical GT supplies pitch labels and validates the expected note count only."
            ),
        },
        "context_path": _display_path(context_path),
        "notehead_count": len(noteheads),
        "noteheads": noteheads,
    }


def _canonical_notes_for_measure(
    canonical: dict[str, Any],
    *,
    global_measure_index: int,
) -> list[dict[str, Any]]:
    notes = canonical.get("notes")
    if not isinstance(notes, list):
        raise ValueError("Canonical GT has no notes list")
    return [
        note
        for note in notes
        if isinstance(note, dict) and int(note.get("measure", -1)) == global_measure_index
    ]


def _pitch_name(pitch_midi: int, accidental: int) -> str:
    natural_midi = pitch_midi - accidental
    natural_names = {0: "C", 2: "D", 4: "E", 5: "F", 7: "G", 9: "A", 11: "B"}
    step = natural_names.get(natural_midi % 12)
    if step is None:
        raise ValueError(f"Cannot spell MIDI pitch {pitch_midi} with accidental {accidental}")
    accidental_symbol = {0: "", -1: "b", 1: "#", -2: "bb", 2: "##"}.get(accidental)
    if accidental_symbol is None:
        raise ValueError(f"Unsupported canonical accidental: {accidental}")
    octave = natural_midi // 12 - 1
    return f"{step}{accidental_symbol}{octave}"


def _is_blue(pixel: tuple[int, int, int, int]) -> bool:
    red, green, blue, alpha = pixel
    return (
        alpha > 0
        and blue >= BLUE_MIN
        and blue - red >= BLUE_RED_GAP
        and blue - green >= BLUE_GREEN_GAP
    )


def _neighbors(x: int, y: int) -> Iterable[tuple[int, int]]:
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                yield x + dx, y + dy


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return round(2 * precision * recall / (precision + recall), 6) if precision + recall else 0.0


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(sum(values) / len(values), 6) if values else 0.0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _system_dir(out_dir: Path, slug: str, system_index: int) -> Path:
    return out_dir / slug / "vlm_melody_inputs" / f"system_{system_index:03d}"


def _source_path(out_dir: Path, slug: str, system_index: int, measure: int) -> Path:
    return _system_dir(out_dir, slug, system_index) / f"measure_{measure:03d}_raw.png"


def _annotation_path(out_dir: Path, slug: str, system_index: int, measure: int) -> Path:
    source_path = _source_path(out_dir, slug, system_index, measure)
    return source_path.with_name(f"{source_path.stem}_notehead_gt{source_path.suffix}")


def _context_path(out_dir: Path, slug: str, system_index: int, measure: int) -> Path:
    return _system_dir(out_dir, slug, system_index) / f"measure_{measure:03d}_context.json"


def _artifact_paths(
    out_dir: Path,
    slug: str,
    system_index: int,
    measures: Sequence[int],
) -> list[Path]:
    system_dir = _system_dir(out_dir, slug, system_index)
    return [
        system_dir / f"measure_{measure:03d}_raw_notehead_candidates_staff-grid-density_v2.json"
        for measure in measures
    ]


def _default_summary_path(out_dir: Path, slug: str, system_index: int) -> Path:
    return _system_dir(out_dir, slug, system_index) / (
        "notehead_candidates_staff-grid-density_v2_summary.json"
    )


def _fixture_name(slug: str, system_index: int, measure: int) -> str:
    return f"{slug}_system_{system_index:03d}_measure_{measure:03d}.json"


if __name__ == "__main__":
    raise SystemExit(main())
