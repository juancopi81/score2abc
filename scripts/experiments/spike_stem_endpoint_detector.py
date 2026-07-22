"""GT-blind stem-endpoint notehead localization experiment.

This bounded spike fits fixed visual parameters only against the promoted
system-1 measures 1-4 coordinate reviews. It freezes validation predictions
for systems 7 and 8 before opening validation truth. Heldout system-3
predictions are generated only when the preregistered validation gate passes,
and are sealed before heldout truth is read.

The detector recognizes note locations and natural staff pitches only. Rhythm,
onsets, durations, rests, and accidentals are explicitly out of scope.

Example:
    uv run python scripts/experiments/spike_stem_endpoint_detector.py out
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from score2abc.utils.imaging import estimate_ink_threshold  # noqa: E402

DEFAULT_SLUG = "jaime-llanos_12_aviador_pasillo_fulgencio-garcia"
DEFAULT_REVIEW_DIR = REPO_ROOT / "tests/fixtures/vlm_melody/notehead_reviews"
OUTPUT_SUBDIR = "stem_endpoint_detector"
DEVELOPMENT_TARGETS = tuple((1, measure) for measure in range(1, 5))
VALIDATION_SYSTEMS = (7, 8)
HELDOUT_SYSTEM = 3
NATURAL_PITCH_CLASSES = (0, 2, 4, 5, 7, 9, 11)

# Preregistered before validation truth is opened. Both conditions must pass.
VALIDATION_GATE = {
    "minimum_pitch_only_note_f1": 0.70,
    "minimum_exact_note_count_rate": 0.50,
}


@dataclass(frozen=True, order=True)
class DetectorConfig:
    """Fixed, staff-spacing-normalized visual parameters."""

    min_run_spaces: float
    max_run_spaces: float
    max_gap_spaces: float
    endpoint_window_spaces: float
    min_endpoint_score: float
    dedupe_x_spaces: float

    @property
    def key(self) -> str:
        return (
            f"run{self.min_run_spaces:.2f}-{self.max_run_spaces:.2f}_"
            f"gap{self.max_gap_spaces:.2f}_end{self.endpoint_window_spaces:.2f}_"
            f"score{self.min_endpoint_score:.2f}_nms{self.dedupe_x_spaces:.2f}"
        )


@dataclass(frozen=True)
class StemRun:
    x: float
    top: int
    bottom: int
    width: int
    ink_fraction: float

    @property
    def length(self) -> int:
        return self.bottom - self.top + 1


@dataclass(frozen=True)
class EndpointCandidate:
    x: float
    y: float
    score: float
    stem_x: float
    stem_top: int
    stem_bottom: int
    endpoint: str
    pitch_midi: int
    staff_position: int


@dataclass(frozen=True)
class PreparedRequest:
    request: dict[str, Any]
    image_path: Path
    image: Image.Image
    staff_lines: tuple[float, ...]
    spacing: float
    threshold: int


class SealedTruthGate:
    """Enforce prediction-before-truth and validation-before-heldout ordering."""

    def __init__(self) -> None:
        self.seals: dict[str, dict[str, Any]] = {}
        self.validation_result: dict[str, Any] | None = None
        self.access_log: list[dict[str, Any]] = []

    def seal_predictions(self, split: str, path: Path) -> dict[str, Any]:
        if split not in {"validation", "heldout"}:
            raise ValueError(f"Cannot seal unsupported split: {split}")
        if split == "heldout" and not self.validation_passed:
            raise RuntimeError("Heldout predictions require a passing validation gate")
        seal = {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
        self.seals[split] = seal
        return seal

    @property
    def validation_passed(self) -> bool:
        return bool(self.validation_result and self.validation_result.get("passed"))

    def record_validation_result(self, result: Mapping[str, Any]) -> None:
        if "validation" not in self.seals:
            raise RuntimeError("Validation metrics require sealed validation predictions")
        self.validation_result = dict(result)

    def read_truth(self, split: str, path: Path) -> list[dict[str, Any]]:
        if split not in self.seals:
            raise RuntimeError(f"{split.title()} truth requires sealed {split} predictions")
        if split == "heldout" and not self.validation_passed:
            raise RuntimeError("Heldout truth requires a passing validation gate")
        rows = _read_jsonl(path)
        self.access_log.append(
            {
                "split": split,
                "truth_path": str(path),
                "after_prediction_sha256": self.seals[split]["sha256"],
            }
        )
        return rows


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    benchmark_dir = args.out_dir / args.slug / "vlm_melody_event_benchmark"
    output_dir = benchmark_dir / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        development_requests = _selected_requests(
            _read_jsonl(benchmark_dir / "development/requests.jsonl"),
            DEVELOPMENT_TARGETS,
        )
        prepared_development = [
            prepare_request(request, out_dir=args.out_dir) for request in development_requests
        ]
        reviews = load_development_reviews(args.review_dir, args.slug)
        config, fitting = fit_config(prepared_development, reviews)
        development_predictions = [predict(prepared, config) for prepared in prepared_development]
        _write_jsonl(output_dir / "development_predictions.jsonl", development_predictions)
        _write_overlays(
            prepared_development,
            development_predictions,
            output_dir / "development_overlays",
            truth_by_identity=reviews,
        )

        validation_requests = _read_jsonl(benchmark_dir / "validation/requests.jsonl")
        _require_systems(validation_requests, VALIDATION_SYSTEMS, "validation")
        prepared_validation = [
            prepare_request(request, out_dir=args.out_dir) for request in validation_requests
        ]
        validation_predictions = [predict(prepared, config) for prepared in prepared_validation]
        validation_path = output_dir / "validation_predictions.sealed.jsonl"
        _write_jsonl(validation_path, validation_predictions)
        _write_overlays(
            prepared_validation,
            validation_predictions,
            output_dir / "validation_overlays",
        )

        truth_gate = SealedTruthGate()
        validation_seal = truth_gate.seal_predictions("validation", validation_path)
        validation_truth = truth_gate.read_truth(
            "validation", benchmark_dir / "validation/truth.jsonl"
        )
        validation_metrics = evaluate_pitch_only(validation_truth, validation_predictions)
        validation_gate = apply_validation_gate(validation_metrics)
        truth_gate.record_validation_result(validation_gate)

        heldout_metrics: dict[str, Any] | None = None
        heldout_seal: dict[str, Any] | None = None
        heldout_status = "skipped_validation_gate_failed"
        if validation_gate["passed"]:
            heldout_requests = _read_jsonl(benchmark_dir / "heldout/requests.jsonl")
            _require_systems(heldout_requests, (HELDOUT_SYSTEM,), "heldout")
            prepared_heldout = [
                prepare_request(request, out_dir=args.out_dir) for request in heldout_requests
            ]
            heldout_predictions = [predict(prepared, config) for prepared in prepared_heldout]
            heldout_path = output_dir / "heldout_predictions.sealed.jsonl"
            _write_jsonl(heldout_path, heldout_predictions)
            _write_overlays(
                prepared_heldout,
                heldout_predictions,
                output_dir / "heldout_overlays",
            )
            heldout_seal = truth_gate.seal_predictions("heldout", heldout_path)
            heldout_truth = truth_gate.read_truth("heldout", benchmark_dir / "heldout/truth.jsonl")
            heldout_metrics = evaluate_pitch_only(heldout_truth, heldout_predictions)
            heldout_status = "evaluated_once"

        report = {
            "schema_version": 1,
            "kind": "gt_blind_stem_endpoint_notehead_experiment",
            "scope": {
                "localization": "vertical_stem_endpoint_candidates",
                "pitch": "natural_staff_pitch_only",
                "accidentals": "out_of_scope",
                "rhythm": "out_of_scope",
                "onsets": "out_of_scope",
                "durations": "out_of_scope",
                "rests": "out_of_scope",
            },
            "development_fitting": fitting,
            "selected_config": asdict(config),
            "validation": {
                "prediction_seal": validation_seal,
                "metrics": validation_metrics,
                "gate": validation_gate,
            },
            "heldout": {
                "status": heldout_status,
                "prediction_seal": heldout_seal,
                "metrics": heldout_metrics,
            },
            "truth_access_log": truth_gate.access_log,
        }
        _write_json(output_dir / "report.json", report)
        (output_dir / "report.md").write_text(_markdown_report(report), encoding="utf-8")
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(output_dir / "report.json")
    print(output_dir / "report.md")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", type=Path, default=Path("out"))
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    return parser


def parameter_grid() -> tuple[DetectorConfig, ...]:
    """Small preregistered search space fitted only on four coordinate reviews."""
    return tuple(
        DetectorConfig(min_run, max_run, max_gap, endpoint_window, min_score, dedupe_x)
        for min_run in (0.80, 1.05, 1.30)
        for max_run in (3.25, 4.25)
        for max_gap in (0.12, 0.20)
        for endpoint_window in (0.34, 0.46)
        for min_score in (0.12, 0.20, 0.28)
        for dedupe_x in (0.30, 0.46)
    )


def prepare_request(request: Mapping[str, Any], *, out_dir: Path) -> PreparedRequest:
    image_record = request["images"]["raw"]
    image_path = out_dir / str(image_record["path_relative_to_out"])
    if _sha256(image_path) != image_record["sha256"]:
        raise ValueError(f"Image hash mismatch: {image_path}")
    with Image.open(image_path) as opened:
        image = opened.convert("L")
    lines = tuple(float(value) for value in request["staff_geometry"]["raw_staff_lines_y_px"])
    if len(lines) != 5 or list(lines) != sorted(lines):
        raise ValueError(f"Expected ordered five-line staff geometry: {lines}")
    spacing = statistics.median(b - a for a, b in zip(lines, lines[1:], strict=False))
    if spacing <= 0:
        raise ValueError(f"Invalid staff spacing: {spacing}")
    return PreparedRequest(
        request=dict(request),
        image_path=image_path,
        image=image,
        staff_lines=lines,
        spacing=spacing,
        threshold=estimate_ink_threshold(image),
    )


def build_ink_mask(
    image: Image.Image,
    *,
    threshold: int,
    staff_lines: Sequence[float],
    spacing: float,
) -> list[list[bool]]:
    """Threshold ink and suppress line-only pixels while preserving crossings."""
    gray = ImageOps.grayscale(image)
    width, height = gray.size
    pixels = gray.load()
    raw = [[pixels[x, y] < threshold for x in range(width)] for y in range(height)]
    mask = [row[:] for row in raw]
    line_radius = max(1, round(spacing * 0.055))
    horizontal_radius = max(4, round(spacing * 0.40))
    vertical_radius = max(2, round(spacing * 0.18))
    for line in staff_lines:
        center_y = round(line)
        for y in range(max(0, center_y - line_radius), min(height, center_y + line_radius + 1)):
            for x in range(width):
                if not raw[y][x]:
                    continue
                horizontal = sum(
                    raw[y][sample_x]
                    for sample_x in range(
                        max(0, x - horizontal_radius), min(width, x + horizontal_radius + 1)
                    )
                )
                vertical = sum(
                    raw[sample_y][x]
                    for sample_y in range(
                        max(0, y - vertical_radius), min(height, y + vertical_radius + 1)
                    )
                )
                if horizontal >= horizontal_radius and vertical <= vertical_radius + 1:
                    mask[y][x] = False
    return mask


def find_stem_runs(
    mask: Sequence[Sequence[bool]],
    *,
    staff_lines: Sequence[float],
    spacing: float,
    config: DetectorConfig,
) -> list[StemRun]:
    """Find and merge near-vertical runs, allowing short staff-crossing gaps."""
    height = len(mask)
    width = len(mask[0]) if height else 0
    if not width:
        return []
    y_min = max(0, math.floor(staff_lines[0] - spacing * 2.1))
    y_max = min(height - 1, math.ceil(staff_lines[-1] + spacing * 2.1))
    max_gap = max(1, round(spacing * config.max_gap_spaces))
    min_length = max(3, round(spacing * config.min_run_spaces))
    max_length = max(min_length, round(spacing * config.max_run_spaces))
    per_column: list[StemRun] = []
    for x in range(1, width - 1):
        row_ink = [
            sum(mask[y][sample_x] for sample_x in range(x - 1, x + 2)) >= 1
            for y in range(y_min, y_max + 1)
        ]
        start: int | None = None
        last_ink: int | None = None
        ink_rows = 0
        for offset, has_ink in enumerate((*row_ink, *(False for _ in range(max_gap + 1)))):
            y = y_min + offset
            if has_ink:
                if start is None:
                    start = y
                    ink_rows = 0
                last_ink = y
                ink_rows += 1
                continue
            if start is None or last_ink is None or y - last_ink <= max_gap:
                continue
            length = last_ink - start + 1
            fraction = ink_rows / length
            if min_length <= length <= max_length and fraction >= 0.72:
                per_column.append(StemRun(float(x), start, last_ink, 1, fraction))
            start = None
            last_ink = None
            ink_rows = 0

    merged: list[list[StemRun]] = []
    for run in per_column:
        if (
            merged
            and run.x - merged[-1][-1].x <= 1.01
            and _vertical_overlap(run, merged[-1][-1]) >= 0.65
            and abs(run.top - merged[-1][-1].top) <= spacing * 0.35
            and abs(run.bottom - merged[-1][-1].bottom) <= spacing * 0.35
        ):
            merged[-1].append(run)
        else:
            merged.append([run])
    stems = []
    max_stem_width = max(2, round(spacing * 0.40))
    for group in merged:
        if len(group) > max_stem_width:
            continue
        stems.append(
            StemRun(
                x=statistics.mean(run.x for run in group),
                top=round(statistics.median(run.top for run in group)),
                bottom=round(statistics.median(run.bottom for run in group)),
                width=len(group),
                ink_fraction=statistics.mean(run.ink_fraction for run in group),
            )
        )
    return stems


def infer_endpoint_candidates(
    mask: Sequence[Sequence[bool]],
    stems: Sequence[StemRun],
    *,
    staff_lines: Sequence[float],
    spacing: float,
    config: DetectorConfig,
) -> list[EndpointCandidate]:
    """Choose the endpoint with strongest compact lateral attachment per stem."""
    candidates: list[EndpointCandidate] = []
    radius_y = max(2, round(spacing * config.endpoint_window_spaces))
    for stem in stems:
        endpoint_options = []
        for endpoint, endpoint_y, inward in (
            ("top", stem.top, 1),
            ("bottom", stem.bottom, -1),
        ):
            best: tuple[float, float, float] | None = None
            for y in range(endpoint_y - radius_y, endpoint_y + radius_y + 1):
                for side in (-1, 1):
                    x = stem.x + side * spacing * 0.27
                    score = endpoint_attachment_score(
                        mask,
                        center_x=x,
                        center_y=y,
                        stem_x=stem.x,
                        endpoint_y=endpoint_y,
                        inward=inward,
                        spacing=spacing,
                    )
                    option = (score, -abs(y - endpoint_y), -x)
                    if best is None or option > (best[0], best[1], best[2]):
                        best = (score, float(-abs(y - endpoint_y)), float(-x))
                        best_x = x
                        best_y = float(y)
            assert best is not None
            endpoint_options.append((best[0], endpoint, best_x, best_y))
        score, endpoint, x, y = max(
            endpoint_options,
            key=lambda row: (row[0], row[1] == "bottom", -row[2]),
        )
        if score < config.min_endpoint_score:
            continue
        staff_position = round((staff_lines[-1] - y) / (spacing / 2))
        candidates.append(
            EndpointCandidate(
                x=x,
                y=y,
                score=score + stem.ink_fraction * 0.05,
                stem_x=stem.x,
                stem_top=stem.top,
                stem_bottom=stem.bottom,
                endpoint=endpoint,
                pitch_midi=natural_midi_for_staff_position(staff_position),
                staff_position=staff_position,
            )
        )
    return candidates


def endpoint_attachment_score(
    mask: Sequence[Sequence[bool]],
    *,
    center_x: float,
    center_y: float,
    stem_x: float,
    endpoint_y: int,
    inward: int,
    spacing: float,
) -> float:
    """Score compact ink beside a stem end, discounting stem-only columns."""
    height = len(mask)
    width = len(mask[0]) if height else 0
    radius_x = max(3, round(spacing * 0.38))
    radius_y = max(2, round(spacing * 0.24))
    left = max(0, round(center_x) - radius_x)
    right = min(width - 1, round(center_x) + radius_x)
    top = max(0, round(center_y) - radius_y)
    bottom = min(height - 1, round(center_y) + radius_y)
    if left > right or top > bottom:
        return 0.0
    total = 0
    occupied_rows = set()
    occupied_columns = set()
    lateral = 0
    for y in range(top, bottom + 1):
        for x in range(left, right + 1):
            if not mask[y][x]:
                continue
            total += 1
            occupied_rows.add(y)
            occupied_columns.add(x)
            if abs(x - stem_x) >= max(2, spacing * 0.10):
                lateral += 1
    area = max(1, (right - left + 1) * (bottom - top + 1))
    density = total / area
    lateral_fraction = lateral / max(1, total)
    compactness = min(len(occupied_rows), len(occupied_columns)) / max(1, radius_y * 2 + 1)
    endpoint_distance = abs(center_y - endpoint_y) / max(1.0, spacing)
    inward_probe_y = round(center_y + inward * spacing * 0.28)
    inward_connection = 0.0
    if 0 <= inward_probe_y < height:
        inward_connection = (
            sum(
                mask[inward_probe_y][x]
                for x in range(max(0, round(stem_x) - 1), min(width, round(stem_x) + 2))
            )
            / 3
        )
    return (
        0.42 * min(1.0, density / 0.38)
        + 0.34 * lateral_fraction
        + 0.18 * min(1.0, compactness)
        + 0.06 * inward_connection
        - 0.12 * endpoint_distance
    )


def dedupe_candidates(
    candidates: Sequence[EndpointCandidate],
    *,
    spacing: float,
    x_tolerance_spaces: float,
) -> list[EndpointCandidate]:
    """Deterministic score-first NMS for duplicate columns and split stems."""
    retained: list[EndpointCandidate] = []
    x_tolerance = max(1.0, spacing * x_tolerance_spaces)
    y_tolerance = max(1.0, spacing * 0.62)
    for candidate in sorted(candidates, key=lambda row: (-row.score, row.x, row.y, row.endpoint)):
        if any(
            abs(candidate.x - previous.x) <= x_tolerance
            and abs(candidate.y - previous.y) <= y_tolerance
            for previous in retained
        ):
            continue
        retained.append(candidate)
    return sorted(retained, key=lambda row: (row.x, row.y, -row.score))


def detect(prepared: PreparedRequest, config: DetectorConfig) -> list[EndpointCandidate]:
    mask = build_ink_mask(
        prepared.image,
        threshold=prepared.threshold,
        staff_lines=prepared.staff_lines,
        spacing=prepared.spacing,
    )
    stems = find_stem_runs(
        mask,
        staff_lines=prepared.staff_lines,
        spacing=prepared.spacing,
        config=config,
    )
    endpoints = infer_endpoint_candidates(
        mask,
        stems,
        staff_lines=prepared.staff_lines,
        spacing=prepared.spacing,
        config=config,
    )
    return dedupe_candidates(
        endpoints,
        spacing=prepared.spacing,
        x_tolerance_spaces=config.dedupe_x_spaces,
    )


def predict(prepared: PreparedRequest, config: DetectorConfig) -> dict[str, Any]:
    candidates = detect(prepared, config)
    return {
        "schema_version": 1,
        "identity": dict(prepared.request["identity"]),
        "method": "vertical_stem_endpoint",
        "config_key": config.key,
        "natural_pitch_only": True,
        "predicted_note_count": len(candidates),
        "ordered_pitches": [candidate.pitch_midi for candidate in candidates],
        "candidates": [
            {
                "id": f"stem{index:03d}",
                "center": {"x": round(candidate.x, 3), "y": round(candidate.y, 3)},
                "score": round(candidate.score, 6),
                "pitch_midi": candidate.pitch_midi,
                "staff_position": candidate.staff_position,
                "stem": {
                    "x": round(candidate.stem_x, 3),
                    "top": candidate.stem_top,
                    "bottom": candidate.stem_bottom,
                    "endpoint": candidate.endpoint,
                },
            }
            for index, candidate in enumerate(candidates, start=1)
        ],
        "onsets": "out_of_scope",
        "durations": "out_of_scope",
        "rhythm": "out_of_scope",
    }


def fit_config(
    prepared: Sequence[PreparedRequest],
    reviews: Mapping[tuple[Any, ...], Sequence[Mapping[str, Any]]],
) -> tuple[DetectorConfig, dict[str, Any]]:
    """Select fixed visual parameters using only S1M1-4 coordinate labels."""
    rows = []
    for config in parameter_grid():
        totals = Counter()
        exact_counts = 0
        for item in prepared:
            identity = _identity_key(item.request["identity"])
            truth = reviews[identity]
            candidates = detect(item, config)
            metrics = coordinate_metrics(candidates, truth, spacing=item.spacing)
            totals.update({key: metrics[key] for key in ("tp", "fp", "fn")})
            exact_counts += int(len(candidates) == len(truth))
        precision = _ratio(totals["tp"], totals["tp"] + totals["fp"])
        recall = _ratio(totals["tp"], totals["tp"] + totals["fn"])
        f1 = _f1(totals["tp"], totals["fp"], totals["fn"])
        rows.append(
            {
                "config": config,
                "tp": totals["tp"],
                "fp": totals["fp"],
                "fn": totals["fn"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "exact_count_rate": _ratio(exact_counts, len(prepared)),
            }
        )
    best = max(
        rows,
        key=lambda row: (
            row["f1"],
            row["exact_count_rate"],
            row["recall"],
            -row["fp"],
            row["config"].key,
        ),
    )
    return best["config"], {
        "allowed_coordinate_labels": "promoted S1M1-4 notehead reviews only",
        "targets": len(prepared),
        "parameterizations": len(rows),
        "selection_metric": "coordinate F1, exact count rate, recall, fewer false positives",
        "coordinate_tolerance": {"x_staff_spaces": 0.62, "y_staff_spaces": 0.50},
        "selected": {key: value for key, value in best.items() if key != "config"},
        "selected_config_key": best["config"].key,
    }


def coordinate_metrics(
    candidates: Sequence[EndpointCandidate],
    truth: Sequence[Mapping[str, Any]],
    *,
    spacing: float,
) -> dict[str, int]:
    pairs = []
    for candidate_index, candidate in enumerate(candidates):
        for truth_index, notehead in enumerate(truth):
            center = notehead["center"]
            dx = abs(candidate.x - float(center["x"])) / spacing
            dy = abs(candidate.y - float(center["y"])) / spacing
            if dx <= 0.62 and dy <= 0.50:
                pairs.append((math.hypot(dx, dy), candidate_index, truth_index))
    matched_candidates = set()
    matched_truth = set()
    for _, candidate_index, truth_index in sorted(pairs):
        if candidate_index in matched_candidates or truth_index in matched_truth:
            continue
        matched_candidates.add(candidate_index)
        matched_truth.add(truth_index)
    tp = len(matched_candidates)
    return {"tp": tp, "fp": len(candidates) - tp, "fn": len(truth) - tp}


def load_development_reviews(
    review_dir: Path, slug: str
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    reviews = {}
    for system, measure in DEVELOPMENT_TARGETS:
        path = review_dir / f"{slug}_system_{system:03d}_measure_{measure:03d}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        identity = payload["identity"]
        if (int(identity["system_index"]), int(identity["system_measure_index"])) != (
            system,
            measure,
        ):
            raise ValueError(f"Review identity mismatch: {path}")
        reviews[_identity_key(identity)] = list(payload["final_noteheads"])
    return reviews


def natural_midi_for_staff_position(position: int) -> int:
    """Map diatonic steps from treble-clef bottom-line E4 to natural MIDI."""
    e4_diatonic = 4 * 7 + 2
    absolute = e4_diatonic + position
    octave, letter_index = divmod(absolute, 7)
    return (octave + 1) * 12 + NATURAL_PITCH_CLASSES[letter_index]


def naturalize_truth_pitch(note: Mapping[str, Any]) -> int:
    """Remove only an explicitly encoded accidental from canonical truth."""
    return int(note["pitch_midi"]) - int(note.get("accidental") or 0)


def evaluate_pitch_only(
    truth_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    predictions = {_identity_key(row["identity"]): row for row in prediction_rows}
    if len(predictions) != len(prediction_rows):
        raise ValueError("Duplicate prediction identity")
    results = []
    totals = Counter()
    for truth in truth_rows:
        key = _identity_key(truth["identity"])
        prediction = predictions.get(key)
        predicted = list(prediction["ordered_pitches"]) if prediction else []
        expected = [naturalize_truth_pitch(note) for note in truth.get("notes") or []]
        compared = max(len(predicted), len(expected))
        ordered_correct = sum(
            actual == wanted for actual, wanted in zip(predicted, expected, strict=False)
        )
        predicted_counts = Counter(predicted)
        expected_counts = Counter(expected)
        tp = sum((predicted_counts & expected_counts).values())
        fp = len(predicted) - tp
        fn = len(expected) - tp
        totals.update(
            {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "ordered_correct": ordered_correct,
                "ordered_compared": compared,
                "exact_count": int(len(predicted) == len(expected)),
            }
        )
        results.append(
            {
                "identity": dict(truth["identity"]),
                "predicted_note_count": len(predicted),
                "truth_note_count": len(expected),
                "exact_note_count": len(predicted) == len(expected),
                "predicted_ordered_natural_pitches": predicted,
                "truth_ordered_natural_pitches": expected,
                "ordered_pitch_correct": ordered_correct,
                "ordered_pitch_compared": compared,
                "pitch_only_tp": tp,
                "pitch_only_fp": fp,
                "pitch_only_fn": fn,
            }
        )
    return {
        "scope": {
            "ordered_natural_pitch": "evaluated",
            "note_count": "evaluated",
            "pitch_only_event_compatibility": "multiset natural-pitch precision/recall/F1",
            "onset": "out_of_scope",
            "duration": "out_of_scope",
            "accidental": "out_of_scope",
        },
        "summary": {
            "targets": len(results),
            "predicted_note_count": totals["tp"] + totals["fp"],
            "truth_note_count": totals["tp"] + totals["fn"],
            "exact_note_count_rate": _ratio(totals["exact_count"], len(results)),
            "ordered_natural_pitch_accuracy": _ratio(
                totals["ordered_correct"], totals["ordered_compared"]
            ),
            "pitch_only_note_precision": _ratio(totals["tp"], totals["tp"] + totals["fp"]),
            "pitch_only_note_recall": _ratio(totals["tp"], totals["tp"] + totals["fn"]),
            "pitch_only_note_f1": _f1(totals["tp"], totals["fp"], totals["fn"]),
        },
        "results": results,
    }


def apply_validation_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    summary = metrics["summary"]
    f1 = float(summary["pitch_only_note_f1"])
    count_rate = float(summary["exact_note_count_rate"])
    passed = (
        f1 >= VALIDATION_GATE["minimum_pitch_only_note_f1"]
        and count_rate >= VALIDATION_GATE["minimum_exact_note_count_rate"]
    )
    return {
        **VALIDATION_GATE,
        "observed_pitch_only_note_f1": f1,
        "observed_exact_note_count_rate": count_rate,
        "passed": passed,
        "failure_action": None if passed else "skip_heldout_without_reading_heldout_truth",
    }


def _write_overlays(
    prepared_rows: Sequence[PreparedRequest],
    prediction_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    truth_by_identity: Mapping[tuple[Any, ...], Sequence[Mapping[str, Any]]] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for prepared, prediction in zip(prepared_rows, prediction_rows, strict=True):
        image = prepared.image.convert("RGB")
        draw = ImageDraw.Draw(image)
        for line in prepared.staff_lines:
            draw.line((0, line, image.width - 1, line), fill="#2d73d5", width=1)
        for candidate in prediction["candidates"]:
            center = candidate["center"]
            stem = candidate["stem"]
            x = float(center["x"])
            y = float(center["y"])
            draw.line(
                (float(stem["x"]), int(stem["top"]), float(stem["x"]), int(stem["bottom"])),
                fill="#e67e22",
                width=2,
            )
            radius = max(3, round(prepared.spacing * 0.18))
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline="#d62728",
                width=2,
            )
            draw.text((x + radius + 1, y - radius), str(candidate["pitch_midi"]), fill="#d62728")
        if truth_by_identity is not None:
            for truth in truth_by_identity[_identity_key(prepared.request["identity"])]:
                x = float(truth["center"]["x"])
                y = float(truth["center"]["y"])
                radius = max(4, round(prepared.spacing * 0.24))
                draw.rectangle((x - radius, y - radius, x + radius, y + radius), outline="#16a34a")
        identity = prepared.request["identity"]
        name = (
            f"system_{int(identity['system_index']):03d}_"
            f"measure_{int(identity['system_measure_index']):03d}.png"
        )
        image.save(output_dir / name)


def _markdown_report(report: Mapping[str, Any]) -> str:
    development = report["development_fitting"]
    validation = report["validation"]
    validation_summary = validation["metrics"]["summary"]
    gate = validation["gate"]
    heldout = report["heldout"]
    lines = [
        "# Stem Endpoint Detector Spike",
        "",
        "## Scope",
        "",
        "The detector uses vertical stem runs, staff-line-aware pixel suppression, compact "
        "endpoint attachment evidence, treble-staff natural-pitch quantization, and deterministic "
        "deduplication. It does not recognize rhythm, onset, duration, rests, or accidentals.",
        "",
        "## Development Fitting",
        "",
        f"- Labels: `{development['allowed_coordinate_labels']}`",
        f"- Parameterizations: `{development['parameterizations']}`",
        f"- Selected config: `{development['selected_config_key']}`",
        f"- Coordinate F1: `{development['selected']['f1']:.3f}`",
        f"- Exact count rate: `{development['selected']['exact_count_rate']:.3f}`",
        "",
        "## Validation",
        "",
        f"- Prediction SHA256: `{validation['prediction_seal']['sha256']}`",
        f"- Targets: `{validation_summary['targets']}`",
        f"- Predicted/truth notes: `{validation_summary['predicted_note_count']}/"
        f"{validation_summary['truth_note_count']}`",
        f"- Exact count rate: `{validation_summary['exact_note_count_rate']:.3f}`",
        f"- Ordered natural-pitch accuracy: "
        f"`{validation_summary['ordered_natural_pitch_accuracy']:.3f}`",
        f"- Pitch-only note P/R/F1: `{validation_summary['pitch_only_note_precision']:.3f}/"
        f"{validation_summary['pitch_only_note_recall']:.3f}/"
        f"{validation_summary['pitch_only_note_f1']:.3f}`",
        f"- Gate: F1 >= `{gate['minimum_pitch_only_note_f1']:.2f}` and exact count >= "
        f"`{gate['minimum_exact_note_count_rate']:.2f}`; passed `{gate['passed']}`",
        "",
        "Validation predictions and overlays were persisted before validation truth was opened.",
        "",
        "## Heldout",
        "",
        f"- Status: `{heldout['status']}`",
    ]
    if heldout["metrics"] is not None:
        summary = heldout["metrics"]["summary"]
        lines.extend(
            [
                f"- Prediction SHA256: `{heldout['prediction_seal']['sha256']}`",
                f"- Exact count rate: `{summary['exact_note_count_rate']:.3f}`",
                f"- Ordered natural-pitch accuracy: "
                f"`{summary['ordered_natural_pitch_accuracy']:.3f}`",
                f"- Pitch-only note F1: `{summary['pitch_only_note_f1']:.3f}`",
                "",
                "Heldout predictions were sealed after the validation gate passed and before "
                "heldout truth was opened. Heldout truth was evaluated once.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "The validation gate failed. No heldout predictions were generated and heldout "
                "truth was not read.",
            ]
        )
    return "\n".join(lines) + "\n"


def _selected_requests(
    rows: Sequence[dict[str, Any]], targets: Sequence[tuple[int, int]]
) -> list[dict[str, Any]]:
    by_target = {
        (
            int(row["identity"]["system_index"]),
            int(row["identity"]["system_measure_index"]),
        ): row
        for row in rows
    }
    missing = [target for target in targets if target not in by_target]
    if missing:
        raise ValueError(f"Missing development requests: {missing}")
    return [by_target[target] for target in targets]


def _require_systems(
    rows: Sequence[Mapping[str, Any]], expected: Sequence[int], split: str
) -> None:
    actual = sorted({int(row["identity"]["system_index"]) for row in rows})
    if actual != sorted(expected):
        raise ValueError(f"Unexpected {split} systems: {actual}; expected {sorted(expected)}")


def _identity_key(identity: Mapping[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(identity["slug"]),
        int(identity["system_index"]),
        int(identity["system_measure_index"]),
        int(identity["global_measure_index"]),
    )


def _vertical_overlap(left: StemRun, right: StemRun) -> float:
    overlap = max(0, min(left.bottom, right.bottom) - max(left.top, right.top) + 1)
    return overlap / max(1, min(left.length, right.length))


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    return _ratio(2 * tp, 2 * tp + fp + fn)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
