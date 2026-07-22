"""Build and score a leak-resistant melody-event benchmark for the VLM spike.

The benchmark deliberately separates inference requests from ground truth. Image
paths, hashes, staff geometry, and allowed musical context are frozen first. Only
after every request file is written does the builder read canonical events and
write the private truth side of the benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from score2abc.events import measure_length_beats  # noqa: E402
from score2abc.metrics import compare_events  # noqa: E402

SCHEMA_VERSION = 1
DEFAULT_SLUG = "jaime-llanos_12_aviador_pasillo_fulgencio-garcia"
DEFAULT_TIME_SIGNATURE = "3/4"


@dataclass(frozen=True)
class BenchmarkTarget:
    system_index: int
    system_measure_index: int
    global_measure_index: int
    manifest_global_measure_index: int


def _targets(
    system_index: int,
    count: int,
    global_start: int,
    *,
    manifest_global_start: int | None = None,
) -> tuple[BenchmarkTarget, ...]:
    stored_start = global_start if manifest_global_start is None else manifest_global_start
    return tuple(
        BenchmarkTarget(
            system_index,
            local_index,
            global_start + local_index - 1,
            stored_start + local_index - 1,
        )
        for local_index in range(1, count + 1)
    )


# Development includes every system already inspected during this spike. The
# validation systems have exact physical crops but their old manifest indices
# are off by one; both indices are frozen here so the correction is auditable.
# System 3 remains sealed for a final held-out score.
BENCHMARK_SPLITS: dict[str, tuple[BenchmarkTarget, ...]] = {
    "development": (*_targets(1, 8, 0), *_targets(2, 9, 8)),
    "validation": (
        *_targets(7, 7, 46, manifest_global_start=47),
        *_targets(8, 7, 53, manifest_global_start=54),
    ),
    "heldout": _targets(3, 9, 17),
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            selected_splits = tuple(args.split or ("development", "validation", "heldout"))
            root = build_benchmark(
                args.out_dir,
                slug=args.slug,
                ground_truth_dir=args.ground_truth,
                split_names=selected_splits,
                clef=args.clef,
                time_signature=args.time_signature,
                key_hint=args.key_hint,
            )
            print(root)
        else:
            report_path = evaluate_prediction_file(
                args.benchmark_dir,
                split_name=args.split,
                predictions_path=args.predictions,
            )
            print(report_path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Freeze requests, then materialize truth.")
    build.add_argument("out_dir", type=Path)
    build.add_argument("--slug", default=DEFAULT_SLUG)
    build.add_argument("--ground-truth", type=Path, default=Path("dataset/ground_truth"))
    build.add_argument("--split", action="append", choices=tuple(BENCHMARK_SPLITS))
    build.add_argument("--clef", default="treble")
    build.add_argument("--time-signature", default=DEFAULT_TIME_SIGNATURE)
    build.add_argument("--key-hint", default=None)

    evaluate = subparsers.add_parser("evaluate", help="Score a prediction JSONL file.")
    evaluate.add_argument("benchmark_dir", type=Path)
    evaluate.add_argument("--split", required=True, choices=tuple(BENCHMARK_SPLITS))
    evaluate.add_argument("--predictions", type=Path, required=True)
    return parser


def build_benchmark(
    out_dir: Path,
    *,
    slug: str,
    ground_truth_dir: Path,
    split_names: Sequence[str],
    clef: str,
    time_signature: str,
    key_hint: str | None,
) -> Path:
    _validate_split_names(split_names)
    measure_length = measure_length_beats(time_signature)
    root = out_dir / slug / "vlm_melody_event_benchmark"

    requests_by_split: dict[str, list[dict[str, Any]]] = {}
    for split_name in split_names:
        requests = prepare_requests(
            out_dir,
            slug=slug,
            targets=BENCHMARK_SPLITS[split_name],
            split_name=split_name,
            clef=clef,
            time_signature=time_signature,
            key_hint=key_hint,
        )
        split_dir = root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(split_dir / "requests.jsonl", requests)
        requests_by_split[split_name] = requests

    # This read intentionally happens after all request files have been frozen.
    truth_payload = _load_truth(ground_truth_dir / f"{slug}.json")
    truth_time_signature = str(truth_payload.get("time_signature") or "")
    if truth_time_signature != time_signature:
        raise ValueError(
            "Benchmark context and canonical truth disagree on time signature: "
            f"{time_signature!r} != {truth_time_signature!r}"
        )

    split_metadata: list[dict[str, Any]] = []
    for split_name, requests in requests_by_split.items():
        truths = build_truth_rows(
            requests,
            truth_payload,
            measure_length=measure_length,
        )
        split_dir = root / split_name
        truth_path = split_dir / "truth.jsonl"
        _write_jsonl(truth_path, truths)
        request_path = split_dir / "requests.jsonl"
        split_metadata.append(
            {
                "name": split_name,
                "targets": len(requests),
                "global_measure_indices": [
                    request["identity"]["global_measure_index"] for request in requests
                ],
                "requests_sha256": _sha256(request_path),
                "truth_sha256": _sha256(truth_path),
            }
        )

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "kind": "vlm_melody_event_benchmark",
        "slug": slug,
        "inference_contract": {
            "ground_truth_available_during_inference": False,
            "allowed_context": {
                "clef": clef,
                "time_signature": time_signature,
                "key_hint": key_hint,
                "expected_measure_beats": _fraction_text(measure_length),
            },
            "prediction_schema": {
                "identity": "Copy the request identity object exactly.",
                "notes": ["onset_beats", "duration_beats", "pitch_midi", "accidental?"],
                "rests": ["onset_beats", "duration_beats"],
            },
        },
        "mapping_scope": (
            "Physical-system mapping is frozen independently of generated global indices. "
            "Systems 4-6 and 9-10 are quarantined for segmentation ambiguity or prior review."
        ),
        "splits": split_metadata,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "benchmark.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return root


def prepare_requests(
    out_dir: Path,
    *,
    slug: str,
    targets: Iterable[BenchmarkTarget],
    split_name: str,
    clef: str,
    time_signature: str,
    key_hint: str | None,
) -> list[dict[str, Any]]:
    """Build inference requests without accepting or reading a truth path."""
    measure_length = measure_length_beats(time_signature)
    records = _load_measure_records(out_dir, slug)
    by_identity = {
        (int(record["system_index"]), int(record["system_measure_index"])): record
        for record in records
    }
    requests: list[dict[str, Any]] = []
    for target in targets:
        key = (target.system_index, target.system_measure_index)
        if key not in by_identity:
            raise ValueError(
                "Missing melody-input record for "
                f"system={target.system_index}, measure={target.system_measure_index}"
            )
        record = by_identity[key]
        actual_global_index = int(record["global_measure_index"])
        if actual_global_index != target.manifest_global_measure_index:
            raise ValueError(
                "Crop-to-canonical mapping drift for "
                f"system={target.system_index}, measure={target.system_measure_index}: "
                f"manifest={actual_global_index}, "
                f"expected_stored={target.manifest_global_measure_index}"
            )

        images: dict[str, dict[str, Any]] = {}
        for input_kind, path_key in (("raw", "measure_raw"), ("staff", "measure_staff")):
            image_path = _resolve_path(out_dir, str(record["paths"][path_key]))
            with Image.open(image_path) as image:
                width, height = image.size
            images[input_kind] = {
                "path_relative_to_out": _output_relative_path(image_path, out_dir),
                "sha256": _sha256(image_path),
                "width_px": width,
                "height_px": height,
            }

        requests.append(
            {
                "schema_version": SCHEMA_VERSION,
                "split": split_name,
                "identity": {
                    "slug": slug,
                    "system_index": target.system_index,
                    "system_measure_index": target.system_measure_index,
                    "global_measure_index": target.global_measure_index,
                },
                "mapping": {
                    "manifest_global_measure_index": actual_global_index,
                    "canonical_global_measure_index": target.global_measure_index,
                    "index_correction": target.global_measure_index - actual_global_index,
                },
                "images": images,
                "staff_geometry": {
                    "raw_staff_lines_y_px": record["staff_lines_y_px_in_system"],
                    "staff_crop_lines_y_px": record["staff_lines_y_px_in_staff_crop"],
                },
                "allowed_context": {
                    "clef": clef,
                    "time_signature": time_signature,
                    "key_hint": key_hint,
                    "expected_measure_beats": _fraction_text(measure_length),
                    "allow_pickup": bool(record.get("allow_pickup", False)),
                },
            }
        )
    return requests


def build_truth_rows(
    requests: Iterable[Mapping[str, Any]],
    truth_payload: Mapping[str, Any],
    *,
    measure_length: Fraction,
) -> list[dict[str, Any]]:
    notes_by_measure: dict[int, list[dict[str, Any]]] = {}
    for raw_note in truth_payload.get("notes") or []:
        note = dict(raw_note)
        notes_by_measure.setdefault(int(note["measure"]), []).append(note)
    for notes in notes_by_measure.values():
        notes.sort(key=lambda item: (float(item["onset_beats"]), int(item["pitch_midi"])))

    rows: list[dict[str, Any]] = []
    for request in requests:
        identity = dict(request["identity"])
        global_measure_index = int(identity["global_measure_index"])
        notes = [
            _canonical_note(note, global_measure_index)
            for note in notes_by_measure.get(global_measure_index, [])
        ]
        allow_pickup = bool(request["allowed_context"]["allow_pickup"])
        extent = _measure_extent(notes, measure_length, allow_pickup=allow_pickup)
        rests = _derive_rests(notes, extent)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "identity": identity,
                "measure_extent_beats": _fraction_number(extent),
                "notes": notes,
                "rests": rests,
            }
        )
    return rows


def evaluate_prediction_file(
    benchmark_dir: Path,
    *,
    split_name: str,
    predictions_path: Path,
) -> Path:
    metadata = json.loads((benchmark_dir / "benchmark.json").read_text(encoding="utf-8"))
    split_metadata = next(
        (split for split in metadata["splits"] if split.get("name") == split_name),
        None,
    )
    if split_metadata is None:
        raise ValueError(f"Split {split_name!r} is not included in the current benchmark metadata")

    split_dir = benchmark_dir / split_name
    request_path = split_dir / "requests.jsonl"
    truth_path = split_dir / "truth.jsonl"
    for path, hash_key in (
        (request_path, "requests_sha256"),
        (truth_path, "truth_sha256"),
    ):
        expected_hash = str(split_metadata[hash_key])
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Benchmark {path.name} hash mismatch for split {split_name!r}: "
                f"metadata={expected_hash}, current={actual_hash}"
            )

    truth_rows = _read_jsonl(truth_path)
    prediction_rows = _read_jsonl(predictions_path)
    report = evaluate_predictions(truth_rows, prediction_rows)
    report_path = split_dir / f"report_{predictions_path.stem}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report_path


def evaluate_predictions(
    truth_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    predictions: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for prediction in prediction_rows:
        key = _identity_key(prediction["identity"])
        if key in predictions:
            raise ValueError(f"Duplicate prediction identity: {key}")
        predictions[key] = prediction

    truth_keys = {_identity_key(row["identity"]) for row in truth_rows}
    unknown_keys = set(predictions) - truth_keys
    if unknown_keys:
        raise ValueError(
            f"Predictions contain identities outside this split: {sorted(unknown_keys)}"
        )

    results: list[dict[str, Any]] = []
    aggregate_truth_notes: list[dict[str, Any]] = []
    aggregate_pred_notes: list[dict[str, Any]] = []
    rest_tp = rest_fp = rest_fn = 0
    for truth in truth_rows:
        identity = dict(truth["identity"])
        key = _identity_key(identity)
        prediction = predictions.get(key)
        pred_notes = list(prediction.get("notes") or []) if prediction else []
        pred_rests = list(prediction.get("rests") or []) if prediction else []
        global_measure_index = int(identity["global_measure_index"])
        _validate_prediction_events(
            pred_notes,
            pred_rests,
            measure_extent=Fraction(str(truth["measure_extent_beats"])),
            identity=identity,
        )
        normalized_truth_notes = _notes_for_metrics(truth.get("notes") or [], global_measure_index)
        normalized_pred_notes = _notes_for_metrics(pred_notes, global_measure_index)
        note_metrics = compare_events(
            {"notes": normalized_pred_notes},
            {"notes": normalized_truth_notes},
        )
        truth_rest_set = _normalize_rests(truth.get("rests") or [])
        pred_rest_set = _normalize_rests(pred_rests)
        local_rest_tp = len(truth_rest_set & pred_rest_set)
        local_rest_fp = len(pred_rest_set - truth_rest_set)
        local_rest_fn = len(truth_rest_set - pred_rest_set)
        rest_tp += local_rest_tp
        rest_fp += local_rest_fp
        rest_fn += local_rest_fn
        ordered = _ordered_note_metrics(normalized_pred_notes, normalized_truth_notes)
        exact_measure = (
            bool(note_metrics["note_count_match"])
            and float(note_metrics["note_f1"]) == 1.0
            and pred_rest_set == truth_rest_set
        )
        results.append(
            {
                "identity": identity,
                "status": "evaluated" if prediction else "missing_prediction",
                **ordered,
                "note_f1": note_metrics["note_f1"],
                "rest_tp": local_rest_tp,
                "rest_fp": local_rest_fp,
                "rest_fn": local_rest_fn,
                "exact_measure": exact_measure,
            }
        )
        aggregate_truth_notes.extend(normalized_truth_notes)
        aggregate_pred_notes.extend(normalized_pred_notes)

    note_summary = compare_events(
        {"notes": aggregate_pred_notes},
        {"notes": aggregate_truth_notes},
    )
    evaluated_count = len(results)
    summary = {
        "targets": evaluated_count,
        "predicted": sum(1 for result in results if result["status"] == "evaluated"),
        "exact_measures": sum(1 for result in results if result["exact_measure"]),
        "exact_measure_rate": (
            round(sum(1 for result in results if result["exact_measure"]) / evaluated_count, 6)
            if evaluated_count
            else 0.0
        ),
        "note_f1": note_summary["note_f1"],
        "note_precision": note_summary["note_precision"],
        "note_recall": note_summary["note_recall"],
        "ordered_pitch_accuracy": _weighted_accuracy(results, "pitch_matches", "compared_notes"),
        "ordered_onset_accuracy": _weighted_accuracy(results, "onset_matches", "compared_notes"),
        "ordered_duration_accuracy": _weighted_accuracy(
            results, "duration_matches", "compared_notes"
        ),
        "rest_precision": _ratio(rest_tp, rest_tp + rest_fp),
        "rest_recall": _ratio(rest_tp, rest_tp + rest_fn),
        "rest_f1": _f1(rest_tp, rest_fp, rest_fn),
    }
    return {"schema_version": SCHEMA_VERSION, "summary": summary, "results": results}


def _measure_extent(
    notes: Sequence[Mapping[str, Any]],
    measure_length: Fraction,
    *,
    allow_pickup: bool,
) -> Fraction:
    if not allow_pickup:
        return measure_length
    if not notes:
        raise ValueError("A pickup measure cannot infer its extent without notes")
    return max(
        Fraction(str(note["onset_beats"])) + Fraction(str(note["duration_beats"])) for note in notes
    )


def _derive_rests(
    notes: Sequence[Mapping[str, Any]], measure_extent: Fraction
) -> list[dict[str, float | int]]:
    intervals: list[tuple[Fraction, Fraction]] = []
    for note in notes:
        onset = Fraction(str(note["onset_beats"]))
        duration = Fraction(str(note["duration_beats"]))
        if onset < 0:
            raise ValueError(f"Negative canonical note onset: {note}")
        if duration <= 0:
            raise ValueError(f"Non-positive note duration: {note}")
        intervals.append((onset, onset + duration))

    merged: list[tuple[Fraction, Fraction]] = []
    for onset, end in sorted(intervals):
        if merged and onset <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((onset, end))

    rests: list[dict[str, float | int]] = []
    cursor = Fraction(0)
    for onset, end in merged:
        if onset > cursor:
            rests.append(
                {
                    "onset_beats": _fraction_number(cursor),
                    "duration_beats": _fraction_number(onset - cursor),
                }
            )
        cursor = max(cursor, end)
    if cursor > measure_extent:
        raise ValueError(
            f"Canonical notes extend past measure: {float(cursor)} > {float(measure_extent)}"
        )
    if cursor < measure_extent:
        rests.append(
            {
                "onset_beats": _fraction_number(cursor),
                "duration_beats": _fraction_number(measure_extent - cursor),
            }
        )
    return rests


def _validate_prediction_events(
    notes: Sequence[Mapping[str, Any]],
    rests: Sequence[Mapping[str, Any]],
    *,
    measure_extent: Fraction,
    identity: Mapping[str, Any],
) -> None:
    note_intervals: list[tuple[Fraction, Fraction]] = []
    for note in notes:
        onset, end = _validated_interval(note, measure_extent, "note", identity)
        pitch = int(note["pitch_midi"])
        if not 0 <= pitch <= 127:
            raise ValueError(f"MIDI pitch outside 0..127 for {identity}: {pitch}")
        note_intervals.append((onset, end))

    rest_intervals = [_validated_interval(rest, measure_extent, "rest", identity) for rest in rests]
    for first_index, first in enumerate(rest_intervals):
        for second in rest_intervals[first_index + 1 :]:
            if _intervals_overlap(first, second):
                raise ValueError(f"Overlapping predicted rests for {identity}: {first}, {second}")
        for note_interval in note_intervals:
            if _intervals_overlap(first, note_interval):
                raise ValueError(
                    f"Predicted rest overlaps a sounding note for {identity}: "
                    f"{first}, {note_interval}"
                )


def _validated_interval(
    event: Mapping[str, Any],
    measure_extent: Fraction,
    kind: str,
    identity: Mapping[str, Any],
) -> tuple[Fraction, Fraction]:
    onset = Fraction(str(event["onset_beats"]))
    duration = Fraction(str(event["duration_beats"]))
    if onset < 0:
        raise ValueError(f"Negative predicted {kind} onset for {identity}: {event}")
    if duration <= 0:
        raise ValueError(f"Non-positive predicted {kind} duration for {identity}: {event}")
    end = onset + duration
    if end > measure_extent:
        raise ValueError(
            f"Predicted {kind} extends past measure for {identity}: "
            f"{float(end)} > {float(measure_extent)}"
        )
    return onset, end


def _intervals_overlap(first: tuple[Fraction, Fraction], second: tuple[Fraction, Fraction]) -> bool:
    return first[0] < second[1] and second[0] < first[1]


def _canonical_note(note: Mapping[str, Any], measure: int) -> dict[str, Any]:
    canonical = {
        "onset_beats": note["onset_beats"],
        "duration_beats": note["duration_beats"],
        "pitch_midi": int(note["pitch_midi"]),
    }
    if note.get("accidental") is not None:
        canonical["accidental"] = int(note["accidental"])
    if int(note["measure"]) != measure:
        raise ValueError(f"Canonical note measure mismatch: {note}")
    return canonical


def _ordered_note_metrics(
    pred_notes: Sequence[Mapping[str, Any]], truth_notes: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    compared = max(len(pred_notes), len(truth_notes))
    pairs = zip(pred_notes, truth_notes, strict=False)
    pitch_matches = onset_matches = duration_matches = 0
    for pred, truth in pairs:
        pitch_matches += int(int(pred["pitch_midi"]) == int(truth["pitch_midi"]))
        onset_matches += int(_float_key(pred["onset_beats"]) == _float_key(truth["onset_beats"]))
        duration_matches += int(
            _float_key(pred["duration_beats"]) == _float_key(truth["duration_beats"])
        )
    return {
        "pred_note_count": len(pred_notes),
        "truth_note_count": len(truth_notes),
        "compared_notes": compared,
        "pitch_matches": pitch_matches,
        "onset_matches": onset_matches,
        "duration_matches": duration_matches,
    }


def _notes_for_metrics(notes: Iterable[Mapping[str, Any]], measure: int) -> list[dict[str, Any]]:
    normalized = []
    for note in notes:
        normalized.append(
            {
                "measure": measure,
                "onset_beats": note["onset_beats"],
                "duration_beats": note["duration_beats"],
                "pitch_midi": note["pitch_midi"],
            }
        )
    return sorted(
        normalized,
        key=lambda note: (
            _float_key(note["onset_beats"]),
            int(note["pitch_midi"]),
            _float_key(note["duration_beats"]),
        ),
    )


def _normalize_rests(rests: Iterable[Mapping[str, Any]]) -> set[tuple[float, float]]:
    return {(_float_key(rest["onset_beats"]), _float_key(rest["duration_beats"])) for rest in rests}


def _weighted_accuracy(results: Iterable[Mapping[str, Any]], matches: str, total: str) -> float:
    rows = list(results)
    denominator = sum(int(row[total]) for row in rows)
    return _ratio(sum(int(row[matches]) for row in rows), denominator)


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return round(2 * precision * recall / (precision + recall), 6) if precision + recall else 0.0


def _load_measure_records(out_dir: Path, slug: str) -> list[dict[str, Any]]:
    candidates = (
        out_dir / slug / "vlm_melody_inputs" / "manifest.jsonl",
        out_dir / "vlm_melody_inputs_manifest.jsonl",
    )
    for path in candidates:
        if path.exists():
            return [row for row in _read_jsonl(path) if row.get("slug") == slug]
    raise ValueError(f"No VLM melody-input manifest found under {out_dir}")


def _load_truth(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(out_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    candidates = (path, out_dir / path, out_dir.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise ValueError(f"Referenced benchmark image does not exist: {raw_path}")


def _output_relative_path(path: Path, out_dir: Path) -> str:
    try:
        return path.resolve().relative_to(out_dir.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Benchmark image is outside the output directory: {path}") from exc


def _identity_key(identity: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(identity["slug"]),
        int(identity["system_index"]),
        int(identity["system_measure_index"]),
        int(identity["global_measure_index"]),
    )


def _validate_split_names(split_names: Sequence[str]) -> None:
    if not split_names:
        raise ValueError("At least one benchmark split is required")
    if len(split_names) != len(set(split_names)):
        raise ValueError(f"Duplicate benchmark splits: {split_names}")
    unknown = set(split_names) - set(BENCHMARK_SPLITS)
    if unknown:
        raise ValueError(f"Unknown benchmark splits: {sorted(unknown)}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fraction_text(value: Fraction) -> str:
    return (
        str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    )


def _fraction_number(value: Fraction) -> float | int:
    return value.numerator if value.denominator == 1 else float(value)


def _float_key(value: Any) -> float:
    return round(float(value), 6)


if __name__ == "__main__":
    raise SystemExit(main())
