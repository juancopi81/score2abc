"""Test whether later handwritten measures repeat development measures visually.

This is a repetition-propagation spike, not a general OMR system. It freezes
staff-normalized, GT-blind image features for every benchmark split before
opening any truth. Development truth is then used only to label the template
bank and to select one feature variant plus a rejection threshold by strict
leave-one-measure-out retrieval. Validation and heldout predictions are sealed
before their corresponding truth files are opened.

Example:
    uv run python scripts/experiments/spike_measure_retrieval_transcriber.py out
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_vlm_melody_event_benchmark as benchmark  # noqa: E402

DEFAULT_OUT_DIR = REPO_ROOT / "out"
DEFAULT_SLUG = benchmark.DEFAULT_SLUG
OUTPUT_SUBDIR = "measure_retrieval_transcriber"
FEATURE_HEIGHT = 96
FEATURE_WIDTH = 160
TARGET_STAFF_SPACING = 12.0
VALIDATION_NOTE_F1_FLOOR = 0.17284
HELDOUT_EVALUATION_ALLOWED = False
FEATURE_VARIANTS = (
    "staff_remove_fixed",
    "staff_downweight_fixed",
    "raw_remove_fixed",
)

TruthLoader = Callable[[Path], list[dict[str, Any]]]


@dataclass(frozen=True)
class BinaryFeature:
    """Fixed-size binary ink map stored as rows of zero/one integers."""

    variant: str
    width: int
    height: int
    rows: tuple[tuple[int, ...], ...]
    bit_rows: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "bit_rows",
            tuple(sum(value << x for x, value in enumerate(row)) for row in self.rows),
        )

    @property
    def ink_pixels(self) -> int:
        return sum(sum(row) for row in self.rows)

    def image(self) -> Image.Image:
        image = Image.new("L", (self.width, self.height), color=255)
        image.putdata([0 if value else 255 for row in self.rows for value in row])
        return image


@dataclass(frozen=True)
class FeatureRecord:
    request: dict[str, Any]
    features: Mapping[str, BinaryFeature]

    @property
    def identity(self) -> dict[str, Any]:
        return dict(self.request["identity"])


@dataclass(frozen=True)
class Template:
    identity: dict[str, Any]
    feature: BinaryFeature
    notes: tuple[dict[str, Any], ...]
    rests: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class Retrieval:
    source: Template
    distance: float
    second_distance: float | None

    @property
    def margin(self) -> float:
        if self.second_distance is None:
            return 0.0
        return max(0.0, self.second_distance - self.distance)


@dataclass(frozen=True)
class SplitArtifacts:
    split: str
    predictions_path: Path
    retrievals_path: Path
    contact_sheet_path: Path
    predictions_sha256: str
    retrievals_sha256: str
    contact_sheet_sha256: str


class ProtocolGate:
    """Make prediction-before-truth and validation-before-heldout enforceable."""

    def __init__(self) -> None:
        self.feature_freeze_complete = False
        self.seals: dict[str, str] = {}
        self.validation_gate: dict[str, Any] | None = None
        self.truth_access_log: list[dict[str, Any]] = []

    def mark_feature_freeze_complete(self) -> None:
        self.feature_freeze_complete = True

    def read_development_truth(self, path: Path, loader: TruthLoader) -> list[dict[str, Any]]:
        if not self.feature_freeze_complete:
            raise RuntimeError("Development truth requires all split features to be frozen")
        rows = loader(path)
        self.truth_access_log.append({"split": "development", "after_all_feature_freezes": True})
        return rows

    def seal_predictions(self, split: str, path: Path) -> str:
        if split not in {"validation", "heldout"}:
            raise ValueError(f"Unsupported prediction seal split: {split}")
        if split == "heldout" and not self.validation_passed:
            raise RuntimeError("Heldout predictions require a passing validation gate")
        digest = _sha256(path)
        self.seals[split] = digest
        return digest

    @property
    def validation_passed(self) -> bool:
        return bool(self.validation_gate and self.validation_gate["passed"])

    def record_validation_gate(self, result: Mapping[str, Any]) -> None:
        if "validation" not in self.seals:
            raise RuntimeError("Validation gate requires sealed validation predictions")
        self.validation_gate = dict(result)

    def read_sealed_truth(
        self, split: str, path: Path, loader: TruthLoader
    ) -> list[dict[str, Any]]:
        if split not in self.seals:
            raise RuntimeError(f"{split.title()} truth requires sealed predictions")
        if split == "heldout" and not self.validation_passed:
            raise RuntimeError("Heldout truth requires a passing validation gate")
        rows = loader(path)
        self.truth_access_log.append(
            {
                "split": split,
                "after_prediction_sha256": self.seals[split],
            }
        )
        return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    args = parser.parse_args(argv)
    try:
        report = run_experiment(args.out_dir, slug=args.slug)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    validation = report["validation"]
    print(report["artifacts"]["report_json"])
    print(
        "validation: "
        f"note_f1={validation['metrics']['summary']['note_f1']:.6f} "
        f"exact={validation['metrics']['summary']['exact_measures']} "
        f"gate={'pass' if validation['gate']['passed'] else 'fail'}"
    )
    print(f"heldout: {report['heldout']['status']}")
    return 0


def run_experiment(
    out_dir: Path,
    *,
    slug: str = DEFAULT_SLUG,
    truth_loader: TruthLoader | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the sealed retrieval protocol and return its deterministic report."""
    truth_loader = truth_loader or _read_jsonl
    benchmark_dir = out_dir / slug / "vlm_melody_event_benchmark"
    experiment_dir = output_dir or benchmark_dir / OUTPUT_SUBDIR
    experiment_dir.mkdir(parents=True, exist_ok=True)
    report_path = experiment_dir / "report.json"
    previous_core_hashes = _previous_core_hashes(report_path)

    requests: dict[str, list[dict[str, Any]]] = {}
    feature_records: dict[str, list[FeatureRecord]] = {}
    feature_artifacts: dict[str, dict[str, Any]] = {}
    for split in ("development", "validation", "heldout"):
        request_path = benchmark_dir / split / "requests.jsonl"
        requests[split] = _read_jsonl(request_path)
        feature_records[split], feature_artifacts[split] = freeze_split_features(
            requests[split],
            out_dir=out_dir,
            split=split,
            output_dir=experiment_dir / "features" / split,
            requests_path=request_path,
        )

    feature_freeze_path = experiment_dir / "feature_freeze.json"
    feature_freeze = {
        "schema_version": 1,
        "kind": "all_split_gt_blind_feature_freeze",
        "feature_variants": list(FEATURE_VARIANTS),
        "splits": feature_artifacts,
        "truth_read_before_freeze": False,
    }
    _write_json(feature_freeze_path, feature_freeze)
    protocol = ProtocolGate()
    protocol.mark_feature_freeze_complete()

    development_truth = protocol.read_development_truth(
        benchmark_dir / "development" / "truth.jsonl", truth_loader
    )
    truth_by_key = {_identity_key(row["identity"]): row for row in development_truth}
    _validate_matching_identities(feature_records["development"], truth_by_key)
    tuning = tune_retriever(feature_records["development"], truth_by_key)
    variant = str(tuning["selected_variant"])
    threshold = float(tuning["selected_threshold"])
    templates = build_template_bank(feature_records["development"], truth_by_key, variant)
    template_bank_path = experiment_dir / "template_bank.json"
    _write_json(
        template_bank_path,
        {
            "schema_version": 1,
            "variant": variant,
            "threshold": threshold,
            "templates": [
                {
                    "identity": template.identity,
                    "feature_ink_pixels": template.feature.ink_pixels,
                    "notes": list(template.notes),
                    "rests": list(template.rests),
                }
                for template in templates
            ],
        },
    )

    development_predictions, development_retrievals = predict_records(
        feature_records["development"],
        templates,
        variant=variant,
        threshold=threshold,
        exclude_self=True,
    )
    development_metrics = benchmark.evaluate_predictions(development_truth, development_predictions)
    development_artifacts = freeze_split_predictions(
        "development",
        feature_records["development"],
        development_predictions,
        development_retrievals,
        output_dir=experiment_dir / "development",
        out_dir=out_dir,
        template_records=feature_records["development"],
    )

    validation_predictions, validation_retrievals = predict_records(
        feature_records["validation"],
        templates,
        variant=variant,
        threshold=threshold,
    )
    validation_artifacts = freeze_split_predictions(
        "validation",
        feature_records["validation"],
        validation_predictions,
        validation_retrievals,
        output_dir=experiment_dir / "validation",
        out_dir=out_dir,
        template_records=feature_records["development"],
    )
    protocol.seal_predictions("validation", validation_artifacts.predictions_path)
    validation_truth = protocol.read_sealed_truth(
        "validation", benchmark_dir / "validation" / "truth.jsonl", truth_loader
    )
    validation_metrics = benchmark.evaluate_predictions(validation_truth, validation_predictions)
    validation_gate = apply_validation_gate(validation_metrics["summary"])
    protocol.record_validation_gate(validation_gate)

    heldout: dict[str, Any] = {
        "status": "not_opened_validation_gate_failed",
        "metrics": None,
        "artifacts": None,
    }
    if validation_gate["passed"] and HELDOUT_EVALUATION_ALLOWED:
        heldout_predictions, heldout_retrievals = predict_records(
            feature_records["heldout"],
            templates,
            variant=variant,
            threshold=threshold,
        )
        heldout_artifacts = freeze_split_predictions(
            "heldout",
            feature_records["heldout"],
            heldout_predictions,
            heldout_retrievals,
            output_dir=experiment_dir / "heldout",
            out_dir=out_dir,
            template_records=feature_records["development"],
        )
        protocol.seal_predictions("heldout", heldout_artifacts.predictions_path)
        heldout_truth = protocol.read_sealed_truth(
            "heldout", benchmark_dir / "heldout" / "truth.jsonl", truth_loader
        )
        heldout_metrics = benchmark.evaluate_predictions(heldout_truth, heldout_predictions)
        heldout = {
            "status": "evaluated_once_after_validation_pass",
            "metrics": heldout_metrics,
            "artifacts": _artifact_summary(heldout_artifacts),
        }
    elif validation_gate["passed"]:
        heldout["status"] = "skipped_not_presealed_before_prior_s3_open"

    report_md_path = experiment_dir / "report.md"
    current_core_hashes = {
        "feature_freeze_sha256": _sha256(feature_freeze_path),
        "template_bank_sha256": _sha256(template_bank_path),
        "development_predictions_sha256": development_artifacts.predictions_sha256,
        "validation_predictions_sha256": validation_artifacts.predictions_sha256,
    }
    report = {
        "schema_version": 1,
        "kind": "gt_blind_repeated_measure_retrieval_transcriber",
        "slug": slug,
        "hypothesis": "later handwritten measures may visually repeat development measures",
        "limitation": (
            "This copies canonical events from a visually retrieved development measure. "
            "It is repetition propagation, not general optical music recognition."
        ),
        "protocol": {
            "feature_freeze_before_any_truth": True,
            "development_training_scope": "systems 1-2 only",
            "selection": "development leave-one-measure-out retrieval",
            "validation": "systems 7-8 predictions sealed before truth",
            "heldout": (
                "system 3 features frozen; prediction and truth skipped by coordination lock"
            ),
            "heldout_coordination_lock": (
                "Another preregistered arm opened S3 before this arm sealed predictions. "
                "This arm therefore skips S3 prediction and evaluation."
            ),
            "network_used": False,
            "production_code_changed": False,
        },
        "feature_freeze": {
            "path": _display_path(feature_freeze_path),
            "sha256": _sha256(feature_freeze_path),
            "splits": feature_artifacts,
        },
        "tuning": tuning,
        "template_bank": {
            "path": _display_path(template_bank_path),
            "sha256": _sha256(template_bank_path),
            "count": len(templates),
        },
        "development": {
            "metrics": development_metrics,
            "artifacts": _artifact_summary(development_artifacts),
        },
        "validation": {
            "metrics": validation_metrics,
            "gate": validation_gate,
            "artifacts": _artifact_summary(validation_artifacts),
        },
        "heldout": heldout,
        "truth_access_log": protocol.truth_access_log,
        "determinism": {
            "algorithm": "fixed ordering, integer pixels, deterministic PNG and JSON",
            "core_hashes": current_core_hashes,
            "previous_core_hashes": previous_core_hashes,
            "unchanged_rerun_verified": (
                previous_core_hashes is not None and previous_core_hashes == current_core_hashes
            ),
            "rerun_instruction": (
                "Run the identical command and compare feature, template-bank, and prediction "
                "SHA256 values."
            ),
        },
        "artifacts": {
            "report_json": _display_path(report_path),
            "report_markdown": _display_path(report_md_path),
        },
    }
    _write_json(report_path, report)
    report_md_path.write_text(_markdown_report(report), encoding="utf-8")
    return report


def extract_binary_feature(
    image: Image.Image,
    staff_lines: Sequence[float],
    *,
    variant: str,
) -> BinaryFeature:
    """Normalize staff height/spacing and emit a line-suppressed binary ink map."""
    if variant not in FEATURE_VARIANTS:
        raise ValueError(f"Unknown feature variant: {variant}")
    if len(staff_lines) != 5:
        raise ValueError(f"Expected five staff lines, got {staff_lines}")
    lines = tuple(float(value) for value in staff_lines)
    spacing = sum(b - a for a, b in zip(lines, lines[1:], strict=False)) / 4.0
    if spacing <= 0:
        raise ValueError(f"Invalid staff-line spacing: {staff_lines}")

    gray = image.convert("L")
    pixels = gray.load()
    suppression = "downweight" if "downweight" in variant else "remove"
    radius = max(1, round(spacing * 0.08))
    for line in lines:
        for y in range(max(0, round(line) - radius), min(gray.height, round(line) + radius + 1)):
            for x in range(gray.width):
                value = int(pixels[x, y])
                if suppression == "remove":
                    pixels[x, y] = 255
                else:
                    pixels[x, y] = min(255, round(value * 0.25 + 255 * 0.75))

    top = max(0, math.floor(lines[0] - 2.0 * spacing))
    bottom = min(gray.height, math.ceil(lines[-1] + 2.0 * spacing) + 1)
    cropped = gray.crop((0, top, gray.width, bottom))
    scale = TARGET_STAFF_SPACING / spacing
    normalized_width = max(8, round(cropped.width * scale))
    normalized_height = max(8, round(cropped.height * scale))
    normalized = cropped.resize((normalized_width, normalized_height), Image.Resampling.BILINEAR)
    normalized = _trim_horizontal_whitespace(normalized)
    fixed = normalized.resize((FEATURE_WIDTH, FEATURE_HEIGHT), Image.Resampling.BILINEAR)
    threshold = _otsu_threshold(fixed)
    rows = tuple(
        tuple(1 if fixed.getpixel((x, y)) <= threshold else 0 for x in range(FEATURE_WIDTH))
        for y in range(FEATURE_HEIGHT)
    )
    return BinaryFeature(
        variant=variant,
        width=FEATURE_WIDTH,
        height=FEATURE_HEIGHT,
        rows=rows,
    )


def binary_shift_distance(
    first: BinaryFeature,
    second: BinaryFeature,
    *,
    max_dx: int = 3,
    max_dy: int = 2,
) -> float:
    """Return minimum binary Jaccard distance over small x/y translations."""
    if (first.width, first.height) != (second.width, second.height):
        raise ValueError("Shift distance requires equal feature dimensions")
    best = 1.0
    for dy in range(-max_dy, max_dy + 1):
        for dx in range(-max_dx, max_dx + 1):
            intersection = union = 0
            for y in range(first.height):
                other_y = y + dy
                a = first.bit_rows[y]
                b = second.bit_rows[other_y] if 0 <= other_y < second.height else 0
                if dx > 0:
                    b >>= dx
                elif dx < 0:
                    b = (b << -dx) & ((1 << first.width) - 1)
                intersection += (a & b).bit_count()
                union += (a | b).bit_count()
            distance = 1.0 - intersection / union if union else 0.0
            best = min(best, distance)
    return round(best, 9)


def retrieve_nearest(
    query_identity: Mapping[str, Any],
    query: BinaryFeature,
    templates: Sequence[Template],
    *,
    exclude_self: bool = False,
) -> Retrieval:
    candidates = [
        template
        for template in templates
        if not exclude_self or _identity_key(template.identity) != _identity_key(query_identity)
    ]
    if not candidates:
        raise ValueError("Retrieval has no eligible templates after self exclusion")
    ranked = sorted(
        (
            binary_shift_distance(query, template.feature),
            _identity_key(template.identity),
            template,
        )
        for template in candidates
    )
    return Retrieval(
        source=ranked[0][2],
        distance=ranked[0][0],
        second_distance=ranked[1][0] if len(ranked) > 1 else None,
    )


def prediction_from_retrieval(
    identity: Mapping[str, Any],
    retrieval: Retrieval,
    *,
    threshold: float,
) -> dict[str, Any]:
    accepted = retrieval.distance <= threshold
    confidence = _retrieval_confidence(retrieval, threshold)
    return {
        "identity": dict(identity),
        "notes": copy.deepcopy(list(retrieval.source.notes)) if accepted else [],
        "rests": copy.deepcopy(list(retrieval.source.rests)) if accepted else [],
        "retrieval": {
            "accepted": accepted,
            "abstained": not accepted,
            "source_identity": dict(retrieval.source.identity),
            "distance": retrieval.distance,
            "second_distance": retrieval.second_distance,
            "margin": round(retrieval.margin, 9),
            "confidence": confidence,
            "threshold": round(threshold, 9),
        },
    }


def build_template_bank(
    records: Sequence[FeatureRecord],
    truth_by_key: Mapping[tuple[Any, ...], Mapping[str, Any]],
    variant: str,
) -> list[Template]:
    templates = []
    for record in records:
        truth = truth_by_key[_identity_key(record.identity)]
        templates.append(
            Template(
                identity=record.identity,
                feature=record.features[variant],
                notes=tuple(copy.deepcopy(truth.get("notes") or [])),
                rests=tuple(copy.deepcopy(truth.get("rests") or [])),
            )
        )
    return templates


def tune_retriever(
    development: Sequence[FeatureRecord],
    truth_by_key: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> dict[str, Any]:
    """Select feature variant and rejection threshold using development LOOCV only."""
    variants = []
    best_key: tuple[Any, ...] | None = None
    selected: dict[str, Any] | None = None
    for variant in FEATURE_VARIANTS:
        templates = build_template_bank(development, truth_by_key, variant)
        retrievals = [
            retrieve_nearest(
                record.identity,
                record.features[variant],
                templates,
                exclude_self=True,
            )
            for record in development
        ]
        thresholds = _threshold_candidates([item.distance for item in retrievals])
        threshold_rows = []
        for threshold in thresholds:
            predictions = [
                _prediction_for_record(record, retrieval, threshold=threshold)
                for record, retrieval in zip(development, retrievals, strict=True)
            ]
            metrics = benchmark.evaluate_predictions(
                [truth_by_key[_identity_key(record.identity)] for record in development],
                predictions,
            )
            accepted = sum(bool(row["retrieval"]["accepted"]) for row in predictions)
            summary = metrics["summary"]
            row = {
                "threshold": round(threshold, 9),
                "accepted": accepted,
                "note_f1": summary["note_f1"],
                "exact_measures": summary["exact_measures"],
                "rest_f1": summary["rest_f1"],
            }
            threshold_rows.append(row)
            key = (
                float(summary["note_f1"]),
                int(summary["exact_measures"]),
                float(summary["rest_f1"]),
                -accepted,
                -threshold,
                -FEATURE_VARIANTS.index(variant),
            )
            if best_key is None or key > best_key:
                best_key = key
                selected = {
                    "selected_variant": variant,
                    "selected_threshold": round(threshold, 9),
                    "selected_metrics": metrics,
                }
        variants.append(
            {
                "variant": variant,
                "nearest_retrievals": [
                    {
                        "query_identity": record.identity,
                        "source_identity": retrieval.source.identity,
                        "distance": retrieval.distance,
                        "second_distance": retrieval.second_distance,
                    }
                    for record, retrieval in zip(development, retrievals, strict=True)
                ],
                "threshold_sweep": threshold_rows,
            }
        )
    if selected is None:
        raise ValueError("No development feature variant could be selected")
    return {
        "scope": "development systems 1-2 canonical events only",
        "self_exclusion": True,
        "direct_expected_counts_used": False,
        "objective": "note_f1, exact measures, rest_f1, then fewer accepted retrievals",
        **selected,
        "variants": variants,
    }


def predict_records(
    records: Sequence[FeatureRecord],
    templates: Sequence[Template],
    *,
    variant: str,
    threshold: float,
    exclude_self: bool = False,
) -> tuple[list[dict[str, Any]], list[Retrieval]]:
    retrievals = [
        retrieve_nearest(
            record.identity,
            record.features[variant],
            templates,
            exclude_self=exclude_self,
        )
        for record in records
    ]
    predictions = [
        _prediction_for_record(record, retrieval, threshold=threshold)
        for record, retrieval in zip(records, retrievals, strict=True)
    ]
    return predictions, retrievals


def _prediction_for_record(
    record: FeatureRecord,
    retrieval: Retrieval,
    *,
    threshold: float,
) -> dict[str, Any]:
    prediction = prediction_from_retrieval(record.identity, retrieval, threshold=threshold)
    if bool(record.request.get("allowed_context", {}).get("allow_pickup", False)):
        prediction["notes"] = []
        prediction["rests"] = []
        prediction["retrieval"].update(
            {
                "accepted": False,
                "abstained": True,
                "rejection_reason": "pickup_extent_unknown_at_inference",
            }
        )
    return prediction


def apply_validation_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    note_f1 = float(summary["note_f1"])
    exact_measures = int(summary["exact_measures"])
    passed = note_f1 > VALIDATION_NOTE_F1_FLOOR and exact_measures >= 1
    return {
        "canonical_note_f1_operator": ">",
        "canonical_note_f1_floor": VALIDATION_NOTE_F1_FLOOR,
        "minimum_exact_measures": 1,
        "observed_canonical_note_f1": note_f1,
        "observed_exact_measures": exact_measures,
        "passed": passed,
        "failure_action": None if passed else "do_not_open_heldout_truth",
    }


def freeze_split_features(
    requests: Sequence[Mapping[str, Any]],
    *,
    out_dir: Path,
    split: str,
    output_dir: Path,
    requests_path: Path,
) -> tuple[list[FeatureRecord], dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[FeatureRecord] = []
    manifest_rows = []
    for request in requests:
        identity = dict(request["identity"])
        features: dict[str, BinaryFeature] = {}
        feature_files = {}
        for variant in FEATURE_VARIANTS:
            input_kind = "raw" if variant.startswith("raw_") else "staff"
            image_record = request["images"][input_kind]
            image_path = _resolve_image(out_dir, str(image_record["path_relative_to_out"]))
            if _sha256(image_path) != str(image_record["sha256"]):
                raise ValueError(f"Request image hash mismatch: {image_path}")
            line_key = "raw_staff_lines_y_px" if input_kind == "raw" else "staff_crop_lines_y_px"
            with Image.open(image_path) as source:
                feature = extract_binary_feature(
                    source,
                    request["staff_geometry"][line_key],
                    variant=variant,
                )
            feature_path = output_dir / _feature_filename(identity, variant)
            feature.image().save(feature_path)
            features[variant] = feature
            feature_files[variant] = {
                "path": _display_path(feature_path),
                "sha256": _sha256(feature_path),
                "width": feature.width,
                "height": feature.height,
                "ink_pixels": feature.ink_pixels,
            }
        records.append(FeatureRecord(request=dict(request), features=features))
        manifest_rows.append(
            {
                "split": split,
                "identity": identity,
                "request_image_hashes": {
                    kind: request["images"][kind]["sha256"] for kind in ("raw", "staff")
                },
                "features": feature_files,
            }
        )
    manifest_path = output_dir / "manifest.jsonl"
    _write_jsonl(manifest_path, manifest_rows)
    return records, {
        "request_path": _display_path(requests_path),
        "requests_sha256": _sha256(requests_path),
        "manifest_path": _display_path(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "targets": len(records),
    }


def freeze_split_predictions(
    split: str,
    records: Sequence[FeatureRecord],
    predictions: Sequence[Mapping[str, Any]],
    retrievals: Sequence[Retrieval],
    *,
    output_dir: Path,
    out_dir: Path,
    template_records: Sequence[FeatureRecord],
) -> SplitArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.sealed.jsonl"
    retrievals_path = output_dir / "retrievals.jsonl"
    _write_jsonl(predictions_path, predictions)
    _write_jsonl(
        retrievals_path,
        [
            {
                "identity": prediction["identity"],
                **prediction["retrieval"],
            }
            for prediction in predictions
        ],
    )
    by_key = {_identity_key(record.identity): record for record in template_records}
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay_paths = []
    for record, prediction, retrieval in zip(records, predictions, retrievals, strict=True):
        source_record = by_key[_identity_key(retrieval.source.identity)]
        path = overlay_dir / _overlay_filename(record.identity)
        _write_retrieval_overlay(record, source_record, prediction, out_dir=out_dir, path=path)
        overlay_paths.append(path)
    contact_sheet_path = output_dir / "contact_sheet.png"
    _write_contact_sheet(overlay_paths, contact_sheet_path)
    return SplitArtifacts(
        split=split,
        predictions_path=predictions_path,
        retrievals_path=retrievals_path,
        contact_sheet_path=contact_sheet_path,
        predictions_sha256=_sha256(predictions_path),
        retrievals_sha256=_sha256(retrievals_path),
        contact_sheet_sha256=_sha256(contact_sheet_path),
    )


def _write_retrieval_overlay(
    query: FeatureRecord,
    source: FeatureRecord,
    prediction: Mapping[str, Any],
    *,
    out_dir: Path,
    path: Path,
) -> None:
    query_image = _open_request_staff(query.request, out_dir)
    source_image = _open_request_staff(source.request, out_dir)
    panel_width = 360
    panel_height = 170
    query_image.thumbnail((panel_width, panel_height), Image.Resampling.LANCZOS)
    source_image.thumbnail((panel_width, panel_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (panel_width * 2 + 30, panel_height + 54), "white")
    canvas.paste(query_image, (10, 36))
    canvas.paste(source_image, (panel_width + 20, 36))
    draw = ImageDraw.Draw(canvas)
    retrieval = prediction["retrieval"]
    query_text = _short_identity(query.identity)
    source_text = _short_identity(source.identity)
    draw.text((10, 8), f"query {query_text}", fill="black")
    draw.text((panel_width + 20, 8), f"source {source_text}", fill="black")
    status = "MATCH" if retrieval["accepted"] else "ABSTAIN"
    draw.text(
        (10, panel_height + 38),
        f"{status} d={retrieval['distance']:.4f} c={retrieval['confidence']:.4f}",
        fill="#166534" if retrieval["accepted"] else "#991b1b",
    )
    canvas.save(path)


def _write_contact_sheet(paths: Sequence[Path], path: Path) -> None:
    images = []
    for source_path in paths:
        with Image.open(source_path) as source:
            image = source.convert("RGB")
            image.thumbnail((370, 120), Image.Resampling.LANCZOS)
            images.append(image.copy())
    columns = 2
    cell_width = 380
    cell_height = 130
    rows = max(1, math.ceil(len(images) / columns))
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    for index, image in enumerate(images):
        sheet.paste(image, ((index % columns) * cell_width, (index // columns) * cell_height))
    sheet.save(path)


def _artifact_summary(artifacts: SplitArtifacts) -> dict[str, Any]:
    return {
        "predictions_path": _display_path(artifacts.predictions_path),
        "predictions_sha256": artifacts.predictions_sha256,
        "retrievals_path": _display_path(artifacts.retrievals_path),
        "retrievals_sha256": artifacts.retrievals_sha256,
        "contact_sheet_path": _display_path(artifacts.contact_sheet_path),
        "contact_sheet_sha256": artifacts.contact_sheet_sha256,
    }


def _retrieval_confidence(retrieval: Retrieval, threshold: float) -> float:
    if threshold <= 0:
        return 0.0
    proximity = max(0.0, 1.0 - retrieval.distance / threshold)
    denominator = max(retrieval.second_distance or 0.0, 1e-9)
    margin = retrieval.margin / denominator
    return round(min(1.0, 0.75 * proximity + 0.25 * margin), 9)


def _threshold_candidates(distances: Sequence[float]) -> list[float]:
    unique = sorted(set(float(value) for value in distances))
    if not unique:
        return [-1.0]
    candidates = [-1.0]
    candidates.extend(unique)
    return candidates


def _trim_horizontal_whitespace(image: Image.Image) -> Image.Image:
    threshold = _otsu_threshold(image)
    active = []
    minimum_ink = max(1, round(image.height * 0.01))
    for x in range(image.width):
        ink = sum(image.getpixel((x, y)) <= threshold for y in range(image.height))
        if ink >= minimum_ink:
            active.append(x)
    if not active:
        return image
    pad = max(2, round(image.height * 0.04))
    left = max(0, active[0] - pad)
    right = min(image.width, active[-1] + pad + 1)
    return image.crop((left, 0, right, image.height))


def _otsu_threshold(image: Image.Image) -> int:
    histogram = image.histogram()[:256]
    total = sum(histogram)
    if total == 0:
        return 127
    weighted_sum = sum(index * count for index, count in enumerate(histogram))
    background_weight = 0
    background_sum = 0
    best_variance = -1.0
    best_threshold = 127
    for threshold, count in enumerate(histogram):
        background_weight += count
        if not background_weight:
            continue
        foreground_weight = total - background_weight
        if not foreground_weight:
            break
        background_sum += threshold * count
        background_mean = background_sum / background_weight
        foreground_mean = (weighted_sum - background_sum) / foreground_weight
        variance = background_weight * foreground_weight * (background_mean - foreground_mean) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold
    return min(235, max(80, best_threshold))


def _open_request_staff(request: Mapping[str, Any], out_dir: Path) -> Image.Image:
    path = _resolve_image(out_dir, str(request["images"]["staff"]["path_relative_to_out"]))
    with Image.open(path) as source:
        return source.convert("RGB")


def _resolve_image(out_dir: Path, relative: str) -> Path:
    path = Path(relative)
    candidate = path if path.is_absolute() else out_dir / path
    if not candidate.exists():
        raise ValueError(f"Request image does not exist: {candidate}")
    return candidate.resolve()


def _validate_matching_identities(
    records: Sequence[FeatureRecord],
    truth_by_key: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> None:
    request_keys = {_identity_key(record.identity) for record in records}
    if request_keys != set(truth_by_key):
        raise ValueError("Development request and truth identities do not match")


def _identity_key(identity: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(identity["slug"]),
        int(identity["system_index"]),
        int(identity["system_measure_index"]),
        int(identity["global_measure_index"]),
    )


def _short_identity(identity: Mapping[str, Any]) -> str:
    return (
        f"S{int(identity['system_index'])}M{int(identity['system_measure_index'])}"
        f"/G{int(identity['global_measure_index'])}"
    )


def _feature_filename(identity: Mapping[str, Any], variant: str) -> str:
    return (
        f"system_{int(identity['system_index']):03d}_"
        f"measure_{int(identity['system_measure_index']):03d}_{variant}.png"
    )


def _overlay_filename(identity: Mapping[str, Any]) -> str:
    return (
        f"system_{int(identity['system_index']):03d}_"
        f"measure_{int(identity['system_measure_index']):03d}.png"
    )


def _markdown_report(report: Mapping[str, Any]) -> str:
    development = report["development"]["metrics"]["summary"]
    validation = report["validation"]["metrics"]["summary"]
    gate = report["validation"]["gate"]
    heldout = report["heldout"]
    lines = [
        "# Repeated-Measure Retrieval Transcriber",
        "",
        report["limitation"],
        "",
        "## Frozen Method",
        "",
        f"- Feature variant: `{report['tuning']['selected_variant']}`",
        f"- Rejection threshold: `{report['tuning']['selected_threshold']:.9f}`",
        f"- Development templates: `{report['template_bank']['count']}`",
        "- Staff height and spacing normalized; staff lines removed or downweighted.",
        "- Binary ink is compared with a small shift-tolerant Jaccard distance.",
        "",
        "## Development LOOCV",
        "",
        f"- Canonical note F1: `{development['note_f1']:.6f}`",
        f"- Exact measures: `{development['exact_measures']}/{development['targets']}`",
        "",
        "## Validation S7+S8",
        "",
        f"- Canonical note F1: `{validation['note_f1']:.6f}`",
        f"- Exact measures: `{validation['exact_measures']}/{validation['targets']}`",
        f"- Gate: note F1 > `{gate['canonical_note_f1_floor']}` and at least one exact; "
        f"passed `{gate['passed']}`",
        f"- Predictions: `{report['validation']['artifacts']['predictions_path']}`",
        f"- Contact sheet: `{report['validation']['artifacts']['contact_sheet_path']}`",
        "",
        "Validation predictions, retrieval diagnostics, overlays, and contact sheet were "
        "persisted before validation truth was opened.",
        "",
        "## Heldout S3",
        "",
        f"- Status: `{heldout['status']}`",
    ]
    if heldout["metrics"] is not None:
        summary = heldout["metrics"]["summary"]
        lines.extend(
            [
                f"- Canonical note F1: `{summary['note_f1']:.6f}`",
                f"- Exact measures: `{summary['exact_measures']}/{summary['targets']}`",
                f"- Predictions: `{heldout['artifacts']['predictions_path']}`",
            ]
        )
    else:
        lines.append("- Heldout truth was not opened because the validation gate failed.")
    return "\n".join(lines) + "\n"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _previous_core_hashes(report_path: Path) -> dict[str, str] | None:
    if not report_path.exists():
        return None
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    hashes = payload.get("determinism", {}).get("core_hashes")
    if not isinstance(hashes, dict):
        return None
    return {str(key): str(value) for key, value in hashes.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
