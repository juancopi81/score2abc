"""Bootstrap notehead selection from development-only ordered event labels.

The experiment freezes blind cap-24 proposals for every benchmark split before
opening development truth. Ordered development onset groups are then aligned to
the proposals with a monotonic dynamic program. Independent coordinate labels
gate the resulting spatial pseudo labels before any selector is trained.

The original failed arm is preserved. A preregistered pickup-aware follow-up can
use exact promoted reviews plus filtered non-pickup pseudo positives. Validation
predictions are hashed before validation truth; heldout predictions are created
only after a fixed validation gate passes. Onset and duration remain out of scope.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable, Mapping, Sequence

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_vlm_notehead_candidates as detector  # noqa: E402
from scripts.experiments import spike_notehead_candidate_classifier as classifier  # noqa: E402
from scripts.run_vlm_notehead_localization_spike import treble_pitch_for_y  # noqa: E402

DEFAULT_SLUG = "jaime-llanos_12_aviador_pasillo_fulgencio-garcia"
BENCHMARK_SUBDIR = Path("vlm_melody_event_benchmark")
OUTPUT_SUBDIR = Path("experiments/notehead_sequence_bootstrap")
SPLITS = ("development", "validation", "heldout")
EVALUATION_SPLITS = ("validation", "heldout")
AUDIT_SYSTEM = 1
AUDIT_MEASURES = (1, 2, 3, 4)
REVIEW_MEASURES = (1, 2, 3, 4)
PSEUDO_CALIBRATION_MEASURES = (2, 3, 4)
UNSUPPORTED_PSEUDO_MEASURES = ((1, 1),)
CANDIDATE_CAP = 24
GATE_PRECISION = 0.85
GATE_RECALL = 0.85
HYBRID_CALIBRATION_PRECISION = 0.90
HYBRID_CALIBRATION_RECALL = 0.75
PSEUDO_RULE_MAX_COSTS = (0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00)
PSEUDO_RULE_MAX_PITCH_ERRORS = (0, 1, 2)
VALIDATION_BASELINE_SYSTEM = 7
VALIDATION_BASELINE_ORDERED_PITCH = 0.130435
VALIDATION_BASELINE_COUNT_MAE = 1.714286
VALIDATION_ORDERED_PITCH_MARGIN = 0.05
VALIDATION_REQUIRED_ORDERED_PITCH = round(
    VALIDATION_BASELINE_ORDERED_PITCH + VALIDATION_ORDERED_PITCH_MARGIN, 6
)
VALIDATION_BASELINE_PATH = (
    "experiments/vlm_system_transcription/"
    "openai-gpt56sol-medium-system-contact-validation-s7/evaluation.json"
)
VALIDATION_BASELINE_SHA256 = "9453236ce2ba0b59c8c2db712a3d029d8f1b5a5490675713bae8ab2a95b94075"
FROZEN_CROSS_SYSTEM_RULE_MAX_COST = 0.30
FROZEN_CROSS_SYSTEM_RULE_MAX_PITCH_ERROR = 0
FROZEN_RULE_CALIBRATION_SHA256 = "2be218fc7f8a8f8be7c2fa792f86929ea2e3b498e0e2f0506c6e6bb59038ef31"
CROSS_SYSTEM_TRAIN_SYSTEM = 7
CROSS_SYSTEM_MODEL_SELECTION_SYSTEM = 8
CROSS_SYSTEM_MIN_ACCEPTED_LABELS = 7
CROSS_SYSTEM_MIN_PSEUDO_COVERAGE = 0.20
S8_GATE_ORDERED_PITCH = 0.20
S8_GATE_COUNT_MAE = 1.5
S3_COORDINATION_EVALUATION_ALLOWED = False
S3_COORDINATION_STATUS = (
    "opened_once_by_another_preregistered_arm; unavailable as an independent final test"
)
CHORD_X_TOLERANCE_STAFF_SPACES = 0.6
PITCH_RE = re.compile(r"^([A-G])([#b]?)(-?\d+)$")
FEATURE_NAMES = (
    classifier.DETECTOR_FEATURES + classifier.PATCH_FEATURES + classifier.GEOMETRY_FEATURES
)
TruthReader = Callable[[Path], list[dict[str, Any]]]


@dataclass(frozen=True)
class OnsetGroup:
    onset_beats: float
    pitches: tuple[int, ...]


@dataclass(frozen=True)
class PseudoAcceptanceRule:
    max_match_cost: float
    max_absolute_pitch_error: int

    def accepts(self, assignment: Mapping[str, Any]) -> bool:
        return (
            assignment.get("status") == "aligned"
            and float(assignment["match_cost"]) <= self.max_match_cost
            and all(
                int(pair["absolute_pitch_error"]) <= self.max_absolute_pitch_error
                for pair in assignment["pairs"]
            )
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "max_match_cost": self.max_match_cost,
            "max_absolute_pitch_error": self.max_absolute_pitch_error,
        }


@dataclass(frozen=True)
class LinearSelector:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    weights: tuple[float, ...]
    intercept: float
    threshold: float

    def score(self, candidate: Mapping[str, Any]) -> float:
        features = candidate["features"]
        standardized = [
            (float(features[name]) - mean) / scale
            for name, mean, scale in zip(self.feature_names, self.means, self.scales, strict=True)
        ]
        return _sigmoid(self.intercept + _dot(self.weights, standardized))


@dataclass(frozen=True)
class PitchCalibrator:
    global_offset: int
    offsets_by_letter: dict[str, int]

    def predict(self, candidate: Mapping[str, Any]) -> int:
        auto_pitch = str(candidate["auto_pitch"])
        offset = self.offsets_by_letter.get(auto_pitch[0], self.global_offset)
        return max(0, min(127, int(candidate["auto_pitch_midi"]) + offset))


class TruthAccessGuard:
    """Enforce candidate/prediction freeze points before truth reads."""

    def __init__(self, split_names: Sequence[str] = SPLITS) -> None:
        self.split_names = tuple(split_names)
        self.candidate_hashes: dict[str, str] = {}
        self.prediction_hashes: dict[str, str] = {}
        self.validation_gate_passed: bool | None = None
        self.access_log: list[dict[str, Any]] = []

    def register_candidate_file(self, split_name: str, path: Path) -> str:
        digest = _sha256(path)
        self.candidate_hashes[split_name] = digest
        return digest

    def read_development_truth(self, path: Path, reader: TruthReader) -> list[dict[str, Any]]:
        missing = sorted(set(self.split_names) - self.candidate_hashes.keys())
        if missing:
            raise RuntimeError(
                "Development truth cannot be read before every candidate file is hashed: "
                f"{missing}"
            )
        rows = reader(path)
        self.access_log.append(
            {
                "truth": "development",
                "path": str(path),
                "after_candidate_hashes": dict(sorted(self.candidate_hashes.items())),
            }
        )
        return rows

    def register_prediction_file(self, split_name: str, path: Path) -> str:
        if split_name not in EVALUATION_SPLITS:
            raise ValueError(f"Predictions are not expected for split: {split_name}")
        digest = _sha256(path)
        self.prediction_hashes[split_name] = digest
        return digest

    def read_validation_truth(self, path: Path, reader: TruthReader) -> list[dict[str, Any]]:
        if "validation" not in self.prediction_hashes:
            raise RuntimeError("Validation truth requires a persisted validation prediction hash")
        rows = reader(path)
        self.access_log.append(
            {
                "truth": "validation",
                "path": str(path),
                "after_prediction_hashes": dict(sorted(self.prediction_hashes.items())),
            }
        )
        return rows

    def record_shared_validation_truth_read(self, path: Path) -> None:
        if "validation" not in self.prediction_hashes:
            raise RuntimeError("Validation truth requires a persisted validation prediction hash")
        self.access_log.append(
            {
                "truth": "validation",
                "path": str(path),
                "after_prediction_hashes": dict(sorted(self.prediction_hashes.items())),
                "shared_with_cross_system_arm": True,
            }
        )

    def register_validation_gate(self, passed: bool) -> None:
        self.validation_gate_passed = passed

    def read_heldout_truth(self, path: Path, reader: TruthReader) -> list[dict[str, Any]]:
        if self.validation_gate_passed is not True:
            raise RuntimeError("Heldout truth requires a passing validation gate")
        if "heldout" not in self.prediction_hashes:
            raise RuntimeError("Heldout truth requires a persisted heldout prediction hash")
        rows = reader(path)
        self.access_log.append(
            {
                "truth": "heldout",
                "path": str(path),
                "after_prediction_hashes": dict(sorted(self.prediction_hashes.items())),
                "after_validation_gate": True,
            }
        )
        return rows

    def read_evaluation_truth(
        self,
        truth_paths: Mapping[str, Path],
        reader: TruthReader,
    ) -> dict[str, list[dict[str, Any]]]:
        missing = sorted(set(EVALUATION_SPLITS) - self.prediction_hashes.keys())
        if missing:
            raise RuntimeError(
                "Evaluation truth cannot be read before every prediction file is hashed: "
                f"{missing}"
            )
        rows_by_split: dict[str, list[dict[str, Any]]] = {}
        for split_name in EVALUATION_SPLITS:
            path = truth_paths[split_name]
            rows_by_split[split_name] = reader(path)
            self.access_log.append(
                {
                    "truth": split_name,
                    "path": str(path),
                    "after_prediction_hashes": dict(sorted(self.prediction_hashes.items())),
                }
            )
        return rows_by_split


class CrossSystemTruthGuard:
    """Enforce S7-train, S8-model-selection, then sealed-S3 ordering."""

    def __init__(self) -> None:
        self.candidate_hashes: dict[str, str] = {}
        self.s8_prediction_hash: str | None = None
        self.s8_gate_passed: bool | None = None
        self.heldout_prediction_hash: str | None = None
        self.access_log: list[dict[str, Any]] = []

    def register_candidate_file(self, split_name: str, path: Path) -> str:
        digest = _sha256(path)
        self.candidate_hashes[split_name] = digest
        return digest

    def read_s7_training_truth(
        self,
        path: Path,
        reader: Callable[[Path, int], list[dict[str, Any]]],
        *,
        row_count: int,
    ) -> list[dict[str, Any]]:
        missing = sorted(set(SPLITS) - self.candidate_hashes.keys())
        if missing:
            raise RuntimeError(f"S7 training truth requires all candidate hashes: {missing}")
        rows = reader(path, row_count)
        systems = {int(row["identity"]["system_index"]) for row in rows}
        if systems != {CROSS_SYSTEM_TRAIN_SYSTEM}:
            raise ValueError(f"S7 training prefix contains unexpected systems: {systems}")
        self.access_log.append(
            {
                "truth": "system7_adaptation_training",
                "path": str(path),
                "rows_read": len(rows),
                "system8_truth_parsed": False,
                "after_candidate_hashes": dict(sorted(self.candidate_hashes.items())),
            }
        )
        return rows

    def register_s8_prediction_file(self, path: Path) -> str:
        self.s8_prediction_hash = _sha256(path)
        return self.s8_prediction_hash

    def read_validation_truth_after_s8_prediction(
        self,
        path: Path,
        reader: TruthReader,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if self.s8_prediction_hash is None:
            raise RuntimeError("S8 truth requires a persisted S8 prediction hash")
        rows = reader(path)
        s7_rows, s8_rows = partition_cross_system_validation_rows(rows)
        self.access_log.append(
            {
                "truth": "system8_model_selection",
                "path": str(path),
                "after_s8_prediction_hash": self.s8_prediction_hash,
                "system7_training_rows": len(s7_rows),
                "system8_evaluation_rows": len(s8_rows),
            }
        )
        return s7_rows, s8_rows

    def register_s8_gate(self, passed: bool) -> None:
        self.s8_gate_passed = passed

    def register_heldout_prediction_file(self, path: Path) -> str:
        if self.s8_gate_passed is not True:
            raise RuntimeError("Heldout predictions require a passing S8 gate")
        self.heldout_prediction_hash = _sha256(path)
        return self.heldout_prediction_hash

    def read_heldout_truth(self, path: Path, reader: TruthReader) -> list[dict[str, Any]]:
        if self.s8_gate_passed is not True:
            raise RuntimeError("Heldout truth requires a passing S8 gate")
        if self.heldout_prediction_hash is None:
            raise RuntimeError("Heldout truth requires a persisted heldout prediction hash")
        rows = reader(path)
        systems = {int(row["identity"]["system_index"]) for row in rows}
        if systems != {3}:
            raise ValueError(f"Heldout truth contains unexpected systems: {systems}")
        self.access_log.append(
            {
                "truth": "system3_sealed_final_test",
                "path": str(path),
                "after_heldout_prediction_hash": self.heldout_prediction_hash,
                "after_s8_gate": True,
            }
        )
        return rows


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = run_experiment(args.out_dir, slug=args.slug)
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(args.out_dir / OUTPUT_SUBDIR / "report.json")
    print(args.out_dir / OUTPUT_SUBDIR / "report.md")
    return 0 if str(report["status"]).startswith("completed") else 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", type=Path, default=Path("out"))
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    return parser


def run_experiment(
    out_dir: Path,
    *,
    slug: str = DEFAULT_SLUG,
    truth_reader: TruthReader = None,
) -> dict[str, Any]:
    truth_reader = truth_reader or _read_jsonl
    benchmark_dir = out_dir / slug / BENCHMARK_SUBDIR
    output_dir = out_dir / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    hybrid_dir = output_dir / "pickup_aware_hybrid"
    if hybrid_dir.exists():
        shutil.rmtree(hybrid_dir)
    cross_system_dir = output_dir / "cross_system_weak_supervision"
    if cross_system_dir.exists():
        shutil.rmtree(cross_system_dir)
    guard = TruthAccessGuard()

    requests_by_split = {
        split_name: _read_jsonl(benchmark_dir / split_name / "requests.jsonl")
        for split_name in SPLITS
    }
    proposals_by_split: dict[str, list[dict[str, Any]]] = {}
    proposal_manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "notehead_sequence_bootstrap_candidate_manifest",
        "candidate_cap": CANDIDATE_CAP,
        "generation_uses_ground_truth": False,
        "splits": {},
    }
    for split_name in SPLITS:
        proposals = [
            generate_measure_proposals(out_dir, request)
            for request in requests_by_split[split_name]
        ]
        proposals_by_split[split_name] = proposals
        path = output_dir / "candidate_proposals" / f"{split_name}.jsonl"
        _write_jsonl(path, proposals)
        digest = guard.register_candidate_file(split_name, path)
        proposal_manifest["splits"][split_name] = {
            "path": _relative_to_out(path, out_dir),
            "sha256": digest,
            "measure_count": len(proposals),
            "candidate_count": sum(len(row["candidates"]) for row in proposals),
        }
    _write_json(output_dir / "candidate_proposals" / "manifest.json", proposal_manifest)

    development_truth = guard.read_development_truth(
        benchmark_dir / "development" / "truth.jsonl", truth_reader
    )
    pseudo_rows = build_development_pseudo_labels(
        development_truth,
        proposals_by_split["development"],
    )
    pseudo_path = output_dir / "pseudo_labels" / "development.jsonl"
    _write_jsonl(pseudo_path, pseudo_rows)

    audit = audit_pseudo_labels(pseudo_rows, proposals_by_split["development"])
    gate = bootstrap_gate(audit)
    _write_json(output_dir / "pseudo_label_audit.json", {"audit": audit, "gate": gate})

    original_arm = {
        "name": "original_canonical_sequence_pseudo_label_bootstrap",
        "status": "stopped_at_pseudo_label_gate",
        "pseudo_label_path": _relative_to_out(pseudo_path, out_dir),
        "pseudo_label_sha256": _sha256(pseudo_path),
        "audit": audit,
        "gate": gate,
        "interpretation": (
            "Original arm remains failed; its gate is not relaxed or reused by the hybrid arm."
        ),
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pickup_aware_hybrid_notehead_sequence_bootstrap",
        "scope": {
            "slug": slug,
            "fitting_split": "development",
            "fitting_systems": [1, 2],
            "candidate_cap": CANDIDATE_CAP,
            "validation_truth_used_for_fitting": False,
            "heldout_truth_used_for_fitting": False,
            "onset_prediction": "out_of_scope",
            "duration_prediction": "out_of_scope",
        },
        "candidate_manifest": proposal_manifest,
        "pseudo_label_path": _relative_to_out(pseudo_path, out_dir),
        "pseudo_label_sha256": _sha256(pseudo_path),
        "audit": audit,
        "gate": gate,
        "original_failed_arm": original_arm,
        "hybrid_arm": None,
        "cross_system_arm": None,
        "truth_access_log": guard.access_log,
        "status": "hybrid_not_started",
        "model": None,
        "predictions": {},
        "metrics": {},
    }

    exact_reviews = load_exact_review_labels(
        proposals_by_split["development"],
        slug=slug,
    )
    exact_review_path = hybrid_dir / "exact_review_labels.jsonl"
    _write_jsonl(exact_review_path, exact_reviews)
    rule, calibration = calibrate_pseudo_acceptance_rule(pseudo_rows, exact_reviews)
    calibration_path = hybrid_dir / "pseudo_acceptance_calibration.json"
    _write_json(calibration_path, calibration)
    hybrid_arm: dict[str, Any] = {
        "name": "pickup_aware_hybrid_bootstrap",
        "hypothesis_status": "preregistered_follow_up",
        "pickup_policy": {
            "system_1_measure_1": "unsupported_by_canonical_sequence_pseudo_labeler",
            "failed_original_pseudo_assignments_used": False,
            "exact_review_labels_used_for_supervision": True,
        },
        "exact_review_labels": {
            "path": _relative_to_out(exact_review_path, out_dir),
            "sha256": _sha256(exact_review_path),
            "measure_count": len(exact_reviews),
            "positive_count": sum(
                int(label["label"]) for row in exact_reviews for label in row["labels"]
            ),
            "negative_count": sum(
                int(not label["label"]) for row in exact_reviews for label in row["labels"]
            ),
        },
        "pseudo_acceptance_calibration": {
            "path": _relative_to_out(calibration_path, out_dir),
            "sha256": _sha256(calibration_path),
            **calibration,
        },
        "accepted_pseudo_labels": None,
        "validation_gate": _validation_gate_preregistration(),
        "status": "stopped_at_confidence_filter_gate",
    }
    report["hybrid_arm"] = hybrid_arm
    if rule is None:
        report["status"] = "stopped_at_hybrid_confidence_filter_gate"
        report["truth_access_log"] = guard.access_log
        _write_reports(output_dir, report)
        return report

    accepted_pseudo_rows = apply_pseudo_acceptance_rule(pseudo_rows, rule)
    accepted_pseudo_path = hybrid_dir / "accepted_pseudo_labels.jsonl"
    _write_jsonl(accepted_pseudo_path, accepted_pseudo_rows)
    accepted_count = sum(len(row["accepted_labels"]) for row in accepted_pseudo_rows)
    hybrid_arm["accepted_pseudo_labels"] = {
        "path": _relative_to_out(accepted_pseudo_path, out_dir),
        "sha256": _sha256(accepted_pseudo_path),
        "measure_count": len(accepted_pseudo_rows),
        "accepted_positive_count": accepted_count,
        "rule": rule.to_json(),
        "eligible_scope": "S1M5-8 and S2M1-9 only",
        "unaccepted_candidates_treated_as_negative": False,
    }

    selector, pitch_calibrator, max_chord_size, training_summary = train_hybrid_models(
        exact_reviews,
        accepted_pseudo_rows,
        proposals_by_split["development"],
    )
    model_payload = _model_payload(selector, pitch_calibrator, max_chord_size)
    model_payload["kind"] = "pure_python_pickup_aware_hybrid_bootstrap_model"
    model_payload["training"] = training_summary
    model_path = hybrid_dir / "model.json"
    _write_json(model_path, model_payload)
    report["model"] = {
        "path": _relative_to_out(model_path, out_dir),
        "sha256": _sha256(model_path),
        **model_payload["summary"],
        "training": training_summary,
    }

    prediction_manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pickup_aware_hybrid_prediction_manifest",
        "validation_truth_opened_before_validation_prediction_hash": False,
        "heldout_truth_opened_before_passing_validation_gate": False,
        "splits": {},
    }
    validation_predictions = [
        predict_measure(
            proposal,
            selector=selector,
            pitch_calibrator=pitch_calibrator,
            max_chord_size=max_chord_size,
        )
        for proposal in proposals_by_split["validation"]
    ]
    validation_prediction_path = hybrid_dir / "predictions" / "validation.jsonl"
    _write_jsonl(validation_prediction_path, validation_predictions)
    validation_prediction_hash = guard.register_prediction_file(
        "validation", validation_prediction_path
    )
    prediction_manifest["splits"]["validation"] = {
        "path": _relative_to_out(validation_prediction_path, out_dir),
        "sha256": validation_prediction_hash,
        "measure_count": len(validation_predictions),
        "systems": [7, 8],
    }
    prediction_manifest_path = hybrid_dir / "predictions" / "manifest.json"
    _write_json(prediction_manifest_path, prediction_manifest)
    baseline_path = out_dir / VALIDATION_BASELINE_PATH
    if _sha256(baseline_path) != VALIDATION_BASELINE_SHA256:
        raise ValueError(f"Preregistered validation baseline hash drift: {baseline_path}")

    frozen_cross_rule = PseudoAcceptanceRule(
        FROZEN_CROSS_SYSTEM_RULE_MAX_COST,
        FROZEN_CROSS_SYSTEM_RULE_MAX_PITCH_ERROR,
    )
    if rule != frozen_cross_rule or _sha256(calibration_path) != FROZEN_RULE_CALIBRATION_SHA256:
        raise ValueError("Frozen S1M2-4 confidence rule or calibration artifact drifted")
    cross_guard = CrossSystemTruthGuard()
    for split_name in SPLITS:
        cross_guard.register_candidate_file(
            split_name,
            output_dir / "candidate_proposals" / f"{split_name}.jsonl",
        )
    validation_truth_path = benchmark_dir / "validation" / "truth.jsonl"
    s7_truth_for_training = cross_guard.read_s7_training_truth(
        validation_truth_path,
        _read_jsonl_prefix,
        row_count=7,
    )
    s7_proposals = [
        row
        for row in proposals_by_split["validation"]
        if int(row["identity"]["system_index"]) == CROSS_SYSTEM_TRAIN_SYSTEM
    ]
    s8_proposals = [
        row
        for row in proposals_by_split["validation"]
        if int(row["identity"]["system_index"]) == CROSS_SYSTEM_MODEL_SELECTION_SYSTEM
    ]
    if len(s7_proposals) != 7 or len(s8_proposals) != 7:
        raise ValueError(
            f"Expected seven S7 and seven S8 proposals, got {len(s7_proposals)}/"
            f"{len(s8_proposals)}"
        )
    s7_pseudo_rows = build_development_pseudo_labels(
        s7_truth_for_training,
        s7_proposals,
        source_split="system7_adaptation_training",
    )
    s7_pseudo_path = cross_system_dir / "s7_sequence_pseudo_labels.jsonl"
    _write_jsonl(s7_pseudo_path, s7_pseudo_rows)
    accepted_s7_rows = apply_frozen_rule_to_s7(s7_pseudo_rows, frozen_cross_rule)
    accepted_s7_path = cross_system_dir / "accepted_s7_pseudo_labels.jsonl"
    _write_jsonl(accepted_s7_path, accepted_s7_rows)
    s7_coverage = evaluate_s7_pseudo_coverage(accepted_s7_rows, s7_truth_for_training)
    s7_coverage_path = cross_system_dir / "s7_pseudo_coverage_gate.json"
    _write_json(s7_coverage_path, s7_coverage)
    cross_system_arm: dict[str, Any] = {
        "name": "cross_system_weak_supervision_s7_to_s8",
        "hypothesis_status": "explicitly_new_preregistered_final_arm",
        "evidence_roles": {
            "system7": "adaptation_training",
            "system8": "model_selection_evidence_not_untouched_validation",
            "system3": S3_COORDINATION_STATUS,
        },
        "frozen_confidence_rule": {
            **frozen_cross_rule.to_json(),
            "calibrated_on": "independently reviewed S1M2-4",
            "calibration_sha256": FROZEN_RULE_CALIBRATION_SHA256,
            "recalibrated_on_system7": False,
        },
        "training_contract": {
            "exact_labels": "S1M1-4 promoted review positives and negatives",
            "weak_labels": "accepted S7 pseudo positives only",
            "unlabeled_s7_candidates_treated_as_negative": False,
            "system8_truth_or_expected_counts_used": False,
            "rhythm_duration_claim": False,
        },
        "s7_pseudo_labels": {
            "all_path": _relative_to_out(s7_pseudo_path, out_dir),
            "all_sha256": _sha256(s7_pseudo_path),
            "accepted_path": _relative_to_out(accepted_s7_path, out_dir),
            "accepted_sha256": _sha256(accepted_s7_path),
            "coverage_gate_path": _relative_to_out(s7_coverage_path, out_dir),
            "coverage_gate_sha256": _sha256(s7_coverage_path),
            **s7_coverage,
        },
        "s8_gate": _s8_gate_preregistration(),
        "model": None,
        "predictions": {},
        "metrics": {},
        "heldout": {
            "status": "not_reached",
            "independent_evaluation_available": S3_COORDINATION_EVALUATION_ALLOWED,
            "coordination_status": S3_COORDINATION_STATUS,
        },
        "truth_access_log": cross_guard.access_log,
        "status": "stopped_at_s7_pseudo_coverage_gate",
    }
    report["cross_system_arm"] = cross_system_arm
    if not s7_coverage["passed"]:
        report["status"] = "stopped_at_cross_system_s7_pseudo_coverage_gate"
        report["truth_access_log"] = guard.access_log
        _write_reports(output_dir, report)
        return report

    (
        cross_selector,
        cross_pitch_calibrator,
        cross_max_chord_size,
        cross_training_summary,
    ) = train_hybrid_models(
        exact_reviews,
        accepted_s7_rows,
        [*proposals_by_split["development"], *s7_proposals],
        pseudo_source_description="accepted S7 adaptation pseudo positives",
    )
    cross_model_payload = _model_payload(
        cross_selector,
        cross_pitch_calibrator,
        cross_max_chord_size,
    )
    cross_model_payload["kind"] = "pure_python_cross_system_weak_supervision_model"
    cross_model_payload["training"] = cross_training_summary
    cross_model_path = cross_system_dir / "model.json"
    _write_json(cross_model_path, cross_model_payload)
    cross_system_arm["model"] = {
        "path": _relative_to_out(cross_model_path, out_dir),
        "sha256": _sha256(cross_model_path),
        **cross_model_payload["summary"],
        "training": cross_training_summary,
    }

    s8_predictions = [
        predict_measure(
            proposal,
            selector=cross_selector,
            pitch_calibrator=cross_pitch_calibrator,
            max_chord_size=cross_max_chord_size,
        )
        for proposal in s8_proposals
    ]
    s8_prediction_path = cross_system_dir / "predictions" / "system8.jsonl"
    _write_jsonl(s8_prediction_path, s8_predictions)
    s8_prediction_hash = cross_guard.register_s8_prediction_file(s8_prediction_path)
    cross_prediction_manifest = {
        "schema_version": 1,
        "kind": "cross_system_weak_supervision_prediction_manifest",
        "system8_truth_opened_before_prediction_hash": False,
        "inference_uses_expected_note_counts": False,
        "splits": {
            "system8_model_selection": {
                "path": _relative_to_out(s8_prediction_path, out_dir),
                "sha256": s8_prediction_hash,
                "measure_count": len(s8_predictions),
                "system_index": 8,
            }
        },
    }
    cross_prediction_manifest_path = cross_system_dir / "predictions" / "manifest.json"
    _write_json(cross_prediction_manifest_path, cross_prediction_manifest)
    cross_system_arm["predictions"] = cross_prediction_manifest

    shared_s7_truth, s8_truth = cross_guard.read_validation_truth_after_s8_prediction(
        validation_truth_path,
        truth_reader,
    )
    validation_truth = [*shared_s7_truth, *s8_truth]
    guard.record_shared_validation_truth_read(validation_truth_path)
    validation_metrics = evaluate_ordered_predictions(validation_truth, validation_predictions)
    s7_truth = [
        row
        for row in validation_truth
        if int(row["identity"]["system_index"]) == VALIDATION_BASELINE_SYSTEM
    ]
    s7_predictions = [
        row
        for row in validation_predictions
        if int(row["identity"]["system_index"]) == VALIDATION_BASELINE_SYSTEM
    ]
    s7_metrics = evaluate_ordered_predictions(s7_truth, s7_predictions)
    validation_gate = evaluate_validation_gate(s7_metrics["summary"])
    guard.register_validation_gate(validation_gate["passed"])
    validation_metrics_path = hybrid_dir / "metrics" / "validation.json"
    validation_s7_metrics_path = hybrid_dir / "metrics" / "validation_s7_gate_scope.json"
    validation_gate_path = hybrid_dir / "validation_gate.json"
    _write_json(validation_metrics_path, validation_metrics)
    _write_json(validation_s7_metrics_path, s7_metrics)
    _write_json(validation_gate_path, validation_gate)
    report["predictions"] = prediction_manifest
    report["metrics"] = {
        "validation": validation_metrics,
        "validation_s7_gate_scope": s7_metrics,
    }
    hybrid_arm["validation_gate"] = {
        **validation_gate,
        "path": _relative_to_out(validation_gate_path, out_dir),
        "sha256": _sha256(validation_gate_path),
        "full_validation_metrics_path": _relative_to_out(validation_metrics_path, out_dir),
        "s7_metrics_path": _relative_to_out(validation_s7_metrics_path, out_dir),
    }
    if validation_gate["passed"]:
        hybrid_arm["status"] = "historical_validation_gate_passed"
    else:
        hybrid_arm["status"] = "stopped_at_validation_gate"

    s8_metrics = evaluate_ordered_predictions(s8_truth, s8_predictions)
    s8_metrics_path = cross_system_dir / "metrics" / "system8.json"
    _write_json(s8_metrics_path, s8_metrics)
    s8_gate = evaluate_s8_gate(s8_metrics["summary"])
    cross_guard.register_s8_gate(s8_gate["passed"])
    s8_gate_path = cross_system_dir / "system8_gate.json"
    _write_json(s8_gate_path, s8_gate)
    cross_system_arm["s8_gate"] = {
        **s8_gate,
        "path": _relative_to_out(s8_gate_path, out_dir),
        "sha256": _sha256(s8_gate_path),
    }
    cross_system_arm["metrics"] = {
        "system8": {
            "path": _relative_to_out(s8_metrics_path, out_dir),
            "sha256": _sha256(s8_metrics_path),
            **s8_metrics,
        }
    }
    cross_system_arm["truth_access_log"] = cross_guard.access_log
    if not s8_gate["passed"]:
        cross_system_arm["status"] = "stopped_at_system8_model_selection_gate"
        cross_system_arm["heldout"] = {
            "status": "not_reached",
            "prediction_written": False,
            "truth_read": False,
            "independent_evaluation_available": S3_COORDINATION_EVALUATION_ALLOWED,
            "coordination_status": S3_COORDINATION_STATUS,
        }
        report["status"] = "stopped_at_cross_system_s8_gate"
        report["truth_access_log"] = guard.access_log
        _write_reports(output_dir, report)
        return report

    if not S3_COORDINATION_EVALUATION_ALLOWED:
        cross_system_arm["status"] = "completed_at_system8_model_selection_only"
        cross_system_arm["heldout"] = {
            "status": "skipped_due_coordination",
            "prediction_written": False,
            "truth_read": False,
            "independent_evaluation_available": False,
            "coordination_status": S3_COORDINATION_STATUS,
        }
        report["status"] = "completed_without_system3_due_coordination"
        report["truth_access_log"] = guard.access_log
        _write_reports(output_dir, report)
        return report

    heldout_predictions = [
        predict_measure(
            proposal,
            selector=cross_selector,
            pitch_calibrator=cross_pitch_calibrator,
            max_chord_size=cross_max_chord_size,
        )
        for proposal in proposals_by_split["heldout"]
    ]
    heldout_prediction_path = cross_system_dir / "predictions" / "system3_heldout.jsonl"
    _write_jsonl(heldout_prediction_path, heldout_predictions)
    heldout_prediction_hash = cross_guard.register_heldout_prediction_file(heldout_prediction_path)
    cross_prediction_manifest["splits"]["system3_sealed_final_test"] = {
        "path": _relative_to_out(heldout_prediction_path, out_dir),
        "sha256": heldout_prediction_hash,
        "measure_count": len(heldout_predictions),
        "system_index": 3,
        "created_after_system8_gate_passed": True,
    }
    _write_json(cross_prediction_manifest_path, cross_prediction_manifest)

    heldout_truth = cross_guard.read_heldout_truth(
        benchmark_dir / "heldout" / "truth.jsonl", truth_reader
    )
    heldout_metrics = evaluate_ordered_predictions(heldout_truth, heldout_predictions)
    heldout_metrics_path = cross_system_dir / "metrics" / "system3_heldout.json"
    _write_json(heldout_metrics_path, heldout_metrics)
    cross_system_arm["predictions"] = cross_prediction_manifest
    cross_system_arm["metrics"]["system3_heldout"] = {
        "path": _relative_to_out(heldout_metrics_path, out_dir),
        "sha256": _sha256(heldout_metrics_path),
        **heldout_metrics,
    }
    cross_system_arm["heldout"] = {
        "status": "evaluated_once",
        "evaluation_count": 1,
        "metrics_path": _relative_to_out(heldout_metrics_path, out_dir),
        "metrics_sha256": _sha256(heldout_metrics_path),
    }
    cross_system_arm["truth_access_log"] = cross_guard.access_log
    cross_system_arm["status"] = "completed"
    report["truth_access_log"] = guard.access_log
    report["status"] = "completed"
    _write_reports(output_dir, report)
    return report


def generate_measure_proposals(out_dir: Path, request: Mapping[str, Any]) -> dict[str, Any]:
    image_record = request["images"]["raw"]
    image_path = out_dir / str(image_record["path_relative_to_out"])
    if _sha256(image_path) != str(image_record["sha256"]):
        raise ValueError(f"Benchmark image hash drift: {image_path}")
    staff_lines = tuple(
        int(round(float(value))) for value in request["staff_geometry"]["raw_staff_lines_y_px"]
    )
    if len(staff_lines) != 5:
        raise ValueError(f"Expected five raw staff lines for {request['identity']}")

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    stages = detector._detect_staff_grid_stages(
        image,
        staff_lines=list(staff_lines),
        max_candidates=CANDIDATE_CAP,
    )
    candidates = []
    for rank, candidate in enumerate(stages.candidates, start=1):
        features = classifier._candidate_feature_row(
            image,
            stages.threshold_mask,
            candidate,
            rank=rank,
            max_candidates=CANDIDATE_CAP,
            staff_lines=staff_lines,
            staff_spacing=stages.staff_spacing,
        )
        auto_pitch = treble_pitch_for_y(float(candidate.center[1]), staff_lines)
        candidates.append(
            {
                "id": f"c{rank:03d}",
                "rank": rank,
                "bbox": {
                    "left": int(candidate.bbox[0]),
                    "top": int(candidate.bbox[1]),
                    "right": int(candidate.bbox[2]) + 1,
                    "bottom": int(candidate.bbox[3]) + 1,
                },
                "center": {
                    "x": round(float(candidate.center[0]), 3),
                    "y": round(float(candidate.center[1]), 3),
                },
                "score": round(float(candidate.score), 6),
                "auto_pitch": auto_pitch,
                "auto_pitch_midi": pitch_name_to_midi(auto_pitch),
                "features": {name: round(float(value), 9) for name, value in features.items()},
            }
        )
    return {
        "schema_version": 1,
        "identity": dict(request["identity"]),
        "request_split": str(request["split"]),
        "request_image_sha256": str(image_record["sha256"]),
        "image_size_px": {"width": image.width, "height": image.height},
        "staff_lines_y_px": list(staff_lines),
        "staff_spacing_px": round(float(stages.staff_spacing), 6),
        "candidate_cap": CANDIDATE_CAP,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "provenance": {
            "candidate_generation_uses_ground_truth": False,
            "feature_generation_uses_ground_truth": False,
            "detector": "staff-grid-density-v2",
        },
    }


def ordered_onset_groups(truth_row: Mapping[str, Any]) -> list[OnsetGroup]:
    grouped: dict[float, list[int]] = {}
    for note in truth_row.get("notes") or []:
        onset = round(float(note["onset_beats"]), 9)
        grouped.setdefault(onset, []).append(int(note["pitch_midi"]))
    return [OnsetGroup(onset, tuple(sorted(pitches))) for onset, pitches in sorted(grouped.items())]


def align_onset_groups(
    groups: Sequence[OnsetGroup],
    candidates: Sequence[Mapping[str, Any]],
    *,
    image_width: float,
    staff_spacing: float,
) -> dict[str, Any]:
    """Align ordered pitch groups to x-monotone candidate subsets with DP."""
    ordered_candidates = sorted(
        candidates,
        key=lambda item: (
            float(item["center"]["x"]),
            float(item["center"]["y"]),
            str(item["id"]),
        ),
    )
    chord_tolerance = max(1.0, staff_spacing * CHORD_X_TOLERANCE_STAFF_SPACES)
    cache: dict[tuple[int, int], tuple[float, tuple[dict[str, Any], ...]]] = {}

    def solve(group_index: int, candidate_index: int) -> tuple[float, tuple[dict[str, Any], ...]]:
        key = (group_index, candidate_index)
        if key in cache:
            return cache[key]
        if group_index >= len(groups):
            return 0.0, ()
        if candidate_index >= len(ordered_candidates):
            remaining = sum(len(group.pitches) for group in groups[group_index:])
            missing = tuple(
                {
                    "group_index": index,
                    "onset_beats": groups[index].onset_beats,
                    "canonical_pitches": list(groups[index].pitches),
                    "candidate_ids": [],
                    "pairs": [],
                    "status": "missing",
                }
                for index in range(group_index, len(groups))
            )
            return 8.0 * remaining, missing

        options: list[tuple[float, tuple[dict[str, Any], ...]]] = []
        skipped_cost, skipped_path = solve(group_index, candidate_index + 1)
        options.append((skipped_cost + 0.002, skipped_path))

        group = groups[group_index]
        missed_cost, missed_path = solve(group_index + 1, candidate_index)
        missing_assignment = {
            "group_index": group_index,
            "onset_beats": group.onset_beats,
            "canonical_pitches": list(group.pitches),
            "candidate_ids": [],
            "pairs": [],
            "status": "missing",
        }
        options.append((missed_cost + 8.0 * len(group.pitches), (missing_assignment, *missed_path)))

        chord_size = len(group.pitches)
        first = ordered_candidates[candidate_index]
        eligible_tail = [
            index
            for index in range(candidate_index + 1, len(ordered_candidates))
            if float(ordered_candidates[index]["center"]["x"]) - float(first["center"]["x"])
            <= chord_tolerance
        ]
        tail_size = chord_size - 1
        if tail_size == 0:
            combinations: Iterable[tuple[int, ...]] = [()]
        else:
            combinations = itertools.combinations(eligible_tail, tail_size)
        for tail in combinations:
            indices = (candidate_index, *tail)
            selected = [ordered_candidates[index] for index in indices]
            match_cost, pairs = _group_match_cost(
                group,
                selected,
                group_index=group_index,
                group_count=len(groups),
                image_width=image_width,
                staff_spacing=staff_spacing,
            )
            suffix_cost, suffix = solve(group_index + 1, max(indices) + 1)
            assignment = {
                "group_index": group_index,
                "onset_beats": group.onset_beats,
                "canonical_pitches": list(group.pitches),
                "candidate_ids": [str(candidate["id"]) for candidate in selected],
                "pairs": pairs,
                "status": "aligned",
                "match_cost": round(match_cost, 9),
            }
            options.append((match_cost + suffix_cost, (assignment, *suffix)))

        result = min(
            options,
            key=lambda item: (round(item[0], 12), _alignment_signature(item[1])),
        )
        cache[key] = result
        return result

    total_cost, assignments = solve(0, 0)
    return {
        "algorithm": "monotonic_dynamic_program_v1",
        "total_cost": round(total_cost, 9),
        "group_count": len(groups),
        "aligned_group_count": sum(row["status"] == "aligned" for row in assignments),
        "assignments": list(assignments),
    }


def _group_match_cost(
    group: OnsetGroup,
    selected: Sequence[Mapping[str, Any]],
    *,
    group_index: int,
    group_count: int,
    image_width: float,
    staff_spacing: float,
) -> tuple[float, list[dict[str, Any]]]:
    ordered = sorted(
        selected,
        key=lambda item: (int(item["auto_pitch_midi"]), str(item["id"])),
    )
    pairs = []
    pitch_cost = 0.0
    for canonical_pitch, candidate in zip(group.pitches, ordered, strict=True):
        auto_pitch_midi = int(candidate["auto_pitch_midi"])
        difference = abs(canonical_pitch - auto_pitch_midi)
        pitch_cost += min(difference, 12) / 3.0
        pairs.append(
            {
                "candidate_id": str(candidate["id"]),
                "canonical_pitch_midi": canonical_pitch,
                "auto_pitch_midi": auto_pitch_midi,
                "auto_pitch": str(candidate["auto_pitch"]),
                "absolute_pitch_error": difference,
            }
        )
    pitch_cost /= max(1, len(group.pitches))
    evidence_cost = sum(1.0 - float(candidate["score"]) for candidate in selected) / len(selected)
    mean_x = sum(float(candidate["center"]["x"]) for candidate in selected) / len(selected)
    expected_fraction = (group_index + 1) / (group_count + 1)
    position_cost = abs(mean_x / max(1.0, image_width) - expected_fraction)
    x_values = [float(candidate["center"]["x"]) for candidate in selected]
    spread_cost = (max(x_values) - min(x_values)) / max(1.0, staff_spacing)
    return (
        pitch_cost + 0.7 * evidence_cost + 0.18 * position_cost + 0.25 * spread_cost,
        pairs,
    )


def build_development_pseudo_labels(
    truth_rows: Sequence[Mapping[str, Any]],
    proposal_rows: Sequence[Mapping[str, Any]],
    *,
    source_split: str = "development",
) -> list[dict[str, Any]]:
    proposals = {_identity_key(row["identity"]): row for row in proposal_rows}
    result = []
    for truth in truth_rows:
        key = _identity_key(truth["identity"])
        proposal = proposals[key]
        groups = ordered_onset_groups(truth)
        alignment = align_onset_groups(
            groups,
            proposal["candidates"],
            image_width=float(proposal["image_size_px"]["width"]),
            staff_spacing=float(proposal["staff_spacing_px"]),
        )
        pitch_by_candidate = {
            pair["candidate_id"]: int(pair["canonical_pitch_midi"])
            for assignment in alignment["assignments"]
            for pair in assignment["pairs"]
        }
        labels = [
            {
                "candidate_id": str(candidate["id"]),
                "label": int(str(candidate["id"]) in pitch_by_candidate),
                "canonical_pitch_midi": pitch_by_candidate.get(str(candidate["id"])),
            }
            for candidate in proposal["candidates"]
        ]
        result.append(
            {
                "schema_version": 1,
                "identity": dict(truth["identity"]),
                "source": {
                    "split": source_split,
                    "canonical_fields_used": ["onset_beats", "pitch_midi"],
                    "duration_used": False,
                    "candidate_generation_completed_before_truth_read": True,
                },
                "alignment": alignment,
                "labels": labels,
            }
        )
    return result


def audit_pseudo_labels(
    pseudo_rows: Sequence[Mapping[str, Any]],
    proposal_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pseudo_by_key = {_identity_key(row["identity"]): row for row in pseudo_rows}
    proposal_by_key = {_identity_key(row["identity"]): row for row in proposal_rows}
    measures = []
    total_tp = total_fp = total_fn = 0
    for measure in AUDIT_MEASURES:
        key = (DEFAULT_SLUG, AUDIT_SYSTEM, measure, measure - 1)
        if key not in pseudo_by_key:
            matching = [
                candidate_key
                for candidate_key in pseudo_by_key
                if candidate_key[1] == AUDIT_SYSTEM and candidate_key[2] == measure
            ]
            if len(matching) != 1:
                raise ValueError(f"Cannot resolve audit identity for system 1 measure {measure}")
            key = matching[0]
        pseudo = pseudo_by_key[key]
        proposal = proposal_by_key[key]
        gt_path = (
            REPO_ROOT
            / "tests/fixtures/vlm_melody/notehead_ground_truth"
            / f"{key[0]}_system_{AUDIT_SYSTEM:03d}_measure_{measure:03d}.json"
        )
        ellipses = classifier._load_annotation_ellipses(gt_path)
        selected_ids = {
            str(label["candidate_id"]) for label in pseudo["labels"] if int(label["label"]) == 1
        }
        selected = [
            candidate
            for candidate in proposal["candidates"]
            if str(candidate["id"]) in selected_ids
        ]
        metric = _coordinate_selection_metrics(selected, ellipses)
        total_tp += metric["tp"]
        total_fp += metric["fp"]
        total_fn += metric["fn"]
        measures.append(
            {
                "identity": dict(pseudo["identity"]),
                "coordinate_label_path": str(gt_path.relative_to(REPO_ROOT)),
                "coordinate_label_sha256": _sha256(gt_path),
                **metric,
            }
        )
    return {
        "scope": "independent S1M1-4 coordinate labels",
        "annotation_ellipse_margin": classifier.ANNOTATION_REGION_MARGIN,
        "selected_count": total_tp + total_fp,
        "ground_truth_count": total_tp + total_fn,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": _ratio(total_tp, total_tp + total_fp),
        "recall": _ratio(total_tp, total_tp + total_fn),
        "f1": _ratio(2 * total_tp, 2 * total_tp + total_fp + total_fn),
        "measures": measures,
    }


def _coordinate_selection_metrics(
    selected: Sequence[Mapping[str, Any]],
    ellipses: Sequence[classifier.Ellipse],
) -> dict[str, Any]:
    possible = []
    for candidate_index, candidate in enumerate(selected):
        x = float(candidate["center"]["x"])
        y = float(candidate["center"]["y"])
        for ellipse_index, ellipse in enumerate(ellipses):
            distance = math.sqrt(
                ((x - ellipse.center_x) / ellipse.radius_x) ** 2
                + ((y - ellipse.center_y) / ellipse.radius_y) ** 2
            )
            if distance <= 1.0:
                possible.append(
                    (distance, str(candidate["id"]), ellipse.id, candidate_index, ellipse_index)
                )
    used_candidates: set[int] = set()
    used_ellipses: set[int] = set()
    assignments = []
    for distance, candidate_id, ellipse_id, candidate_index, ellipse_index in sorted(possible):
        if candidate_index in used_candidates or ellipse_index in used_ellipses:
            continue
        used_candidates.add(candidate_index)
        used_ellipses.add(ellipse_index)
        assignments.append(
            {
                "candidate_id": candidate_id,
                "coordinate_label_id": ellipse_id,
                "normalized_ellipse_distance": round(distance, 6),
            }
        )
    tp = len(assignments)
    fp = len(selected) - tp
    fn = len(ellipses) - tp
    return {
        "selected_count": len(selected),
        "ground_truth_count": len(ellipses),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "assignments": assignments,
    }


def bootstrap_gate(
    audit: Mapping[str, Any],
    *,
    minimum_precision: float = GATE_PRECISION,
    minimum_recall: float = GATE_RECALL,
) -> dict[str, Any]:
    precision = float(audit["precision"])
    recall = float(audit["recall"])
    passed = precision >= minimum_precision and recall >= minimum_recall
    return {
        "minimum_precision": minimum_precision,
        "minimum_recall": minimum_recall,
        "observed_precision": precision,
        "observed_recall": recall,
        "passed": passed,
        "failure_action": None if passed else "stop_without_training_or_evaluation",
    }


def load_exact_review_labels(
    proposal_rows: Sequence[Mapping[str, Any]],
    *,
    slug: str,
) -> list[dict[str, Any]]:
    proposals = {_identity_key(row["identity"]): row for row in proposal_rows}
    rows = []
    for measure in REVIEW_MEASURES:
        review_path = (
            REPO_ROOT
            / "tests/fixtures/vlm_melody/notehead_reviews"
            / f"{slug}_system_001_measure_{measure:03d}.json"
        )
        review = json.loads(review_path.read_text(encoding="utf-8"))
        identity = dict(review["identity"])
        key = _identity_key(identity)
        proposal = proposals.get(key)
        if proposal is None:
            raise ValueError(f"Reviewed measure is absent from development proposals: {identity}")
        if review.get("manual_noteheads"):
            raise ValueError(f"Hybrid supervision requires candidate-only reviews: {review_path}")
        if str(review["source"]["image_sha256"]) != str(proposal["request_image_sha256"]):
            raise ValueError(f"Review image hash does not match frozen proposal: {review_path}")

        reviewed_candidates = {str(row["id"]): row for row in review["candidates"]}
        proposal_candidates = {str(row["id"]): row for row in proposal["candidates"]}
        if reviewed_candidates.keys() != proposal_candidates.keys():
            raise ValueError(f"Review candidate IDs do not match frozen cap-24 set: {review_path}")
        for candidate_id, candidate in proposal_candidates.items():
            reviewed_center = reviewed_candidates[candidate_id]["center"]
            if float(reviewed_center["x"]) != float(candidate["center"]["x"]) or float(
                reviewed_center["y"]
            ) != float(candidate["center"]["y"]):
                raise ValueError(
                    f"Review candidate geometry drift for {candidate_id}: {review_path}"
                )

        positive_pitches = {}
        for note in review["final_noteheads"]:
            source = note.get("source") or {}
            if source.get("kind") != "candidate":
                raise ValueError(f"Review contains a non-candidate final notehead: {review_path}")
            candidate_id = str(source["candidate_id"])
            positive_pitches[candidate_id] = pitch_name_to_midi(str(note["pitch"]))
        labels = [
            {
                "candidate_id": candidate_id,
                "label": int(candidate_id in positive_pitches),
                "pitch_midi": positive_pitches.get(candidate_id),
            }
            for candidate_id in proposal_candidates
        ]
        rows.append(
            {
                "schema_version": 1,
                "identity": identity,
                "source": {
                    "kind": "promoted_exact_candidate_review",
                    "review_path": str(review_path.relative_to(REPO_ROOT)),
                    "review_sha256": _sha256(review_path),
                    "candidate_cap": CANDIDATE_CAP,
                },
                "labels": labels,
            }
        )
    return rows


def calibrate_pseudo_acceptance_rule(
    pseudo_rows: Sequence[Mapping[str, Any]],
    exact_review_rows: Sequence[Mapping[str, Any]],
) -> tuple[PseudoAcceptanceRule | None, dict[str, Any]]:
    calibration_pseudo = [
        row
        for row in pseudo_rows
        if int(row["identity"]["system_index"]) == 1
        and int(row["identity"]["system_measure_index"]) in PSEUDO_CALIBRATION_MEASURES
    ]
    exact_by_key = {_identity_key(row["identity"]): row for row in exact_review_rows}
    candidate_rules = [
        PseudoAcceptanceRule(cost, pitch_error)
        for pitch_error in PSEUDO_RULE_MAX_PITCH_ERRORS
        for cost in PSEUDO_RULE_MAX_COSTS
    ]
    evaluations = [
        _evaluate_pseudo_acceptance_rule(rule, calibration_pseudo, exact_by_key)
        for rule in candidate_rules
    ]
    selected_index = next(
        (
            index
            for index, evaluation in enumerate(evaluations)
            if evaluation["precision"] >= HYBRID_CALIBRATION_PRECISION
            and evaluation["recall"] >= HYBRID_CALIBRATION_RECALL
        ),
        None,
    )
    selected_rule = candidate_rules[selected_index] if selected_index is not None else None
    selected_metrics = evaluations[selected_index] if selected_index is not None else None
    return selected_rule, {
        "schema_version": 1,
        "kind": "pickup_aware_pseudo_acceptance_calibration",
        "calibration_scope": "reviewed non-pickup S1M2-4 only",
        "pickup_s1m1_used": False,
        "fixed_rule_grid": [rule.to_json() for rule in candidate_rules],
        "selection_rule": (
            "first rule in ascending max pitch error then ascending max match cost "
            "meeting both calibration gates"
        ),
        "minimum_precision": HYBRID_CALIBRATION_PRECISION,
        "minimum_recall": HYBRID_CALIBRATION_RECALL,
        "evaluations": evaluations,
        "selected_rule": selected_rule.to_json() if selected_rule else None,
        "selected_metrics": selected_metrics,
        "passed": selected_rule is not None,
        "failure_action": (
            None if selected_rule is not None else "stop_without_training_or_validation"
        ),
    }


def _evaluate_pseudo_acceptance_rule(
    rule: PseudoAcceptanceRule,
    pseudo_rows: Sequence[Mapping[str, Any]],
    exact_by_key: Mapping[tuple[str, int, int, int], Mapping[str, Any]],
) -> dict[str, Any]:
    predicted: set[tuple[tuple[str, int, int, int], str, int]] = set()
    truth: set[tuple[tuple[str, int, int, int], str, int]] = set()
    for pseudo in pseudo_rows:
        key = _identity_key(pseudo["identity"])
        exact = exact_by_key[key]
        truth.update(
            (key, str(label["candidate_id"]), int(label["pitch_midi"]))
            for label in exact["labels"]
            if int(label["label"]) == 1
        )
        for assignment in pseudo["alignment"]["assignments"]:
            if not rule.accepts(assignment):
                continue
            predicted.update(
                (key, str(pair["candidate_id"]), int(pair["canonical_pitch_midi"]))
                for pair in assignment["pairs"]
            )
    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    return {
        "rule": rule.to_json(),
        "accepted_count": len(predicted),
        "truth_count": len(truth),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }


def apply_pseudo_acceptance_rule(
    pseudo_rows: Sequence[Mapping[str, Any]],
    rule: PseudoAcceptanceRule,
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in pseudo_rows
        if (
            int(row["identity"]["system_index"]) == 1
            and 5 <= int(row["identity"]["system_measure_index"]) <= 8
        )
        or int(row["identity"]["system_index"]) == 2
    ]
    if len(eligible) != 13:
        raise ValueError(f"Expected 13 unreviewed development measures, got {len(eligible)}")
    results = []
    for pseudo in eligible:
        results.append(
            {
                "schema_version": 1,
                "identity": dict(pseudo["identity"]),
                "source": {
                    "kind": "accepted_development_sequence_pseudo_positive",
                    "rule": rule.to_json(),
                    "unaccepted_candidates_are_unlabeled": True,
                },
                "accepted_labels": _accepted_labels_for_pseudo(pseudo, rule),
            }
        )
    return results


def apply_frozen_rule_to_s7(
    pseudo_rows: Sequence[Mapping[str, Any]],
    rule: PseudoAcceptanceRule,
) -> list[dict[str, Any]]:
    systems = {int(row["identity"]["system_index"]) for row in pseudo_rows}
    if systems != {CROSS_SYSTEM_TRAIN_SYSTEM} or len(pseudo_rows) != 7:
        raise ValueError(
            f"Cross-system adaptation requires seven S7 pseudo rows, got {len(pseudo_rows)} "
            f"from systems {systems}"
        )
    return [
        {
            "schema_version": 1,
            "identity": dict(pseudo["identity"]),
            "source": {
                "kind": "accepted_system7_sequence_pseudo_positive",
                "rule": rule.to_json(),
                "unaccepted_candidates_are_unlabeled": True,
                "system8_truth_or_expected_counts_used": False,
            },
            "accepted_labels": _accepted_labels_for_pseudo(pseudo, rule),
        }
        for pseudo in pseudo_rows
    ]


def _accepted_labels_for_pseudo(
    pseudo: Mapping[str, Any], rule: PseudoAcceptanceRule
) -> list[dict[str, Any]]:
    accepted_labels = []
    for assignment in pseudo["alignment"]["assignments"]:
        if not rule.accepts(assignment):
            continue
        for pair in assignment["pairs"]:
            accepted_labels.append(
                {
                    "candidate_id": str(pair["candidate_id"]),
                    "pitch_midi": int(pair["canonical_pitch_midi"]),
                    "auto_pitch_midi": int(pair["auto_pitch_midi"]),
                    "group_index": int(assignment["group_index"]),
                    "group_size": len(assignment["pairs"]),
                    "match_cost": float(assignment["match_cost"]),
                    "absolute_pitch_error": int(pair["absolute_pitch_error"]),
                }
            )
    return accepted_labels


def evaluate_s7_pseudo_coverage(
    accepted_rows: Sequence[Mapping[str, Any]],
    truth_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    accepted_count = sum(len(row["accepted_labels"]) for row in accepted_rows)
    canonical_note_count = sum(len(row.get("notes") or []) for row in truth_rows)
    coverage = _ratio(accepted_count, canonical_note_count)
    count_passed = accepted_count >= CROSS_SYSTEM_MIN_ACCEPTED_LABELS
    coverage_passed = coverage >= CROSS_SYSTEM_MIN_PSEUDO_COVERAGE
    return {
        "minimum_accepted_labels": CROSS_SYSTEM_MIN_ACCEPTED_LABELS,
        "minimum_canonical_note_coverage": CROSS_SYSTEM_MIN_PSEUDO_COVERAGE,
        "accepted_label_count": accepted_count,
        "canonical_note_count": canonical_note_count,
        "canonical_note_coverage": coverage,
        "accepted_count_requirement_passed": count_passed,
        "coverage_requirement_passed": coverage_passed,
        "passed": count_passed and coverage_passed,
        "failure_action": (
            None if count_passed and coverage_passed else "stop_without_system8_prediction_or_truth"
        ),
    }


def train_hybrid_models(
    exact_review_rows: Sequence[Mapping[str, Any]],
    accepted_pseudo_rows: Sequence[Mapping[str, Any]],
    proposal_rows: Sequence[Mapping[str, Any]],
    *,
    pseudo_source_description: str = "accepted S1M5-8/S2M1-9 pseudo positives",
) -> tuple[LinearSelector, PitchCalibrator, int, dict[str, Any]]:
    proposals = {_identity_key(row["identity"]): row for row in proposal_rows}
    matrix: list[list[float]] = []
    labels: list[int] = []
    exact_matrix: list[list[float]] = []
    exact_labels: list[int] = []
    residuals_by_letter: dict[str, list[int]] = {}
    all_residuals = []

    def add_pitch_residual(candidate: Mapping[str, Any], pitch_midi: int) -> None:
        residual = pitch_midi - int(candidate["auto_pitch_midi"])
        letter = str(candidate["auto_pitch"])[0]
        residuals_by_letter.setdefault(letter, []).append(residual)
        all_residuals.append(residual)

    for review in exact_review_rows:
        proposal = proposals[_identity_key(review["identity"])]
        candidates = {str(row["id"]): row for row in proposal["candidates"]}
        for label in review["labels"]:
            candidate = candidates[str(label["candidate_id"])]
            features = [float(candidate["features"][name]) for name in FEATURE_NAMES]
            target = int(label["label"])
            matrix.append(features)
            labels.append(target)
            exact_matrix.append(features)
            exact_labels.append(target)
            if target:
                add_pitch_residual(candidate, int(label["pitch_midi"]))

    pseudo_positive_count = 0
    max_chord_size = 1
    for pseudo in accepted_pseudo_rows:
        proposal = proposals[_identity_key(pseudo["identity"])]
        candidates = {str(row["id"]): row for row in proposal["candidates"]}
        for label in pseudo["accepted_labels"]:
            candidate = candidates[str(label["candidate_id"])]
            matrix.append([float(candidate["features"][name]) for name in FEATURE_NAMES])
            labels.append(1)
            pseudo_positive_count += 1
            max_chord_size = max(max_chord_size, int(label["group_size"]))
            add_pitch_residual(candidate, int(label["pitch_midi"]))

    means, scales, standardized = classifier._standardize(matrix)
    weights, intercept = _fit_weighted_logistic(standardized, labels)
    exact_standardized = [
        [(value - mean) / scale for value, mean, scale in zip(row, means, scales, strict=True)]
        for row in exact_matrix
    ]
    exact_probabilities = [_sigmoid(intercept + _dot(weights, row)) for row in exact_standardized]
    threshold = _select_probability_threshold(exact_probabilities, exact_labels)
    selector = LinearSelector(
        feature_names=tuple(FEATURE_NAMES),
        means=tuple(means),
        scales=tuple(scales),
        weights=tuple(weights),
        intercept=intercept,
        threshold=threshold,
    )
    pitch_calibrator = PitchCalibrator(
        _integer_median(all_residuals),
        {
            letter: _integer_median(values)
            for letter, values in sorted(residuals_by_letter.items())
            if len(values) >= 2
        },
    )
    threshold_metrics = _threshold_metrics(exact_probabilities, exact_labels, threshold)
    return (
        selector,
        pitch_calibrator,
        max_chord_size,
        {
            "exact_review_candidate_count": len(exact_labels),
            "exact_review_positive_count": sum(exact_labels),
            "exact_review_negative_count": len(exact_labels) - sum(exact_labels),
            "accepted_pseudo_positive_count": pseudo_positive_count,
            "unreviewed_pseudo_negative_count": 0,
            "selector_threshold_calibration": "exact S1M1-4 reviews only",
            "selector_threshold_metrics": threshold_metrics,
            "pitch_calibration_sources": [
                "exact S1M1-4 review positives",
                pseudo_source_description,
            ],
        },
    )


def _threshold_metrics(
    probabilities: Sequence[float], labels: Sequence[int], threshold: float
) -> dict[str, Any]:
    predicted = [probability >= threshold for probability in probabilities]
    tp = sum(prediction and label for prediction, label in zip(predicted, labels, strict=True))
    fp = sum(prediction and not label for prediction, label in zip(predicted, labels, strict=True))
    fn = sum(not prediction and label for prediction, label in zip(predicted, labels, strict=True))
    return {
        "threshold": round(threshold, 12),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }


def _validation_gate_preregistration() -> dict[str, Any]:
    return {
        "status": "preregistered_before_validation_truth_read",
        "comparison_scope": "system 7, matching the existing full-system VLM baseline",
        "baseline": {
            "path_relative_to_out": VALIDATION_BASELINE_PATH,
            "sha256": VALIDATION_BASELINE_SHA256,
            "ordered_pitch_accuracy": VALIDATION_BASELINE_ORDERED_PITCH,
            "mean_absolute_note_count_error": VALIDATION_BASELINE_COUNT_MAE,
        },
        "requirements": {
            "ordered_pitch_accuracy": f">={VALIDATION_REQUIRED_ORDERED_PITCH}",
            "ordered_pitch_absolute_margin": VALIDATION_ORDERED_PITCH_MARGIN,
            "mean_absolute_note_count_error": f"<={VALIDATION_BASELINE_COUNT_MAE}",
            "all_must_pass": True,
        },
        "heldout_action": "write heldout predictions only after this gate passes",
    }


def evaluate_validation_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    ordered_pitch = float(summary["ordered_pitch_accuracy"])
    count_mae = float(summary["mean_absolute_note_count_error"])
    pitch_passed = ordered_pitch >= VALIDATION_REQUIRED_ORDERED_PITCH
    count_passed = count_mae <= VALIDATION_BASELINE_COUNT_MAE
    return {
        **_validation_gate_preregistration(),
        "status": "evaluated",
        "observed": {
            "ordered_pitch_accuracy": ordered_pitch,
            "mean_absolute_note_count_error": count_mae,
        },
        "pitch_requirement_passed": pitch_passed,
        "count_requirement_passed": count_passed,
        "passed": pitch_passed and count_passed,
        "failure_action": (
            None if pitch_passed and count_passed else "stop_without_heldout_prediction_or_truth"
        ),
    }


def partition_cross_system_validation_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    s7_rows = [
        dict(row)
        for row in rows
        if int(row["identity"]["system_index"]) == CROSS_SYSTEM_TRAIN_SYSTEM
    ]
    s8_rows = [
        dict(row)
        for row in rows
        if int(row["identity"]["system_index"]) == CROSS_SYSTEM_MODEL_SELECTION_SYSTEM
    ]
    unexpected = {int(row["identity"]["system_index"]) for row in rows} - {
        CROSS_SYSTEM_TRAIN_SYSTEM,
        CROSS_SYSTEM_MODEL_SELECTION_SYSTEM,
    }
    if unexpected or len(s7_rows) != 7 or len(s8_rows) != 7:
        raise ValueError(
            "Cross-system validation partition requires seven S7 and seven S8 rows; "
            f"got S7={len(s7_rows)}, S8={len(s8_rows)}, unexpected={sorted(unexpected)}"
        )
    s7_keys = {_identity_key(row["identity"]) for row in s7_rows}
    s8_keys = {_identity_key(row["identity"]) for row in s8_rows}
    if s7_keys & s8_keys:
        raise ValueError("S7 training and S8 model-selection identities overlap")
    return s7_rows, s8_rows


def _s8_gate_preregistration() -> dict[str, Any]:
    return {
        "status": "preregistered_before_system8_truth_read",
        "evidence_role": "model_selection_not_untouched_validation",
        "requirements": {
            "ordered_pitch_accuracy": f">={S8_GATE_ORDERED_PITCH}",
            "mean_absolute_note_count_error": f"<={S8_GATE_COUNT_MAE}",
            "all_must_pass": True,
        },
        "inference_uses_expected_note_counts": False,
        "heldout_action_original_preregistration": (
            "persist S3 predictions only after this gate passes"
        ),
        "coordination_override": {
            "evaluation_allowed": S3_COORDINATION_EVALUATION_ALLOWED,
            "status": S3_COORDINATION_STATUS,
            "action": "skip S3 and report only S8",
        },
    }


def evaluate_s8_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    ordered_pitch = float(summary["ordered_pitch_accuracy"])
    count_mae = float(summary["mean_absolute_note_count_error"])
    pitch_passed = ordered_pitch >= S8_GATE_ORDERED_PITCH
    count_passed = count_mae <= S8_GATE_COUNT_MAE
    return {
        **_s8_gate_preregistration(),
        "status": "evaluated",
        "observed": {
            "ordered_pitch_accuracy": ordered_pitch,
            "mean_absolute_note_count_error": count_mae,
        },
        "pitch_requirement_passed": pitch_passed,
        "count_requirement_passed": count_passed,
        "passed": pitch_passed and count_passed,
        "failure_action": (
            None if pitch_passed and count_passed else "stop_without_system3_prediction_or_truth"
        ),
    }


def train_bootstrap_models(
    pseudo_rows: Sequence[Mapping[str, Any]],
    proposal_rows: Sequence[Mapping[str, Any]],
) -> tuple[LinearSelector, PitchCalibrator, int]:
    labels_by_key = {
        (_identity_key(row["identity"]), str(label["candidate_id"])): label
        for row in pseudo_rows
        for label in row["labels"]
    }
    matrix: list[list[float]] = []
    labels: list[int] = []
    labeled_candidates: list[Mapping[str, Any]] = []
    residuals_by_letter: dict[str, list[int]] = {}
    all_residuals = []
    for proposal in proposal_rows:
        identity = _identity_key(proposal["identity"])
        for candidate in proposal["candidates"]:
            label = labels_by_key[(identity, str(candidate["id"]))]
            matrix.append([float(candidate["features"][name]) for name in FEATURE_NAMES])
            labels.append(int(label["label"]))
            labeled_candidates.append(candidate)
            if label["canonical_pitch_midi"] is not None:
                residual = int(label["canonical_pitch_midi"]) - int(candidate["auto_pitch_midi"])
                letter = str(candidate["auto_pitch"])[0]
                residuals_by_letter.setdefault(letter, []).append(residual)
                all_residuals.append(residual)
    if not any(labels) or all(labels):
        raise ValueError("Pseudo labels must contain positive and negative candidates")
    means, scales, standardized = classifier._standardize(matrix)
    weights, intercept = _fit_weighted_logistic(standardized, labels)
    probabilities = [_sigmoid(intercept + _dot(weights, row)) for row in standardized]
    threshold = _select_probability_threshold(probabilities, labels)
    selector = LinearSelector(
        feature_names=tuple(FEATURE_NAMES),
        means=tuple(means),
        scales=tuple(scales),
        weights=tuple(weights),
        intercept=intercept,
        threshold=threshold,
    )
    global_offset = _integer_median(all_residuals)
    offsets_by_letter = {
        letter: _integer_median(values)
        for letter, values in sorted(residuals_by_letter.items())
        if len(values) >= 2
    }
    pitch_calibrator = PitchCalibrator(global_offset, offsets_by_letter)
    max_chord_size = max(
        len(assignment["canonical_pitches"])
        for row in pseudo_rows
        for assignment in row["alignment"]["assignments"]
    )
    return selector, pitch_calibrator, max_chord_size


def _fit_weighted_logistic(
    matrix: Sequence[Sequence[float]], labels: Sequence[int]
) -> tuple[list[float], float]:
    feature_count = len(matrix[0])
    weights = [0.0] * feature_count
    intercept = 0.0
    positives = sum(labels)
    negatives = len(labels) - positives
    positive_weight = len(labels) / (2.0 * positives)
    negative_weight = len(labels) / (2.0 * negatives)
    weight_total = sum(positive_weight if label else negative_weight for label in labels)
    for iteration in range(1200):
        gradient = [0.0] * feature_count
        intercept_gradient = 0.0
        for row, label in zip(matrix, labels, strict=True):
            probability = _sigmoid(intercept + _dot(weights, row))
            class_weight = positive_weight if label else negative_weight
            error = (probability - label) * class_weight
            intercept_gradient += error
            for index, value in enumerate(row):
                gradient[index] += error * value
        learning_rate = 0.08 / (1.0 + iteration / 500.0)
        intercept -= learning_rate * intercept_gradient / weight_total
        for index in range(feature_count):
            regularized = gradient[index] / weight_total + weights[index] / len(labels)
            weights[index] -= learning_rate * regularized
    return weights, intercept


def _select_probability_threshold(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    thresholds = [max(probabilities) + 1e-12, *sorted(set(probabilities), reverse=True)]
    best_threshold = thresholds[0]
    best_rank = (-1.0, -1.0, -1.0, 0, 0.0)
    for threshold in thresholds:
        predicted = [probability >= threshold for probability in probabilities]
        tp = sum(prediction and label for prediction, label in zip(predicted, labels, strict=True))
        fp = sum(
            prediction and not label for prediction, label in zip(predicted, labels, strict=True)
        )
        fn = sum(
            not prediction and label for prediction, label in zip(predicted, labels, strict=True)
        )
        precision = _raw_ratio(tp, tp + fp)
        recall = _raw_ratio(tp, tp + fn)
        f1 = _raw_ratio(2 * tp, 2 * tp + fp + fn)
        rank = (f1, precision, recall, -sum(predicted), threshold)
        if rank > best_rank:
            best_rank = rank
            best_threshold = threshold
    return best_threshold


def predict_measure(
    proposal: Mapping[str, Any],
    *,
    selector: LinearSelector,
    pitch_calibrator: PitchCalibrator,
    max_chord_size: int,
) -> dict[str, Any]:
    scored = [
        {**candidate, "selector_probability": selector.score(candidate)}
        for candidate in proposal["candidates"]
    ]
    selected = [
        candidate
        for candidate in scored
        if float(candidate["selector_probability"]) >= selector.threshold
    ]
    groups = _group_selected_candidates(
        selected,
        staff_spacing=float(proposal["staff_spacing_px"]),
        max_chord_size=max_chord_size,
    )
    prediction_groups = []
    for group_index, group in enumerate(groups):
        notes = sorted(
            [
                {
                    "candidate_id": str(candidate["id"]),
                    "pitch_midi": pitch_calibrator.predict(candidate),
                    "auto_pitch": str(candidate["auto_pitch"]),
                    "auto_pitch_midi": int(candidate["auto_pitch_midi"]),
                    "selector_probability": round(float(candidate["selector_probability"]), 9),
                }
                for candidate in group
            ],
            key=lambda item: (item["pitch_midi"], item["candidate_id"]),
        )
        prediction_groups.append(
            {
                "group_index": group_index,
                "x_px": round(
                    sum(float(candidate["center"]["x"]) for candidate in group) / len(group),
                    3,
                ),
                "notes": notes,
            }
        )
    ordered_pitches = [note["pitch_midi"] for group in prediction_groups for note in group["notes"]]
    return {
        "schema_version": 1,
        "identity": dict(proposal["identity"]),
        "candidate_file_split": str(proposal["request_split"]),
        "selector_threshold": round(selector.threshold, 12),
        "grouping_x_tolerance_staff_spaces": CHORD_X_TOLERANCE_STAFF_SPACES,
        "predicted_group_count": len(prediction_groups),
        "predicted_note_count": len(ordered_pitches),
        "groups": prediction_groups,
        "ordered_pitches": ordered_pitches,
        "onsets": "out_of_scope",
        "durations": "out_of_scope",
    }


def _group_selected_candidates(
    selected: Sequence[Mapping[str, Any]],
    *,
    staff_spacing: float,
    max_chord_size: int,
) -> list[list[Mapping[str, Any]]]:
    tolerance = max(1.0, staff_spacing * CHORD_X_TOLERANCE_STAFF_SPACES)
    ordered = sorted(
        selected,
        key=lambda item: (float(item["center"]["x"]), float(item["center"]["y"]), str(item["id"])),
    )
    clusters: list[list[Mapping[str, Any]]] = []
    for candidate in ordered:
        if (
            clusters
            and float(candidate["center"]["x"]) - float(clusters[-1][0]["center"]["x"]) <= tolerance
        ):
            clusters[-1].append(candidate)
        else:
            clusters.append([candidate])
    groups = []
    for cluster in clusters:
        best_by_auto_pitch: dict[int, Mapping[str, Any]] = {}
        for candidate in cluster:
            pitch = int(candidate["auto_pitch_midi"])
            previous = best_by_auto_pitch.get(pitch)
            if previous is None or (
                float(candidate["selector_probability"]),
                -int(candidate["rank"]),
                str(candidate["id"]),
            ) > (
                float(previous["selector_probability"]),
                -int(previous["rank"]),
                str(previous["id"]),
            ):
                best_by_auto_pitch[pitch] = candidate
        retained = sorted(
            best_by_auto_pitch.values(),
            key=lambda item: (
                -float(item["selector_probability"]),
                int(item["rank"]),
                str(item["id"]),
            ),
        )[:max_chord_size]
        groups.append(
            sorted(retained, key=lambda item: (float(item["center"]["y"]), str(item["id"])))
        )
    return groups


def evaluate_ordered_predictions(
    truth_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    predictions = {_identity_key(row["identity"]): row for row in prediction_rows}
    if len(predictions) != len(prediction_rows):
        raise ValueError("Duplicate prediction identity")
    results = []
    total_predicted = total_truth = total_pitch_correct = total_compared = 0
    for truth in truth_rows:
        key = _identity_key(truth["identity"])
        prediction = predictions.get(key)
        if prediction is None:
            predicted_groups: list[Mapping[str, Any]] = []
            predicted_pitches: list[int] = []
        else:
            predicted_groups = list(prediction["groups"])
            predicted_pitches = [int(value) for value in prediction["ordered_pitches"]]
        truth_groups = ordered_onset_groups(truth)
        truth_pitches = [pitch for group in truth_groups for pitch in group.pitches]
        correct = sum(
            predicted == expected
            for predicted, expected in zip(predicted_pitches, truth_pitches, strict=False)
        )
        compared = max(len(predicted_pitches), len(truth_pitches))
        total_predicted += len(predicted_pitches)
        total_truth += len(truth_pitches)
        total_pitch_correct += correct
        total_compared += compared
        results.append(
            {
                "identity": dict(truth["identity"]),
                "predicted_note_count": len(predicted_pitches),
                "truth_note_count": len(truth_pitches),
                "exact_note_count": len(predicted_pitches) == len(truth_pitches),
                "absolute_note_count_error": abs(len(predicted_pitches) - len(truth_pitches)),
                "predicted_group_count": len(predicted_groups),
                "truth_group_count": len(truth_groups),
                "exact_group_count": len(predicted_groups) == len(truth_groups),
                "ordered_pitch_correct": correct,
                "ordered_pitch_compared": compared,
                "ordered_pitch_accuracy": _ratio(correct, compared),
                "exact_ordered_pitches": predicted_pitches == truth_pitches,
                "predicted_ordered_pitches": predicted_pitches,
                "truth_ordered_pitches": truth_pitches,
            }
        )
    count = len(results)
    return {
        "schema_version": 1,
        "scope": {
            "note_count": "evaluated",
            "group_count": "evaluated",
            "ordered_pitch": "evaluated",
            "onset": "out_of_scope",
            "duration": "out_of_scope",
        },
        "summary": {
            "targets": count,
            "predicted_note_count": total_predicted,
            "truth_note_count": total_truth,
            "exact_note_count_rate": _ratio(sum(row["exact_note_count"] for row in results), count),
            "mean_absolute_note_count_error": _ratio(
                sum(row["absolute_note_count_error"] for row in results), count
            ),
            "exact_group_count_rate": _ratio(
                sum(row["exact_group_count"] for row in results), count
            ),
            "ordered_pitch_accuracy": _ratio(total_pitch_correct, total_compared),
            "exact_ordered_pitch_rate": _ratio(
                sum(row["exact_ordered_pitches"] for row in results), count
            ),
        },
        "results": results,
    }


def pitch_name_to_midi(pitch: str) -> int:
    match = PITCH_RE.fullmatch(pitch)
    if match is None:
        raise ValueError(f"Invalid scientific pitch: {pitch!r}")
    letter, accidental, octave_text = match.groups()
    pitch_class = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[letter]
    pitch_class += {"": 0, "#": 1, "b": -1}[accidental]
    midi = (int(octave_text) + 1) * 12 + pitch_class
    if not 0 <= midi <= 127:
        raise ValueError(f"Pitch outside MIDI range: {pitch!r}")
    return midi


def _model_payload(
    selector: LinearSelector,
    pitch_calibrator: PitchCalibrator,
    max_chord_size: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "pure_python_notehead_bootstrap_model",
        "summary": {
            "selector": "class_weighted_logistic_l2",
            "feature_count": len(selector.feature_names),
            "threshold": round(selector.threshold, 12),
            "max_chord_size": max_chord_size,
            "pitch_calibration": "development pseudo-label median MIDI residual by natural letter",
        },
        "selector": {
            "feature_names": list(selector.feature_names),
            "means": [round(value, 12) for value in selector.means],
            "scales": [round(value, 12) for value in selector.scales],
            "weights": [round(value, 12) for value in selector.weights],
            "intercept": round(selector.intercept, 12),
            "threshold": round(selector.threshold, 12),
        },
        "pitch_calibrator": {
            "global_offset": pitch_calibrator.global_offset,
            "offsets_by_letter": pitch_calibrator.offsets_by_letter,
        },
    }


def _write_reports(output_dir: Path, report: Mapping[str, Any]) -> None:
    _write_json(output_dir / "report.json", report)
    (output_dir / "report.md").write_text(_markdown_report(report), encoding="utf-8")


def _markdown_report(report: Mapping[str, Any]) -> str:
    audit = report["audit"]
    gate = report["gate"]
    hybrid = report["hybrid_arm"]
    lines = [
        "# Pickup-aware Hybrid Notehead Sequence Bootstrap",
        "",
        "## Scope",
        "",
        "Cap-24 candidates and features were frozen for development, validation, and sealed "
        "heldout before development truth was opened. Only development systems 1 and 2 were "
        "used for pseudo labels and fitting.",
        "",
        "Onset and duration prediction are out of scope. Reported downstream metrics cover "
        "candidate groups, note counts, and ordered pitches.",
        "",
        "## Original Failed Arm",
        "",
        f"- Precision: `{audit['precision']:.3f}` (minimum `{gate['minimum_precision']:.2f}`)",
        f"- Recall: `{audit['recall']:.3f}` (minimum `{gate['minimum_recall']:.2f}`)",
        f"- TP/FP/FN: `{audit['tp']}/{audit['fp']}/{audit['fn']}`",
        f"- Passed: `{gate['passed']}`",
        "",
        "This original canonical-sequence arm remains failed. Its gate was not relaxed, and "
        "its S1M1 pseudo assignments were not used by the follow-up arm.",
        "",
        "## Preregistered Hybrid Arm",
        "",
        "Promoted S1M1-4 reviews provide exact positive and negative candidate labels. S1M1 "
        "is unsupported for sequence pseudo labeling but remains exact supervised training "
        "data. The confidence filter is calibrated only on reviewed non-pickup S1M2-4.",
        "",
    ]
    calibration = hybrid["pseudo_acceptance_calibration"]
    selected = calibration["selected_metrics"]
    lines.append(f"- Confidence calibration passed: `{calibration['passed']}`")
    if selected:
        lines.extend(
            [
                f"- Frozen rule: `{calibration['selected_rule']}`",
                f"- Calibration TP/FP/FN: `{selected['tp']}/{selected['fp']}/{selected['fn']}`",
                f"- Calibration precision/recall: `{selected['precision']:.3f}` / "
                f"`{selected['recall']:.3f}`",
                "",
            ]
        )
    if not calibration["passed"]:
        lines.extend(
            [
                "The confidence filter could not clear its preregistered gate. The hybrid "
                "arm stopped without training or validation.",
                "",
            ]
        )
        return "\n".join(lines)

    accepted = hybrid["accepted_pseudo_labels"]
    lines.extend(
        [
            f"- Accepted unreviewed pseudo positives: "
            f"`{accepted['accepted_positive_count']}` across "
            f"`{accepted['measure_count']}` measures",
            f"- Exact review positives/negatives: "
            f"`{hybrid['exact_review_labels']['positive_count']}` / "
            f"`{hybrid['exact_review_labels']['negative_count']}`",
            "",
            "## Validation Gate",
            "",
        ]
    )
    validation_gate = hybrid["validation_gate"]
    if validation_gate.get("observed"):
        observed = validation_gate["observed"]
        lines.extend(
            [
                f"- S7 ordered pitch: `{observed['ordered_pitch_accuracy']:.3f}` "
                f"(required `>={VALIDATION_REQUIRED_ORDERED_PITCH}`)",
                f"- S7 count MAE: `{observed['mean_absolute_note_count_error']:.3f}` "
                f"(required `<={VALIDATION_BASELINE_COUNT_MAE}`)",
                f"- Passed: `{validation_gate['passed']}`",
                "",
            ]
        )
    if validation_gate.get("observed") and not validation_gate["passed"]:
        lines.extend(
            [
                "The prior hybrid arm remains stopped at its validation gate; this result "
                "was not changed by the cross-system follow-up.",
                "",
            ]
        )

    cross = report.get("cross_system_arm")
    if not cross:
        return "\n".join(lines)
    coverage = cross["s7_pseudo_labels"]
    lines.extend(
        [
            "## Cross-system Weak Supervision",
            "",
            "System 7 is adaptation/training data. System 8 is model-selection evidence "
            "because prior arms already evaluated it globally. A concurrent preregistered "
            "arm has since opened System 3, so it is unavailable here as an independent "
            "final test. No onset, rhythm, or duration claim is made.",
            "",
            f"- Frozen S1 rule: `{cross['frozen_confidence_rule']['max_match_cost']}` "
            "maximum match cost and zero pitch error",
            f"- Accepted S7 labels: `{coverage['accepted_label_count']}` / "
            f"`{coverage['canonical_note_count']}` canonical notes "
            f"(`{coverage['canonical_note_coverage']:.3f}` coverage)",
            f"- S7 coverage gate passed: `{coverage['passed']}`",
            "- Unaccepted S7 candidates remained unlabeled and were not negative examples.",
            "",
        ]
    )
    if not coverage["passed"]:
        lines.extend(
            [
                "The S7 weak-label coverage gate failed. No S8 prediction or truth read "
                "occurred for this arm.",
                "",
            ]
        )
        return "\n".join(lines)

    s8_gate = cross["s8_gate"]
    observed = s8_gate.get("observed")
    if observed:
        lines.extend(
            [
                "## System 8 Gate",
                "",
                f"- Ordered pitch: `{observed['ordered_pitch_accuracy']:.3f}` "
                f"(required `>={S8_GATE_ORDERED_PITCH}`)",
                f"- Count MAE: `{observed['mean_absolute_note_count_error']:.3f}` "
                f"(required `<={S8_GATE_COUNT_MAE}`)",
                f"- Passed: `{s8_gate['passed']}`",
                "",
            ]
        )
    heldout = cross["heldout"]
    if heldout["status"] in {"not_reached", "skipped_due_coordination"}:
        lines.extend(
            [
                "All S8 predictions were persisted and hashed before S8 truth was parsed. "
                "No S3 prediction was written and this arm did not read S3 truth. S3 is "
                "reported as unavailable for independent evaluation because another "
                "preregistered arm opened it once.",
                "",
            ]
        )
        return "\n".join(lines)

    heldout_summary = cross["metrics"]["system3_heldout"]["summary"]
    lines.extend(
        [
            "## System 3 Final Test",
            "",
            f"- Ordered pitch: `{heldout_summary['ordered_pitch_accuracy']:.3f}`",
            f"- Count MAE: `{heldout_summary['mean_absolute_note_count_error']:.3f}`",
            f"- Exact note-count rate: `{heldout_summary['exact_note_count_rate']:.3f}`",
            "- Evaluation count: `1`",
            "",
        ]
    )
    return "\n".join(lines)


def _alignment_signature(assignments: Sequence[Mapping[str, Any]]) -> str:
    return "|".join(
        f"{assignment['group_index']}:{assignment['status']}:"
        f"{','.join(assignment['candidate_ids'])}"
        for assignment in assignments
    )


def _identity_key(identity: Mapping[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(identity["slug"]),
        int(identity["system_index"]),
        int(identity["system_measure_index"]),
        int(identity["global_measure_index"]),
    )


def _integer_median(values: Sequence[int]) -> int:
    if not values:
        return 0
    value = float(median(values))
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponential = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(max(value, -700.0))
    return exponential / (1.0 + exponential)


def _raw_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _relative_to_out(path: Path, out_dir: Path) -> str:
    return path.resolve().relative_to(out_dir.resolve()).as_posix()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _read_jsonl_prefix(path: Path, row_count: int) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for _ in range(row_count):
            line = handle.readline()
            if not line:
                raise ValueError(f"Expected {row_count} JSONL rows in prefix: {path}")
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
