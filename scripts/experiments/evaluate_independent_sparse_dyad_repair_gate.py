"""Evaluate the frozen Desde Lejos sparse-dyad repair gate exactly once.

The evaluator verifies every frozen artifact and replays the exact truth-blind
repair contract before opening either a raw-image-only review or MusicXML.
Candidate identity and augmentation-dot evidence are scored from raw-image
coordinates; MusicXML is used only for note count, diatonic pitch, and frozen
onset-group chord-size scoring.
"""

from __future__ import annotations

import argparse
import ast
import math
import shutil
import sys
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import evaluate_frozen_third_score_heldout as heldout  # noqa: E402
from scripts.experiments import (  # noqa: E402
    evaluate_independent_multihead_recovery_gate as multi_eval,
)
from scripts.experiments import freeze_independent_sparse_dyad_repair_gate as gate  # noqa: E402
from scripts.experiments import freeze_third_score_heldout as freezer  # noqa: E402
from scripts.experiments import run_independent_sparse_dyad_repair_gate as runner  # noqa: E402
from scripts.experiments import spike_consumed_polyphonic_pitch_repair as recovery  # noqa: E402
from scripts.experiments import spike_consumed_sparse_stem_dyad_repair as repair  # noqa: E402

SCHEMA_VERSION = 1
DEFAULT_EVALUATION_VERSION = "v1"
TARGET_SLUG = gate.DESDE_LEJOS_SLUG
TARGET_SYSTEM_INDEX = gate.TARGET_SYSTEM_INDEX
OUTPUT_SUBDIR = gate.OUTPUT_SUBDIR
DEFAULT_NAMESPACE = gate.DEFAULT_NAMESPACE
EXPECTED_TRANSCRIPTION_FILENAME = "desde_lejos_system_007.musicxml"
EXPECTED_RAW_REVIEW_FILENAME = "desde_lejos_system_007_raw_review.json"
EXPECTED_GATE_ROOT = (
    Path("out/local_restricted")
    / TARGET_SLUG
    / OUTPUT_SUBDIR
    / DEFAULT_NAMESPACE
    / f"system_{TARGET_SYSTEM_INDEX:03d}"
)
EXPECTED_TRANSCRIPTION_PATH = EXPECTED_GATE_ROOT / EXPECTED_TRANSCRIPTION_FILENAME
EXPECTED_RAW_REVIEW_PATH = EXPECTED_GATE_ROOT / EXPECTED_RAW_REVIEW_FILENAME

LANE_COMPARISON = "multihead_recovery"
LANE_REPAIRED = "sparse_dyad_repair"
LANES = (LANE_COMPARISON, LANE_REPAIRED)
RUNNER_SOURCE_NAME = "run_third_score_heldout_inference.py"
RUNNER_EVALUATION_EXCLUDED_FUNCTIONS = {
    "main",
    "materialize_third_score_inference",
    "freeze_inference",
    "_materialize_sparse_dyad_repair_sidecar",
    "_sparse_dyad_repair_row",
    "_sparse_group_indices",
    "_sparse_candidate_record",
    "_write_sparse_dyad_repair_overlay",
    "_verify_sparse_dyad_repair_sidecar",
    "_sparse_repair_module",
}
RUNNER_EVALUATION_EXCLUDED_CONSTANTS = {
    "SPARSE_DYAD_REPAIR_SIDECAR_VERSION",
    "SPARSE_DYAD_REPAIR_DIRNAME",
}
RAW_REVIEW_KIND = "independent_sparse_dyad_repair_raw_image_review"
RAW_REVIEW_STATUS = "completed_after_frozen_predictions"
DEFAULT_HEAD_TOLERANCE_PX = 7.0
DEFAULT_DOT_TOLERANCE_PX = 7.0

TruthLoader = Callable[[Path], heldout.VisibleMusicXMLTruth]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Expected local inputs:\n"
            f"  MusicXML: {EXPECTED_TRANSCRIPTION_PATH.as_posix()}\n"
            f"  raw review: {EXPECTED_RAW_REVIEW_PATH.as_posix()}"
        ),
    )
    parser.add_argument("sealed_manifest", type=Path)
    parser.add_argument("--musicxml", type=Path, required=True)
    parser.add_argument("--raw-review", type=Path, required=True)
    parser.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help="Optional crop-to-physical-measure mapping when counts are not one-to-one.",
    )
    parser.add_argument(
        "--evaluation-version",
        default=DEFAULT_EVALUATION_VERSION,
        help="Create-once output version (default: v1, written as evaluation_v1).",
    )
    args = parser.parse_args(argv)
    try:
        result = evaluate_independent_sparse_dyad_repair_gate(
            args.sealed_manifest,
            musicxml_path=args.musicxml,
            raw_review_path=args.raw_review,
            mapping_path=args.mapping,
            evaluation_version=args.evaluation_version,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        OSError,
        ValueError,
        ET.ParseError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(result["report"])
    return 0


def evaluate_independent_sparse_dyad_repair_gate(
    sealed_manifest_path: Path,
    *,
    musicxml_path: Path,
    raw_review_path: Path,
    mapping_path: Path | None = None,
    evaluation_version: str = DEFAULT_EVALUATION_VERSION,
    truth_loader: TruthLoader = heldout.load_visible_musicxml_truth,
) -> dict[str, str]:
    """Verify the blind gate, then score raw review and MusicXML exactly once."""
    heldout._validate_version(evaluation_version)
    sealed_manifest_path = sealed_manifest_path.expanduser().resolve()
    musicxml_path = musicxml_path.expanduser().resolve()
    raw_review_path = raw_review_path.expanduser().resolve()
    mapping_path = mapping_path.expanduser().resolve() if mapping_path is not None else None

    # This must remain the first operation that opens any caller-supplied input.
    frozen = verify_frozen_sparse_dyad_gate(sealed_manifest_path)
    namespace_root = frozen["namespace_root"]
    output_dir = namespace_root / f"evaluation_{evaluation_version}"
    temp_dir = namespace_root / f".evaluation_{evaluation_version}.tmp"
    prior_evaluations = sorted(namespace_root.glob("evaluation_*"))
    if prior_evaluations:
        raise FileExistsError(
            "Independent sparse-dyad evaluation already exists: "
            + ", ".join(str(path) for path in prior_evaluations)
        )
    stale_temps = sorted(namespace_root.glob(".evaluation_*.tmp"))
    if stale_temps:
        raise FileExistsError(
            "Stale independent sparse-dyad evaluation directory exists: "
            + ", ".join(str(path) for path in stale_temps)
        )

    # Raw-image evidence is opened before the transcription and is validated
    # without using MusicXML count, pitch, or onset information.
    if not raw_review_path.is_file():
        raise FileNotFoundError(f"Raw-image review does not exist: {raw_review_path}")
    raw_review_sha256 = freezer._sha256(raw_review_path)
    raw_review_payload = heldout._read_json(raw_review_path)
    raw_review = validate_raw_image_review(raw_review_payload, frozen=frozen)

    if not musicxml_path.is_file():
        raise FileNotFoundError(f"User MusicXML does not exist: {musicxml_path}")
    if mapping_path is not None and not mapping_path.is_file():
        raise FileNotFoundError(f"Mapping JSON does not exist: {mapping_path}")
    musicxml_sha256 = freezer._sha256(musicxml_path)
    mapping_sha256 = freezer._sha256(mapping_path) if mapping_path is not None else None
    truth = truth_loader(musicxml_path)

    crop_indices = tuple(sorted(frozen["paired_rows_by_crop"]))
    if mapping_path is None:
        mapping_payload = multi_eval._default_mapping(truth.measure_numbers, crop_indices)
        mapping_mode = "deterministic_equal_count_one_to_one"
    else:
        mapping_payload = heldout._load_mapping(mapping_path)
        mapping_mode = "explicit_user_mapping"
    mapping = heldout.validate_and_materialize_mapping(
        mapping_payload,
        truth=truth,
        crop_indices=crop_indices,
        mode=mapping_mode,
    )
    truth_rows = heldout.build_truth_rows(frozen["requests_by_crop"], truth, mapping)
    structure_support = multi_eval._mapping_structure_support(mapping, truth)
    lane_reports = {
        lane: multi_eval._score_lane(
            truth_rows,
            frozen["paired_rows_by_crop"],
            lane=lane,
            structure_support=structure_support,
        )
        for lane in LANES
    }
    report = _build_report(
        target=frozen["target"],
        mapping_mode=mapping_mode,
        truth=truth,
        lane_reports=lane_reports,
        structure_support=structure_support,
        raw_review=raw_review,
    )

    temp_dir.mkdir(parents=False, exist_ok=False)
    try:
        source_snapshot = temp_dir / "source.musicxml"
        raw_review_snapshot = temp_dir / "raw_image_review.json"
        mapping_snapshot = temp_dir / "mapping.json"
        truth_snapshot = temp_dir / "truth.jsonl"
        paired_snapshot = temp_dir / "frozen_paired_predictions.jsonl"
        diagnostics_snapshot = temp_dir / "frozen_diagnostics.jsonl"
        freeze_snapshot = temp_dir / "frozen_freeze.json"
        sealed_snapshot = temp_dir / "frozen_sealed_manifest.json"
        evaluator_snapshot = temp_dir / "evaluator.py"
        report_path = temp_dir / "report.json"

        shutil.copyfile(musicxml_path, source_snapshot)
        shutil.copyfile(raw_review_path, raw_review_snapshot)
        heldout._write_json(mapping_snapshot, mapping)
        heldout._write_jsonl(truth_snapshot, truth_rows)
        shutil.copyfile(frozen["predictions_path"], paired_snapshot)
        shutil.copyfile(frozen["diagnostics_path"], diagnostics_snapshot)
        shutil.copyfile(frozen["freeze_path"], freeze_snapshot)
        shutil.copyfile(sealed_manifest_path, sealed_snapshot)
        shutil.copyfile(Path(__file__).resolve(), evaluator_snapshot)

        pins = {
            "source_musicxml": heldout._snapshot_record(
                source_snapshot,
                source_path=musicxml_path,
                source_sha256=musicxml_sha256,
            ),
            "raw_image_review": heldout._snapshot_record(
                raw_review_snapshot,
                source_path=raw_review_path,
                source_sha256=raw_review_sha256,
            ),
            "mapping": heldout._snapshot_record(
                mapping_snapshot,
                source_path=mapping_path,
                source_sha256=mapping_sha256,
                require_source_match=False,
            ),
            "truth": heldout._snapshot_record(truth_snapshot),
            "paired_predictions": heldout._snapshot_record(
                paired_snapshot,
                source_path=frozen["predictions_path"],
                source_sha256=frozen["predictions_sha256"],
            ),
            "diagnostics": heldout._snapshot_record(
                diagnostics_snapshot,
                source_path=frozen["diagnostics_path"],
                source_sha256=frozen["diagnostics_sha256"],
            ),
            "freeze_manifest": heldout._snapshot_record(
                freeze_snapshot,
                source_path=frozen["freeze_path"],
                source_sha256=frozen["freeze_sha256"],
            ),
            "sealed_manifest": heldout._snapshot_record(
                sealed_snapshot,
                source_path=sealed_manifest_path,
                source_sha256=frozen["sealed_sha256"],
            ),
            "evaluator": heldout._snapshot_record(
                evaluator_snapshot,
                source_path=Path(__file__).resolve(),
                source_sha256=freezer._sha256(Path(__file__).resolve()),
            ),
        }
        report["pins"] = pins
        heldout._write_json(report_path, report)
        pins["report"] = heldout._snapshot_record(report_path)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "independent_sparse_dyad_repair_post_freeze_evaluation_manifest",
            "status": "evaluated_exactly_once_after_frozen_predictions",
            "create_once": True,
            "evaluation_version": evaluation_version,
            "target": frozen["target"],
            "truth_opened_after_all_frozen_hashes_verified": True,
            "raw_review_opened_after_all_frozen_hashes_verified": True,
            "raw_review_opened_before_musicxml": True,
            "mapping_mode": mapping_mode,
            "pins": pins,
        }
        heldout._write_json(temp_dir / "manifest.json", manifest)

        verified_again = verify_frozen_sparse_dyad_gate(sealed_manifest_path)
        if (
            verified_again["freeze_sha256"] != frozen["freeze_sha256"]
            or verified_again["sealed_sha256"] != frozen["sealed_sha256"]
        ):
            raise ValueError("Frozen independent sparse-dyad gate changed during evaluation")
        temp_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return {
        "evaluation_dir": str(output_dir),
        "manifest": str(output_dir / "manifest.json"),
        "report": str(output_dir / "report.json"),
        "truth": str(output_dir / "truth.jsonl"),
        "mapping": str(output_dir / "mapping.json"),
        "raw_review": str(output_dir / "raw_image_review.json"),
    }


def verify_frozen_sparse_dyad_gate(sealed_manifest_path: Path) -> dict[str, Any]:
    """Verify every frozen role and replay the exact fixed repair contract."""
    if not sealed_manifest_path.is_file():
        raise FileNotFoundError(f"Sealed manifest does not exist: {sealed_manifest_path}")
    sealed = heldout._read_json(sealed_manifest_path)
    expected_target = {"slug": TARGET_SLUG, "system_index": TARGET_SYSTEM_INDEX}
    if sealed.get("kind") != "independent_sparse_dyad_repair_sealed_manifest":
        raise ValueError(f"Unexpected sparse-dyad sealed kind: {sealed.get('kind')}")
    if sealed.get("status") != "frozen_awaiting_truth":
        raise ValueError("Independent sparse-dyad seal is not awaiting truth")
    if sealed.get("truth_accessed") is not False or sealed.get("truth_used") is not False:
        raise ValueError("Independent sparse-dyad seal is not truth-blind")
    if sealed.get("target") != expected_target:
        raise ValueError(f"Unexpected independent sparse-dyad target: {sealed.get('target')}")

    frozen_dir = sealed_manifest_path.parent.resolve()
    namespace_root = frozen_dir.parent.resolve()
    freeze_path = heldout._safe_child(frozen_dir, str(sealed["freeze"]["path"]))
    if freeze_path != frozen_dir / "freeze.json":
        raise ValueError(f"Unexpected sparse-dyad freeze path: {freeze_path}")
    freeze_sha256 = freezer._sha256(freeze_path)
    if freeze_sha256 != str(sealed["freeze"]["sha256"]):
        raise ValueError("Independent sparse-dyad freeze manifest hash drift")
    runner.verify_freeze(frozen_dir)
    freeze = heldout._read_json(freeze_path)
    if freeze.get("kind") != "independent_sparse_dyad_repair_freeze":
        raise ValueError(f"Unexpected sparse-dyad freeze kind: {freeze.get('kind')}")
    if freeze.get("target") != expected_target:
        raise ValueError("Independent sparse-dyad frozen target mismatch")
    if freeze.get("truth_accessed") is not False or freeze.get("truth_used") is not False:
        raise ValueError("Independent sparse-dyad freeze is not truth-blind")
    if (
        freeze.get("comparison_config_id") != recovery.EDGE_SAFE_STEM_MULTIHEAD_CONFIG_ID
        or freeze.get("repair_config_id") != repair.CONFIG_ID
        or freeze.get("repair_parameters") != repair.PARAMETERS
    ):
        raise ValueError("Independent sparse-dyad fixed-rule contract drift")
    pins = freeze.get("pins")
    if not isinstance(pins, Mapping):
        raise ValueError("Independent sparse-dyad freeze has no role-based pins")
    _verify_current_implementation_bindings(pins, namespace_root=namespace_root)

    predictions_pin = _only_pin(pins, "paired_predictions")
    predictions_path = _pin_snapshot(namespace_root, predictions_pin)
    paired_rows = heldout._read_jsonl(predictions_path)
    paired_rows_by_crop = heldout._rows_by_crop(paired_rows, label="paired predictions")

    prepared_pin = _find_pin(
        pins,
        "prepared_and_source",
        filename="prepared_manifest.json",
        expected_sha256=str(freeze["prepared_manifest_sha256"]),
    )
    prepared_path = _pin_snapshot(namespace_root, prepared_pin)
    prepared = heldout._read_json(prepared_path)
    _validate_prepared_snapshot(prepared, expected_target=expected_target)
    prepared_source = _pin_source_path(prepared_pin)
    requests_record = prepared["artifacts"]["requests"]
    requests_source = prepared_source.parent / str(requests_record["path"])
    requests_pin = _find_pin_by_source(pins, "prepared_and_source", requests_source)
    requests_path = _pin_snapshot(namespace_root, requests_pin)
    request_rows = heldout._read_jsonl(requests_path)
    expected_row_hashes = tuple(str(value) for value in requests_record.get("row_sha256") or [])
    actual_row_hashes = tuple(freezer._hash_json(row) for row in request_rows)
    if actual_row_hashes != expected_row_hashes:
        raise ValueError("Independent sparse-dyad request row hash drift")
    requests_by_crop = heldout._rows_by_crop(request_rows, label="requests")

    generic_pin = _find_unique_pin(pins, "baseline_inference", filename="inference.jsonl")
    generic_path = _pin_snapshot(namespace_root, generic_pin)
    generic_rows = heldout._read_jsonl(generic_path)
    generic_rows_by_crop = heldout._rows_by_crop(generic_rows, label="generic inference")
    if set(generic_rows_by_crop) != set(paired_rows_by_crop):
        raise ValueError("Independent sparse-dyad generic/paired crop identities differ")
    _verify_generic_source_images(generic_rows, pins=pins, namespace_root=namespace_root)

    model_pin = _find_unique_pin(pins, "model_and_training", filename="model.json")
    model_payload = heldout._read_json(_pin_snapshot(namespace_root, model_pin))
    selector = recovery.selector_config_from_model(model_payload)
    contract = runner.verify_paired_contract(
        paired_rows,
        generic_rows,
        selector=selector,
        expected_target=expected_target,
    )
    contract_pin = _find_pin(
        pins,
        "paired_artifacts",
        filename="truth_blind_contract.json",
        expected_sha256=str(freeze["truth_blind_contract_sha256"]),
    )
    if heldout._read_json(_pin_snapshot(namespace_root, contract_pin)) != contract:
        raise ValueError("Independent sparse-dyad truth-blind contract drift")
    if int(contract["accepted_repair_count"]) != int(freeze["accepted_repair_count"]):
        raise ValueError("Independent sparse-dyad accepted-repair count drift")

    diagnostics_pin = _find_unique_pin(pins, "paired_artifacts", filename="diagnostics.jsonl")
    diagnostics_path = _pin_snapshot(namespace_root, diagnostics_pin)
    diagnostics_rows = heldout._read_jsonl(diagnostics_path)
    diagnostics_by_crop = heldout._rows_by_crop(diagnostics_rows, label="diagnostics")
    if set(diagnostics_by_crop) != set(paired_rows_by_crop):
        raise ValueError("Independent sparse-dyad diagnostics/prediction crop identities differ")
    _validate_targets(request_rows, label="request")
    _validate_targets(generic_rows, label="generic inference")
    _validate_targets(paired_rows, label="paired prediction")
    _validate_targets(diagnostics_rows, label="diagnostics")

    return {
        "namespace_root": namespace_root,
        "sealed_sha256": freezer._sha256(sealed_manifest_path),
        "freeze_path": freeze_path,
        "freeze_sha256": freeze_sha256,
        "predictions_path": predictions_path,
        "predictions_sha256": str(predictions_pin["snapshot_sha256"]),
        "diagnostics_path": diagnostics_path,
        "diagnostics_sha256": str(diagnostics_pin["snapshot_sha256"]),
        "target": expected_target,
        "requests_by_crop": requests_by_crop,
        "generic_rows_by_crop": generic_rows_by_crop,
        "paired_rows_by_crop": paired_rows_by_crop,
        "diagnostics_by_crop": diagnostics_by_crop,
        "prepared_pins": tuple(pins["prepared_and_source"]),
    }


def validate_raw_image_review(
    payload: Mapping[str, Any],
    *,
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    """Match raw-only centers to the accepted frozen repair without MusicXML."""
    if payload.get("kind") != RAW_REVIEW_KIND or payload.get("status") != RAW_REVIEW_STATUS:
        raise ValueError("Raw-image review kind/status mismatch")
    if payload.get("review_mode") != "raw_image_only":
        raise ValueError("Sparse-dyad review must use raw_image_only mode")
    if payload.get("target") != frozen["target"]:
        raise ValueError("Raw-image review target mismatch")
    if payload.get("automatic_overlay_visible") is not False:
        raise ValueError("Raw-image review must not expose automatic overlays")
    if payload.get("musicxml_visible") is not False:
        raise ValueError("Raw-image review must be completed without MusicXML")
    measures = payload.get("measures")
    if not isinstance(measures, list) or any(not isinstance(item, Mapping) for item in measures):
        raise ValueError("Raw-image review requires a measures list")

    accepted = {
        crop: row["sparse_repair"]
        for crop, row in frozen["diagnostics_by_crop"].items()
        if row["sparse_repair"].get("accepted") is True
    }
    reviewed = {int(item["automatic_crop_index"]): item for item in measures}
    if set(reviewed) != set(accepted):
        raise ValueError(
            "Raw-image review must cover exactly the accepted repair crops: "
            f"expected {sorted(accepted)}, got {sorted(reviewed)}"
        )

    crop_reports = []
    for crop in sorted(accepted):
        review = reviewed[crop]
        generic = frozen["generic_rows_by_crop"][crop]
        source = generic.get("source")
        if not isinstance(source, Mapping) or not isinstance(source.get("sha256"), str):
            raise ValueError(f"Frozen generic row {crop} has no source image hash")
        if review.get("raw_image_sha256") != source["sha256"]:
            raise ValueError(f"Raw-image review source hash mismatch for crop {crop}")
        head_points = _review_points(review, "notehead_centers")
        dot_points = _review_points(review, "augmentation_dot_centers")
        decision = accepted[crop]
        chosen = decision.get("chosen_pair")
        if not isinstance(chosen, Mapping):
            raise ValueError(f"Accepted repair crop {crop} has no chosen pair")
        candidates = _candidate_centers(frozen["generic_rows_by_crop"][crop])
        proposed_ids = tuple(str(value) for value in decision["proposed_ids"])
        displaced_ids = tuple(str(value) for value in decision["current_ids"])
        head_tolerance = float(review.get("head_tolerance_px", DEFAULT_HEAD_TOLERANCE_PX))
        dot_tolerance = float(review.get("dot_tolerance_px", DEFAULT_DOT_TOLERANCE_PX))
        head_matches = _match_candidate_ids(
            proposed_ids,
            candidates,
            head_points,
            tolerance=head_tolerance,
        )
        displaced_matches = _match_candidate_ids(
            displaced_ids,
            candidates,
            head_points,
            tolerance=head_tolerance,
            require_all=False,
        )
        confirmed_dot_pairs = []
        rejected_dot_pairs = []
        for pair in chosen.get("augmentation_dot_pairs") or []:
            candidate_ids = tuple(str(value) for value in pair.get("candidate_ids") or [])
            matches = _match_candidate_ids(
                candidate_ids,
                candidates,
                dot_points,
                tolerance=dot_tolerance,
                require_all=False,
            )
            record = {
                "candidate_ids": list(candidate_ids),
                "matched_candidate_ids": sorted(matches),
            }
            if len(matches) == len(candidate_ids) == 2:
                confirmed_dot_pairs.append(record)
            else:
                rejected_dot_pairs.append(record)
        if len(head_matches) != len(proposed_ids):
            raise ValueError(f"Raw review does not confirm every proposed head in crop {crop}")
        if displaced_matches:
            raise ValueError(
                f"Raw review still identifies displaced anchors as noteheads in crop {crop}: "
                f"{sorted(displaced_matches)}"
            )
        if not confirmed_dot_pairs:
            raise ValueError(f"Raw review confirms no frozen augmentation-dot pair in crop {crop}")
        crop_reports.append(
            {
                "automatic_crop_index": crop,
                "raw_image_sha256": source["sha256"],
                "reviewed_notehead_count": len(head_points),
                "reviewed_augmentation_dot_count": len(dot_points),
                "proposed_head_matches": head_matches,
                "displaced_head_matches": displaced_matches,
                "confirmed_augmentation_dot_pairs": confirmed_dot_pairs,
                "unconfirmed_augmentation_dot_pairs": rejected_dot_pairs,
                "head_pixel_identity_passed": True,
                "augmentation_dot_evidence_passed": True,
            }
        )
    return {
        "status": "scored_raw_image_only",
        "accepted_repair_crop_count": len(crop_reports),
        "head_pixel_identity_passed": all(
            item["head_pixel_identity_passed"] for item in crop_reports
        ),
        "augmentation_dot_evidence_passed": all(
            item["augmentation_dot_evidence_passed"] for item in crop_reports
        ),
        "crops": crop_reports,
    }


def _review_points(review: Mapping[str, Any], key: str) -> list[dict[str, float]]:
    values = review.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"Raw-image review requires non-empty {key}")
    points = []
    for value in values:
        if not isinstance(value, Mapping) or value.get("x") is None or value.get("y") is None:
            raise ValueError(f"Raw-image review {key} contains an invalid point")
        points.append({"x": float(value["x"]), "y": float(value["y"])})
    return points


def _candidate_centers(row: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    candidates = row.get("candidate_predictions")
    if not isinstance(candidates, list):
        raise ValueError("Frozen generic inference has no candidate_predictions")
    result = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("Frozen candidate prediction is not an object")
        candidate_id = str(candidate.get("candidate_id", ""))
        center = candidate.get("center")
        if (
            not candidate_id
            or not isinstance(center, Mapping)
            or center.get("x") is None
            or center.get("y") is None
        ):
            raise ValueError("Frozen candidate prediction has no identity/center")
        result[candidate_id] = {"x": float(center["x"]), "y": float(center["y"])}
    return result


def _match_candidate_ids(
    candidate_ids: Sequence[str],
    candidates: Mapping[str, Mapping[str, float]],
    review_points: Sequence[Mapping[str, float]],
    *,
    tolerance: float,
    require_all: bool = True,
) -> dict[str, dict[str, Any]]:
    if tolerance <= 0:
        raise ValueError("Raw-review matching tolerance must be positive")
    unmatched_points = set(range(len(review_points)))
    matches: dict[str, dict[str, Any]] = {}
    for candidate_id in candidate_ids:
        center = candidates.get(candidate_id)
        if center is None:
            raise ValueError(f"Frozen raw-review candidate is missing: {candidate_id}")
        distances = [
            (
                math.hypot(
                    float(center["x"]) - float(review_points[index]["x"]),
                    float(center["y"]) - float(review_points[index]["y"]),
                ),
                index,
            )
            for index in unmatched_points
        ]
        if not distances:
            continue
        distance, index = min(distances)
        if distance <= tolerance:
            unmatched_points.remove(index)
            matches[candidate_id] = {
                "candidate_center": dict(center),
                "review_center": dict(review_points[index]),
                "distance_px": round(distance, 3),
            }
    if require_all and len(matches) != len(candidate_ids):
        missing = sorted(set(candidate_ids) - set(matches))
        raise ValueError(f"Raw review did not match candidate centers: {missing}")
    return matches


def _build_report(
    *,
    target: Mapping[str, Any],
    mapping_mode: str,
    truth: heldout.VisibleMusicXMLTruth,
    lane_reports: Mapping[str, Mapping[str, Any]],
    structure_support: Mapping[int, Mapping[str, Any]],
    raw_review: Mapping[str, Any],
) -> dict[str, Any]:
    comparison = lane_reports[LANE_COMPARISON]["summary"]
    repaired = lane_reports[LANE_REPAIRED]["summary"]
    mapping_unsupported = sum(
        not bool(record["well_defined"]) for record in structure_support.values()
    )
    structure_supported = min(
        comparison["structure_supported_crop_count"],
        repaired["structure_supported_crop_count"],
    )
    structure_status = "scored"
    if mapping_unsupported == len(structure_support):
        structure_status = multi_eval.NOT_SCORED_AMBIGUOUS_MAPPING
    elif structure_supported == 0:
        structure_status = multi_eval.NOT_SCORED_MISSING_GROUPS
    elif structure_supported < len(structure_support):
        structure_status = "partially_scored_mapping_safe_crops_only"
    comparison_crops = {
        int(row["automatic_crop_index"]): row for row in lane_reports[LANE_COMPARISON]["crops"]
    }
    repaired_crops = {
        int(row["automatic_crop_index"]): row for row in lane_reports[LANE_REPAIRED]["crops"]
    }
    repaired_measure_comparisons = []
    for raw_crop in raw_review["crops"]:
        crop = int(raw_crop["automatic_crop_index"])
        repaired_measure_comparisons.append(
            {
                "automatic_crop_index": crop,
                "raw_image_review": raw_crop,
                "comparison_lane": comparison_crops[crop],
                "repair_lane": repaired_crops[crop],
            }
        )
    metric_support = {
        "candidate_pixel_identity": "scored_raw_image_only",
        "augmentation_dot_evidence": "scored_raw_image_only",
        "note_count_precision_recall_f1": "scored",
        "ordered_diatonic_pitch_staff_position": "scored_ignore_key_signature_accidentals",
        "onset_group_chord_size_structure": structure_status,
        "key_signature_and_accidentals": multi_eval.NOT_SCORED_LOCALIZATION_GATE,
        "meter": multi_eval.NOT_SCORED_LOCALIZATION_GATE,
        "absolute_onset_and_rhythm": multi_eval.NOT_SCORED_LOCALIZATION_GATE,
        "duration": multi_eval.NOT_SCORED_LOCALIZATION_GATE,
        "rests": multi_eval.NOT_SCORED_LOCALIZATION_GATE,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "independent_sparse_dyad_repair_one_shot_evaluation",
        "status": "evaluated_exactly_once_after_frozen_predictions",
        "target": dict(target),
        "mapping_mode": mapping_mode,
        "raw_image_review": dict(raw_review),
        "metric_support": metric_support,
        "frozen_context": {
            "key_signature": "unknown",
            "meter": "unsupported_not_frozen",
            "rhythm": "unsupported_not_frozen",
            "rests": "unsupported_not_frozen",
        },
        "source_musicxml_context_not_scored": {
            "time_signature": truth.time_signature,
            "key_fifths": truth.key_fifths,
            "key_events": [list(event) for event in truth.key_events],
            "clef": list(truth.clef) if truth.clef is not None else None,
        },
        "lanes": dict(lane_reports),
        "accepted_repair_crops": repaired_measure_comparisons,
        "comparison": {
            "predicted_note_count_delta": repaired["predicted_note_count"]
            - comparison["predicted_note_count"],
            "note_count_f1_delta": round(
                repaired["note_count_f1"] - comparison["note_count_f1"], 6
            ),
            "exact_diatonic_staff_position_match_delta": (
                repaired["exact_diatonic_staff_position_matches"]
                - comparison["exact_diatonic_staff_position_matches"]
            ),
            "ordered_diatonic_alignment_accuracy_delta": round(
                repaired["ordered_diatonic_alignment_accuracy"]
                - comparison["ordered_diatonic_alignment_accuracy"],
                6,
            ),
            "exact_chord_size_match_delta": repaired["exact_chord_size_matches"]
            - comparison["exact_chord_size_matches"],
            "exact_structure_crop_delta": repaired["exact_structure_crops"]
            - comparison["exact_structure_crops"],
        },
        "unsupported": {
            name: {"status": status}
            for name, status in metric_support.items()
            if status.startswith("not_scored")
        },
    }


def _validate_prepared_snapshot(
    prepared: Mapping[str, Any], *, expected_target: Mapping[str, Any]
) -> None:
    if prepared.get("kind") != gate.INDEPENDENT_SPARSE_DYAD_REPAIR_GATE.prepare_kind:
        raise ValueError("Independent sparse-dyad pinned prepared-manifest kind mismatch")
    if prepared.get("status") != "prepared_awaiting_model_predictions":
        raise ValueError("Independent sparse-dyad pinned preparation has wrong status")
    if prepared.get("split") != freezer.SPLIT_NAME or prepared.get("truth_accessed") is not False:
        raise ValueError("Independent sparse-dyad pinned preparation is not truth-blind")
    if prepared.get("target") != expected_target:
        raise ValueError("Independent sparse-dyad pinned prepared target mismatch")
    config = prepared.get("independent_sparse_dyad_repair_gate")
    if not isinstance(config, Mapping):
        raise ValueError("Independent sparse-dyad pinned preparation has no repair contract")
    if (
        config.get("config_id") != repair.CONFIG_ID
        or config.get("parameters") != repair.PARAMETERS
        or config.get("truth_accessed") is not False
        or config.get("truth_used") is not False
    ):
        raise ValueError("Independent sparse-dyad pinned repair contract drift")


def _verify_current_implementation_bindings(
    pins: Mapping[str, Any], *, namespace_root: Path
) -> None:
    records = pins.get("implementations")
    if not isinstance(records, list) or not records:
        raise ValueError("Independent sparse-dyad freeze has no implementation pins")
    for pin in records:
        if not isinstance(pin, Mapping):
            raise ValueError("Independent sparse-dyad implementation pin is invalid")
        source = _pin_source_path(pin)
        if not source.is_file():
            raise FileNotFoundError(f"Frozen implementation source is missing: {source}")
        if freezer._sha256(source) == str(pin["snapshot_sha256"]):
            continue
        if source.name == RUNNER_SOURCE_NAME:
            snapshot = _pin_snapshot(namespace_root, pin)
            if _runner_evaluation_surface(source) == _runner_evaluation_surface(snapshot):
                continue
        raise ValueError(f"Current implementation differs from frozen source: {source}")


def _runner_evaluation_surface(path: Path) -> str:
    """Fingerprint runner code used while verifying an existing frozen gate.

    Inference materialization and optional sidecars are intentionally excluded: the
    evaluator consumes already sealed inference rows and never executes those paths.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    retained = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.name in RUNNER_EVALUATION_EXCLUDED_FUNCTIONS
        ):
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and (
            _assigned_names(node) & RUNNER_EVALUATION_EXCLUDED_CONSTANTS
        ):
            continue
        retained.append(node)
    surface = ast.Module(body=retained, type_ignores=[])
    return ast.dump(surface, annotate_fields=True, include_attributes=False)


def _assigned_names(node: ast.Assign | ast.AnnAssign) -> set[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return {target.id for target in targets if isinstance(target, ast.Name)}


def _verify_generic_source_images(
    rows: Sequence[Mapping[str, Any]],
    *,
    pins: Mapping[str, Any],
    namespace_root: Path,
) -> None:
    prepared_pins = pins.get("prepared_and_source")
    if not isinstance(prepared_pins, list):
        raise ValueError("Independent sparse-dyad freeze has no prepared/source pins")
    pinned_hashes = {
        str(pin["snapshot_sha256"])
        for pin in prepared_pins
        if isinstance(pin, Mapping)
        and _pin_snapshot(namespace_root, pin).suffix.lower() in {".png", ".jpg", ".jpeg"}
    }
    for row in rows:
        source = row.get("source")
        if not isinstance(source, Mapping):
            raise ValueError("Independent sparse-dyad generic row has no source")
        image_path = Path(str(source.get("image", ""))).expanduser()
        if not image_path.is_absolute():
            image_path = REPO_ROOT / image_path
        image_path = image_path.resolve()
        expected_sha256 = str(source.get("sha256", ""))
        if not image_path.is_file() or freezer._sha256(image_path) != expected_sha256:
            raise ValueError(f"Independent sparse-dyad source image drift: {image_path}")
        if expected_sha256 not in pinned_hashes:
            raise ValueError(f"Independent sparse-dyad source image is not frozen: {image_path}")


def _validate_targets(rows: Sequence[Mapping[str, Any]], *, label: str) -> None:
    for row in rows:
        identity = row.get("identity")
        if not isinstance(identity, Mapping):
            raise ValueError(f"Independent sparse-dyad {label} has no identity")
        if (
            identity.get("slug") != TARGET_SLUG
            or int(identity.get("system_index", -1)) != TARGET_SYSTEM_INDEX
        ):
            raise ValueError(f"Independent sparse-dyad {label} target mismatch")
        if (
            row.get("truth_accessed", False) is not False
            or row.get("truth_used", False) is not False
        ):
            raise ValueError(f"Independent sparse-dyad {label} is not truth-blind")


def _only_pin(pins: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    records = pins.get(role)
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], Mapping):
        raise ValueError(f"Independent sparse-dyad freeze requires one {role} pin")
    return records[0]


def _find_pin(
    pins: Mapping[str, Any],
    role: str,
    *,
    filename: str,
    expected_sha256: str,
) -> Mapping[str, Any]:
    records = pins.get(role)
    if not isinstance(records, list):
        raise ValueError(f"Independent sparse-dyad freeze has no {role} role")
    matches = [
        pin
        for pin in records
        if isinstance(pin, Mapping)
        and Path(str(pin.get("source_path", ""))).name == filename
        and str(pin.get("snapshot_sha256")) == expected_sha256
    ]
    if len(matches) != 1:
        raise ValueError(f"Cannot uniquely locate frozen {role}/{filename} at pinned hash")
    return matches[0]


def _find_unique_pin(pins: Mapping[str, Any], role: str, *, filename: str) -> Mapping[str, Any]:
    records = pins.get(role)
    if not isinstance(records, list):
        raise ValueError(f"Independent sparse-dyad freeze has no {role} role")
    matches = [
        pin
        for pin in records
        if isinstance(pin, Mapping) and Path(str(pin.get("source_path", ""))).name == filename
    ]
    if len(matches) != 1:
        raise ValueError(f"Cannot uniquely locate frozen {role}/{filename}")
    return matches[0]


def _find_pin_by_source(
    pins: Mapping[str, Any], role: str, expected_source: Path
) -> Mapping[str, Any]:
    records = pins.get(role)
    if not isinstance(records, list):
        raise ValueError(f"Independent sparse-dyad freeze has no {role} role")
    expected = expected_source.expanduser().resolve()
    matches = [
        pin for pin in records if isinstance(pin, Mapping) and _pin_source_path(pin) == expected
    ]
    if len(matches) != 1:
        raise ValueError(f"Cannot locate frozen source: {expected}")
    return matches[0]


def _pin_source_path(pin: Mapping[str, Any]) -> Path:
    source = Path(str(pin["source_path"])).expanduser()
    if not source.is_absolute():
        source = REPO_ROOT / source
    return source.resolve()


def _pin_snapshot(namespace_root: Path, pin: Mapping[str, Any]) -> Path:
    return heldout._safe_child(namespace_root, str(pin["snapshot_path_relative_to_namespace"]))


if __name__ == "__main__":
    raise SystemExit(main())
