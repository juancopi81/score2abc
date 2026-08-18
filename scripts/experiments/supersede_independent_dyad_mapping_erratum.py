"""Supersede an invalid dyad-gate evaluation without rewriting its audit trail.

This tool is intentionally narrow. It reuses the immutable MusicXML and frozen
prediction snapshots from a completed evaluation, applies a corrected crop
mapping, and writes one create-once erratum directory beside the original.
"""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import evaluate_frozen_third_score_heldout as heldout  # noqa: E402
from scripts.experiments import evaluate_independent_dyad_recovery_gate as evaluator  # noqa: E402
from scripts.experiments import freeze_third_score_heldout as freezer  # noqa: E402

SCHEMA_VERSION = 1
OUTPUT_VERSION = "v2_mapping_erratum"
ERRATUM_REASON = (
    "The original post-freeze mapping assigned three onset groups to automatic crop 1 "
    "and one to crop 2. Frozen x-group evidence and the source crop show two onset "
    "groups in each crop; only the measure-1 note slices are corrected."
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sealed_manifest", type=Path)
    parser.add_argument("--prior-evaluation", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = supersede_mapping_erratum(
            args.sealed_manifest,
            prior_evaluation_manifest=args.prior_evaluation,
            corrected_mapping_path=args.mapping,
        )
    except (FileExistsError, FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(result["report"])
    return 0


def supersede_mapping_erratum(
    sealed_manifest_path: Path,
    *,
    prior_evaluation_manifest: Path,
    corrected_mapping_path: Path,
) -> dict[str, str]:
    sealed_manifest_path = sealed_manifest_path.expanduser().resolve()
    prior_evaluation_manifest = prior_evaluation_manifest.expanduser().resolve()
    corrected_mapping_path = corrected_mapping_path.expanduser().resolve()

    frozen = evaluator.verify_frozen_dyad_gate(sealed_manifest_path)
    prior = _verify_prior_evaluation(
        prior_evaluation_manifest,
        expected_target=frozen["target"],
        expected_sealed_sha256=frozen["sealed_sha256"],
        expected_predictions_sha256=frozen["predictions_sha256"],
    )
    output_dir = frozen["namespace_root"] / f"evaluation_{OUTPUT_VERSION}"
    temp_dir = frozen["namespace_root"] / f".evaluation_{OUTPUT_VERSION}.tmp"
    if output_dir.exists():
        raise FileExistsError(f"Mapping erratum already exists: {output_dir}")
    if temp_dir.exists():
        raise FileExistsError(f"Stale mapping-erratum directory exists: {temp_dir}")
    if not corrected_mapping_path.is_file():
        raise FileNotFoundError(f"Corrected mapping does not exist: {corrected_mapping_path}")

    truth = heldout.load_visible_musicxml_truth(prior["source_musicxml"])
    crop_indices = tuple(sorted(frozen["paired_rows_by_crop"]))
    prior_mapping = heldout._load_mapping(prior["mapping"])
    corrected_payload = heldout._load_mapping(corrected_mapping_path)
    prior_materialized = heldout.validate_and_materialize_mapping(
        prior_mapping,
        truth=truth,
        crop_indices=crop_indices,
        mode="superseded_invalid_mapping",
    )
    corrected_mapping = heldout.validate_and_materialize_mapping(
        corrected_payload,
        truth=truth,
        crop_indices=crop_indices,
        mode="explicit_mapping_erratum",
    )
    mapping_diff = _mapping_diff(prior_materialized, corrected_mapping)
    _validate_narrow_no_lo_creas_correction(mapping_diff)

    truth_rows = heldout.build_truth_rows(frozen["requests_by_crop"], truth, corrected_mapping)
    structure_support = evaluator._mapping_structure_support(corrected_mapping, truth)
    lane_reports = {
        lane: evaluator._score_lane(
            truth_rows,
            frozen["paired_rows_by_crop"],
            lane=lane,
            structure_support=structure_support,
        )
        for lane in evaluator.LANES
    }
    report = evaluator._build_report(
        target=frozen["target"],
        mapping_mode="explicit_mapping_erratum",
        truth=truth,
        lane_reports=lane_reports,
        structure_support=structure_support,
    )
    report["status"] = "evaluated_mapping_erratum_preserving_original"
    report["erratum"] = {
        "reason": ERRATUM_REASON,
        "superseded_evaluation": freezer._repo_display_path(prior_evaluation_manifest.parent),
        "original_preserved": True,
        "mapping_diff": mapping_diff,
    }

    temp_dir.mkdir(parents=False, exist_ok=False)
    try:
        source_snapshot = temp_dir / "source.musicxml"
        mapping_snapshot = temp_dir / "mapping.json"
        truth_snapshot = temp_dir / "truth.jsonl"
        paired_snapshot = temp_dir / "frozen_paired_predictions.jsonl"
        freeze_snapshot = temp_dir / "frozen_freeze.json"
        sealed_snapshot = temp_dir / "frozen_sealed_manifest.json"
        evaluator_snapshot = temp_dir / "evaluator.py"
        erratum_snapshot = temp_dir / "erratum_evaluator.py"
        superseded_dir = temp_dir / "superseded_evaluation_v1"
        report_path = temp_dir / "report.json"

        shutil.copyfile(prior["source_musicxml"], source_snapshot)
        heldout._write_json(mapping_snapshot, corrected_mapping)
        heldout._write_jsonl(truth_snapshot, truth_rows)
        shutil.copyfile(frozen["predictions_path"], paired_snapshot)
        shutil.copyfile(frozen["freeze_path"], freeze_snapshot)
        shutil.copyfile(sealed_manifest_path, sealed_snapshot)
        shutil.copyfile(Path(evaluator.__file__).resolve(), evaluator_snapshot)
        shutil.copyfile(Path(__file__).resolve(), erratum_snapshot)
        superseded_dir.mkdir()
        for label, source in (
            ("manifest.json", prior_evaluation_manifest),
            ("report.json", prior["report"]),
            ("mapping.json", prior["mapping"]),
        ):
            shutil.copyfile(source, superseded_dir / label)

        pins = {
            "source_musicxml": heldout._snapshot_record(source_snapshot),
            "mapping": heldout._snapshot_record(
                mapping_snapshot,
                source_path=corrected_mapping_path,
                source_sha256=freezer._sha256(corrected_mapping_path),
                require_source_match=False,
            ),
            "truth": heldout._snapshot_record(truth_snapshot),
            "paired_predictions": heldout._snapshot_record(
                paired_snapshot,
                source_path=frozen["predictions_path"],
                source_sha256=frozen["predictions_sha256"],
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
            "original_evaluator": heldout._snapshot_record(evaluator_snapshot),
            "erratum_evaluator": heldout._snapshot_record(
                erratum_snapshot,
                source_path=Path(__file__).resolve(),
                source_sha256=freezer._sha256(Path(__file__).resolve()),
            ),
            "superseded_manifest": heldout._snapshot_record(
                superseded_dir / "manifest.json",
                source_path=prior_evaluation_manifest,
                source_sha256=freezer._sha256(prior_evaluation_manifest),
            ),
            "superseded_report": heldout._snapshot_record(
                superseded_dir / "report.json",
                source_path=prior["report"],
                source_sha256=freezer._sha256(prior["report"]),
            ),
            "superseded_mapping": heldout._snapshot_record(
                superseded_dir / "mapping.json",
                source_path=prior["mapping"],
                source_sha256=freezer._sha256(prior["mapping"]),
            ),
        }
        report["pins"] = pins
        heldout._write_json(report_path, report)
        pins["report"] = heldout._snapshot_record(report_path)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "independent_dyad_recovery_mapping_erratum_manifest",
            "status": "evaluated_mapping_erratum_preserving_original",
            "create_once": True,
            "evaluation_version": OUTPUT_VERSION,
            "target": frozen["target"],
            "reason": ERRATUM_REASON,
            "supersedes": freezer._repo_display_path(prior_evaluation_manifest.parent),
            "original_preserved": True,
            "pins": pins,
        }
        heldout._write_json(temp_dir / "manifest.json", manifest)

        verified_again = evaluator.verify_frozen_dyad_gate(sealed_manifest_path)
        if verified_again["sealed_sha256"] != frozen["sealed_sha256"]:
            raise ValueError("Frozen dyad gate changed during mapping erratum")
        temp_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return {
        "evaluation_dir": str(output_dir),
        "manifest": str(output_dir / "manifest.json"),
        "report": str(output_dir / "report.json"),
        "mapping": str(output_dir / "mapping.json"),
    }


def _verify_prior_evaluation(
    manifest_path: Path,
    *,
    expected_target: Mapping[str, Any],
    expected_sealed_sha256: str,
    expected_predictions_sha256: str,
) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prior evaluation manifest does not exist: {manifest_path}")
    payload = heldout._read_json(manifest_path)
    if payload.get("kind") != "independent_dyad_recovery_post_freeze_evaluation_manifest":
        raise ValueError("Prior evaluation has the wrong manifest kind")
    if payload.get("target") != expected_target:
        raise ValueError("Prior evaluation target differs from the frozen gate")
    pins = payload.get("pins")
    if not isinstance(pins, Mapping):
        raise ValueError("Prior evaluation manifest has no pins")
    paths = {}
    for key in ("source_musicxml", "mapping", "report", "paired_predictions"):
        record = pins.get(key)
        if not isinstance(record, Mapping):
            raise ValueError(f"Prior evaluation is missing pin: {key}")
        path = heldout._safe_child(manifest_path.parent, str(record["snapshot_path"]))
        if freezer._sha256(path) != record.get("snapshot_sha256"):
            raise ValueError(f"Prior evaluation snapshot hash mismatch: {key}")
        paths[key] = path
    sealed_pin = pins.get("sealed_manifest")
    if not isinstance(sealed_pin, Mapping) or sealed_pin.get("source_sha256") != (
        expected_sealed_sha256
    ):
        raise ValueError("Prior evaluation is not tied to the current sealed gate")
    paired_pin = pins["paired_predictions"]
    if paired_pin.get("source_sha256") != expected_predictions_sha256:
        raise ValueError("Prior evaluation predictions differ from the frozen gate")
    return paths


def _mapping_diff(prior: Mapping[str, Any], corrected: Mapping[str, Any]) -> list[dict[str, Any]]:
    prior_by_crop = {int(row["automatic_crop_index"]): row for row in prior["automatic_crops"]}
    corrected_by_crop = {
        int(row["automatic_crop_index"]): row for row in corrected["automatic_crops"]
    }
    changes = []
    for crop in sorted(prior_by_crop):
        before = prior_by_crop[crop]["physical_measure_spans"]
        after = corrected_by_crop[crop]["physical_measure_spans"]
        if before != after:
            changes.append(
                {
                    "automatic_crop_index": crop,
                    "before": copy.deepcopy(before),
                    "after": copy.deepcopy(after),
                }
            )
    if not changes:
        raise ValueError("Corrected mapping is identical to the superseded mapping")
    return changes


def _validate_narrow_no_lo_creas_correction(changes: list[dict[str, Any]]) -> None:
    if [change["automatic_crop_index"] for change in changes] != [1, 2]:
        raise ValueError("Mapping erratum may change only No lo Creas crops 1 and 2")
    expected = [
        {
            "automatic_crop_index": 1,
            "before": [{"measure_number": 1, "note_start": 0, "note_end": 9}],
            "after": [{"measure_number": 1, "note_start": 0, "note_end": 6}],
        },
        {
            "automatic_crop_index": 2,
            "before": [{"measure_number": 1, "note_start": 9, "note_end": 12}],
            "after": [{"measure_number": 1, "note_start": 6, "note_end": 12}],
        },
    ]
    if changes != expected:
        raise ValueError("Mapping change does not match the audited No lo Creas correction")


if __name__ == "__main__":
    raise SystemExit(main())
