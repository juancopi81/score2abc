"""Run and seal one prepared independent automatic-key-state gate.

The score-disjoint selector runs once. Its candidate IDs and coordinates are
then replayed through two global staff-pitch lanes: no key signature and the
truth-blind visual key state pinned during preparation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import freeze_third_score_heldout as base  # noqa: E402
from scripts.experiments import run_third_score_heldout_inference as inference  # noqa: E402
from scripts.experiments.spike_consumed_polyphonic_pitch_repair import (  # noqa: E402
    _fifths_accidentals,
    _staff_pitch,
)

SCHEMA_VERSION = 1
PAIR_VERSION = "independent-key-paired-predictions-v1"
DEFAULT_INFERENCE_DIRNAME = "inference_v1"
DEFAULT_MODEL_DIR = (
    REPO_ROOT / "out/vlm_melody_consumed_training/cross_score_notehead_v1_replay_20260722"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared_manifest", type=Path)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args(argv)
    try:
        result = run_and_seal(args.prepared_manifest, model_dir=args.model_dir)
    except (FileExistsError, FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def run_and_seal(prepared_manifest_path: Path, *, model_dir: Path) -> dict[str, Any]:
    prepared_manifest_path = prepared_manifest_path.resolve()
    model_dir = model_dir.resolve()
    prepared = base._read_json(prepared_manifest_path)
    _validate_prepared_gate(prepared)

    result = inference.materialize_third_score_inference(
        prepared_manifest_path,
        model_dir=model_dir,
        inference_dirname=DEFAULT_INFERENCE_DIRNAME,
    )
    inference_dir = Path(result["inference_dir"])
    detailed_path = inference_dir / "inference.jsonl"
    detailed = inference._read_jsonl(detailed_path)
    paired_rows, invariance = build_paired_predictions(
        detailed,
        automatic_fifths=int(prepared["independent_key_gate"]["automatic_fifths"]),
    )
    paired_path = inference_dir / "paired_predictions.jsonl"
    invariance_path = inference_dir / "selection_invariance.json"
    manifest_path = inference_dir / "paired_manifest.json"
    inference._write_jsonl(paired_path, paired_rows)
    inference._write_json(invariance_path, invariance)
    pair_manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "independent_key_state_paired_prediction_manifest",
        "version": PAIR_VERSION,
        "status": "predicted_before_truth",
        "truth_accessed": False,
        "truth_used": False,
        "target": prepared["target"],
        "localization_contract": "one selector pass shared by both pitch lanes",
        "baseline_fifths": None,
        "automatic_fifths": int(prepared["independent_key_gate"]["automatic_fifths"]),
        "measure_count": len(paired_rows),
        "artifacts": {
            "detailed_inference": _local_record(detailed_path, inference_dir),
            "paired_predictions": _local_record(paired_path, inference_dir),
            "selection_invariance": _local_record(invariance_path, inference_dir),
        },
    }
    inference._write_json(manifest_path, pair_manifest)

    frozen = _freeze_paired_gate(
        prepared_manifest_path,
        model_dir=model_dir,
        inference_dir=inference_dir,
        paired_path=paired_path,
        pair_manifest_path=manifest_path,
        invariance_path=invariance_path,
    )
    return {
        "target": prepared["target"],
        "inference_dir": str(inference_dir),
        "paired_predictions": str(paired_path),
        "paired_predictions_sha256": inference._sha256(paired_path),
        "selection_invariance": invariance,
        "freeze": frozen,
    }


def build_paired_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    automatic_fifths: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        raise ValueError("Independent key gate requires at least one inference row")
    automatic_alterations = _fifths_accidentals(automatic_fifths)
    paired_rows = []
    total_notes = 0
    for row in rows:
        if row.get("truth_used") is not False:
            raise ValueError("Inference row did not preserve truth_used=false")
        geometry = row.get("staff_geometry")
        if not isinstance(geometry, Mapping):
            raise ValueError("Inference row has no staff geometry")
        staff_lines = geometry.get("raw_staff_lines_y_px")
        if not isinstance(staff_lines, Sequence) or isinstance(staff_lines, (str, bytes)):
            raise ValueError("Inference row has no raw staff lines")
        anchors = row.get("automatic_anchors")
        if not isinstance(anchors, list):
            raise ValueError("Inference row has no automatic anchors")

        baseline_notes = []
        automatic_notes = []
        for anchor in anchors:
            source = anchor.get("source")
            center = anchor.get("center")
            if not isinstance(source, Mapping) or not isinstance(center, Mapping):
                raise ValueError("Malformed automatic anchor")
            candidate_id = str(source["candidate_id"])
            x = float(center["x"])
            y = float(center["y"])
            baseline = _staff_pitch(y, staff_lines, {})
            automatic = _staff_pitch(y, staff_lines, automatic_alterations)
            shared = {
                "candidate_id": candidate_id,
                "center": {"x": x, "y": y},
                "order": int(anchor["order"]),
            }
            baseline_notes.append({**shared, **_pitch_fields(baseline)})
            automatic_notes.append({**shared, **_pitch_fields(automatic)})
        total_notes += len(baseline_notes)
        identity = dict(row["identity"])
        paired_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "identity": identity,
                "truth_accessed": False,
                "truth_used": False,
                "source": dict(row["source"]),
                "staff_geometry": dict(geometry),
                "baseline_fifths": None,
                "automatic_fifths": automatic_fifths,
                "lanes": {
                    "global_no_key": {"notes": baseline_notes},
                    "global_automatic_key": {"notes": automatic_notes},
                },
            }
        )

    invariance = selection_invariance(paired_rows)
    invariance.update(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "independent_key_state_selection_invariance",
            "truth_used": False,
            "measure_count": len(paired_rows),
            "note_count": total_notes,
        }
    )
    return paired_rows, invariance


def selection_invariance(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    comparisons = []
    for row in rows:
        lanes = row["lanes"]
        baseline = lanes["global_no_key"]["notes"]
        automatic = lanes["global_automatic_key"]["notes"]
        baseline_ids = [note["candidate_id"] for note in baseline]
        automatic_ids = [note["candidate_id"] for note in automatic]
        baseline_centers = [note["center"] for note in baseline]
        automatic_centers = [note["center"] for note in automatic]
        comparison = {
            "identity": dict(row["identity"]),
            "candidate_ids_equal": baseline_ids == automatic_ids,
            "coordinates_equal": baseline_centers == automatic_centers,
            "counts_equal": len(baseline) == len(automatic),
        }
        comparisons.append(comparison)
    passed = all(
        row["candidate_ids_equal"] and row["coordinates_equal"] and row["counts_equal"]
        for row in comparisons
    )
    if not passed:
        raise ValueError("Automatic-key lane changed candidate localization")
    return {"passed": True, "comparisons": comparisons}


def _freeze_paired_gate(
    prepared_manifest_path: Path,
    *,
    model_dir: Path,
    inference_dir: Path,
    paired_path: Path,
    pair_manifest_path: Path,
    invariance_path: Path,
) -> dict[str, Any]:
    manifest = inference._read_json(inference_dir / "manifest.json")
    validated = inference._verify_inference_binding(
        prepared_manifest_path,
        model_dir=model_dir,
        inference_dir=inference_dir,
        manifest=manifest,
    )
    model_paths = inference._deduplicate_paths(
        [model_dir / "model.json", validated["model_implementation_path"]]
    )
    selected_model_training_outputs = [
        path
        for path in validated["model_artifact_paths"]
        if path != (model_dir / "model.json").resolve()
    ]
    training_paths = inference._deduplicate_paths(
        [
            *selected_model_training_outputs,
            model_dir / "manifest.json",
            *validated["training_input_paths"],
            inference_dir / "manifest.json",
            Path(inference.__file__),
            Path(__file__),
            inference._resolve_record_path(validated["metadata_record"]),
            inference_dir / "assumptions.json",
            inference_dir / "requests.jsonl",
            inference_dir / "replay.json",
            inference_dir / "inference.jsonl",
            pair_manifest_path,
            invariance_path,
            *validated.get("prepared_context_paths", ()),
        ]
    )
    return base.freeze_prepared_heldout_score(
        prepared_manifest_path,
        predictions_path=paired_path,
        model_artifact_paths=model_paths,
        training_artifact_paths=training_paths,
        gate=validated["gate_config"]["gate"],
    )


def _validate_prepared_gate(prepared: Mapping[str, Any]) -> None:
    config = prepared.get("independent_key_gate")
    if not isinstance(config, Mapping):
        raise ValueError("Prepared manifest is not an independent key-state gate")
    if config.get("baseline_fifths") is not None:
        raise ValueError("Independent key baseline must have no key signature")
    if not isinstance(config.get("automatic_fifths"), int):
        raise ValueError("Independent key gate has no automatic fifths")
    if config.get("truth_used") is not False or prepared.get("truth_accessed") is not False:
        raise ValueError("Prepared independent key gate is not truth-blind")


def _pitch_fields(pitch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pitch": str(pitch["pitch"]),
        "pitch_midi": int(pitch["pitch_midi"]),
        "staff_position": int(pitch["staff_position"]),
    }


def _local_record(path: Path, root: Path) -> dict[str, str]:
    return {"path": path.relative_to(root).as_posix(), "sha256": inference._sha256(path)}


if __name__ == "__main__":
    raise SystemExit(main())
