"""Prepare the truth-blind Tio Climaco full-event melody gate.

The fixed target is local-restricted Tio Climaco system 7. Preparation seals
eight automatic crops plus pre-transcription meter/key metadata. The generic
score-disjoint selector, both optional recovery lanes, and repaired full-event
composition must all be materialized and sealed before target MusicXML exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_vlm_melody_inputs import _detect_initial_signature  # noqa: E402
from scripts.experiments import freeze_third_score_heldout as base  # noqa: E402
from scripts.experiments import strict_initial_key_context as strict_key  # noqa: E402

OUTPUT_SUBDIR = "vlm_melody_independent_full_event_gate"
DEFAULT_NAMESPACE = "v1"
EVALUATOR_VERSION = "independent-full-event-gate-v1"
TIO_CLIMACO_SLUG = "jaime-llanos_94_tio-climaco_pasillo_bonifacio-bautista"
TARGET_SYSTEM_INDEX = 7
EXPECTED_CROP_COUNT = 8
EXPECTED_TIME_SIGNATURE = "3/4"
EXPECTED_KEY_HINT = "1 flat(s): Bb"

INDEPENDENT_FULL_EVENT_GATE = base.HeldoutGateSpec(
    key="independent_full_event",
    output_subdir=OUTPUT_SUBDIR,
    evaluator_version=EVALUATOR_VERSION,
    implementation_path=Path(__file__),
)

DEFAULT_CANDIDATE_POOL = (
    base.Candidate(TIO_CLIMACO_SLUG, TARGET_SYSTEM_INDEX, "fixed_preregistered_target"),
)
DEFAULT_LAYOUT_POLICY = base.LayoutPolicy(
    min_measure_count=EXPECTED_CROP_COUNT,
    max_measure_count=EXPECTED_CROP_COUNT,
    min_crop_width_px=100,
    max_spacing_cv=0.45,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", type=Path, default=Path("out/local_restricted"))
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    args = parser.parse_args(argv)
    try:
        result = prepare_independent_full_event_gate(args.out_dir, namespace=args.namespace)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(result["prepared_manifest"])
    return 0


def prepare_independent_full_event_gate(
    out_dir: Path,
    *,
    namespace: str = DEFAULT_NAMESPACE,
    candidate_pool: Sequence[base.Candidate] = DEFAULT_CANDIDATE_POOL,
    policy: base.LayoutPolicy | None = None,
) -> dict[str, Any]:
    """Seal the fixed eight-crop target without opening target truth."""
    _validate_target(candidate_pool)
    work_dir = out_dir / TIO_CLIMACO_SLUG
    metadata_path = work_dir / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"Missing pipeline metadata for full-event gate: {metadata_path}")
    metadata = base._read_json(metadata_path)
    _validate_metadata(metadata)

    result = base.prepare_heldout_score(
        out_dir,
        namespace=namespace,
        candidate_pool=candidate_pool,
        policy=policy or DEFAULT_LAYOUT_POLICY,
        gate=INDEPENDENT_FULL_EVENT_GATE,
    )
    prepared_path = Path(result["prepared_manifest"])
    prepared = base._read_json(prepared_path)
    expected_target = {"slug": TIO_CLIMACO_SLUG, "system_index": TARGET_SYSTEM_INDEX}
    if prepared.get("target") != expected_target:
        raise ValueError(f"Unexpected independent full-event target: {prepared.get('target')}")
    crops = prepared["artifacts"]["crops"]
    if len(crops) != EXPECTED_CROP_COUNT:
        raise ValueError(
            f"Independent full-event gate requires {EXPECTED_CROP_COUNT} crops, got {len(crops)}"
        )

    context_records = _write_context(
        prepared_path.parent,
        metadata_path=metadata_path,
        initial_system_path=work_dir / "systems/system_001.png",
    )
    prepared["independent_full_event_gate"] = {
        "expected_crop_count": EXPECTED_CROP_COUNT,
        "required_pretruth_lanes": [
            "generic_score_disjoint_baseline",
            "edge_safe_stem_multihead_recovery_v1",
            "sparse_stem_dyad_repair_v1",
            "repaired_full_event_v1",
        ],
        "full_event_contract": (
            "compose repaired candidates through bounded pitch, rhythm, rest, and meter "
            "inference; fail closed on missing meter or request-only synthetic rests"
        ),
        "context_contract": (
            "treble clef plus hash-pinned pre-transcription 3/4 and one-flat metadata; "
            "the automatic visual-key detector remains an explicit diagnostic"
        ),
        "required_final_seal": "independent_full_event_frozen_awaiting_truth",
        "truth_accessed": False,
        "truth_used": False,
    }
    prepared["forbidden_truth_paths"].extend(
        [
            f"out/local_restricted/{TIO_CLIMACO_SLUG}/**/*truth*",
            f"out/local_restricted/{TIO_CLIMACO_SLUG}/**/*.musicxml",
            f"out/local_restricted/{TIO_CLIMACO_SLUG}/**/*.mxl",
        ]
    )
    evaluator_record = _write_evaluator_scope(prepared_path.parent, prepared=prepared)
    prepared["artifacts"]["context"] = context_records
    prepared["artifacts"]["evaluator"] = evaluator_record
    base._write_json(prepared_path, prepared)
    result.update(
        {
            "prepared_manifest_sha256": base._sha256(prepared_path),
            "context": context_records,
            "evaluator": evaluator_record,
        }
    )
    return result


def _validate_target(candidate_pool: Sequence[base.Candidate]) -> None:
    expected = (
        base.Candidate(TIO_CLIMACO_SLUG, TARGET_SYSTEM_INDEX, "fixed_preregistered_target"),
    )
    if tuple(candidate_pool) != expected:
        raise ValueError("Independent full-event gate target is fixed and cannot be substituted")


def _validate_metadata(metadata: dict[str, Any]) -> None:
    if metadata.get("time_signature") != EXPECTED_TIME_SIGNATURE:
        raise ValueError("Tio Climaco full-event gate requires pre-transcription 3/4 metadata")
    if metadata.get("key_hint") != EXPECTED_KEY_HINT:
        raise ValueError("Tio Climaco full-event gate requires pre-transcription one-flat metadata")


def _write_context(
    namespace_root: Path,
    *,
    metadata_path: Path,
    initial_system_path: Path,
) -> dict[str, dict[str, str]]:
    if not initial_system_path.is_file():
        raise ValueError(f"Missing initial system for visual-key diagnostic: {initial_system_path}")
    context_dir = namespace_root / "context"
    context_dir.mkdir(parents=False, exist_ok=False)

    prediction = _detect_initial_signature(initial_system_path)
    visual_state = strict_key.strict_initial_key_state(prediction, source_system_index=1)
    prediction_path = context_dir / "visual_key_prediction.json"
    state_path = context_dir / "visual_key_state.json"
    base._write_json(prediction_path, prediction)
    base._write_json(state_path, visual_state)

    allowed_path = context_dir / "allowed_context.json"
    allowed = {
        "schema_version": 1,
        "kind": "independent_full_event_allowed_context",
        "truth_accessed": False,
        "truth_used": False,
        "allowed_context": {
            "clef": "treble",
            "time_signature": EXPECTED_TIME_SIGNATURE,
            "key_hint": EXPECTED_KEY_HINT,
            "expected_measure_beats": "3",
            "allow_pickup": False,
        },
        "provenance": {
            "clef": "fixed treble-clef spike contract; not target truth",
            "time_signature": "hash-pinned pre-transcription pipeline metadata",
            "key_hint": "hash-pinned pre-transcription pipeline metadata",
            "expected_measure_beats": "derived from the pinned 3/4 metadata",
            "allow_pickup": "false because this isolated continuation system cannot be a pickup",
            "metadata": {
                "path": base._repo_display_path(metadata_path),
                "sha256": base._sha256(metadata_path),
            },
            "visual_key_diagnostic": {
                "prediction_path": "visual_key_prediction.json",
                "prediction_sha256": base._sha256(prediction_path),
                "state_path": "visual_key_state.json",
                "state_sha256": base._sha256(state_path),
                "accepted_as_context": False,
                "status": visual_state.get("status"),
            },
        },
        "warnings": [
            "The automatic visual-key detector is diagnostic only for this target.",
            "Key context comes from metadata supplied and pinned before target transcription.",
        ],
    }
    base._write_json(allowed_path, allowed)
    return {
        "allowed_context": _relative_record(allowed_path, namespace_root),
        "visual_key_prediction": _relative_record(prediction_path, namespace_root),
        "visual_key_state": _relative_record(state_path, namespace_root),
    }


def _write_evaluator_scope(
    namespace_root: Path,
    *,
    prepared: dict[str, Any],
) -> dict[str, str]:
    evaluator_path = namespace_root / "evaluator_spec.json"
    payload = {
        "schema_version": 1,
        "version": EVALUATOR_VERSION,
        "split": "fresh_heldout",
        "status": "preregistered_before_prediction_and_truth",
        "truth_gate": (
            "target MusicXML and physical-measure mapping may be added only after the "
            "complete repaired full-event seal is fixed"
        ),
        "supported_metrics": [
            "candidate_recovery",
            "note_count",
            "chromatic_pitch",
            "onset",
            "duration",
            "rests",
            "meter_validity",
            "exact_measure",
        ],
        "required_checks": [
            "verify every prepared, model, inference, recovery, composition, and seal hash",
            "require an explicit crop-to-physical-measure mapping before evaluation",
            "score baseline and repaired full events separately",
            "report any request-only meter completion as rejected rather than transcription",
        ],
        "forbidden_before_freeze": prepared["forbidden_truth_paths"],
    }
    base._write_json(evaluator_path, payload)
    return {
        "version": EVALUATOR_VERSION,
        "path": evaluator_path.relative_to(namespace_root).as_posix(),
        "sha256": base._sha256(evaluator_path),
        "implementation_path": base._repo_display_path(Path(__file__)),
        "implementation_sha256": base._sha256(Path(__file__)),
    }


def _relative_record(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": base._sha256(path),
    }


if __name__ == "__main__":
    raise SystemExit(main())
