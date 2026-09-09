"""Prepare the truth-blind No Lo Creas dyad-recovery gate.

The fixed target is local-restricted No Lo Creas system 8. Preparation may
inspect only pipeline images and metadata. It seals eleven automatic crops and
an explicitly limited musical context before baseline or recovery inference.
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

from scripts.experiments import freeze_third_score_heldout as base  # noqa: E402
from scripts.experiments import spike_consumed_polyphonic_pitch_repair as recovery  # noqa: E402

OUTPUT_SUBDIR = "vlm_melody_independent_dyad_recovery_gate"
DEFAULT_NAMESPACE = "v1"
EVALUATOR_VERSION = "independent-dyad-recovery-gate-v1"
NO_LO_CREAS_SLUG = "jaime-llanos_73_no-lo-creas_pasillo_a-vasquez-pedrero"
TARGET_SYSTEM_INDEX = 8
EXPECTED_CROP_COUNT = 11

INDEPENDENT_DYAD_RECOVERY_GATE = base.HeldoutGateSpec(
    key="independent_dyad_recovery",
    output_subdir=OUTPUT_SUBDIR,
    evaluator_version=EVALUATOR_VERSION,
    implementation_path=Path(__file__),
)

DEFAULT_CANDIDATE_POOL = (
    base.Candidate(NO_LO_CREAS_SLUG, TARGET_SYSTEM_INDEX, "fixed_preregistered_target"),
)
DEFAULT_LAYOUT_POLICY = base.LayoutPolicy(
    min_measure_count=EXPECTED_CROP_COUNT,
    max_measure_count=EXPECTED_CROP_COUNT,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", type=Path, default=Path("out/local_restricted"))
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    args = parser.parse_args(argv)
    try:
        result = prepare_independent_dyad_recovery_gate(
            args.out_dir,
            namespace=args.namespace,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(result["prepared_manifest"])
    return 0


def prepare_independent_dyad_recovery_gate(
    out_dir: Path,
    *,
    namespace: str = DEFAULT_NAMESPACE,
    candidate_pool: Sequence[base.Candidate] = DEFAULT_CANDIDATE_POOL,
    policy: base.LayoutPolicy | None = None,
) -> dict[str, Any]:
    """Seal the fixed eleven-crop target without opening target truth."""
    _validate_target(candidate_pool)
    metadata_path = out_dir / NO_LO_CREAS_SLUG / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"Missing pipeline metadata for dyad gate: {metadata_path}")

    result = base.prepare_heldout_score(
        out_dir,
        namespace=namespace,
        candidate_pool=candidate_pool,
        policy=policy or DEFAULT_LAYOUT_POLICY,
        gate=INDEPENDENT_DYAD_RECOVERY_GATE,
    )
    prepared_path = Path(result["prepared_manifest"])
    prepared = base._read_json(prepared_path)
    expected_target = {"slug": NO_LO_CREAS_SLUG, "system_index": TARGET_SYSTEM_INDEX}
    if prepared.get("target") != expected_target:
        raise ValueError(f"Unexpected independent dyad target: {prepared.get('target')}")
    crops = prepared["artifacts"]["crops"]
    if len(crops) != EXPECTED_CROP_COUNT:
        raise ValueError(
            f"Independent dyad gate requires {EXPECTED_CROP_COUNT} crops, got {len(crops)}"
        )

    context_record = _write_context(prepared_path.parent, metadata_path=metadata_path)
    prepared["independent_dyad_recovery_gate"] = {
        "config_id": recovery.EDGE_SAFE_STEM_DYAD_CONFIG_ID,
        "parameters": dict(recovery.EDGE_SAFE_STEM_DYAD_PARAMETERS),
        "baseline": "generic_score_disjoint_configuration_c",
        "recovery_contract": (
            "add at most one companion to each existing x group; never delete or reposition "
            "baseline candidates"
        ),
        "paired_note_contract": (
            "each lane exposes deterministic notes mapped directly from frozen candidate "
            "coordinates to natural treble staff_position/pitch, with left-to-right baseline "
            "onset_group_index and recovered status; additions reuse an existing baseline "
            "group ID while generic canonical prediction remains unchanged as provenance"
        ),
        "expected_crop_count": EXPECTED_CROP_COUNT,
        "supported_evaluation": ["candidate_localization", "note_count", "diatonic_pitch"],
        "unsupported_evaluation": [
            "chromatic_key_accuracy",
            "onset",
            "duration",
            "rests",
            "meter",
        ],
        "truth_accessed": False,
        "truth_used": False,
    }
    prepared["forbidden_truth_paths"].extend(
        [
            f"out/local_restricted/{NO_LO_CREAS_SLUG}/**/*truth*",
            f"out/local_restricted/{NO_LO_CREAS_SLUG}/**/*.musicxml",
            f"out/local_restricted/{NO_LO_CREAS_SLUG}/**/*.mxl",
        ]
    )
    evaluator_record = _write_evaluator_scope(prepared_path.parent, prepared=prepared)
    prepared["artifacts"]["context"] = {"allowed_context": context_record}
    prepared["artifacts"]["evaluator"] = evaluator_record
    base._write_json(prepared_path, prepared)
    result.update(
        {
            "prepared_manifest_sha256": base._sha256(prepared_path),
            "context": context_record,
            "evaluator": evaluator_record,
        }
    )
    return result


def _validate_target(candidate_pool: Sequence[base.Candidate]) -> None:
    expected = (
        base.Candidate(NO_LO_CREAS_SLUG, TARGET_SYSTEM_INDEX, "fixed_preregistered_target"),
    )
    if tuple(candidate_pool) != expected:
        raise ValueError("Independent dyad gate target is fixed and cannot be substituted")


def _write_context(namespace_root: Path, *, metadata_path: Path) -> dict[str, str]:
    context_dir = namespace_root / "context"
    context_dir.mkdir(parents=False, exist_ok=False)
    context_path = context_dir / "allowed_context.json"
    payload = {
        "schema_version": 1,
        "kind": "independent_dyad_recovery_allowed_context",
        "truth_accessed": False,
        "truth_used": False,
        "allowed_context": {
            "clef": "treble",
            "time_signature": None,
            "key_hint": None,
            "expected_measure_beats": None,
            "allow_pickup": False,
        },
        "provenance": {
            "clef": "fixed treble-clef spike contract; not target truth",
            "time_signature": "withheld for this localization/count/diatonic-pitch gate",
            "key_hint": "explicitly unknown; natural diatonic pitch mapping only",
            "expected_measure_beats": "unsupported in this gate",
            "allow_pickup": "false because this isolated system cannot establish pickup status",
            "metadata": {
                "path": base._repo_display_path(metadata_path),
                "sha256": base._sha256(metadata_path),
            },
        },
        "evaluation_scope": {
            "supported": ["candidate_localization", "note_count", "diatonic_pitch"],
            "unsupported": [
                "chromatic_key_accuracy",
                "onset",
                "duration",
                "rests",
                "meter",
            ],
        },
        "warnings": [
            "Key state is unknown; pitches are natural treble-clef diatonic positions.",
            "Meter decoding, rhythm, duration, onset, and rest inference are not applied.",
        ],
    }
    base._write_json(context_path, payload)
    return {
        "path": context_path.relative_to(namespace_root).as_posix(),
        "sha256": base._sha256(context_path),
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
        "truth_gate": "transcription may be opened only after both lanes are frozen",
        "supported_metrics": ["candidate_localization", "note_count", "diatonic_pitch"],
        "unsupported_metrics": [
            "chromatic_key_accuracy",
            "onset",
            "duration",
            "rests",
            "meter",
        ],
        "required_checks": [
            "verify all prepared, baseline-inference, paired-lane, model, training, source, "
            "parameter, and implementation hashes before loading truth",
            "verify the baseline lane is identical to generic inference localization",
            "verify recovery only adds one companion per existing x group",
            "verify each paired note has staff_position, onset_group_index, and recovered, and "
            "that recovery introduces no new group IDs",
            "report baseline and recovery localization/count/diatonic-pitch metrics separately",
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


if __name__ == "__main__":
    raise SystemExit(main())
