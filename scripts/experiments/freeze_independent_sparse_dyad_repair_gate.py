"""Prepare the truth-blind Desde Lejos dotted-hollow dyad repair gate.

The fixed target is local-restricted Desde Lejos system 7. Preparation may
inspect only pipeline images and metadata. It seals ten automatic crops and an
unknown key/meter context before either prediction lane is materialized.
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

OUTPUT_SUBDIR = "vlm_melody_independent_sparse_dyad_repair_gate"
DEFAULT_NAMESPACE = "v1"
EVALUATOR_VERSION = "independent-sparse-dyad-repair-gate-v1"
DESDE_LEJOS_SLUG = "jaime-llanos_26_desde-lejos_pasillo_b-b"
TARGET_SYSTEM_INDEX = 7
EXPECTED_CROP_COUNT = 10

INDEPENDENT_SPARSE_DYAD_REPAIR_GATE = base.HeldoutGateSpec(
    key="independent_sparse_dyad_repair",
    output_subdir=OUTPUT_SUBDIR,
    evaluator_version=EVALUATOR_VERSION,
    implementation_path=Path(__file__),
)

DEFAULT_CANDIDATE_POOL = (
    base.Candidate(DESDE_LEJOS_SLUG, TARGET_SYSTEM_INDEX, "fixed_preregistered_target"),
)
DEFAULT_LAYOUT_POLICY = base.LayoutPolicy(
    min_measure_count=EXPECTED_CROP_COUNT,
    max_measure_count=EXPECTED_CROP_COUNT,
    min_crop_width_px=140,
    max_spacing_cv=0.35,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", type=Path, default=Path("out/local_restricted"))
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    args = parser.parse_args(argv)
    try:
        result = prepare_independent_sparse_dyad_repair_gate(
            args.out_dir,
            namespace=args.namespace,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(result["prepared_manifest"])
    return 0


def prepare_independent_sparse_dyad_repair_gate(
    out_dir: Path,
    *,
    namespace: str = DEFAULT_NAMESPACE,
    candidate_pool: Sequence[base.Candidate] = DEFAULT_CANDIDATE_POOL,
    policy: base.LayoutPolicy | None = None,
) -> dict[str, Any]:
    """Seal the fixed ten-crop target without opening target truth."""
    from scripts.experiments import spike_consumed_sparse_stem_dyad_repair as repair

    _validate_target(candidate_pool)
    metadata_path = out_dir / DESDE_LEJOS_SLUG / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"Missing pipeline metadata for sparse-dyad gate: {metadata_path}")

    result = base.prepare_heldout_score(
        out_dir,
        namespace=namespace,
        candidate_pool=candidate_pool,
        policy=policy or DEFAULT_LAYOUT_POLICY,
        gate=INDEPENDENT_SPARSE_DYAD_REPAIR_GATE,
    )
    prepared_path = Path(result["prepared_manifest"])
    prepared = base._read_json(prepared_path)
    expected_target = {"slug": DESDE_LEJOS_SLUG, "system_index": TARGET_SYSTEM_INDEX}
    if prepared.get("target") != expected_target:
        raise ValueError(f"Unexpected independent sparse-dyad target: {prepared.get('target')}")
    if len(prepared["artifacts"]["crops"]) != EXPECTED_CROP_COUNT:
        raise ValueError(
            "Independent sparse-dyad gate requires "
            f"{EXPECTED_CROP_COUNT} crops, got {len(prepared['artifacts']['crops'])}"
        )

    context_record = _write_context(prepared_path.parent, metadata_path=metadata_path)
    prepared["independent_sparse_dyad_repair_gate"] = {
        "config_id": repair.CONFIG_ID,
        "parameters": dict(repair.PARAMETERS),
        "baseline": "fixed edge-safe stem multi-head recovery lane",
        "repair_contract": (
            "replace only a two-candidate/two-onset lane with one in-staff shared-stem dyad "
            "when paired weak augmentation-dot evidence is present and displaced anchors are weak"
        ),
        "expected_crop_count": EXPECTED_CROP_COUNT,
        "supported_evaluation": [
            "candidate_pixel_identity",
            "augmentation_dot_evidence",
            "note_count",
            "diatonic_pitch",
            "onset_group_chord_size",
        ],
        "unsupported_evaluation": [
            "chromatic_key_accuracy",
            "duration",
            "rests",
            "meter",
        ],
        "truth_accessed": False,
        "truth_used": False,
    }
    prepared["forbidden_truth_paths"].extend(
        [
            f"out/local_restricted/{DESDE_LEJOS_SLUG}/**/*truth*",
            f"out/local_restricted/{DESDE_LEJOS_SLUG}/**/*.musicxml",
            f"out/local_restricted/{DESDE_LEJOS_SLUG}/**/*.mxl",
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
        base.Candidate(DESDE_LEJOS_SLUG, TARGET_SYSTEM_INDEX, "fixed_preregistered_target"),
    )
    if tuple(candidate_pool) != expected:
        raise ValueError("Independent sparse-dyad gate target is fixed and cannot be substituted")


def _write_context(namespace_root: Path, *, metadata_path: Path) -> dict[str, str]:
    context_dir = namespace_root / "context"
    context_dir.mkdir(parents=False, exist_ok=False)
    context_path = context_dir / "allowed_context.json"
    payload = {
        "schema_version": 1,
        "kind": "independent_sparse_dyad_repair_allowed_context",
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
            "time_signature": "withheld for this pixel/pitch/onset-group gate",
            "key_hint": "explicitly unknown; natural diatonic pitch mapping only",
            "metadata": {
                "path": base._repo_display_path(metadata_path),
                "sha256": base._sha256(metadata_path),
            },
        },
        "warnings": [
            "Key state is unknown; pitches are natural treble-clef diatonic positions.",
            "Rhythm, duration, rests, and meter are not inferred by this gate.",
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
        "truth_gate": "transcription and pixel review may be opened only after both lanes freeze",
        "supported_metrics": [
            "candidate_pixel_identity",
            "augmentation_dot_evidence",
            "note_count",
            "diatonic_pitch",
            "onset_group_chord_size",
        ],
        "unsupported_metrics": [
            "chromatic_key_accuracy",
            "duration",
            "rests",
            "meter",
        ],
        "required_checks": [
            "verify every prepared, inference, paired-lane, model, source, parameter, and "
            "implementation hash before opening truth",
            "verify the multi-head comparison lane is reproduced from the fixed prior rule",
            "verify each accepted replacement has one shared-stem pair plus two aligned weak "
            "augmentation-dot candidates",
            "score head pixel identity and onset-group chord size, not MusicXML count alone",
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
