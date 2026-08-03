"""Prepare and freeze the truth-blind Coqueteos fifth-score melody gate.

The gate is fixed to Coqueteos system 2 and expects six automatic crops. It
pins a provisional Pasillo meter prior and an explicitly unknown key because
the automatic initial-key detector is inconclusive on this score.
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
from scripts.experiments import spike_consumed_key_signature_detector as key_detector  # noqa: E402

OUTPUT_SUBDIR = "vlm_melody_fifth_score_heldout"
DEFAULT_NAMESPACE = "v1"
EVALUATOR_VERSION = "fifth-score-melody-gate-v1"
COQUETEOS_SLUG = "jaime-llanos_22_coqueteos_pasillo_fulgencio-garcia"

FIFTH_SCORE_GATE = base.HeldoutGateSpec(
    key="fifth_score",
    output_subdir=OUTPUT_SUBDIR,
    evaluator_version=EVALUATOR_VERSION,
    implementation_path=Path(__file__),
)

DEFAULT_CANDIDATE_POOL = (base.Candidate(COQUETEOS_SLUG, 2, "primary_layout_only"),)
DEFAULT_LAYOUT_POLICY = base.LayoutPolicy(
    min_measure_count=6,
    max_measure_count=6,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("out_dir", nargs="?", type=Path, default=Path("out"))
    prepare_parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("prepared_manifest", type=Path)
    freeze_parser.add_argument("--predictions", type=Path, required=True)
    freeze_parser.add_argument("--model-artifact", type=Path, action="append", required=True)
    freeze_parser.add_argument("--training-artifact", type=Path, action="append", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            report = prepare_fifth_score(args.out_dir, namespace=args.namespace)
            print(report["prepared_manifest"])
        else:
            report = freeze_prepared_fifth_score(
                args.prepared_manifest,
                predictions_path=args.predictions,
                model_artifact_paths=args.model_artifact,
                training_artifact_paths=args.training_artifact,
            )
            print(report["sealed_manifest"])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def prepare_fifth_score(
    out_dir: Path,
    *,
    namespace: str = DEFAULT_NAMESPACE,
    candidate_pool: Sequence[base.Candidate] = DEFAULT_CANDIDATE_POOL,
    policy: base.LayoutPolicy | None = None,
) -> dict[str, Any]:
    """Prepare the six-crop Coqueteos gate and pin truth-blind context."""
    context_inputs = _inspect_context_inputs(out_dir, slug=COQUETEOS_SLUG)
    report = base.prepare_heldout_score(
        out_dir,
        namespace=namespace,
        candidate_pool=candidate_pool,
        policy=policy or DEFAULT_LAYOUT_POLICY,
        gate=FIFTH_SCORE_GATE,
    )
    prepared_path = Path(report["prepared_manifest"])
    prepared = base._read_json(prepared_path)
    target = prepared["target"]
    if target != {"slug": COQUETEOS_SLUG, "system_index": 2}:
        raise ValueError(f"Unexpected fifth-score target: {target}")
    context_records = _write_context(
        namespace_root=prepared_path.parent,
        **context_inputs,
    )
    prepared["artifacts"]["context"] = context_records
    base._write_json(prepared_path, prepared)
    report["prepared_manifest_sha256"] = base._sha256(prepared_path)
    report["context"] = context_records
    return report


def freeze_prepared_fifth_score(
    prepared_manifest_path: Path,
    *,
    predictions_path: Path,
    model_artifact_paths: Sequence[Path],
    training_artifact_paths: Sequence[Path],
) -> dict[str, Any]:
    return base.freeze_prepared_heldout_score(
        prepared_manifest_path,
        predictions_path=predictions_path,
        model_artifact_paths=model_artifact_paths,
        training_artifact_paths=training_artifact_paths,
        gate=FIFTH_SCORE_GATE,
    )


def _inspect_context_inputs(out_dir: Path, *, slug: str) -> dict[str, Any]:
    metadata_path = out_dir / slug / "metadata.json"
    initial_system_path = out_dir / slug / "systems/system_001.png"
    if not metadata_path.is_file():
        raise ValueError(f"Missing pipeline metadata for fifth-score context: {metadata_path}")
    if not initial_system_path.is_file():
        raise ValueError(f"Missing first system for initial key detection: {initial_system_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected metadata object: {metadata_path}")
    if str(metadata.get("rhythm", "")).strip().lower() != "pasillo":
        raise ValueError("Fifth-score provisional meter prior requires Pasillo metadata")

    key_prediction = key_detector.detect_signature(
        initial_system_path,
        mode=key_detector.MODE_INITIAL,
    )
    if key_prediction.get("gate_passed") is not False:
        raise ValueError(
            "Fifth-score gate was preregistered with an inconclusive visual key; "
            "refusing changed key-detector context"
        )
    return {
        "metadata_path": metadata_path,
        "initial_system_path": initial_system_path,
        "key_prediction": key_prediction,
    }


def _write_context(
    *,
    namespace_root: Path,
    metadata_path: Path,
    initial_system_path: Path,
    key_prediction: dict[str, Any],
) -> dict[str, Any]:
    context_dir = namespace_root / "context"
    context_dir.mkdir(parents=False, exist_ok=False)
    key_report_path = context_dir / "initial_key_signature.json"
    base._write_json(key_report_path, key_prediction)
    key_overlay_path = context_dir / "initial_key_signature_overlay.png"
    key_detector._draw_overlay(key_prediction, key_overlay_path)

    context_path = context_dir / "allowed_context.json"
    context = {
        "schema_version": 1,
        "kind": "fifth_score_truth_blind_allowed_context",
        "truth_accessed": False,
        "truth_used": False,
        "target_slug": COQUETEOS_SLUG,
        "allowed_context": {
            "clef": "treble",
            "time_signature": "3/4",
            "key_hint": None,
            "expected_measure_beats": 3.0,
            "allow_pickup": False,
        },
        "provenance": {
            "clef": "melody-spike fixed treble-clef contract; not target truth",
            "time_signature": "provisional_metadata_rhythm_prior:Pasillo",
            "key_hint": "automatic visual initial-key detector inconclusive; fail closed",
            "expected_measure_beats": "provisional_metadata_rhythm_prior:Pasillo",
            "allow_pickup": "false for an interior held-out system",
            "metadata": {
                "path": base._repo_display_path(metadata_path),
                "sha256": base._sha256(metadata_path),
            },
            "initial_key_image": {
                "path": base._repo_display_path(initial_system_path),
                "sha256": base._sha256(initial_system_path),
            },
        },
        "warnings": [
            "Meter is a provisional rhythm-genre prior, not visual or target truth.",
            "Automatic visual key detection was inconclusive; key remains unknown.",
        ],
    }
    base._write_json(context_path, context)
    return {
        "allowed_context": _relative_record(context_path, namespace_root),
        "initial_key_signature": _relative_record(key_report_path, namespace_root),
        "initial_key_signature_overlay": _relative_record(key_overlay_path, namespace_root),
    }


def _relative_record(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": base._sha256(path),
    }


if __name__ == "__main__":
    raise SystemExit(main())
