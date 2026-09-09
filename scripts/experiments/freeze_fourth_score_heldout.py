"""Prepare and freeze the truth-blind fourth-score melody gate.

The target is selected from system images only. Preparation also pins the
pipeline metadata, an automatic initial key-signature prediction, and a
conservative rhythm-derived meter prior before model inference or truth access.
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

OUTPUT_SUBDIR = "vlm_melody_fourth_score_heldout"
DEFAULT_NAMESPACE = "v1"
EVALUATOR_VERSION = "fourth-score-melody-gate-v1"
GATOE_FIQUE_SLUG = "jaime-llanos_49_gatoe-fique_pasillo_emilio-murillo"

FOURTH_SCORE_GATE = base.HeldoutGateSpec(
    key="fourth_score",
    output_subdir=OUTPUT_SUBDIR,
    evaluator_version=EVALUATOR_VERSION,
    implementation_path=Path(__file__),
)

DEFAULT_CANDIDATE_POOL = (
    base.Candidate(GATOE_FIQUE_SLUG, 3, "primary_layout_only"),
    base.Candidate(
        "jaime-llanos_22_coqueteos_pasillo_fulgencio-garcia",
        4,
        "fallback_layout_only",
    ),
    base.Candidate(
        "jaime-llanos_92_sobre-el-humo_bambuco_fulgencio-garcia",
        3,
        "harder_fallback_layout_only",
    ),
)

RHYTHM_CONTEXT_PRIORS = {
    "pasillo": {"time_signature": "3/4", "expected_measure_beats": 3.0},
}


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
            report = prepare_fourth_score(args.out_dir, namespace=args.namespace)
            print(report["prepared_manifest"])
        else:
            report = freeze_prepared_fourth_score(
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


def prepare_fourth_score(
    out_dir: Path,
    *,
    namespace: str = DEFAULT_NAMESPACE,
    candidate_pool: Sequence[base.Candidate] = DEFAULT_CANDIDATE_POOL,
    policy: base.LayoutPolicy | None = None,
) -> dict[str, Any]:
    """Select a fresh system and pin its truth-blind musical context."""
    report = base.prepare_heldout_score(
        out_dir,
        namespace=namespace,
        candidate_pool=candidate_pool,
        policy=policy,
        gate=FOURTH_SCORE_GATE,
    )
    prepared_path = Path(report["prepared_manifest"])
    prepared = base._read_json(prepared_path)
    target = prepared["target"]
    slug = str(target["slug"])
    context_records = _prepare_context(out_dir, slug=slug, namespace_root=prepared_path.parent)
    prepared["artifacts"]["context"] = context_records
    base._write_json(prepared_path, prepared)
    report["prepared_manifest_sha256"] = base._sha256(prepared_path)
    report["context"] = context_records
    return report


def freeze_prepared_fourth_score(
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
        gate=FOURTH_SCORE_GATE,
    )


def _prepare_context(out_dir: Path, *, slug: str, namespace_root: Path) -> dict[str, Any]:
    metadata_path = out_dir / slug / "metadata.json"
    system_path = out_dir / slug / "systems/system_001.png"
    if not metadata_path.is_file():
        raise ValueError(f"Missing pipeline metadata for fourth-score context: {metadata_path}")
    if not system_path.is_file():
        raise ValueError(f"Missing first system for initial key detection: {system_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected metadata object: {metadata_path}")

    context_dir = namespace_root / "context"
    context_dir.mkdir(parents=False, exist_ok=False)
    key_prediction = key_detector.detect_signature(system_path, mode=key_detector.MODE_INITIAL)
    key_report_path = context_dir / "initial_key_signature.json"
    base._write_json(key_report_path, key_prediction)
    key_overlay_path = context_dir / "initial_key_signature_overlay.png"
    key_detector._draw_overlay(key_prediction, key_overlay_path)

    time_signature = metadata.get("time_signature")
    expected_beats = _expected_beats(time_signature)
    meter_source = "pipeline_metadata.time_signature"
    if expected_beats is None:
        prior = RHYTHM_CONTEXT_PRIORS.get(str(metadata.get("rhythm", "")).strip().lower())
        if prior:
            time_signature = prior["time_signature"]
            expected_beats = float(prior["expected_measure_beats"])
            meter_source = f"provisional_metadata_rhythm_prior:{metadata.get('rhythm')}"

    fifths = key_prediction.get("fifths")
    key_hint = _key_hint(int(fifths)) if fifths is not None else metadata.get("key_hint")
    key_source = (
        "automatic_visual_initial_key_signature_detector_v2"
        if fifths is not None
        else "pipeline_metadata.key_hint_or_unknown"
    )
    warnings = []
    if expected_beats is None:
        warnings.append("No usable meter context was available before truth.")
    elif meter_source.startswith("provisional_"):
        warnings.append("Meter is a provisional rhythm-genre prior, not visual or target truth.")
    if fifths is None:
        warnings.append("Automatic visual key detection was inconclusive.")

    context_path = context_dir / "allowed_context.json"
    context = {
        "schema_version": 1,
        "kind": "fourth_score_truth_blind_allowed_context",
        "truth_accessed": False,
        "truth_used": False,
        "target_slug": slug,
        "allowed_context": {
            "clef": "treble",
            "time_signature": time_signature,
            "key_hint": key_hint,
            "expected_measure_beats": expected_beats,
            "allow_pickup": False,
        },
        "provenance": {
            "clef": "melody-spike fixed treble-clef contract; not target truth",
            "time_signature": meter_source,
            "key_hint": key_source,
            "expected_measure_beats": meter_source,
            "allow_pickup": "false for an interior held-out system",
            "metadata": {
                "path": base._repo_display_path(metadata_path),
                "sha256": base._sha256(metadata_path),
            },
            "initial_key_image": {
                "path": base._repo_display_path(system_path),
                "sha256": base._sha256(system_path),
            },
        },
        "warnings": warnings,
    }
    base._write_json(context_path, context)
    return {
        "allowed_context": _relative_record(context_path, namespace_root),
        "initial_key_signature": _relative_record(key_report_path, namespace_root),
        "initial_key_signature_overlay": _relative_record(key_overlay_path, namespace_root),
    }


def _expected_beats(time_signature: Any) -> float | None:
    if not time_signature:
        return None
    try:
        numerator, denominator = str(time_signature).split("/", 1)
        value = int(numerator) * 4 / int(denominator)
    except (ValueError, ZeroDivisionError):
        return None
    return float(value) if value > 0 else None


def _key_hint(fifths: int) -> str:
    sharps = ("F#", "C#", "G#", "D#", "A#", "E#", "B#")
    flats = ("Bb", "Eb", "Ab", "Db", "Gb", "Cb", "Fb")
    if fifths > 0:
        names = sharps[:fifths]
        return f"{fifths} sharp(s): {', '.join(names)}"
    if fifths < 0:
        names = flats[: abs(fifths)]
        return f"{abs(fifths)} flat(s): {', '.join(names)}"
    return "no sharps or flats"


def _relative_record(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": base._sha256(path),
    }


if __name__ == "__main__":
    raise SystemExit(main())
