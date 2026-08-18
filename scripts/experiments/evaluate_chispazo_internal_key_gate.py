"""Evaluate the sealed Chispazo internal-key diagnostic exactly once.

The frozen selector and paired pitch lanes are hash-verified before the human
MusicXML is opened. The ``-2`` lane remains diagnostic because the strict
truth-blind detector returned unknown before transcription.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import evaluate_frozen_third_score_heldout as heldout  # noqa: E402
from scripts.experiments import evaluate_independent_key_state_gates as paired  # noqa: E402
from scripts.experiments import freeze_independent_key_state_gates as freezer  # noqa: E402

SCHEMA_VERSION = 1
DEFAULT_EVALUATION_VERSION = "v1"
CASE = paired.CHISPAZO_INTERNAL_CHANGE_CASE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", type=Path, default=Path("out"))
    parser.add_argument("--evaluation-version", default=DEFAULT_EVALUATION_VERSION)
    args = parser.parse_args(argv)
    try:
        result = evaluate_chispazo_internal_key_gate(
            args.out_dir,
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


def evaluate_chispazo_internal_key_gate(
    out_dir: Path,
    *,
    evaluation_version: str = DEFAULT_EVALUATION_VERSION,
) -> dict[str, str]:
    """Verify the freeze, evaluate both lanes, and publish an atomic report."""
    heldout._validate_version(evaluation_version)
    out_dir = out_dir.expanduser().resolve()

    # Frozen hashes and paired-lane invariants are checked before MusicXML truth.
    frozen = paired._verify_case(out_dir, CASE)
    namespace_root = Path(frozen["namespace_root"])
    output_dir = namespace_root / f"evaluation_{evaluation_version}"
    temp_dir = namespace_root / f".evaluation_{evaluation_version}.tmp"
    if output_dir.exists():
        raise FileExistsError(f"Chispazo evaluation already exists: {output_dir}")
    if temp_dir.exists():
        raise FileExistsError(f"Stale Chispazo evaluation exists: {temp_dir}")

    result = paired._evaluate_case(CASE, frozen)
    prepared = heldout._read_json(Path(frozen["prepared_path"]))
    key_context = prepared["independent_key_gate"]
    if key_context.get("strict_detector_fifths") is not None:
        raise ValueError("Chispazo diagnostic unexpectedly has a strict detector result")
    if key_context.get("automatic_lane_kind") != "inconclusive_top_candidate_diagnostic":
        raise ValueError("Chispazo automatic lane is not marked diagnostic-only")

    comparison = result["comparison"]
    decision = _diagnostic_decision(comparison)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "chispazo_internal_key_change_diagnostic_evaluation",
        "status": "evaluated_exactly_once_after_frozen_hashes_verified",
        "truth_opened_after_all_frozen_hashes_verified": True,
        "metric_scope": "pitch_only_with_fixed_candidate_localization",
        "annotation_scope": {
            "visible_evidence_only": True,
            "initial_key_signature_not_repeated_at_system_left": True,
            "source_final_note_clipped_or_missing": True,
            "annotation_errors_assigned_for_source_omissions": False,
        },
        "case": result,
        "diagnostic_decision": decision,
    }

    temp_dir.mkdir(parents=False, exist_ok=False)
    try:
        case_dir = temp_dir / "case"
        case_dir.mkdir()
        snapshots = paired._snapshot_case(case_dir, frozen, result)
        report["pins"] = {
            "evaluator": paired._snapshot_file(Path(__file__), temp_dir / "evaluator.py"),
            "paired_evaluator": paired._snapshot_file(
                Path(paired.__file__), temp_dir / "paired_evaluator.py"
            ),
            "shared_verifier": paired._snapshot_file(
                Path(heldout.__file__), temp_dir / "shared_verifier.py"
            ),
            "case_artifacts": snapshots,
        }
        report_path = temp_dir / "report.json"
        heldout._write_json(report_path, report)
        _write_markdown(temp_dir / "report.md", report)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "chispazo_internal_key_change_diagnostic_evaluation_manifest",
            "status": report["status"],
            "create_once": True,
            "evaluation_version": evaluation_version,
            "truth_opened_after_all_frozen_hashes_verified": True,
            "diagnostic_status": decision["status"],
            "promotable": False,
            "report_sha256": freezer.base._sha256(report_path),
            "pins": report["pins"],
        }
        heldout._write_json(temp_dir / "manifest.json", manifest)

        repeated = paired._verify_case(out_dir, CASE)
        if (
            repeated["sealed_sha256"] != frozen["sealed_sha256"]
            or repeated["freeze_sha256"] != frozen["freeze_sha256"]
            or repeated["prepared_sha256"] != frozen["prepared_sha256"]
        ):
            raise ValueError("Frozen Chispazo gate changed during evaluation")
        temp_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return {
        "evaluation_dir": str(output_dir),
        "manifest": str(output_dir / "manifest.json"),
        "report": str(output_dir / "report.json"),
        "report_markdown": str(output_dir / "report.md"),
    }


def _diagnostic_decision(comparison: Mapping[str, Any]) -> dict[str, Any]:
    diagnostic_supported = int(comparison["exact_pitch_match_delta"]) > 0
    return {
        "status": "diagnostic_supported" if diagnostic_supported else "diagnostic_not_supported",
        "promotable": False,
        "reason": (
            "The -2 lane was selected only as an inconclusive top-candidate diagnostic after "
            "the strict truth-blind detector returned unknown."
        ),
        "runtime_action": "keep automatic internal key changes out of runtime",
        "next_gate": (
            "repair the strict detector using consumed evidence, then freeze a new "
            "score-disjoint internal-change target"
        ),
    }


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    result = report["case"]
    context = result["source_musicxml_context"]
    comparison = result["comparison"]
    decision = report["diagnostic_decision"]
    key_events = ", ".join(
        f"m{event['measure_number']}={event['fifths']:+d}" for event in context["key_events"]
    )
    lines = [
        "# Chispazo Internal-Key Diagnostic",
        "",
        "The frozen gate was verified before the human MusicXML was opened.",
        "Candidate IDs, coordinates, and counts are identical in both pitch lanes.",
        "",
        f"- Physical measures: `{context['physical_measure_count']}`",
        f"- Automatic crops: `{result['selection_invariance']['automatic_crop_count']}`",
        f"- Visible transcribed noteheads: `{context['visible_notehead_count']}`",
        "- Frozen selected noteheads: "
        f"`{result['selection_invariance']['selected_notehead_count']}`",
        f"- MusicXML key events: `{key_events}`",
        f"- Baseline exact pitch matches: `{comparison['baseline_exact_pitch_matches']}`",
        "- Diagnostic -2 lane exact pitch matches: "
        f"`{comparison['automatic_exact_pitch_matches']}`",
        f"- Delta: `{comparison['exact_pitch_match_delta']:+d}`",
        "",
        "## Decision",
        "",
        f"**{decision['status']}**, but non-promotable: {decision['runtime_action']}.",
        "",
        "The transcription follows visible evidence. Source omissions at the system left and",
        "clipped final note are recorded as image limitations, not annotation mistakes.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
