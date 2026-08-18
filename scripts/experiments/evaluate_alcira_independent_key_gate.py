"""Evaluate the sealed Alcira new-PDF key-state gate exactly once.

The frozen selector and paired pitch lanes are hash-verified before the human
MusicXML is opened. This gate evaluates a strict two-sharp system-entry state;
candidate IDs, coordinates, and counts remain identical across both lanes.
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
CASE = paired.EvaluationCase(
    case_id="alcira_system_entry_change_s6",
    musicxml_name="alcira_system_006.musicxml",
    expected_time_signature="3/4",
    expected_key_fifths=2,
    expected_clef=("G", 2),
    mapping=(
        (1, ((1, 0, 6),)),
        (2, ((2, 0, 6),)),
        (3, ((3, 0, 6),)),
        (4, ((4, 0, 3),)),
        (5, ((5, 0, 6),)),
        (6, ((6, 0, 6),)),
    ),
    expected_key_events=((1, 2),),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "out_dir",
        nargs="?",
        type=Path,
        default=Path("out/local_restricted"),
    )
    parser.add_argument("--evaluation-version", default=DEFAULT_EVALUATION_VERSION)
    args = parser.parse_args(argv)
    try:
        result = evaluate_alcira_independent_key_gate(
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


def evaluate_alcira_independent_key_gate(
    out_dir: Path,
    *,
    evaluation_version: str = DEFAULT_EVALUATION_VERSION,
) -> dict[str, str]:
    """Verify the Alcira freeze, evaluate both lanes, and publish atomically."""
    paired._validate_version(evaluation_version)
    out_dir = out_dir.expanduser().resolve()

    # Verify every frozen hash and paired-lane invariant before opening MusicXML.
    frozen = paired._verify_case(out_dir, CASE)
    namespace_root = Path(frozen["namespace_root"])
    output_dir = namespace_root / f"evaluation_{evaluation_version}"
    temp_dir = namespace_root / f".evaluation_{evaluation_version}.tmp"
    if output_dir.exists():
        raise FileExistsError(f"Alcira evaluation already exists: {output_dir}")
    if temp_dir.exists():
        raise FileExistsError(f"Stale Alcira evaluation exists: {temp_dir}")

    result = paired._evaluate_case(CASE, frozen)
    prepared = heldout._read_json(Path(frozen["prepared_path"]))
    key_context = prepared["independent_key_gate"]
    decision = _gate_decision(result, key_context)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "alcira_new_pdf_system_entry_key_state_evaluation",
        "status": "evaluated_exactly_once_after_frozen_hashes_verified",
        "truth_opened_after_all_frozen_hashes_verified": True,
        "metric_scope": "pitch_only_with_fixed_candidate_localization",
        "annotation_scope": {
            "visible_evidence_only": True,
            "visible_dyads_encoded_as_musicxml_chords": True,
            "absolute_note_recall_includes_both_notes_of_each_visible_dyad": True,
            "key_state_delta_uses_identical_selected_candidates": True,
        },
        "case": result,
        "gate_decision": decision,
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
            "kind": "alcira_new_pdf_system_entry_key_state_evaluation_manifest",
            "status": report["status"],
            "create_once": True,
            "evaluation_version": evaluation_version,
            "truth_opened_after_all_frozen_hashes_verified": True,
            "gate_status": decision["status"],
            "promotion_scope": decision["promotion_scope"],
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
            raise ValueError("Frozen Alcira gate changed during evaluation")
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


def _gate_decision(
    result: Mapping[str, Any],
    key_context: Mapping[str, Any],
) -> dict[str, Any]:
    comparison = result["comparison"]
    strict_key_matches_truth = (
        key_context.get("strict_detector_fifths")
        == result["source_musicxml_context"]["key_fifths"]
        == 2
    )
    strict_lane = key_context.get("automatic_lane_kind") == "strict_automatic_key"
    localization_invariant = bool(result["selection_invariance"]["passed"])
    improved = int(comparison["exact_pitch_match_delta"]) > 0
    passed = strict_key_matches_truth and strict_lane and localization_invariant and improved
    return {
        "status": "passed" if passed else "failed",
        "acceptance_rule": (
            "Require a strict truth-blind key read matching the transcription, more exact "
            "ordered-pitch matches than the no-key lane, and unchanged candidate localization."
        ),
        "strict_key_matches_truth": strict_key_matches_truth,
        "strict_automatic_lane": strict_lane,
        "localization_invariant": localization_invariant,
        "exact_pitch_improved": improved,
        "promotion_scope": "strict_initial_or_system_entry_key_state" if passed else "none",
        "runtime_action": (
            "advance strict initial/system-entry key state to a bounded integration slice"
            if passed
            else "keep automatic key state out of runtime"
        ),
        "internal_change_scope": "not_evaluated_by_this_gate",
    }


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    result = report["case"]
    context = result["source_musicxml_context"]
    comparison = result["comparison"]
    decision = report["gate_decision"]
    baseline = result["lanes"][paired.LANE_BASELINE]["metrics"]["summary"]
    automatic = result["lanes"][paired.LANE_AUTOMATIC]["metrics"]["summary"]
    lines = [
        "# Alcira New-PDF System-Entry Key-State Evaluation",
        "",
        "The frozen gate was verified before the human MusicXML was opened.",
        "Candidate IDs, coordinates, and counts are identical in both pitch lanes.",
        "",
        f"- Physical measures: `{context['physical_measure_count']}`",
        f"- Visible MusicXML noteheads: `{context['visible_notehead_count']}`",
        "- Frozen selected noteheads: "
        f"`{result['selection_invariance']['selected_notehead_count']}`",
        f"- Human key: `{context['key_fifths']:+d}` fifths",
        f"- Frozen automatic key: `{result['frozen_key_context']['automatic_fifths']:+d}` fifths",
        f"- Baseline exact pitch matches: `{comparison['baseline_exact_pitch_matches']}`",
        f"- Automatic-key exact pitch matches: `{comparison['automatic_exact_pitch_matches']}`",
        f"- Exact-pitch delta: `{comparison['exact_pitch_match_delta']:+d}`",
        f"- Baseline alignment accuracy: `{comparison['baseline_alignment_accuracy']:.6f}`",
        f"- Automatic alignment accuracy: `{comparison['automatic_alignment_accuracy']:.6f}`",
        f"- Automatic note-count F1: `{automatic['note_count_f1']:.6f}`",
        "",
        "The source contains visible dyads, and the transcription preserves both noteheads.",
        "The frozen selector emits fewer anchors, so absolute note recall is limited; this does",
        "not confound the paired key-state delta because both lanes use the same anchors.",
        "",
        "## Decision",
        "",
        f"**{decision['status']}**: {decision['runtime_action']}.",
        "",
        f"The no-key lane note-count F1 is also `{baseline['note_count_f1']:.6f}` because key",
        "state changes pitch only, not localization or count.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
