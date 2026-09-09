"""Evaluate both sealed independent automatic-key gates exactly once.

Both frozen gates are fully verified before either human MusicXML file is
opened. The evaluator then scores the no-key and automatic-key pitch lanes
against explicit post-transcription note-span mappings. It never changes the
frozen requests, candidate selection, or predictions.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import evaluate_frozen_third_score_heldout as heldout  # noqa: E402
from scripts.experiments import freeze_independent_key_state_gates as freezer  # noqa: E402
from scripts.experiments import run_independent_key_state_gate as runner  # noqa: E402

SCHEMA_VERSION = 1
DEFAULT_EVALUATION_VERSION = "v1"
OUTPUT_SUBDIR = "vlm_melody_independent_key_gate_evaluation"
LANE_BASELINE = "global_no_key"
LANE_AUTOMATIC = "global_automatic_key"


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    musicxml_name: str
    expected_time_signature: str
    expected_key_fifths: int
    expected_clef: tuple[str, int]
    mapping: tuple[tuple[int, tuple[tuple[int, int, int], ...]], ...]
    expected_key_events: tuple[tuple[int, int], ...] = ()

    @property
    def gate_case(self) -> freezer.KeyGateCase:
        available = {**freezer.CASES, **freezer.CHALLENGE_CASES}
        return available[self.case_id]


# These note spans were fixed only after the independent transcriptions exposed
# one false internal split in each automatic segmentation. They are evaluation
# provenance and were never available to candidate selection or pitch inference.
CASES = (
    EvaluationCase(
        case_id="estrella_initial_s3",
        musicxml_name="estrella_system_003.musicxml",
        expected_time_signature="2/4",
        expected_key_fifths=-1,
        expected_clef=("G", 2),
        mapping=(
            (1, ((1, 0, 7),)),
            (2, ((2, 0, 7),)),
            (3, ((3, 0, 2),)),
            (4, ((4, 0, 5),)),
            (5, ((4, 5, 7),)),
            (6, ((5, 0, 7),)),
        ),
    ),
    EvaluationCase(
        case_id="sobre_change",
        musicxml_name="sobre_el_humo_system_007.musicxml",
        expected_time_signature="3/4",
        expected_key_fifths=2,
        expected_clef=("G", 2),
        mapping=(
            (1, ((1, 0, 2),)),
            (2, ((1, 2, 6),)),
            (3, ((2, 0, 6),)),
            (4, ((3, 0, 5),)),
            (5, ((4, 0, 3),)),
            (6, ((5, 0, 6),)),
            (7, ((6, 0, 6),)),
        ),
    ),
)


# This mapping was fixed after the Chispazo gate was sealed and the user
# transcription exposed one missed physical barline: automatic crop 5 contains
# physical measures 5 and 6, including the internal key change.
CHISPAZO_INTERNAL_CHANGE_CASE = EvaluationCase(
    case_id="chispazo_internal_change_s3",
    musicxml_name="chispazo_system_003.musicxml",
    expected_time_signature="3/4",
    expected_key_fifths=-2,
    expected_clef=("G", 2),
    mapping=(
        (1, ((1, 0, 6),)),
        (2, ((2, 0, 3),)),
        (3, ((3, 0, 3),)),
        (4, ((4, 0, 7),)),
        (5, ((5, 0, 2), (6, 0, 1))),
        (6, ((7, 0, 5),)),
        (7, ((8, 0, 1),)),
        (8, ((9, 0, 5),)),
    ),
    expected_key_events=((1, 2), (6, -2)),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", type=Path, default=Path("out"))
    parser.add_argument(
        "--evaluation-version",
        default=DEFAULT_EVALUATION_VERSION,
        help="Create-once output version (default: v1).",
    )
    args = parser.parse_args(argv)
    try:
        result = evaluate_independent_key_gates(
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


def evaluate_independent_key_gates(
    out_dir: Path,
    *,
    evaluation_version: str = DEFAULT_EVALUATION_VERSION,
    cases: Sequence[EvaluationCase] = CASES,
) -> dict[str, str]:
    """Verify both gates, score both lanes, and publish one atomic report."""
    _validate_version(evaluation_version)
    out_dir = out_dir.expanduser().resolve()
    output_root = out_dir / OUTPUT_SUBDIR
    output_dir = output_root / evaluation_version
    temp_dir = output_root / f".{evaluation_version}.tmp"
    if output_dir.exists():
        raise FileExistsError(f"Independent key evaluation already exists: {output_dir}")
    if temp_dir.exists():
        raise FileExistsError(f"Stale independent key evaluation exists: {temp_dir}")

    # The complete frozen state of every case is verified before any truth is opened.
    verified = {case.case_id: _verify_case(out_dir, case) for case in cases}

    case_results = []
    for case in cases:
        frozen = verified[case.case_id]
        case_results.append(_evaluate_case(case, frozen))

    decision = _promotion_decision(case_results)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "independent_automatic_key_state_two_score_evaluation",
        "status": "evaluated_exactly_once_after_both_gates_verified",
        "truth_opened_after_all_frozen_hashes_verified": True,
        "metric_scope": "pitch_only_with_fixed_candidate_localization",
        "cases": case_results,
        "aggregate": _aggregate(case_results),
        "promotion_decision": decision,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=False, exist_ok=False)
    try:
        case_pins = {}
        for case, result in zip(cases, case_results, strict=True):
            frozen = verified[case.case_id]
            case_dir = temp_dir / "cases" / case.case_id
            case_dir.mkdir(parents=True)
            snapshots = _snapshot_case(case_dir, frozen, result)
            result["pins"] = snapshots
            case_pins[case.case_id] = snapshots

        report["pins"] = {
            "evaluator": _snapshot_file(Path(__file__).resolve(), temp_dir / "evaluator.py"),
            "shared_verifier": _snapshot_file(
                Path(heldout.__file__).resolve(), temp_dir / "shared_verifier.py"
            ),
            "case_artifacts": case_pins,
        }
        report_path = temp_dir / "report.json"
        heldout._write_json(report_path, report)
        _write_markdown(temp_dir / "report.md", report)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "independent_automatic_key_state_evaluation_manifest",
            "status": report["status"],
            "create_once": True,
            "evaluation_version": evaluation_version,
            "truth_opened_after_all_frozen_hashes_verified": True,
            "promotion_status": decision["status"],
            "report_sha256": freezer.base._sha256(report_path),
            "pins": report["pins"],
        }
        heldout._write_json(temp_dir / "manifest.json", manifest)

        for case in cases:
            repeated = _verify_case(out_dir, case)
            original = verified[case.case_id]
            if (
                repeated["sealed_sha256"] != original["sealed_sha256"]
                or repeated["freeze_sha256"] != original["freeze_sha256"]
                or repeated["prepared_sha256"] != original["prepared_sha256"]
            ):
                raise ValueError(f"Frozen gate changed during evaluation: {case.case_id}")
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


def _verify_case(out_dir: Path, case: EvaluationCase) -> dict[str, Any]:
    gate = case.gate_case
    namespace_root = (
        out_dir
        / gate.slug
        / freezer.OUTPUT_SUBDIR
        / gate.namespace
        / f"system_{gate.target_system_index:03d}"
    )
    sealed_path = namespace_root / "frozen" / "sealed_manifest.json"
    frozen = heldout.verify_frozen_gate(sealed_path)
    expected_target = {"slug": gate.slug, "system_index": gate.target_system_index}
    if frozen["target"] != expected_target:
        raise ValueError(f"Unexpected frozen target for {case.case_id}: {frozen['target']}")

    prediction_rows = [
        frozen["predictions_by_crop"][index] for index in sorted(frozen["predictions_by_crop"])
    ]
    invariance = runner.selection_invariance(prediction_rows)
    return {
        **frozen,
        "namespace_root": namespace_root,
        "sealed_path": sealed_path,
        "musicxml_path": namespace_root / case.musicxml_name,
        "selection_invariance": invariance,
        "prediction_rows": prediction_rows,
    }


def _evaluate_case(case: EvaluationCase, frozen: Mapping[str, Any]) -> dict[str, Any]:
    musicxml_path = Path(frozen["musicxml_path"])
    if not musicxml_path.is_file():
        raise FileNotFoundError(f"Human MusicXML does not exist: {musicxml_path}")
    truth = heldout.load_visible_musicxml_truth(musicxml_path)
    _validate_truth_context(case, truth)

    mapping_payload = _mapping_payload(case)
    mapping = heldout.validate_and_materialize_mapping(
        mapping_payload,
        truth=truth,
        crop_indices=sorted(frozen["predictions_by_crop"]),
        mode="explicit_post_transcription_note_spans",
    )
    truth_rows = heldout.build_truth_rows(frozen["requests_by_crop"], truth, mapping)
    lane_reports = {}
    for lane in (LANE_BASELINE, LANE_AUTOMATIC):
        predictions = {
            crop: {"notes": row["lanes"][lane]["notes"]}
            for crop, row in frozen["predictions_by_crop"].items()
        }
        lane_reports[lane] = heldout.evaluate_pitch_only(
            truth_rows,
            predictions,
            target=frozen["target"],
            mapping_mode="explicit_post_transcription_note_spans",
            report_kind=f"independent_key_state_{lane}_pitch_evaluation",
        )

    baseline = lane_reports[LANE_BASELINE]["metrics"]["summary"]
    automatic = lane_reports[LANE_AUTOMATIC]["metrics"]["summary"]
    prepared = heldout._read_json(Path(frozen["prepared_path"]))
    key_config = prepared["independent_key_gate"]
    return {
        "case_id": case.case_id,
        "target": dict(frozen["target"]),
        "source_musicxml_context": {
            "time_signature": truth.time_signature,
            "key_fifths": truth.key_fifths,
            "key_events": [
                {"measure_number": measure, "fifths": fifths}
                for measure, fifths in truth.key_events
            ],
            "clef": list(truth.clef) if truth.clef is not None else None,
            "physical_measure_count": len(truth.measure_numbers),
            "visible_notehead_count": sum(
                len(truth.notes_by_measure[number]) for number in truth.measure_numbers
            ),
        },
        "frozen_key_context": {
            "baseline_fifths": key_config["baseline_fifths"],
            "automatic_fifths": key_config["automatic_fifths"],
        },
        "selection_invariance": {
            "passed": bool(frozen["selection_invariance"]["passed"]),
            "automatic_crop_count": len(frozen["predictions_by_crop"]),
            "selected_notehead_count": sum(
                len(row["lanes"][LANE_BASELINE]["notes"])
                for row in frozen["predictions_by_crop"].values()
            ),
        },
        "mapping": mapping,
        "truth_rows": truth_rows,
        "lanes": lane_reports,
        "comparison": {
            "baseline_exact_pitch_matches": baseline["exact_pitch_matches"],
            "automatic_exact_pitch_matches": automatic["exact_pitch_matches"],
            "exact_pitch_match_delta": (
                automatic["exact_pitch_matches"] - baseline["exact_pitch_matches"]
            ),
            "baseline_alignment_accuracy": baseline["ordered_pitch_alignment_accuracy"],
            "automatic_alignment_accuracy": automatic["ordered_pitch_alignment_accuracy"],
            "score_improved": (automatic["exact_pitch_matches"] > baseline["exact_pitch_matches"]),
            "score_regressed": (automatic["exact_pitch_matches"] < baseline["exact_pitch_matches"]),
        },
        "musicxml_path": str(musicxml_path),
        "sealed_path": str(frozen["sealed_path"]),
    }


def _validate_truth_context(case: EvaluationCase, truth: heldout.VisibleMusicXMLTruth) -> None:
    if truth.time_signature != case.expected_time_signature:
        raise ValueError(f"Unexpected {case.case_id} time signature: {truth.time_signature}")
    if truth.key_fifths != case.expected_key_fifths:
        raise ValueError(f"Unexpected {case.case_id} key fifths: {truth.key_fifths}")
    if case.expected_key_events and truth.key_events != case.expected_key_events:
        raise ValueError(f"Unexpected {case.case_id} key events: {truth.key_events}")
    if truth.clef != case.expected_clef:
        raise ValueError(f"Unexpected {case.case_id} clef: {truth.clef}")


def _mapping_payload(case: EvaluationCase) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "automatic_crops": [
            {
                "automatic_crop_index": crop,
                "physical_measure_spans": [
                    {
                        "measure_number": measure,
                        "note_start": start,
                        "note_end": end,
                    }
                    for measure, start, end in spans
                ],
            }
            for crop, spans in case.mapping
        ],
    }


def _promotion_decision(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    improved = [
        str(result["case_id"]) for result in case_results if result["comparison"]["score_improved"]
    ]
    regressed = [
        str(result["case_id"]) for result in case_results if result["comparison"]["score_regressed"]
    ]
    invariant = all(result["selection_invariance"]["passed"] for result in case_results)
    passed = len(improved) == len(case_results) and not regressed and invariant
    return {
        "status": "promoted" if passed else "not_promoted",
        "preregistered_rule": (
            "Require more exact ordered-pitch matches on both scores, no score-level "
            "regression, and unchanged candidate localization."
        ),
        "localization_invariant": invariant,
        "improved_cases": improved,
        "regressed_cases": regressed,
        "runtime_action": (
            "integrate automatic key state" if passed else "keep automatic key state out of runtime"
        ),
    }


def _aggregate(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baseline = sum(
        int(result["comparison"]["baseline_exact_pitch_matches"]) for result in case_results
    )
    automatic = sum(
        int(result["comparison"]["automatic_exact_pitch_matches"]) for result in case_results
    )
    truth_notes = sum(
        int(result["source_musicxml_context"]["visible_notehead_count"]) for result in case_results
    )
    return {
        "score_count": len(case_results),
        "truth_visible_notehead_count": truth_notes,
        "baseline_exact_pitch_matches": baseline,
        "automatic_exact_pitch_matches": automatic,
        "exact_pitch_match_delta": automatic - baseline,
    }


def _snapshot_case(
    case_dir: Path,
    frozen: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    mapping_path = case_dir / "mapping.json"
    truth_path = case_dir / "truth.jsonl"
    heldout._write_json(mapping_path, result["mapping"])
    heldout._write_jsonl(truth_path, result["truth_rows"])
    snapshots = {
        "source_musicxml": _snapshot_file(
            Path(frozen["musicxml_path"]), case_dir / "source.musicxml"
        ),
        "mapping": heldout._snapshot_record(mapping_path),
        "truth": heldout._snapshot_record(truth_path),
        "sealed_manifest": _snapshot_file(
            Path(frozen["sealed_path"]), case_dir / "frozen_sealed_manifest.json"
        ),
        "freeze_manifest": _snapshot_file(
            Path(frozen["freeze_path"]), case_dir / "frozen_freeze.json"
        ),
        "paired_predictions": _snapshot_file(
            Path(frozen["namespace_root"])
            / frozen["freeze"]["predictions"]["snapshot_path_relative_to_namespace"],
            case_dir / "frozen_paired_predictions.jsonl",
        ),
    }
    return snapshots


def _snapshot_file(source: Path, destination: Path) -> dict[str, Any]:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Snapshot source does not exist: {source}")
    shutil.copyfile(source, destination)
    return heldout._snapshot_record(
        destination,
        source_path=source,
        source_sha256=freezer.base._sha256(source),
    )


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# Independent Automatic-Key Evaluation",
        "",
        "Both gates were frozen and verified before either MusicXML transcription was opened.",
        "Only pitch changes are scored; candidate IDs, coordinates, and counts are identical.",
        "",
        "| Case | Human key | Automatic key | Baseline matches | Automatic matches | Delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in report["cases"]:
        comparison = result["comparison"]
        lines.append(
            f"| {result['case_id']} | {result['source_musicxml_context']['key_fifths']} "
            f"| {result['frozen_key_context']['automatic_fifths']} "
            f"| {comparison['baseline_exact_pitch_matches']} "
            f"| {comparison['automatic_exact_pitch_matches']} "
            f"| {comparison['exact_pitch_match_delta']:+d} |"
        )
    aggregate = report["aggregate"]
    decision = report["promotion_decision"]
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Baseline exact pitch matches: `{aggregate['baseline_exact_pitch_matches']}`",
            f"- Automatic-key exact pitch matches: `{aggregate['automatic_exact_pitch_matches']}`",
            f"- Delta: `{aggregate['exact_pitch_match_delta']:+d}`",
            "",
            "## Decision",
            "",
            f"**{decision['status']}**: {decision['runtime_action']}.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _validate_version(value: str) -> None:
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789_-"
    if not value or any(character not in allowed for character in value):
        raise ValueError(f"Invalid evaluation version: {value}")


if __name__ == "__main__":
    raise SystemExit(main())
