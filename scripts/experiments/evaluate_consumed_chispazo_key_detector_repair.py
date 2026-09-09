"""Replay the consumed Chispazo gate with the repaired strict key detector."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import evaluate_frozen_third_score_heldout as heldout  # noqa: E402
from scripts.experiments import evaluate_independent_key_state_gates as paired  # noqa: E402
from scripts.experiments import run_independent_key_state_gate as runner  # noqa: E402
from scripts.experiments import run_third_score_heldout_inference as inference  # noqa: E402

SCHEMA_VERSION = 1
CASE = paired.CHISPAZO_INTERNAL_CHANGE_CASE
EXPECTED_BOUNDARY_X_PX = 1284
EXPECTED_FIFTHS = -2
EXPECTED_METHOD = "hinted_one_strong_two_flat_shape_sequence"
DEFAULT_DETECTOR_REPORT = (
    "vlm_melody_consumed_training/internal_key_change_scan_v3_chispazo_repair/report.json"
)
DEFAULT_OUTPUT_SUBDIR = "vlm_melody_consumed_training/chispazo_key_detector_repair_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _detector_event(report: Mapping[str, Any]) -> dict[str, Any]:
    if report.get("truth_used_for_prediction") is not False:
        raise ValueError("Detector report must declare truth_used_for_prediction=false")
    matches = []
    for system in report.get("systems", []):
        if not isinstance(system, Mapping):
            continue
        scan = system.get("scan")
        if not isinstance(scan, Mapping):
            continue
        source = scan.get("input")
        if not isinstance(source, Mapping):
            continue
        source_path = Path(str(source.get("path", ""))).expanduser().resolve()
        if source_path.stem != "system_003" or source_path.parents[1].name != CASE.gate_case.slug:
            continue
        if not source_path.is_file() or _sha256(source_path) != source.get("sha256"):
            raise ValueError("Chispazo detector source image is missing or changed")
        for hit in scan.get("hits", []):
            boundary = hit.get("structural_boundary", {})
            if int(boundary.get("x_px", -1)) == EXPECTED_BOUNDARY_X_PX:
                matches.append((source, hit))
    if len(matches) != 1:
        raise ValueError(f"Expected one repaired Chispazo detector event, found {len(matches)}")
    source, hit = matches[0]
    if hit.get("fifths") != EXPECTED_FIFTHS or hit.get("selection_method") != EXPECTED_METHOD:
        raise ValueError("Chispazo detector event does not match the repaired strict rule")
    return {
        "source": dict(source),
        "boundary_x_px": EXPECTED_BOUNDARY_X_PX,
        "fifths": EXPECTED_FIFTHS,
        "selection_method": EXPECTED_METHOD,
        "selected_glyph_ids": list(hit.get("selected_glyph_ids", [])),
    }


def _replay_matches_frozen_diagnostic(
    repaired_rows: Sequence[Mapping[str, Any]],
    frozen_rows: Sequence[Mapping[str, Any]],
) -> bool:
    if len(repaired_rows) != len(frozen_rows):
        return False
    for repaired, frozen in zip(repaired_rows, frozen_rows, strict=True):
        if repaired["identity"] != frozen["identity"]:
            return False
        if repaired["lanes"] != frozen["lanes"]:
            return False
    return True


def evaluate(out_dir: Path, detector_report_path: Path, output_dir: Path) -> Path:
    out_dir = out_dir.expanduser().resolve()
    detector_report_path = detector_report_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    temp_dir = output_dir.with_name(f".{output_dir.name}.tmp")
    if output_dir.exists() or temp_dir.exists():
        raise FileExistsError(f"Refusing to overwrite consumed Chispazo repair: {output_dir}")

    detector_report = _read_json(detector_report_path)
    event = _detector_event(detector_report)
    frozen = paired._verify_case(out_dir, CASE)
    namespace_root = Path(frozen["namespace_root"])
    prepared = heldout._read_json(Path(frozen["prepared_path"]))
    original_key = prepared["independent_key_gate"]
    if (
        original_key.get("automatic_fifths") != event["fifths"]
        or original_key.get("key_event_x_px") != event["boundary_x_px"]
        or original_key.get("automatic_lane_kind") != "inconclusive_top_candidate_diagnostic"
    ):
        raise ValueError("Repaired detector event differs from the frozen diagnostic event")

    inference_rows = inference._read_jsonl(namespace_root / "inference_v1/inference.jsonl")
    request_rows = inference._read_jsonl(namespace_root / "requests.jsonl")
    crop_left_by_measure = {
        int(row["identity"]["automatic_measure_index"]): int(row["input"]["bbox_px"][0])
        for row in request_rows
    }
    repaired_rows, invariance = runner.build_paired_predictions(
        inference_rows,
        automatic_fifths=int(event["fifths"]),
        key_event_x_px=int(event["boundary_x_px"]),
        crop_left_by_measure=crop_left_by_measure,
    )
    if not invariance["passed"] or not _replay_matches_frozen_diagnostic(
        repaired_rows, frozen["prediction_rows"]
    ):
        raise ValueError("Repaired detector replay differs from frozen Chispazo localization")

    # Chispazo is consumed; truth is opened only after detector and replay checks.
    result = paired._evaluate_case(CASE, frozen)
    comparison = result["comparison"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "consumed_chispazo_key_detector_repair",
        "claim_boundary": {
            "independent_heldout_claim": False,
            "pipeline_ready": False,
            "reason": "strict acceptance was repaired after Chispazo transcription was opened",
        },
        "detector_event": event,
        "frozen_gate": {
            "sealed_manifest": str(frozen["sealed_path"]),
            "sealed_manifest_sha256": frozen["sealed_sha256"],
            "original_lane_kind": original_key["automatic_lane_kind"],
        },
        "selection_invariance": {
            "passed": True,
            "measure_count": len(repaired_rows),
            "note_count": invariance["note_count"],
            "replay_matches_frozen_diagnostic": True,
        },
        "pitch_metrics": {
            "baseline_exact_pitch_matches": comparison["baseline_exact_pitch_matches"],
            "repaired_detector_exact_pitch_matches": comparison["automatic_exact_pitch_matches"],
            "exact_pitch_match_delta": comparison["exact_pitch_match_delta"],
            "baseline_alignment_accuracy": comparison["baseline_alignment_accuracy"],
            "repaired_detector_alignment_accuracy": comparison["automatic_alignment_accuracy"],
        },
        "next_gate": (
            "freeze a new score-disjoint positive internal-change target from source material "
            "that was not part of this 112-system model-selection sweep"
        ),
    }

    temp_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir()
    try:
        repaired_path = temp_dir / "repaired_paired_predictions.jsonl"
        inference._write_jsonl(repaired_path, repaired_rows)
        report["pins"] = {
            "detector_report": {
                "path": str(detector_report_path),
                "sha256": _sha256(detector_report_path),
            },
            "repaired_predictions": {
                "path": repaired_path.name,
                "sha256": _sha256(repaired_path),
            },
            "evaluator": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__))},
        }
        report_path = temp_dir / "report.json"
        heldout._write_json(report_path, report)
        _write_markdown(temp_dir / "report.md", report)
        temp_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return output_dir / "report.json"


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    metrics = report["pitch_metrics"]
    lines = [
        "# Consumed Chispazo Key-Detector Repair",
        "",
        "The repaired strict detector now accepts the two-flat event at x=1284.",
        "This is consumed model-selection evidence, not a new heldout result.",
        "",
        f"- Baseline exact pitch matches: `{metrics['baseline_exact_pitch_matches']}`",
        "- Repaired detector exact pitch matches: "
        f"`{metrics['repaired_detector_exact_pitch_matches']}`",
        f"- Delta: `{metrics['exact_pitch_match_delta']:+d}`",
        "- Candidate IDs, coordinates, counts, and replayed pitches unchanged: `True`",
        "",
        f"Next gate: {report['next_gate']}.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", type=Path, default=Path("out"))
    parser.add_argument("--detector-report", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    detector_report = args.detector_report or args.out_dir / DEFAULT_DETECTOR_REPORT
    output_dir = args.output_dir or args.out_dir / DEFAULT_OUTPUT_SUBDIR
    try:
        report = evaluate(args.out_dir, detector_report, output_dir)
    except (FileExistsError, FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
