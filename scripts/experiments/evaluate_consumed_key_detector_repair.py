"""Score the consumed Sobre key-detector repair without rewriting its frozen gate."""

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
from scripts.experiments import evaluate_independent_key_state_gates as evaluation  # noqa: E402
from scripts.experiments import run_independent_key_state_gate as runner  # noqa: E402
from scripts.experiments import run_third_score_heldout_inference as inference  # noqa: E402

SCHEMA_VERSION = 1
OUTPUT_SUBDIR = "vlm_melody_consumed_training/independent_key_detector_repair_v1"
DEFAULT_DETECTOR_REPORT = (
    "vlm_melody_consumed_training/consumed_key_signature_detector_v4_sobre_repair/report.json"
)
CASE_ID = "sobre_change"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pin(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _detector_fifths(report: Mapping[str, Any], *, expected_slug: str) -> int:
    if report.get("truth_used_for_prediction") is not False:
        raise ValueError("Detector report must declare truth_used_for_prediction=false")
    predictions = report.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != 1:
        raise ValueError("Repair evaluator requires exactly one detector prediction")
    prediction = predictions[0]
    if not isinstance(prediction, Mapping) or prediction.get("slug") != expected_slug:
        raise ValueError("Detector prediction targets the wrong score")
    source = prediction.get("input")
    if not isinstance(source, Mapping):
        raise ValueError("Detector prediction has no source image pin")
    image_path = Path(str(source.get("path", ""))).expanduser().resolve()
    if not image_path.is_file() or _sha256(image_path) != source.get("sha256"):
        raise ValueError("Detector source image is missing or changed")
    fifths = prediction.get("fifths")
    if not isinstance(fifths, int) or isinstance(fifths, bool):
        raise ValueError("Detector repair did not produce exact fifths")
    return fifths


def _selection_matches_frozen(
    repaired_rows: Sequence[Mapping[str, Any]], frozen_rows: Sequence[Mapping[str, Any]]
) -> bool:
    if len(repaired_rows) != len(frozen_rows):
        return False
    for repaired, frozen in zip(repaired_rows, frozen_rows, strict=True):
        if repaired["identity"] != frozen["identity"]:
            return False
        repaired_notes = repaired["lanes"]["global_no_key"]["notes"]
        frozen_notes = frozen["lanes"]["global_no_key"]["notes"]
        if [note["candidate_id"] for note in repaired_notes] != [
            note["candidate_id"] for note in frozen_notes
        ]:
            return False
        if [note["center"] for note in repaired_notes] != [note["center"] for note in frozen_notes]:
            return False
    return True


def _markdown(report: Mapping[str, Any]) -> str:
    metrics = report["pitch_metrics"]
    return "\n".join(
        [
            "# Consumed Independent-Key Detector Repair",
            "",
            "This is post-transcription model-selection evidence. The original sealed gate remains "
            "unchanged and its independent result remains `not_promoted`.",
            "",
            "| Lane | Key fifths | Exact pitch matches | Alignment accuracy |",
            "| --- | ---: | ---: | ---: |",
            f"| No key | unknown | {metrics['baseline']['exact_pitch_matches']} | "
            f"{metrics['baseline']['ordered_pitch_alignment_accuracy']} |",
            f"| Frozen automatic | {report['key_states']['frozen_automatic']} | "
            f"{metrics['frozen_automatic']['exact_pitch_matches']} | "
            f"{metrics['frozen_automatic']['ordered_pitch_alignment_accuracy']} |",
            f"| Repaired detector | {report['key_states']['repaired_detector']} | "
            f"{metrics['repaired_detector']['exact_pitch_matches']} | "
            f"{metrics['repaired_detector']['ordered_pitch_alignment_accuracy']} |",
            "",
            "Candidate IDs and coordinates unchanged: "
            f"`{report['selection_invariance']['passed']}`.",
            "",
        ]
    )


def evaluate(out_dir: Path, detector_report_path: Path, output_dir: Path) -> Path:
    out_dir = out_dir.expanduser().resolve()
    detector_report_path = detector_report_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    temp_dir = output_dir.with_name(f".{output_dir.name}.tmp")
    if output_dir.exists() or temp_dir.exists():
        raise FileExistsError(f"Refusing to overwrite consumed repair evidence: {output_dir}")

    case = next(item for item in evaluation.CASES if item.case_id == CASE_ID)
    frozen = evaluation._verify_case(out_dir, case)
    detector_report = _read_json(detector_report_path)
    repaired_fifths = _detector_fifths(detector_report, expected_slug=case.gate_case.slug)

    inference_path = Path(frozen["namespace_root"]) / "inference_v1/inference.jsonl"
    inference_rows = inference._read_jsonl(inference_path)
    repaired_rows, repaired_invariance = runner.build_paired_predictions(
        inference_rows, automatic_fifths=repaired_fifths
    )
    if not _selection_matches_frozen(repaired_rows, frozen["prediction_rows"]):
        raise ValueError("Repaired key replay changed frozen candidate localization")

    temp_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir()
    try:
        repaired_path = temp_dir / "repaired_paired_predictions.jsonl"
        inference._write_jsonl(repaired_path, repaired_rows)

        # Consumed truth is opened only after detector and localization checks complete.
        original = evaluation._evaluate_case(case, frozen)
        truth = heldout.load_visible_musicxml_truth(Path(frozen["musicxml_path"]))
        mapping = heldout.validate_and_materialize_mapping(
            evaluation._mapping_payload(case),
            truth=truth,
            crop_indices=sorted(frozen["predictions_by_crop"]),
            mode="explicit_post_transcription_note_spans",
        )
        truth_rows = heldout.build_truth_rows(frozen["requests_by_crop"], truth, mapping)
        repaired_predictions = {
            int(row["identity"]["automatic_measure_index"]): {
                "notes": row["lanes"]["global_automatic_key"]["notes"]
            }
            for row in repaired_rows
        }
        repaired_evaluation = heldout.evaluate_pitch_only(
            truth_rows,
            repaired_predictions,
            target=frozen["target"],
            mapping_mode="explicit_post_transcription_note_spans",
            report_kind="consumed_independent_key_detector_repair",
        )
        baseline = original["lanes"][evaluation.LANE_BASELINE]["metrics"]["summary"]
        frozen_automatic = original["lanes"][evaluation.LANE_AUTOMATIC]["metrics"]["summary"]
        repaired = repaired_evaluation["metrics"]["summary"]
        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": "consumed_independent_key_detector_repair",
            "claim_boundary": {
                "independent_heldout_claim": False,
                "pipeline_ready": False,
                "reason": "repair designed after Sobre transcription was opened",
            },
            "target": dict(frozen["target"]),
            "key_states": {
                "truth": case.expected_key_fifths,
                "frozen_automatic": original["frozen_key_context"]["automatic_fifths"],
                "repaired_detector": repaired_fifths,
            },
            "selection_invariance": {
                "passed": bool(repaired_invariance["passed"]),
                "matches_original_frozen_baseline": True,
                "selected_notehead_count": int(repaired_invariance["note_count"]),
            },
            "pitch_metrics": {
                "baseline": baseline,
                "frozen_automatic": frozen_automatic,
                "repaired_detector": repaired,
                "repair_delta_vs_frozen": repaired["exact_pitch_matches"]
                - frozen_automatic["exact_pitch_matches"],
                "repair_delta_vs_baseline": repaired["exact_pitch_matches"]
                - baseline["exact_pitch_matches"],
            },
            "pins": {
                "detector_report": _pin(detector_report_path),
                "inference": _pin(inference_path),
                "sealed_gate": _pin(Path(frozen["sealed_path"])),
                "musicxml": _pin(Path(frozen["musicxml_path"])),
                "repaired_predictions": _pin(repaired_path),
            },
        }
        heldout._write_json(temp_dir / "report.json", report)
        (temp_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
        heldout._write_json(
            temp_dir / "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "kind": report["kind"],
                "create_once": True,
                "report_sha256": _sha256(temp_dir / "report.json"),
                "original_frozen_gate_modified": False,
            },
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temp_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return output_dir / "report.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", type=Path, default=Path("out"))
    parser.add_argument("--detector-report", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    detector_report = args.detector_report or args.out_dir / DEFAULT_DETECTOR_REPORT
    output_dir = args.output_dir or args.out_dir / OUTPUT_SUBDIR
    try:
        report = evaluate(args.out_dir, detector_report, output_dir)
    except (FileExistsError, FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
