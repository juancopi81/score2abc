"""Import and score the frozen Carrizal second-score heldout exactly once.

The recognizer produced seven automatic crops before canonical truth existed.
The approved MusicXML contains eight physical measures because automatic crop 2
spans physical measures 2 and 3. This evaluator preserves that segmentation
failure instead of rewriting either the requests or predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from score2abc.events import measure_length_beats  # noqa: E402
from score2abc.musicxml import parse_musicxml_events  # noqa: E402
from scripts import build_vlm_melody_event_benchmark as benchmark  # noqa: E402
from scripts.experiments import spike_composed_melody_chain as composed  # noqa: E402

TARGET_SLUG = "jaime-llanos_19_carrizal_pasillo_emilio-murillo"
TARGET_SYSTEM = 4
OUTPUT_SUBDIR = "vlm_melody_fresh_heldout"
SPLIT_NAME = "fresh_heldout"
DEFAULT_MUSICXML_NAME = "carrizal_system_4.musicxml"
EXPECTED_TIME_SIGNATURE = "3/4"
EXPECTED_KEY_FIFTHS = -1
EXPECTED_CLEF = ("G", 2)
EXPECTED_FREEZE_SHA256 = "af7814c0d00888fbc7af3a579cfd746781558062cb9c5b498faadd65e9f28b96"
EXPECTED_REQUESTS_SHA256 = "bf061a1224eef6e473a5e0569d444d107b65724b0a7be19ef0a5d7575c428cf0"
EXPECTED_PREDICTIONS_SHA256 = "654f7671da37fe3d62e9b776a5b4ad5ab0cc0a44383ad5bddd864065c0cb430d"

# Defined only after the independent transcription exposed the missed barline.
# This mapping is evaluation provenance, never inference input.
CROP_TO_PHYSICAL_MEASURES: dict[int, tuple[int, ...]] = {
    1: (1,),
    2: (2, 3),
    3: (4,),
    4: (5,),
    5: (6,),
    6: (7,),
    7: (8,),
}


@dataclass(frozen=True)
class MusicXMLTruth:
    payload: dict[str, Any]
    measure_numbers: tuple[int, ...]
    measure_extents: dict[int, Fraction]
    rests_by_measure: dict[int, tuple[dict[str, Any], ...]]
    key_fifths: int | None
    clef: tuple[str, int] | None


TruthLoader = Callable[[Path], MusicXMLTruth]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", type=Path, default=Path("out"))
    parser.add_argument("--musicxml", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        report = evaluate_fresh_heldout(args.out_dir, musicxml_path=args.musicxml)
    except (FileNotFoundError, KeyError, OSError, ValueError, ET.ParseError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(report["artifacts"]["evaluation"])
    print(json.dumps(report["metrics"]["summary"], indent=2, sort_keys=True))
    return 0


def evaluate_fresh_heldout(
    out_dir: Path,
    *,
    musicxml_path: Path | None = None,
    slug: str = TARGET_SLUG,
    system_index: int = TARGET_SYSTEM,
    crop_to_physical_measures: Mapping[int, Sequence[int]] = CROP_TO_PHYSICAL_MEASURES,
    expected_time_signature: str = EXPECTED_TIME_SIGNATURE,
    expected_key_fifths: int = EXPECTED_KEY_FIFTHS,
    expected_clef: tuple[str, int] = EXPECTED_CLEF,
    expected_freeze_sha256: str | None = EXPECTED_FREEZE_SHA256,
    expected_requests_sha256: str | None = EXPECTED_REQUESTS_SHA256,
    expected_predictions_sha256: str | None = EXPECTED_PREDICTIONS_SHA256,
    truth_loader: TruthLoader = lambda path: load_musicxml_truth(path),
) -> dict[str, Any]:
    root = out_dir / slug / OUTPUT_SUBDIR / f"system_{system_index:03d}"
    split_dir = root / SPLIT_NAME
    source_path = musicxml_path or root / DEFAULT_MUSICXML_NAME
    source_path = source_path.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Canonical MusicXML does not exist: {source_path}")

    frozen = _verify_frozen_state(
        root,
        split_dir,
        expected_freeze_sha256=expected_freeze_sha256,
        expected_requests_sha256=expected_requests_sha256,
        expected_predictions_sha256=expected_predictions_sha256,
    )

    evaluation_path = split_dir / "evaluation.json"
    truth_path = split_dir / "truth.jsonl"
    import_path = split_dir / "truth_import.json"
    report_path = root / "evaluation.md"
    source_sha256 = composed._sha256(source_path)
    if evaluation_path.exists():
        return _load_existing_evaluation(
            evaluation_path,
            source_sha256=source_sha256,
            freeze_sha256=frozen["freeze_sha256"],
        )
    partial = [path for path in (truth_path, import_path, report_path) if path.exists()]
    if partial:
        raise ValueError(
            "Refusing to overwrite partial one-shot evaluation artifacts: "
            + ", ".join(str(path) for path in partial)
        )

    # No MusicXML parsing occurs before all frozen requests/predictions are verified.
    truth = truth_loader(source_path)
    mapping = {
        int(crop): tuple(int(measure) for measure in physical)
        for crop, physical in crop_to_physical_measures.items()
    }
    measure_length = _validate_musicxml_truth(
        truth,
        mapping=mapping,
        expected_time_signature=expected_time_signature,
        expected_key_fifths=expected_key_fifths,
        expected_clef=expected_clef,
    )
    truth_rows = build_mapped_truth_rows(
        frozen["requests"],
        truth,
        mapping=mapping,
        measure_length=measure_length,
    )
    metrics = benchmark.evaluate_predictions(truth_rows, frozen["predictions"])

    benchmark._write_jsonl(truth_path, truth_rows)
    truth_sha256 = composed._sha256(truth_path)
    evaluator_path = Path(__file__).resolve()
    segmentation = _segmentation_summary(mapping, truth.measure_numbers)
    import_payload = {
        "schema_version": 1,
        "kind": "post_freeze_canonical_musicxml_import",
        "truth_read_after_freeze_verification": True,
        "source": {
            "path": composed._display_path(source_path),
            "sha256": source_sha256,
            "time_signature": truth.payload["time_signature"],
            "key_fifths": truth.key_fifths,
            "clef": {"sign": truth.clef[0], "line": truth.clef[1]},
        },
        "freeze": {
            "path": composed._display_path(frozen["freeze_path"]),
            "sha256": frozen["freeze_sha256"],
            "requests_sha256": frozen["requests_sha256"],
            "predictions_sha256": frozen["predictions_sha256"],
        },
        "mapping": segmentation,
        "truth": {
            "path": composed._display_path(truth_path),
            "sha256": truth_sha256,
        },
        "evaluator": {
            "path": composed._display_path(evaluator_path),
            "sha256": composed._sha256(evaluator_path),
        },
    }
    _write_json(import_path, import_payload)

    decision = _integration_decision(metrics["summary"])
    report = {
        "schema_version": 1,
        "kind": "one_shot_second_score_evaluation",
        "status": "evaluated_once_after_frozen_predictions",
        "freeze_sha256": frozen["freeze_sha256"],
        "musicxml_sha256": source_sha256,
        "truth_sha256": truth_sha256,
        "truth_import_sha256": composed._sha256(import_path),
        "segmentation": segmentation,
        "metrics": metrics,
        "pipeline_integration_decision": decision,
        "artifacts": {
            "musicxml": composed._display_path(source_path),
            "truth": composed._display_path(truth_path),
            "truth_import": composed._display_path(import_path),
            "evaluation": composed._display_path(evaluation_path),
            "report_markdown": composed._display_path(report_path),
        },
    }
    _write_json(evaluation_path, report)
    _write_markdown(report_path, report)
    return report


def load_musicxml_truth(path: Path) -> MusicXMLTruth:
    payload = parse_musicxml_events(path)
    root = ET.parse(path).getroot()
    _strip_namespaces(root)
    parts = root.findall("part")
    if len(parts) != 1:
        raise ValueError(f"Expected exactly one MusicXML part, found {len(parts)}")

    divisions = 1
    key_fifths: int | None = None
    clef: tuple[str, int] | None = None
    measure_numbers: list[int] = []
    extents: dict[int, Fraction] = {}
    rests: dict[int, tuple[dict[str, Any], ...]] = {}
    for fallback, measure in enumerate(parts[0].findall("measure"), start=1):
        number = _measure_number(measure, fallback)
        measure_numbers.append(number)
        attributes = measure.find("attributes")
        if attributes is not None:
            divisions = int(attributes.findtext("divisions", str(divisions)))
            fifths_text = attributes.findtext("key/fifths")
            if fifths_text is not None:
                key_fifths = int(fifths_text)
            sign = attributes.findtext("clef/sign")
            line = attributes.findtext("clef/line")
            if sign is not None and line is not None:
                clef = (sign.strip(), int(line))

        cursor = 0
        last_note_onset = 0
        max_cursor = 0
        measure_rests: list[dict[str, Any]] = []
        for child in measure:
            if child.tag == "backup":
                cursor -= int(child.findtext("duration", "0"))
                continue
            if child.tag == "forward":
                cursor += int(child.findtext("duration", "0"))
                max_cursor = max(max_cursor, cursor)
                continue
            if child.tag != "note":
                continue
            duration = int(child.findtext("duration", "0"))
            if duration <= 0:
                continue
            is_chord_tone = child.find("chord") is not None
            onset = last_note_onset if is_chord_tone else cursor
            if child.find("rest") is not None:
                measure_rests.append(
                    {
                        "onset_beats": _fraction_number(Fraction(onset, divisions)),
                        "duration_beats": _fraction_number(Fraction(duration, divisions)),
                    }
                )
            if not is_chord_tone:
                cursor += duration
                last_note_onset = onset
                max_cursor = max(max_cursor, cursor)
        extents[number] = Fraction(max_cursor, divisions)
        rests[number] = tuple(measure_rests)

    return MusicXMLTruth(
        payload=dict(payload),
        measure_numbers=tuple(measure_numbers),
        measure_extents=extents,
        rests_by_measure=rests,
        key_fifths=key_fifths,
        clef=clef,
    )


def build_mapped_truth_rows(
    requests: Sequence[Mapping[str, Any]],
    truth: MusicXMLTruth,
    *,
    mapping: Mapping[int, Sequence[int]],
    measure_length: Fraction,
) -> list[dict[str, Any]]:
    requests_by_crop = {int(row["identity"]["system_measure_index"]): row for row in requests}
    if set(requests_by_crop) != set(mapping):
        raise ValueError(
            f"Frozen crop identities do not match evaluation mapping: "
            f"requests={sorted(requests_by_crop)}, mapping={sorted(mapping)}"
        )

    notes_by_measure: dict[int, list[dict[str, Any]]] = {}
    for raw_note in truth.payload.get("notes") or []:
        notes_by_measure.setdefault(int(raw_note["measure"]), []).append(dict(raw_note))

    rows: list[dict[str, Any]] = []
    for crop_index in sorted(mapping):
        physical_measures = tuple(mapping[crop_index])
        notes: list[dict[str, Any]] = []
        rests: list[dict[str, Any]] = []
        for offset_index, physical_measure in enumerate(physical_measures):
            offset = measure_length * offset_index
            for raw_note in notes_by_measure.get(physical_measure, []):
                note = {
                    "onset_beats": _fraction_number(
                        offset + Fraction(str(raw_note["onset_beats"]))
                    ),
                    "duration_beats": raw_note["duration_beats"],
                    "pitch_midi": int(raw_note["pitch_midi"]),
                }
                if raw_note.get("accidental") is not None:
                    note["accidental"] = int(raw_note["accidental"])
                notes.append(note)
            for raw_rest in truth.rests_by_measure.get(physical_measure, ()):
                rests.append(
                    {
                        "onset_beats": _fraction_number(
                            offset + Fraction(str(raw_rest["onset_beats"]))
                        ),
                        "duration_beats": raw_rest["duration_beats"],
                    }
                )
        notes.sort(key=lambda row: (float(row["onset_beats"]), int(row["pitch_midi"])))
        rests.sort(key=lambda row: float(row["onset_beats"]))
        rows.append(
            {
                "schema_version": 1,
                "identity": dict(requests_by_crop[crop_index]["identity"]),
                "measure_extent_beats": _fraction_number(measure_length * len(physical_measures)),
                "physical_measure_numbers": list(physical_measures),
                "notes": notes,
                "rests": rests,
            }
        )
    return rows


def _verify_frozen_state(
    root: Path,
    split_dir: Path,
    *,
    expected_freeze_sha256: str | None,
    expected_requests_sha256: str | None,
    expected_predictions_sha256: str | None,
) -> dict[str, Any]:
    freeze_path = split_dir / "freeze.json"
    predictions_path = split_dir / "predictions.jsonl"
    if not freeze_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError(f"Frozen heldout artifacts are missing under {split_dir}")

    freeze_sha256 = composed._sha256(freeze_path)
    if expected_freeze_sha256 and freeze_sha256 != expected_freeze_sha256:
        raise ValueError(
            f"Frozen prediction manifest changed: {freeze_sha256} != {expected_freeze_sha256}"
        )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen_before_truth" or freeze.get("split") != SPLIT_NAME:
        raise ValueError(f"Invalid fresh heldout freeze: {freeze_path}")

    requests_path = composed._path_from_display(str(freeze["requests"]["path"]))
    requests_sha256 = composed._sha256(requests_path)
    if requests_sha256 != str(freeze["requests"]["sha256"]):
        raise ValueError("Frozen request file changed")
    if expected_requests_sha256 and requests_sha256 != expected_requests_sha256:
        raise ValueError("Frozen request hash no longer matches the preregistered request hash")

    for artifact in freeze.get("artifacts") or []:
        path = composed._path_from_display(str(artifact["path"]))
        if not path.is_file() or composed._sha256(path) != str(artifact["sha256"]):
            raise ValueError(f"Frozen prediction artifact changed or is missing: {path}")

    predictions_sha256 = composed._sha256(predictions_path)
    if expected_predictions_sha256 and predictions_sha256 != expected_predictions_sha256:
        raise ValueError("Frozen predictions no longer match the preregistered prediction hash")
    requests = composed._read_jsonl(requests_path)
    predictions = composed._read_jsonl(predictions_path)
    target_count = int(freeze["target_count"])
    if len(requests) != target_count or len(predictions) != target_count:
        raise ValueError("Frozen request/prediction count does not match the freeze manifest")
    request_keys = {composed._identity_key(row["identity"]) for row in requests}
    prediction_keys = {composed._identity_key(row["identity"]) for row in predictions}
    if request_keys != prediction_keys:
        raise ValueError("Frozen request and prediction identities differ")

    sealed_path = root / "sealed_manifest.json"
    if sealed_path.exists():
        sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
        if str(sealed.get("prediction_freeze_sha256")) != freeze_sha256:
            raise ValueError("Sealed manifest does not reference the current prediction freeze")

    return {
        "freeze_path": freeze_path,
        "freeze_sha256": freeze_sha256,
        "requests_sha256": requests_sha256,
        "predictions_sha256": predictions_sha256,
        "requests": requests,
        "predictions": predictions,
    }


def _validate_musicxml_truth(
    truth: MusicXMLTruth,
    *,
    mapping: Mapping[int, Sequence[int]],
    expected_time_signature: str,
    expected_key_fifths: int,
    expected_clef: tuple[str, int],
) -> Fraction:
    measure_length = measure_length_beats(expected_time_signature)
    if truth.payload.get("time_signature") != expected_time_signature:
        raise ValueError(
            f"Unexpected MusicXML time signature: {truth.payload.get('time_signature')}"
        )
    if truth.key_fifths != expected_key_fifths:
        raise ValueError(f"Unexpected MusicXML key fifths: {truth.key_fifths}")
    if truth.clef != expected_clef:
        raise ValueError(f"Unexpected MusicXML clef: {truth.clef}")

    mapped = tuple(measure for crop in sorted(mapping) for measure in mapping[crop])
    if mapped != truth.measure_numbers:
        raise ValueError(
            f"Physical-measure mapping does not exactly cover MusicXML measures: "
            f"mapping={mapped}, musicxml={truth.measure_numbers}"
        )
    invalid_extents = {
        measure: float(extent)
        for measure, extent in truth.measure_extents.items()
        if extent != measure_length
    }
    if invalid_extents:
        raise ValueError(
            f"MusicXML measures are not complete {expected_time_signature}: {invalid_extents}"
        )
    return measure_length


def _segmentation_summary(
    mapping: Mapping[int, Sequence[int]], measure_numbers: Sequence[int]
) -> dict[str, Any]:
    merged = [
        {
            "automatic_crop": crop,
            "physical_measures": list(mapping[crop]),
            "diagnosis": "missed interior barline",
        }
        for crop in sorted(mapping)
        if len(mapping[crop]) > 1
    ]
    return {
        "mapping_defined_after_independent_transcription_review": True,
        "mapping_used_at_inference": False,
        "automatic_crop_count": len(mapping),
        "physical_measure_count": len(measure_numbers),
        "count_match": len(mapping) == len(measure_numbers),
        "missed_barline_count": len(measure_numbers) - len(mapping),
        "automatic_crop_to_physical_measures": {
            str(crop): list(mapping[crop]) for crop in sorted(mapping)
        },
        "merged_automatic_crops": merged,
    }


def _integration_decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    has_exact = int(summary["exact_measures"]) > 0
    has_rest_evidence = float(summary["rest_f1"]) > 0
    return {
        "status": "evidence_present" if has_exact and has_rest_evidence else "not_ready",
        "numeric_threshold_preregistered": False,
        "existing_qualitative_rule": (
            "Require independent exact-event and rest evidence before pipeline integration."
        ),
        "exact_event_evidence": has_exact,
        "rest_evidence": has_rest_evidence,
    }


def _load_existing_evaluation(
    path: Path, *, source_sha256: str, freeze_sha256: str
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("freeze_sha256") != freeze_sha256:
        raise ValueError("Evaluation freeze hash differs; refusing to reopen heldout truth")
    if report.get("musicxml_sha256") != source_sha256:
        raise ValueError("MusicXML changed after the one-shot evaluation")
    return report


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    summary = report["metrics"]["summary"]
    segmentation = report["segmentation"]
    decision = report["pipeline_integration_decision"]
    lines = [
        "# Carrizal System 4 One-Shot Evaluation",
        "",
        "Predictions were frozen before the independent MusicXML transcription was opened.",
        "The seven automatic crops are scored against eight physical measures through the",
        "recorded post-freeze segmentation mapping; no prediction was regenerated.",
        "",
        "## Segmentation",
        "",
        f"- Automatic crops: `{segmentation['automatic_crop_count']}`",
        f"- Physical measures: `{segmentation['physical_measure_count']}`",
        f"- Missed barlines: `{segmentation['missed_barline_count']}`",
        "- Automatic crop 2 maps to physical measures 2 and 3.",
        "",
        "## Metrics",
        "",
        f"- Strict note F1: `{float(summary['note_f1']):.6f}`",
        f"- Ordered pitch accuracy: `{float(summary['ordered_pitch_accuracy']):.6f}`",
        f"- Ordered onset accuracy: `{float(summary['ordered_onset_accuracy']):.6f}`",
        f"- Ordered duration accuracy: `{float(summary['ordered_duration_accuracy']):.6f}`",
        f"- Rest F1: `{float(summary['rest_f1']):.6f}`",
        f"- Exact automatic crops: `{summary['exact_measures']}/{summary['targets']}`",
        "",
        "## Decision",
        "",
        f"Pipeline integration status: **{decision['status']}**.",
        "",
        "The qualitative gate requires independent exact-event and rest evidence; no numeric",
        "threshold was preregistered for this second-score run.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _strip_namespaces(root: ET.Element) -> None:
    for element in root.iter():
        if isinstance(element.tag, str) and "}" in element.tag:
            element.tag = element.tag.split("}", 1)[1]


def _measure_number(measure: ET.Element, fallback: int) -> int:
    try:
        return int(measure.get("number", str(fallback)))
    except ValueError:
        return fallback


def _fraction_number(value: Fraction) -> int | float:
    return value.numerator if value.denominator == 1 else float(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
