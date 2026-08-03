import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw

from scripts.experiments import evaluate_second_score_heldout as truth_tools
from scripts.experiments import spike_consumed_coqueteos_corrected_replay as spike


def test_publishes_predictions_before_truth_and_scores_seven_measures(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    temp_dir = fixture["output"].with_name(f".{fixture['output'].name}.tmp")
    truth_opened = False

    def load_truth(path: Path) -> truth_tools.MusicXMLTruth:
        nonlocal truth_opened
        truth_opened = True
        assert (temp_dir / "pretruth_manifest.json").is_file()
        assert (temp_dir / "predictions.jsonl").is_file()
        return _truth()

    result = spike.run_consumed_replay(
        fixture["out_dir"],
        output_dir=fixture["output"],
        model_dir=fixture["model_dir"],
        truth_loader=load_truth,
        inference_runner=_fake_inference,
        triage_runner=_fake_triage,
    )

    assert truth_opened is True
    report = _json(Path(result["report"]))
    assert report["evidence_scope"]["independent_heldout"] is False
    assert report["evidence_scope"]["truth_used_for_prediction"] is False
    assert report["corrected"]["summary"]["note_f1"] == 1.0
    assert report["corrected"]["summary"]["exact_measures"] == 7
    assert report["corrected"]["summary"]["meter_valid_crops"] == 7
    assert report["review_triage"]["corrected"]["flagged_measure_indices"] == list(range(1, 8))
    assert report["review_triage"]["corrected"]["metrics"]["fp"] == 7
    assert len(report["coordinate_review_priorities"]) == 7

    pretruth = _json(Path(result["pretruth_manifest"]))
    assert pretruth["truth_accessed_for_prediction"] is False
    assert pretruth["truth_used_for_prediction"] is False
    assert pretruth["measure_count"] == 7
    assert Path(result["contact_sheet"]).is_file()
    consumption = _json(Path(result["consumption_mapping"]))
    assert consumption["split_status"] == "consumed_training"
    assert [row["physical_measure_numbers"] for row in consumption["crops"]] == [
        [index] for index in range(1, 8)
    ]


def test_create_once_refuses_before_truth_is_reopened(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    spike.run_consumed_replay(
        fixture["out_dir"],
        output_dir=fixture["output"],
        model_dir=fixture["model_dir"],
        truth_loader=lambda path: _truth(),
        inference_runner=_fake_inference,
        triage_runner=_fake_triage,
    )
    reopened = False

    def reject_reopen(path: Path) -> truth_tools.MusicXMLTruth:
        nonlocal reopened
        reopened = True
        raise AssertionError("truth must not be reopened")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        spike.run_consumed_replay(
            fixture["out_dir"],
            output_dir=fixture["output"],
            model_dir=fixture["model_dir"],
            truth_loader=reject_reopen,
            inference_runner=_fake_inference,
            triage_runner=_fake_triage,
        )

    assert reopened is False


def test_rejects_model_other_than_frozen_coqueteos_model(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    (fixture["model_dir"] / "model.json").write_text('{"changed": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="differs from the model frozen"):
        spike.run_consumed_replay(
            fixture["out_dir"],
            output_dir=fixture["output"],
            model_dir=fixture["model_dir"],
            truth_loader=lambda path: _truth(),
            inference_runner=_fake_inference,
            triage_runner=_fake_triage,
        )


def _fixture(tmp_path: Path) -> dict[str, Path]:
    out_dir = tmp_path / "out"
    work_dir = out_dir / spike.SLUG
    namespace = work_dir / "vlm_melody_training_inputs" / spike.DEFAULT_NAMESPACE
    measure_dir = namespace / "measure_inputs/system_002"
    measure_dir.mkdir(parents=True)
    rows = []
    for index in range(1, spike.MEASURE_COUNT + 1):
        image_path = measure_dir / f"measure_{index:03d}_raw.png"
        _staff_image(image_path)
        rows.append(
            {
                "slug": spike.SLUG,
                "system_index": spike.SYSTEM_INDEX,
                "system_measure_index": index,
                "global_measure_index": index,
                "x_bounds_px": {"left": (index - 1) * 100, "right": index * 100},
                "paths": {"measure_raw": str(image_path.resolve())},
            }
        )
    _write_jsonl(namespace / "inputs_manifest.jsonl", rows)
    _write_json(
        namespace / "manifest.json",
        {
            "kind": "vlm_melody_consumed_cross_score_training_inputs",
            "identity": {"slug": spike.SLUG, "system_index": spike.SYSTEM_INDEX},
            "measure_count": spike.MEASURE_COUNT,
            "eligible_for_training": False,
        },
    )

    heldout = work_dir / "vlm_melody_fifth_score_heldout/v1/system_002"
    context_path = heldout / "context/allowed_context.json"
    _write_json(
        context_path,
        {
            "truth_accessed": False,
            "truth_used": False,
            "allowed_context": {
                "clef": "treble",
                "time_signature": "3/4",
                "key_hint": None,
                "expected_measure_beats": 3,
                "allow_pickup": False,
            },
            "provenance": {"source": "synthetic truth-blind test context"},
        },
    )
    musicxml = heldout / "coqueteos_system_002.musicxml"
    musicxml.parent.mkdir(parents=True, exist_ok=True)
    musicxml.write_text("<score-partwise/>\n", encoding="utf-8")
    _write_json(
        heldout / "evaluation_v1/report.json",
        {
            "metrics": {
                "summary": _summary(targets=6, exact=0),
                "results": [_metric_result(index, exact=False) for index in range(1, 7)],
            }
        },
    )
    _write_json(
        heldout / "pretruth_meter_triage_v1/report.json",
        {"flagged_automatic_measure_indices": [1, 2, 3]},
    )
    _write_json(work_dir / "metadata.json", {"rhythm": "Pasillo"})

    model_dir = out_dir / "model"
    _write_json(model_dir / "model.json", {"kind": "synthetic-model"})
    model_hash = spike._sha256(model_dir / "model.json")
    _write_json(
        heldout / "frozen/freeze.json",
        {
            "model_artifacts": [
                {
                    "source_path": str(model_dir / "model.json"),
                    "source_sha256": model_hash,
                }
            ]
        },
    )
    return {
        "out_dir": out_dir,
        "model_dir": model_dir,
        "output": namespace / spike.DEFAULT_OUTPUT_DIRNAME,
    }


def _fake_inference(
    requests: list[dict[str, Any]],
    model: dict[str, Any],
    out_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    predictions = []
    details = []
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir()
    for request in requests:
        identity = dict(request["identity"])
        prediction = {
            "identity": identity,
            "measure_extent_beats": 3,
            "notes": [{"pitch_midi": 60, "onset_beats": 0, "duration_beats": 3}],
            "rests": [],
        }
        predictions.append(prediction)
        details.append({"identity": identity, "truth_used": False})
        _staff_image(overlay_dir / f"measure_{identity['system_measure_index']:03d}.png")
    _staff_image(output_dir / "contact_sheet.png")
    return {
        "predictions": predictions,
        "inference": details,
        "replay": {"kind": "synthetic-replay"},
    }


def _fake_triage(
    requests: list[dict[str, Any]],
    details: list[dict[str, Any]],
    *,
    model_payload: dict[str, Any],
    metadata: dict[str, Any],
    out_root: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    _staff_image(output_dir / "triage_contact_sheet.png")
    return [
        {
            "key": f"measure-{index}",
            "identity": {
                "slug": spike.SLUG,
                "system_index": spike.SYSTEM_INDEX,
                "system_measure_index": index,
            },
            "review_flag": True,
            "truth_used": False,
        }
        for index in range(1, spike.MEASURE_COUNT + 1)
    ]


def _truth() -> truth_tools.MusicXMLTruth:
    notes = [
        {
            "measure": index,
            "onset_beats": 0,
            "duration_beats": 3,
            "pitch_midi": 60,
        }
        for index in range(1, spike.MEASURE_COUNT + 1)
    ]
    return truth_tools.MusicXMLTruth(
        payload={"time_signature": "3/4", "notes": notes},
        measure_numbers=tuple(range(1, spike.MEASURE_COUNT + 1)),
        measure_extents={index: Fraction(3) for index in range(1, spike.MEASURE_COUNT + 1)},
        rests_by_measure={index: () for index in range(1, spike.MEASURE_COUNT + 1)},
        key_fifths=-1,
        clef=("G", 2),
    )


def _summary(*, targets: int, exact: int) -> dict[str, Any]:
    return {
        "targets": targets,
        "predicted": targets,
        "exact_measures": exact,
        "exact_measure_rate": exact / targets,
        "note_f1": 0.0,
        "note_precision": 0.0,
        "note_recall": 0.0,
        "ordered_pitch_accuracy": 0.0,
        "ordered_onset_accuracy": 0.0,
        "ordered_duration_accuracy": 0.0,
        "rest_precision": 0.0,
        "rest_recall": 0.0,
        "rest_f1": 0.0,
        "meter_valid_crop_rate": 0.0,
    }


def _metric_result(index: int, *, exact: bool) -> dict[str, Any]:
    return {
        "identity": {"system_measure_index": index},
        "compared_notes": 1,
        "pred_note_count": 0 if not exact else 1,
        "truth_note_count": 1,
        "onset_matches": 0 if not exact else 1,
    }


def _staff_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (140, 100), "white")
    draw = ImageDraw.Draw(image)
    for y in (30, 40, 50, 60, 70):
        draw.line((5, y, 135, y), fill="black", width=2)
    image.save(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
