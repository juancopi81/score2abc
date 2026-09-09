from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from scripts.experiments.spike_consumed_visual_key_pitch_replay import (
    LANE_BASELINE,
    LANE_VISUAL,
    main,
    run_replay,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    image_dir = tmp_path / "crops"
    image_dir.mkdir(parents=True)
    image_paths = []
    for index in range(2):
        path = image_dir / f"measure_{index + 1:03d}.png"
        Image.new("L", (8, 8), 255).save(path)
        image_paths.append(path)

    detector_measures = []
    for index, path in enumerate(image_paths):
        detector_measures.append(
            {
                "sequence_index": index,
                "fifths": 1 if index else None,
                "fifths_status": "confirmed_explicit" if index else "unknown",
                "input": {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                },
                "state": {
                    "kind": "explicit_change" if index else "unknown_initial",
                    "fifths": 1 if index else None,
                },
                "truth_used_for_prediction": False,
            }
        )
    detector_path = tmp_path / "detector.json"
    _write_json(
        detector_path,
        {
            "schema_version": 1,
            "truth_used_for_prediction": False,
            "measures": detector_measures,
        },
    )

    rows = []
    for index in range(2):
        y = 132.5
        candidate = {
            "candidate_id": f"c{index + 1}",
            "center": {"x": 20.0, "y": y},
            "score": 0.9,
            "detector_rank": 1,
        }
        rows.append(
            {
                "identity": {"automatic_measure_index": index + 1, "slug": "demo-work"},
                "staff_geometry": {"raw_staff_lines_y_px": [115, 120, 125, 130, 135]},
                "candidate_predictions": [candidate],
                "truth_used": False,
                "canonical_prediction": {
                    "notes": [
                        {
                            "candidate_id": candidate["candidate_id"],
                            "pitch_midi": 65,
                        }
                    ]
                },
            }
        )
    inference_path = tmp_path / "inference.jsonl"
    inference_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    model_path = tmp_path / "model.json"
    _write_json(
        model_path,
        {
            "replay": {
                "selector": {
                    "threshold": 0.5,
                    "nms_x_spaces": 0.85,
                    "minimum_selected_count": 1,
                    "maximum_selected_count": 2,
                }
            }
        },
    )
    context_path = tmp_path / "context_hints.json"
    _write_json(
        context_path,
        {"events": [{"start_measure": 2, "key_hint": "one sharp: F#"}]},
    )
    truth_path = tmp_path / "truth.jsonl"
    truth_path.write_text(
        "".join(
            json.dumps(
                {
                    "automatic_crop_index": index + 1,
                    "notes": [
                        {
                            "pitch": "F#4" if index else "F4",
                            "pitch_midi": 66 if index else 65,
                            "onset_divisions": 0,
                            "xml_order": 0,
                            "physical_note_index": 0,
                        }
                    ],
                },
                sort_keys=True,
            )
            + "\n"
            for index in range(2)
        ),
        encoding="utf-8",
    )
    return {
        "detector": detector_path,
        "context": context_path,
        "inference": inference_path,
        "model": model_path,
        "truth": truth_path,
    }


def _replace_with_expanded_detector(paths: dict[str, Path]) -> None:
    image_paths = sorted(paths["detector"].parent.joinpath("crops").glob("measure_*.png"))
    predictions = [
        {
            "input": {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
            "truth_used_for_prediction": False,
        }
        for path in image_paths
    ]
    _write_json(
        paths["detector"],
        {
            "schema_version": 1,
            "truth_used_for_prediction": False,
            "predictions": predictions,
            "context_hints": {
                "truth_used": False,
                "events": [
                    {
                        "start_measure": 1,
                        "key_hint": {"fifths": -1},
                        "source": {"slug": "different-work"},
                    },
                    {
                        "start_measure": 2,
                        "key_hint": {"fifths": 1},
                        "source": {"slug": "demo-work"},
                    },
                ],
            },
        },
    )


def _run(tmp_path: Path, truth_path: Path, output_name: str) -> dict:
    paths = _fixture(tmp_path / output_name)
    paths["truth"] = truth_path
    return run_replay(
        paths["detector"],
        paths["context"],
        paths["inference"],
        paths["model"],
        paths["truth"],
        tmp_path / output_name / "result",
    )


def test_visual_key_changes_pitch_without_changing_selection(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    report = run_replay(
        paths["detector"],
        paths["context"],
        paths["inference"],
        paths["model"],
        paths["truth"],
        tmp_path / "result",
    )

    assert report["detector"]["truth_used_for_prediction"] is False
    assert report["detector"]["state_summary"]["final_fifths"] == 1
    assert report["access_audit"]["detector_context_used_for_prediction"] is True
    assert report["access_audit"]["evaluation_label_context_used_for_prediction"] is False
    assert report["access_audit"]["source_image_hashes_verified"] is True
    assert report["candidate_selection"]["identical_selection"] is True
    assert report["consumed_evidence_gate"]["passed"] is True
    assert report["comparison"]["baseline_metrics"]["exact_pitch_matches"] == 1
    assert report["comparison"]["visual_metrics"]["exact_pitch_matches"] == 2
    assert report["comparison"]["visual_metrics"]["exact_group_count"] == 2
    assert report["prediction_artifacts"]["baseline"]["sha256"]
    assert report["prediction_artifacts"]["visual_key"]["sha256"]

    baseline_rows = [
        json.loads(line)
        for line in Path(report["prediction_artifacts"]["baseline"]["path"])
        .read_text()
        .splitlines()
    ]
    visual_rows = [
        json.loads(line)
        for line in Path(report["prediction_artifacts"]["visual_key"]["path"])
        .read_text()
        .splitlines()
    ]
    assert baseline_rows[1]["candidate_ids"] == visual_rows[1]["candidate_ids"]
    assert baseline_rows[1]["notes"][0]["pitch"] == "F4"
    assert visual_rows[1]["notes"][0]["pitch"] == "F#4"
    assert baseline_rows[0]["key_fifths"] is None
    assert visual_rows[1]["key_fifths"] == 1


def test_expanded_detector_events_are_scoped_to_inference_slug(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _replace_with_expanded_detector(paths)

    report = run_replay(
        paths["detector"],
        paths["context"],
        paths["inference"],
        paths["model"],
        paths["truth"],
        tmp_path / "result",
    )

    assert report["detector"]["target_slug"] == "demo-work"
    assert report["detector"]["state_summary"]["source_format"] == ("expanded_signature_events")
    assert report["detector"]["state_summary"]["explicit_change_count"] == 1
    assert report["detector"]["visual_context_used_for_prediction"] == {2: {"key_fifths": 1}}
    assert report["comparison"]["visual_metrics"]["exact_pitch_matches"] == 2


def test_truth_changes_cannot_change_persisted_predictions(tmp_path: Path) -> None:
    first_paths = _fixture(tmp_path / "first_inputs")
    second_paths = _fixture(tmp_path / "second_inputs")
    altered_truth = tmp_path / "altered_truth.jsonl"
    altered_truth.write_text(
        first_paths["truth"].read_text().replace('"F4"', '"G4"').replace("65", "67"),
        encoding="utf-8",
    )
    first = run_replay(
        first_paths["detector"],
        first_paths["context"],
        first_paths["inference"],
        first_paths["model"],
        first_paths["truth"],
        tmp_path / "result_first",
    )
    second = run_replay(
        second_paths["detector"],
        second_paths["context"],
        second_paths["inference"],
        second_paths["model"],
        altered_truth,
        tmp_path / "result_second",
    )

    for lane in (LANE_BASELINE, LANE_VISUAL):
        left = first["prediction_artifacts"]["baseline" if lane == LANE_BASELINE else "visual_key"]
        right = second["prediction_artifacts"][
            "baseline" if lane == LANE_BASELINE else "visual_key"
        ]
        assert left["sha256"] == right["sha256"]
        assert Path(left["path"]).read_bytes() == Path(right["path"]).read_bytes()
    assert first["comparison"]["baseline_metrics"] != second["comparison"]["baseline_metrics"]


def test_cli_smoke_writes_report(tmp_path: Path, capsys) -> None:
    paths = _fixture(tmp_path)
    exit_code = main(
        [
            "--detector-report",
            str(paths["detector"]),
            "--context-hints",
            str(paths["context"]),
            "--inference",
            str(paths["inference"]),
            "--model",
            str(paths["model"]),
            "--truth",
            str(paths["truth"]),
            "--output-dir",
            str(tmp_path / "cli-result"),
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "cli-result" / "report.json").is_file()
    assert "report.json" in capsys.readouterr().out
