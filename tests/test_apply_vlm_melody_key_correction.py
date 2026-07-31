from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.experiments import apply_vlm_melody_key_correction as correction


def _write_inference(tmp_path: Path) -> Path:
    rows = []
    for measure, original_pitch in ((1, 77), (2, 76)):
        image = tmp_path / f"measure_{measure:03d}.png"
        Image.new("RGB", (120, 140), "white").save(image)
        candidate_id = f"d{measure:03d}"
        rows.append(
            {
                "schema_version": 1,
                "truth_used": False,
                "identity": {
                    "slug": "demo",
                    "system_index": 1,
                    "system_measure_index": measure,
                    "automatic_measure_index": measure,
                },
                "source": {
                    "image": str(image),
                    "sha256": correction._sha256(image),
                },
                "staff_geometry": {"raw_staff_lines_y_px": [20, 40, 60, 80, 100]},
                "canonical_prediction": {
                    "measure_extent_beats": 3,
                    "notes": [
                        {
                            "onset_beats": 0,
                            "duration_beats": 1,
                            "pitch_midi": original_pitch,
                        }
                    ],
                    "rests": [{"onset_beats": 1, "duration_beats": 2}],
                    "rhythm_tokens": [
                        {
                            "kind": "note",
                            "onset_beats": 0,
                            "duration_beats": 1,
                            "note_count": 1,
                        },
                        {"kind": "rest", "onset_beats": 1, "duration_beats": 2},
                    ],
                },
                "automatic_anchors": [
                    {
                        "order": 1,
                        "center": {"x": 40.0, "y": 20.0},
                        "source": {"candidate_id": candidate_id},
                    }
                ],
                "candidate_predictions": [
                    {
                        "candidate_id": candidate_id,
                        "center": {"x": 40.0, "y": 20.0},
                        "selected": True,
                    }
                ],
            }
        )
    path = tmp_path / "inference.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_key_correction_changes_only_pitch_fields(tmp_path: Path) -> None:
    inference = _write_inference(tmp_path)
    output = tmp_path / "corrected"

    report = correction.apply_key_correction(
        inference,
        key_events=[
            correction.KeyEvent(start_measure=1, fifths=-1),
            correction.KeyEvent(start_measure=2, fifths=2),
        ],
        output_dir=output,
    )

    predictions = [
        json.loads(line)
        for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["key_fifths"] for row in predictions] == [-1, 2]
    assert [row["candidate_ids"] for row in predictions] == [["d001"], ["d002"]]
    assert predictions[0]["notes"][0]["pitch"] == "F5"
    assert predictions[1]["notes"][0]["pitch"] == "F#5"
    assert predictions[0]["notes"][0]["onset_beats"] == 0
    assert predictions[0]["rests"] == [{"duration_beats": 2, "onset_beats": 1}]
    assert report["summary"] == {
        "measure_count": 2,
        "note_count": 2,
        "changed_pitch_count": 1,
        "all_candidate_ids_unchanged": True,
        "all_coordinates_unchanged": True,
        "all_note_counts_unchanged": True,
        "all_note_rhythm_unchanged": True,
    }
    assert sorted(path.name for path in (output / "overlays").iterdir()) == [
        "measure_001.png",
        "measure_002.png",
    ]


def test_key_correction_is_create_once(tmp_path: Path) -> None:
    inference = _write_inference(tmp_path)
    output = tmp_path / "corrected"
    correction.apply_key_correction(
        inference,
        key_events=[correction.KeyEvent(start_measure=1, fifths=0)],
        output_dir=output,
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        correction.apply_key_correction(
            inference,
            key_events=[correction.KeyEvent(start_measure=1, fifths=2)],
            output_dir=output,
        )


def test_key_correction_rejects_anchor_candidate_drift(tmp_path: Path) -> None:
    inference = _write_inference(tmp_path)
    rows = [json.loads(line) for line in inference.read_text(encoding="utf-8").splitlines()]
    rows[0]["candidate_predictions"][0]["center"]["y"] = 21.0
    inference.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="candidate center changed"):
        correction.apply_key_correction(
            inference,
            key_events=[correction.KeyEvent(start_measure=1, fifths=0)],
            output_dir=tmp_path / "corrected",
        )
