from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from scripts.experiments import spike_cross_score_pitch_mapping as spike


def test_materializes_sealed_pitch_lanes_before_scoring_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_dir = _fixture(tmp_path)
    monkeypatch.setattr(
        spike,
        "detect_signature",
        lambda _path, *, mode: {
            "fifths": 1,
            "gate_passed": True,
            "predicted_signature_family": "sharp",
            "selected_glyph_ids": [f"{mode}-sharp"],
        },
    )
    monkeypatch.setattr(
        spike,
        "track_common_staff_shift",
        lambda image, *, base_lines, spacing: [0] * image.width,
    )

    output = spike.run_experiment(
        fixture_dir=fixture_dir,
        output_dir=tmp_path / "result",
    )
    report = _json(output / "report.json")
    seal = _json(output / "prediction_seal.json")

    assert seal["truth_opened"] is False
    assert report["protocol"]["prediction_seal_written_before_truth_open"] is True
    assert report["selection_invariance"]["all_candidate_ids_equal"] is True
    assert report["selection_invariance"]["all_coordinates_equal"] is True
    assert report["selection_invariance"]["all_note_counts_equal"] is True
    assert (
        report["evaluation"]["lanes"][spike.LANE_GLOBAL_FROZEN]["aggregate"]["exact_pitch_matches"]
        == 0
    )
    assert (
        report["evaluation"]["lanes"][spike.LANE_GLOBAL_AUTOMATIC]["aggregate"][
            "exact_pitch_matches"
        ]
        == 2
    )
    assert report["decisions"]["automatic_key_state"] == {
        "aggregate_exact_pitch_delta": 2,
        "aggregate_exact_pitch_delta_vs_frozen_baseline": 2,
        "component": "automatic key-state mapping",
        "consumed_development_gate_passed": True,
        "heldout_status": "not_run",
        "improved_scores": [
            {"score_id": "alpha", "exact_pitch_delta": 1},
            {"score_id": "beta", "exact_pitch_delta": 1},
        ],
        "localization_invariant": True,
        "regressed_scores": [],
        "unchanged_scores": [],
    }
    assert report["decisions"]["runtime_promotion"]["eligible"] is False

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        spike.run_experiment(fixture_dir=fixture_dir, output_dir=output)


def test_rejects_fixture_hash_drift(tmp_path: Path) -> None:
    fixture_dir = _fixture(tmp_path)
    requests_path = fixture_dir / "requests.jsonl"
    requests_path.write_text(
        requests_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        spike.run_experiment(
            fixture_dir=fixture_dir,
            output_dir=tmp_path / "result",
        )


def test_selection_invariance_detects_changed_candidate_identity() -> None:
    predictions = {
        lane: [
            {
                "score_id": "alpha",
                "measure_index": 1,
                "notes": [
                    {
                        "candidate_id": "c001",
                        "x": 10.0,
                        "y": 45.0,
                        "pitch_midi": 65,
                    }
                ],
            }
        ]
        for lane in spike.LANES
    }
    predictions[spike.LANE_GLOBAL_AUTOMATIC][0]["notes"][0]["candidate_id"] = "c002"

    invariance = spike._selection_invariance(predictions)

    assert invariance["all_candidate_ids_equal"] is False
    assert invariance["all_coordinates_equal"] is True
    assert invariance["all_note_counts_equal"] is True


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "fixture"
    images_dir = root / "images"
    images_dir.mkdir(parents=True)
    requests = []
    truth = []
    scores = []
    key_events = []
    for score_id in ("alpha", "beta"):
        image_path = images_dir / f"{score_id}.png"
        image = Image.new("L", (80, 70), 255)
        draw = ImageDraw.Draw(image)
        for y in (10, 20, 30, 40, 50):
            draw.line((0, y, 79, y), fill=0, width=1)
        image.save(image_path)
        image_pin = _pin(image_path, root)
        scores.append({"score_id": score_id})
        key_events.append(
            {
                "score_id": score_id,
                "mode": "initial",
                "start_measure": 1,
                "fallback_fifths": None,
                "image": image_pin,
            }
        )
        requests.append(
            {
                "score_id": score_id,
                "measure_index": 1,
                "segmentation_confounded": False,
                "frozen_fifths": None,
                "staff_lines_y_px": [10, 20, 30, 40, 50],
                "image": image_pin,
                "notes": [
                    {
                        "candidate_id": "c001",
                        "x": 20.0,
                        "y": 45.0,
                        "baseline_pitch": "F4",
                    }
                ],
            }
        )
        truth.append({"score_id": score_id, "measure_index": 1, "pitch_midi": [66]})

    requests_path = root / "requests.jsonl"
    truth_path = root / "truth.jsonl"
    _write_jsonl(requests_path, requests)
    _write_jsonl(truth_path, truth)
    _write_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "kind": spike.FIXTURE_KIND,
            "evidence_tier": "consumed_postmortem",
            "eligible_for_heldout_claim": False,
            "scores": scores,
            "key_events": key_events,
            "requests": _pin(requests_path, root),
            "truth": _pin(truth_path, root),
        },
    )
    return root


def _pin(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
