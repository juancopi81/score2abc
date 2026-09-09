from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from scripts.experiments import triage_frozen_heldout_meter_deficits as triage


def test_missing_and_invalid_freezes_fail_before_loading_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing" / "sealed_manifest.json"
    with pytest.raises(FileNotFoundError, match="Sealed manifest does not exist"):
        triage.triage_frozen_heldout_meter_deficits(missing)

    fixture = _fixture(tmp_path / "invalid")
    opened = False

    def fail_verifier(path: Path) -> dict[str, Any]:
        nonlocal opened
        opened = True
        raise ValueError("invalid frozen gate")

    monkeypatch.setattr(triage.evaluator, "verify_frozen_gate", fail_verifier)
    with pytest.raises(ValueError, match="invalid frozen gate"):
        triage.triage_frozen_heldout_meter_deficits(fixture["sealed"])
    assert opened is True
    assert not (fixture["namespace"] / triage.DEFAULT_OUTPUT_DIRNAME).exists()


def test_flags_are_deterministic_and_canonical_predictions_are_not_mutated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path / "deterministic")
    _patch_replay(monkeypatch, fixture)
    before = fixture["canonical_predictions"].read_bytes()

    first = triage.triage_frozen_heldout_meter_deficits(
        fixture["sealed"],
        output_dirname="pretruth_meter_triage_a",
    )
    second = triage.triage_frozen_heldout_meter_deficits(
        fixture["sealed"],
        output_dirname="pretruth_meter_triage_b",
    )

    assert Path(first["predictions"]).read_bytes() == Path(second["predictions"]).read_bytes()
    assert first["flagged_measure_indices"] == [2]
    assert fixture["canonical_predictions"].read_bytes() == before
    rows = _read_jsonl(Path(first["predictions"]))
    assert [row["review_flag"] for row in rows] == [False, True]
    assert all(row["truth_used"] is False for row in rows)
    assert all(row["canonical_prediction_mutated"] is False for row in rows)
    assert all("notes" not in row and "rests" not in row for row in rows)
    manifest = json.loads(Path(first["manifest"]).read_text(encoding="utf-8"))
    assert manifest["truth_used"] is False
    assert manifest["truth_or_musicxml_opened"] is False
    assert manifest["review_flags_only"] is True
    assert manifest["canonical_predictions_mutated"] is False


def test_create_once_and_stale_temp_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path / "create-once")
    _patch_replay(monkeypatch, fixture)
    triage.triage_frozen_heldout_meter_deficits(fixture["sealed"])

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        triage.triage_frozen_heldout_meter_deficits(fixture["sealed"])

    stale_name = "pretruth_meter_triage_stale"
    (fixture["namespace"] / f".{stale_name}.tmp").mkdir()
    with pytest.raises(FileExistsError, match="stale temporary"):
        triage.triage_frozen_heldout_meter_deficits(
            fixture["sealed"],
            output_dirname=stale_name,
        )


def test_frozen_snapshot_hash_drift_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path / "hash-drift")
    _patch_replay(monkeypatch, fixture)
    fixture["detailed_inference"].write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="detailed_inference snapshot hash drift"):
        triage.triage_frozen_heldout_meter_deficits(fixture["sealed"])
    assert not (fixture["namespace"] / triage.DEFAULT_OUTPUT_DIRNAME).exists()


def test_rejects_truth_looking_output_names_and_paths(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "truth-path")
    with pytest.raises(ValueError, match="Truth/MusicXML-looking output"):
        triage.triage_frozen_heldout_meter_deficits(
            fixture["sealed"],
            output_dirname="musicxml_review",
        )
    with pytest.raises(ValueError, match="Truth/MusicXML-looking path"):
        triage.triage_frozen_heldout_meter_deficits(
            tmp_path / "dataset" / "ground_truth" / "sealed_manifest.json"
        )


def _fixture(root: Path) -> dict[str, Any]:
    out_dir = root / "out"
    slug = "synthetic-heldout"
    namespace = out_dir / slug / "vlm_melody_fifth_score_heldout" / "v1" / "system_002"
    frozen_dir = namespace / "frozen"
    artifacts = frozen_dir / "artifacts"
    model_path = artifacts / "model" / "001_model.json"
    requests_path = artifacts / "training" / "001_requests.jsonl"
    inference_path = artifacts / "training" / "002_inference.jsonl"
    metadata_path = artifacts / "training" / "003_metadata.json"
    predictions_path = artifacts / "predictions" / "001_predictions.jsonl"
    for path in (model_path, requests_path, inference_path, metadata_path, predictions_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    requests = []
    inference_rows = []
    predictions = []
    for measure in (1, 2):
        image_path = namespace / "crops" / f"measure_{measure:03d}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (60, 40), "white").save(image_path)
        image_sha = triage.freezer._sha256(image_path)
        identity = {
            "slug": slug,
            "system_index": 2,
            "system_measure_index": measure,
            "automatic_measure_index": measure,
        }
        prediction = {
            "identity": identity,
            "notes": [{"pitch_midi": 60 + measure}],
            "rests": [],
            "inference_provenance": {"truth_used": False},
        }
        request = {
            "identity": identity,
            "truth_used": False,
            "allowed_context": {
                "expected_measure_beats": 3.0,
                "allow_pickup": False,
            },
            "images": {
                "raw": {
                    "path_relative_to_out": image_path.relative_to(out_dir).as_posix(),
                    "sha256": image_sha,
                }
            },
        }
        detail = {
            "identity": identity,
            "truth_used": False,
            "allowed_context": request["allowed_context"],
            "canonical_prediction": prediction,
            "source": {"image": str(image_path), "sha256": image_sha},
        }
        requests.append(request)
        inference_rows.append(detail)
        predictions.append(prediction)

    model_path.write_text("{}\n", encoding="utf-8")
    _write_jsonl(requests_path, requests)
    _write_jsonl(inference_path, inference_rows)
    metadata_path.write_text('{"rhythm":"Pasillo"}\n', encoding="utf-8")
    _write_jsonl(predictions_path, predictions)

    def record(path: Path) -> dict[str, str]:
        return {
            "snapshot_path_relative_to_namespace": path.relative_to(namespace).as_posix(),
            "snapshot_sha256": triage.freezer._sha256(path),
        }

    freeze = {
        "inference_binding": {
            "selected_model": {"artifacts": {"model.json": record(model_path)}},
            "inference": {
                "requests": record(requests_path),
                "detailed_inference": record(inference_path),
                "metadata": record(metadata_path),
                "predictions": record(predictions_path),
            },
        }
    }
    freeze_path = frozen_dir / "freeze.json"
    prepared_path = namespace / "prepared_manifest.json"
    sealed_path = frozen_dir / "sealed_manifest.json"
    prepared_path.write_text("{}\n", encoding="utf-8")
    freeze_path.write_text(json.dumps(freeze) + "\n", encoding="utf-8")
    sealed_path.write_text("{}\n", encoding="utf-8")
    requests_by_crop = {measure: row for measure, row in enumerate(requests, start=1)}
    predictions_by_crop = {measure: row for measure, row in enumerate(predictions, start=1)}
    verified = {
        "namespace_root": namespace,
        "sealed_sha256": triage.freezer._sha256(sealed_path),
        "freeze_path": freeze_path,
        "freeze_sha256": triage.freezer._sha256(freeze_path),
        "freeze": freeze,
        "prepared_path": prepared_path,
        "prepared_sha256": triage.freezer._sha256(prepared_path),
        "target": {"slug": slug, "system_index": 2},
        "requests_by_crop": requests_by_crop,
        "predictions_by_crop": predictions_by_crop,
    }
    return {
        "namespace": namespace,
        "sealed": sealed_path,
        "verified": verified,
        "canonical_predictions": predictions_path,
        "detailed_inference": inference_path,
    }


def _patch_replay(
    monkeypatch: pytest.MonkeyPatch,
    fixture: dict[str, Any],
) -> None:
    monkeypatch.setattr(
        triage.evaluator,
        "verify_frozen_gate",
        lambda path: fixture["verified"],
    )
    monkeypatch.setattr(
        triage.heldout,
        "reconstruct_model",
        lambda payload: (object(), object(), {"truth_used": False}),
    )
    monkeypatch.setattr(
        triage.recovery,
        "selector_config_from_model",
        lambda payload: {},
    )
    replays = [
        SimpleNamespace(
            measure_index=measure,
            key=f"synthetic:S02M{measure:02d}",
            source_image=(fixture["namespace"] / "crops" / f"measure_{measure:03d}.png"),
        )
        for measure in (1, 2)
    ]
    monkeypatch.setattr(
        triage.onset,
        "_holdout_replays",
        lambda *args, **kwargs: replays,
    )

    def observe(replay: SimpleNamespace, **kwargs: Any) -> dict[str, Any]:
        flagged = replay.measure_index == 2
        return {
            "key": replay.key,
            "identity": {
                "slug": "synthetic-heldout",
                "system_index": 2,
                "system_measure_index": replay.measure_index,
            },
            "source_image": str(replay.source_image),
            "source_sha256": triage.freezer._sha256(replay.source_image),
            "visual_total_beats": 2.0 if flagged else 3.0,
            "expected_measure_beats": 3.0,
            "deficit_beats": 1.0 if flagged else 0.0,
            "review_flag": flagged,
            "status": ("review_visual_meter_deficit" if flagged else "meter_not_underfilled"),
            "symbols": [],
            "truth_used": False,
        }

    monkeypatch.setattr(triage.meter, "observe_replay", observe)

    def overlay(observation: dict[str, Any], replay: SimpleNamespace, path: Path) -> None:
        Image.new("RGB", (60, 40), "white").save(path)

    monkeypatch.setattr(triage.meter, "_write_overlay", overlay)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
