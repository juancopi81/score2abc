from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from scripts.experiments import materialize_repaired_full_event_sidecar as sidecar


def test_materializes_atomic_meter_valid_sidecar_and_replays_deterministically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    canonical_before = _canonical_bytes(fixture["inference_dir"])

    result = sidecar.materialize_repaired_full_event_sidecar(
        fixture["inference_dir"], model_dir=fixture["model_dir"]
    )

    output_dir = Path(result["output_dir"])
    assert output_dir == fixture["inference_dir"] / sidecar.DEFAULT_OUTPUT_DIRNAME
    assert output_dir.is_dir()
    assert not _temp_dir(output_dir).exists()
    assert result["measure_count"] == 1
    assert result["truth_used"] is False
    assert result["freeze_supported"] is False
    assert _canonical_bytes(fixture["inference_dir"]) == canonical_before

    required = {
        "config.json",
        "predictions.jsonl",
        "inference.jsonl",
        "diagnostics.jsonl",
        "overlays/measure_001.png",
        "contact_sheet.png",
        "manifest.json",
    }
    actual = {
        path.relative_to(output_dir).as_posix() for path in output_dir.rglob("*") if path.is_file()
    }
    assert actual == required

    manifest = _read_json(output_dir / "manifest.json")
    assert manifest["kind"] == sidecar.MANIFEST_KIND
    assert manifest["version"] == sidecar.SIDECAR_VERSION
    assert manifest["status"] == "spike_only_full_events_materialized"
    assert manifest["measure_count"] == 1
    assert manifest["truth_accessed"] is False
    assert manifest["truth_used"] is False
    assert manifest["create_once"] is True
    assert manifest["freeze_supported"] is False
    assert manifest["contract"] == {
        "canonical_bytes_unchanged": True,
        "exact_sparse_candidate_lane_consumed": True,
        "candidate_reranking_applied": False,
        "full_events_materialized": True,
        "all_measures_meter_valid": True,
        "truth_used": False,
        "freeze_supported": False,
        "spike_only": True,
    }
    assert manifest["model_and_training"] == fixture["pins"]
    assert set(manifest["canonical"]) == {
        "main_manifest",
        "requests",
        "predictions",
        "inference",
    }
    assert set(manifest["upstream"]) == {
        "multihead_manifest",
        "multihead_lane",
        "sparse_manifest",
        "sparse_lane",
    }
    assert set(manifest["artifacts"]) == required - {"manifest.json"}
    for record in (
        *manifest["canonical"].values(),
        *manifest["upstream"].values(),
        *manifest["artifacts"].values(),
    ):
        assert len(record["sha256"]) == 64

    prediction = _read_jsonl(output_dir / "predictions.jsonl")[0]
    assert prediction["identity"] == fixture["identity"]
    assert prediction["measure_extent_beats"] == 3.0
    assert prediction["decoder_status"] == "synthetic:3/4"
    assert prediction["rests"] == []
    assert prediction["notes"] == [
        {"duration_beats": 3.0, "onset_beats": 0.0, "pitch_midi": 60},
        {"duration_beats": 3.0, "onset_beats": 0.0, "pitch_midi": 64},
    ]
    assert prediction["rhythm_tokens"] == [
        {
            "duration_beats": 3.0,
            "kind": "note",
            "note_count": 2,
            "onset_beats": 0.0,
        }
    ]

    detail = _read_jsonl(output_dir / "inference.jsonl")[0]
    diagnostic = _read_jsonl(output_dir / "diagnostics.jsonl")[0]
    assert detail["meter_valid"] is True
    assert detail["decoded_extent_beats"] == 3.0
    assert len(detail["groups"]) == 1
    assert diagnostic["candidate_ids"] == ["d001", "d002"]
    assert diagnostic["onset_groups"] == [
        {
            "candidate_ids": ["d001", "d002"],
            "group_id": "g001",
            "onset_group_index": 1,
        }
    ]
    assert diagnostic["decoder"]["meter_valid"] is True
    with Image.open(output_dir / "overlays/measure_001.png") as overlay:
        assert overlay.size == (120, 80)
    with Image.open(output_dir / "contact_sheet.png") as contact_sheet:
        assert contact_sheet.width == 120
        assert contact_sheet.height > 80

    verified = sidecar.verify_repaired_full_event_sidecar(
        output_dir, model_dir=fixture["model_dir"]
    )
    assert verified == {
        "output_dir": str(output_dir),
        "measure_count": 1,
        "verified": True,
        "truth_used": False,
    }


def test_refuses_to_overwrite_create_once_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    sidecar.materialize_repaired_full_event_sidecar(
        fixture["inference_dir"], model_dir=fixture["model_dir"]
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite create-once sidecar"):
        sidecar.materialize_repaired_full_event_sidecar(
            fixture["inference_dir"], model_dir=fixture["model_dir"]
        )


def test_missing_meter_fails_without_output_or_temporary_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch, expected_measure_beats=None)
    output_dir = fixture["inference_dir"] / sidecar.DEFAULT_OUTPUT_DIRNAME

    with pytest.raises(ValueError, match="Full-event composition failed closed"):
        sidecar.materialize_repaired_full_event_sidecar(
            fixture["inference_dir"], model_dir=fixture["model_dir"]
        )

    assert not output_dir.exists()
    assert not _temp_dir(output_dir).exists()


def test_request_only_meter_fallback_fails_without_publishing_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output_dir = fixture["inference_dir"] / sidecar.DEFAULT_OUTPUT_DIRNAME

    def compose_with_fallback(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = _compose(*args, **kwargs)
        result.update(
            status=sidecar.compositor.STATUS_REQUEST_METER_FALLBACK,
            prediction=None,
            meter_valid=False,
            decoder_status="request_meter_fallback:meter_repaired",
        )
        return result

    monkeypatch.setattr(
        sidecar.compositor, "compose_repaired_candidate_events", compose_with_fallback
    )

    with pytest.raises(ValueError, match="Full-event composition failed closed"):
        sidecar.materialize_repaired_full_event_sidecar(
            fixture["inference_dir"], model_dir=fixture["model_dir"]
        )

    assert not output_dir.exists()
    assert not _temp_dir(output_dir).exists()


def test_rejects_optional_lane_manifest_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    main_path = fixture["inference_dir"] / "manifest.json"
    main = _read_json(main_path)
    main["optional_lanes"]["edge_safe_stem_multihead_recovery"]["sha256"] = "0" * 64
    _write_json(main_path, main)

    with pytest.raises(ValueError, match="Multi-head recovery optional manifest hash drift"):
        sidecar.materialize_repaired_full_event_sidecar(
            fixture["inference_dir"], model_dir=fixture["model_dir"]
        )


def test_verifier_rejects_main_manifest_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output_dir = Path(
        sidecar.materialize_repaired_full_event_sidecar(
            fixture["inference_dir"], model_dir=fixture["model_dir"]
        )["output_dir"]
    )
    main_path = fixture["inference_dir"] / "manifest.json"
    main = _read_json(main_path)
    main["post_materialization_drift"] = True
    _write_json(main_path, main)

    with pytest.raises(ValueError, match="Repaired full-event manifest drift"):
        sidecar.verify_repaired_full_event_sidecar(output_dir, model_dir=fixture["model_dir"])


def test_verifier_rejects_upstream_lane_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output_dir = Path(
        sidecar.materialize_repaired_full_event_sidecar(
            fixture["inference_dir"], model_dir=fixture["model_dir"]
        )["output_dir"]
    )
    recovery_lane = fixture["multihead_dir"] / "recovery_lane.jsonl"
    recovery_lane.write_bytes(recovery_lane.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="Repaired full-event manifest drift"):
        sidecar.verify_repaired_full_event_sidecar(output_dir, model_dir=fixture["model_dir"])


def test_rejects_supplied_model_directory_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    wrong_model_dir = tmp_path / "wrong_model"
    wrong_model_dir.mkdir()

    with pytest.raises(ValueError, match="Supplied model directory does not match"):
        sidecar.materialize_repaired_full_event_sidecar(
            fixture["inference_dir"], model_dir=wrong_model_dir
        )


def test_rejects_request_inference_sparse_row_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    repair_path = fixture["sparse_dir"] / "repair_lane.jsonl"
    repair_row = _read_jsonl(repair_path)[0]
    repair_row["identity"] = {**repair_row["identity"], "automatic_measure_index": 2}
    _write_jsonl(repair_path, [repair_row])

    with pytest.raises(ValueError, match="Request/inference/sparse lane identity mismatch"):
        sidecar.materialize_repaired_full_event_sidecar(
            fixture["inference_dir"], model_dir=fixture["model_dir"]
        )


def test_verifier_rejects_output_artifact_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output_dir = Path(
        sidecar.materialize_repaired_full_event_sidecar(
            fixture["inference_dir"], model_dir=fixture["model_dir"]
        )["output_dir"]
    )
    predictions_path = output_dir / "predictions.jsonl"
    predictions_path.write_bytes(predictions_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="Repaired full-event manifest drift"):
        sidecar.verify_repaired_full_event_sidecar(output_dir, model_dir=fixture["model_dir"])


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    expected_measure_beats: float | None = 3.0,
) -> dict[str, Any]:
    inference_dir = tmp_path / "inference_v1"
    multihead_dir = inference_dir / sidecar.runner.MULTIHEAD_RECOVERY_DIRNAME
    sparse_dir = inference_dir / sidecar.runner.SPARSE_DYAD_REPAIR_DIRNAME
    model_dir = tmp_path / "model"
    image_dir = tmp_path / "images"
    for path in (inference_dir, multihead_dir, sparse_dir, model_dir, image_dir):
        path.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / "measure.png"
    Image.new("RGB", (120, 80), "white").save(image_path)
    image_sha256 = _sha256(image_path)
    identity = {
        "slug": "synthetic-score",
        "system_index": 1,
        "system_measure_index": 1,
        "automatic_measure_index": 1,
    }
    context = {
        "allow_pickup": False,
        "clef": "treble",
        "expected_measure_beats": expected_measure_beats,
        "key_hint": "one flat: Bb",
        "time_signature": "3/4" if expected_measure_beats is not None else None,
    }
    staff_geometry = {"raw_staff_lines_y_px": [20, 30, 40, 50, 60]}
    request = {
        "identity": identity,
        "truth_accessed": False,
        "images": {
            "raw": {
                "path_relative_to_out": "images/measure.png",
                "sha256": image_sha256,
            }
        },
        "allowed_context": context,
        "allowed_context_provenance": {
            "key_hint": "synthetic visual key",
            "time_signature": "metadata",
        },
        "staff_geometry": staff_geometry,
    }
    source = {"image": str(image_path), "sha256": image_sha256}
    inference_row = {
        "identity": identity,
        "truth_used": False,
        "source": source,
        "allowed_context": context,
        "allowed_context_provenance": request["allowed_context_provenance"],
        "staff_geometry": staff_geometry,
    }
    canonical_prediction = {
        "identity": identity,
        "notes": [],
        "rests": [],
        "rhythm_tokens": [],
        "measure_extent_beats": None,
        "decoder_status": "canonical-unchanged",
    }
    candidate_lane = [
        {
            "candidate_id": "d001",
            "center": {"x": 35.0, "y": 36.0},
            "bbox": {"left": 31, "top": 33, "right": 39, "bottom": 39},
            "onset_group_index": 1,
            "score": 0.91,
        },
        {
            "candidate_id": "d002",
            "center": {"x": 36.0, "y": 46.0},
            "bbox": {"left": 32, "top": 43, "right": 40, "bottom": 49},
            "onset_group_index": 1,
            "score": 0.89,
        },
    ]
    repair_row = {
        "identity": identity,
        "truth_used": False,
        "source": source,
        "lanes": {"sparse_dyad_repair": {"candidate_lane": candidate_lane}},
    }

    requests_path = inference_dir / "requests.jsonl"
    predictions_path = inference_dir / "predictions.jsonl"
    inference_path = inference_dir / "inference.jsonl"
    _write_jsonl(requests_path, [request])
    _write_jsonl(predictions_path, [canonical_prediction])
    _write_jsonl(inference_path, [inference_row])
    _write_jsonl(multihead_dir / "recovery_lane.jsonl", [{"identity": identity}])
    _write_jsonl(sparse_dir / "repair_lane.jsonl", [repair_row])

    implementation_path = model_dir / "selector.py"
    implementation_path.write_text("METHOD = 'synthetic-selector'\n", encoding="utf-8")
    implementation_record = _record(implementation_path)
    model_path = model_dir / "model.json"
    _write_json(
        model_path,
        {"replay": {"method": {"method_id": "synthetic-repaired-selector-v1"}}},
    )
    artifact_records = {"model.json": _record(model_path)}
    model_manifest_path = model_dir / "manifest.json"
    _write_json(
        model_manifest_path,
        {"implementation": implementation_record, "artifacts": artifact_records},
    )
    pins = {
        "model_manifest": _record(model_manifest_path),
        "implementation": implementation_record,
        "artifacts": artifact_records,
    }

    _write_json(multihead_dir / "manifest.json", {"model_and_training": pins})
    _write_json(sparse_dir / "manifest.json", {"model_and_training": pins})
    main_manifest = {
        "status": "inferred_spike_only_no_freeze",
        "create_once": True,
        "truth_accessed": False,
        "truth_used": False,
        "target": {"slug": identity["slug"], "system_index": identity["system_index"]},
        "output_count": 1,
        "model_and_training": pins,
        "optional_lanes": {
            "edge_safe_stem_multihead_recovery": {
                "path": f"{sidecar.runner.MULTIHEAD_RECOVERY_DIRNAME}/manifest.json",
                "sha256": _sha256(multihead_dir / "manifest.json"),
            },
            "sparse_stem_dyad_repair": {
                "path": f"{sidecar.runner.SPARSE_DYAD_REPAIR_DIRNAME}/manifest.json",
                "sha256": _sha256(sparse_dir / "manifest.json"),
            },
        },
        "artifacts": {
            "requests.jsonl": {"path": "requests.jsonl", "sha256": _sha256(requests_path)},
            "predictions.jsonl": {
                "path": "predictions.jsonl",
                "sha256": _sha256(predictions_path),
            },
            "inference.jsonl": {
                "path": "inference.jsonl",
                "sha256": _sha256(inference_path),
            },
        },
    }
    _write_json(inference_dir / "manifest.json", main_manifest)

    monkeypatch.setattr(sidecar.runner, "_verify_multihead_recovery_sidecar", lambda path: None)
    monkeypatch.setattr(sidecar.runner, "_verify_sparse_dyad_repair_sidecar", lambda path: None)
    monkeypatch.setattr(
        sidecar.runner,
        "reconstruct_model",
        lambda payload: (object(), object(), object()),
    )
    monkeypatch.setattr(sidecar.runner.freezer, "_find_out_dir", lambda path: tmp_path)
    monkeypatch.setattr(sidecar.compositor, "compose_repaired_candidate_events", _compose)

    return {
        "inference_dir": inference_dir,
        "multihead_dir": multihead_dir,
        "sparse_dir": sparse_dir,
        "model_dir": model_dir,
        "pins": pins,
        "identity": identity,
    }


def _compose(
    request: dict[str, Any],
    inference_row: dict[str, Any],
    lane: list[dict[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    del inference_row, kwargs
    identity = dict(request["identity"])
    expected_beats = request["allowed_context"].get("expected_measure_beats")
    anchors = [
        {
            "source": {
                "candidate_id": row["candidate_id"],
                "onset_group_index": row["onset_group_index"],
            }
        }
        for row in lane
    ]
    common = {
        "identity": identity,
        "groups": [
            {
                "group_id": "g001",
                "onset_group_index": 1,
                "anchors": anchors,
                "pitches": ["C4", "E4"],
            }
        ],
        "automatic_key_context": {"key_hint": "one flat: Bb", "truth_used": False},
        "expected_measure_beats": expected_beats,
        "observed_extent_beats": expected_beats,
        "decoded_extent_beats": expected_beats,
        "truth_used": False,
    }
    if expected_beats is None:
        return {
            **common,
            "status": sidecar.compositor.STATUS_MISSING_METER,
            "prediction": None,
            "decoder_status": "not_applied_missing_expected_measure_beats",
            "meter_valid": None,
        }
    prediction = {
        "identity": identity,
        "notes": [
            {"onset_beats": 0.0, "duration_beats": 3.0, "pitch_midi": 60},
            {"onset_beats": 0.0, "duration_beats": 3.0, "pitch_midi": 64},
        ],
        "rests": [],
        "rhythm_tokens": [
            {
                "kind": "note",
                "onset_beats": 0.0,
                "duration_beats": 3.0,
                "note_count": 2,
            }
        ],
        "measure_extent_beats": 3.0,
        "decoder_status": "synthetic:3/4",
        "inference_provenance": {"truth_used": False},
    }
    return {
        **common,
        "status": sidecar.compositor.STATUS_MATERIALIZED,
        "prediction": prediction,
        "decoder_status": "synthetic:3/4",
        "meter_valid": True,
    }


def _canonical_bytes(inference_dir: Path) -> dict[str, bytes]:
    return {
        name: (inference_dir / name).read_bytes()
        for name in ("requests.jsonl", "predictions.jsonl", "inference.jsonl")
    }


def _temp_dir(output_dir: Path) -> Path:
    return output_dir.with_name(f".{output_dir.name}.tmp")


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
