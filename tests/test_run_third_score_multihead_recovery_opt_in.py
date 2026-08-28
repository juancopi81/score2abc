from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from scripts.experiments import run_third_score_heldout_inference as spike


def test_default_artifacts_stay_unchanged_and_opt_in_writes_additive_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, model_dir = _stub_inference_environment(tmp_path, monkeypatch)

    baseline = spike.materialize_third_score_inference(
        prepared,
        model_dir=model_dir,
        inference_dirname="baseline",
    )
    opt_in = spike.materialize_third_score_inference(
        prepared,
        model_dir=model_dir,
        inference_dirname="with_recovery",
        multihead_recovery=True,
    )

    baseline_dir = Path(baseline["inference_dir"])
    opt_in_dir = Path(opt_in["inference_dir"])
    assert (baseline_dir / "predictions.jsonl").read_bytes() == (
        opt_in_dir / "predictions.jsonl"
    ).read_bytes()
    assert (baseline_dir / "inference.jsonl").read_bytes() == (
        opt_in_dir / "inference.jsonl"
    ).read_bytes()

    expected_artifacts = {
        "requests.jsonl",
        "predictions.jsonl",
        "inference.jsonl",
        "assumptions.json",
        "replay.json",
        "contact_sheet.png",
        "overlays",
    }
    baseline_manifest = spike._read_json(baseline_dir / "manifest.json")
    opt_in_manifest = spike._read_json(opt_in_dir / "manifest.json")
    assert baseline_manifest["status"] == "inferred_awaiting_freeze"
    assert "optional_lanes" not in baseline_manifest
    assert set(baseline_manifest["artifacts"]) == expected_artifacts
    assert set(opt_in_manifest["artifacts"]) == expected_artifacts
    assert opt_in_manifest["status"] == "inferred_spike_only_no_freeze"

    sidecar_dir = opt_in_dir / spike.MULTIHEAD_RECOVERY_DIRNAME
    sidecar_manifest = spike._read_json(sidecar_dir / "manifest.json")
    config = spike._read_json(sidecar_dir / "config.json")
    lanes = spike._read_jsonl(sidecar_dir / "recovery_lane.jsonl")
    diagnostics = spike._read_jsonl(sidecar_dir / "diagnostics.jsonl")
    assert config["config_id"] == spike.recovery.EDGE_SAFE_STEM_MULTIHEAD_CONFIG_ID
    assert config["parameters"] == spike.recovery.EDGE_SAFE_STEM_MULTIHEAD_PARAMETERS
    assert sidecar_manifest["contract"] == {
        "additive_candidate_selection_only": True,
        "baseline_canonical_predictions_unchanged": True,
        "baseline_predictions_unchanged": True,
        "canonical_pitch_and_rhythm_recomposition_applied": False,
        "freeze_supported": False,
        "recovered_candidates_reuse_existing_onset_groups": True,
    }
    assert sidecar_manifest["recovered_head_count"] == 1
    assert lanes[0]["lanes"]["baseline_generic"]["candidate_lane"] == [
        {
            "bbox": {"bottom": 55, "left": 55, "right": 65, "top": 45},
            "candidate_id": "c001",
            "center": {"x": 60.0, "y": 50.0},
            "onset_group_index": 1,
            "recovered": False,
            "score": 0.9,
        }
    ]
    recovery_lane = lanes[0]["lanes"]["edge_safe_stem_multihead_recovery"]
    assert recovery_lane["recovered_candidate_ids"] == ["c002"]
    assert recovery_lane["canonical_prediction_materialized"] is False
    assert [item["candidate_id"] for item in recovery_lane["candidate_lane"]] == [
        "c001",
        "c002",
    ]
    assert diagnostics[0]["baseline_onset_group_count"] == 1
    assert diagnostics[0]["recovered_onset_group_count"] == 1
    assert diagnostics[0]["recovery"][0]["stem_attachment_score"] == pytest.approx(0.9)
    assert (sidecar_dir / "overlays/measure_001.png").is_file()
    assert (sidecar_dir / "contact_sheet.png").is_file()
    spike._verify_multihead_recovery_sidecar(sidecar_dir)

    optional = opt_in_manifest["optional_lanes"]["edge_safe_stem_multihead_recovery"]
    assert optional == {
        "path": f"{spike.MULTIHEAD_RECOVERY_DIRNAME}/manifest.json",
        "sha256": spike._sha256(sidecar_dir / "manifest.json"),
    }
    assert opt_in["multihead_recovery"]["recovery_lane_sha256"] == spike._sha256(
        sidecar_dir / "recovery_lane.jsonl"
    )


def test_sidecar_hash_drift_and_canonical_freeze_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, model_dir = _stub_inference_environment(tmp_path, monkeypatch)
    result = spike.materialize_third_score_inference(
        prepared,
        model_dir=model_dir,
        multihead_recovery=True,
    )
    inference_dir = Path(result["inference_dir"])

    with pytest.raises(ValueError, match="spike-only optional recovery artifacts"):
        spike.freeze_inference(
            prepared,
            inference_dir=inference_dir,
            model_dir=model_dir,
        )

    sidecar_dir = inference_dir / spike.MULTIHEAD_RECOVERY_DIRNAME
    config_path = sidecar_dir / "config.json"
    config_path.write_text(config_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar artifact hash drift"):
        spike._verify_multihead_recovery_sidecar(sidecar_dir)


def test_cli_requires_no_freeze_for_multihead_recovery(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = spike.main([str(tmp_path / "missing.json"), "--multihead-recovery"])

    assert result == 1
    assert "requires --no-freeze" in capsys.readouterr().err


def _stub_inference_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    namespace = tmp_path / "namespace"
    namespace.mkdir()
    prepared = namespace / "prepared_manifest.json"
    prepared.write_text("{}\n", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text("{}\n", encoding="utf-8")
    image_path = tmp_path / "measure.png"
    image = Image.new("RGB", (120, 100), "white")
    draw = ImageDraw.Draw(image)
    for y in (30, 40, 50, 60, 70):
        draw.line((0, y, 119, y), fill="black", width=1)
    draw.ellipse((55, 45, 65, 55), fill="black")
    draw.ellipse((57, 65, 67, 75), fill="black")
    image.save(image_path)

    identity = {
        "slug": "synthetic",
        "system_index": 7,
        "automatic_measure_index": 1,
    }
    request = {
        "identity": identity,
        "allowed_context": {
            "allow_pickup": False,
            "clef": "treble",
            "expected_measure_beats": None,
            "key_hint": None,
            "time_signature": None,
        },
        "allowed_context_provenance": {"source": "test"},
        "staff_geometry": {"raw_staff_lines_y_px": [30, 40, 50, 60, 70]},
    }
    candidate_predictions = [
        {
            "candidate_id": "c001",
            "detector_rank": 1,
            "score": 0.9,
            "center": {"x": 60.0, "y": 50.0},
            "bbox": {"left": 55, "top": 45, "right": 65, "bottom": 55},
        },
        {
            "candidate_id": "c002",
            "detector_rank": 2,
            "score": 0.7,
            "center": {"x": 62.0, "y": 70.0},
            "bbox": {"left": 57, "top": 65, "right": 67, "bottom": 75},
        },
    ]
    prediction = {
        "identity": identity,
        "notes": [
            {
                "candidate_id": "c001",
                "center": {"x": 60.0, "y": 50.0},
                "pitch": "B4",
                "pitch_midi": 71,
                "onset_beats": None,
                "duration_beats": None,
            }
        ],
        "rests": [],
        "rhythm_tokens": [],
        "decoder_status": "not_applied_missing_expected_measure_beats",
        "inference_provenance": {"truth_used": False},
    }
    item = spike.composed.ComposedMeasure(
        request=request,
        image_path=image_path,
        image=image,
        staff_spacing=10.0,
        candidate_predictions=candidate_predictions,
        anchors=[],
        groups=[],
        anchor_features=[],
        rest_features=[],
        visual_symbols=[],
        decoded_symbols=[],
        prediction=prediction,
    )
    model_payload = {
        "replay": {
            "method": {"method_id": "synthetic"},
            "selector": {
                "threshold": 0.0,
                "nms_x_spaces": 0.85,
                "minimum_selected_count": 1,
                "maximum_selected_count": 1,
            },
        }
    }
    prepared_payload = {
        "target": {"slug": "synthetic", "system_index": 7},
        "artifacts": {"requests": {"row_sha256": ["request-sha"]}},
    }
    validated = {
        "prepared": prepared_payload,
        "gate_config": {
            "manifest_kind": "synthetic_inference_manifest",
            "inference_version": "synthetic-v1",
        },
        "expected_count": 1,
        "model": model_payload,
        "prepared_requests": [request],
        "metadata": {},
        "metadata_record": spike._file_record(metadata_path),
        "pins": {"model_manifest": {"path": "model", "sha256": "a"}, "artifacts": {}},
    }
    monkeypatch.setattr(spike, "_validate_inputs", lambda *_args, **_kwargs: validated)
    monkeypatch.setattr(spike.freezer, "_find_out_dir", lambda _path: tmp_path)
    monkeypatch.setattr(
        spike,
        "reconstruct_model",
        lambda _payload: (object(), object(), {"truth_used": False}),
    )
    monkeypatch.setattr(
        spike,
        "_context_assumptions",
        lambda *_args, **_kwargs: {
            "warnings": [],
            "allowed_context": request["allowed_context"],
            "provenance": request["allowed_context_provenance"],
        },
    )
    monkeypatch.setattr(spike, "_materialize_request", lambda row, **_kwargs: row)
    monkeypatch.setattr(spike, "_infer_request", lambda *_args, **_kwargs: item)
    monkeypatch.setattr(
        spike.composed,
        "_write_overlay",
        lambda composed_item, output_path: composed_item.image.save(output_path),
    )
    stem_features = {
        candidate_id: {
            "score": score,
            "direction": "up",
            "x": 65,
            "candidate_bbox": candidate["bbox"],
        }
        for candidate_id, score, candidate in (
            ("c001", 0.8, candidate_predictions[0]),
            ("c002", 0.9, candidate_predictions[1]),
        )
    }
    monkeypatch.setattr(
        spike.recovery,
        "candidate_local_stem_features",
        lambda _row: (
            stem_features,
            {
                "source_image": spike._file_record(image_path),
                "staff_spacing_px": 10.0,
            },
        ),
    )
    return prepared, model_dir
