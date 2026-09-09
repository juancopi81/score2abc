import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from scripts.experiments import freeze_independent_multihead_recovery_gate as gate
from scripts.experiments import freeze_third_score_heldout as freezer
from scripts.experiments import run_independent_multihead_recovery_gate as runner
from scripts.experiments import run_third_score_heldout_inference as inference
from scripts.experiments import spike_consumed_polyphonic_pitch_repair as recovery


def test_gate_contract_is_fixed_and_registered() -> None:
    assert gate.A_MEDIO_PALO_SLUG == "jaime-llanos_7_a-medio-palo_pasillo_m-garavito-w"
    assert gate.TARGET_SYSTEM_INDEX == 7
    assert gate.EXPECTED_CROP_COUNT == 7
    assert gate.DEFAULT_LAYOUT_POLICY.min_crop_width_px == 80
    assert gate.DEFAULT_LAYOUT_POLICY.max_spacing_cv == pytest.approx(0.45)
    assert recovery.EDGE_SAFE_STEM_MULTIHEAD_PARAMETERS == {
        "minimum_y_gap_staff_spaces": 1.0,
        "maximum_y_gap_staff_spaces": 3.0,
        "minimum_score_ratio": 0.5,
        "minimum_stem_score": 0.55,
        "minimum_group_x_staff_spaces": 1.0,
        "maximum_recovered_heads_per_group": 2,
    }
    config = inference.GATE_CONFIGS[gate.INDEPENDENT_MULTIHEAD_RECOVERY_GATE.prepare_kind]
    assert config["inference_version"] == "independent-multihead-baseline-inference-v1"


def test_preparation_seals_seven_crops_and_unknown_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score_root = tmp_path / gate.A_MEDIO_PALO_SLUG
    system_path = score_root / "systems" / "system_007.png"
    system_path.parent.mkdir(parents=True)
    image = Image.new("L", (1400, 120), "white")
    draw = ImageDraw.Draw(image)
    for y in (35, 45, 55, 65, 75):
        draw.line((0, y, 1399, y), fill="black", width=2)
    image.save(system_path)
    (score_root / "metadata.json").write_text(
        json.dumps({"title": "A medio palo", "time_signature": None, "key_hint": None}) + "\n",
        encoding="utf-8",
    )
    boundaries = [index / gate.EXPECTED_CROP_COUNT for index in range(8)]
    monkeypatch.setattr(freezer, "detect_barlines", lambda _path: boundaries)
    monkeypatch.setattr(
        freezer,
        "measure_boundaries_for_system",
        lambda _path, _detected: boundaries,
    )

    result = gate.prepare_independent_multihead_recovery_gate(tmp_path)

    prepared_path = Path(result["prepared_manifest"])
    prepared = freezer._read_json(prepared_path)
    assert prepared["target"] == {
        "slug": gate.A_MEDIO_PALO_SLUG,
        "system_index": gate.TARGET_SYSTEM_INDEX,
    }
    assert len(prepared["artifacts"]["crops"]) == gate.EXPECTED_CROP_COUNT
    assert prepared["independent_multihead_recovery_gate"]["truth_used"] is False
    assert prepared["independent_multihead_recovery_gate"]["parameters"] == (
        recovery.EDGE_SAFE_STEM_MULTIHEAD_PARAMETERS
    )
    context = freezer._read_json(prepared_path.parent / "context/allowed_context.json")
    assert context["allowed_context"] == {
        "clef": "treble",
        "time_signature": None,
        "key_hint": None,
        "expected_measure_beats": None,
        "allow_pickup": False,
    }
    assert any("musicxml" in path for path in prepared["forbidden_truth_paths"])
    assert any("truth" in path for path in prepared["forbidden_truth_paths"])


def test_multihead_lane_adds_two_companions_without_new_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = _measure_image(tmp_path)
    rows = [_inference_row(image_path, measure=index) for index in range(1, 8)]
    monkeypatch.setattr(recovery, "candidate_local_stem_features", _stem_features)
    inference_dir = tmp_path / "baseline_inference_v1"
    inference_dir.mkdir()
    inference._write_jsonl(inference_dir / "inference.jsonl", rows)
    inference._write_jsonl(
        inference_dir / "predictions.jsonl", [row["canonical_prediction"] for row in rows]
    )
    inference._write_json(inference_dir / "manifest.json", {"kind": "synthetic"})

    result = runner.materialize_paired_recovery(
        rows,
        model_payload=_model_payload(),
        inference_dir=inference_dir,
        expected_target={"slug": gate.A_MEDIO_PALO_SLUG, "system_index": 7},
    )

    paired = inference._read_jsonl(Path(result["paired_predictions"]))
    first = paired[0]["lanes"]
    recovered = first["multihead_recovery"]
    assert recovered["recovered_head_count"] == 2
    assert recovered["recovered_candidate_ids"] == ["companion_a", "companion_b"]
    assert [
        item["onset_group_index"] for item in recovered["candidate_lane"] if item["recovered"]
    ] == [1, 1]
    assert result["invariance"]["maximum_two_companions_per_existing_x_group"] is True
    assert result["invariance"]["recovered_head_count"] == 14
    with pytest.raises(FileExistsError, match="create-once"):
        runner.materialize_paired_recovery(
            rows,
            model_payload=_model_payload(),
            inference_dir=inference_dir,
            expected_target={"slug": gate.A_MEDIO_PALO_SLUG, "system_index": 7},
        )


def test_multihead_contract_rejects_three_companions_for_one_group() -> None:
    row = {"staff_geometry": {"raw_staff_lines_y_px": [20, 30, 40, 50, 60]}}
    selector = _model_payload()["replay"]["selector"]
    baseline = [{"candidate_id": "anchor", "center": {"x": 30, "y": 40}}]
    recovered = [
        {"candidate_id": "a", "center": {"x": 30, "y": 20}, "recovery_group_index": 1},
        {"candidate_id": "b", "center": {"x": 30, "y": 60}, "recovery_group_index": 1},
        {"candidate_id": "c", "center": {"x": 30, "y": 10}, "recovery_group_index": 1},
    ]

    with pytest.raises(ValueError, match="per-group companion cap"):
        runner._verify_additive_recovery(
            row,
            selector=selector,
            baseline=baseline,
            recovered=recovered,
        )


def _model_payload() -> dict:
    return {
        "replay": {
            "selector": {
                "threshold": 0.5,
                "nms_x_spaces": 1.0,
                "minimum_selected_count": 0,
                "maximum_selected_count": 2,
            }
        }
    }


def _inference_row(image_path: Path, *, measure: int) -> dict:
    identity = {
        "slug": gate.A_MEDIO_PALO_SLUG,
        "system_index": 7,
        "system_measure_index": measure,
        "automatic_measure_index": measure,
    }
    candidates = [
        _candidate("anchor", 30.0, 40.0, 0.9, 1),
        _candidate("companion_a", 30.0, 20.0, 0.8, 2),
        _candidate("companion_b", 30.0, 60.0, 0.79, 3),
        _candidate("later", 80.0, 40.0, 0.85, 4),
    ]
    notes = [
        _generic_note("anchor", 30.0, 40.0, "A4", 69),
        _generic_note("later", 80.0, 40.0, "A5", 81),
    ]
    canonical = {
        "identity": identity,
        "notes": notes,
        "rests": [],
        "rhythm_tokens": [],
        "measure_extent_beats": None,
        "decoder_status": "not_applied_missing_expected_measure_beats",
        "inference_provenance": {"truth_used": False},
    }
    return {
        "schema_version": 1,
        "identity": identity,
        "truth_used": False,
        "source": {"image": str(image_path), "sha256": _sha256(image_path)},
        "allowed_context": {
            "allow_pickup": False,
            "clef": "treble",
            "expected_measure_beats": None,
            "key_hint": None,
            "time_signature": None,
        },
        "staff_geometry": {"raw_staff_lines_y_px": [20, 30, 40, 50, 60]},
        "candidate_predictions": candidates,
        "automatic_anchors": [],
        "anchor_features": [],
        "residual_rest_features": [],
        "visual_symbols": [],
        "decoded_symbols": [],
        "decoder_status": "not_applied_missing_expected_measure_beats",
        "canonical_prediction": canonical,
    }


def _candidate(candidate_id: str, x: float, y: float, score: float, rank: int) -> dict:
    return {
        "candidate_id": candidate_id,
        "center": {"x": x, "y": y},
        "score": score,
        "detector_rank": rank,
        "bbox": {"left": int(x - 4), "top": int(y - 4), "right": int(x + 4), "bottom": int(y + 4)},
    }


def _generic_note(candidate_id: str, x: float, y: float, pitch: str, midi: int) -> dict:
    return {
        "pitch": pitch,
        "pitch_midi": midi,
        "onset_beats": None,
        "duration_beats": None,
        "candidate_id": candidate_id,
        "center": {"x": x, "y": y},
    }


def _stem_features(_row: dict) -> tuple[dict, dict]:
    return (
        {
            "anchor": {"score": 0.8},
            "companion_a": {"score": 0.8},
            "companion_b": {"score": 0.8},
            "later": {"score": 0.8},
        },
        {"method": "synthetic"},
    )


def _measure_image(root: Path) -> Path:
    path = root / "measure.png"
    image = Image.new("L", (120, 80), "white")
    draw = ImageDraw.Draw(image)
    for y in (20, 30, 40, 50, 60):
        draw.line((0, y, 119, y), fill="black", width=1)
    image.save(path)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
