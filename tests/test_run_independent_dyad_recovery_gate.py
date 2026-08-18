import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from scripts.experiments import freeze_independent_dyad_recovery_gate as gate
from scripts.experiments import freeze_third_score_heldout as base
from scripts.experiments import run_independent_dyad_recovery_gate as runner
from scripts.experiments import run_third_score_heldout_inference as inference
from scripts.experiments import spike_consumed_polyphonic_pitch_repair as recovery


def test_materializes_additive_pair_and_preserves_generic_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = _measure_image(tmp_path)
    rows = [_inference_row(image_path, measure=index) for index in range(1, 12)]
    monkeypatch.setattr(recovery, "candidate_local_stem_features", _stem_features)
    inference_dir = tmp_path / "baseline_inference_v1"
    inference_dir.mkdir()
    _write_generic_files(inference_dir, rows)

    result = runner.materialize_paired_recovery(
        rows,
        model_payload=_model_payload(),
        inference_dir=inference_dir,
        expected_target={
            "slug": gate.NO_LO_CREAS_SLUG,
            "system_index": gate.TARGET_SYSTEM_INDEX,
        },
    )

    paired = inference._read_jsonl(Path(result["paired_predictions"]))
    manifest = inference._read_json(Path(result["manifest"]))
    diagnostics = inference._read_jsonl(Path(result["pair_dir"]) / "recovery_diagnostics.jsonl")
    assert len(paired) == len(diagnostics) == 11
    assert manifest["measure_count"] == 11
    assert manifest["recovered_head_count"] == 11
    assert manifest["version"] == runner.PAIR_VERSION
    assert "one notes list" in manifest["paired_note_contract"]
    first = paired[0]["lanes"]
    assert first["baseline_generic"]["canonical_prediction"] == rows[0]["canonical_prediction"]
    assert [item["candidate_id"] for item in first["baseline_generic"]["candidate_lane"]] == [
        "anchor",
        "later",
    ]
    assert first["edge_safe_recovery"]["recovered_candidate_ids"] == ["companion"]
    assert [item["candidate_id"] for item in first["edge_safe_recovery"]["candidate_lane"]] == [
        "companion",
        "anchor",
        "later",
    ]
    assert [item["onset_group_index"] for item in first["baseline_generic"]["candidate_lane"]] == [
        1,
        2,
    ]
    assert [
        item["onset_group_index"] for item in first["edge_safe_recovery"]["candidate_lane"]
    ] == [1, 1, 2]
    assert first["baseline_generic"]["notes"] == [
        {
            "pitch": "B4",
            "pitch_midi": 71,
            "staff_position": 4,
            "onset_group_index": 1,
            "recovered": False,
            "onset_beats": None,
            "duration_beats": None,
            "candidate_id": "anchor",
            "center": {"x": 30.0, "y": 40.0},
            "generic_pitch_diagnostic": {"pitch": "A4", "pitch_midi": 69},
        },
        {
            "pitch": "B4",
            "pitch_midi": 71,
            "staff_position": 4,
            "onset_group_index": 2,
            "recovered": False,
            "onset_beats": None,
            "duration_beats": None,
            "candidate_id": "later",
            "center": {"x": 80.0, "y": 40.0},
            "generic_pitch_diagnostic": {"pitch": "A5", "pitch_midi": 81},
        },
    ]
    recovery_notes = first["edge_safe_recovery"]["notes"]
    assert [note["candidate_id"] for note in recovery_notes] == [
        "companion",
        "anchor",
        "later",
    ]
    assert [note["onset_group_index"] for note in recovery_notes] == [1, 1, 2]
    assert [note["staff_position"] for note in recovery_notes] == [8, 4, 4]
    assert [note["recovered"] for note in recovery_notes] == [True, False, False]
    assert {note["onset_group_index"] for note in first["edge_safe_recovery"]["notes"]} == {
        note["onset_group_index"] for note in first["baseline_generic"]["notes"]
    }
    assert diagnostics[0]["recovery"][0]["recovery_group_index"] == 1
    assert result["invariance"]["passed"] is True
    assert result["invariance"]["paired_lane_pitches_use_direct_natural_treble_mapping"] is True
    with pytest.raises(FileExistsError, match="create-once"):
        runner.materialize_paired_recovery(
            rows,
            model_payload=_model_payload(),
            inference_dir=inference_dir,
            expected_target={
                "slug": gate.NO_LO_CREAS_SLUG,
                "system_index": gate.TARGET_SYSTEM_INDEX,
            },
        )


def test_paired_contract_rejects_baseline_repositioning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = _measure_image(tmp_path)
    rows = [_inference_row(image_path, measure=index) for index in range(1, 12)]
    monkeypatch.setattr(recovery, "candidate_local_stem_features", _stem_features)
    selector = recovery.selector_config_from_model(_model_payload())
    paired_rows = [
        runner._pair_row(
            row,
            selector=selector,
            expected_target={
                "slug": gate.NO_LO_CREAS_SLUG,
                "system_index": gate.TARGET_SYSTEM_INDEX,
            },
        )[0]
        for row in rows
    ]
    paired_rows[0]["lanes"]["edge_safe_recovery"]["candidate_lane"][1]["center"]["x"] += 1

    with pytest.raises(ValueError, match="repositioned"):
        runner.verify_paired_contract(
            paired_rows,
            rows,
            expected_target={
                "slug": gate.NO_LO_CREAS_SLUG,
                "system_index": gate.TARGET_SYSTEM_INDEX,
            },
        )


def test_resumes_only_validated_generic_inference_before_pair_or_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared_path = tmp_path / "prepared_manifest.json"
    prepared_path.write_text(
        json.dumps(
            {
                "kind": gate.INDEPENDENT_DYAD_RECOVERY_GATE.prepare_kind,
                "target": {
                    "slug": gate.NO_LO_CREAS_SLUG,
                    "system_index": gate.TARGET_SYSTEM_INDEX,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    inference_dir = tmp_path / runner.DEFAULT_INFERENCE_DIRNAME
    inference_dir.mkdir()
    image_path = _measure_image(tmp_path)
    rows = [_inference_row(image_path, measure=index) for index in range(1, 12)]
    _write_generic_files(inference_dir, rows)
    config = inference.GATE_CONFIGS[gate.INDEPENDENT_DYAD_RECOVERY_GATE.prepare_kind]
    inference._write_json(
        inference_dir / "manifest.json",
        {
            "kind": config["manifest_kind"],
            "version": config["inference_version"],
            "status": "inferred_awaiting_freeze",
            "target": {
                "slug": gate.NO_LO_CREAS_SLUG,
                "system_index": gate.TARGET_SYSTEM_INDEX,
            },
            "truth_accessed": False,
            "truth_used": False,
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        inference,
        "_verify_inference_binding",
        lambda *_args, **_kwargs: {"expected_count": gate.EXPECTED_CROP_COUNT},
    )
    monkeypatch.setattr(
        inference,
        "materialize_third_score_inference",
        lambda *_args, **_kwargs: pytest.fail("resume must not rerun generic inference"),
    )

    result = runner._materialize_or_resume_baseline(
        prepared_path,
        model_dir=tmp_path / "model",
    )

    assert result["resumed_after_validated_pre_pair_failure"] is True
    assert result["output_count"] == 11
    pair_dir = inference_dir / runner.PAIR_DIRNAME
    pair_dir.mkdir()
    with pytest.raises(FileExistsError, match="paired or frozen"):
        runner._materialize_or_resume_baseline(
            prepared_path,
            model_dir=tmp_path / "model",
        )
    pair_dir.rmdir()
    (tmp_path / runner.PAIR_FREEZE_DIRNAME).mkdir()
    with pytest.raises(FileExistsError, match="paired or frozen"):
        runner._materialize_or_resume_baseline(
            prepared_path,
            model_dir=tmp_path / "model",
        )


def test_fixed_recovery_rejects_two_companions_for_one_group() -> None:
    row = {"staff_geometry": {"raw_staff_lines_y_px": [20, 30, 40, 50, 60]}}
    selector = _model_payload()["replay"]["selector"]
    baseline = [{"candidate_id": "anchor", "center": {"x": 30, "y": 40}}]
    recovered = [
        {"candidate_id": "a", "center": {"x": 30, "y": 20}, "recovery_group_index": 1},
        {"candidate_id": "b", "center": {"x": 31, "y": 60}, "recovery_group_index": 1},
    ]

    with pytest.raises(ValueError, match="more than one companion"):
        runner._verify_additive_recovery(
            row,
            selector=selector,
            baseline=baseline,
            recovered=recovered,
        )


def test_paired_contract_rejects_recovered_note_group_reassignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = _measure_image(tmp_path)
    rows = [_inference_row(image_path, measure=index) for index in range(1, 12)]
    monkeypatch.setattr(recovery, "candidate_local_stem_features", _stem_features)
    selector = recovery.selector_config_from_model(_model_payload())
    paired_rows = [
        runner._pair_row(
            row,
            selector=selector,
            expected_target={
                "slug": gate.NO_LO_CREAS_SLUG,
                "system_index": gate.TARGET_SYSTEM_INDEX,
            },
        )[0]
        for row in rows
    ]
    recovered_note = next(
        note for note in paired_rows[0]["lanes"]["edge_safe_recovery"]["notes"] if note["recovered"]
    )
    recovered_note["onset_group_index"] = 3

    with pytest.raises(ValueError, match="note records or onset groups drifted"):
        runner.verify_paired_contract(
            paired_rows,
            rows,
            expected_target={
                "slug": gate.NO_LO_CREAS_SLUG,
                "system_index": gate.TARGET_SYSTEM_INDEX,
            },
        )


def test_paired_freeze_pins_all_provenance_roles_and_rejects_snapshot_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    boundaries = [index / gate.EXPECTED_CROP_COUNT for index in range(12)]
    monkeypatch.setattr(base, "detect_barlines", lambda _path: boundaries)
    monkeypatch.setattr(base, "measure_boundaries_for_system", lambda _path, _raw: boundaries)
    _target_system_and_metadata(tmp_path)
    prepared_result = gate.prepare_independent_dyad_recovery_gate(tmp_path)
    prepared_path = Path(prepared_result["prepared_manifest"])
    namespace_root = prepared_path.parent

    measure_image = _measure_image(tmp_path)
    rows = [_inference_row(measure_image, measure=index) for index in range(1, 12)]
    monkeypatch.setattr(recovery, "candidate_local_stem_features", _stem_features)
    inference_dir = namespace_root / runner.DEFAULT_INFERENCE_DIRNAME
    inference_dir.mkdir()
    _write_generic_files(inference_dir, rows)
    pair_result = runner.materialize_paired_recovery(
        rows,
        model_payload=_model_payload(),
        inference_dir=inference_dir,
        expected_target={
            "slug": gate.NO_LO_CREAS_SLUG,
            "system_index": gate.TARGET_SYSTEM_INDEX,
        },
    )

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    model_artifact = model_dir / "model.json"
    training_input = tmp_path / "training.json"
    model_artifact.write_text("{}\n", encoding="utf-8")
    training_input.write_text("{}\n", encoding="utf-8")
    metadata_path = tmp_path / gate.NO_LO_CREAS_SLUG / "metadata.json"
    monkeypatch.setattr(
        inference,
        "_verify_inference_binding",
        lambda *_args, **_kwargs: {
            "metadata_record": inference._file_record(metadata_path),
            "model_artifact_paths": (model_artifact,),
            "training_input_paths": (training_input,),
            "model_implementation_path": Path(recovery.__file__).resolve(),
        },
    )

    result = runner.freeze_paired_recovery(
        prepared_path,
        model_dir=model_dir,
        inference_dir=inference_dir,
        pair_dir=Path(pair_result["pair_dir"]),
    )

    freeze = inference._read_json(Path(result["freeze"]))
    assert set(freeze["pins"]) == {
        "paired_predictions",
        "paired_artifacts",
        "prepared_and_source",
        "baseline_inference",
        "model_and_training",
        "implementations",
    }
    assert all(freeze["pins"][role] for role in freeze["pins"])
    assert freeze["parameters"] == recovery.EDGE_SAFE_STEM_DYAD_PARAMETERS
    implementation_pin = freeze["pins"]["implementations"][0]
    implementation_snapshot = (
        Path(result["freeze"]).parent.parent
        / implementation_pin["snapshot_path_relative_to_namespace"]
    )
    implementation_snapshot.write_text(
        implementation_snapshot.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="snapshot hash drift"):
        runner.verify_paired_freeze(Path(result["freeze"]).parent)


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
        "slug": gate.NO_LO_CREAS_SLUG,
        "system_index": gate.TARGET_SYSTEM_INDEX,
        "system_measure_index": measure,
        "automatic_measure_index": measure,
    }
    candidates = [
        {
            "candidate_id": "anchor",
            "center": {"x": 30.0, "y": 40.0},
            "score": 0.9,
            "detector_rank": 1,
            "bbox": {"left": 26, "top": 36, "right": 34, "bottom": 44},
        },
        {
            "candidate_id": "companion",
            "center": {"x": 30.0, "y": 20.0},
            "score": 0.8,
            "detector_rank": 2,
            "bbox": {"left": 26, "top": 16, "right": 34, "bottom": 24},
        },
        {
            "candidate_id": "later",
            "center": {"x": 80.0, "y": 40.0},
            "score": 0.85,
            "detector_rank": 3,
            "bbox": {"left": 76, "top": 36, "right": 84, "bottom": 44},
        },
    ]
    generic_pitches = (("anchor", 30.0, "A4", 69), ("later", 80.0, "A5", 81))
    notes = [
        {
            "pitch": pitch,
            "pitch_midi": pitch_midi,
            "onset_beats": None,
            "duration_beats": None,
            "candidate_id": candidate_id,
            "center": {"x": x, "y": 40.0},
        }
        for candidate_id, x, pitch, pitch_midi in generic_pitches
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


def _stem_features(_row: dict) -> tuple[dict, dict]:
    return (
        {
            "anchor": {"score": 0.8},
            "companion": {"score": 0.8},
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


def _write_generic_files(root: Path, rows: list[dict]) -> None:
    inference._write_jsonl(root / "inference.jsonl", rows)
    inference._write_jsonl(
        root / "predictions.jsonl", [row["canonical_prediction"] for row in rows]
    )
    inference._write_json(root / "manifest.json", {"kind": "synthetic"})


def _target_system_and_metadata(root: Path) -> None:
    score_root = root / gate.NO_LO_CREAS_SLUG
    system = score_root / "systems" / f"system_{gate.TARGET_SYSTEM_INDEX:03d}.png"
    system.parent.mkdir(parents=True)
    image = Image.new("L", (1320, 120), "white")
    draw = ImageDraw.Draw(image)
    for y in (35, 45, 55, 65, 75):
        draw.line((0, y, 1319, y), fill="black", width=2)
    image.save(system)
    (score_root / "metadata.json").write_text(
        json.dumps(
            {
                "title": "No lo Creas",
                "composer": "A. Vasquez Pedrero",
                "rhythm": "Pasillo",
                "time_signature": None,
                "key_hint": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
