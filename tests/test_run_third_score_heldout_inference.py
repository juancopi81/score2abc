import json
import shutil
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from scripts.experiments import freeze_fourth_score_heldout as fourth_freezer
from scripts.experiments import freeze_third_score_heldout as freezer
from scripts.experiments import run_third_score_heldout_inference as spike


@pytest.fixture
def sealed_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    candidate = freezer.Candidate(spike.LA_CHATA_SLUG, 7, "test")
    boundaries = tuple(index / 7 for index in range(8))
    monkeypatch.setattr(freezer, "detect_barlines", lambda _path: list(boundaries))
    monkeypatch.setattr(
        freezer,
        "measure_boundaries_for_system",
        lambda _path, _detected: list(boundaries),
    )
    _system_image(tmp_path, candidate)
    metadata = tmp_path / candidate.slug / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "title": "Synthetic",
                "composer": "Test",
                "rhythm": "Pasillo",
                "time_signature": None,
                "key_hint": None,
                "tempo_hint": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prepared = freezer.prepare_third_score(
        tmp_path,
        namespace="test-v1",
        candidate_pool=(candidate,),
        policy=freezer.LayoutPolicy(
            min_width_px=500,
            min_height_px=50,
            min_measure_count=7,
            max_measure_count=7,
            min_crop_width_px=50,
            max_spacing_cv=0.1,
        ),
    )
    model_dir = _model_dir(tmp_path)
    return Path(prepared["prepared_manifest"]), model_dir


def test_reconstructs_serialized_selector_recovery_and_pitch(tmp_path: Path) -> None:
    payload = json.loads((_model_dir(tmp_path) / "model.json").read_text(encoding="utf-8"))

    model, _predictor, audit = spike.reconstruct_model(payload)

    assert model.learned_threshold == pytest.approx(0.0)
    assert model.base.nms_x_spaces == pytest.approx(0.85)
    assert model.base.minimum_selected_count == 2
    assert model.base.maximum_selected_count == 5
    assert model.recovery.leading_gap_spaces == pytest.approx(3.5)
    assert model.recovery.score_margin == pytest.approx(0.0025)
    assert audit["scorer"]["positive_vector_count"] == 1
    assert audit["scorer"]["negative_vector_count"] == 1
    assert audit["pitch"]["method"] == "key_signature_only"


def test_materializes_exactly_seven_outputs_deterministically_and_create_once(
    sealed_inputs: tuple[Path, Path],
) -> None:
    prepared, model_dir = sealed_inputs

    first = spike.materialize_third_score_inference(
        prepared, model_dir=model_dir, inference_dirname="inference_a"
    )
    second = spike.materialize_third_score_inference(
        prepared, model_dir=model_dir, inference_dirname="inference_b"
    )

    assert first["output_count"] == second["output_count"] == 7
    assert first["predictions_sha256"] == second["predictions_sha256"]
    rows = spike._read_jsonl(Path(first["predictions"]))
    assert len(rows) == 7
    assert all(
        row["decoder_status"] == "not_applied_missing_expected_measure_beats" for row in rows
    )
    assert all(row["inference_provenance"]["truth_used"] is False for row in rows)
    manifest = spike._read_json(Path(first["manifest"]))
    assert manifest["version"] == "third-score-inference-v2"
    assert manifest["prepared_manifest"] == spike._file_record(prepared)
    assert manifest["context"]["metadata"]["sha256"] == spike._sha256(
        freezer._find_out_dir(prepared.parent) / spike.LA_CHATA_SLUG / "metadata.json"
    )
    with pytest.raises(FileExistsError, match="create-once"):
        spike.materialize_third_score_inference(
            prepared, model_dir=model_dir, inference_dirname="inference_a"
        )


def test_rejects_prepared_request_hash_drift(sealed_inputs: tuple[Path, Path]) -> None:
    prepared, model_dir = sealed_inputs
    requests = prepared.parent / "requests.jsonl"
    requests.write_text(requests.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Prepared requests hash drift"):
        spike.materialize_third_score_inference(prepared, model_dir=model_dir)


def test_rejects_la_chata_truth_path(tmp_path: Path) -> None:
    truth = tmp_path / "dataset/ground_truth" / f"{spike.LA_CHATA_SLUG}.json"
    truth.parent.mkdir(parents=True)
    truth.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden"):
        spike.materialize_third_score_inference(truth, model_dir=tmp_path)


def test_materializes_and_freezes_variable_count_fourth_score_with_pinned_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = freezer.Candidate(fourth_freezer.GATOE_FIQUE_SLUG, 3, "test")
    boundaries = tuple(index / 6 for index in range(7))
    monkeypatch.setattr(freezer, "detect_barlines", lambda _path: list(boundaries))
    monkeypatch.setattr(
        freezer,
        "measure_boundaries_for_system",
        lambda _path, _detected: list(boundaries),
    )
    target = _system_image(tmp_path, candidate)
    initial = target.with_name("system_001.png")
    shutil.copyfile(target, initial)
    metadata = tmp_path / candidate.slug / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "title": "Synthetic",
                "composer": "Test",
                "rhythm": "Pasillo",
                "time_signature": None,
                "key_hint": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        fourth_freezer.key_detector,
        "detect_signature",
        lambda path, mode: {
            "input": {"path": str(path), "sha256": spike._sha256(path)},
            "mode": mode,
            "fifths": -1,
            "gate_passed": True,
            "truth_used_for_prediction": False,
        },
    )
    monkeypatch.setattr(
        fourth_freezer.key_detector,
        "_draw_overlay",
        lambda _prediction, path: Image.open(initial).save(path),
    )
    prepared_report = fourth_freezer.prepare_fourth_score(
        tmp_path,
        candidate_pool=(candidate,),
        policy=freezer.LayoutPolicy(
            min_width_px=500,
            min_height_px=50,
            min_measure_count=6,
            max_measure_count=6,
            min_crop_width_px=50,
            max_spacing_cv=0.1,
        ),
    )
    prepared = Path(prepared_report["prepared_manifest"])
    model_dir = _model_dir(tmp_path)

    inference = spike.materialize_third_score_inference(prepared, model_dir=model_dir)

    assert inference["output_count"] == 6
    manifest = spike._read_json(Path(inference["manifest"]))
    assumptions = spike._read_json(Path(inference["inference_dir"]) / "assumptions.json")
    assert manifest["kind"] == "fourth_score_truth_blind_inference_manifest"
    assert manifest["version"] == "fourth-score-inference-v1"
    assert assumptions["allowed_context"]["expected_measure_beats"] == 3.0
    assert "Bb" in assumptions["allowed_context"]["key_hint"]

    frozen = spike.freeze_inference(
        prepared,
        inference_dir=Path(inference["inference_dir"]),
        model_dir=model_dir,
    )
    freeze = spike._read_json(Path(frozen["freeze"]))
    assert freeze["kind"] == "fourth_score_fresh_heldout_freeze"
    assert freeze["inference_binding"]["version"] == "fourth-score-inference-v1"
    assert "prepared_context" in freeze["inference_binding"]["inference"]
    spike.verify_frozen_outputs(prepared.parent / "frozen")


def test_freeze_pins_predictions_model_and_training(
    sealed_inputs: tuple[Path, Path],
) -> None:
    prepared, model_dir = sealed_inputs
    inference = spike.materialize_third_score_inference(prepared, model_dir=model_dir)

    result = spike.freeze_inference(
        prepared,
        inference_dir=Path(inference["inference_dir"]),
        model_dir=model_dir,
    )
    freeze = spike._read_json(Path(result["freeze"]))

    assert freeze["status"] == "frozen_awaiting_truth"
    assert freeze["truth_accessed"] is False
    assert freeze["inference_binding"]["version"] == "third-score-inference-v2"
    binding = freeze["inference_binding"]
    assert binding["prepared_manifest"] == spike._file_record(prepared)
    assert binding["inference"]["manifest"]["source_path"].endswith("/inference_v2/manifest.json")
    assert binding["inference"]["implementation"]["source_sha256"] == spike._sha256(
        Path(spike.__file__)
    )
    assert binding["inference"]["metadata"]["source_path"].endswith("/metadata.json")
    assert binding["inference"]["assumptions"]["source_path"].endswith(
        "/inference_v2/assumptions.json"
    )
    assert binding["inference"]["requests"]["source_path"].endswith("/inference_v2/requests.jsonl")
    assert binding["inference"]["replay"]["source_path"].endswith("/inference_v2/replay.json")
    assert binding["inference"]["detailed_inference"]["source_path"].endswith(
        "/inference_v2/inference.jsonl"
    )
    assert binding["selected_model"]["manifest"]["source_path"].endswith("/model/manifest.json")
    assert binding["selected_model"]["artifacts"]["report.md"]["source_path"].endswith(
        "/model/report.md"
    )
    spike.verify_frozen_outputs(prepared.parent / "frozen")


def test_frozen_verifier_rejects_coherent_cross_role_model_manifest_rebinding(
    sealed_inputs: tuple[Path, Path],
) -> None:
    prepared, model_dir = sealed_inputs
    inference = spike.materialize_third_score_inference(prepared, model_dir=model_dir)
    result = spike.freeze_inference(
        prepared,
        inference_dir=Path(inference["inference_dir"]),
        model_dir=model_dir,
    )
    freeze_path = Path(result["freeze"])
    sealed_path = freeze_path.parent / "sealed_manifest.json"
    freeze = spike._read_json(freeze_path)
    binding = freeze["inference_binding"]
    binding["selected_model"]["manifest"] = binding["inference"]["requests"]
    freeze["inference_binding"] = binding
    spike._write_json(freeze_path, freeze)
    sealed = spike._read_json(sealed_path)
    sealed["freeze"]["sha256"] = spike._sha256(freeze_path)
    sealed["inference_binding_sha256"] = spike._hash_json(binding)
    spike._write_json(sealed_path, sealed)

    with pytest.raises(ValueError, match="selected-model manifest binding mismatch"):
        spike.verify_frozen_outputs(prepared.parent / "frozen")


def test_freeze_rejects_cross_namespace_prepared_substitution(
    sealed_inputs: tuple[Path, Path],
) -> None:
    prepared, model_dir = sealed_inputs
    inference = spike.materialize_third_score_inference(prepared, model_dir=model_dir)
    out_dir = prepared.parents[4]
    candidate = freezer.Candidate(spike.LA_CHATA_SLUG, 7, "test")
    second = freezer.prepare_third_score(
        out_dir,
        namespace="test-v2",
        candidate_pool=(candidate,),
        policy=freezer.LayoutPolicy(
            min_width_px=500,
            min_height_px=50,
            min_measure_count=7,
            max_measure_count=7,
            min_crop_width_px=50,
            max_spacing_cv=0.1,
        ),
    )
    second_prepared = Path(second["prepared_manifest"])

    with pytest.raises(ValueError, match="prepared manifest path substitution"):
        spike.freeze_inference(
            second_prepared,
            inference_dir=Path(inference["inference_dir"]),
            model_dir=model_dir,
        )
    assert not (second_prepared.parent / "frozen").exists()


def test_freeze_rejects_cross_directory_model_substitution(
    sealed_inputs: tuple[Path, Path],
) -> None:
    prepared, model_dir = sealed_inputs
    inference = spike.materialize_third_score_inference(prepared, model_dir=model_dir)
    copied_model_dir = model_dir.with_name("copied-model")
    shutil.copytree(model_dir, copied_model_dir)

    with pytest.raises(ValueError, match="artifact path substitution"):
        spike.freeze_inference(
            prepared,
            inference_dir=Path(inference["inference_dir"]),
            model_dir=copied_model_dir,
        )
    assert not (prepared.parent / "frozen").exists()


def test_freeze_rejects_model_artifact_changed_after_inference(
    sealed_inputs: tuple[Path, Path],
) -> None:
    prepared, model_dir = sealed_inputs
    inference = spike.materialize_third_score_inference(prepared, model_dir=model_dir)
    model_path = model_dir / "model.json"
    model_path.write_text(model_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Selected model artifact hash drift"):
        spike.freeze_inference(
            prepared,
            inference_dir=Path(inference["inference_dir"]),
            model_dir=model_dir,
        )
    assert not (prepared.parent / "frozen").exists()


def test_freeze_rejects_metadata_changed_after_inference(
    sealed_inputs: tuple[Path, Path],
) -> None:
    prepared, model_dir = sealed_inputs
    inference = spike.materialize_third_score_inference(prepared, model_dir=model_dir)
    metadata_path = freezer._find_out_dir(prepared.parent) / spike.LA_CHATA_SLUG / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["tempo_hint"] = "changed after inference"
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="metadata/context provenance substitution"):
        spike.freeze_inference(
            prepared,
            inference_dir=Path(inference["inference_dir"]),
            model_dir=model_dir,
        )
    assert not (prepared.parent / "frozen").exists()


def test_freeze_rejects_inference_implementation_record_substitution(
    sealed_inputs: tuple[Path, Path],
) -> None:
    prepared, model_dir = sealed_inputs
    inference = spike.materialize_third_score_inference(prepared, model_dir=model_dir)
    manifest_path = Path(inference["manifest"])
    manifest = spike._read_json(manifest_path)
    manifest["implementation"]["sha256"] = "0" * 64
    spike._write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="Inference implementation hash drift"):
        spike.freeze_inference(
            prepared,
            inference_dir=Path(inference["inference_dir"]),
            model_dir=model_dir,
        )
    assert not (prepared.parent / "frozen").exists()


def _system_image(out_dir: Path, candidate: freezer.Candidate) -> Path:
    path = out_dir / candidate.slug / "systems" / f"system_{candidate.system_index:03d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (840, 120), "white")
    draw = ImageDraw.Draw(image)
    for y in (35, 45, 55, 65, 75):
        draw.line((0, y, 839, y), fill="black", width=2)
    for measure in range(7):
        x = measure * 120 + 60
        draw.ellipse((x - 7, 50, x + 7, 58), fill="black")
        draw.line((x + 7, 54, x + 7, 31), fill="black", width=2)
    image.save(path)
    return path


def _model_dir(root: Path) -> Path:
    model_dir = root / "model"
    if model_dir.exists():
        return model_dir
    model_dir.mkdir()
    vector_size = 17 * 13
    model = {
        "schema_version": 1,
        "kind": "vlm_melody_consumed_training_notehead_model",
        "experiment_version": "cross_score_notehead_v1",
        "configuration": "C",
        "blocked": False,
        "replay": {
            "feature_order": list(spike.dense.DENSE_FEATURES),
            "selection_mode": spike.dense.SELECTION_MODE,
            "method": {
                "method_id": "test_replay",
                "patch_id": "grayscale_staff_suppressed",
                "scorer_kind": "class_template",
                "selection": {"metrics": {}},
                "recovery": {
                    "leading_gap_spaces": 3.5,
                    "score_margin": 0.0025,
                    "metrics": {},
                    "base_metrics": {},
                    "searched": [],
                },
            },
            "selector": {
                "scorer": {
                    "kind": "patch_scorer",
                    "patch_id": "grayscale_staff_suppressed",
                    "scorer_kind": "class_template",
                    "positive_vectors": [[0.0] * vector_size],
                    "negative_vectors": [[1.0] * vector_size],
                },
                "threshold": 0.0,
                "nms_x_spaces": 0.85,
                "minimum_selected_count": 2,
                "maximum_selected_count": 5,
                "leading_gap_spaces": 3.5,
                "score_margin": 0.0025,
            },
            "pitch": {
                "method": "key_signature_only",
                "oof_selection": {},
                "accidental_samples": [],
            },
        },
        "training": {"keys": ["train:S01M01"], "scores": ["train"], "review_hashes": ["a"]},
        "third_score_truth_used": False,
    }
    training = {
        "selection": {"selected_configuration": "C"},
        "final_training": {"keys": model["training"]["keys"], "review_hashes": ["a"]},
        "input_provenance": {"test": []},
    }
    report = {"selection": {"selected_configuration": "C"}}
    for name, payload in (
        ("model.json", model),
        ("training_selection.json", training),
        ("report.json", report),
    ):
        (model_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (model_dir / "report.md").write_text("# Synthetic report\n", encoding="utf-8")
    manifest = {
        "kind": "vlm_melody_cross_score_consumed_retraining_manifest",
        "la_chata_truth_accessed": False,
        "artifacts": {
            name: {"path": str(model_dir / name), "sha256": spike._sha256(model_dir / name)}
            for name in ("model.json", "training_selection.json", "report.json", "report.md")
        },
        "implementation": {
            "path": str(Path(spike.__file__).resolve()),
            "sha256": spike._sha256(Path(spike.__file__).resolve()),
        },
    }
    (model_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return model_dir
