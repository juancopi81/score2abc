import json
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.experiments import freeze_fourth_score_heldout as spike
from scripts.experiments import freeze_third_score_heldout as base


def test_prepare_pins_layout_visual_key_and_rhythm_meter_prior(tmp_path: Path, monkeypatch) -> None:
    candidate = base.Candidate(spike.GATOE_FIQUE_SLUG, 3, "only")
    boundaries = tuple(index / 6 for index in range(7))
    monkeypatch.setattr(base, "detect_barlines", lambda _path: list(boundaries))
    monkeypatch.setattr(
        base,
        "measure_boundaries_for_system",
        lambda _path, _detected: list(boundaries),
    )
    _system_image(tmp_path, candidate.slug, 3)
    initial = _system_image(tmp_path, candidate.slug, 1)
    _metadata(tmp_path, candidate.slug)
    monkeypatch.setattr(
        spike.key_detector,
        "detect_signature",
        lambda path, mode: {
            "input": {"path": str(path), "sha256": base._sha256(path)},
            "mode": mode,
            "fifths": -1,
            "gate_passed": True,
            "truth_used_for_prediction": False,
        },
    )
    monkeypatch.setattr(
        spike.key_detector,
        "_draw_overlay",
        lambda _prediction, path: Image.open(initial).save(path),
    )

    result = spike.prepare_fourth_score(
        tmp_path,
        candidate_pool=(candidate,),
        policy=base.LayoutPolicy(
            min_width_px=500,
            min_height_px=50,
            min_measure_count=6,
            max_measure_count=6,
            min_crop_width_px=50,
            max_spacing_cv=0.1,
        ),
    )

    prepared_path = Path(result["prepared_manifest"])
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    context_record = prepared["artifacts"]["context"]["allowed_context"]
    context = json.loads(
        (prepared_path.parent / context_record["path"]).read_text(encoding="utf-8")
    )
    assert prepared["kind"] == "fourth_score_fresh_heldout_prepare"
    assert spike.OUTPUT_SUBDIR in prepared_path.parts
    assert context["truth_used"] is False
    assert context["allowed_context"]["time_signature"] == "3/4"
    assert context["allowed_context"]["expected_measure_beats"] == 3.0
    assert context["allowed_context"]["key_hint"] == "1 flat(s): Bb"
    assert context["provenance"]["time_signature"].startswith("provisional_")
    base._verify_prepared_manifest(
        prepared_path.parent,
        prepared_path,
        prepared,
        expected_kind=spike.FOURTH_SCORE_GATE.prepare_kind,
    )


def test_freeze_uses_fourth_score_artifact_kinds(tmp_path: Path, monkeypatch) -> None:
    candidate = base.Candidate(spike.GATOE_FIQUE_SLUG, 3, "only")
    boundaries = tuple(index / 6 for index in range(7))
    monkeypatch.setattr(base, "detect_barlines", lambda _path: list(boundaries))
    monkeypatch.setattr(
        base,
        "measure_boundaries_for_system",
        lambda _path, _detected: list(boundaries),
    )
    _system_image(tmp_path, candidate.slug, 3)
    initial = _system_image(tmp_path, candidate.slug, 1)
    _metadata(tmp_path, candidate.slug)
    monkeypatch.setattr(
        spike.key_detector,
        "detect_signature",
        lambda path, mode: {
            "input": {"path": str(path), "sha256": base._sha256(path)},
            "mode": mode,
            "fifths": -1,
            "gate_passed": True,
            "truth_used_for_prediction": False,
        },
    )
    monkeypatch.setattr(
        spike.key_detector,
        "_draw_overlay",
        lambda _prediction, path: Image.open(initial).save(path),
    )
    prepared = spike.prepare_fourth_score(
        tmp_path,
        candidate_pool=(candidate,),
        policy=base.LayoutPolicy(min_width_px=500, min_height_px=50),
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    predictions = artifacts / "predictions.jsonl"
    model = artifacts / "model.json"
    training = artifacts / "training.json"
    predictions.write_text('{"notes":[]}\n', encoding="utf-8")
    model.write_text("{}\n", encoding="utf-8")
    training.write_text("{}\n", encoding="utf-8")

    frozen = spike.freeze_prepared_fourth_score(
        Path(prepared["prepared_manifest"]),
        predictions_path=predictions,
        model_artifact_paths=(model,),
        training_artifact_paths=(training,),
    )

    freeze = json.loads(Path(frozen["freeze"]).read_text(encoding="utf-8"))
    sealed = json.loads(Path(frozen["sealed_manifest"]).read_text(encoding="utf-8"))
    assert freeze["kind"] == "fourth_score_fresh_heldout_freeze"
    assert sealed["kind"] == "fourth_score_fresh_heldout_sealed_manifest"


def _system_image(root: Path, slug: str, index: int) -> Path:
    path = root / slug / "systems" / f"system_{index:03d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (840, 120), "white")
    draw = ImageDraw.Draw(image)
    for y in (35, 45, 55, 65, 75):
        draw.line((0, y, 839, y), fill="black", width=2)
    for measure in range(6):
        x = measure * 140 + 70
        draw.ellipse((x - 7, 50, x + 7, 58), fill="black")
        draw.line((x + 7, 54, x + 7, 31), fill="black", width=2)
    image.save(path)
    return path


def _metadata(root: Path, slug: str) -> None:
    path = root / slug / "metadata.json"
    path.write_text(
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
