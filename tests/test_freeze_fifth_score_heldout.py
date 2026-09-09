import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from scripts.experiments import freeze_fifth_score_heldout as spike
from scripts.experiments import freeze_third_score_heldout as base


def test_prepare_pins_six_crops_and_inconclusive_key_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundaries = tuple(index / 6 for index in range(7))
    monkeypatch.setattr(base, "detect_barlines", lambda _path: list(boundaries))
    monkeypatch.setattr(
        base,
        "measure_boundaries_for_system",
        lambda _path, _detected: list(boundaries),
    )
    _system_image(tmp_path, index=1)
    _system_image(tmp_path, index=2)
    _metadata(tmp_path)
    key_prediction = {
        "input": {"path": "synthetic", "sha256": "0" * 64},
        "mode": "initial",
        "fifths": -1,
        "gate_passed": False,
        "truth_used_for_prediction": False,
    }
    monkeypatch.setattr(spike.key_detector, "detect_signature", lambda _path, mode: key_prediction)
    monkeypatch.setattr(
        spike.key_detector,
        "_draw_overlay",
        lambda _prediction, path: Image.open(
            tmp_path / spike.COQUETEOS_SLUG / "systems/system_001.png"
        ).save(path),
    )

    result = spike.prepare_fifth_score(tmp_path)

    prepared_path = Path(result["prepared_manifest"])
    prepared = base._read_json(prepared_path)
    selection = base._read_json(prepared_path.parent / "selection.json")
    context = base._read_json(prepared_path.parent / "context/allowed_context.json")
    assert prepared_path.parent == (
        tmp_path / spike.COQUETEOS_SLUG / spike.OUTPUT_SUBDIR / "v1" / "system_002"
    )
    assert prepared["kind"] == "fifth_score_fresh_heldout_prepare"
    assert prepared["target"] == {"slug": spike.COQUETEOS_SLUG, "system_index": 2}
    assert len(prepared["artifacts"]["crops"]) == 6
    assert selection["policy"]["max_spacing_cv"] == 0.35
    assert selection["policy"]["min_measure_count"] == 6
    assert selection["policy"]["max_measure_count"] == 6
    assert context["truth_accessed"] is False
    assert context["truth_used"] is False
    assert context["allowed_context"] == {
        "allow_pickup": False,
        "clef": "treble",
        "expected_measure_beats": 3.0,
        "key_hint": None,
        "time_signature": "3/4",
    }
    assert prepared["forbidden_truth_paths"] == base._forbidden_truth_paths(spike.COQUETEOS_SLUG)
    base._verify_prepared_manifest(
        prepared_path.parent,
        prepared_path,
        prepared,
        expected_kind=spike.FIFTH_SCORE_GATE.prepare_kind,
    )


def test_prepare_rejects_changed_conclusive_key_before_creating_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _system_image(tmp_path, index=1)
    _system_image(tmp_path, index=2)
    _metadata(tmp_path)
    monkeypatch.setattr(
        spike.key_detector,
        "detect_signature",
        lambda _path, mode: {
            "mode": mode,
            "fifths": -1,
            "gate_passed": True,
            "truth_used_for_prediction": False,
        },
    )

    with pytest.raises(ValueError, match="inconclusive visual key"):
        spike.prepare_fifth_score(tmp_path)

    assert not (tmp_path / spike.COQUETEOS_SLUG / spike.OUTPUT_SUBDIR).exists()


def test_freeze_uses_fifth_score_artifact_kinds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundaries = tuple(index / 6 for index in range(7))
    monkeypatch.setattr(base, "detect_barlines", lambda _path: list(boundaries))
    monkeypatch.setattr(
        base,
        "measure_boundaries_for_system",
        lambda _path, _detected: list(boundaries),
    )
    initial = _system_image(tmp_path, index=1)
    _system_image(tmp_path, index=2)
    _metadata(tmp_path)
    monkeypatch.setattr(
        spike.key_detector,
        "detect_signature",
        lambda _path, mode: {
            "mode": mode,
            "fifths": None,
            "gate_passed": False,
            "truth_used_for_prediction": False,
        },
    )
    monkeypatch.setattr(
        spike.key_detector,
        "_draw_overlay",
        lambda _prediction, path: Image.open(initial).save(path),
    )
    prepared = spike.prepare_fifth_score(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    predictions = artifacts / "predictions.jsonl"
    model = artifacts / "model.json"
    training = artifacts / "training.json"
    predictions.write_text('{"notes":[]}\n', encoding="utf-8")
    model.write_text("{}\n", encoding="utf-8")
    training.write_text("{}\n", encoding="utf-8")

    frozen = spike.freeze_prepared_fifth_score(
        Path(prepared["prepared_manifest"]),
        predictions_path=predictions,
        model_artifact_paths=(model,),
        training_artifact_paths=(training,),
    )

    freeze = base._read_json(Path(frozen["freeze"]))
    sealed = base._read_json(Path(frozen["sealed_manifest"]))
    assert freeze["kind"] == "fifth_score_fresh_heldout_freeze"
    assert sealed["kind"] == "fifth_score_fresh_heldout_sealed_manifest"


def _system_image(root: Path, *, index: int) -> Path:
    path = root / spike.COQUETEOS_SLUG / "systems" / f"system_{index:03d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (1400, 120), "white")
    draw = ImageDraw.Draw(image)
    for y in (35, 45, 55, 65, 75):
        draw.line((0, y, 1399, y), fill="black", width=2)
    for measure in range(6):
        x = measure * 230 + 115
        draw.ellipse((x - 7, 50, x + 7, 58), fill="black")
        draw.line((x + 7, 54, x + 7, 31), fill="black", width=2)
    image.save(path)
    return path


def _metadata(root: Path) -> None:
    path = root / spike.COQUETEOS_SLUG / "metadata.json"
    path.write_text(
        json.dumps(
            {
                "title": "Coqueteos",
                "composer": "Fulgencio Garcia",
                "rhythm": "Pasillo",
                "time_signature": None,
                "key_hint": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
