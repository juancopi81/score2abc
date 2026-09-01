import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from scripts.experiments import freeze_independent_full_event_gate as gate
from scripts.experiments import freeze_third_score_heldout as base
from scripts.experiments import run_third_score_heldout_inference as inference
from scripts.experiments import seal_independent_full_event_gate as sealer


def test_prepares_fixed_eight_crop_full_event_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundaries = [index / gate.EXPECTED_CROP_COUNT for index in range(9)]
    monkeypatch.setattr(base, "detect_barlines", lambda _path: boundaries)
    monkeypatch.setattr(base, "measure_boundaries_for_system", lambda _path, _raw: boundaries)
    monkeypatch.setattr(
        gate,
        "_detect_initial_signature",
        lambda _path: {"kind": "synthetic_visual_key_prediction"},
    )
    monkeypatch.setattr(
        gate.strict_key,
        "strict_initial_key_state",
        lambda _prediction, source_system_index: {
            "status": "unknown",
            "source_system_index": source_system_index,
        },
    )
    _write_target(tmp_path)

    result = gate.prepare_independent_full_event_gate(tmp_path, namespace="test-v1")

    prepared_path = Path(result["prepared_manifest"])
    prepared = base._read_json(prepared_path)
    context = base._read_json(prepared_path.parent / "context/allowed_context.json")
    evaluator = base._read_json(prepared_path.parent / "evaluator_spec.json")
    assert prepared["target"] == {
        "slug": gate.TIO_CLIMACO_SLUG,
        "system_index": gate.TARGET_SYSTEM_INDEX,
    }
    assert len(prepared["artifacts"]["crops"]) == gate.EXPECTED_CROP_COUNT
    assert context["allowed_context"] == {
        "allow_pickup": False,
        "clef": "treble",
        "expected_measure_beats": "3",
        "key_hint": gate.EXPECTED_KEY_HINT,
        "time_signature": gate.EXPECTED_TIME_SIGNATURE,
    }
    assert context["provenance"]["visual_key_diagnostic"]["accepted_as_context"] is False
    assert evaluator["supported_metrics"] == [
        "candidate_recovery",
        "note_count",
        "chromatic_pitch",
        "onset",
        "duration",
        "rests",
        "meter_validity",
        "exact_measure",
    ]
    base._verify_prepared_manifest(
        prepared_path.parent,
        prepared_path,
        prepared,
        expected_kind=gate.INDEPENDENT_FULL_EVENT_GATE.prepare_kind,
    )
    config = inference.GATE_CONFIGS[gate.INDEPENDENT_FULL_EVENT_GATE.prepare_kind]
    assert config["inference_version"] == "independent-full-event-baseline-inference-v1"


def test_preparation_rejects_unpinned_context(tmp_path: Path) -> None:
    _write_target(tmp_path, key_hint="unknown")

    with pytest.raises(ValueError, match="one-flat metadata"):
        gate.prepare_independent_full_event_gate(tmp_path)


def test_seal_snapshots_complete_chain_and_detects_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated = _write_upstream_manifests(tmp_path)
    monkeypatch.setattr(sealer, "_validate_upstream", lambda *_args, **_kwargs: validated)

    result = sealer.seal_independent_full_event_gate(
        validated["repaired_full_event_manifest_path"].parent,
        model_dir=validated["model_manifest_path"].parent,
    )

    sealed_path = Path(result["sealed_manifest"])
    verified = sealer.verify_independent_full_event_gate(
        sealed_path,
        model_dir=validated["model_manifest_path"].parent,
    )
    assert verified["verified"] is True
    assert verified["target"] == {
        "slug": gate.TIO_CLIMACO_SLUG,
        "system_index": gate.TARGET_SYSTEM_INDEX,
    }
    freeze = base._read_json(sealed_path.parent / "freeze.json")
    assert set(freeze["snapshots"]) == {
        "prepared_manifest",
        "inference_manifest",
        "multihead_manifest",
        "sparse_manifest",
        "repaired_full_event_manifest",
        "model_manifest",
    }
    assert freeze["contract"]["repaired_full_events_frozen"] is True
    with pytest.raises(FileExistsError, match="overwrite"):
        sealer.seal_independent_full_event_gate(
            validated["repaired_full_event_manifest_path"].parent,
            model_dir=validated["model_manifest_path"].parent,
        )

    validated["sparse_manifest_path"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Full-event sparse_manifest"):
        sealer.verify_independent_full_event_gate(
            sealed_path,
            model_dir=validated["model_manifest_path"].parent,
        )


def test_seal_rejects_musicxml_before_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated = _write_upstream_manifests(tmp_path)
    monkeypatch.setattr(sealer, "_validate_upstream", lambda *_args, **_kwargs: validated)
    truth_path = validated["prepared_manifest_path"].parent / "premature.musicxml"
    truth_path.write_text("<score-partwise/>\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exists before"):
        sealer.seal_independent_full_event_gate(
            validated["repaired_full_event_manifest_path"].parent,
            model_dir=validated["model_manifest_path"].parent,
        )


def _write_target(root: Path, *, key_hint: str = gate.EXPECTED_KEY_HINT) -> None:
    score_root = root / gate.TIO_CLIMACO_SLUG
    systems_dir = score_root / "systems"
    systems_dir.mkdir(parents=True)
    for index in (1, gate.TARGET_SYSTEM_INDEX):
        image = Image.new("L", (1600, 140), "white")
        draw = ImageDraw.Draw(image)
        for y in (45, 55, 65, 75, 85):
            draw.line((0, y, 1599, y), fill="black", width=2)
        image.save(systems_dir / f"system_{index:03d}.png")
    (score_root / "metadata.json").write_text(
        json.dumps(
            {
                "title": "Tío Clímaco",
                "composer": "Bonifacio Bautista",
                "rhythm": "Pasillo",
                "time_signature": gate.EXPECTED_TIME_SIGNATURE,
                "key_hint": key_hint,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_upstream_manifests(root: Path) -> dict[str, Path | dict[str, object]]:
    namespace_root = (
        root / gate.TIO_CLIMACO_SLUG / gate.OUTPUT_SUBDIR / gate.DEFAULT_NAMESPACE / "system_007"
    )
    inference_dir = namespace_root / "full_event_inference_v1"
    model_dir = root / "model"
    paths = {
        "prepared_manifest_path": namespace_root / "prepared_manifest.json",
        "inference_manifest_path": inference_dir / "manifest.json",
        "multihead_manifest_path": inference_dir
        / inference.MULTIHEAD_RECOVERY_DIRNAME
        / "manifest.json",
        "sparse_manifest_path": inference_dir
        / inference.SPARSE_DYAD_REPAIR_DIRNAME
        / "manifest.json",
        "repaired_full_event_manifest_path": inference_dir
        / "repaired_full_event_v1"
        / "manifest.json",
        "model_manifest_path": model_dir / "manifest.json",
    }
    for role, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"role": role}) + "\n", encoding="utf-8")
    return {
        "target": {
            "slug": gate.TIO_CLIMACO_SLUG,
            "system_index": gate.TARGET_SYSTEM_INDEX,
        },
        **paths,
    }
