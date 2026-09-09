import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from scripts.experiments import freeze_independent_sparse_dyad_repair_gate as gate
from scripts.experiments import freeze_third_score_heldout as freezer
from scripts.experiments import run_independent_multihead_recovery_gate as multihead
from scripts.experiments import run_independent_sparse_dyad_repair_gate as runner
from scripts.experiments import run_third_score_heldout_inference as inference
from scripts.experiments import spike_consumed_polyphonic_pitch_repair as recovery
from scripts.experiments import spike_consumed_sparse_stem_dyad_repair as repair


def test_gate_contract_is_fixed_and_registered() -> None:
    assert gate.DESDE_LEJOS_SLUG == "jaime-llanos_26_desde-lejos_pasillo_b-b"
    assert gate.TARGET_SYSTEM_INDEX == 7
    assert gate.EXPECTED_CROP_COUNT == 10
    assert gate.DEFAULT_LAYOUT_POLICY.min_crop_width_px == 140
    config = inference.GATE_CONFIGS[gate.INDEPENDENT_SPARSE_DYAD_REPAIR_GATE.prepare_kind]
    assert config["inference_version"] == "independent-sparse-dyad-baseline-inference-v1"


def test_preparation_seals_ten_crops_and_unknown_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score_root = tmp_path / gate.DESDE_LEJOS_SLUG
    system_path = score_root / "systems/system_007.png"
    system_path.parent.mkdir(parents=True)
    image = Image.new("L", (2000, 160), "white")
    draw = ImageDraw.Draw(image)
    for y in (40, 50, 60, 70, 80):
        draw.line((0, y, 1999, y), fill="black", width=2)
    image.save(system_path)
    (score_root / "metadata.json").write_text(
        json.dumps({"title": "Desde Lejos", "time_signature": None, "key_hint": None}) + "\n",
        encoding="utf-8",
    )
    boundaries = [index / gate.EXPECTED_CROP_COUNT for index in range(11)]
    monkeypatch.setattr(freezer, "detect_barlines", lambda _path: boundaries)
    monkeypatch.setattr(
        freezer,
        "measure_boundaries_for_system",
        lambda _path, _detected: boundaries,
    )

    result = gate.prepare_independent_sparse_dyad_repair_gate(tmp_path)

    prepared_path = Path(result["prepared_manifest"])
    prepared = freezer._read_json(prepared_path)
    assert prepared["target"] == {
        "slug": gate.DESDE_LEJOS_SLUG,
        "system_index": gate.TARGET_SYSTEM_INDEX,
    }
    assert len(prepared["artifacts"]["crops"]) == gate.EXPECTED_CROP_COUNT
    declaration = prepared["independent_sparse_dyad_repair_gate"]
    assert declaration["config_id"] == repair.CONFIG_ID
    assert declaration["parameters"] == repair.PARAMETERS
    assert declaration["truth_used"] is False
    context = freezer._read_json(prepared_path.parent / "context/allowed_context.json")
    assert context["allowed_context"] == {
        "clef": "treble",
        "time_signature": None,
        "key_hint": None,
        "expected_measure_beats": None,
        "allow_pickup": False,
    }


def test_pair_row_replaces_outside_staff_noise_with_dotted_shared_stem_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _row()
    candidates = repair._normalized_candidates(row)
    by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}
    current = [by_id["noise_a"], by_id["noise_b"]]
    monkeypatch.setattr(
        multihead,
        "_pair_row",
        lambda _row, selector, expected_target: (
            _multihead_row(_row, current),
            {"truth_used": False},
            current,
            [],
        ),
    )
    monkeypatch.setattr(
        recovery,
        "candidate_local_stem_features",
        lambda _row: (_stem_features(), {"kind": "synthetic"}),
    )

    paired, diagnostics, prior, repaired = runner._pair_row(
        row,
        selector=_selector(),
        expected_target={"slug": gate.DESDE_LEJOS_SLUG, "system_index": 7},
    )

    assert [item["candidate_id"] for item in prior] == ["noise_a", "noise_b"]
    assert [item["candidate_id"] for item in repaired] == ["head_upper", "head_lower"]
    assert diagnostics["sparse_repair"]["accepted"] is True
    lane = paired["lanes"]["sparse_dyad_repair"]
    assert {item["onset_group_index"] for item in lane["candidate_lane"]} == {1}
    assert lane["displaced_candidate_ids"] == ["noise_a", "noise_b"]
    assert lane["added_candidate_ids"] == ["head_lower", "head_upper"]


def test_freeze_verifier_rejects_snapshot_drift(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen"
    snapshot = frozen / "artifacts/paired_predictions/001_predictions.jsonl"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("{}\n", encoding="utf-8")
    pin = {
        "snapshot_path_relative_to_namespace": (
            "frozen/artifacts/paired_predictions/001_predictions.jsonl"
        ),
        "snapshot_sha256": _sha256(snapshot),
    }
    roles = {
        role: [pin]
        for role in (
            "paired_predictions",
            "paired_artifacts",
            "prepared_and_source",
            "baseline_inference",
            "model_and_training",
            "implementations",
        )
    }
    freeze_path = frozen / "freeze.json"
    inference._write_json(
        freeze_path,
        {
            "status": "frozen_awaiting_truth",
            "truth_accessed": False,
            "repair_config_id": repair.CONFIG_ID,
            "repair_parameters": repair.PARAMETERS,
            "pins": roles,
        },
    )
    inference._write_json(
        frozen / "sealed_manifest.json",
        {
            "status": "frozen_awaiting_truth",
            "truth_accessed": False,
            "freeze": {"sha256": _sha256(freeze_path)},
        },
    )
    runner.verify_freeze(frozen)

    snapshot.write_text("drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot hash drift"):
        runner.verify_freeze(frozen)


def _row() -> dict:
    candidates = [
        _candidate("head_upper", 40, 30, 0.1),
        _candidate("head_lower", 41, 40, 0.1),
        _candidate("dot_upper", 56, 30, -0.1),
        _candidate("dot_lower", 56, 40, -0.1),
        _candidate("noise_a", 75, 72, 0.9),
        _candidate("noise_b", 90, 72, 0.8),
    ]
    return {
        "identity": {
            "slug": gate.DESDE_LEJOS_SLUG,
            "system_index": 7,
            "system_measure_index": 2,
            "automatic_measure_index": 2,
        },
        "truth_used": False,
        "source": {"image": "synthetic.png", "sha256": "0" * 64},
        "allowed_context": {
            "allow_pickup": False,
            "clef": "treble",
            "expected_measure_beats": None,
            "key_hint": None,
            "time_signature": None,
        },
        "decoder_status": "not_applied_missing_expected_measure_beats",
        "staff_geometry": {"raw_staff_lines_y_px": [20, 30, 40, 50, 60]},
        "candidate_predictions": candidates,
    }


def _candidate(candidate_id: str, x: float, y: float, score: float) -> dict:
    return {
        "candidate_id": candidate_id,
        "center": {"x": x, "y": y},
        "score": score,
        "detector_rank": 1,
        "bbox": {"left": x - 4, "top": y - 4, "right": x + 4, "bottom": y + 4},
    }


def _selector() -> dict:
    return {
        "threshold": 0.5,
        "nms_x_spaces": 0.85,
        "minimum_selected_count": 2,
        "maximum_selected_count": 8,
    }


def _stem_features() -> dict:
    return {
        "head_upper": {"x": 42, "score": 0.9},
        "head_lower": {"x": 42, "score": 0.4},
        "dot_upper": {"x": 56, "score": 0.1},
        "dot_lower": {"x": 56, "score": 0.1},
        "noise_a": {"x": 75, "score": 0.9},
        "noise_b": {"x": 90, "score": 0.8},
    }


def _multihead_row(row: dict, current: list[dict]) -> dict:
    return {
        "identity": row["identity"],
        "context": {
            "clef": "treble",
            "key_hint": None,
            "time_signature": None,
            "rhythm_rest_supported": False,
        },
        "lanes": {
            "multihead_recovery": {
                "candidate_lane": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "center": candidate["center"],
                        "score": candidate["score"],
                        "onset_group_index": index,
                    }
                    for index, candidate in enumerate(current, start=1)
                ],
                "notes": [],
            }
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
