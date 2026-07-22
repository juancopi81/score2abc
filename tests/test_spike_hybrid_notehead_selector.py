from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.experiments import spike_hybrid_notehead_selector as spike
from scripts.experiments import spike_notehead_patch_templates as patch_spike
from scripts.experiments import spike_stem_endpoint_detector as stem_spike


def _grid_candidate(
    candidate_id: str, x: float, y: float, *, rank: int = 1
) -> patch_spike.CandidatePatch:
    return patch_spike.CandidatePatch(
        measure=1,
        id=candidate_id,
        rank=rank,
        center_x=x,
        center_y=y,
        bbox=(round(x) - 3, round(y) - 2, round(x) + 4, round(y) + 3),
        detector_score=0.8,
        patches={spike.PATCH_ID: (0.0,) * (patch_spike.PATCH_WIDTH * patch_spike.PATCH_HEIGHT)},
    )


def _stem_candidate(x: float, y: float, score: float = 0.9) -> stem_spike.EndpointCandidate:
    return stem_spike.EndpointCandidate(
        x=x,
        y=y,
        score=score,
        stem_x=x + 2,
        stem_top=5,
        stem_bottom=25,
        endpoint="bottom",
        pitch_midi=64,
        staff_position=0,
    )


def test_union_nms_merges_grid_stem_duplicates_and_preserves_distinct_rows() -> None:
    image = Image.new("L", (80, 50), 255)
    result = spike.build_candidate_union(
        [_grid_candidate("c001", 10, 20)],
        [_stem_candidate(11, 21), _stem_candidate(50, 20)],
        image=image,
        suppressed_image=image,
        staff_spacing=10,
        threshold=127,
    )

    assert len(result) == 2
    assert (result[0].center_x, result[0].center_y) == (10, 20)
    assert result[0].grid_ids == ("c001",)
    assert result[0].stem_ids == ("stem001",)
    assert result[0].agreement is True
    assert result[1].has_grid is False
    assert result[1].has_stem is True


def test_provenance_features_distinguish_union_intersection() -> None:
    candidate = spike.HybridCandidate(
        id="u001",
        center_x=10,
        center_y=20,
        bbox=(5, 15, 16, 26),
        patch=(0.0,) * (patch_spike.PATCH_WIDTH * patch_spike.PATCH_HEIGHT),
        grid_ids=("c003",),
        stem_ids=("stem002",),
        grid_rank=3,
        grid_score=0.75,
        stem_score=0.88,
        agreement_distance=0.2,
    )

    features = spike.hybrid_features(candidate, patch_knn_score=0.4)

    assert features["has_grid"] == 1.0
    assert features["has_stem"] == 1.0
    assert features["agreement"] == 1.0
    assert features["agreement_distance"] == 0.2
    assert features["patch_knn_score"] == 0.4


def test_loocv_splits_never_train_on_held_measure() -> None:
    splits = spike.loocv_splits((1, 2, 3, 4))

    assert len(splits) == 4
    assert splits[0] == ((2, 3, 4), 1)
    assert all(held not in training for training, held in splits)
    assert {held for _, held in splits} == {1, 2, 3, 4}
    with pytest.raises(ValueError, match="unique measures"):
        spike.loocv_splits((1, 1, 2))


def test_prediction_seal_is_required_before_truth_access(tmp_path: Path) -> None:
    gate = spike.PredictionTruthGate()
    truth_path = tmp_path / "truth.jsonl"
    truth_path.write_text('{"identity":{}}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="sealed validation predictions"):
        gate.read_truth("validation", truth_path)

    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text("{}\n", encoding="utf-8")
    seal = gate.seal_predictions("validation", predictions_path)

    assert gate.read_truth("validation", truth_path) == [{"identity": {}}]
    assert gate.access_log == [
        {
            "split": "validation",
            "truth_path": str(truth_path),
            "after_prediction_sha256": seal["sha256"],
        }
    ]


def test_deterministic_union_and_jsonl_output(tmp_path: Path) -> None:
    image = Image.new("L", (80, 50), 255)

    def build() -> tuple[spike.HybridCandidate, ...]:
        return spike.build_candidate_union(
            [_grid_candidate("c002", 30, 20, rank=2), _grid_candidate("c001", 10, 20)],
            [_stem_candidate(31, 20), _stem_candidate(11, 21)],
            image=image,
            suppressed_image=image,
            staff_spacing=10,
            threshold=127,
        )

    first = build()
    second = build()
    first_payload = [
        {
            "id": row.id,
            "center": [row.center_x, row.center_y],
            "grid_ids": row.grid_ids,
            "stem_ids": row.stem_ids,
        }
        for row in first
    ]
    second_payload = [
        {
            "id": row.id,
            "center": [row.center_x, row.center_y],
            "grid_ids": row.grid_ids,
            "stem_ids": row.stem_ids,
        }
        for row in second
    ]
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    spike._write_jsonl(first_path, first_payload)
    spike._write_jsonl(second_path, second_payload)

    assert first_payload == second_payload
    assert first_path.read_bytes() == second_path.read_bytes()
    assert (
        hashlib.sha256(first_path.read_bytes()).hexdigest()
        == hashlib.sha256(second_path.read_bytes()).hexdigest()
    )
    assert [row["id"] for row in map(json.loads, first_path.read_text().splitlines())] == [
        "u001",
        "u002",
    ]
