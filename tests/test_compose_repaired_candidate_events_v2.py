from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.experiments import compose_repaired_candidate_events_v2 as compositor


def test_applies_full_measure_dotted_half_to_exact_sparse_dyad(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        compositor.v1,
        "compose_repaired_candidate_events",
        lambda *args, **kwargs: _base_result(status="not_materialized_request_only_meter_fallback"),
    )

    result = compositor.compose_repaired_candidate_events_v2(
        _request(),
        {"truth_used": False},
        _candidate_lane(),
        _accepted_decision(),
        pitch_predictor=object(),
        out_dir=tmp_path,
        selector_method_id="selector-v1",
    )

    assert result["status"] == compositor.v1.STATUS_MATERIALIZED
    assert result["meter_valid"] is True
    assert result["durations"] == [3.0, 3.0]
    assert result["onsets"] == [0.0, 0.0]
    assert result["rests"] == []
    assert result["prediction"]["notes"] == [
        {"duration_beats": 3.0, "onset_beats": 0.0, "pitch_midi": 60},
        {"duration_beats": 3.0, "onset_beats": 0.0, "pitch_midi": 64},
    ]
    assert result["prediction"]["rhythm_tokens"] == [
        {"duration_beats": 3.0, "kind": "note", "note_count": 2, "onset_beats": 0.0}
    ]
    assert result["duration_evidence"]["applied"] is True
    assert result["v1_base_composition"]["status"].startswith("not_materialized")
    assert result["inference_provenance"]["meter_fallback"]["applied"] is False


def test_nonaccepted_sparse_decision_replays_v1_without_rhythm_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = _base_result(status=compositor.v1.STATUS_MATERIALIZED)
    monkeypatch.setattr(
        compositor.v1,
        "compose_repaired_candidate_events",
        lambda *args, **kwargs: base,
    )

    result = compositor.compose_repaired_candidate_events_v2(
        _request(),
        {"truth_used": False},
        _candidate_lane(),
        {"accepted": False, "reason": "no_pair", "truth_used": False},
        pitch_predictor=object(),
        out_dir=tmp_path,
        selector_method_id="selector-v1",
    )

    assert result["prediction"] == base["prediction"]
    assert result["decoder_status"] == base["decoder_status"]
    assert result["duration_evidence"] == {
        "schema_version": 1,
        "kind": "sparse_shared_stem_dotted_half",
        "applied": False,
        "reason": "sparse_repair_not_accepted",
        "truth_used": False,
    }


def test_applied_evidence_rejects_group_identity_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = _base_result(status=compositor.v1.STATUS_MATERIALIZED)
    base["groups"][0]["anchors"][1]["source"]["candidate_id"] = "d999"
    monkeypatch.setattr(
        compositor.v1,
        "compose_repaired_candidate_events",
        lambda *args, **kwargs: base,
    )

    with pytest.raises(ValueError, match="candidate identities drifted"):
        compositor.compose_repaired_candidate_events_v2(
            _request(),
            {"truth_used": False},
            _candidate_lane(),
            _accepted_decision(),
            pitch_predictor=object(),
            out_dir=tmp_path,
            selector_method_id="selector-v1",
        )


def _request() -> dict[str, Any]:
    return {
        "identity": {
            "slug": "synthetic-score",
            "system_index": 1,
            "system_measure_index": 1,
            "automatic_measure_index": 1,
        },
        "allowed_context": {"expected_measure_beats": "3"},
    }


def _candidate_lane() -> list[dict[str, Any]]:
    return [
        {"candidate_id": "d001", "onset_group_index": 1},
        {"candidate_id": "d002", "onset_group_index": 1},
    ]


def _accepted_decision() -> dict[str, Any]:
    return {
        "accepted": True,
        "reason": "accepted",
        "truth_used": False,
        "chosen_pair": {
            "candidate_ids": ["d001", "d002"],
            "augmentation_dot_pairs": [{"candidate_ids": ["d050", "d051"]}],
        },
    }


def _base_result(*, status: str) -> dict[str, Any]:
    identity = _request()["identity"]
    prediction = {
        "identity": identity,
        "notes": [
            {"duration_beats": 1.0, "onset_beats": 0.0, "pitch_midi": 60},
            {"duration_beats": 1.0, "onset_beats": 0.0, "pitch_midi": 64},
        ],
        "rests": [],
        "rhythm_tokens": [
            {"duration_beats": 1.0, "kind": "note", "note_count": 2, "onset_beats": 0.0}
        ],
        "measure_extent_beats": 1.0,
        "decoder_status": "v1-synthetic",
    }
    return {
        "schema_version": 1,
        "identity": identity,
        "status": status,
        "prediction": prediction,
        "groups": [
            {
                "group_id": "g001",
                "onset_group_index": 1,
                "center_x": 35.5,
                "pitches": ["C4", "E4"],
                "anchors": [
                    {
                        "source": {"candidate_id": "d001", "onset_group_index": 1},
                    },
                    {
                        "source": {"candidate_id": "d002", "onset_group_index": 1},
                    },
                ],
            }
        ],
        "automatic_key_context": {"truth_used": False},
        "notes": prediction["notes"],
        "onsets": [0.0, 0.0],
        "durations": [1.0, 1.0],
        "rests": [],
        "decoder_status": "v1-synthetic",
        "expected_measure_beats": 3.0,
        "observed_extent_beats": 1.0,
        "decoded_extent_beats": 1.0,
        "meter_valid": status == compositor.v1.STATUS_MATERIALIZED,
        "anchor_features": [],
        "residual_rest_features": [],
        "visual_symbols": [],
        "decoded_symbols": [],
        "inference_provenance": {"truth_used": False},
    }
