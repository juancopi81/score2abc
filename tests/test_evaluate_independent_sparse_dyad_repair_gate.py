from pathlib import Path
from typing import Any

import pytest

from scripts.experiments import evaluate_independent_sparse_dyad_repair_gate as evaluator


def test_raw_review_confirms_proposed_heads_and_one_frozen_dot_pair() -> None:
    result = evaluator.validate_raw_image_review(_raw_review(), frozen=_frozen_review_fixture())

    crop = result["crops"][0]
    assert result["head_pixel_identity_passed"] is True
    assert result["augmentation_dot_evidence_passed"] is True
    assert set(crop["proposed_head_matches"]) == {"head_upper", "head_lower"}
    assert crop["displaced_head_matches"] == {}
    assert crop["confirmed_augmentation_dot_pairs"] == [
        {
            "candidate_ids": ["dot_upper", "dot_lower"],
            "matched_candidate_ids": ["dot_lower", "dot_upper"],
        }
    ]
    assert crop["unconfirmed_augmentation_dot_pairs"] == [
        {
            "candidate_ids": ["noise_dot_upper", "noise_dot_lower"],
            "matched_candidate_ids": [],
        }
    ]


def test_raw_review_rejects_candidate_overlay_exposure() -> None:
    review = _raw_review()
    review["automatic_overlay_visible"] = True

    with pytest.raises(ValueError, match="must not expose automatic overlays"):
        evaluator.validate_raw_image_review(review, frozen=_frozen_review_fixture())


def test_raw_review_rejects_displaced_anchor_labeled_as_notehead() -> None:
    review = _raw_review()
    review["measures"][0]["notehead_centers"].append({"x": 10.0, "y": 80.0})

    with pytest.raises(ValueError, match="displaced anchors as noteheads"):
        evaluator.validate_raw_image_review(review, frozen=_frozen_review_fixture())


def test_frozen_verification_precedes_raw_review_and_musicxml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = False

    def fail_verification(_path: Path) -> dict[str, Any]:
        raise ValueError("frozen drift")

    def truth_loader(_path: Path) -> evaluator.heldout.VisibleMusicXMLTruth:
        nonlocal opened
        opened = True
        raise AssertionError("truth must not open")

    monkeypatch.setattr(evaluator, "verify_frozen_sparse_dyad_gate", fail_verification)

    with pytest.raises(ValueError, match="frozen drift"):
        evaluator.evaluate_independent_sparse_dyad_repair_gate(
            tmp_path / "sealed_manifest.json",
            musicxml_path=tmp_path / "truth.musicxml",
            raw_review_path=tmp_path / "review.json",
            truth_loader=truth_loader,
        )

    assert opened is False


def test_help_names_expected_desde_lejos_inputs(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        evaluator.main(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert evaluator.EXPECTED_TRANSCRIPTION_PATH.as_posix() in output
    assert evaluator.EXPECTED_RAW_REVIEW_PATH.as_posix() in output


def test_runner_evaluation_surface_allows_only_optional_materialization_changes(
    tmp_path: Path,
) -> None:
    frozen = tmp_path / "frozen.py"
    current = tmp_path / "current.py"
    frozen.write_text(
        """\
BASELINE = 1

def materialize_third_score_inference():
    return 'old'

def _read_json():
    return 'stable'
""",
        encoding="utf-8",
    )
    current.write_text(
        """\
BASELINE = 1
SPARSE_DYAD_REPAIR_DIRNAME = 'sidecar'
INDEPENDENT_FULL_EVENT_INFERENCE_VERSION = 'new-gate'
GATE_CONFIGS = {'new': 'gate'}

from scripts.experiments import freeze_independent_full_event_gate

def materialize_third_score_inference():
    return 'new optional lane'

def _sparse_dyad_repair_row():
    return 'new'

def _multihead_recovery_row():
    return 'new'

def _verify_multihead_baseline():
    return 'new'

def _read_json():
    return 'stable'
""",
        encoding="utf-8",
    )

    assert evaluator._runner_evaluation_surface(current) == evaluator._runner_evaluation_surface(
        frozen
    )

    current.write_text(current.read_text(encoding="utf-8").replace("'stable'", "'drift'"))
    assert evaluator._runner_evaluation_surface(current) != evaluator._runner_evaluation_surface(
        frozen
    )


def _raw_review() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": evaluator.RAW_REVIEW_KIND,
        "status": evaluator.RAW_REVIEW_STATUS,
        "review_mode": "raw_image_only",
        "automatic_overlay_visible": False,
        "musicxml_visible": False,
        "target": {
            "slug": evaluator.TARGET_SLUG,
            "system_index": evaluator.TARGET_SYSTEM_INDEX,
        },
        "measures": [
            {
                "automatic_crop_index": 2,
                "raw_image_sha256": "a" * 64,
                "head_tolerance_px": 5.0,
                "dot_tolerance_px": 5.0,
                "notehead_centers": [{"x": 20.0, "y": 30.0}, {"x": 20.0, "y": 40.0}],
                "augmentation_dot_centers": [
                    {"x": 35.0, "y": 30.0},
                    {"x": 35.0, "y": 40.0},
                ],
            }
        ],
    }


def _frozen_review_fixture() -> dict[str, Any]:
    candidate_predictions = [
        _candidate("head_upper", 20, 30),
        _candidate("head_lower", 20, 40),
        _candidate("dot_upper", 35, 30),
        _candidate("dot_lower", 35, 40),
        _candidate("noise_dot_upper", 48, 30),
        _candidate("noise_dot_lower", 48, 40),
        _candidate("displaced_a", 10, 80),
        _candidate("displaced_b", 30, 80),
    ]
    return {
        "target": {
            "slug": evaluator.TARGET_SLUG,
            "system_index": evaluator.TARGET_SYSTEM_INDEX,
        },
        "diagnostics_by_crop": {
            2: {
                "sparse_repair": {
                    "accepted": True,
                    "proposed_ids": ["head_upper", "head_lower"],
                    "current_ids": ["displaced_a", "displaced_b"],
                    "chosen_pair": {
                        "augmentation_dot_pairs": [
                            {"candidate_ids": ["dot_upper", "dot_lower"]},
                            {"candidate_ids": ["noise_dot_upper", "noise_dot_lower"]},
                        ]
                    },
                }
            }
        },
        "generic_rows_by_crop": {
            2: {
                "source": {"sha256": "a" * 64},
                "candidate_predictions": candidate_predictions,
            }
        },
    }


def _candidate(candidate_id: str, x: float, y: float) -> dict[str, Any]:
    return {"candidate_id": candidate_id, "center": {"x": x, "y": y}}
