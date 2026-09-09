from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.experiments import evaluate_hollow_notehead_unseen_gate as evaluator
from scripts.experiments import review_hollow_notehead_unseen_gate as reviewer


def test_promotes_precise_recovery_on_sufficient_heldout_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _build_gate(
        tmp_path,
        candidates=[(10.0, 20.0), (30.0, 20.0), (50.0, 20.0)],
        proposals=[(70.0, 20.0), (90.0, 20.0)],
    )
    app = reviewer.load_review_app(manifest_path)
    app.save(
        0,
        _review_payload(
            (10.0, 20.0),
            (30.0, 20.0),
            (50.0, 20.0),
            (70.0, 20.0),
            (90.0, 20.0),
        ),
    )
    monkeypatch.setattr(evaluator.freezer, "verify_sealed_manifest", lambda _: {})

    output = evaluator.evaluate_hollow_notehead_unseen_gate(manifest_path)
    report = _json(output / "report.json")

    assert report["promotion_gate"]["decision"] == "promote"
    assert report["promotion_gate"]["eligible_for_candidate_pipeline_integration"] is True
    assert report["summary"] == {
        "measure_count": 1,
        "truth_notehead_count": 5,
        "baseline_matched_truth_count": 3,
        "baseline_recall": 0.6,
        "recovery_opportunity_count": 2,
        "considered_pair_count": 0,
        "frozen_proposal_count": 2,
        "recovered_truth_count": 2,
        "duplicate_proposal_count": 0,
        "false_proposal_count": 0,
        "proposal_precision": 1.0,
        "recovery_rate": 1.0,
        "augmented_matched_truth_count": 5,
        "augmented_recall": 1.0,
    }
    assert (output / "overlays/measure_001_evaluation.png").is_file()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        evaluator.evaluate_hollow_notehead_unseen_gate(manifest_path)


def test_does_not_promote_undersized_gate_without_recovery_opportunity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _build_gate(
        tmp_path,
        candidates=[(20.0, 30.0), (80.0, 30.0)],
        proposals=[],
    )
    reviewer.load_review_app(manifest_path).save(
        0,
        _review_payload((20.0, 30.0), (80.0, 30.0)),
    )
    monkeypatch.setattr(evaluator.freezer, "verify_sealed_manifest", lambda _: {})

    output = evaluator.evaluate_hollow_notehead_unseen_gate(manifest_path)
    report = _json(output / "report.json")

    assert report["promotion_gate"]["decision"] == "not_promoted"
    assert report["promotion_gate"]["eligible_for_candidate_pipeline_integration"] is False
    assert report["promotion_gate"]["checks"]["sample_sufficient"] is False
    assert report["promotion_gate"]["checks"]["has_recovery_opportunity"] is False
    assert report["summary"]["baseline_recall"] == 1.0
    assert report["summary"]["proposal_precision"] is None
    assert report["summary"]["recovery_rate"] is None
    assert report["promotion_gate"]["reasons"] == [
        "review contains fewer than the minimum independent hollow noteheads",
        "baseline candidates already cover all reviewed hollow noteheads",
        "frozen rule emitted no heldout proposals",
    ]


def test_refuses_incomplete_human_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _build_gate(tmp_path, candidates=[], proposals=[])
    monkeypatch.setattr(evaluator.freezer, "verify_sealed_manifest", lambda _: {})

    with pytest.raises(ValueError, match="incomplete for measures"):
        evaluator.evaluate_hollow_notehead_unseen_gate(manifest_path)


def _build_gate(
    tmp_path: Path,
    *,
    candidates: list[tuple[float, float]],
    proposals: list[tuple[float, float]],
) -> Path:
    gate_dir = tmp_path / "gate"
    source_dir = gate_dir / "measures/measure_001"
    source_dir.mkdir(parents=True)
    raw_path = source_dir / "raw.png"
    Image.new("RGB", (120, 80), "white").save(raw_path)
    identity = {
        "slug": "heldout-example",
        "system_index": 4,
        "system_measure_index": 1,
    }
    candidate_path = source_dir / "candidates.json"
    _write_json(
        candidate_path,
        {
            "kind": "vlm_notehead_candidates",
            "identity": identity,
            "staff_spacing_px": 20.0,
            "candidate_count": len(candidates),
            "candidates": [
                {
                    "id": f"c{index:03d}",
                    "center": {"x": x, "y": y},
                    "bbox": {
                        "left": x - 4,
                        "top": y - 4,
                        "right": x + 4,
                        "bottom": y + 4,
                    },
                }
                for index, (x, y) in enumerate(candidates, start=1)
            ],
        },
    )
    proposal_path = source_dir / "proposals.json"
    _write_json(
        proposal_path,
        {
            "kind": "vlm_melody_hollow_notehead_unseen_gate_proposals",
            "identity": identity,
            "considered_pair_count": 0,
            "proposal_count": len(proposals),
            "proposals": [
                {
                    "center": {"x": x, "y": y},
                    "support_candidate_ids": [],
                    "score": 1.0,
                }
                for x, y in proposals
            ],
        },
    )
    manifest_path = gate_dir / "sealed_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "kind": reviewer.MANIFEST_KIND,
            "status": "frozen_awaiting_human_review",
            "split": "fresh_heldout_morphology",
            "create_once": True,
            "truth_accessed": False,
            "measure_count": 1,
            "measures": [
                {
                    "identity": identity,
                    "raw_image": _record(raw_path, gate_dir),
                    "candidate_artifact": _record(candidate_path, gate_dir),
                    "proposal_artifact": _record(proposal_path, gate_dir),
                }
            ],
            "provenance": {
                "ground_truth_files_read": [],
                "review_files_read": [],
                "musicxml_files_read": [],
            },
        },
    )
    return manifest_path


def _review_payload(*centers: tuple[float, float]) -> dict[str, object]:
    return {
        "centers": [{"x": x, "y": y} for x, y in centers],
        "completion_confirmed": True,
    }


def _record(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
