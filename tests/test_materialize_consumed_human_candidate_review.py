import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from scripts.experiments import materialize_consumed_human_candidate_review as materializer


def test_materializes_complete_review_and_proposal_comparison(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    result = materializer.materialize_human_candidate_review(
        fixture["out_dir"],
        decision_fixture=fixture["decision"],
        repo_root=fixture["repo_root"],
    )

    review = _json(Path(result["review"]))
    assert review["human_reviewed"] is True
    assert review["eligible_for_spike_training"] is True
    assert [head["candidate_id"] for head in review["accepted_heads"]] == ["c001", "c003"]
    assert review["accepted_heads"][1]["automatic_pitch_corrected"] is True
    assert review["rejected_candidates"][0]["rejection_class"] == "accidental"
    comparison = review["comparison_to_automatic_proposal"]
    assert comparison["true_positive_candidate_ids"] == ["c001"]
    assert comparison["false_positive_candidate_ids"] == ["c002"]
    assert comparison["false_negative_candidate_ids"] == ["c003"]
    assert comparison["metrics"] == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
    assert Path(result["comparison_overlay"]).is_file()
    assert Path(result["manifest"]).is_file()


def test_materializes_derived_head_from_hollow_rim_candidates(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    decision = _json(fixture["decision"])
    decision["accepted_heads"] = [
        {
            "order": 1,
            "derived_from": {
                "method": "mean_candidate_centers",
                "support_candidate_ids": ["c001", "c002"],
            },
            "sounding_pitch": "F4",
            "staff_pitch": "F4",
        }
    ]
    decision["rejected_candidate_classes"][0]["candidate_ids"] = ["c003"]
    _write_json(fixture["decision"], decision)

    result = materializer.materialize_human_candidate_review(
        fixture["out_dir"],
        decision_fixture=fixture["decision"],
        repo_root=fixture["repo_root"],
    )

    review = _json(Path(result["review"]))
    assert review["schema_version"] == 2
    assert review["provenance"]["derived_head_count"] == 1
    assert review["accepted_heads"] == [
        {
            "automatic_pitch_corrected": False,
            "automatic_staff_pitch": "F4",
            "bbox": {"bottom": 60, "left": 15, "right": 75, "top": 50},
            "center": {"x": 45.0, "y": 55.0},
            "derivation_method": "mean_candidate_centers",
            "localization_kind": "derived",
            "order": 1,
            "sounding_pitch": "F4",
            "staff_pitch": "F4",
            "support_candidate_ids": ["c001", "c002"],
            "support_candidates": [
                {
                    "bbox": {"bottom": 60, "left": 15, "right": 25, "top": 50},
                    "candidate_id": "c001",
                    "candidate_score": 0.9,
                    "center": {"x": 20.0, "y": 55.0},
                },
                {
                    "bbox": {"bottom": 60, "left": 65, "right": 75, "top": 50},
                    "candidate_id": "c002",
                    "candidate_score": 0.8,
                    "center": {"x": 70.0, "y": 55.0},
                },
            ],
        }
    ]
    comparison = review["comparison_to_automatic_proposal"]
    assert comparison["matches"] == [
        {
            "automatic_candidate_id": "c001",
            "human_order": 1,
            "human_reference": "derived:1",
            "match_kind": "derived",
        }
    ]
    assert comparison["false_positive_candidate_ids"] == ["c002"]
    assert comparison["false_negative_head_references"] == []
    assert "additional rim assignments" in comparison["metric_semantics"]
    assert comparison["metrics"] == {
        "tp": 1,
        "fp": 1,
        "fn": 0,
        "precision": 0.5,
        "recall": 1.0,
        "f1": 0.666667,
    }


def test_rejects_support_candidate_reused_as_rejection(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    decision = _json(fixture["decision"])
    decision["accepted_heads"] = [
        {
            "order": 1,
            "derived_from": {
                "method": "mean_candidate_centers",
                "support_candidate_ids": ["c001", "c002"],
            },
            "sounding_pitch": "F4",
            "staff_pitch": "F4",
        }
    ]
    decision["rejected_candidate_classes"][0]["candidate_ids"] = ["c002", "c003"]
    _write_json(fixture["decision"], decision)

    with pytest.raises(ValueError, match="Invalid rejected candidate id"):
        materializer.materialize_human_candidate_review(
            fixture["out_dir"],
            decision_fixture=fixture["decision"],
            repo_root=fixture["repo_root"],
        )


def test_rejects_incomplete_candidate_partition(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    decision = _json(fixture["decision"])
    decision["rejected_candidate_classes"][0]["candidate_ids"] = []
    _write_json(fixture["decision"], decision)

    with pytest.raises(ValueError, match="must list candidate ids"):
        materializer.materialize_human_candidate_review(
            fixture["out_dir"],
            decision_fixture=fixture["decision"],
            repo_root=fixture["repo_root"],
        )


def test_rejects_stale_source_hash_before_writing(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    decision = _json(fixture["decision"])
    decision["source"]["candidate_artifact"]["sha256"] = "0" * 64
    _write_json(fixture["decision"], decision)

    with pytest.raises(ValueError, match="hash drift"):
        materializer.materialize_human_candidate_review(
            fixture["out_dir"],
            decision_fixture=fixture["decision"],
            repo_root=fixture["repo_root"],
        )


def test_create_once_refuses_to_overwrite_review(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    materializer.materialize_human_candidate_review(
        fixture["out_dir"],
        decision_fixture=fixture["decision"],
        repo_root=fixture["repo_root"],
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        materializer.materialize_human_candidate_review(
            fixture["out_dir"],
            decision_fixture=fixture["decision"],
            repo_root=fixture["repo_root"],
        )


@pytest.mark.parametrize(
    ("assignments", "message"),
    [
        ([None], "Automatic assignment 0 must be an object"),
        ([{"candidate_id": "missing"}], "Unknown automatic candidate id"),
        (
            [{"candidate_id": "c001"}, {"candidate_id": "c001"}],
            "Duplicate automatic candidate id",
        ),
    ],
)
def test_rejects_invalid_automatic_assignments(
    tmp_path: Path,
    assignments: list,
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    decision = _json(fixture["decision"])
    proposals_path = fixture["repo_root"] / decision["source"]["proposals"]["path"]
    proposals = _json(proposals_path)
    proposals["tasks"][0]["proposal"]["assignments"] = assignments
    _write_json(proposals_path, proposals)
    decision["source"]["proposals"]["sha256"] = hashlib.sha256(
        proposals_path.read_bytes()
    ).hexdigest()
    _write_json(fixture["decision"], decision)

    with pytest.raises(ValueError, match=message):
        materializer.materialize_human_candidate_review(
            fixture["out_dir"],
            decision_fixture=fixture["decision"],
            repo_root=fixture["repo_root"],
        )


@pytest.mark.parametrize(("measure", "accepted_count"), [(3, 1), (5, 6), (7, 1)])
def test_committed_decision_replays_from_pinned_fixture_bundle(
    tmp_path: Path,
    measure: int,
    accepted_count: int,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    decision = (
        repo_root
        / "tests/fixtures/vlm_melody/human_candidate_reviews"
        / f"coqueteos_system_002_measure_{measure:03d}.json"
    )

    result = materializer.materialize_human_candidate_review(
        tmp_path / "out",
        decision_fixture=decision,
        repo_root=repo_root,
    )

    review = _json(Path(result["review"]))
    assert review["identity"]["system_measure_index"] == measure
    assert len(review["accepted_heads"]) == accepted_count
    for key in ("raw_image", "candidate_artifact", "proposals"):
        assert review["source"][key]["path"].startswith(
            "tests/fixtures/vlm_melody/hollow_notehead_inputs/"
        )


def _fixture(tmp_path: Path) -> dict[str, Path]:
    repo_root = tmp_path / "repo"
    out_dir = repo_root / "out"
    namespace = out_dir / "test-score/vlm_melody_training_inputs/test_namespace"
    raw_path = namespace / "measure_inputs/system_002/measure_005_raw.png"
    raw_path.parent.mkdir(parents=True)
    _staff_image(raw_path)
    candidates_path = namespace / "candidates/system_002/measure_005/candidates.json"
    _write_json(
        candidates_path,
        {
            "kind": materializer.CANDIDATES_KIND,
            "slug": "test-score",
            "system_index": 2,
            "system_measure_index": 5,
            "global_measure_index": 5,
            "candidate_count": 3,
            "source_image_path": raw_path.relative_to(repo_root).as_posix(),
            "source_image_size_px": {"width": 180, "height": 100},
            "staff_lines_y_px": [20, 30, 40, 50, 60],
            "candidates": [
                _candidate("c001", 20, 55, 0.9),
                _candidate("c002", 70, 55, 0.8),
                _candidate("c003", 120, 70, 0.7),
            ],
        },
    )
    proposals_path = namespace / "proposals/proposals.json"
    _write_json(
        proposals_path,
        {
            "kind": materializer.PROPOSALS_KIND,
            "identity": {"slug": "test-score", "system_index": 2},
            "segmentation_namespace": "test_namespace",
            "tasks": [
                {
                    "identity": {"system_measure_index": 5},
                    "proposal": {
                        "assignments": [
                            {"candidate_id": "c001"},
                            {"candidate_id": "c002"},
                        ]
                    },
                }
            ],
        },
    )
    decision_path = repo_root / "tests/fixture.json"
    _write_json(
        decision_path,
        {
            "schema_version": 1,
            "kind": materializer.DECISION_KIND,
            "split_status": "consumed_training",
            "reviewer_type": "human_candidate_adjudication",
            "human_reviewed": True,
            "eligible_for_spike_training": True,
            "eligible_for_heldout_claim": False,
            "identity": {
                "slug": "test-score",
                "system_index": 2,
                "system_measure_index": 5,
                "physical_measure_number": 5,
            },
            "segmentation_namespace": "test_namespace",
            "source": {
                "raw_image": _record(raw_path, repo_root),
                "candidate_artifact": _record(candidates_path, repo_root),
                "proposals": _record(proposals_path, repo_root),
            },
            "accepted_heads": [
                {
                    "order": 1,
                    "candidate_id": "c001",
                    "sounding_pitch": "F4",
                    "staff_pitch": "F4",
                },
                {
                    "order": 2,
                    "candidate_id": "c003",
                    "sounding_pitch": "D4",
                    "staff_pitch": "D4",
                },
            ],
            "rejected_candidate_classes": [
                {
                    "class": "accidental",
                    "description": "Synthetic accidental fragment.",
                    "candidate_ids": ["c002"],
                }
            ],
        },
    )
    return {
        "repo_root": repo_root,
        "out_dir": out_dir,
        "decision": decision_path,
    }


def _candidate(candidate_id: str, x: int, y: int, score: float) -> dict:
    return {
        "id": candidate_id,
        "rank": int(candidate_id[1:]),
        "bbox": {"left": x - 5, "top": y - 5, "right": x + 5, "bottom": y + 5},
        "center": {"x": float(x), "y": float(y)},
        "score": score,
    }


def _staff_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (180, 100), "white")
    draw = ImageDraw.Draw(image)
    for y in (20, 30, 40, 50, 60):
        draw.line((0, y, 179, y), fill="black", width=1)
    image.save(path)


def _record(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
