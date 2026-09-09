from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import promote_vlm_notehead_reviews as promoter


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_saved_review(
    repo_root: Path,
    *,
    slug: str = "demo",
    system: int = 1,
    measure: int = 2,
) -> dict[str, Path]:
    out_dir = repo_root / "out"
    image_path = (
        out_dir
        / slug
        / "vlm_melody_inputs"
        / f"system_{system:03d}"
        / f"measure_{measure:03d}_raw.png"
    )
    candidate_path = (
        out_dir
        / slug
        / "vlm_notehead_localization"
        / f"system_{system:03d}"
        / f"measure_{measure:03d}"
        / "candidates.json"
    )
    review_path = (
        out_dir
        / slug
        / "vlm_melody_reviews"
        / f"system_{system:03d}"
        / f"measure_{measure:03d}"
        / "review.json"
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"deterministic image bytes")
    candidate_path.write_text('{"candidate_count": 2}\n', encoding="utf-8")
    review = {
        "schema_version": 1,
        "kind": promoter.REVIEW_KIND,
        "identity": {
            "slug": slug,
            "system_index": system,
            "system_measure_index": measure,
            "global_measure_index": measure - 1,
        },
        "source": {
            "image_path": str(image_path),
            "image_sha256": _sha256(image_path),
            "candidate_artifact_path": str(candidate_path),
            "candidate_artifact_sha256": _sha256(candidate_path),
            "candidate_cap": 24,
            "coordinate_space": "source image pixels, origin at top-left",
        },
        "candidates": [
            {
                "id": "c001",
                "rank": 1,
                "center": {"x": 10.0, "y": 20.0},
                "label": "accepted",
                "auto_pitch": "C4",
            },
            {
                "id": "c002",
                "rank": 2,
                "center": {"x": 30.0, "y": 40.0},
                "label": "rejected",
                "auto_pitch": "A3",
            },
        ],
        "manual_noteheads": [
            {
                "id": "manual_001",
                "center": {"x": 50.0, "y": 24.0},
                "auto_pitch": "B3",
                "pitch": "Bb3",
                "pitch_corrected": True,
            }
        ],
        "final_noteheads": [
            {
                "source": {"kind": "candidate", "candidate_id": "c001"},
                "center": {"x": 10.0, "y": 20.0},
                "auto_pitch": "C4",
                "pitch": "C#4",
                "pitch_corrected": True,
                "order": 1,
            },
            {
                "source": {"kind": "manual", "manual_id": "manual_001"},
                "center": {"x": 50.0, "y": 24.0},
                "auto_pitch": "B3",
                "pitch": "Bb3",
                "pitch_corrected": True,
                "order": 2,
            },
        ],
        "timing": {
            "active_review_ms": 4321,
            "saved_at": "2026-07-15T12:00:00+00:00",
            "inactivity_timeout_ms": 30000,
        },
        "metrics": {
            "ground_truth_path": "/private/hidden/notehead_ground_truth.json",
            "selection": {"f1": 1.0},
        },
    }
    review_path.write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
    return {
        "out_dir": out_dir,
        "image": image_path,
        "candidate": candidate_path,
        "review": review_path,
    }


def test_cli_promotes_selected_reviews_with_portable_complete_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    first = _write_saved_review(repo_root, slug="alpha", system=1, measure=2)
    _write_saved_review(repo_root, slug="beta", system=2, measure=3)
    _write_saved_review(repo_root, slug="ignored", system=1, measure=2)
    destination = repo_root / "fixtures"
    monkeypatch.setattr(promoter, "REPO_ROOT", repo_root)

    result = promoter.main(
        [
            str(first["out_dir"]),
            "--slug",
            "alpha",
            "--slug",
            "beta",
            "--system",
            "1",
            "--system",
            "2",
            "--measure",
            "2",
            "--measure",
            "3",
            "--destination",
            str(destination),
        ]
    )

    assert result == 0
    assert sorted(path.name for path in destination.iterdir()) == [
        "alpha_system_001_measure_002.json",
        "beta_system_002_measure_003.json",
    ]
    fixture = json.loads(
        (destination / "alpha_system_001_measure_002.json").read_text(encoding="utf-8")
    )
    assert fixture["source"]["image_path"] == first["image"].relative_to(repo_root).as_posix()
    assert (
        fixture["source"]["candidate_artifact_path"]
        == first["candidate"].relative_to(repo_root).as_posix()
    )
    assert [item["label"] for item in fixture["candidates"]] == ["accepted", "rejected"]
    assert fixture["manual_noteheads"][0]["pitch"] == "Bb3"
    assert fixture["final_noteheads"][0]["pitch_corrected"] is True
    assert fixture["timing"]["active_review_ms"] == 4321
    assert fixture["provenance"] == {
        "review_type": "human",
        "review_tool_path": promoter.REVIEW_TOOL_PATH,
        "scope": "spike_only",
        "source_review_path": first["review"].relative_to(repo_root).as_posix(),
    }


@pytest.mark.parametrize(
    ("stale_source", "message"),
    [
        ("image", "Source image SHA256 is stale"),
        ("candidate", "Candidate artifact SHA256 is stale"),
    ],
)
def test_promotion_rejects_stale_source_hashes(
    tmp_path: Path, stale_source: str, message: str
) -> None:
    repo_root = tmp_path / "repo"
    paths = _write_saved_review(repo_root)
    paths[stale_source].write_bytes(b"changed after human review")

    with pytest.raises(ValueError, match=message):
        promoter.promote_reviews(
            paths["out_dir"],
            destination=repo_root / "fixtures",
            repo_root=repo_root,
        )

    assert not (repo_root / "fixtures").exists()


def test_promotion_refuses_overwrite_and_force_reproduces_identical_bytes(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    paths = _write_saved_review(repo_root)
    destination = repo_root / "fixtures"
    target = promoter.promote_reviews(
        paths["out_dir"], destination=destination, repo_root=repo_root
    )[0]
    first_bytes = target.read_bytes()

    with pytest.raises(FileExistsError, match="Rerun with --force"):
        promoter.promote_reviews(paths["out_dir"], destination=destination, repo_root=repo_root)
    assert target.read_bytes() == first_bytes

    promoter.promote_reviews(
        paths["out_dir"],
        destination=destination,
        force=True,
        repo_root=repo_root,
    )
    assert target.read_bytes() == first_bytes


def test_promoted_fixture_excludes_metrics_and_absolute_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    paths = _write_saved_review(repo_root)
    review = json.loads(paths["review"].read_text(encoding="utf-8"))
    review["ground_truth_path"] = "/private/also-hidden.json"
    review["source"]["ground_truth_path"] = "/private/source-hidden.json"
    review["candidates"][0]["metrics"] = {"ground_truth_path": "/private/candidate-hidden.json"}
    paths["review"].write_text(json.dumps(review), encoding="utf-8")

    target = promoter.promote_reviews(
        paths["out_dir"],
        destination=repo_root / "fixtures",
        repo_root=repo_root,
    )[0]
    content = target.read_text(encoding="utf-8")
    fixture = json.loads(content)

    assert "metrics" not in content
    assert "ground_truth_path" not in content
    assert "/private/" not in content
    assert str(repo_root) not in content
    assert fixture["candidates"][0]["label"] == "accepted"
    assert fixture["source"]["image_path"].startswith("out/")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda review: review["final_noteheads"][0]["source"].update(candidate_id="missing"),
            "references unknown candidate",
        ),
        (
            lambda review: review["candidates"][0].update(label="rejected"),
            "references rejected candidate",
        ),
        (
            lambda review: review["final_noteheads"][1]["source"].update(manual_id="missing"),
            "references unknown manual notehead",
        ),
    ],
)
def test_promotion_rejects_invalid_final_source_references(
    tmp_path: Path, mutation: object, message: str
) -> None:
    repo_root = tmp_path / "repo"
    paths = _write_saved_review(repo_root)
    review = json.loads(paths["review"].read_text(encoding="utf-8"))
    mutation(review)  # type: ignore[operator]
    paths["review"].write_text(json.dumps(review), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        promoter.build_portable_fixture(paths["review"], paths["out_dir"], repo_root=repo_root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda review: review["final_noteheads"][1].update(order=1),
            "Duplicate final notehead order",
        ),
        (
            lambda review: review["final_noteheads"].__setitem__(
                1, {**review["final_noteheads"][0], "order": 2}
            ),
            "Duplicate final notehead source reference",
        ),
        (
            lambda review: review["final_noteheads"][0]["source"].update(manual_id="manual_001"),
            "contradictory manual_id",
        ),
    ],
)
def test_promotion_rejects_duplicate_or_contradictory_review_graph_edges(
    tmp_path: Path, mutation: object, message: str
) -> None:
    repo_root = tmp_path / "repo"
    paths = _write_saved_review(repo_root)
    review = json.loads(paths["review"].read_text(encoding="utf-8"))
    mutation(review)  # type: ignore[operator]
    paths["review"].write_text(json.dumps(review), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        promoter.build_portable_fixture(paths["review"], paths["out_dir"], repo_root=repo_root)
