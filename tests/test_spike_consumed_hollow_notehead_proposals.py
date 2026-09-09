import json
import math
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from scripts.experiments import spike_consumed_hollow_notehead_proposals as spike


def test_proposes_center_for_closed_hollow_notehead() -> None:
    image, payload = _case("closed_hollow")

    proposals, considered = spike.propose_hollow_notehead_centers(image, payload)

    assert len(considered) == 1
    assert len(proposals) == 1
    assert proposals[0].support_candidate_ids == ("c001", "c002")
    assert proposals[0].center == (85.0, 70.0)
    assert proposals[0].contour_kind == "closed"
    assert proposals[0].features["hole_support_alignment"] >= 0.9


def test_proposes_center_for_open_hollow_notehead() -> None:
    image, payload = _case("open_hollow")

    proposals, considered = spike.propose_hollow_notehead_centers(image, payload)

    assert len(considered) == 1
    assert len(proposals) == 1
    assert proposals[0].contour_kind == "open"
    assert proposals[0].features["hole_area_spacing_squared"] == 0.0
    assert proposals[0].features["ring_density"] >= spike.MIN_OPEN_RING_DENSITY


def test_rejects_filled_notehead() -> None:
    image, payload = _case("filled")

    proposals, considered = spike.propose_hollow_notehead_centers(image, payload)

    assert proposals == []
    assert len(considered) == 1
    assert considered[0]["features"]["inner_density"] > spike.MAX_OPEN_INNER_DENSITY


def test_rejects_staff_stem_rectangle() -> None:
    image, payload = _case("rectangle")

    proposals, considered = spike.propose_hollow_notehead_centers(image, payload)

    assert proposals == []
    assert len(considered) == 1
    features = considered[0]["features"]
    assert (
        features["hole_support_alignment"] < spike.MIN_HOLE_SUPPORT_ALIGNMENT
        or features["hole_axis_ratio"] > spike.MAX_HOLE_AXIS_RATIO
    )


def test_rejects_opposite_diagonal_even_with_closed_hole() -> None:
    image, payload = _case("closed_hollow", rises_to_right=False)

    proposals, considered = spike.propose_hollow_notehead_centers(image, payload)

    assert proposals == []
    assert len(considered) == 1
    assert considered[0]["features"]["rises_to_right"] == 0.0


def test_proposal_outcomes_match_truth_one_to_one(tmp_path: Path) -> None:
    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(
        json.dumps(
            {
                "staff_lines_y_px": [20, 40, 60, 80, 100],
                "staff_spacing_px": 20,
                "candidates": [_candidate("c001", (150.0, 90.0))],
            }
        )
    )
    measure = spike.ReviewedMeasure(
        identity={"slug": "test", "system_index": 1, "system_measure_index": 1},
        lane="human_promoted_aviador",
        image_path=tmp_path / "unused.png",
        candidates_path=candidates_path,
        truth_source_path=tmp_path / "unused.json",
        truth_source_kind="promoted_notehead_review",
        truth_measure_index=None,
    )
    row = {
        "proposal_artifact": {
            "proposal_count": 2,
            "proposals": [
                {"center": {"x": 49.0, "y": 50.0}},
                {"center": {"x": 51.0, "y": 50.0}},
            ],
        }
    }

    evaluated = spike._evaluate_frozen_row(row, measure, [(50.0, 50.0)])

    assert evaluated["evaluation"]["recovered_truth_count"] == 1
    assert evaluated["evaluation"]["augmented_matched_truth"] == 1
    assert evaluated["evaluation"]["proposal_outcomes"] == {
        "duplicate_truth": 1,
        "recovered_truth": 1,
    }


def test_fixture_manifest_rejects_hash_drift(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / spike.DEFAULT_FIXTURE_MANIFEST
    manifest = json.loads(source.read_text())
    manifest["measures"][0]["candidates"]["sha256"] = "0" * 64
    drifted = tmp_path / "manifest.json"
    drifted.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="hash drift"):
        spike.load_reviewed_measures(repo_root, manifest_path=drifted)


def test_consumed_spike_replays_from_committed_fixture_bundle(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    destination = tmp_path / "hollow-notehead-spike"

    result = spike.run_consumed_hollow_notehead_spike(
        repo_root=repo_root,
        output_dir=destination,
    )

    report = json.loads((result / "report.json").read_text())
    assert report["summary"]["measure_count"] == 22
    assert report["summary"]["proposal_count"] == 5
    assert report["summary"]["recovered_truth_count"] == 5
    assert report["summary"]["proposal_outcomes"] == {"recovered_truth": 5}
    assert report["provenance"]["fixture_manifest"]["path"] == (spike.DEFAULT_FIXTURE_MANIFEST)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        spike.run_consumed_hollow_notehead_spike(
            repo_root=repo_root,
            output_dir=destination,
        )


def _case(
    shape: str,
    *,
    rises_to_right: bool = True,
) -> tuple[Image.Image, dict]:
    image = Image.new("RGB", (180, 120), "white")
    draw = ImageDraw.Draw(image)
    staff_lines = [20, 40, 60, 80, 100]
    if shape != "open_hollow":
        for y in staff_lines:
            draw.line((0, y, image.width - 1, y), fill="black", width=2)

    center = (85.0, 70.0)
    direction = (0.45, -0.89 if rises_to_right else 0.89)
    if shape == "closed_hollow":
        _draw_ellipse_outline(draw, center, direction, start=0.0, stop=2.0 * math.pi)
    elif shape == "open_hollow":
        _draw_ellipse_outline(
            draw,
            center,
            direction,
            start=0.60,
            stop=2.0 * math.pi - 0.60,
            width=6,
        )
    elif shape == "filled":
        _draw_filled_ellipse(draw, center, direction)
    elif shape == "rectangle":
        draw.rectangle((78, 58, 92, 82), outline="black", width=3)
    else:
        raise ValueError(f"Unknown shape: {shape}")

    support = (
        (
            center[0] - direction[0] * 12,
            center[1] - direction[1] * 12,
        ),
        (
            center[0] + direction[0] * 12,
            center[1] + direction[1] * 12,
        ),
    )
    payload = {
        "staff_lines_y_px": staff_lines,
        "staff_spacing_px": 20,
        "candidates": [
            _candidate("c001", support[0]),
            _candidate("c002", support[1]),
        ],
    }
    return image, payload


def _draw_ellipse_outline(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    direction: tuple[float, float],
    *,
    start: float,
    stop: float,
    width: int = 3,
) -> None:
    points = [
        _ellipse_point(center, direction, angle)
        for angle in (start + index * (stop - start) / 80 for index in range(81))
    ]
    draw.line(points, fill="black", width=width, joint="curve")


def _draw_filled_ellipse(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    direction: tuple[float, float],
) -> None:
    points = [_ellipse_point(center, direction, index * 2.0 * math.pi / 80) for index in range(80)]
    draw.polygon(points, fill="black")


def _ellipse_point(
    center: tuple[float, float],
    direction: tuple[float, float],
    angle: float,
) -> tuple[float, float]:
    major = 14.0
    minor = 7.0
    perpendicular = (-direction[1], direction[0])
    return (
        center[0]
        + major * direction[0] * math.cos(angle)
        + minor * perpendicular[0] * math.sin(angle),
        center[1]
        + major * direction[1] * math.cos(angle)
        + minor * perpendicular[1] * math.sin(angle),
    )


def _candidate(candidate_id: str, center: tuple[float, float]) -> dict:
    return {
        "id": candidate_id,
        "score": 0.9,
        "center": {"x": center[0], "y": center[1]},
        "bbox": {
            "left": round(center[0] - 5),
            "top": round(center[1] - 5),
            "right": round(center[0] + 5),
            "bottom": round(center[1] + 5),
        },
    }
