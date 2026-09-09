import json
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.build_vlm_notehead_candidates import (
    build_notehead_candidates_for_measure,
    detect_notehead_candidates,
    detect_staff_grid_density_candidates,
    evaluate_notehead_candidates,
    main,
)


def test_detect_notehead_candidates_keeps_notes_on_and_between_staff_lines() -> None:
    image = _make_staff_measure_image()

    candidates = detect_notehead_candidates(image, staff_lines=[10, 20, 30, 40, 50])

    centers = [(candidate.center[0], candidate.center[1]) for candidate in candidates]
    assert any(abs(x - 34) <= 4 and abs(y - 30) <= 5 for x, y in centers)
    assert any(abs(x - 72) <= 4 and abs(y - 24) <= 5 for x, y in centers)
    assert all(candidate.width < 30 for candidate in candidates)


def test_staff_grid_generation_does_not_require_ground_truth(monkeypatch) -> None:
    image = _make_staff_measure_image()

    def fail_if_ground_truth_is_loaded(*args, **kwargs):
        raise AssertionError("candidate generation must not load ground truth")

    monkeypatch.setattr(
        "scripts.build_vlm_notehead_candidates._load_ground_truth",
        fail_if_ground_truth_is_loaded,
    )
    candidates = detect_staff_grid_density_candidates(image, staff_lines=[10, 20, 30, 40, 50])

    assert candidates


def test_staff_grid_lines_alone_do_not_become_many_candidates() -> None:
    image = Image.new("RGB", (120, 64), "white")
    draw = ImageDraw.Draw(image)
    for y in (10, 20, 30, 40, 50):
        draw.line((0, y, image.width - 1, y), fill="black", width=1)

    candidates = detect_staff_grid_density_candidates(image, staff_lines=[10, 20, 30, 40, 50])

    assert candidates == []


def test_staff_grid_proposes_on_line_and_between_line_noteheads() -> None:
    candidates = detect_staff_grid_density_candidates(
        _make_staff_measure_image(),
        staff_lines=[10, 20, 30, 40, 50],
    )
    centers = [candidate.center for candidate in candidates]

    assert any(abs(x - 34) <= 6 and abs(y - 30) <= 6 for x, y in centers)
    assert any(abs(x - 72) <= 6 and abs(y - 25) <= 6 for x, y in centers)


def test_notehead_evaluation_is_separate_and_deterministic() -> None:
    candidates = [
        {"id": "c001", "center": {"x": 10, "y": 10}},
        {"id": "c002", "center": {"x": 50, "y": 50}},
    ]
    ground_truth = [
        {"id": "n001", "center": {"x": 11, "y": 10}},
        {"id": "n002", "center": {"x": 90, "y": 90}},
    ]

    first = evaluate_notehead_candidates(candidates, ground_truth, tolerance_px=3)
    second = evaluate_notehead_candidates(candidates, ground_truth, tolerance_px=3)

    assert first == second
    assert first["true_positives"] == 1
    assert first["false_positives"] == 1
    assert first["false_negatives"] == 1
    assert first["assignments"] == [
        {
            "candidate_index": 0,
            "candidate_id": "c001",
            "ground_truth_index": 0,
            "ground_truth_id": "n001",
            "distance_px": 1.0,
        }
    ]


def test_build_notehead_candidates_writes_json_overlay_and_contact_sheet(
    tmp_path: Path,
) -> None:
    out_dir = _make_vlm_melody_inputs(tmp_path)

    artifact = build_notehead_candidates_for_measure(
        out_dir,
        slug="demo",
        system_index=1,
        measure_index=3,
        source_variant="staff",
        overwrite=True,
    )

    json_path = artifact["json_path"]
    overlay_path = artifact["overlay_path"]
    contact_sheet_path = artifact["contact_sheet_path"]
    assert json_path == (
        out_dir
        / "demo"
        / "vlm_melody_inputs"
        / "system_001"
        / "measure_003_notehead_candidates.json"
    )
    assert overlay_path.exists()
    assert contact_sheet_path.exists()
    assert set(artifact["diagnostic_paths"]) == {
        "threshold_ink_mask",
        "staff_line_mask",
        "staff_suppressed_mask",
        "raw_components",
        "contact_sheet",
    }
    assert all(path.exists() for path in artifact["diagnostic_paths"].values())

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "vlm_notehead_candidates"
    assert payload["slug"] == "demo"
    assert payload["system_index"] == 1
    assert payload["system_measure_index"] == 3
    assert payload["source_variant"] == "staff"
    assert payload["staff_lines_y_px"] == [10, 20, 30, 40, 50]
    assert payload["candidate_count"] >= 2
    assert payload["candidates"][0]["id"] == "c001"
    assert {"bbox", "center", "area", "width", "height", "staff_position"} <= set(
        payload["candidates"][0]
    )
    assert payload["staff_suppression"]["selected_rows_y_px"]
    assert payload["pre_filter_component_count"] >= payload["candidate_count"]
    assert payload["diagnostics"]["threshold_ink_mask"].endswith(
        "measure_003_notehead_ink_mask.png"
    )


def test_raw_diagnostics_show_configured_staff_rows_removed(tmp_path: Path) -> None:
    out_dir = _make_vlm_melody_inputs(tmp_path)

    artifact = build_notehead_candidates_for_measure(
        out_dir,
        slug="demo",
        system_index=1,
        measure_index=3,
        source_variant="raw",
        overwrite=True,
    )

    payload = json.loads(artifact["json_path"].read_text(encoding="utf-8"))
    selected_rows = payload["staff_suppression"]["selected_rows_y_px"]
    assert selected_rows
    threshold = Image.open(artifact["diagnostic_paths"]["threshold_ink_mask"]).convert("L")
    staff_mask = Image.open(artifact["diagnostic_paths"]["staff_line_mask"]).convert("L")
    suppressed = Image.open(artifact["diagnostic_paths"]["staff_suppressed_mask"]).convert("L")

    row = selected_rows[0]
    assert threshold.getpixel((0, row)) == 0
    assert staff_mask.getpixel((0, row)) == 0
    assert suppressed.getpixel((0, row)) == 255
    assert Image.open(artifact["diagnostic_paths"]["raw_components"]).size == (120, 80)


def test_notehead_candidates_cli_accepts_source_variant(tmp_path: Path) -> None:
    out_dir = _make_vlm_melody_inputs(tmp_path)

    assert (
        main(
            [
                str(out_dir),
                "--slug",
                "demo",
                "--system",
                "1",
                "--measure",
                "3",
                "--source-variant",
                "raw",
                "--overwrite",
            ]
        )
        == 0
    )

    json_path = (
        out_dir
        / "demo"
        / "vlm_melody_inputs"
        / "system_001"
        / "measure_003_raw_notehead_candidates.json"
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["source_variant"] == "raw"
    assert payload["source_image_path"].endswith("measure_003_raw.png")


def test_staff_grid_artifacts_are_strategy_specific_and_evaluated_after_generation(
    tmp_path: Path,
) -> None:
    out_dir = _make_vlm_melody_inputs(tmp_path)
    ground_truth_path = tmp_path / "ground_truth.json"
    ground_truth_path.write_text(
        json.dumps(
            {
                "noteheads": [
                    {"id": "n001", "center": {"x": 34, "y": 30}},
                    {"id": "n002", "center": {"x": 72, "y": 25}},
                ]
            }
        ),
        encoding="utf-8",
    )

    artifact = build_notehead_candidates_for_measure(
        out_dir,
        slug="demo",
        system_index=1,
        measure_index=3,
        source_variant="staff",
        strategy="staff-grid-density",
        ground_truth_path=ground_truth_path,
        overwrite=True,
    )

    assert artifact["json_path"].name == (
        "measure_003_notehead_candidates_staff-grid-density_v2.json"
    )
    assert artifact["overlay_path"].name.endswith("_v2_overlay.png")
    assert artifact["heatmap_path"].name.endswith("_v2_heatmap.png")
    assert artifact["comparison_overlay_path"].name.endswith("_v2_gt_compare.png")
    assert all(path.exists() for key, path in artifact.items() if key.endswith("_path"))
    payload = json.loads(artifact["json_path"].read_text(encoding="utf-8"))
    assert payload["strategy"] == "staff-grid-density"
    assert payload["heuristic"]["staff_lines_removed_from_detection_image"] is False
    assert payload["evaluation"]["gt_count"] == 2
    assert payload["evaluation"]["candidate_count"] == payload["candidate_count"]
    assert payload["candidate_count"] <= 12


def _make_vlm_melody_inputs(tmp_path: Path) -> Path:
    out_dir = tmp_path / "out"
    system_dir = out_dir / "demo" / "vlm_melody_inputs" / "system_001"
    system_dir.mkdir(parents=True)

    raw_path = system_dir / "measure_003_raw.png"
    staff_path = system_dir / "measure_003_staff.png"
    staff_overlay_path = system_dir / "measure_003_staff_overlay.png"
    context_path = system_dir / "measure_003_context.json"

    _make_staff_measure_image().save(staff_path)
    _make_staff_measure_image(height=80, staff_lines=[20, 30, 40, 50, 60]).save(raw_path)
    overlay = Image.open(staff_path).convert("RGB")
    overlay_draw = ImageDraw.Draw(overlay)
    for y in (10, 20, 30, 40, 50):
        overlay_draw.line((0, y, overlay.width - 1, y), fill=(220, 20, 60), width=1)
    overlay.save(staff_overlay_path)

    paths = {
        "measure_raw": str(raw_path),
        "measure_staff": str(staff_path),
        "measure_staff_overlay": str(staff_overlay_path),
        "context": str(context_path),
    }
    context = {
        "slug": "demo",
        "system_index": 1,
        "system_measure_index": 3,
        "global_measure_index": 2,
        "display_measure_number": 3,
        "clef_hint": "treble",
        "time_signature_hint": "3/4",
        "expected_measure_beats": "3",
        "paths": paths,
        "staff_lines_y_px_in_staff_crop": [10, 20, 30, 40, 50],
        "staff_lines_y_px_in_system": [20, 30, 40, 50, 60],
    }
    context_path.write_text(json.dumps(context), encoding="utf-8")
    (out_dir / "vlm_melody_inputs_manifest.jsonl").write_text(
        json.dumps(context) + "\n",
        encoding="utf-8",
    )
    return out_dir


def _make_staff_measure_image(
    *,
    width: int = 120,
    height: int = 64,
    staff_lines: list[int] | None = None,
) -> Image.Image:
    lines = staff_lines or [10, 20, 30, 40, 50]
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for y in lines:
        draw.line((0, y, width - 1, y), fill="black", width=1)

    draw.ellipse((27, lines[2] - 5, 41, lines[2] + 5), fill="black")
    draw.line((41, lines[2] - 4, 41, max(0, lines[2] - 26)), fill="black", width=2)
    draw.ellipse((65, lines[1] - 1, 79, lines[1] + 9), fill="black")
    draw.rectangle((95, 5, 99, 9), fill="black")
    return image
