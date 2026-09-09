import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

import scripts.build_vlm_notehead_localization_inputs as localization_inputs
from scripts.build_vlm_notehead_candidates import GridCandidate
from scripts.build_vlm_notehead_localization_inputs import (
    CANDIDATE_ASSISTED_LOCALIZATION,
    DIRECT_LOCALIZATION,
    GALLERY_HEADER_HEIGHT_PX,
    build_vlm_notehead_localization_inputs,
    make_gallery_cell,
)


def test_builder_writes_task_specific_roles_and_unique_experiment_ids(tmp_path: Path) -> None:
    out_dir = _make_base_inputs(tmp_path, measure_order=(2, 1))

    records = build_vlm_notehead_localization_inputs(
        out_dir,
        selected_slugs={"demo"},
        selected_systems={1},
        selected_measures={1, 2},
        task_kind="all",
    )

    assert [(record["system_measure_index"], record["task_kind"]) for record in records] == [
        (1, DIRECT_LOCALIZATION),
        (1, CANDIDATE_ASSISTED_LOCALIZATION),
        (2, DIRECT_LOCALIZATION),
        (2, CANDIDATE_ASSISTED_LOCALIZATION),
    ]
    assert len({record["experiment_id"] for record in records}) == len(records)

    direct = records[0]
    assisted = records[1]
    assert [image["role"] for image in direct["images"]] == [
        "context",
        "detail",
        "binary",
    ]
    assert [image["role"] for image in assisted["images"]] == [
        "context",
        "detail",
        "candidate_gallery",
    ]
    assert direct["candidate_artifact_path"] is None
    assert assisted["candidate_artifact_path"].endswith("/candidates.json")
    assert direct["context_path"].endswith("/measure_001_context.json")
    assert direct["images"][0]["path"].endswith("/context.png")
    assert direct["source_context"] == {
        "clef": "treble",
        "time_signature": "3/4",
        "key": "C",
        "allow_pickup": True,
        "expected_measure_beats": "3",
        "staff_lines_y_px": [15, 25, 35, 45, 55],
        "raw_image_size": {"width": 110, "height": 72},
    }


def test_candidate_generation_does_not_read_discoverable_ground_truth(
    tmp_path: Path, monkeypatch
) -> None:
    out_dir = _make_base_inputs(tmp_path, measure_order=(1,))
    ground_truth_dir = tmp_path / "coordinate_gt"
    ground_truth_dir.mkdir()
    ground_truth_path = ground_truth_dir / "demo_system_001_measure_001.json"
    ground_truth_path.write_text('{"must_not_be_read": true}\n', encoding="utf-8")
    monkeypatch.setattr(
        localization_inputs,
        "COORDINATE_GROUND_TRUTH_DIR",
        ground_truth_dir,
    )

    original_read_text = Path.read_text

    def guarded_read_text(path: Path, *args, **kwargs):
        if path == ground_truth_path:
            raise AssertionError("ground truth must not be read during generation")
        return original_read_text(path, *args, **kwargs)

    def fail_if_legacy_loader_runs(*args, **kwargs):
        raise AssertionError("ground-truth evaluator must not run during generation")

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(
        "scripts.build_vlm_notehead_candidates._load_ground_truth",
        fail_if_legacy_loader_runs,
    )

    records = build_vlm_notehead_localization_inputs(
        out_dir,
        selected_measures={1},
        task_kind=CANDIDATE_ASSISTED_LOCALIZATION,
    )

    assert records[0]["coordinate_ground_truth_path"] == str(ground_truth_path)
    artifact = json.loads(Path(records[0]["candidate_artifact_path"]).read_text())
    assert artifact["provenance"]["candidate_generation_is_blind"] is True
    assert artifact["provenance"]["ground_truth_files_read"] == []


def test_gallery_headers_are_composed_outside_untouched_patch_pixels() -> None:
    patch = Image.new("L", (40, 30), 255)
    draw = ImageDraw.Draw(patch)
    draw.ellipse((10, 6, 28, 24), fill=0)

    cell = make_gallery_cell(patch, "c007  x=0.625")

    assert cell.size == (patch.width, patch.height + GALLERY_HEADER_HEIGHT_PX)
    pasted_patch = cell.crop((0, GALLERY_HEADER_HEIGHT_PX, cell.width, cell.height))
    assert ImageChops.difference(patch, pasted_patch).getbbox() is None
    assert cell.crop((0, 0, cell.width, GALLERY_HEADER_HEIGHT_PX)).getbbox() is not None


def test_max_candidates_is_forwarded_and_candidate_ids_preserve_rank(
    tmp_path: Path, monkeypatch
) -> None:
    out_dir = _make_base_inputs(tmp_path, measure_order=(1,))
    requested_caps: list[int] = []
    ranked_candidates = [
        GridCandidate(
            bbox=(x - 2, 31, x + 2, 35),
            score=round(1.0 - rank / 100, 3),
            features={"synthetic_rank": float(rank)},
        )
        for rank, x in enumerate((80, 15, 55, 35, 95), start=1)
    ]

    def fake_detector(image, *, staff_lines, max_candidates):
        assert image.size == (110, 72)
        assert staff_lines == [15, 25, 35, 45, 55]
        requested_caps.append(max_candidates)
        return ranked_candidates

    monkeypatch.setattr(
        localization_inputs,
        "detect_staff_grid_density_candidates",
        fake_detector,
    )

    records = build_vlm_notehead_localization_inputs(
        out_dir,
        selected_measures={1},
        task_kind=CANDIDATE_ASSISTED_LOCALIZATION,
        max_candidates=3,
    )

    artifact = json.loads(Path(records[0]["candidate_artifact_path"]).read_text())
    assert requested_caps == [3]
    assert artifact["max_candidates"] == 3
    assert artifact["candidate_count"] == 3
    assert [candidate["id"] for candidate in artifact["candidates"]] == [
        "c001",
        "c002",
        "c003",
    ]
    assert [candidate["center"]["x"] for candidate in artifact["candidates"]] == [
        80.0,
        15.0,
        55.0,
    ]
    assert artifact["gallery"]["candidate_ids_left_to_right"] == [
        "c002",
        "c003",
        "c001",
    ]
    assert artifact["gallery"]["labels_touch_patch_pixels"] is False
    for cell in artifact["gallery"]["cells"]:
        assert cell["header_bbox_px_in_cell"]["bottom"] == cell["patch_bbox_px_in_cell"]["top"]


def test_outputs_are_stable_and_context_markers_do_not_touch_music(
    tmp_path: Path,
) -> None:
    out_dir = _make_base_inputs(tmp_path, measure_order=(3, 1, 2))

    first = build_vlm_notehead_localization_inputs(
        out_dir,
        selected_measures={2},
        task_kind="all",
        overwrite=True,
    )
    manifest_path = out_dir / "vlm_notehead_localization_manifest.jsonl"
    candidate_path = Path(first[1]["candidate_artifact_path"])
    gallery_path = Path(first[1]["images"][2]["path"])
    first_bytes = (
        manifest_path.read_bytes(),
        candidate_path.read_bytes(),
        gallery_path.read_bytes(),
    )

    second = build_vlm_notehead_localization_inputs(
        out_dir,
        selected_measures={2},
        task_kind="all",
        overwrite=True,
    )

    assert first == second
    assert first_bytes == (
        manifest_path.read_bytes(),
        candidate_path.read_bytes(),
        gallery_path.read_bytes(),
    )
    context_path = Path(first[0]["images"][0]["path"])
    context = Image.open(context_path).convert("RGB")
    source_system = Image.open(out_dir / "demo/systems/system_001.png").convert("RGB")
    margin = first[0]["provenance"]["views"]["context"]["white_margin_px"]
    crop_bounds = first[0]["provenance"]["views"]["context"]["source_crop_x_bounds_px"]
    musical_region = context.crop((0, margin, context.width, margin + source_system.height))
    expected_region = source_system.crop(
        (crop_bounds["left"], 0, crop_bounds["right"], source_system.height)
    )
    assert ImageChops.difference(musical_region, expected_region).getbbox() is None


def _make_base_inputs(tmp_path: Path, *, measure_order: tuple[int, ...]) -> Path:
    out_dir = tmp_path / "out"
    systems_dir = out_dir / "demo" / "systems"
    measure_dir = out_dir / "demo" / "vlm_melody_inputs" / "system_001"
    systems_dir.mkdir(parents=True)
    measure_dir.mkdir(parents=True)

    system_image = Image.new("RGB", (300, 72), "white")
    draw = ImageDraw.Draw(system_image)
    for y in (15, 25, 35, 45, 55):
        draw.line((0, y, system_image.width - 1, y), fill="black", width=1)
    for x, y in ((30, 35), (75, 30), (130, 40), (170, 25), (230, 35), (270, 45)):
        draw.ellipse((x - 5, y - 3, x + 5, y + 3), fill="black")
        draw.line((x + 5, y, x + 5, y - 16), fill="black", width=2)
    system_path = systems_dir / "system_001.png"
    system_image.save(system_path)

    bounds = {1: (0, 110), 2: (95, 205), 3: (190, 300)}
    records = []
    for measure_index in (1, 2, 3):
        left, right = bounds[measure_index]
        raw_path = measure_dir / f"measure_{measure_index:03d}_raw.png"
        context_path = measure_dir / f"measure_{measure_index:03d}_context.json"
        system_image.crop((left, 0, right, system_image.height)).save(raw_path)
        context = {
            "slug": "demo",
            "clef_hint": "treble",
            "time_signature_hint": "3/4",
            "key_hint": "C",
            "allow_pickup": measure_index == 1,
            "expected_measure_beats": "3",
            "system_index": 1,
            "system_measure_index": measure_index,
            "global_measure_index": measure_index - 1,
            "staff_lines_y_px_in_system": [15, 25, 35, 45, 55],
            "paths": {},
        }
        context_path.write_text(json.dumps(context), encoding="utf-8")
        records.append(
            {
                **context,
                "paths": {
                    "source_system": str(system_path),
                    "measure_raw": str(raw_path),
                    "context": str(context_path),
                },
                "x_bounds_px": {"left": left, "right": right},
            }
        )

    records_by_measure = {record["system_measure_index"]: record for record in records}
    manifest_records = [records_by_measure[index] for index in measure_order]
    (out_dir / "vlm_melody_inputs_manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in manifest_records),
        encoding="utf-8",
    )
    return out_dir
