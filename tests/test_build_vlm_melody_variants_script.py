import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

from scripts.build_vlm_melody_variants import (
    VARIANT_IDS,
    build_vlm_melody_variants,
    main,
)
from scripts.build_vlm_pitch_ruler_inputs import DEFAULT_LABEL_WIDTH_PX


def test_build_vlm_melody_variants_writes_all_variants_and_manifests(tmp_path: Path) -> None:
    out_dir = _make_vlm_melody_inputs(tmp_path)

    records = build_vlm_melody_variants(
        out_dir,
        selected_slugs={"demo"},
        selected_systems={1},
        selected_measures={3},
        variant_ids=("all",),
        overwrite=True,
    )

    assert [record["variant_id"] for record in records] == list(VARIANT_IDS)
    assert {record["system_measure_index"] for record in records} == {3}

    staff_record = _record_by_variant(records, "staff")
    assert staff_record["input_kind"] == "staff"
    assert staff_record["image_path"].endswith("measure_003_staff.png")
    assert staff_record["recipe"] == {"kind": "base", "path_key": "measure_staff"}

    context_record = _record_by_variant(records, "neighbor_context")
    context_path = Path(context_record["image_path"])
    assert context_record["input_kind"] == "neighbor_context"
    assert context_path.name == "measure_003_neighbor_context.png"
    assert context_path.exists()
    assert context_record["recipe"]["target_markers_touch_music_pixels"] is False

    panel_record = _record_by_variant(records, "pitch_ruler_panel_from_staff_overlay")
    panel_path = Path(panel_record["image_path"])
    assert panel_record["input_kind"] == "pitch_ruler_panel"
    assert panel_path.name == "measure_003_pitch_ruler_panel_from_staff_overlay.png"
    assert panel_path.exists()
    assert panel_record["recipe"]["source_kind"] == "staff_overlay"
    assert panel_record["recipe"]["style"] == "panel"
    assert panel_record["recipe"]["source_path"].endswith("measure_003_staff_overlay.png")
    assert panel_record["context_path"].endswith("measure_003_context.json")

    top_manifest = out_dir / "vlm_melody_variants_manifest.jsonl"
    slug_manifest = out_dir / "demo" / "vlm_melody_inputs" / "variants_manifest.jsonl"
    assert _read_jsonl(top_manifest) == records
    assert _read_jsonl(slug_manifest) == records


def test_build_vlm_melody_variants_supports_repeated_variant_subset(
    tmp_path: Path,
) -> None:
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
                "4",
                "--variant",
                "staff",
                "--variant",
                "pitch_ruler_soft_from_staff",
                "--overwrite",
            ]
        )
        == 0
    )

    records = _read_jsonl(out_dir / "vlm_melody_variants_manifest.jsonl")
    assert [record["variant_id"] for record in records] == [
        "staff",
        "pitch_ruler_soft_from_staff",
    ]
    assert {record["system_measure_index"] for record in records} == {4}
    assert records[1]["input_kind"] == "pitch_ruler_soft"


def test_pitch_ruler_variants_use_unique_source_style_filenames(tmp_path: Path) -> None:
    out_dir = _make_vlm_melody_inputs(tmp_path)

    records = build_vlm_melody_variants(
        out_dir,
        selected_slugs={"demo"},
        selected_systems={1},
        selected_measures={3},
        variant_ids=(
            "pitch_ruler_soft_from_staff",
            "pitch_ruler_soft_from_staff_overlay",
        ),
        overwrite=True,
    )

    paths = {record["variant_id"]: Path(record["image_path"]) for record in records}
    assert paths["pitch_ruler_soft_from_staff"].name == (
        "measure_003_pitch_ruler_soft_from_staff.png"
    )
    assert paths["pitch_ruler_soft_from_staff_overlay"].name == (
        "measure_003_pitch_ruler_soft_from_staff_overlay.png"
    )
    assert paths["pitch_ruler_soft_from_staff"] != paths["pitch_ruler_soft_from_staff_overlay"]
    assert paths["pitch_ruler_soft_from_staff"].exists()
    assert paths["pitch_ruler_soft_from_staff_overlay"].exists()


def test_staff_and_overlay_derived_variants_do_not_overwrite_each_other(
    tmp_path: Path,
) -> None:
    out_dir = _make_vlm_melody_inputs(tmp_path)

    records = build_vlm_melody_variants(
        out_dir,
        selected_slugs={"demo"},
        selected_systems={1},
        selected_measures={3},
        variant_ids=(
            "pitch_ruler_panel_from_staff",
            "pitch_ruler_panel_from_staff_overlay",
        ),
        overwrite=True,
    )

    staff_path = Path(_record_by_variant(records, "pitch_ruler_panel_from_staff")["image_path"])
    overlay_path = Path(
        _record_by_variant(records, "pitch_ruler_panel_from_staff_overlay")["image_path"]
    )
    assert staff_path.name == "measure_003_pitch_ruler_panel_from_staff.png"
    assert overlay_path.name == "measure_003_pitch_ruler_panel_from_staff_overlay.png"

    staff_image = Image.open(staff_path)
    overlay_image = Image.open(overlay_path)
    staff_music = staff_image.crop(
        (DEFAULT_LABEL_WIDTH_PX, 0, staff_image.width, staff_image.height)
    )
    overlay_music = overlay_image.crop(
        (DEFAULT_LABEL_WIDTH_PX, 0, overlay_image.width, overlay_image.height)
    )
    assert ImageChops.difference(staff_music, overlay_music).getbbox() is not None

    manifest_records = _read_jsonl(out_dir / "vlm_melody_variants_manifest.jsonl")
    assert [Path(record["image_path"]) for record in manifest_records] == [staff_path, overlay_path]


def _record_by_variant(records: list[dict], variant_id: str) -> dict:
    for record in records:
        if record["variant_id"] == variant_id:
            return record
    raise AssertionError(f"missing variant: {variant_id}")


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _make_vlm_melody_inputs(tmp_path: Path) -> Path:
    out_dir = tmp_path / "out"
    system_dir = out_dir / "demo" / "vlm_melody_inputs" / "system_001"
    system_dir.mkdir(parents=True)
    source_system_path = out_dir / "demo" / "systems" / "system_001.png"
    source_system_path.parent.mkdir(parents=True)
    source_system = Image.new("RGB", (270, 80), color="white")
    source_draw = ImageDraw.Draw(source_system)
    for y in (10, 20, 30, 40, 50):
        source_draw.line((0, y, 269, y), fill="black", width=1)
    source_system.save(source_system_path)
    records = []
    for measure_index in (2, 3, 4):
        stem = f"measure_{measure_index:03d}"
        raw_path = system_dir / f"{stem}_raw.png"
        staff_path = system_dir / f"{stem}_staff.png"
        staff_overlay_path = system_dir / f"{stem}_staff_overlay.png"
        context_path = system_dir / f"{stem}_context.json"

        _write_measure_images(raw_path, staff_path, staff_overlay_path)
        paths = {
            "measure_raw": str(raw_path),
            "measure_staff": str(staff_path),
            "measure_staff_overlay": str(staff_overlay_path),
            "context": str(context_path),
            "source_system": str(source_system_path),
        }
        context = {
            "slug": "demo",
            "title": "Demo",
            "rhythm": "pasillo",
            "clef_hint": "treble",
            "time_signature_hint": "3/4",
            "key_hint": None,
            "system_index": 1,
            "system_measure_index": measure_index,
            "global_measure_index": measure_index - 1,
            "display_measure_number": measure_index,
            "allow_pickup": False,
            "expected_measure_beats": "3",
            "paths": paths,
            "staff_lines_y_px_in_staff_crop": [5, 15, 25, 35, 45],
            "x_bounds_px": {
                "left": (measure_index - 2) * 90,
                "right": (measure_index - 1) * 90,
            },
        }
        context_path.write_text(json.dumps(context), encoding="utf-8")
        records.append(context)

    (out_dir / "vlm_melody_inputs_manifest.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return out_dir


def _write_measure_images(raw_path: Path, staff_path: Path, staff_overlay_path: Path) -> None:
    raw = Image.new("RGB", (90, 60), color="white")
    raw_draw = ImageDraw.Draw(raw)
    raw_draw.rectangle((5, 5, 84, 54), outline=(150, 150, 150))
    raw.save(raw_path)

    staff = Image.new("RGB", (90, 60), color="white")
    staff_draw = ImageDraw.Draw(staff)
    for y in (5, 15, 25, 35, 45):
        staff_draw.line((0, y, 89, y), fill="black", width=1)
    staff_draw.ellipse((35, 24, 45, 31), fill="black")
    staff.save(staff_path)

    overlay = staff.copy()
    overlay_draw = ImageDraw.Draw(overlay)
    for y in (5, 15, 25, 35, 45):
        overlay_draw.line((0, y, 89, y), fill=(220, 20, 60), width=1)
    overlay.save(staff_overlay_path)
