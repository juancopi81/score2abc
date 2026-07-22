import json
from pathlib import Path

from PIL import Image, ImageChops

from scripts.build_vlm_pitch_ruler_inputs import (
    DEFAULT_LABEL_WIDTH_PX,
    build_vlm_pitch_ruler_inputs,
    main,
)


def test_build_vlm_pitch_ruler_inputs_writes_image_and_manifest(tmp_path: Path) -> None:
    out_dir = _make_vlm_melody_inputs(tmp_path, clef_hint="treble")

    records = build_vlm_pitch_ruler_inputs(
        out_dir,
        selected_slugs={"demo"},
        selected_systems={1},
        selected_measures={3},
    )

    assert len(records) == 1
    record = records[0]
    output_path = Path(record["paths"]["measure_pitch_ruler"])
    assert output_path.name == "measure_003_pitch_ruler.png"
    assert output_path.exists()
    assert record["pitch_ruler"]["source_kind"] == "staff_overlay"
    assert record["pitch_ruler"]["clef"] == "treble"

    source = Image.open(out_dir / "demo/vlm_melody_inputs/system_001/measure_003_staff_overlay.png")
    output = Image.open(output_path)
    assert output.size == (source.width + DEFAULT_LABEL_WIDTH_PX, source.height)

    pasted_source = output.crop((DEFAULT_LABEL_WIDTH_PX, 0, output.width, output.height))
    assert ImageChops.difference(source.convert("RGB"), pasted_source.convert("RGB")).getbbox()

    top_manifest = out_dir / "vlm_pitch_ruler_inputs_manifest.jsonl"
    slug_manifest = out_dir / "demo" / "vlm_melody_inputs" / "pitch_ruler_manifest.jsonl"
    assert top_manifest.exists()
    assert slug_manifest.exists()
    manifest_record = json.loads(top_manifest.read_text(encoding="utf-8").strip())
    assert manifest_record["paths"]["measure_pitch_ruler"] == str(output_path)


def test_build_vlm_pitch_ruler_inputs_writes_soft_variant(tmp_path: Path) -> None:
    out_dir = _make_vlm_melody_inputs(tmp_path, clef_hint="treble")

    records = build_vlm_pitch_ruler_inputs(
        out_dir,
        selected_slugs={"demo"},
        selected_systems={1},
        selected_measures={3},
        style="soft",
    )

    assert len(records) == 1
    record = records[0]
    output_path = Path(record["paths"]["measure_pitch_ruler_soft"])
    assert output_path.name == "measure_003_pitch_ruler_soft.png"
    assert output_path.exists()
    assert record["pitch_ruler"]["style"] == "soft"

    source = Image.open(out_dir / "demo/vlm_melody_inputs/system_001/measure_003_staff_overlay.png")
    output = Image.open(output_path)
    assert output.size == (source.width + DEFAULT_LABEL_WIDTH_PX, source.height)

    top_manifest = out_dir / "vlm_pitch_ruler_inputs_manifest.jsonl"
    manifest_record = json.loads(top_manifest.read_text(encoding="utf-8").strip())
    assert manifest_record["paths"]["measure_pitch_ruler_soft"] == str(output_path)


def test_build_vlm_pitch_ruler_inputs_writes_panel_variant_without_overlay(
    tmp_path: Path,
) -> None:
    out_dir = _make_vlm_melody_inputs(tmp_path, clef_hint="treble")

    records = build_vlm_pitch_ruler_inputs(
        out_dir,
        selected_slugs={"demo"},
        selected_systems={1},
        selected_measures={3},
        source_kind="staff",
        style="panel",
    )

    assert len(records) == 1
    record = records[0]
    output_path = Path(record["paths"]["measure_pitch_ruler_panel"])
    assert output_path.name == "measure_003_pitch_ruler_panel.png"
    assert output_path.exists()
    assert record["pitch_ruler"]["source_kind"] == "staff"
    assert record["pitch_ruler"]["style"] == "panel"

    source = Image.open(out_dir / "demo/vlm_melody_inputs/system_001/measure_003_staff.png")
    output = Image.open(output_path)
    assert output.size == (source.width + DEFAULT_LABEL_WIDTH_PX, source.height)

    pasted_source = output.crop((DEFAULT_LABEL_WIDTH_PX, 0, output.width, output.height))
    assert (
        ImageChops.difference(source.convert("RGB"), pasted_source.convert("RGB")).getbbox() is None
    )

    top_manifest = out_dir / "vlm_pitch_ruler_inputs_manifest.jsonl"
    manifest_record = json.loads(top_manifest.read_text(encoding="utf-8").strip())
    assert manifest_record["paths"]["measure_pitch_ruler_panel"] == str(output_path)


def test_build_vlm_pitch_ruler_inputs_rejects_unsupported_clef(tmp_path: Path) -> None:
    out_dir = _make_vlm_melody_inputs(tmp_path, clef_hint="bass")

    assert main([str(out_dir), "--slug", "demo", "--system", "1", "--measure", "3"]) == 1


def _make_vlm_melody_inputs(tmp_path: Path, *, clef_hint: str) -> Path:
    out_dir = tmp_path / "out"
    system_dir = out_dir / "demo" / "vlm_melody_inputs" / "system_001"
    system_dir.mkdir(parents=True)
    staff_overlay_path = system_dir / "measure_003_staff_overlay.png"
    staff_path = system_dir / "measure_003_staff.png"
    raw_path = system_dir / "measure_003_raw.png"
    context_path = system_dir / "measure_003_context.json"

    image = Image.new("RGB", (80, 50), color="white")
    for path in (raw_path, staff_path, staff_overlay_path):
        image.save(path)

    context = {
        "slug": "demo",
        "clef_hint": clef_hint,
        "system_index": 1,
        "system_measure_index": 3,
        "global_measure_index": 2,
        "display_measure_number": 3,
        "paths": {
            "measure_raw": str(raw_path),
            "measure_staff": str(staff_path),
            "measure_staff_overlay": str(staff_overlay_path),
            "context": str(context_path),
        },
        "staff_lines_y_px_in_staff_crop": [5, 15, 25, 35, 45],
    }
    context_path.write_text(json.dumps(context), encoding="utf-8")
    (out_dir / "vlm_melody_inputs_manifest.jsonl").write_text(
        json.dumps(context) + "\n",
        encoding="utf-8",
    )
    return out_dir
