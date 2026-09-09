import json
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.build_vlm_melody_prompt_variants import (
    build_vlm_melody_prompt_variants,
    main,
)
from scripts.build_vlm_melody_variants import build_vlm_melody_variants


def test_build_vlm_melody_prompt_variants_associates_prompts_with_images(
    tmp_path: Path,
) -> None:
    out_dir = _make_vlm_melody_inputs(tmp_path)
    build_vlm_melody_variants(
        out_dir,
        selected_slugs={"demo"},
        selected_systems={1},
        selected_measures={3},
        variant_ids=("staff", "pitch_ruler_panel_from_staff"),
        overwrite=True,
    )

    records = build_vlm_melody_prompt_variants(
        out_dir,
        selected_slugs={"demo"},
        selected_systems={1},
        selected_measures={3},
        variant_ids=("staff", "pitch_ruler_panel_from_staff"),
        prompt_ids=("all",),
        overwrite=True,
    )

    prompt_pairs = {(record["variant_id"], record["prompt_id"]) for record in records}
    assert prompt_pairs == {
        ("staff", "direct_pitch_v0"),
        ("staff", "direct_pitch_best_effort_v1"),
        ("staff", "educated_pitch_v2"),
        ("staff", "notehead_count_then_pitch_v1"),
        ("staff", "free_response_describe_v1"),
        ("staff", "describe_then_guess_v2"),
        ("pitch_ruler_panel_from_staff", "pitch_ruler_panel_explained_v1"),
        ("pitch_ruler_panel_from_staff", "educated_pitch_v2"),
        ("pitch_ruler_panel_from_staff", "notehead_count_then_pitch_v1"),
        ("pitch_ruler_panel_from_staff", "free_response_describe_v1"),
        ("pitch_ruler_panel_from_staff", "describe_then_guess_v2"),
    }

    panel_record = _record_by_pair(
        records,
        "pitch_ruler_panel_from_staff",
        "pitch_ruler_panel_explained_v1",
    )
    assert panel_record["input_kind"] == "pitch_ruler_panel"
    assert panel_record["prompt_variant_id"] == (
        "pitch_ruler_panel_from_staff__pitch_ruler_panel_explained_v1"
    )
    assert panel_record["image_path"].endswith("measure_003_pitch_ruler_panel_from_staff.png")

    system_prompt = Path(panel_record["system_prompt_path"])
    user_prompt = Path(panel_record["user_prompt_path"])
    schema_path = Path(panel_record["schema_path"])
    config_path = Path(panel_record["config_path"])
    assert "left pitch-reference gutter" in system_prompt.read_text(encoding="utf-8")
    assert "left gutter contains pitch labels" in user_prompt.read_text(encoding="utf-8")
    assert schema_path.exists()
    assert json.loads(config_path.read_text(encoding="utf-8"))["prompt_id"] == (
        "pitch_ruler_panel_explained_v1"
    )

    educated_record = _record_by_pair(records, "staff", "educated_pitch_v2")
    educated_prompt = Path(educated_record["user_prompt_path"]).read_text(encoding="utf-8")
    assert "Do not return zero notes unless the crop is truly blank" in educated_prompt
    assert "low-confidence note" in educated_prompt
    assert json.loads(Path(educated_record["schema_path"]).read_text(encoding="utf-8"))[
        "required"
    ] == [
        "notehead_count",
        "items",
        "comments",
        "overall_confidence",
        "uncertainties",
    ]

    describe_record = _record_by_pair(records, "staff", "describe_then_guess_v2")
    describe_prompt = Path(describe_record["user_prompt_path"]).read_text(encoding="utf-8")
    assert describe_record["output_mode"] == "free_response"
    assert describe_record["schema_path"] is None
    assert "Final best-guess transcription" in describe_prompt

    top_manifest = out_dir / "vlm_melody_prompt_variants_manifest.jsonl"
    slug_manifest = out_dir / "demo" / "vlm_melody_inputs" / "prompt_variants_manifest.jsonl"
    assert _read_jsonl(top_manifest) == records
    assert _read_jsonl(slug_manifest) == records


def test_build_vlm_melody_prompt_variants_supports_free_response_without_schema(
    tmp_path: Path,
) -> None:
    out_dir = _make_vlm_melody_inputs(tmp_path)

    records = build_vlm_melody_prompt_variants(
        out_dir,
        selected_slugs={"demo"},
        selected_systems={1},
        selected_measures={3},
        variant_ids=("staff",),
        prompt_ids=("free_response_describe_v1",),
        overwrite=True,
    )

    assert len(records) == 1
    record = records[0]
    assert record["output_mode"] == "free_response"
    assert record["schema_path"] is None
    assert "Return concise prose" in Path(record["user_prompt_path"]).read_text(encoding="utf-8")
    prompt_dir = Path(record["config_path"]).parent
    assert not (prompt_dir / "schema.json").exists()


def test_build_vlm_melody_prompt_variants_rejects_incompatible_selection(
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
                "3",
                "--variant",
                "staff",
                "--prompt",
                "pitch_ruler_panel_explained_v1",
            ]
        )
        == 1
    )


def test_build_vlm_melody_prompt_variants_cli_writes_subset(
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
                "3",
                "--variant",
                "pitch_ruler_soft_from_staff",
                "--prompt",
                "pitch_ruler_soft_explained_v1",
                "--overwrite",
            ]
        )
        == 0
    )

    records = _read_jsonl(out_dir / "vlm_melody_prompt_variants_manifest.jsonl")
    assert len(records) == 1
    assert records[0]["variant_id"] == "pitch_ruler_soft_from_staff"
    assert records[0]["prompt_id"] == "pitch_ruler_soft_explained_v1"
    assert "faint dotted pitch guides" in Path(records[0]["user_prompt_path"]).read_text(
        encoding="utf-8"
    )


def test_neighbor_context_prompt_explains_target_ticks_and_internal_stems(
    tmp_path: Path,
) -> None:
    out_dir = _make_vlm_melody_inputs(tmp_path)
    build_vlm_melody_variants(
        out_dir,
        selected_slugs={"demo"},
        selected_systems={1},
        selected_measures={3},
        variant_ids=("neighbor_context",),
        overwrite=True,
    )

    records = build_vlm_melody_prompt_variants(
        out_dir,
        selected_slugs={"demo"},
        selected_systems={1},
        selected_measures={3},
        variant_ids=("neighbor_context",),
        prompt_ids=("neighbor_context_transcribe_v1",),
        overwrite=True,
    )

    assert len(records) == 1
    record = records[0]
    assert record["input_kind"] == "neighbor_context"
    system_prompt = Path(record["system_prompt_path"]).read_text(encoding="utf-8")
    user_prompt = Path(record["user_prompt_path"]).read_text(encoding="utf-8")
    assert "red ticks" in system_prompt
    assert "long internal vertical stroke with an attached notehead is a stem" in user_prompt


def _record_by_pair(records: list[dict], variant_id: str, prompt_id: str) -> dict:
    for record in records:
        if record["variant_id"] == variant_id and record["prompt_id"] == prompt_id:
            return record
    raise AssertionError(f"missing pair: {variant_id} {prompt_id}")


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
    contexts = []
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
        contexts.append(context)
    (out_dir / "vlm_melody_inputs_manifest.jsonl").write_text(
        "\n".join(json.dumps(context) for context in contexts) + "\n",
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
