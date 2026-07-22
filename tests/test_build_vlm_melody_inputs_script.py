import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from score2abc.manifest import write_manifest_jsonl
from score2abc.schemas import WorkItem, WorkMetadata
from score2abc.utils.imaging import estimate_ink_threshold
from scripts.build_vlm_melody_inputs import build_vlm_melody_inputs, main


def test_build_vlm_melody_inputs_writes_measure_crops_and_context(tmp_path: Path) -> None:
    out_dir = _make_pipeline_out(tmp_path)

    records = build_vlm_melody_inputs(out_dir, selected_slugs={"demo"}, selected_systems={1})

    assert len(records) == 3
    first = records[0]
    assert first["slug"] == "demo"
    assert first["system_index"] == 1
    assert first["system_measure_index"] == 1
    assert first["global_measure_index"] == 0
    assert first["allow_pickup"] is True
    assert first["expected_measure_beats"] == "2"
    assert len(first["staff_lines_y_px_in_system"]) == 5

    for record in records:
        for key in ("measure_raw", "measure_staff", "measure_staff_overlay", "context"):
            assert Path(record["paths"][key]).exists()

    context = json.loads(Path(first["paths"]["context"]).read_text(encoding="utf-8"))
    assert context["task"].startswith("Transcribe only the melody notes")
    assert context["paths"]["measure_staff"].endswith("measure_001_staff.png")

    manifest_path = out_dir / "demo" / "vlm_melody_inputs" / "manifest.jsonl"
    manifest_records = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["system_measure_index"] for record in manifest_records] == [1, 2, 3]


def test_build_vlm_melody_inputs_cli_rejects_missing_manifest(tmp_path: Path) -> None:
    assert main([str(tmp_path / "missing")]) == 1


def test_build_vlm_melody_inputs_splits_consumed_carrizal_system(tmp_path: Path) -> None:
    slug = "jaime-llanos_19_carrizal_pasillo_emilio-murillo"
    fixture_dir = Path(__file__).parent / "fixtures" / "barlines" / slug
    out_dir = tmp_path / "out"
    work_dir = out_dir / slug
    systems_dir = work_dir / "systems"
    systems_dir.mkdir(parents=True)
    source_pdf = work_dir / "source.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    shutil.copyfile(fixture_dir / "system_004.png", systems_dir / "system_004.png")
    item = WorkItem(
        slug=slug,
        pdf_path=source_pdf,
        metadata=WorkMetadata(
            title="Carrizal",
            composer="Emilio Murillo",
            rhythm="pasillo",
            time_signature="3/4",
            key_hint="one flat: Bb",
        ),
    )
    write_manifest_jsonl([item], out_dir / "manifest.jsonl")

    records = build_vlm_melody_inputs(
        out_dir,
        selected_slugs={slug},
        selected_systems={4},
    )

    assert len(records) == 8
    assert [record["system_measure_index"] for record in records] == list(range(1, 9))
    for record in records:
        crop_path = Path(record["paths"]["measure_raw"])
        with Image.open(crop_path) as crop:
            gray = crop.convert("L")
            threshold = estimate_ink_threshold(gray)
            staff_lines = record["staff_lines_y_px_in_system"]
            nonstaff_ink = sum(
                gray.getpixel((x, y)) < threshold
                for y in range(gray.height)
                if all(abs(y - line_y) > 3 for line_y in staff_lines)
                for x in range(gray.width)
            )
            assert gray.width >= 180
            assert nonstaff_ink >= 250


def _make_pipeline_out(tmp_path: Path) -> Path:
    out_dir = tmp_path / "out"
    work_dir = out_dir / "demo"
    systems_dir = work_dir / "systems"
    systems_dir.mkdir(parents=True)
    (work_dir / "source.pdf").write_bytes(b"%PDF-1.4\n")

    item = WorkItem(
        slug="demo",
        pdf_path=work_dir / "source.pdf",
        metadata=WorkMetadata(
            title="Demo",
            composer="Tester",
            rhythm="pasillo",
            time_signature="2/4",
            key_hint="D minor",
        ),
    )
    write_manifest_jsonl([item], out_dir / "manifest.jsonl")
    _draw_system(systems_dir / "system_001.png", barline_fractions=[0.35, 0.68])
    _draw_system(systems_dir / "system_001_m_001.png", barline_fractions=[])
    _draw_system(systems_dir / "system_002.png", barline_fractions=[0.5])
    return out_dir


def _draw_system(
    path: Path,
    barline_fractions: list[float],
    *,
    width: int = 800,
    height: int = 180,
    staff_top: int = 50,
    staff_bottom: int = 130,
) -> None:
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    spacing = (staff_bottom - staff_top) // 4
    for idx in range(5):
        y = staff_top + idx * spacing
        draw.line([(0, y), (width - 1, y)], fill="black", width=2)

    for fraction in barline_fractions:
        x = round(fraction * width)
        draw.line([(x, staff_top), (x, staff_bottom)], fill="black", width=3)

    draw.ellipse((100, 80, 120, 96), fill="black")
    image.save(path)
