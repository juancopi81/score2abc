import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.experiments import freeze_second_score_heldout as spike


def test_prepare_target_requests_uses_only_selected_measure_inputs(tmp_path: Path) -> None:
    slug = "fresh-score"
    records = [
        _measure_record(tmp_path, slug=slug, system=4, measure=1, global_measure=21),
        _measure_record(tmp_path, slug=slug, system=4, measure=2, global_measure=22),
        _measure_record(tmp_path, slug=slug, system=5, measure=1, global_measure=28),
    ]
    manifest = tmp_path / slug / "vlm_melody_inputs/manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    requests = spike.prepare_target_requests(
        tmp_path,
        slug=slug,
        system_index=4,
        measures=(1, 2),
        clef="treble",
        time_signature="3/4",
        key_hint="one flat: Bb",
    )

    assert [row["identity"]["system_measure_index"] for row in requests] == [1, 2]
    assert [row["identity"]["global_measure_index"] for row in requests] == [21, 22]
    assert all(row["split"] == "fresh_heldout" for row in requests)
    assert all(row["allowed_context"]["time_signature"] == "3/4" for row in requests)
    assert all(row["allowed_context"]["key_hint"] == "one flat: Bb" for row in requests)
    assert all("truth" not in json.dumps(row).lower() for row in requests)


def test_prepare_target_requests_rejects_missing_measure(tmp_path: Path) -> None:
    slug = "fresh-score"
    record = _measure_record(tmp_path, slug=slug, system=4, measure=1, global_measure=21)
    manifest = tmp_path / slug / "vlm_melody_inputs/manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing target melody-input records"):
        spike.prepare_target_requests(
            tmp_path,
            slug=slug,
            system_index=4,
            measures=(1, 2),
            clef="treble",
            time_signature="3/4",
            key_hint="one flat: Bb",
        )


def test_fresh_heldout_refuses_to_overwrite_existing_freeze(tmp_path: Path) -> None:
    freeze = (
        tmp_path
        / "fresh-score"
        / spike.OUTPUT_SUBDIR
        / "system_004"
        / spike.SPLIT_NAME
        / "freeze.json"
    )
    freeze.parent.mkdir(parents=True)
    freeze.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="already frozen and cannot be overwritten"):
        spike.freeze_fresh_heldout(
            tmp_path,
            reviews_dir=tmp_path / "reviews",
            target_slug="fresh-score",
            target_system=4,
        )


def _measure_record(
    out_dir: Path,
    *,
    slug: str,
    system: int,
    measure: int,
    global_measure: int,
) -> dict:
    root = out_dir / slug / "vlm_melody_inputs" / f"system_{system:03d}"
    root.mkdir(parents=True, exist_ok=True)
    raw = root / f"measure_{measure:03d}_raw.png"
    staff = root / f"measure_{measure:03d}_staff.png"
    Image.new("RGB", (80, 40), "white").save(raw)
    Image.new("RGB", (80, 30), "white").save(staff)
    return {
        "slug": slug,
        "system_index": system,
        "system_measure_index": measure,
        "global_measure_index": global_measure,
        "allow_pickup": False,
        "paths": {
            "measure_raw": str(raw),
            "measure_staff": str(staff),
        },
        "staff_lines_y_px_in_system": [8, 14, 20, 26, 32],
        "staff_lines_y_px_in_staff_crop": [3, 9, 15, 21, 27],
    }
