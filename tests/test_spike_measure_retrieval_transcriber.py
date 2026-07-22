import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from scripts.experiments.spike_measure_retrieval_transcriber import (
    BinaryFeature,
    ProtocolGate,
    Retrieval,
    Template,
    binary_shift_distance,
    extract_binary_feature,
    prediction_from_retrieval,
    retrieve_nearest,
    run_experiment,
)


def test_feature_normalization_and_shift_tolerance() -> None:
    first = _music_image(width=220, height=120, spacing=14, symbol_x=88)
    shifted = _music_image(width=260, height=144, spacing=17, symbol_x=110)
    unrelated = _music_image(width=220, height=120, spacing=14, symbol_x=150, invert=True)

    first_feature = extract_binary_feature(
        first, _staff_lines(120, 14), variant="staff_remove_fixed"
    )
    shifted_feature = extract_binary_feature(
        shifted, _staff_lines(144, 17), variant="staff_remove_fixed"
    )
    unrelated_feature = extract_binary_feature(
        unrelated, _staff_lines(120, 14), variant="staff_remove_fixed"
    )

    assert (first_feature.width, first_feature.height) == (160, 96)
    assert (shifted_feature.width, shifted_feature.height) == (160, 96)
    assert binary_shift_distance(first_feature, shifted_feature) < binary_shift_distance(
        first_feature, unrelated_feature
    )


def test_retrieval_excludes_query_measure_itself() -> None:
    query = _binary_feature((2, 2), (3, 2))
    exact_self = _template(_identity(1), query, pitch=60)
    other = _template(_identity(2), _binary_feature((2, 2), (4, 2)), pitch=62)

    retrieval = retrieve_nearest(_identity(1), query, [exact_self, other], exclude_self=True)

    assert retrieval.source.identity == _identity(2)
    assert retrieval.distance > 0


def test_prediction_copies_exact_source_events_and_can_abstain() -> None:
    source = _template(_identity(1), _binary_feature((2, 2)), pitch=67)
    retrieval = Retrieval(source=source, distance=0.1, second_distance=0.4)

    accepted = prediction_from_retrieval(_identity(7), retrieval, threshold=0.2)
    abstained = prediction_from_retrieval(_identity(8), retrieval, threshold=0.05)

    assert accepted["notes"] == list(source.notes)
    assert accepted["rests"] == list(source.rests)
    assert accepted["retrieval"]["source_identity"] == source.identity
    assert abstained["notes"] == []
    assert abstained["rests"] == []
    assert abstained["retrieval"]["abstained"] is True
    accepted["notes"][0]["pitch_midi"] = 99
    assert source.notes[0]["pitch_midi"] == 67


def test_protocol_requires_feature_and_prediction_freezes_before_truth(tmp_path: Path) -> None:
    truth_path = tmp_path / "truth.jsonl"
    truth_path.write_text("{}\n", encoding="utf-8")
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text("{}\n", encoding="utf-8")
    gate = ProtocolGate()

    with pytest.raises(RuntimeError, match="all split features"):
        gate.read_development_truth(truth_path, _read_jsonl)
    with pytest.raises(RuntimeError, match="passing validation gate"):
        gate.seal_predictions("heldout", predictions_path)

    gate.mark_feature_freeze_complete()
    gate.read_development_truth(truth_path, _read_jsonl)
    gate.seal_predictions("validation", predictions_path)
    gate.record_validation_gate({"passed": False})
    with pytest.raises(RuntimeError, match="passing validation gate"):
        gate.seal_predictions("heldout", predictions_path)


def test_end_to_end_freezes_before_truth_skips_nonindependent_heldout_and_is_deterministic(
    tmp_path: Path,
) -> None:
    out_dir, slug = _make_benchmark(tmp_path)
    access_log: list[str] = []

    def guarded_loader(path: Path) -> list[dict]:
        split = path.parent.name
        experiment_dir = path.parents[1] / "experiment_one"
        assert (experiment_dir / "feature_freeze.json").exists()
        if split == "validation":
            assert (experiment_dir / "validation/predictions.sealed.jsonl").exists()
        if split == "heldout":
            pytest.fail("Heldout truth must stay unopened after the coordination update")
        access_log.append(split)
        return _read_jsonl(path)

    first = run_experiment(
        out_dir,
        slug=slug,
        truth_loader=guarded_loader,
        output_dir=out_dir / slug / "vlm_melody_event_benchmark/experiment_one",
    )
    second = run_experiment(
        out_dir,
        slug=slug,
        output_dir=out_dir / slug / "vlm_melody_event_benchmark/experiment_two",
    )

    assert access_log == ["development", "validation"]
    assert first["validation"]["gate"]["passed"] is True
    assert first["heldout"]["status"] == "skipped_not_presealed_before_prior_s3_open"
    assert first["heldout"]["metrics"] is None
    assert (
        first["validation"]["artifacts"]["predictions_sha256"]
        == second["validation"]["artifacts"]["predictions_sha256"]
    )
    assert first["tuning"]["selected_variant"] == second["tuning"]["selected_variant"]
    assert first["tuning"]["selected_threshold"] == second["tuning"]["selected_threshold"]


def _make_benchmark(tmp_path: Path) -> tuple[Path, str]:
    out_dir = tmp_path / "out"
    slug = "demo"
    benchmark_dir = out_dir / slug / "vlm_melody_event_benchmark"
    specifications = {
        "development": [
            (1, 1, "a", 60),
            (1, 2, "a", 60),
            (2, 1, "b", 67),
        ],
        "validation": [(7, 1, "a", 60)],
        "heldout": [(3, 1, "b", 67)],
    }
    global_index = 0
    for split, rows in specifications.items():
        requests = []
        truths = []
        split_dir = benchmark_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for system, measure, pattern, pitch in rows:
            spacing = 14 + system % 3
            height = 120 + system % 3 * 8
            width = 220 + measure * 7
            symbol_x = 82 if pattern == "a" else 145
            image = _music_image(
                width=width,
                height=height,
                spacing=spacing,
                symbol_x=symbol_x,
                invert=pattern == "b",
            )
            image_dir = out_dir / slug / "inputs" / f"system_{system:03d}"
            image_dir.mkdir(parents=True, exist_ok=True)
            raw_path = image_dir / f"measure_{measure:03d}_raw.png"
            staff_path = image_dir / f"measure_{measure:03d}_staff.png"
            image.save(raw_path)
            image.save(staff_path)
            identity = {
                "slug": slug,
                "system_index": system,
                "system_measure_index": measure,
                "global_measure_index": global_index,
            }
            lines = _staff_lines(height, spacing)
            requests.append(
                {
                    "identity": identity,
                    "images": {
                        "raw": _image_record(raw_path, out_dir),
                        "staff": _image_record(staff_path, out_dir),
                    },
                    "staff_geometry": {
                        "raw_staff_lines_y_px": lines,
                        "staff_crop_lines_y_px": lines,
                    },
                }
            )
            truths.append(
                {
                    "identity": identity,
                    "measure_extent_beats": 3,
                    "notes": [{"onset_beats": 0, "duration_beats": 3, "pitch_midi": pitch}],
                    "rests": [],
                }
            )
            global_index += 1
        _write_jsonl(split_dir / "requests.jsonl", requests)
        _write_jsonl(split_dir / "truth.jsonl", truths)
    return out_dir, slug


def _music_image(
    *,
    width: int,
    height: int,
    spacing: int,
    symbol_x: int,
    invert: bool = False,
) -> Image.Image:
    image = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(image)
    lines = _staff_lines(height, spacing)
    for y in lines:
        draw.line((5, y, width - 6, y), fill=90, width=1)
    center_y = lines[2] + (-spacing if invert else spacing // 2)
    draw.ellipse(
        (
            symbol_x - spacing // 3,
            center_y - spacing // 4,
            symbol_x + spacing // 3,
            center_y + spacing // 4,
        ),
        fill=0,
    )
    stem_x = symbol_x - spacing // 3 if invert else symbol_x + spacing // 3
    stem_end = center_y + spacing * 2 if invert else center_y - spacing * 2
    draw.line((stem_x, center_y, stem_x, stem_end), fill=0, width=2)
    if invert:
        draw.line((symbol_x + spacing, center_y, symbol_x + spacing * 2, center_y), fill=0, width=3)
    return image


def _staff_lines(height: int, spacing: int) -> list[int]:
    center = height // 2
    return [center + (index - 2) * spacing for index in range(5)]


def _binary_feature(*points: tuple[int, int]) -> BinaryFeature:
    width = height = 8
    point_set = set(points)
    rows = tuple(
        tuple(1 if (x, y) in point_set else 0 for x in range(width)) for y in range(height)
    )
    return BinaryFeature("test", width, height, rows)


def _template(identity: dict, feature: BinaryFeature, *, pitch: int) -> Template:
    return Template(
        identity=identity,
        feature=feature,
        notes=({"onset_beats": 0, "duration_beats": 2, "pitch_midi": pitch},),
        rests=({"onset_beats": 2, "duration_beats": 1},),
    )


def _identity(measure: int) -> dict:
    return {
        "slug": "demo",
        "system_index": 1,
        "system_measure_index": measure,
        "global_measure_index": measure - 1,
    }


def _image_record(path: Path, out_dir: Path) -> dict:
    return {
        "path_relative_to_out": path.relative_to(out_dir).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
