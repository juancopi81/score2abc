import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image

from scripts.build_vlm_melody_event_benchmark import (
    BENCHMARK_SPLITS,
    BenchmarkTarget,
    build_benchmark,
    build_truth_rows,
    evaluate_prediction_file,
    evaluate_predictions,
    prepare_requests,
)


def test_prepare_requests_freezes_images_and_validates_mapping_without_truth(
    tmp_path: Path,
) -> None:
    out_dir = _make_out(tmp_path, measure_count=2)

    requests = prepare_requests(
        out_dir,
        slug="demo",
        targets=(BenchmarkTarget(1, 1, 0, 0), BenchmarkTarget(1, 2, 1, 1)),
        split_name="development",
        clef="treble",
        time_signature="3/4",
        key_hint="one flat: Bb",
    )

    assert [row["identity"]["global_measure_index"] for row in requests] == [0, 1]
    assert requests[0]["allowed_context"] == {
        "clef": "treble",
        "time_signature": "3/4",
        "key_hint": "one flat: Bb",
        "expected_measure_beats": "3",
        "allow_pickup": True,
    }
    assert requests[0]["images"]["raw"]["path_relative_to_out"].startswith(
        "demo/vlm_melody_inputs/"
    )
    raw_path = out_dir / requests[0]["images"]["raw"]["path_relative_to_out"]
    assert (
        requests[0]["images"]["raw"]["sha256"] == hashlib.sha256(raw_path.read_bytes()).hexdigest()
    )

    with pytest.raises(ValueError, match="mapping drift"):
        prepare_requests(
            out_dir,
            slug="demo",
            targets=(BenchmarkTarget(1, 2, 7, 7),),
            split_name="heldout",
            clef="treble",
            time_signature="3/4",
            key_hint=None,
        )


def test_build_benchmark_writes_requests_before_truth_is_opened(tmp_path: Path) -> None:
    out_dir = _make_out(tmp_path, measure_count=4)
    ground_truth_dir = tmp_path / "truth"
    ground_truth_dir.mkdir()
    (ground_truth_dir / "demo.json").write_text("not json\n", encoding="utf-8")

    original = BENCHMARK_SPLITS["development"]
    BENCHMARK_SPLITS["development"] = tuple(
        BenchmarkTarget(1, index, index - 1, index - 1) for index in range(1, 5)
    )
    try:
        with pytest.raises(json.JSONDecodeError):
            build_benchmark(
                out_dir,
                slug="demo",
                ground_truth_dir=ground_truth_dir,
                split_names=("development",),
                clef="treble",
                time_signature="3/4",
                key_hint=None,
            )
    finally:
        BENCHMARK_SPLITS["development"] = original

    request_path = out_dir / "demo" / "vlm_melody_event_benchmark/development/requests.jsonl"
    assert request_path.exists()
    assert len(request_path.read_text(encoding="utf-8").splitlines()) == 4


def test_build_truth_rows_derives_pickup_internal_and_full_measure_rests() -> None:
    requests = [
        _request(0, allow_pickup=True),
        _request(1, allow_pickup=False),
        _request(2, allow_pickup=False),
    ]
    payload = {
        "notes": [
            {"measure": 0, "onset_beats": 0, "duration_beats": 0.5, "pitch_midi": 60},
            {"measure": 0, "onset_beats": 1, "duration_beats": 0.5, "pitch_midi": 62},
            {"measure": 1, "onset_beats": 0.5, "duration_beats": 1, "pitch_midi": 64},
            {"measure": 1, "onset_beats": 0.5, "duration_beats": 1, "pitch_midi": 67},
        ]
    }

    rows = build_truth_rows(requests, payload, measure_length=Fraction(3))

    assert rows[0]["measure_extent_beats"] == 1.5
    assert rows[0]["rests"] == [{"onset_beats": 0.5, "duration_beats": 0.5}]
    assert rows[1]["rests"] == [
        {"onset_beats": 0, "duration_beats": 0.5},
        {"onset_beats": 1.5, "duration_beats": 1.5},
    ]
    assert [note["pitch_midi"] for note in rows[1]["notes"]] == [64, 67]
    assert rows[2]["rests"] == [{"onset_beats": 0, "duration_beats": 3}]


def test_evaluate_predictions_scores_exact_events_rests_and_missing_rows() -> None:
    truth = [
        {
            "identity": _identity(0),
            "measure_extent_beats": 3,
            "notes": [{"onset_beats": 0.5, "duration_beats": 1, "pitch_midi": 64}],
            "rests": [
                {"onset_beats": 0, "duration_beats": 0.5},
                {"onset_beats": 1.5, "duration_beats": 1.5},
            ],
        },
        {
            "identity": _identity(1),
            "measure_extent_beats": 3,
            "notes": [{"onset_beats": 0, "duration_beats": 3, "pitch_midi": 67}],
            "rests": [],
        },
    ]
    predictions = [
        {
            "identity": _identity(0),
            "notes": [{"onset_beats": 0.5, "duration_beats": 1, "pitch_midi": 64}],
            "rests": [
                {"onset_beats": 0, "duration_beats": 0.5},
                {"onset_beats": 1.5, "duration_beats": 1.5},
            ],
        }
    ]

    report = evaluate_predictions(truth, predictions)

    assert report["summary"]["predicted"] == 1
    assert report["summary"]["exact_measures"] == 1
    assert report["summary"]["exact_measure_rate"] == 0.5
    assert report["summary"]["note_precision"] == 1.0
    assert report["summary"]["note_recall"] == 0.5
    assert report["summary"]["rest_f1"] == 1.0
    assert report["results"][1]["status"] == "missing_prediction"


def test_evaluate_predictions_rejects_invalid_event_geometry() -> None:
    truth = [
        {
            "identity": _identity(0),
            "measure_extent_beats": 3,
            "notes": [],
            "rests": [{"onset_beats": 0, "duration_beats": 3}],
        }
    ]
    predictions = [
        {
            "identity": _identity(0),
            "notes": [{"onset_beats": -0.5, "duration_beats": 1, "pitch_midi": 64}],
            "rests": [],
        }
    ]

    with pytest.raises(ValueError, match="Negative predicted note onset"):
        evaluate_predictions(truth, predictions)


def test_evaluate_prediction_file_rejects_stale_split_omitted_from_metadata(
    tmp_path: Path,
) -> None:
    benchmark_dir, predictions_path = _make_evaluation_benchmark(tmp_path)
    metadata = json.loads((benchmark_dir / "benchmark.json").read_text(encoding="utf-8"))
    metadata["splits"] = []
    (benchmark_dir / "benchmark.json").write_text(
        json.dumps(metadata) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not included in the current benchmark metadata"):
        evaluate_prediction_file(
            benchmark_dir,
            split_name="development",
            predictions_path=predictions_path,
        )


@pytest.mark.parametrize(
    ("filename", "error_pattern"),
    (
        ("requests.jsonl", "requests.jsonl hash mismatch"),
        ("truth.jsonl", "truth.jsonl hash mismatch"),
    ),
)
def test_evaluate_prediction_file_rejects_hash_drift(
    tmp_path: Path,
    filename: str,
    error_pattern: str,
) -> None:
    benchmark_dir, predictions_path = _make_evaluation_benchmark(tmp_path)
    split_file = benchmark_dir / "development" / filename
    split_file.write_text(split_file.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=error_pattern):
        evaluate_prediction_file(
            benchmark_dir,
            split_name="development",
            predictions_path=predictions_path,
        )


def _make_out(tmp_path: Path, *, measure_count: int) -> Path:
    out_dir = tmp_path / "out"
    melody_dir = out_dir / "demo" / "vlm_melody_inputs" / "system_001"
    melody_dir.mkdir(parents=True)
    records = []
    for index in range(1, measure_count + 1):
        raw_path = melody_dir / f"measure_{index:03d}_raw.png"
        staff_path = melody_dir / f"measure_{index:03d}_staff.png"
        Image.new("L", (80, 40), color=255).save(raw_path)
        Image.new("L", (80, 30), color=255).save(staff_path)
        records.append(
            {
                "slug": "demo",
                "system_index": 1,
                "system_measure_index": index,
                "global_measure_index": index - 1,
                "allow_pickup": index == 1,
                "paths": {"measure_raw": str(raw_path), "measure_staff": str(staff_path)},
                "staff_lines_y_px_in_system": [5, 10, 15, 20, 25],
                "staff_lines_y_px_in_staff_crop": [3, 8, 13, 18, 23],
            }
        )
    manifest_path = out_dir / "demo" / "vlm_melody_inputs" / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return out_dir


def _make_evaluation_benchmark(tmp_path: Path) -> tuple[Path, Path]:
    benchmark_dir = tmp_path / "benchmark"
    split_dir = benchmark_dir / "development"
    split_dir.mkdir(parents=True)
    request_path = split_dir / "requests.jsonl"
    truth_path = split_dir / "truth.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    request_path.write_text(json.dumps({"identity": _identity(0)}) + "\n", encoding="utf-8")
    truth_path.write_text(
        json.dumps(
            {
                "identity": _identity(0),
                "measure_extent_beats": 3,
                "notes": [],
                "rests": [{"onset_beats": 0, "duration_beats": 3}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    predictions_path.write_text("", encoding="utf-8")
    metadata = {
        "splits": [
            {
                "name": "development",
                "requests_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
                "truth_sha256": hashlib.sha256(truth_path.read_bytes()).hexdigest(),
            }
        ]
    }
    (benchmark_dir / "benchmark.json").write_text(
        json.dumps(metadata) + "\n",
        encoding="utf-8",
    )
    return benchmark_dir, predictions_path


def _request(measure: int, *, allow_pickup: bool) -> dict:
    return {
        "identity": _identity(measure),
        "allowed_context": {"allow_pickup": allow_pickup},
    }


def _identity(measure: int) -> dict:
    return {
        "slug": "demo",
        "system_index": 1,
        "system_measure_index": measure + 1,
        "global_measure_index": measure,
    }
