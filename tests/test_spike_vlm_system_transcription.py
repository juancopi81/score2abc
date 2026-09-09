import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.experiments import spike_vlm_system_transcription as system_spike


def test_cli_rejects_heldout_split() -> None:
    parser = system_spike._build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--split", "heldout", "--system", "3"])

    assert exc_info.value.code == 2


def test_run_experiment_rejects_heldout_before_freezing_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_freeze(*_args, **_kwargs):
        raise AssertionError("heldout guard must run before request freezing")

    monkeypatch.setattr(system_spike, "freeze_request", forbidden_freeze)

    with pytest.raises(ValueError, match="only permits development and validation"):
        system_spike.run_experiment(
            tmp_path / "out",
            slug="demo",
            split_name="heldout",
            system_index=3,
        )


def test_live_fake_transport_freezes_request_before_truth_and_converts_chords_and_rests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = _benchmark_out(tmp_path)
    observed_payloads = []
    evaluation_calls = []

    def fake_transport(payload):
        observed_payloads.append(payload)
        assert "ground_truth" not in json.dumps(payload).lower()
        run_dir = out_dir / "experiments/vlm_system_transcription/live-test"
        assert (run_dir / "contact_sheet.png").exists()
        assert (run_dir / "request_payload.json").exists()
        assert not (run_dir / "provider_response.json").exists()
        return {
            "id": "resp_test",
            "output_text": json.dumps(_transcription_payload()),
            "usage": {"input_tokens": 100, "output_tokens": 30, "total_tokens": 130},
        }

    original_evaluate = system_spike.benchmark.evaluate_predictions

    def checking_evaluate(truth_rows, prediction_rows):
        run_dir = out_dir / "experiments/vlm_system_transcription/live-test"
        assert (run_dir / "provider_response.json").exists()
        assert (run_dir / "raw_response.txt").exists()
        assert (run_dir / "parsed_prediction.json").exists()
        assert (run_dir / "predictions.jsonl").exists()
        evaluation_calls.append((truth_rows, prediction_rows))
        return original_evaluate(truth_rows, prediction_rows)

    monkeypatch.setattr(system_spike.benchmark, "evaluate_predictions", checking_evaluate)
    report = system_spike.run_experiment(
        out_dir,
        slug="demo",
        split_name="development",
        system_index=1,
        experiment_id="live-test",
        model="gpt-test",
        reasoning_effort="high",
        detail="original",
        max_output_tokens=512,
        timeout_seconds=5,
        max_calls=1,
        force=True,
        transport=fake_transport,
    )

    assert report["status"] == "called"
    assert report["live_calls"] == 1
    assert report["evaluation_summary"]["targets"] == 2
    assert len(observed_payloads) == 1
    assert len(evaluation_calls) == 1
    payload = observed_payloads[0]
    assert payload["model"] == "gpt-test"
    assert payload["reasoning"] == {"effort": "high"}
    assert payload["max_output_tokens"] == 512
    assert payload["input"][0]["content"][1]["detail"] == "original"
    assert payload["text"]["format"]["strict"] is True
    assert "no refusal" in payload["instructions"].lower()
    assert "best-effort" in payload["instructions"].lower()

    predictions = _read_jsonl(
        out_dir / "experiments/vlm_system_transcription/live-test/predictions.jsonl"
    )
    assert predictions[0]["identity"]["global_measure_index"] == 0
    assert predictions[0]["notes"] == [
        {"onset_beats": 0, "duration_beats": 1, "pitch_midi": 60},
        {
            "onset_beats": 0,
            "duration_beats": 1,
            "pitch_midi": 64,
            "accidental": 0,
        },
    ]
    assert predictions[0]["rests"] == [{"onset_beats": 1, "duration_beats": 2}]
    assert predictions[1]["notes"] == [
        {"onset_beats": 0, "duration_beats": 3, "pitch_midi": 70, "accidental": -1}
    ]

    sheet_manifest = json.loads(
        (out_dir / "experiments/vlm_system_transcription/live-test/contact_sheet.json").read_text(
            encoding="utf-8"
        )
    )
    assert sheet_manifest["labels_outside_score_pixels"] is True
    assert [item["system_measure_index"] for item in sheet_manifest["placements"]] == [1, 2]
    for placement in sheet_manifest["placements"]:
        assert placement["label_box_px"][3] <= placement["score_box_px"][1]
    request_manifest = json.loads(
        (
            out_dir / "experiments/vlm_system_transcription/live-test/request_manifest.json"
        ).read_text(encoding="utf-8")
    )
    sheet_path = out_dir / "experiments/vlm_system_transcription/live-test/contact_sheet.png"
    assert (
        request_manifest["contact_sheet_sha256"]
        == hashlib.sha256(sheet_path.read_bytes()).hexdigest()
    )


def test_dry_run_never_constructs_transport_or_reads_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = _benchmark_out(tmp_path, write_truth=False)

    def forbidden_transport(_payload):
        raise AssertionError("dry run must not call transport")

    def forbidden_evaluation(_truth, _predictions):
        raise AssertionError("dry run must not evaluate")

    monkeypatch.setattr(system_spike.benchmark, "evaluate_predictions", forbidden_evaluation)
    report = system_spike.run_experiment(
        out_dir,
        slug="demo",
        split_name="development",
        system_index=1,
        experiment_id="dry-test",
        max_calls=0,
        transport=forbidden_transport,
    )

    assert report == {
        "schema_version": 1,
        "kind": "vlm_system_transcription_spike",
        "status": "dry_run",
        "live_calls": 0,
        "cache_key": report["cache_key"],
        "paths": {"run_dir": str(out_dir / "experiments/vlm_system_transcription/dry-test")},
    }
    run_dir = out_dir / "experiments/vlm_system_transcription/dry-test"
    assert (run_dir / "request_payload.json").exists()
    assert (run_dir / "replay.sh").exists()
    assert not (run_dir / "provider_response.json").exists()
    assert not (run_dir / "predictions.jsonl").exists()


def test_cached_response_avoids_transport_and_is_keyed_by_request(
    tmp_path: Path,
) -> None:
    out_dir = _benchmark_out(tmp_path)
    calls = 0

    def first_transport(_payload):
        nonlocal calls
        calls += 1
        return {"id": "cached", "output_text": json.dumps(_transcription_payload())}

    first = system_spike.run_experiment(
        out_dir,
        slug="demo",
        split_name="development",
        system_index=1,
        experiment_id="cache-first",
        max_calls=1,
        transport=first_transport,
    )

    def forbidden_transport(_payload):
        raise AssertionError("cache hit must not call transport")

    second = system_spike.run_experiment(
        out_dir,
        slug="demo",
        split_name="development",
        system_index=1,
        experiment_id="cache-second",
        max_calls=0,
        transport=forbidden_transport,
    )

    assert calls == 1
    assert first["cache_key"] == second["cache_key"]
    assert second["status"] == "cached"
    assert second["live_calls"] == 0
    assert (
        out_dir / "experiments/vlm_system_transcription/cache-second/provider_response.json"
    ).exists()


def test_schema_parser_rejects_omitted_measure_and_invalid_rest_pitch(tmp_path: Path) -> None:
    rows = _request_rows(tmp_path / "out")
    omitted = {"measures": [_transcription_payload()["measures"][0]]}
    with pytest.raises(ValueError, match="order/coverage mismatch"):
        system_spike.parse_transcription(json.dumps(omitted), expected_rows=rows)

    invalid = _transcription_payload()
    invalid["measures"][0]["events"][2]["pitch"] = "C4"
    with pytest.raises(ValueError, match="Rest pitch must be null"):
        system_spike.parse_transcription(json.dumps(invalid), expected_rows=rows)


def test_provider_failure_is_counted_and_journaled(tmp_path: Path) -> None:
    out_dir = _benchmark_out(tmp_path, write_truth=False)

    def failing_transport(_payload):
        raise RuntimeError("simulated provider failure")

    with pytest.raises(RuntimeError, match="simulated provider failure"):
        system_spike.run_experiment(
            out_dir,
            slug="demo",
            split_name="development",
            system_index=1,
            experiment_id="failed-call",
            max_calls=1,
            force=True,
            transport=failing_transport,
        )

    result = json.loads(
        (out_dir / "experiments/vlm_system_transcription/failed-call/result.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["status"] == "provider_error"
    assert result["live_calls"] == 1
    assert result["error"] == {
        "type": "RuntimeError",
        "message": "simulated provider failure",
    }


def _benchmark_out(tmp_path: Path, *, write_truth: bool = True) -> Path:
    out_dir = tmp_path / "out"
    split_dir = out_dir / "demo/vlm_melody_event_benchmark/development"
    split_dir.mkdir(parents=True)
    rows = _request_rows(out_dir)
    _write_jsonl(split_dir / "requests.jsonl", rows)
    if write_truth:
        rows_by_local = {row["identity"]["system_measure_index"]: row for row in rows}
        truth_rows = [
            {
                "identity": rows_by_local[1]["identity"],
                "measure_extent_beats": 3,
                "notes": [
                    {"onset_beats": 0, "duration_beats": 1, "pitch_midi": 60},
                    {"onset_beats": 0, "duration_beats": 1, "pitch_midi": 64},
                ],
                "rests": [{"onset_beats": 1, "duration_beats": 2}],
            },
            {
                "identity": rows_by_local[2]["identity"],
                "measure_extent_beats": 3,
                "notes": [{"onset_beats": 0, "duration_beats": 3, "pitch_midi": 70}],
                "rests": [],
            },
            {
                "identity": {
                    "slug": "demo",
                    "system_index": 2,
                    "system_measure_index": 1,
                    "global_measure_index": 2,
                },
                "measure_extent_beats": 3,
                "notes": [],
                "rests": [{"onset_beats": 0, "duration_beats": 3}],
            },
        ]
        _write_jsonl(split_dir / "truth.jsonl", truth_rows)
    return out_dir


def _request_rows(out_dir: Path) -> list[dict]:
    rows = []
    image_dir = out_dir / "demo/staff"
    image_dir.mkdir(parents=True, exist_ok=True)
    for local_index in (2, 1):
        path = image_dir / f"measure_{local_index:03d}.png"
        image = Image.new("L", (70 + local_index * 5, 30 + local_index), 255)
        image.putpixel((10, 10), 0)
        image.save(path)
        rows.append(
            {
                "schema_version": 1,
                "split": "development",
                "identity": {
                    "slug": "demo",
                    "system_index": 1,
                    "system_measure_index": local_index,
                    "global_measure_index": local_index - 1,
                },
                "images": {
                    "staff": {
                        "path_relative_to_out": str(path.relative_to(out_dir)),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "width_px": image.width,
                        "height_px": image.height,
                    }
                },
                "allowed_context": {
                    "clef": "treble",
                    "time_signature": "3/4",
                    "key_hint": "one flat: Bb",
                    "expected_measure_beats": "3",
                    "allow_pickup": False,
                },
            }
        )
    return rows


def _transcription_payload() -> dict:
    return {
        "measures": [
            {
                "system_measure_index": 1,
                "events": [
                    _event("note", 0, 1, "C4", None),
                    _event("note", 0, 1, "E4", "natural"),
                    _event("rest", 1, 2, None, None),
                ],
            },
            {
                "system_measure_index": 2,
                "events": [_event("note", 0, 3, "Bb4", "flat")],
            },
        ]
    }


def _event(kind, onset, duration, pitch, accidental) -> dict:
    return {
        "kind": kind,
        "onset_beats": onset,
        "duration_beats": duration,
        "pitch": pitch,
        "accidental": accidental,
        "confidence": 0.8,
        "evidence": "visible symbol",
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
