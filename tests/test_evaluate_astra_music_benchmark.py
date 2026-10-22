import hashlib
import json
from pathlib import Path

import pytest

from scripts import build_vlm_melody_event_benchmark as benchmark
from scripts.experiments import evaluate_astra_music_benchmark as evaluator


def test_scores_polyphony_and_duration_and_penalizes_missing_measure(tmp_path: Path) -> None:
    experiment = _sealed_fixture(tmp_path)
    arm_b = _submission()
    arm_b["measures"][0]["events"][1]["duration_beats"] = 0.5
    del arm_b["measures"][1]
    _reseal_arm(experiment, "b", arm_b)

    result = evaluator.evaluate_astra_music_benchmark(experiment)
    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))

    assert report["lanes"]["baseline"]["historical_summary_reproduced"] is True
    assert report["lanes"]["baseline"]["metrics"]["summary"]["exact_measures"] == 7
    assert (
        report["lanes"]["a"]["metrics"]["summary"]
        == report["lanes"]["baseline"]["metrics"]["summary"]
    )
    arm_b_report = report["lanes"]["b"]
    assert arm_b_report["submission_validation"]["missing_measure_indices"] == [2]
    assert arm_b_report["submission_validation"]["headline_metrics_valid"] is True
    assert arm_b_report["metrics"]["summary"]["targets"] == 7
    assert arm_b_report["metrics"]["summary"]["predicted"] == 6
    assert arm_b_report["metrics"]["summary"]["exact_measures"] == 5
    assert arm_b_report["metrics"]["summary"]["ordered_duration_accuracy"] < 1.0


def test_extra_measure_is_explicit_and_invalidates_headline_arm(tmp_path: Path) -> None:
    experiment = _sealed_fixture(tmp_path)
    arm_a = _submission()
    arm_a["measures"].append(_measure(8))
    _reseal_arm(experiment, "a", arm_a)

    result = evaluator.evaluate_astra_music_benchmark(experiment)
    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    arm = report["lanes"]["a"]

    assert arm["submission_validation"]["extra_measure_indices"] == [8]
    assert arm["submission_validation"]["headline_metrics_valid"] is False
    assert arm["metrics"]["summary"]["targets"] == 7
    assert arm["metrics"]["summary"]["predicted"] == 0
    assert arm["metrics"]["summary"]["note_f1"] == 0.0
    assert arm["metrics"]["summary"]["exact_measures"] == 0


def test_overfull_event_is_reported_without_aborting_comparison(tmp_path: Path) -> None:
    experiment = _sealed_fixture(tmp_path)
    arm_a = _submission()
    arm_a["measures"][3]["events"][0]["duration_beats"] = 4
    _reseal_arm(experiment, "a", arm_a)

    result = evaluator.evaluate_astra_music_benchmark(experiment)
    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    arm = report["lanes"]["a"]

    assert arm["submission_validation"]["status"] == "invalid_event_geometry"
    assert arm["submission_validation"]["headline_metrics_valid"] is False
    assert arm["submission_validation"]["event_geometry_errors"][0]["measure_index"] == 4
    assert (
        "extends past measure" in arm["submission_validation"]["event_geometry_errors"][0]["error"]
    )
    assert arm["metrics"]["summary"]["note_f1"] == 0.0
    assert report["lanes"]["b"]["metrics"]["summary"]["exact_measures"] == 7


def test_altered_seal_fails_before_truth_loader_and_existing_output_is_refused(
    tmp_path: Path,
) -> None:
    experiment = _sealed_fixture(tmp_path)
    seal_path = experiment / "submission_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["protocol_sha256"] = "0" * 64
    _write_json(seal_path, seal)
    called = False

    def forbidden_truth_loader(_path: Path) -> list[dict]:
        nonlocal called
        called = True
        raise AssertionError("truth must remain unopened")

    with pytest.raises(ValueError, match="Protocol hash"):
        evaluator.evaluate_astra_music_benchmark(
            experiment,
            truth_loader=forbidden_truth_loader,
        )
    assert called is False

    output_dir = experiment / "already-there"
    output_dir.mkdir()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        evaluator.evaluate_astra_music_benchmark(
            experiment,
            output_dir=output_dir,
            truth_loader=forbidden_truth_loader,
        )
    assert called is False


def test_reused_archive_path_is_rejected_before_truth(tmp_path: Path) -> None:
    experiment = _sealed_fixture(tmp_path)
    seal_path = experiment / "submission_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["submissions"]["b"] = seal["submissions"]["a"]
    _write_json(seal_path, seal)
    called = False

    def forbidden(_path: Path) -> list[dict]:
        nonlocal called
        called = True
        return []

    with pytest.raises(ValueError, match="distinct archived submission paths"):
        evaluator.evaluate_astra_music_benchmark(experiment, truth_loader=forbidden)
    assert called is False


def test_arm_input_pin_is_rechecked_after_truth_load(tmp_path: Path) -> None:
    experiment = _sealed_fixture(tmp_path)
    protocol = json.loads((experiment / "protocol.json").read_text(encoding="utf-8"))
    input_path = Path(protocol["arms"]["a"]["input_pins"][0]["path"])

    def mutate_input_then_load(path: Path) -> list[dict]:
        input_path.write_text("changed after initial verification\n", encoding="utf-8")
        return benchmark._read_jsonl(path)

    with pytest.raises(ValueError, match="source recheck hash mismatch"):
        evaluator.evaluate_astra_music_benchmark(
            experiment,
            truth_loader=mutate_input_then_load,
        )
    assert not (experiment / "evaluation_v1").exists()


def _sealed_fixture(tmp_path: Path) -> Path:
    experiment = tmp_path / "astra"
    experiment.mkdir()
    requests_path = experiment / "requests.jsonl"
    truth_path = experiment / "truth.jsonl"
    baseline_path = experiment / "baseline.jsonl"
    inference_path = experiment / "baseline_inference.jsonl"
    freeze_path = experiment / "baseline_freeze.json"
    baseline_report_path = experiment / "baseline_report.json"

    identities = [_identity(index) for index in range(1, 8)]
    requests = [{"identity": identity} for identity in identities]
    truth = [_truth(index, identity) for index, identity in enumerate(identities, start=1)]
    baseline = _predictions(_submission(), identities)
    _write_jsonl(requests_path, requests)
    _write_jsonl(truth_path, truth)
    _write_jsonl(baseline_path, baseline)
    inference_path.write_text("synthetic frozen inference\n", encoding="utf-8")
    _write_json(
        freeze_path,
        {
            "schema_version": 1,
            "status": "frozen_before_truth",
            "split": "validation",
            "target_count": 7,
            "requests": {"path": str(requests_path), "sha256": _sha256(requests_path)},
            "artifacts": [
                {"path": str(baseline_path), "sha256": _sha256(baseline_path)},
                {"path": str(inference_path), "sha256": _sha256(inference_path)},
            ],
        },
    )
    _write_json(
        baseline_report_path,
        {"validation": {"metrics": benchmark.evaluate_predictions(truth, baseline)}},
    )

    original_root = tmp_path / "original"
    for arm in ("a", "b"):
        original_dir = original_root / arm
        archive_dir = experiment / "arms" / arm
        (original_dir / "input").mkdir(parents=True)
        (original_dir / "result").mkdir()
        (archive_dir / "result").mkdir(parents=True)
        (original_dir / "input/task.txt").write_text(f"synthetic arm {arm}\n", encoding="utf-8")
        _write_json(original_dir / "result/submission.json", _submission())
        _write_json(archive_dir / "result/submission.json", _submission())
    protocol = {
        "schema_version": 1,
        "status": "preregistered_before_new_inference",
        "target": {
            "slug": "demo",
            "system_index": 8,
            "expected_measure_indices": list(range(1, 8)),
        },
        "dataset_status": "synthetic_consumed_development",
        "arms": {
            arm: {
                "directory": str(original_root / arm),
                "input": f"synthetic input {arm}",
                "input_pins": [_pin(original_root / arm / "input/task.txt")],
            }
            for arm in ("a", "b")
        },
        "reference_pins": {
            "requests": _pin(requests_path),
            "truth": _pin(truth_path),
            "baseline_predictions": _pin(baseline_path),
            "baseline_freeze": _pin(freeze_path),
            "baseline_report": _pin(baseline_report_path),
            "evaluator": _pin(Path(benchmark.__file__).resolve()),
        },
    }
    protocol_path = experiment / "protocol.json"
    _write_json(protocol_path, protocol)
    _write_json(
        experiment / "submission_seal.json",
        {
            "schema_version": 1,
            "kind": evaluator.SEAL_KIND,
            "status": "frozen_before_truth",
            "protocol_sha256": _sha256(protocol_path),
            "truth_accessed": False,
            "submissions": {
                arm: _pin(experiment / "arms" / arm / "result/submission.json")
                for arm in ("a", "b")
            },
            "artifacts": [
                _pin(experiment / "arms" / arm / "result/submission.json") for arm in ("a", "b")
            ],
        },
    )
    return experiment


def _submission() -> dict:
    return {
        "measures": [_measure(index) for index in range(1, 8)],
        "context": {"clef": "treble", "time_signature": "3/4"},
        "notation_uncertainties": [],
    }


def _measure(index: int) -> dict:
    if index == 1:
        events = [
            _event("note", 0, 1, "E4"),
            _event("note", 0, 1, "C4"),
            _event("rest", 1, 2, None),
        ]
    elif index == 2:
        events = [_event("note", 0, 2, "D4"), _event("rest", 2, 1, None)]
    else:
        letter = ["C", "D", "E", "F", "G"][(index - 3) % 5]
        events = [_event("note", 0, 3, f"{letter}5")]
    return {"measure_index": index, "events": events, "uncertainties": []}


def _event(kind: str, onset: float, duration: float, pitch: str | None) -> dict:
    return {
        "kind": kind,
        "onset_beats": onset,
        "duration_beats": duration,
        "pitch": pitch,
        "accidental": None,
    }


def _truth(index: int, identity: dict) -> dict:
    prediction = _predictions({"measures": [_measure(index)]}, [identity])[0]
    return {
        "identity": identity,
        "measure_extent_beats": 3,
        "notes": prediction["notes"],
        "rests": prediction["rests"],
    }


def _predictions(payload: dict, identities: list[dict]) -> list[dict]:
    identity_by_index = {identity["system_measure_index"]: identity for identity in identities}
    rows = []
    for measure in payload["measures"]:
        notes, rests = [], []
        for event in measure["events"]:
            converted = {
                "onset_beats": event["onset_beats"],
                "duration_beats": event["duration_beats"],
            }
            if event["kind"] == "rest":
                rests.append(converted)
            else:
                notes.append(
                    {**converted, "pitch_midi": evaluator.pitch_name_to_midi(event["pitch"])}
                )
        rows.append(
            {
                "identity": identity_by_index[measure["measure_index"]],
                "notes": notes,
                "rests": rests,
            }
        )
    return rows


def _identity(index: int) -> dict:
    return {
        "slug": "demo",
        "system_index": 8,
        "system_measure_index": index,
        "global_measure_index": 52 + index,
    }


def _reseal_arm(experiment: Path, arm: str, payload: dict) -> None:
    path = experiment / "arms" / arm / "result/submission.json"
    protocol = json.loads((experiment / "protocol.json").read_text(encoding="utf-8"))
    original = Path(protocol["arms"][arm]["directory"]) / "result/submission.json"
    _write_json(path, payload)
    _write_json(original, payload)
    seal_path = experiment / "submission_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["submissions"][arm] = _pin(path)
    for index, record in enumerate(seal["artifacts"]):
        if record["path"] == str(path):
            seal["artifacts"][index] = _pin(path)
    _write_json(seal_path, seal)


def _pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
