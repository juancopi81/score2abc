import json
from pathlib import Path
from typing import Any

import pytest

from scripts.experiments import evaluate_consumed_repaired_full_event_sidecar as evaluator


def test_verifies_sidecar_and_baseline_pin_before_opening_truth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    order: list[str] = []
    original_baseline_verifier = evaluator._verify_baseline_canonical_predictions_pin

    def verify_sidecar(path: Path) -> dict[str, Any]:
        order.append("sidecar")
        return {"verified": True, "measure_count": 1}

    def verify_baseline(path: Path, manifest: dict[str, Any]) -> tuple[Path, str]:
        order.append("baseline")
        return original_baseline_verifier(path, manifest)

    def load_truth(path: Path) -> list[dict[str, Any]]:
        order.append("truth")
        return evaluator._read_jsonl(path)

    monkeypatch.setattr(
        evaluator.materializer, "verify_repaired_full_event_sidecar", verify_sidecar
    )
    monkeypatch.setattr(evaluator, "_verify_baseline_canonical_predictions_pin", verify_baseline)

    evaluator.evaluate_consumed_repaired_full_event_sidecar(
        fixture["sidecar"],
        truth_snapshot=fixture["truth"],
        mapping=fixture["mapping"],
        output_dir=fixture["output"],
        truth_loader=load_truth,
    )

    assert order == ["sidecar", "baseline", "truth"]


def test_writes_consumed_metric_report_and_atomic_snapshots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, baseline_pitch=62, repaired_pitch=60)
    _stub_sidecar_verifier(monkeypatch)

    result = evaluator.evaluate_consumed_repaired_full_event_sidecar(
        fixture["sidecar"],
        truth_snapshot=fixture["truth"],
        mapping=fixture["mapping"],
        output_dir=fixture["output"],
    )

    output_dir = Path(result["evaluation_dir"])
    report = _read_json(output_dir / "report.json")
    manifest = _read_json(output_dir / "manifest.json")
    assert report["evidence_scope"] == "consumed_postmortem"
    assert report["runtime_promotion_supported"] is False
    assert report["lanes"]["repaired"]["metrics"]["summary"]["note_f1"] == 1.0
    assert report["metric_deltas_repaired_minus_baseline"]["note_f1"] > 0
    assert manifest["evidence_scope"] == "consumed_postmortem"
    assert manifest["runtime_promotion_supported"] is False
    assert manifest["truth_opened_after_all_candidate_model_sidecar_hashes_verified"] is True
    assert not (fixture["output"].parent / f".{fixture['output'].name}.tmp").exists()
    for name in (
        "report.json",
        "report.md",
        "baseline_predictions.jsonl",
        "repaired_predictions.jsonl",
        "truth.jsonl",
        "mapping.json",
        "sidecar_manifest.json",
        "manifest.json",
    ):
        assert (output_dir / name).is_file()
    assert (output_dir / "baseline_predictions.jsonl").read_bytes() == fixture[
        "baseline"
    ].read_bytes()
    assert (output_dir / "truth.jsonl").read_bytes() == fixture["truth"].read_bytes()
    assert set(manifest["pins"]) == {"sources", "snapshots", "implementations"}


def test_rejects_identity_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, truth_crop=2)
    _stub_sidecar_verifier(monkeypatch)

    with pytest.raises(ValueError, match="target identities are not aligned"):
        evaluator.evaluate_consumed_repaired_full_event_sidecar(
            fixture["sidecar"],
            truth_snapshot=fixture["truth"],
            mapping=fixture["mapping"],
            output_dir=fixture["output"],
        )

    assert not fixture["output"].exists()


def test_rejects_baseline_source_hash_drift_before_truth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _stub_sidecar_verifier(monkeypatch)
    fixture["baseline"].write_text("drift\n", encoding="utf-8")
    opened = False

    def load_truth(path: Path) -> list[dict[str, Any]]:
        nonlocal opened
        opened = True
        raise AssertionError("truth must remain unopened")

    with pytest.raises(ValueError, match="source hash drift"):
        evaluator.evaluate_consumed_repaired_full_event_sidecar(
            fixture["sidecar"],
            truth_snapshot=fixture["truth"],
            mapping=fixture["mapping"],
            output_dir=fixture["output"],
            truth_loader=load_truth,
        )

    assert opened is False


def test_create_once_refuses_before_reopening_truth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _stub_sidecar_verifier(monkeypatch)
    evaluator.evaluate_consumed_repaired_full_event_sidecar(
        fixture["sidecar"],
        truth_snapshot=fixture["truth"],
        mapping=fixture["mapping"],
        output_dir=fixture["output"],
    )
    opened = False

    def load_truth(path: Path) -> list[dict[str, Any]]:
        nonlocal opened
        opened = True
        raise AssertionError("truth must not reopen")

    with pytest.raises(FileExistsError, match="create-once"):
        evaluator.evaluate_consumed_repaired_full_event_sidecar(
            fixture["sidecar"],
            truth_snapshot=fixture["truth"],
            mapping=fixture["mapping"],
            output_dir=fixture["output"],
            truth_loader=load_truth,
        )

    assert opened is False


def test_meter_valid_extent_does_not_hide_wrong_duration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, repaired_duration=2.0)
    _stub_sidecar_verifier(monkeypatch)

    result = evaluator.evaluate_consumed_repaired_full_event_sidecar(
        fixture["sidecar"],
        truth_snapshot=fixture["truth"],
        mapping=fixture["mapping"],
        output_dir=fixture["output"],
    )

    report = _read_json(Path(result["report"]))
    repaired = report["lanes"]["repaired"]
    assert repaired["meter_validity"]["summary"]["valid_measure_rate"] == 1.0
    assert repaired["meter_validity"]["duration_or_rest_accuracy_included"] is False
    assert repaired["metrics"]["summary"]["ordered_duration_accuracy"] == 0.0
    assert repaired["metrics"]["summary"]["rest_f1"] == 0.0


def test_records_explicit_multi_measure_crop_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, physical_measure_numbers=[6, 7])
    _stub_sidecar_verifier(monkeypatch)

    result = evaluator.evaluate_consumed_repaired_full_event_sidecar(
        fixture["sidecar"],
        truth_snapshot=fixture["truth"],
        mapping=fixture["mapping"],
        output_dir=fixture["output"],
    )

    report = _read_json(Path(result["report"]))
    assert report["identity_adaptation"]["explicit_physical_measure_mapping"] == {"1": [6, 7]}
    assert _read_json(Path(result["evaluation_dir"]) / "mapping.json")["automatic_crops"][0][
        "physical_measure_numbers"
    ] == [6, 7]


def test_rejects_mapping_that_does_not_cover_evaluated_crop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _stub_sidecar_verifier(monkeypatch)
    fixture["mapping"].write_text(
        json.dumps(
            {
                "schema_version": 1,
                "automatic_crops": [
                    {"automatic_crop_index": 1, "physical_measure_numbers": [1]},
                    {"automatic_crop_index": 2, "physical_measure_numbers": [2]},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match evaluated automatic crops"):
        evaluator.evaluate_consumed_repaired_full_event_sidecar(
            fixture["sidecar"],
            truth_snapshot=fixture["truth"],
            mapping=fixture["mapping"],
            output_dir=fixture["output"],
        )


def _stub_sidecar_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        evaluator.materializer,
        "verify_repaired_full_event_sidecar",
        lambda path: {"verified": True, "measure_count": 1},
    )


def _fixture(
    root: Path,
    *,
    baseline_pitch: int = 60,
    repaired_pitch: int = 60,
    repaired_duration: float = 1.0,
    truth_crop: int = 1,
    physical_measure_numbers: list[int] | None = None,
) -> dict[str, Path]:
    inference_dir = root / "inference"
    sidecar_dir = inference_dir / "repaired_full_event_v1"
    sidecar_dir.mkdir(parents=True)
    identity = {
        "slug": "consumed-score",
        "system_index": 2,
        "automatic_measure_index": 1,
    }
    baseline_row = _prediction(identity, pitch=baseline_pitch, duration=1.0)
    repaired_row = _prediction(identity, pitch=repaired_pitch, duration=repaired_duration)
    baseline_path = inference_dir / "predictions.jsonl"
    repaired_path = sidecar_dir / "predictions.jsonl"
    _write_jsonl(baseline_path, [baseline_row])
    _write_jsonl(repaired_path, [repaired_row])
    manifest = {
        "schema_version": 1,
        "kind": "repaired_full_event_sidecar_manifest",
        "target": {"slug": "consumed-score", "system_index": 2},
        "canonical": {
            "predictions": {
                "path": "../predictions.jsonl",
                "sha256": evaluator._sha256(baseline_path),
            }
        },
        "artifacts": {
            "predictions.jsonl": {
                "path": "predictions.jsonl",
                "sha256": evaluator._sha256(repaired_path),
            }
        },
    }
    manifest_path = sidecar_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    truth_identity = {
        "slug": "consumed-score",
        "system_index": 2,
        "automatic_measure_index": truth_crop,
    }
    truth_path = root / "consumed_truth.jsonl"
    _write_jsonl(truth_path, [_truth(truth_identity)])
    mapping_path = root / "crop_mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "automatic_crops": [
                    {
                        "automatic_crop_index": 1,
                        "physical_measure_numbers": physical_measure_numbers or [1],
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "sidecar": sidecar_dir,
        "baseline": baseline_path,
        "repaired": repaired_path,
        "truth": truth_path,
        "mapping": mapping_path,
        "output": root / "consumed_evaluation_v1",
    }


def _prediction(
    identity: dict[str, Any],
    *,
    pitch: int,
    duration: float,
) -> dict[str, Any]:
    rests = [{"onset_beats": duration, "duration_beats": 3.0 - duration}] if duration < 3.0 else []
    return {
        "schema_version": 1,
        "identity": dict(identity),
        "measure_extent_beats": 3,
        "notes": [
            {
                "pitch_midi": pitch,
                "onset_beats": 0.0,
                "duration_beats": duration,
            }
        ],
        "rests": rests,
    }


def _truth(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "identity": dict(identity),
        "measure_extent_beats": 3,
        "notes": [{"pitch_midi": 60, "onset_beats": 0.0, "duration_beats": 1.0}],
        "rests": [{"onset_beats": 1.0, "duration_beats": 2.0}],
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
