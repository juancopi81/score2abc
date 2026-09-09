import json
from pathlib import Path
from typing import Any

import pytest

from scripts.experiments import evaluate_frozen_fifth_score_heldout as spike
from scripts.experiments import evaluate_frozen_third_score_heldout as shared_evaluator
from scripts.experiments import freeze_third_score_heldout as freezer
from tests.test_evaluate_frozen_third_score_heldout import _frozen_fixture


def test_verifies_gate_before_opening_truth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    musicxml = _write_six_measure_musicxml(tmp_path / "truth.musicxml")
    opened = False

    def reject_gate(path: Path) -> dict[str, Any]:
        raise ValueError("synthetic frozen verification failure")

    def truth_loader(path: Path) -> Any:
        nonlocal opened
        opened = True
        raise AssertionError("truth must remain unopened")

    monkeypatch.setattr(spike.heldout, "verify_frozen_gate", reject_gate)

    with pytest.raises(ValueError, match="synthetic frozen verification failure"):
        spike.evaluate_frozen_fifth_score(
            tmp_path / "sealed_manifest.json",
            musicxml_path=musicxml,
            truth_loader=truth_loader,
        )

    assert opened is False


def test_scores_full_events_and_pins_atomic_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _verified_state(tmp_path)
    musicxml = _write_six_measure_musicxml(tmp_path / "six.musicxml")
    monkeypatch.setattr(
        spike.heldout,
        "verify_frozen_gate",
        lambda path: fixture["verified"],
    )

    result = spike.evaluate_frozen_fifth_score(
        fixture["sealed"],
        musicxml_path=musicxml,
    )

    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    summary = report["metrics"]["summary"]
    assert summary["note_f1"] == 1.0
    assert summary["ordered_pitch_accuracy"] == 1.0
    assert summary["ordered_onset_accuracy"] == 1.0
    assert summary["ordered_duration_accuracy"] == 1.0
    assert summary["rest_precision"] == 1.0
    assert summary["rest_recall"] == 1.0
    assert summary["rest_f1"] == 1.0
    assert summary["exact_measures"] == 6
    assert summary["meter_context_match"] is True
    assert summary["meter_valid_crops"] == 6
    assert summary["meter_valid_crop_rate"] == 1.0
    assert report["mapping_mode"] == "deterministic_default_one_to_one"
    assert report["meter"]["summary"]["valid_truth_measures"] == 6

    evaluation_dir = Path(result["evaluation_dir"])
    assert not (fixture["namespace_root"] / ".evaluation_v1.tmp").exists()
    assert (evaluation_dir / "source.musicxml").read_bytes() == musicxml.read_bytes()
    assert len(_read_jsonl(Path(result["truth"]))) == 6
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["truth_opened_after_all_frozen_hashes_verified"] is True
    for role in (
        "source_musicxml",
        "mapping",
        "truth",
        "frozen_prepared_manifest",
        "frozen_requests",
        "frozen_predictions",
        "frozen_freeze_manifest",
        "frozen_sealed_manifest",
        "evaluator",
        "heldout_verifier",
        "musicxml_truth_loader",
        "event_benchmark",
        "report",
    ):
        record = manifest["pins"][role]
        snapshot = evaluation_dir / record["snapshot_path"]
        assert snapshot.is_file()
        assert freezer._sha256(snapshot) == record["snapshot_sha256"]


def test_create_once_refuses_overwrite_before_reopening_truth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _verified_state(tmp_path)
    musicxml = _write_six_measure_musicxml(tmp_path / "six.musicxml")
    monkeypatch.setattr(
        spike.heldout,
        "verify_frozen_gate",
        lambda path: fixture["verified"],
    )
    spike.evaluate_frozen_fifth_score(fixture["sealed"], musicxml_path=musicxml)
    opened = False

    def truth_loader(path: Path) -> Any:
        nonlocal opened
        opened = True
        raise AssertionError("truth must not be reopened")

    with pytest.raises(FileExistsError, match="already exists"):
        spike.evaluate_frozen_fifth_score(
            fixture["sealed"],
            musicxml_path=musicxml,
            evaluation_version="v2",
            truth_loader=truth_loader,
        )

    assert opened is False


def test_frozen_hash_drift_is_rejected_before_truth(tmp_path: Path) -> None:
    fixture = _frozen_fixture(
        tmp_path,
        predictions=[[60]] * 6,
        evaluation_spec=shared_evaluator.FIFTH_SCORE_EVALUATION,
    )
    fixture["model_snapshot"].write_text("drift\n", encoding="utf-8")
    opened = False

    def truth_loader(path: Path) -> Any:
        nonlocal opened
        opened = True
        raise AssertionError("truth must remain unopened")

    with pytest.raises(ValueError, match="Frozen snapshot hash drift"):
        spike.evaluate_frozen_fifth_score(
            fixture["sealed"],
            musicxml_path=_write_six_measure_musicxml(tmp_path / "truth.musicxml"),
            truth_loader=truth_loader,
        )

    assert opened is False


def test_partial_measure_mapping_fails_without_partial_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _verified_state(tmp_path)
    musicxml = _write_six_measure_musicxml(tmp_path / "six.musicxml")
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "automatic_crops": [
                    {
                        "automatic_crop_index": index,
                        "physical_measure_spans": [
                            {
                                "measure_number": index,
                                **({"note_start": 0} if index == 1 else {}),
                            }
                        ],
                    }
                    for index in range(1, 7)
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        spike.heldout,
        "verify_frozen_gate",
        lambda path: fixture["verified"],
    )

    with pytest.raises(ValueError, match="Partial-measure note spans"):
        spike.evaluate_frozen_fifth_score(
            fixture["sealed"],
            musicxml_path=musicxml,
            mapping_path=mapping,
        )

    assert not (fixture["namespace_root"] / "evaluation_v1").exists()
    assert not (fixture["namespace_root"] / ".evaluation_v1.tmp").exists()


def _verified_state(tmp_path: Path) -> dict[str, Any]:
    namespace_root = tmp_path / "synthetic" / "fifth" / "v1" / "system_002"
    namespace_root.mkdir(parents=True)
    requests_path = namespace_root / "requests.jsonl"
    predictions_path = namespace_root / "frozen_predictions.jsonl"
    prepared_path = namespace_root / "prepared_manifest.json"
    freeze_path = namespace_root / "freeze.json"
    sealed_path = namespace_root / "sealed_manifest.json"
    context_path = namespace_root / "context/allowed_context.json"
    context_path.parent.mkdir()

    requests = []
    predictions = []
    for index in range(1, 7):
        identity = {
            "slug": "synthetic-score",
            "system_index": 2,
            "automatic_measure_index": index,
        }
        requests.append({"identity": identity})
        predictions.append(
            {
                "identity": {
                    **identity,
                    "system_measure_index": index,
                },
                "measure_extent_beats": 3,
                "notes": [
                    {"pitch_midi": 60, "onset_beats": 0, "duration_beats": 1},
                    {"pitch_midi": 62, "onset_beats": 2, "duration_beats": 1},
                ],
                "rests": [{"onset_beats": 1, "duration_beats": 1}],
            }
        )
    freezer._write_jsonl(requests_path, requests)
    freezer._write_jsonl(predictions_path, predictions)
    freezer._write_json(
        context_path,
        {
            "truth_accessed": False,
            "truth_used": False,
            "allowed_context": {
                "time_signature": "3/4",
                "expected_measure_beats": 3,
                "allow_pickup": False,
                "clef": "treble",
                "key_hint": None,
            },
        },
    )
    freezer._write_json(
        prepared_path,
        {
            "artifacts": {
                "context": {
                    "allowed_context": {
                        "path": context_path.relative_to(namespace_root).as_posix(),
                        "sha256": freezer._sha256(context_path),
                    }
                }
            }
        },
    )
    freezer._write_json(
        freeze_path,
        {
            "requests": {
                "path": requests_path.relative_to(namespace_root).as_posix(),
                "sha256": freezer._sha256(requests_path),
            },
            "predictions": {
                "snapshot_path_relative_to_namespace": predictions_path.relative_to(
                    namespace_root
                ).as_posix(),
                "snapshot_sha256": freezer._sha256(predictions_path),
            },
        },
    )
    freezer._write_json(sealed_path, {"kind": "synthetic-sealed"})
    verified = {
        "namespace_root": namespace_root,
        "sealed_sha256": freezer._sha256(sealed_path),
        "freeze_path": freeze_path,
        "freeze_sha256": freezer._sha256(freeze_path),
        "freeze": json.loads(freeze_path.read_text(encoding="utf-8")),
        "prepared_path": prepared_path,
        "prepared_sha256": freezer._sha256(prepared_path),
        "target": {"slug": "synthetic-score", "system_index": 2},
        "requests_by_crop": {index: row for index, row in enumerate(requests, start=1)},
        "predictions_by_crop": {index: row for index, row in enumerate(predictions, start=1)},
        "evaluation_spec": shared_evaluator.FIFTH_SCORE_EVALUATION,
    }
    return {
        "namespace_root": namespace_root,
        "sealed": sealed_path,
        "verified": verified,
    }


def _write_six_measure_musicxml(path: Path) -> Path:
    measures = []
    for number in range(1, 7):
        attributes = ""
        if number == 1:
            attributes = (
                "<attributes><divisions>2</divisions><key><fifths>-1</fifths></key>"
                "<time><beats>3</beats><beat-type>4</beat-type></time>"
                "<clef><sign>G</sign><line>2</line></clef></attributes>"
            )
        measures.append(
            f'<measure number="{number}">{attributes}'
            "<note><pitch><step>C</step><octave>4</octave></pitch><duration>2</duration></note>"
            "<note><rest/><duration>2</duration></note>"
            "<note><pitch><step>D</step><octave>4</octave></pitch><duration>2</duration></note>"
            "</measure>"
        )
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<score-partwise version="4.0">'
        '<part-list><score-part id="P1"><part-name>Music</part-name></score-part></part-list>'
        f'<part id="P1">{"".join(measures)}</part>'
        "</score-partwise>\n",
        encoding="utf-8",
    )
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
