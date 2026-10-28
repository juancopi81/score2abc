import copy
import json

import pytest

from scripts.experiments import evaluate_astra_collection_pilot as pilot


def test_coqueteos_baseline_adapter_changes_only_legacy_identity_fields():
    rows = [
        {
            "identity": {
                "slug": "coqueteos",
                "system_index": 2,
                "system_measure_index": crop,
                "automatic_measure_index": crop,
            },
            "notes": [{"pitch_midi": 60, "onset_beats": 0, "duration_beats": 1}],
            "rests": [],
        }
        for crop in (2, 1)
    ]
    before = copy.deepcopy(rows)
    adapted = pilot.coqueteos_benchmark_identities(rows)
    assert rows == before
    assert [row["identity"]["global_measure_index"] for row in adapted] == [1, 2]
    assert [row["identity"]["system_measure_index"] for row in adapted] == [1, 2]
    assert adapted[0]["notes"] == rows[1]["notes"]
    assert adapted[1]["rests"] == rows[0]["rests"]


def test_written_truth_keeps_tied_continuation_chord_and_explicit_rest(tmp_path):
    path = tmp_path / "source.musicxml"
    path.write_text("""<score-partwise><part id="P1">
      <measure number="8"><attributes><divisions>2</divisions>
      <time><beats>3</beats><beat-type>4</beat-type></time></attributes>
      <note><pitch><step>B</step><alter>-1</alter><octave>4</octave></pitch>
      <duration>6</duration><tie type="start"/></note></measure>
      <measure number="9">
      <note><pitch><step>B</step><alter>-1</alter><octave>4</octave></pitch>
      <duration>2</duration><tie type="stop"/></note>
      <note><chord/><pitch><step>D</step><octave>5</octave></pitch><duration>2</duration></note>
      <note><rest/><duration>2</duration></note>
      <note><pitch><step>B</step><octave>4</octave></pitch><duration>2</duration>
      <accidental>natural</accidental></note></measure></part></score-partwise>""")
    rows = pilot.written_truth(path, "example", 2, 100)
    assert [row["identity"]["global_measure_index"] for row in rows] == [101, 102]
    assert [row["source_measure_number"] for row in rows] == ["8", "9"]
    assert rows[0]["notes"] == [{"onset_beats": 0, "duration_beats": 3, "pitch_midi": 70}]
    assert rows[1]["notes"] == [
        {"onset_beats": 0, "duration_beats": 1, "pitch_midi": 70},
        {"onset_beats": 0, "duration_beats": 1, "pitch_midi": 74},
        {"onset_beats": 2, "duration_beats": 1, "pitch_midi": 71},
    ]
    assert rows[1]["rests"] == [{"onset_beats": 1, "duration_beats": 1}]
    assert pilot.meter_valid_count(rows, rows) == 2


def test_historical_mapping_keeps_missing_measures_empty_and_offsets_without_mutation():
    rows = [
        {
            "identity": {"system_measure_index": 3},
            "rests": [],
            "notes": [{"onset_beats": 0.5, "duration_beats": 0.5, "pitch_midi": 71}],
        }
    ]
    before = copy.deepcopy(rows)
    historical = [{"identity": {"system_measure_index": 2}}]
    result = pilot.compose(rows, historical, {"2": [2, 3]})
    assert result[0]["notes"] == [{"onset_beats": 3.5, "duration_beats": 0.5, "pitch_midi": 71}]
    assert rows == before


def test_changed_sealed_file_rejected_before_reference_access(tmp_path, monkeypatch):
    plan = {"targets": [{"id": target} for target in pilot.TARGET_IDS]}
    (tmp_path / "selection_plan.json").write_text(json.dumps(plan))
    protocol = {
        "status": pilot.PROTOCOL_STATUS,
        "model": pilot.MODEL,
        "workers": dict.fromkeys(pilot.TARGET_IDS, {}),
        "selection_plan_sha256": pilot.benchmark._sha256(tmp_path / "selection_plan.json"),
    }
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    changed = tmp_path / "changed.json"
    changed.write_text("original")
    digest = pilot.benchmark._sha256(changed)
    changed.write_text("tampered")
    deviation = {
        "source_only_protocol_passed": False,
        "allowed_label": pilot.REPORT_LABEL,
    }
    (tmp_path / "protocol_deviation.json").write_text(json.dumps(deviation))
    seal = {
        "kind": pilot.SEAL_KIND,
        "status": pilot.SEAL_STATUS,
        "protocol_sha256": pilot.benchmark._sha256(tmp_path / "protocol.json"),
        "protocol_deviation_sha256": pilot.benchmark._sha256(tmp_path / "protocol_deviation.json"),
        "submissions": dict.fromkeys(pilot.TARGET_IDS, {}),
        "files": [{"path": "changed.json", "sha256": digest}],
    }
    (tmp_path / "submission_seal.json").write_text(json.dumps(seal))

    def no_truth(*args, **kwargs):
        pytest.fail("Truth was opened before all submissions passed verification")

    monkeypatch.setattr(pilot, "written_truth", no_truth)
    with pytest.raises(ValueError, match="Pinned file missing or changed"):
        pilot.evaluate(tmp_path)
    assert not (tmp_path / "evaluation_v1").exists()


def test_malformed_submission_is_a_target_diagnostic_instead_of_an_exception():
    identity = {
        "slug": "synthetic",
        "system_index": 1,
        "system_measure_index": 1,
        "global_measure_index": 1,
    }
    predictions, validation = pilot.adapt_submission(
        {"measures": "not-an-array", "context": {}}, [identity], [1]
    )
    assert predictions == []
    assert validation["status"] == "invalid_submission"
    assert validation["headline_metrics_valid"] is False
    assert validation["validation_errors"] == ["Submission requires measures and context"]


def test_historical_mapping_must_partition_physical_measures_exactly_once():
    historical = [
        {"identity": {"system_measure_index": 1}},
        {"identity": {"system_measure_index": 2}},
    ]
    with pytest.raises(ValueError, match="partition every physical measure exactly once"):
        pilot.validate_mapping({"1": [1, 1], "2": [2]}, 3, historical)
    with pytest.raises(ValueError, match="exactly cover historical crops"):
        pilot.validate_mapping({"1": [1], "3": [2, 3]}, 3, historical)


def test_historical_full_event_comparison_withholds_mismatched_tie_semantics(monkeypatch):
    identity = {
        "slug": "synthetic",
        "system_index": 1,
        "system_measure_index": 1,
        "global_measure_index": 1,
    }
    old_truth = [
        {
            "identity": identity,
            "measure_extent_beats": 6,
            "notes": [{"onset_beats": 0, "duration_beats": 4, "pitch_midi": 70}],
            "rests": [],
        }
    ]
    baseline = copy.deepcopy(old_truth)
    native_truth = [
        {
            "identity": {**identity, "system_measure_index": 1},
            "notes": [{"onset_beats": 0, "duration_beats": 3, "pitch_midi": 70}],
            "rests": [],
        },
        {
            "identity": {**identity, "system_measure_index": 2},
            "notes": [{"onset_beats": 0, "duration_beats": 1, "pitch_midi": 70}],
            "rests": [],
        },
    ]
    target = {
        "id": "synthetic",
        "human_truth": {"canonical_full_event_snapshot": {"path": "truth.jsonl"}},
        "historical_baseline": {
            "predictions_path": "predictions.jsonl",
            "report_path": "report.json",
        },
        "historical_mapping": {"automatic_crop_to_physical_measures": {"1": [1, 2]}},
        "metric_comparability": {"limitation": "synthetic"},
    }

    def rows(path):
        return old_truth if str(path).endswith("truth.jsonl") else baseline

    monkeypatch.setattr(pilot.benchmark, "_read_jsonl", rows)
    monkeypatch.setattr(
        pilot,
        "read_json",
        lambda path: {"metrics": pilot.benchmark.evaluate_predictions(old_truth, baseline)},
    )
    comparison = pilot.full_event_comparison(target, native_truth, [], False)
    assert comparison["baseline_reproduced"] is True
    assert comparison["supported"] is False
    assert comparison["astra"] is None
    assert "tie or event semantics" in comparison["unsupported_reason"]


def test_aggregate_identities_do_not_collide_across_scores():
    truths, predictions = [], []
    for ordinal in range(3):
        identity = {
            "slug": f"score-{ordinal}",
            "system_index": 1,
            "system_measure_index": 1,
            "global_measure_index": ordinal * 100 + 1,
        }
        note = {"pitch_midi": 60 + ordinal, "onset_beats": 0, "duration_beats": 3}
        truths.append(
            {"identity": identity, "measure_extent_beats": 3, "notes": [note], "rests": []}
        )
        predictions.append({"identity": identity, "notes": [note], "rests": []})
    assert pilot.benchmark.evaluate_predictions(truths, predictions)["summary"]["note_f1"] == 1
