"""Hermetic checks of edition metrics and the registered-run boundary."""

import copy
import hashlib
import json

import pytest

from score2abc import score_eval


def note(pitch, *, onset=0, duration=384):
    return {
        "onset_ticks": onset,
        "duration_ticks": duration,
        "pitch_midi": pitch,
        "voice": "1",
        "tie_start": False,
        "tie_end": False,
    }


def bar(number, pitch, *, chord="C", notes=None, meter_valid=True):
    return {
        "number": number,
        "duration_ticks": 1152,
        "expected_ticks": 1152,
        "meter": {"numerator": 3, "denominator": 4},
        "meter_valid": meter_valid,
        "partial_reason": None,
        "notes": [note(pitch)] if notes is None else notes,
        "rests": [],
        "chords": [{"onset_ticks": 0, "symbol": chord, "voice": "1"}],
    }


def score(measures, *, route=None, valid=True):
    return {
        "valid": valid,
        "errors": [] if valid else ["bad notation"],
        "warnings": [],
        "ticks_per_quarter": 384,
        "measures": measures,
        "route": route if route is not None else [m["number"] for m in measures],
        "route_status": "resolved" if valid else "unavailable",
        "counts": {
            "measures": len(measures),
            "noteheads": sum(len(m["notes"]) for m in measures),
            "rests": sum(len(m["rests"]) for m in measures),
            "chord_labels": sum(len(m["chords"]) for m in measures),
            "performed_measures": len(route) if route is not None else len(measures),
        },
    }


def test_equivalent_normalized_music_is_exact_even_with_different_voice_ids():
    reference = score([bar(1, 60), bar(2, 62)], route=[1, 2, 1])
    candidate = copy.deepcopy(reference)
    candidate["measures"][0]["notes"][0]["voice"] = "renamed_voice"
    candidate["measures"][1]["chords"][0]["voice"] = "renamed_voice"
    result = score_eval.compare_normalized(reference, candidate)
    assert result["metrics"]["edition_note_f1"] == 1
    assert result["metrics"]["edition_exact_bar_rate"] == 1
    assert result["metrics"]["chord_f1"] == 1
    assert result["metrics"]["form_exact_match"] is True
    assert result["metrics"]["source_note_f1"] is None
    assert result["metrics"]["editorial_acceptance"] is None
    assert result["coverage"]["source_scored_measures"] == 0


def test_pitch_duration_and_chord_errors_lower_agreement():
    reference = score([bar(1, 60, notes=[note(60), note(64, onset=384)])])
    candidate = score([bar(1, 61, chord="G", notes=[note(61), note(64, onset=384, duration=768)])])
    result = score_eval.compare_normalized(reference, candidate)
    assert result["metrics"]["edition_note_f1"] == 0
    assert result["metrics"]["chord_f1"] == 0
    assert result["metrics"]["edition_exact_bar_rate"] == 0
    assert result["mismatches"][0]["kind"] == "notation"


@pytest.mark.parametrize(
    "candidate_pitches,expected_alignment",
    [
        ([60, 64], [(1, 1), (2, None), (3, 2)]),
        ([60, 65, 62, 64], [(1, 1), (None, 2), (2, 3), (3, 4)]),
    ],
)
def test_inserted_or_missing_bar_does_not_shift_following_match(
    candidate_pitches, expected_alignment
):
    reference = score([bar(i, pitch) for i, pitch in enumerate([60, 62, 64], 1)])
    candidate = score([bar(i, pitch) for i, pitch in enumerate(candidate_pitches, 1)])
    result = score_eval.compare_normalized(reference, candidate)
    assert [(x["reference"], x["candidate"]) for x in result["alignment"]] == expected_alignment
    expected_matches = 2 if len(candidate_pitches) == 2 else 3
    assert result["event_counts"]["notes"]["matched"] == expected_matches
    assert result["metrics"]["edition_exact_bar_rate"] < 1


def test_duplicate_simultaneous_heads_are_not_collapsed():
    reference = score([bar(1, 60, notes=[note(60), note(60)])])
    candidate = score([bar(1, 60)])
    result = score_eval.compare_normalized(reference, candidate)
    assert result["event_counts"]["notes"] == {"matched": 1, "predicted": 1, "expected": 2}
    assert result["metrics"]["edition_note_f1"] == pytest.approx(2 / 3)
    assert result["metrics"]["edition_exact_bar_rate"] == 0


def test_wrong_performed_route_fails_form_with_identical_written_bars():
    reference = score([bar(1, 60), bar(2, 62)], route=[1, 2, 1])
    candidate = score([bar(1, 60), bar(2, 62)], route=[1, 2])
    result = score_eval.compare_normalized(reference, candidate)
    assert result["metrics"]["edition_exact_bar_rate"] == 1
    assert result["metrics"]["form_exact_match"] is False
    assert any(x["kind"] == "form" for x in result["mismatches"])


def test_invalid_output_has_no_accuracy_or_cost_metrics():
    reference = score([bar(1, 60)])
    candidate = score([], valid=False)
    result = score_eval.compare_normalized(reference, candidate)
    assert all(value is None for value in result["metrics"].values())
    assert result["coverage"]["source_scored_measures"] == 0
    assert result["metric_status"]["source_note_f1"] == "pending_source_labels"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def registry_fixture(tmp_path, monkeypatch):
    frozen = tmp_path / "frozen"
    labels = tmp_path / "labels"
    frozen.mkdir()
    labels.mkdir()
    (frozen / "manifest.json").write_text("{}")
    (labels / "manifest.json").write_text("{}")
    (frozen / "review.abc").write_text("reference")
    (frozen / "source.pdf").write_bytes(b"PDF input")
    candidate_path = tmp_path / "candidate.abc"
    candidate_path.write_text("candidate")
    decision_path = tmp_path / "decisions.json"
    decision_path.write_text("[]")
    slug = "synthetic_work"
    frozen_data = {
        "works": [
            {
                "slug": slug,
                "title": "Synthetic",
                "answer_abc_path": "review.abc",
                "source_pdf_path": "source.pdf",
            }
        ],
        "files": {"source.pdf": {"sha256": _sha(b"PDF input")}},
    }
    monkeypatch.setattr(score_eval, "verify_freeze", lambda _: frozen_data)
    monkeypatch.setattr(
        score_eval, "verify_labels", lambda *_: {"works": [{"slug": slug, "decisions": []}]}
    )
    monkeypatch.setattr(score_eval, "normalize_abc", lambda _: score([bar(1, 60)]))
    model = score_eval.MODELS[0]
    record = {
        "slug": slug,
        "model_id": model["id"],
        "model": model["model"],
        "reasoning_effort": model["reasoning_effort"],
        "status": "complete",
        "authorship": "model",
        "input_pdf_sha256": _sha(b"PDF input"),
        "abc_path": str(candidate_path),
        "abc_sha256": _sha(b"candidate"),
        "decision_log_path": str(decision_path),
        "decision_log_sha256": _sha(b"[]"),
    }

    from score2abc.score_eval_harness import AUTHOR_INSTRUCTIONS

    trace_path = tmp_path / "run.json"
    trace = {
        **record,
        "schema_version": 1,
        "instructions_sha256": _sha(AUTHOR_INSTRUCTIONS.encode()),
        "isolation": "input_only",
        "fresh_context": True,
        "network_access": False,
        "runner_name": "test-runner",
        "revision_passes": 0,
    }
    trace_path.write_text(json.dumps(trace))
    record["run_record_path"] = str(trace_path)
    record["run_record_sha256"] = _sha(trace_path.read_bytes())

    def run(records, *, evidence_kind="actual"):
        registry = tmp_path / "registry.json"
        registry.write_text(
            json.dumps({"schema_version": 1, "evidence_kind": evidence_kind, "runs": records})
        )
        return score_eval.build_report(frozen, labels, registry)

    return record, run, candidate_path


def test_registered_run_keeps_unmeasured_cost_and_source_accuracy_null(tmp_path, monkeypatch):
    record, run, _ = registry_fixture(tmp_path, monkeypatch)
    report = run([record])
    result = report["cases"][0]["results"]["astra"]
    assert result["status"] == "complete"
    assert result["metrics"]["cost_usd"] is None
    assert result["metrics"]["source_note_f1"] is None
    assert result["metrics"]["editorial_acceptance"] is None
    assert report["summary"]["source_accuracy_available"] is False
    assert report["cases"][0]["results"]["terra"]["status"] == "not_run"


@pytest.mark.parametrize("pinned_usage", [None, [{"input_tokens": 40, "output_tokens": 10}, None]])
def test_legacy_registry_only_usage_is_ignored_without_blocking_report(
    tmp_path, monkeypatch, pinned_usage
):
    record, run, _ = registry_fixture(tmp_path, monkeypatch)
    record["usage"] = {"input_tokens": 100, "output_tokens": 20}
    if pinned_usage is not None:
        path = tmp_path / "run.json"
        trace = json.loads(path.read_text())
        trace["usage"] = pinned_usage
        path.write_text(json.dumps(trace))
        record["run_record_sha256"] = _sha(path.read_bytes())
    result = run([record])["cases"][0]["results"]["astra"]
    assert result["status"] == "complete"
    assert result["usage"] == (pinned_usage if pinned_usage is not None else [])
    assert result["metrics"]["cost_usd"] is None


def test_unbacked_cost_is_rejected_and_pinned_provider_cost_is_preserved(tmp_path, monkeypatch):
    record, run, _ = registry_fixture(tmp_path, monkeypatch)
    record["cost_usd"] = 0.25
    with pytest.raises(ValueError, match="pinned measurement"):
        run([record])
    measurement = tmp_path / "measurement.json"
    measurement.write_text(
        json.dumps(
            {"cost_usd": {"value": 0.25, "basis": "provider_reported", "evidence": "receipt-123"}}
        )
    )
    record.update(
        measurement_path=str(measurement), measurement_sha256=_sha(measurement.read_bytes())
    )
    result = run([record])["cases"][0]["results"]["astra"]
    assert result["metrics"]["cost_usd"] == 0.25
    assert result["metric_status"]["cost_usd"] == "provider_reported"
    measurement.write_text("{}")
    with pytest.raises(ValueError, match="hash/size mismatch"):
        run([record])


@pytest.mark.parametrize(
    "change",
    [
        {"fresh_context": False},
        {"network_access": True},
        {"revision_passes": 3},
        {"instructions_sha256": "0" * 64},
    ],
)
def test_runner_provenance_must_satisfy_shared_contract(tmp_path, monkeypatch, change):
    record, run, _ = registry_fixture(tmp_path, monkeypatch)
    path = tmp_path / "run.json"
    trace = json.loads(path.read_text())
    trace.update(change)
    path.write_text(json.dumps(trace))
    record["run_record_sha256"] = _sha(path.read_bytes())
    with pytest.raises(ValueError, match="PDF-only paired contract"):
        run([record])


def test_missing_edition_bar_is_partial_even_when_abc_is_valid(tmp_path, monkeypatch):
    record, run, _ = registry_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        score_eval,
        "normalize_abc",
        lambda text: score([bar(1, 60), bar(2, 64)] if text == "reference" else [bar(1, 60)]),
    )
    result = run([record])["cases"][0]["results"]["astra"]
    assert result["status"] == "partial"
    assert result["metrics"]["bar_coverage"] == 0.5


def test_unconfirmed_pickup_needs_inspection_not_automatic_completion(monkeypatch):
    parsed = score([bar(1, 60, meter_valid=None)])
    parsed["measures"][0]["partial_reason"] = "pickup_candidate"
    monkeypatch.setattr(score_eval, "normalize_abc", lambda _: parsed)
    feedback = score_eval.validate_candidate("draft", [])
    assert feedback["issues"][0]["partial_reason"] == "pickup_candidate"
    assert "do not automatically pad" in feedback["issues"][0]["detail"]


def test_registered_invalid_abc_is_failed_without_agreement_metrics(tmp_path, monkeypatch):
    record, run, candidate_path = registry_fixture(tmp_path, monkeypatch)
    candidate_path.write_text("invalid")
    record["abc_sha256"] = _sha(b"invalid")
    trace_path = tmp_path / "run.json"
    trace = json.loads(trace_path.read_text())
    trace["abc_sha256"] = record["abc_sha256"]
    trace_path.write_text(json.dumps(trace))
    record["run_record_sha256"] = _sha(trace_path.read_bytes())
    monkeypatch.setattr(
        score_eval,
        "normalize_abc",
        lambda text: score([], valid=False) if text == "invalid" else score([bar(1, 60)]),
    )
    result = run([record])["cases"][0]["results"]["astra"]
    assert result["status"] == "failed"
    assert result["error"] == "bad notation"
    assert result["metrics"]["edition_note_f1"] is None
    assert result["metrics"]["source_note_f1"] is None


def test_synthetic_registry_cannot_publish_results_under_real_model_names(tmp_path, monkeypatch):
    record, run, _ = registry_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="require actual evidence"):
        run([record], evidence_kind="synthetic_demo")


@pytest.mark.parametrize(
    "change,match",
    [
        ({"model_id": "unknown"}, "unknown run identity"),
        ({"model": "gpt-6.1-terra"}, "model/settings differ"),
        ({"reasoning_effort": "low"}, "model/settings differ"),
        ({"status": "success"}, "explicit outcome"),
        ({"input_pdf_sha256": "0" * 64}, "input PDF differs"),
        ({"authorship": "synthetic_test"}, "model authorship"),
        ({"cost_usd": -1}, "Invalid measured usage"),
        ({"cost_usd": True}, "Invalid measured usage"),
    ],
)
def test_registry_rejects_identity_outcome_source_and_false_cost(
    tmp_path, monkeypatch, change, match
):
    record, run, _ = registry_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match=match):
        run([{**record, **change}])


def test_registry_rejects_changed_candidate_bytes_and_duplicate_run(tmp_path, monkeypatch):
    record, run, candidate_path = registry_fixture(tmp_path, monkeypatch)
    candidate_path.write_text("changed after registration")
    with pytest.raises(ValueError, match="hash/size mismatch"):
        run([record])
    with pytest.raises(ValueError, match="Duplicate or unknown run identity"):
        run([record, record])
