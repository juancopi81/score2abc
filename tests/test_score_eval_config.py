"""Versioned cohorts separate generation from grading and reject mixed settings."""

import copy
import json

import pytest
from test_score_eval import bar, registry_fixture, score
from test_score_eval_harness import frozen_fixture

from score2abc import score_eval
from score2abc import score_eval_config as config
from score2abc import score_eval_harness as harness
from score2abc.score_eval_source import NOTE_FIELDS, note_events_sha256


@pytest.fixture
def implementation(tmp_path, monkeypatch):
    root = tmp_path / "implementation"
    root.mkdir()
    for name in (
        "score_eval_config.py",
        "score_eval_harness.py",
        "score_eval_runner.py",
        "score_eval.py",
        "score_eval_abc.py",
        "score_eval_parse.js",
        "review_playback.js",
        "eval_labels.py",
        "score_eval_source.py",
    ):
        (root / name).write_text(f"synthetic implementation: {name}")
    monkeypatch.setattr(config, "_PACKAGE", root)
    monkeypatch.setattr(config, "renderer_identity", lambda: {"fixture": "renderer-v1"})
    monkeypatch.setattr(
        config, "runner_identity", lambda: {"status": "prepared", "fixture": "runner-v1"}
    )
    monkeypatch.setattr(config, "code_provenance", lambda: {"commit": "test", "dirty": True})
    return root


def test_arbitrary_systems_and_deterministic_systems_have_explicit_settings():
    value = config.load_config()
    value["systems"].append(
        {
            "id": "pipeline",
            "label": "Deterministic pipeline",
            "provider": "local",
            "adapter": "deterministic",
            "model": None,
            "reasoning_effort": None,
            "settings": {},
        }
    )
    value["systems"].append(
        {
            "id": "another-model",
            "label": "Adapter not installed",
            "provider": "another",
            "adapter": "not_installed",
            "model": "exact-model-id",
            "reasoning_effort": "high",
            "settings": {"temperature": 0},
        }
    )
    assert len(config.load_config(value)["systems"]) == 4
    assert config.load_config()["systems"] != value["systems"]


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda c: c["systems"].append(copy.deepcopy(c["systems"][0])), "Duplicate system"),
        (lambda c: c["harness"].update(max_revision_passes=3), "zero to two"),
        (lambda c: c["harness"].update(max_revision_passes=True), "zero to two"),
        (lambda c: c["systems"][0].update(model=None), "system model"),
        (lambda c: c["systems"][0].update(settings=[]), "settings must"),
        (lambda c: c.update(unknown="ignored?"), "configuration schema"),
    ],
)
def test_bad_configuration_fails_before_scheduling(mutation, match):
    value = config.load_config()
    mutation(value)
    with pytest.raises(ValueError, match=match):
        config.load_config(value)


def test_source_packet_schedules_all_configured_systems_without_answers(
    tmp_path, monkeypatch, implementation
):
    frozen, _ = frozen_fixture(tmp_path, monkeypatch)
    settings = config.load_config()
    extra = copy.deepcopy(settings["systems"][1])
    extra.update(id="terra-high", reasoning_effort="high")
    settings["systems"].append(extra)
    packet = harness.prepare_inputs(frozen, tmp_path / "packet", settings)
    assert packet["schema_version"] == 2
    assert len(packet["tasks"]) == 18
    assert len({item["task_id"] for item in packet["tasks"]}) == 18
    assert all(item["task_id"] == config.task_id(item) for item in packet["tasks"])
    assert not any("answers" in str(path) for path in (tmp_path / "packet").rglob("*"))
    assert "review.abc" not in json.dumps(packet)
    assert {item["model_id"] for item in packet["tasks"]} == {"astra", "terra", "terra-high"}
    assert all(item["status"] == "not_run" for item in packet["tasks"])
    assert config.verify_prepared_task(packet["tasks"][0], packet) == packet["tasks"][0]
    repeated = harness.prepare_inputs(frozen, tmp_path / "trial-2", settings, trial=2)
    assert repeated["generation"]["sha256"] == packet["generation"]["sha256"]
    assert {item["trial"] for item in repeated["tasks"]} == {2}
    assert not (
        {item["task_id"] for item in repeated["tasks"]}
        & {item["task_id"] for item in packet["tasks"]}
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"model": "gpt-6.1-terra"},
        {"reasoning_effort": "low"},
        {"instructions_sha256": "0" * 64},
        {"max_revision_passes": 99},
        {"input_pdf_sha256": "0" * 64},
        {"trial": 7},
        {"settings": {"temperature": 1}},
    ],
)
def test_execution_fields_must_match_packet_even_when_task_id_is_recomputed(
    tmp_path, monkeypatch, implementation, changes
):
    frozen, _ = frozen_fixture(tmp_path, monkeypatch)
    packet = harness.prepare_inputs(frozen, tmp_path / "packet", config.load_config())
    task = copy.deepcopy(packet["tasks"][0])
    task.update(changes)
    task["task_id"] = config.task_id(task)
    with pytest.raises(ValueError):
        config.verify_prepared_task(task, packet)


def test_versions_track_actual_code_and_keep_commit_metadata_out_of_identity(
    tmp_path, monkeypatch, implementation
):
    _, frozen = frozen_fixture(tmp_path, monkeypatch)
    settings = config.load_config()
    first = config.generation_identity(frozen, settings, "instructions")
    monkeypatch.setattr(config, "code_provenance", lambda: {"commit": "later", "dirty": False})
    metadata_only = config.generation_identity(frozen, settings, "instructions")
    assert first["sha256"] == metadata_only["sha256"]
    assert first["harness"]["provenance"] != metadata_only["harness"]["provenance"]
    (implementation / "score_eval_runner.py").write_text("changed image rendering")
    changed = config.generation_identity(frozen, settings, "instructions")
    assert first["dataset"] == changed["dataset"]
    assert first["harness"]["sha256"] != changed["harness"]["sha256"]
    assert first["sha256"] != changed["sha256"]
    changed["harness"]["manifest"]["instructions_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity hash"):
        config.verify_generation(changed, settings, frozen)


def v2_fixture(tmp_path, monkeypatch, implementation):
    record, _, _ = registry_fixture(tmp_path, monkeypatch)
    frozen, labels = tmp_path / "frozen", tmp_path / "labels"
    settings = config.load_config()
    monkeypatch.setattr(
        score_eval,
        "verify_labels",
        lambda *_: {
            "label_set": settings["labels_version"],
            "works": [{"slug": "synthetic_work", "decisions": []}],
        },
    )
    generation = config.generation_identity(
        score_eval.verify_freeze(frozen), settings, harness.AUTHOR_INSTRUCTIONS
    )
    model = config.configured_systems(settings, generation)[0]
    record.update(
        provider=model["provider"],
        adapter=model["adapter"],
        settings=model["settings"],
        system_sha256=model["system_sha256"],
        generation_sha256=generation["sha256"],
        trial=1,
        run_id="fixture-run-1",
        resolved_model=None,
    )
    record["task_id"] = config.task_id(record)
    trace_path = tmp_path / "run.json"
    trace = json.loads(trace_path.read_text())
    trace.update(record)
    trace["schema_version"] = 2
    trace["runtime_sha256"] = config.digest(generation["harness"]["manifest"]["runner_runtime"])
    trace_path.write_text(json.dumps(trace))
    record["run_record_sha256"] = score_eval.sha256(trace_path.read_bytes())
    registry = tmp_path / "registry.json"
    raw = {
        "schema_version": 2,
        "evidence_kind": "actual",
        "config": settings,
        "generation": generation,
        "runs": [record],
    }

    def run(configuration=None):
        registry.write_text(json.dumps(raw))
        return score_eval.build_report(frozen, labels, registry, configuration)

    return raw, run


def test_versioned_report_supports_archived_regrading_without_reinference(
    tmp_path, monkeypatch, implementation
):
    raw, run = v2_fixture(tmp_path, monkeypatch, implementation)
    first = run()
    assert first["schema_version"] == 2
    assert set(first["versions"]) == {"dataset", "labels", "grader", "harness", "config"}
    assert first["cases"][0]["results"]["astra"]["run"]["resolved_model"] is None
    (implementation / "score_eval_source.py").write_text("fixed source denominator")
    updated = copy.deepcopy(raw["config"])
    updated.update(id="regraded", grader_version="complete-score-v3")
    second = run(updated)
    assert first["generation_sha256"] == second["generation_sha256"]
    assert first["versions"]["harness"] == second["versions"]["harness"]
    assert first["versions"]["grader"]["sha256"] != second["versions"]["grader"]["sha256"]
    assert second["cases"][0]["results"]["astra"]["metrics"]["edition_note_f1"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("system_sha256", "0" * 64),
        ("generation_sha256", "0" * 64),
        ("task_id", "0" * 64),
        ("settings", {"temperature": 0.5}),
        ("provider", "another-provider"),
        ("trial", 0),
    ],
)
def test_versioned_registry_rejects_mixed_or_forged_identity(
    tmp_path, monkeypatch, implementation, field, value
):
    raw, run = v2_fixture(tmp_path, monkeypatch, implementation)
    raw["runs"][0][field] = value
    with pytest.raises(ValueError):
        run()


def test_changed_model_visible_harness_cannot_regrade_as_same_generation(
    tmp_path, monkeypatch, implementation
):
    raw, run = v2_fixture(tmp_path, monkeypatch, implementation)
    changed = copy.deepcopy(raw["config"])
    changed["harness"]["max_revision_passes"] = 1
    with pytest.raises(ValueError, match="configuration differs"):
        run(changed)


def test_unknown_label_version_cannot_be_reported_as_requested_version(
    tmp_path, monkeypatch, implementation
):
    _, run = v2_fixture(tmp_path, monkeypatch, implementation)
    monkeypatch.setattr(
        score_eval,
        "verify_labels",
        lambda *_: {
            "label_set": "old_labels_v1",
            "works": [{"slug": "synthetic_work", "decisions": []}],
        },
    )
    with pytest.raises(ValueError, match="label version differs"):
        run()


def test_source_grade_and_coverage_are_merged_without_claiming_full_audit(
    tmp_path, monkeypatch, implementation
):
    raw, run = v2_fixture(tmp_path, monkeypatch, implementation)
    audited_bar = bar(1, 60)
    monkeypatch.setattr(
        score_eval,
        "verify_labels",
        lambda *_: {
            "label_set": raw["config"]["labels_version"],
            "works": [
                {
                    "slug": "synthetic_work",
                    "decisions": [],
                    "source_audit": {
                        "total_source_measures": 2,
                        "ticks_per_quarter": 384,
                        "measures": [
                            {
                                "source_bar": 1,
                                "edition_bar": 1,
                                "source_location": {"page": 1, "system": 1, "measure": 1},
                                "dimensions": ["notes"],
                                "note_fields": list(NOTE_FIELDS),
                                "expected_noteheads": 1,
                                "note_events_sha256": note_events_sha256(audited_bar),
                            }
                        ],
                    },
                }
            ],
        },
    )
    monkeypatch.setattr(
        score_eval,
        "normalize_abc",
        lambda text: score(
            [
                bar(1, 60 if text == "reference" else 61),
            ]
        ),
    )
    report = run()
    result = report["cases"][0]["results"]["astra"]
    assert report["summary"]["source_accuracy_available"] is True
    assert result["metrics"]["source_note_f1"] == 0
    assert result["coverage"]["source_scored_measures"] == 1
    assert result["coverage"]["source_total_measures"] == 2
    assert result["metric_status"]["source_note_f1"] == "model_audited_source_subset"
    raw["runs"] = []
    planned = run()["cases"][0]["results"]["astra"]
    assert planned["metrics"]["source_note_f1"] is None
    assert planned["metric_status"]["source_note_f1"] == "not_run"
    assert planned["coverage"]["source_eligible_measures"] == 1
    assert planned["coverage"]["source_scored_measures"] == 0


def test_runtime_mismatch_cannot_hide_behind_compatible_model_settings(
    tmp_path, monkeypatch, implementation
):
    raw, run = v2_fixture(tmp_path, monkeypatch, implementation)
    path = tmp_path / "run.json"
    trace = json.loads(path.read_text())
    trace["runtime_sha256"] = "0" * 64
    path.write_text(json.dumps(trace))
    raw["runs"][0]["run_record_sha256"] = score_eval.sha256(path.read_bytes())
    with pytest.raises(ValueError, match="runtime differs"):
        run()


def test_usage_comes_only_from_pinned_trace_including_unmeasured_attempts(
    tmp_path, monkeypatch, implementation
):
    raw, run = v2_fixture(tmp_path, monkeypatch, implementation)
    path = tmp_path / "run.json"
    trace = json.loads(path.read_text())
    trace["usage"] = [{"input_tokens": 40, "output_tokens": 10}, None]
    path.write_text(json.dumps(trace))
    raw["runs"][0]["run_record_sha256"] = score_eval.sha256(path.read_bytes())
    result = run()["cases"][0]["results"]["astra"]
    assert result["usage"] == trace["usage"]
    assert result["metrics"]["cost_usd"] is None
    raw["runs"][0]["usage"] = [{"input_tokens": 0, "output_tokens": 0}]
    with pytest.raises(ValueError, match="usage differs"):
        run()


def test_cohort_cannot_mix_piecewise_or_modelwise_selected_trials(
    tmp_path, monkeypatch, implementation
):
    raw, run = v2_fixture(tmp_path, monkeypatch, implementation)
    model = config.configured_systems(raw["config"], raw["generation"])[1]
    second = copy.deepcopy(raw["runs"][0])
    second.update(
        model_id=model["id"],
        model=model["model"],
        reasoning_effort=model["reasoning_effort"],
        system_sha256=model["system_sha256"],
        run_id="fixture-run-2",
        trial=2,
    )
    second["task_id"] = config.task_id(second)
    raw["runs"].append(second)
    with pytest.raises(ValueError, match="Mixed trial numbers"):
        run()


@pytest.mark.parametrize(
    "candidate_pitches,expected_status",
    [
        ([60, 62, 64], "complete"),
        ([60, 64], "partial"),
    ],
)
def test_completeness_excludes_only_known_edition_insertions_not_source_bars(
    tmp_path, monkeypatch, candidate_pitches, expected_status
):
    record, run, _ = registry_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        score_eval,
        "verify_labels",
        lambda *_: {
            "works": [
                {
                    "slug": "synthetic_work",
                    "decisions": [],
                    "source_to_edition_mapping": {
                        "source_to_edition_bar": {"1": 1, "2": 3, "3": 4}
                    },
                }
            ]
        },
    )
    monkeypatch.setattr(
        score_eval,
        "normalize_abc",
        lambda text: score(
            [
                bar(i, pitch)
                for i, pitch in enumerate(
                    [60, 70, 62, 64] if text == "reference" else candidate_pitches, 1
                )
            ]
        ),
    )
    result = run([record])["cases"][0]["results"]["astra"]
    assert result["status"] == expected_status
    assert result["coverage"]["source_required_edition_measures"] == 3
    assert result["coverage"]["edition_only_measures_excluded_from_completeness"] == 1
    assert result["metrics"]["edition_exact_bar_rate"] < 1
    assert result["coverage"]["source_scored_measures"] == 0
