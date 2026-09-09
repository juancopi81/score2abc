from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from scripts.experiments import build_cross_score_error_breakdown as breakdown


def test_builds_portable_cross_score_breakdown_from_frozen_reports(
    tmp_path: Path,
) -> None:
    output = breakdown.build_cross_score_error_breakdown(
        output_dir=tmp_path / "result",
    )
    report = _json(output / "report.json")

    assert report["kind"] == breakdown.REPORT_KIND
    assert report["evidence"]["source_report_count"] == 4
    assert report["evidence"]["predictions_rerun"] is False
    assert report["evidence"]["musicxml_opened"] is False
    assert [score["score_id"] for score in report["scores"]] == [
        "carrizal",
        "la_chata",
        "gatoe_fique",
        "coqueteos",
    ]

    aggregate = report["aggregate"]
    assert aggregate["segmentation"] == {
        "score_count": 4,
        "automatic_crop_count": 26,
        "physical_measure_count": 28,
        "one_to_one_score_count": 2,
        "merged_crop_count": 2,
        "missing_boundary_count": 2,
        "confounded_crop_count": 2,
        "root_cause_policy": (
            "exclude merged or otherwise non-one-to-one crops from downstream target ranking"
        ),
    }
    clean = aggregate["clean_one_to_one_units"]
    assert clean["unit_count"] == 24
    assert clean["note_count"] == {
        "metric_semantics": (
            "count-capacity upper bound only; min(predicted, truth) does not imply "
            "note identity, pitch, or event matches"
        ),
        "predicted": 98,
        "truth": 100,
        "matched_capacity": 87,
        "surplus": 11,
        "deficit": 13,
        "precision": 0.887755,
        "recall": 0.87,
        "f1": 0.878788,
    }
    assert clean["pitch"] == {
        "status": "scored",
        "exact_matches": 36,
        "mismatches_within_count_capacity": 51,
        "matched_capacity": 87,
        "conditional_accuracy": 0.413793,
    }
    assert clean["full_event_subset"]["onset"]["conditional_accuracy"] == 0.483871
    assert clean["full_event_subset"]["duration"]["conditional_accuracy"] == 0.677419
    assert clean["full_event_subset"]["rests"]["f1"] == 0.222222

    candidate_stage = next(
        stage for stage in report["stage_breakdown"] if stage["stage"] == "candidate_coverage"
    )
    assert candidate_stage["status"] == "not_identifiable_from_frozen_reports"
    assert candidate_stage["ranking_eligible"] is False
    count_stage = next(
        stage for stage in report["stage_breakdown"] if stage["stage"] == "note_count_output"
    )
    assert count_stage["status"] == "scored_but_causally_ambiguous"
    assert count_stage["ranking_eligible"] is False
    segmentation_stage = next(
        stage for stage in report["stage_breakdown"] if stage["stage"] == "segmentation"
    )
    assert segmentation_stage["ranking_eligible"] is False
    assert report["next_engineering_target"]["selected_target"] == ("pitch_mapping_and_key_context")
    assert report["next_engineering_target"]["basis"] == {
        "support_score_count": 4,
        "opportunity_count": 87,
        "observed_error_count": 51,
        "error_rate": 0.586207,
    }
    assert report["next_engineering_target"]["next_experiment"]["human_input_required_now"] is False
    assert (output / "summary.md").is_file()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        breakdown.build_cross_score_error_breakdown(output_dir=output)


def test_preserves_unsupported_rhythm_instead_of_treating_it_as_zero(
    tmp_path: Path,
) -> None:
    output = breakdown.build_cross_score_error_breakdown(
        output_dir=tmp_path / "result",
    )
    report = _json(output / "report.json")
    la_chata = next(score for score in report["scores"] if score["score_id"] == "la_chata")

    assert la_chata["rhythm"] == {
        "onset": {"status": "not_scored_missing_frozen_context"},
        "duration": {"status": "not_scored_missing_frozen_context"},
        "rests": {"status": "not_scored_missing_frozen_context"},
        "meter": {"status": "not_scored_missing_frozen_context"},
        "exact_full_measures": {"status": "not_scored_missing_frozen_context"},
    }


def test_rejects_hash_drift_before_normalization(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "fixtures"
    shutil.copytree(breakdown.DEFAULT_INPUT_MANIFEST.parent, fixture_dir)
    report_path = fixture_dir / "la_chata_system_007_evaluation.json"
    report_path.write_text(report_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        breakdown.build_cross_score_error_breakdown(
            input_manifest=fixture_dir / "manifest.json",
            output_dir=tmp_path / "result",
        )


def test_joins_coqueteos_metrics_and_mapping_by_crop_index(tmp_path: Path) -> None:
    fixture_dir = _fixture_copy(tmp_path)
    report_path = fixture_dir / "coqueteos_system_002_evaluation.json"
    report = _json(report_path)
    report["meter"]["crops"] = list(reversed(report["meter"]["crops"]))
    _write_json(report_path, report)
    _refresh_manifest_hash(fixture_dir, report_path.name)

    output = breakdown.build_cross_score_error_breakdown(
        input_manifest=fixture_dir / "manifest.json",
        output_dir=tmp_path / "result",
    )
    generated = _json(output / "report.json")

    assert (
        generated["aggregate"]["clean_one_to_one_units"]["pitch"]["conditional_accuracy"]
        == 0.413793
    )


def test_rejects_mismatched_parallel_crop_indices(tmp_path: Path) -> None:
    fixture_dir = _fixture_copy(tmp_path)
    report_path = fixture_dir / "coqueteos_system_002_evaluation.json"
    report = _json(report_path)
    report["meter"]["crops"][-1]["automatic_crop_index"] = 5
    _write_json(report_path, report)
    _refresh_manifest_hash(fixture_dir, report_path.name)

    with pytest.raises(ValueError, match="duplicate index"):
        breakdown.build_cross_score_error_breakdown(
            input_manifest=fixture_dir / "manifest.json",
            output_dir=tmp_path / "result",
        )


def test_rejects_scope_or_target_identity_drift(tmp_path: Path) -> None:
    fixture_dir = _fixture_copy(tmp_path)
    manifest_path = fixture_dir / "manifest.json"
    manifest = _json(manifest_path)
    manifest["reports"][1]["scope"] = "full_event"
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="Scope mismatch"):
        breakdown.build_cross_score_error_breakdown(
            input_manifest=manifest_path,
            output_dir=tmp_path / "scope-result",
        )

    manifest["reports"][1]["scope"] = "pitch_only"
    manifest["reports"][1]["target_slug"] = "wrong-slug"
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="target slug"):
        breakdown.build_cross_score_error_breakdown(
            input_manifest=manifest_path,
            output_dir=tmp_path / "identity-result",
        )


def test_rejects_incomplete_physical_measure_coverage(tmp_path: Path) -> None:
    fixture_dir = _fixture_copy(tmp_path)
    report_path = fixture_dir / "la_chata_system_007_evaluation.json"
    report = _json(report_path)
    report["metrics"]["crops"][2]["physical_measure_numbers"] = [8]
    _write_json(report_path, report)
    _refresh_manifest_hash(fixture_dir, report_path.name)

    with pytest.raises(ValueError, match="physical measure coverage mismatch"):
        breakdown.build_cross_score_error_breakdown(
            input_manifest=fixture_dir / "manifest.json",
            output_dir=tmp_path / "result",
        )


def test_rejects_coqueteos_metric_row_identity_drift(tmp_path: Path) -> None:
    fixture_dir = _fixture_copy(tmp_path)
    report_path = fixture_dir / "coqueteos_system_002_evaluation.json"
    report = _json(report_path)
    report["metrics"]["results"][2]["identity"]["slug"] = "wrong-slug"
    _write_json(report_path, report)
    _refresh_manifest_hash(fixture_dir, report_path.name)

    with pytest.raises(ValueError, match="metric row slug"):
        breakdown.build_cross_score_error_breakdown(
            input_manifest=fixture_dir / "manifest.json",
            output_dir=tmp_path / "result",
        )


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fixture_copy(tmp_path: Path) -> Path:
    fixture_dir = tmp_path / "fixtures"
    shutil.copytree(breakdown.DEFAULT_INPUT_MANIFEST.parent, fixture_dir)
    return fixture_dir


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _refresh_manifest_hash(fixture_dir: Path, report_name: str) -> None:
    manifest_path = fixture_dir / "manifest.json"
    manifest = _json(manifest_path)
    report_path = fixture_dir / report_name
    digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    for report in manifest["reports"]:
        if report["path"] == report_name:
            report["sha256"] = digest
            break
    else:
        raise AssertionError(f"Missing report in manifest: {report_name}")
    _write_json(manifest_path, manifest)
