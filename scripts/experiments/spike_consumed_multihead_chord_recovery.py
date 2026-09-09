"""Select a bounded multi-head chord recovery rule on consumed evidence.

This spike extends the previous one-companion dyad rule without creating new
onsets. It evaluates a small preregistered parameter grid against consumed
Alcira, La Chata, corrected No lo Creas, and Aviador/Carrizal review evidence.
No result from this script is eligible for runtime use without a later,
score-disjoint frozen gate.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import evaluate_independent_dyad_recovery_gate as evaluator  # noqa: E402
from scripts.experiments import run_independent_dyad_recovery_gate as runner  # noqa: E402
from scripts.experiments import spike_consumed_edge_safe_dyad_recovery as consumed  # noqa: E402
from scripts.experiments import spike_consumed_polyphonic_pitch_repair as recovery  # noqa: E402

SCHEMA_VERSION = 1
OUTPUT_VERSION = "consumed_multihead_chord_recovery_v1"
NO_LO_CREAS_SLUG = "jaime-llanos_73_no-lo-creas_pasillo_a-vasquez-pedrero"
NO_LO_CREAS_SYSTEM_INDEX = 8

MINIMUM_Y_GAPS = (0.75, 1.0)
MINIMUM_STEM_SCORES = (0.45, 0.55)
MAXIMUM_RECOVERED_HEADS = (2, 3)
FIXED_PARAMETERS = {
    "maximum_y_gap_staff_spaces": 3.0,
    "minimum_score_ratio": 0.5,
    "minimum_group_x_staff_spaces": 1.0,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", type=Path, default=Path("out"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        report = run_spike(out_dir=args.out_dir, output_dir=args.output_dir)
    except (FileExistsError, FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(report["artifacts"]["report_json"])
    print(report["artifacts"]["report_markdown"])
    return 0


def _config_grid() -> list[dict[str, Any]]:
    configs = []
    for minimum_y_gap in MINIMUM_Y_GAPS:
        for minimum_stem_score in MINIMUM_STEM_SCORES:
            for maximum_heads in MAXIMUM_RECOVERED_HEADS:
                parameters = {
                    "minimum_y_gap_staff_spaces": minimum_y_gap,
                    "minimum_stem_score": minimum_stem_score,
                    "maximum_recovered_heads_per_group": maximum_heads,
                    **FIXED_PARAMETERS,
                }
                config_id = (
                    "multihead"
                    f"__y_{minimum_y_gap:g}_3"
                    "__ratio_0.5"
                    f"__stem_{minimum_stem_score:g}"
                    "__leading_x_1"
                    f"__cap_{maximum_heads}"
                )
                configs.append({"config_id": config_id, "parameters": parameters})
    return configs


def _recovery_callback(parameters: Mapping[str, Any]) -> consumed.RecoveryRow:
    fixed = dict(parameters)

    def recover_row(
        row: Mapping[str, Any], selector: Mapping[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
        baseline = recovery.select_candidates(row, selector)
        stem_features, _ = recovery.candidate_local_stem_features(row)
        recovered = recovery.recover_edge_safe_stem_aware_multihead_candidates(
            row,
            selector,
            baseline,
            stem_features=stem_features,
            **fixed,
        )
        return baseline, recovered, stem_features

    return recover_row


def _no_lo_creas_root(out_dir: Path) -> Path:
    return (
        out_dir
        / "local_restricted"
        / NO_LO_CREAS_SLUG
        / "vlm_melody_independent_dyad_recovery_gate/v1"
        / f"system_{NO_LO_CREAS_SYSTEM_INDEX:03d}"
    )


def _evaluate_no_lo_creas(
    out_dir: Path,
    *,
    recover_row: consumed.RecoveryRow,
    config_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = _no_lo_creas_root(out_dir)
    paths = {
        "inference": root / "baseline_inference_v1/inference.jsonl",
        "model": root / "frozen/artifacts/model_and_training/002_model.json",
        "truth": root / "evaluation_v2_mapping_erratum/truth.jsonl",
        "mapping": root / "evaluation_v2_mapping_erratum/mapping.json",
        "musicxml": root / "evaluation_v2_mapping_erratum/source.musicxml",
        "erratum_report": root / "evaluation_v2_mapping_erratum/report.json",
    }
    rows, inference_pin = consumed._read_jsonl(paths["inference"])
    model, model_pin = consumed._read_json(paths["model"])
    truth_rows, truth_pin = consumed._read_jsonl(paths["truth"])
    mapping, mapping_pin = consumed._read_json(paths["mapping"])
    erratum_report, report_pin = consumed._read_json(paths["erratum_report"])
    musicxml_pin = consumed._pin(paths["musicxml"])
    selector = recovery.selector_config_from_model(model)

    paired_rows_by_crop: dict[int, dict[str, Any]] = {}
    records = []
    for row in rows:
        baseline, recovered, _stem_features = recover_row(row, selector)
        canonical = row.get("canonical_prediction")
        if not isinstance(canonical, Mapping):
            raise ValueError("No lo Creas inference row has no canonical prediction")
        runner._verify_generic_baseline(row, baseline, canonical)
        group_by_id = runner._baseline_group_indices(row, selector=selector, baseline=baseline)
        generic_note_by_id = {
            str(note["candidate_id"]): note for note in canonical.get("notes") or []
        }
        baseline_notes = [
            runner._normalized_note(
                row,
                candidate,
                onset_group_index=group_by_id[str(candidate["candidate_id"])],
                recovered=False,
                generic_note=generic_note_by_id[str(candidate["candidate_id"])],
            )
            for candidate in baseline
        ]
        baseline_notes.sort(key=runner._note_sort_key)
        recovered_notes = [
            runner._normalized_note(
                row,
                candidate,
                onset_group_index=int(candidate["recovery_group_index"]),
                recovered=True,
            )
            for candidate in recovered
        ]
        recovered_notes.sort(key=runner._note_sort_key)
        combined_notes = [*copy.deepcopy(baseline_notes), *recovered_notes]
        combined_notes.sort(key=runner._note_sort_key)
        crop = int(row["identity"]["automatic_measure_index"])
        paired_rows_by_crop[crop] = {
            "identity": dict(row["identity"]),
            "lanes": {
                evaluator.LANE_BASELINE: {"notes": baseline_notes},
                config_id: {"notes": combined_notes},
            },
        }
        records.append(
            {
                "automatic_crop_index": crop,
                "baseline_candidate_ids": [str(item["candidate_id"]) for item in baseline],
                "recovered_candidate_ids": [str(item["candidate_id"]) for item in recovered],
                "recovered_by_group": _recovered_by_group(recovered),
            }
        )

    visible_truth = evaluator.heldout.load_visible_musicxml_truth(paths["musicxml"])
    structure_support = evaluator._mapping_structure_support(mapping, visible_truth)
    baseline_score = evaluator._score_lane(
        truth_rows,
        paired_rows_by_crop,
        lane=evaluator.LANE_BASELINE,
        structure_support=structure_support,
    )
    recovered_score = evaluator._score_lane(
        truth_rows,
        paired_rows_by_crop,
        lane=config_id,
        structure_support=structure_support,
    )
    expected_baseline = erratum_report["lanes"][evaluator.LANE_BASELINE]["summary"]
    if baseline_score["summary"] != expected_baseline:
        raise ValueError("No lo Creas baseline no longer reproduces the mapping erratum")
    previous_dyad = erratum_report["lanes"][evaluator.LANE_RECOVERED]["summary"]
    return (
        {
            "target": {
                "slug": NO_LO_CREAS_SLUG,
                "system_index": NO_LO_CREAS_SYSTEM_INDEX,
            },
            "baseline": baseline_score["summary"],
            "previous_dyad": previous_dyad,
            "multihead_recovery": recovered_score["summary"],
            "delta_vs_previous_dyad": _summary_delta(recovered_score["summary"], previous_dyad),
            "recovery_records": records,
        },
        {
            "inference": inference_pin,
            "model": model_pin,
            "truth": truth_pin,
            "mapping": mapping_pin,
            "source_musicxml": musicxml_pin,
            "mapping_erratum_report": report_pin,
        },
    )


def _recovered_by_group(recovered: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[int, list[str]] = {}
    for candidate in recovered:
        groups.setdefault(int(candidate["recovery_group_index"]), []).append(
            str(candidate["candidate_id"])
        )
    return [
        {"onset_group_index": group, "candidate_ids": sorted(candidate_ids)}
        for group, candidate_ids in sorted(groups.items())
    ]


def _summary_delta(
    recovered: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, float | int]:
    return {
        "predicted_note_count": int(recovered["predicted_note_count"])
        - int(baseline["predicted_note_count"]),
        "note_count_f1": round(
            float(recovered["note_count_f1"]) - float(baseline["note_count_f1"]), 6
        ),
        "exact_diatonic_staff_position_matches": int(
            recovered["exact_diatonic_staff_position_matches"]
        )
        - int(baseline["exact_diatonic_staff_position_matches"]),
        "ordered_diatonic_alignment_accuracy": round(
            float(recovered["ordered_diatonic_alignment_accuracy"])
            - float(baseline["ordered_diatonic_alignment_accuracy"]),
            6,
        ),
        "exact_chord_size_matches": int(recovered["exact_chord_size_matches"])
        - int(baseline["exact_chord_size_matches"]),
        "chord_size_alignment_accuracy": round(
            float(recovered["chord_size_alignment_accuracy"])
            - float(baseline["chord_size_alignment_accuracy"]),
            6,
        ),
        "exact_structure_crops": int(recovered["exact_structure_crops"])
        - int(baseline["exact_structure_crops"]),
    }


def _evaluate_config(
    out_dir: Path,
    *,
    output_stage: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    config_id = str(config["config_id"])
    parameters = dict(config["parameters"])
    recover_row = _recovery_callback(parameters)
    alcira, alcira_pins = consumed._evaluate_alcira(
        out_dir=out_dir,
        output_dir=output_stage / config_id,
        recover_row=recover_row,
        config_id=config_id,
        parameters=parameters,
    )
    la_chata, la_chata_pins = consumed._evaluate_la_chata(
        out_dir,
        recover_row=recover_row,
        config_id=config_id,
        parameters=parameters,
    )
    regression, regression_pins = consumed._evaluate_candidate_regression(
        out_dir, recover_row=recover_row
    )
    no_lo_creas, no_lo_pins = _evaluate_no_lo_creas(
        out_dir, recover_row=recover_row, config_id=config_id
    )
    evaluations = {
        "alcira": alcira,
        "la_chata": la_chata,
        "no_lo_creas": no_lo_creas,
        "aviador_carrizal_candidate_regression": regression,
    }
    return (
        {
            "config_id": config_id,
            "parameters": parameters,
            "evaluations": evaluations,
            "gate": _gate(evaluations),
        },
        {
            "alcira": alcira_pins,
            "la_chata": la_chata_pins,
            "no_lo_creas": no_lo_pins,
            "aviador_carrizal": regression_pins,
        },
    )


def _gate(evaluations: Mapping[str, Any]) -> dict[str, Any]:
    alcira = evaluations["alcira"]
    la_chata = evaluations["la_chata"]
    no_lo = evaluations["no_lo_creas"]
    regression = evaluations["aviador_carrizal_candidate_regression"]
    alcira_preserved = int(alcira["edge_safe_recovery"]["exact_pitch_matches"]) >= int(
        alcira["baseline"]["exact_pitch_matches"]
    ) and float(alcira["edge_safe_recovery"]["note_count_f1"]) >= float(
        alcira["baseline"]["note_count_f1"]
    )
    la_baseline = la_chata["baseline"]
    la_recovered = la_chata["edge_safe_recovery"]
    la_chata_preserved = int(la_recovered["pitch_group_metrics"]["exact_group_count"]) >= int(
        la_baseline["pitch_group_metrics"]["exact_group_count"]
    ) and int(la_recovered["pitch_group_metrics"]["group_edit_distance"]) <= int(
        la_baseline["pitch_group_metrics"]["group_edit_distance"]
    )
    regression_preserved = (
        float(regression["edge_safe_recovery"]["f1"]) >= float(regression["baseline"]["f1"])
        and int(regression["edge_safe_recovery"]["recovered_false_positive_count"]) == 0
    )
    no_lo_delta = no_lo["delta_vs_previous_dyad"]
    no_lo_improved = float(no_lo_delta["note_count_f1"]) >= 0 and (
        int(no_lo_delta["exact_diatonic_staff_position_matches"]) > 0
        or int(no_lo_delta["exact_chord_size_matches"]) > 0
        or int(no_lo_delta["exact_structure_crops"]) > 0
    )
    passed = alcira_preserved and la_chata_preserved and regression_preserved and no_lo_improved
    return {
        "passed": passed,
        "alcira_preserved": alcira_preserved,
        "la_chata_preserved": la_chata_preserved,
        "aviador_carrizal_regression_preserved": regression_preserved,
        "no_lo_creas_improved_over_previous_dyad": no_lo_improved,
    }


def _selection_key(result: Mapping[str, Any]) -> tuple[Any, ...]:
    no_lo = result["evaluations"]["no_lo_creas"]
    summary = no_lo["multihead_recovery"]
    delta = no_lo["delta_vs_previous_dyad"]
    parameters = result["parameters"]
    recovered_count = sum(
        len(record["recovered_candidate_ids"]) for record in no_lo["recovery_records"]
    )
    return (
        bool(result["gate"]["passed"]),
        int(delta["exact_chord_size_matches"]),
        int(delta["exact_diatonic_staff_position_matches"]),
        int(delta["exact_structure_crops"]),
        float(summary["chord_size_alignment_accuracy"]),
        float(summary["ordered_diatonic_alignment_accuracy"]),
        -recovered_count,
        -int(parameters["maximum_recovered_heads_per_group"]),
        float(parameters["minimum_stem_score"]),
        float(parameters["minimum_y_gap_staff_spaces"]),
    )


def _select_best(results: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    passed = [result for result in results if result["gate"]["passed"]]
    return max(passed, key=_selection_key) if passed else None


def run_spike(*, out_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    out_dir = out_dir.expanduser().resolve()
    output_dir = (
        (output_dir or out_dir / "vlm_melody_consumed_training/multihead_chord_recovery_v1")
        .expanduser()
        .resolve()
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite create-once output: {output_dir}")
    stage = output_dir.parent / f".{output_dir.name}.stage"
    if stage.exists():
        raise FileExistsError(f"Stale multi-head recovery stage exists: {stage}")
    stage.mkdir(parents=True)
    try:
        results = []
        pins = {}
        for config in _config_grid():
            result, config_pins = _evaluate_config(
                out_dir,
                output_stage=stage / "config_overlays",
                config=config,
            )
            results.append(result)
            pins[str(config["config_id"])] = config_pins
        selected = _select_best(results)
        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": "consumed_multihead_chord_recovery_spike",
            "experiment_version": OUTPUT_VERSION,
            "status": (
                "advance_to_independent_gate" if selected is not None else "reject_or_revise"
            ),
            "protocol": {
                "fixed_localization": True,
                "no_new_onset_groups": True,
                "bounded_recovered_heads_per_group": True,
                "leading_edge_fail_closed": True,
                "consumed_model_selection_only": True,
                "held_out_claim": False,
                "runtime_adoption_eligible": False,
            },
            "parameter_grid": _config_grid(),
            "results": results,
            "selected": (
                {
                    "config_id": selected["config_id"],
                    "parameters": selected["parameters"],
                    "gate": selected["gate"],
                }
                if selected is not None
                else None
            ),
            "decision": {
                "next_action": (
                    "freeze one unseen polyphonic system before transcription"
                    if selected is not None
                    else "do not freeze or integrate; revise candidate morphology"
                ),
                "runtime_adoption_eligible": False,
            },
            "provenance": {
                "pins_by_config": pins,
                "truth_used_for_model_selection": True,
                "frozen_artifacts_modified": False,
            },
            "artifacts": {
                "output_dir": str(output_dir),
                "report_json": str(output_dir / "report.json"),
                "report_markdown": str(output_dir / "report.md"),
                "config_overlays": str(output_dir / "config_overlays"),
            },
        }
        (stage / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (stage / "report.md").write_text(_write_markdown(report), encoding="utf-8")
        stage.rename(output_dir)
        return report
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _write_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Consumed Multi-Head Chord Recovery",
        "",
        "This is consumed model-selection evidence, not a held-out accuracy claim.",
        "",
        "| Config | Pass | No lo Creas pitch | No lo Creas chord | Candidate FP |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for result in report["results"]:
        no_lo = result["evaluations"]["no_lo_creas"]
        regression = result["evaluations"]["aviador_carrizal_candidate_regression"]
        lines.append(
            f"| `{result['config_id']}` | `{result['gate']['passed']}` | "
            f"`{no_lo['delta_vs_previous_dyad']['exact_diatonic_staff_position_matches']:+d}` | "
            f"`{no_lo['delta_vs_previous_dyad']['exact_chord_size_matches']:+d}` | "
            f"`{regression['edge_safe_recovery']['recovered_false_positive_count']}` |"
        )
    lines.extend(["", "## Decision", ""])
    selected = report.get("selected")
    if selected is None:
        lines.append("No configuration cleared all consumed gates. Do not freeze or integrate.")
    else:
        lines.extend(
            [
                f"Selected `{selected['config_id']}` for one score-disjoint frozen gate.",
                "",
                "The selected rule remains outside runtime until that gate passes.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
