"""Evaluate a fail-closed dyad recovery rule on consumed score evidence.

The fixed rule starts from the frozen x-only selector and can add one
stem-supported companion to an existing onset group. It refuses candidates in
the first staff-space of a crop, where barline and preamble ink are ambiguous.
Alcira selects the rule; La Chata and the Aviador/Carrizal review bundle are
reported as consumed cross-score checks. This script never rewrites a freeze
and cannot make a held-out or runtime-adoption claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import evaluate_frozen_third_score_heldout as heldout  # noqa: E402
from scripts.experiments import run_independent_key_state_gate as key_runner  # noqa: E402
from scripts.experiments import spike_consumed_chord_recovery_regression as regression  # noqa: E402
from scripts.experiments import spike_consumed_polyphonic_pitch_repair as recovery  # noqa: E402

SCHEMA_VERSION = 1
OUTPUT_VERSION = "consumed_edge_safe_dyad_recovery_v2"
CONFIG_ID = recovery.EDGE_SAFE_STEM_DYAD_CONFIG_ID
PARAMETERS = recovery.EDGE_SAFE_STEM_DYAD_PARAMETERS

ALCIRA_SLUG = "jaime-llanos_5_alcira_bambuco_oriol-rangel"
LA_CHATA_SLUG = "jaime-llanos_64_la-chata_pasillo_luis-a-calvo"

RecoveryRow = Callable[
    [Mapping[str, Any], Mapping[str, Any]],
    tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]],
]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pin(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": _sha256(data), "bytes": len(data)}


def _read_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    pin = _pin(path)
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value, pin


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pin = _pin(path)
    rows = []
    for line_number, line in enumerate(
        path.expanduser().resolve().read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"Expected at least one row: {path}")
    return rows, pin


def _measure(row: Mapping[str, Any]) -> int:
    identity = row.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("Inference row has no identity")
    return int(identity["automatic_measure_index"])


def _recover_row(
    row: Mapping[str, Any], selector: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    baseline = recovery.select_candidates(row, selector)
    stem_features, _ = recovery.candidate_local_stem_features(row)
    recovered = recovery.recover_edge_safe_stem_aware_chord_candidates(
        row,
        selector,
        baseline,
        stem_features=stem_features,
        **PARAMETERS,
    )
    return baseline, recovered, stem_features


def _anchors(selected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    anchors = []
    for order, candidate in enumerate(
        sorted(
            selected,
            key=lambda item: (
                float(item["center"]["x"]),
                float(item["center"]["y"]),
                str(item["candidate_id"]),
            ),
        ),
        start=1,
    ):
        source = candidate.get("source")
        if not isinstance(source, Mapping):
            raise ValueError(f"Candidate {candidate['candidate_id']} has no source payload")
        anchors.append(
            {
                "order": order,
                "center": dict(candidate["center"]),
                "source": dict(source),
            }
        )
    return anchors


def _paired_prediction_map(
    rows: Sequence[Mapping[str, Any]], lane: str
) -> dict[int, dict[str, Any]]:
    return {_measure(row): {"notes": list(row["lanes"][lane]["notes"])} for row in rows}


def _evaluate_alcira(
    *,
    out_dir: Path,
    output_dir: Path,
    recover_row: RecoveryRow = _recover_row,
    config_id: str = CONFIG_ID,
    parameters: Mapping[str, Any] = PARAMETERS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = (
        out_dir
        / "local_restricted"
        / ALCIRA_SLUG
        / "vlm_melody_independent_key_gate"
        / "v1_alcira_system_entry_change_s6"
        / "system_006"
    )
    paths = {
        "inference": root / "inference_v1/inference.jsonl",
        "paired": root / "evaluation_v1/case/frozen_paired_predictions.jsonl",
        "truth": root / "evaluation_v1/case/truth.jsonl",
        "requests": root / "requests.jsonl",
        "model": root / "frozen/artifacts/model/001_model.json",
    }
    inference_rows, inference_pin = _read_jsonl(paths["inference"])
    baseline_paired, paired_pin = _read_jsonl(paths["paired"])
    requests, requests_pin = _read_jsonl(paths["requests"])
    model, model_pin = _read_json(paths["model"])
    selector = recovery.selector_config_from_model(model)

    recovered_rows = []
    recovery_records = []
    overlay_dir = output_dir / "overlays" / "alcira_system_006"
    for row in inference_rows:
        baseline, recovered, stem_features = recover_row(row, selector)
        updated = dict(row)
        updated["automatic_anchors"] = _anchors([*baseline, *recovered])
        recovered_rows.append(updated)
        measure = _measure(row)
        recovery_records.append(
            {
                "automatic_measure_index": measure,
                "baseline_candidate_ids": [item["candidate_id"] for item in baseline],
                "recovered_candidate_ids": [item["candidate_id"] for item in recovered],
                "recovered": [
                    {
                        "candidate_id": item["candidate_id"],
                        "center": dict(item["center"]),
                        "score": item["score"],
                        "stem_score": stem_features[str(item["candidate_id"])]["score"],
                        "leading_edge_distance_staff_spaces": item[
                            "leading_edge_distance_staff_spaces"
                        ],
                    }
                    for item in recovered
                ],
            }
        )
        _render_overlay(
            row,
            baseline=baseline,
            recovered=recovered,
            output_path=overlay_dir / f"measure_{measure:03d}.png",
        )

    crop_left_by_measure = {
        int(row["identity"]["automatic_measure_index"]): int(row["input"]["bbox_px"][0])
        for row in requests
    }
    recovered_paired, invariance = key_runner.build_paired_predictions(
        recovered_rows,
        automatic_fifths=2,
        key_event_x_px=328,
        crop_left_by_measure=crop_left_by_measure,
    )

    # Predictions are fixed before opening consumed transcription truth.
    truth_rows, truth_pin = _read_jsonl(paths["truth"])
    target = {"slug": ALCIRA_SLUG, "system_index": 6}
    baseline_report = heldout.evaluate_pitch_only(
        truth_rows,
        _paired_prediction_map(baseline_paired, "global_automatic_key"),
        target=target,
        mapping_mode="existing_consumed_alcira_mapping",
        report_kind="consumed_alcira_edge_safe_dyad_baseline",
    )
    recovered_report = heldout.evaluate_pitch_only(
        truth_rows,
        _paired_prediction_map(recovered_paired, "global_automatic_key"),
        target=target,
        mapping_mode="existing_consumed_alcira_mapping",
        report_kind="consumed_alcira_edge_safe_dyad_recovered",
    )
    baseline_summary = baseline_report["metrics"]["summary"]
    recovered_summary = recovered_report["metrics"]["summary"]
    return (
        {
            "target": target,
            "selector": selector,
            "parameters": dict(parameters),
            "baseline": baseline_summary,
            "edge_safe_recovery": recovered_summary,
            "delta": {
                "predicted_note_count": recovered_summary["predicted_note_count"]
                - baseline_summary["predicted_note_count"],
                "exact_pitch_matches": recovered_summary["exact_pitch_matches"]
                - baseline_summary["exact_pitch_matches"],
                "note_count_f1": round(
                    recovered_summary["note_count_f1"] - baseline_summary["note_count_f1"],
                    6,
                ),
            },
            "selection_invariance": invariance,
            "recovery_records": recovery_records,
            "overlay_dir": "overlays/alcira_system_006",
        },
        {
            "inference": inference_pin,
            "paired_baseline": paired_pin,
            "truth": truth_pin,
            "requests": requests_pin,
            "model": model_pin,
        },
    )


def _materialize_la_chata(
    rows: Sequence[Mapping[str, Any]],
    selector: Mapping[str, Any],
    context_hints: Mapping[int, Any],
    *,
    recover_row: RecoveryRow = _recover_row,
    config_id: str = CONFIG_ID,
    parameters: Mapping[str, Any] = PARAMETERS,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], list[dict[str, Any]]]:
    baseline_by_measure = {}
    recovered_by_measure = {}
    records = []
    for row in rows:
        measure = _measure(row)
        baseline, recovered, stem_features = recover_row(row, selector)
        hint = context_hints.get(measure, recovery._row_key_hint(row))
        alterations = recovery._key_accidentals(hint)
        baseline_by_measure[measure] = recovery._materialize_selected_prediction(
            row,
            selector,
            baseline,
            config_id="x_only",
            config_family="x_only_baseline",
            lane="consumed_context",
            alterations=alterations,
            truth_alterations=None,
        )
        recovered_by_measure[measure] = recovery._materialize_selected_prediction(
            row,
            selector,
            [*baseline, *recovered],
            config_id=config_id,
            config_family="edge_safe_stem_dyad_recovery",
            lane="consumed_context",
            alterations=alterations,
            truth_alterations=None,
            recovered=recovered,
            baseline_group_count=recovery._onset_group_count(
                baseline,
                recovery._staff_spacing(row) * float(selector["nms_x_spaces"]),
            ),
            recovery_parameters=parameters,
            stem_features=stem_features,
        )
        records.append(
            {
                "automatic_measure_index": measure,
                "recovered_candidate_ids": [item["candidate_id"] for item in recovered],
            }
        )
    return baseline_by_measure, recovered_by_measure, records


def _evaluate_la_chata(
    out_dir: Path,
    *,
    recover_row: RecoveryRow = _recover_row,
    config_id: str = CONFIG_ID,
    parameters: Mapping[str, Any] = PARAMETERS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = out_dir / LA_CHATA_SLUG / "vlm_melody_third_score_heldout/v2/system_007"
    paths = {
        "inference": root / "inference_v2/inference.jsonl",
        "truth": root / "evaluation_v1/truth.jsonl",
        "context": root / "context_hints_v1.json",
        "model": out_dir
        / "vlm_melody_consumed_training/cross_score_notehead_v1_replay_v2/model.json",
    }
    rows, inference_pin = _read_jsonl(paths["inference"])
    model, model_pin = _read_json(paths["model"])
    context_hints, context_pin = recovery._load_context_hints(
        paths["context"], measure_indices=[_measure(row) for row in rows]
    )
    if context_pin is None:
        raise ValueError("La Chata consumed context was not loaded")
    selector = recovery.selector_config_from_model(model)
    baseline, recovered, records = _materialize_la_chata(
        rows,
        selector,
        context_hints,
        recover_row=recover_row,
        config_id=config_id,
        parameters=parameters,
    )

    # Both candidate lanes are fixed before opening consumed truth.
    truth_rows, truth_pin = _read_jsonl(paths["truth"])
    truth_by_measure = {int(row["automatic_crop_index"]): row for row in truth_rows}
    baseline_score = recovery._score_lane(
        {"x_only": baseline}, truth_by_measure, selector=selector
    )[0]
    recovered_score = recovery._score_lane(
        {config_id: recovered}, truth_by_measure, selector=selector
    )[0]
    return (
        {
            "target": {"slug": LA_CHATA_SLUG, "system_index": 7},
            "baseline": baseline_score["metrics"],
            "edge_safe_recovery": recovered_score["metrics"],
            "recovery_records": records,
        },
        {
            "inference": inference_pin,
            "truth": truth_pin,
            "context": context_pin,
            "model": model_pin,
        },
    )


def _evaluate_candidate_regression(
    out_dir: Path,
    *,
    recover_row: RecoveryRow = _recover_row,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_path = (
        out_dir / "vlm_melody_consumed_training/cross_score_notehead_v1_replay_v2/model.json"
    )
    model_payload, model_pin = _read_json(model_path)
    model, _pitch_predictor, model_audit = regression.heldout.reconstruct_model(model_payload)
    selector = recovery.selector_config_from_model(model_payload)
    examples, source_pins = regression._load_examples(
        out_dir,
        reviews_dir=regression.consumed.DEFAULT_REVIEWS_DIR,
        carrizal_reviews=regression.consumed.DEFAULT_CARRIZAL_REVIEWS,
    )
    baseline_metrics = []
    recovered_metrics = []
    recovered_examples = []
    for example in examples:
        row = regression._inference_row_for_example(example, model)
        baseline, recovered, _stem_features = recover_row(row, selector)
        baseline_metrics.append(
            regression.candidate_metrics(
                regression._candidate_ids(baseline), example.matched_candidate_ids
            )
        )
        recovered_metrics.append(
            regression.candidate_metrics(
                regression._candidate_ids([*baseline, *recovered]),
                example.matched_candidate_ids,
                recovered_ids=regression._candidate_ids(recovered),
            )
        )
        if recovered:
            recovered_examples.append(
                {
                    "example_key": str(example.key),
                    "recovered_candidate_ids": regression._candidate_ids(recovered),
                }
            )
    return (
        {
            "example_count": len(examples),
            "selector": selector,
            "baseline": regression.aggregate_candidate_metrics(baseline_metrics),
            "edge_safe_recovery": regression.aggregate_candidate_metrics(recovered_metrics),
            "recovered_examples": recovered_examples,
            "model_reconstruction": model_audit,
        },
        {"model": model_pin, "consumed_sources": source_pins},
    )


def _render_overlay(
    row: Mapping[str, Any],
    *,
    baseline: Sequence[Mapping[str, Any]],
    recovered: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    source = row.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("Inference row has no source image")
    image_path = Path(str(source["image"]))
    if not image_path.is_absolute():
        image_path = REPO_ROOT / image_path
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    for color, candidates in (((20, 90, 220), baseline), ((0, 160, 70), recovered)):
        for candidate in candidates:
            payload = candidate.get("source")
            bbox = payload.get("bbox") if isinstance(payload, Mapping) else None
            if not isinstance(bbox, Mapping):
                continue
            bounds = (
                int(bbox["left"]),
                int(bbox["top"]),
                int(bbox["right"]),
                int(bbox["bottom"]),
            )
            draw.rectangle(bounds, outline=color, width=2)
            draw.text(
                (bounds[0], max(0, bounds[1] - 11)), str(candidate["candidate_id"]), fill=color
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _decision(
    alcira: Mapping[str, Any],
    la_chata: Mapping[str, Any],
    candidate_regression: Mapping[str, Any],
) -> dict[str, Any]:
    alcira_improved = float(alcira["edge_safe_recovery"]["note_count_f1"]) > float(
        alcira["baseline"]["note_count_f1"]
    ) and int(alcira["edge_safe_recovery"]["exact_pitch_matches"]) > int(
        alcira["baseline"]["exact_pitch_matches"]
    )
    baseline_regression = candidate_regression["baseline"]
    recovered_regression = candidate_regression["edge_safe_recovery"]
    regression_preserved = (
        float(recovered_regression["f1"]) >= float(baseline_regression["f1"])
        and int(recovered_regression["recovered_false_positive_count"]) == 0
    )
    la_chata_baseline = la_chata["baseline"]["selection_count_metrics"]
    la_chata_recovered = la_chata["edge_safe_recovery"]["selection_count_metrics"]
    baseline_groups = la_chata["baseline"]["pitch_group_metrics"]
    recovered_groups = la_chata["edge_safe_recovery"]["pitch_group_metrics"]
    la_chata_capacity_delta = int(la_chata_recovered["predicted_total_head_count"]) - int(
        la_chata_baseline["predicted_total_head_count"]
    )
    la_chata_structural_improved = int(recovered_groups["exact_group_count"]) > int(
        baseline_groups["exact_group_count"]
    ) and int(recovered_groups["group_edit_distance"]) < int(baseline_groups["group_edit_distance"])
    passed = alcira_improved and la_chata_structural_improved and regression_preserved
    return {
        "status": "advance_to_independent_gate" if passed else "reject_or_revise",
        "alcira_consumed_improved": alcira_improved,
        "aviador_carrizal_candidate_regression_preserved": regression_preserved,
        "la_chata_consumed_head_capacity_delta": la_chata_capacity_delta,
        "la_chata_consumed_pitch_groups_improved": la_chata_structural_improved,
        "runtime_adoption_eligible": False,
        "next_action": (
            "freeze one unseen polyphonic system before transcription"
            if passed
            else "revise the candidate-local rule without runtime integration"
        ),
    }


def _write_markdown(report: Mapping[str, Any]) -> str:
    alcira = report["evaluations"]["alcira"]
    la_chata = report["evaluations"]["la_chata"]
    regression_report = report["evaluations"]["aviador_carrizal_candidate_regression"]
    decision = report["decision"]
    lines = [
        "# Consumed Edge-Safe Dyad Recovery",
        "",
        "This is consumed model-selection evidence, not a held-out accuracy claim.",
        "",
        "## Alcira system 6",
        "",
        f"- Selected heads: `{alcira['baseline']['predicted_note_count']} -> "
        f"{alcira['edge_safe_recovery']['predicted_note_count']}`",
        f"- Exact pitch matches: `{alcira['baseline']['exact_pitch_matches']} -> "
        f"{alcira['edge_safe_recovery']['exact_pitch_matches']}`",
        f"- Note-count F1: `{alcira['baseline']['note_count_f1']:.6f} -> "
        f"{alcira['edge_safe_recovery']['note_count_f1']:.6f}`",
        "",
        "## La Chata system 7",
        "",
        "- Selected heads: "
        f"`{la_chata['baseline']['selection_count_metrics']['predicted_total_head_count']} -> "
        f"{la_chata['edge_safe_recovery']['selection_count_metrics']['predicted_total_head_count']}`",
        "- Exact pitch groups: "
        f"`{la_chata['baseline']['pitch_group_metrics']['exact_group_count']} -> "
        f"{la_chata['edge_safe_recovery']['pitch_group_metrics']['exact_group_count']}`",
        "- Pitch-group edit distance: "
        f"`{la_chata['baseline']['pitch_group_metrics']['group_edit_distance']} -> "
        f"{la_chata['edge_safe_recovery']['pitch_group_metrics']['group_edit_distance']}`",
        "",
        "## Aviador/Carrizal candidate regression",
        "",
        f"- Candidate F1: `{regression_report['baseline']['f1']:.6f} -> "
        f"{regression_report['edge_safe_recovery']['f1']:.6f}`",
        "- Recovered false positives: "
        f"`{regression_report['edge_safe_recovery']['recovered_false_positive_count']}`",
        "",
        "## Decision",
        "",
        f"**{decision['status']}**. {decision['next_action']}.",
        "",
        "The rule remains outside runtime until an unseen polyphonic system passes.",
        "",
    ]
    return "\n".join(lines)


def run_spike(*, out_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    out_dir = out_dir.expanduser().resolve()
    output_dir = (
        (
            output_dir
            or out_dir
            / "local_restricted"
            / ALCIRA_SLUG
            / "vlm_melody_independent_key_gate/v1_alcira_system_entry_change_s6/system_006"
            / "postmortem_edge_safe_dyad_recovery_v2"
        )
        .expanduser()
        .resolve()
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite create-once output: {output_dir}")

    # Render overlays to a temporary sibling so the report remains atomic.
    overlay_stage = output_dir.parent / f".{output_dir.name}.overlay-stage"
    if overlay_stage.exists():
        raise FileExistsError(f"Stale overlay staging directory exists: {overlay_stage}")
    try:
        alcira, alcira_pins = _evaluate_alcira(out_dir=out_dir, output_dir=overlay_stage)
        la_chata, la_chata_pins = _evaluate_la_chata(out_dir)
        candidate_regression, regression_pins = _evaluate_candidate_regression(out_dir)
        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": "consumed_edge_safe_dyad_recovery_spike",
            "experiment_version": OUTPUT_VERSION,
            "status": "evaluated_consumed_model_selection_evidence",
            "config_id": CONFIG_ID,
            "parameters": dict(PARAMETERS),
            "protocol": {
                "fixed_localization": True,
                "no_new_onset_groups": True,
                "at_most_one_recovered_head_per_existing_group": True,
                "leading_edge_fail_closed": True,
                "alcira_role": "consumed_rule_selection",
                "la_chata_role": "consumed_polyphonic_cross_score_check",
                "aviador_carrizal_role": "consumed_candidate_regression_check",
                "held_out_claim": False,
                "runtime_adoption_eligible": False,
            },
            "evaluations": {
                "alcira": alcira,
                "la_chata": la_chata,
                "aviador_carrizal_candidate_regression": candidate_regression,
            },
            "decision": _decision(alcira, la_chata, candidate_regression),
            "provenance": {
                "inputs": {
                    "alcira": alcira_pins,
                    "la_chata": la_chata_pins,
                    "aviador_carrizal": regression_pins,
                },
                "frozen_artifacts_modified": False,
                "truth_used_to_train_selector": False,
            },
            "artifacts": {
                "output_dir": str(output_dir),
                "report_json": str(output_dir / "report.json"),
                "report_markdown": str(output_dir / "report.md"),
                "alcira_overlays": str(output_dir / "overlays/alcira_system_006"),
            },
        }
        staged_overlays = overlay_stage / "overlays"
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir()
        if staged_overlays.exists():
            shutil.move(str(staged_overlays), str(output_dir / "overlays"))
        (output_dir / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (output_dir / "report.md").write_text(_write_markdown(report), encoding="utf-8")
        return report
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(overlay_stage, ignore_errors=True)


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


if __name__ == "__main__":
    raise SystemExit(main())
