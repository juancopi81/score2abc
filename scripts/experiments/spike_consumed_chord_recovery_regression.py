"""Audit targeted chord recovery on consumed notehead-review examples.

This is in-sample regression evidence only. It replays the frozen consumed
training model against the consumed Aviador and Carrizal examples, reports the
fixed x-only baseline and 2x2 chord-recovery grid, and never selects a winner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import run_third_score_heldout_inference as heldout  # noqa: E402
from scripts.experiments import spike_consumed_polyphonic_pitch_repair as recovery  # noqa: E402
from scripts.experiments import spike_cross_score_consumed_retraining as consumed  # noqa: E402
from scripts.experiments import spike_notehead_patch_templates as patches  # noqa: E402
from scripts.experiments import spike_review_augmented_selector as dense  # noqa: E402

SCHEMA_VERSION = 1
OUTPUT_VERSION = "consumed_chord_recovery_regression_v1"
DEFAULT_OUT_DIR = REPO_ROOT / "out"
DEFAULT_MODEL = (
    DEFAULT_OUT_DIR
    / "vlm_melody_consumed_training"
    / "cross_score_notehead_v1_replay_v2"
    / "model.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_OUT_DIR / "vlm_melody_consumed_training" / OUTPUT_VERSION


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pin(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": _sha256(data), "bytes": len(data)}


def _read_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.expanduser().resolve()
    data = path.read_bytes()
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value, {"path": str(path), "sha256": _sha256(data), "bytes": len(data)}


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def _recovery_config_id(minimum_y_gap: float, maximum_y_gap: float, ratio: float) -> str:
    return (
        f"chord_recovery__min_y_{minimum_y_gap:g}"
        f"__max_y_{maximum_y_gap:g}__score_ratio_{ratio:g}"
    )


def targeted_recovery_configs() -> tuple[dict[str, float | str], ...]:
    """Return the fixed 2x2 recovery grid in deterministic order."""
    return tuple(
        {
            "config_id": _recovery_config_id(minimum, 3.0, ratio),
            "minimum_y_gap_staff_spaces": minimum,
            "maximum_y_gap_staff_spaces": 3.0,
            "minimum_score_ratio": ratio,
        }
        for minimum in recovery.DEFAULT_RECOVERY_MIN_Y_GAP_GRID
        for ratio in recovery.DEFAULT_RECOVERY_SCORE_RATIO_GRID
    )


def targeted_stem_recovery_configs() -> tuple[dict[str, float | str], ...]:
    """Return the fixed stem-score grid crossed with the promising recovery tuple."""
    return tuple(
        {
            "config_id": recovery._stem_recovery_config_id(
                minimum_y_gap_staff_spaces=1.0,
                maximum_y_gap_staff_spaces=3.0,
                minimum_score_ratio=0.5,
                minimum_stem_score=stem_score,
            ),
            "minimum_y_gap_staff_spaces": 1.0,
            "maximum_y_gap_staff_spaces": 3.0,
            "minimum_score_ratio": 0.5,
            "minimum_stem_score": stem_score,
        }
        for stem_score in recovery.DEFAULT_STEM_SCORE_GRID
    )


def _configuration_order() -> tuple[str, ...]:
    return (
        "x_only",
        *(str(config["config_id"]) for config in targeted_recovery_configs()),
        *(str(config["config_id"]) for config in targeted_stem_recovery_configs()),
    )


def _staff_spacing(row: Mapping[str, Any]) -> float:
    geometry = row.get("staff_geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError("Inference row has no staff_geometry")
    lines = geometry.get("raw_staff_lines_y_px")
    if not isinstance(lines, Sequence) or isinstance(lines, (str, bytes)) or len(lines) < 2:
        raise ValueError("Inference row has no usable staff lines")
    spacing = sum(
        abs(float(right) - float(left)) for left, right in zip(lines, lines[1:], strict=False)
    ) / (len(lines) - 1)
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("Inference row has invalid staff spacing")
    return spacing


def _onset_group_count(selected: Sequence[Mapping[str, Any]], x_radius_px: float) -> int:
    if not selected:
        return 0
    xs = sorted(float(candidate["center"]["x"]) for candidate in selected)
    count = 1
    for previous, current in zip(xs, xs[1:], strict=False):
        if current - previous >= x_radius_px:
            count += 1
    return count


def _candidate_ids(selected: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted(str(candidate["candidate_id"]) for candidate in selected)


def candidate_metrics(
    selected_ids: Sequence[str] | set[str],
    truth_ids: Sequence[str] | set[str],
    *,
    recovered_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Score one consumed example and expose recovery false positives."""
    selected = {str(value) for value in selected_ids}
    truth = {str(value) for value in truth_ids}
    recovered = {str(value) for value in recovered_ids}
    true_positive = selected & truth
    false_positive = selected - truth
    false_negative = truth - selected
    recovered_true_positive = recovered & truth
    recovered_false_positive = recovered - truth
    return {
        "tp": len(true_positive),
        "fp": len(false_positive),
        "fn": len(false_negative),
        "precision": _ratio(len(true_positive), len(selected)),
        "recall": _ratio(len(true_positive), len(truth)),
        "f1": _ratio(
            2 * len(true_positive),
            2 * len(true_positive) + len(false_positive) + len(false_negative),
        ),
        "exact_set": selected == truth,
        "selected_count": len(selected),
        "truth_count": len(truth),
        "recovered_count": len(recovered),
        "recovered_true_positive_count": len(recovered_true_positive),
        "recovered_false_positive_count": len(recovered_false_positive),
        "selected_candidate_ids": sorted(selected),
        "truth_candidate_ids": sorted(truth),
        "recovered_candidate_ids": sorted(recovered),
        "recovered_true_positive_ids": sorted(recovered_true_positive),
        "recovered_false_positive_ids": sorted(recovered_false_positive),
    }


def aggregate_candidate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-example candidate metrics without choosing a config."""
    totals = {
        key: 0 for key in ("tp", "fp", "fn", "selected_count", "truth_count", "recovered_count")
    }
    recovered_totals = {
        key: 0 for key in ("recovered_true_positive_count", "recovered_false_positive_count")
    }
    exact_count = 0
    for row in rows:
        for key in totals:
            totals[key] += int(row[key])
        for key in recovered_totals:
            recovered_totals[key] += int(row[key])
        exact_count += int(bool(row["exact_set"]))
    example_count = len(rows)
    selected_counts = [int(row["selected_count"]) for row in rows]
    return {
        "example_count": example_count,
        "tp": totals["tp"],
        "fp": totals["fp"],
        "fn": totals["fn"],
        "precision": _ratio(totals["tp"], totals["tp"] + totals["fp"]),
        "recall": _ratio(totals["tp"], totals["tp"] + totals["fn"]),
        "f1": _ratio(2 * totals["tp"], 2 * totals["tp"] + totals["fp"] + totals["fn"]),
        "exact_set_count": exact_count,
        "exact_set_rate": _ratio(exact_count, example_count),
        "selected_count_total": totals["selected_count"],
        "selected_count_mean": _ratio(totals["selected_count"], example_count),
        "selected_count_min": min(selected_counts) if selected_counts else 0,
        "selected_count_max": max(selected_counts) if selected_counts else 0,
        "truth_count_total": totals["truth_count"],
        "recovered_count_total": totals["recovered_count"],
        "recovered_true_positive_count": recovered_totals["recovered_true_positive_count"],
        "recovered_false_positive_count": recovered_totals["recovered_false_positive_count"],
    }


def _inference_row_for_example(example: Any, model: Any) -> dict[str, Any]:
    measure = example.measure
    unlabeled = patches.UnlabeledMeasure(
        measure=measure.measure,
        source_image=measure.source_image,
        source_sha256=measure.source_sha256,
        staff_lines=measure.staff_lines,
        staff_spacing=measure.staff_spacing,
        candidates=tuple(row.candidate for row in measure.rows),
    )
    candidate_predictions, _ = model.rank(unlabeled, selection_mode=dense.SELECTION_MODE)
    identity = dict(example.request.get("identity", {}))
    identity["automatic_measure_index"] = int(measure.measure)
    return {
        "identity": identity,
        "staff_geometry": {"raw_staff_lines_y_px": list(measure.staff_lines)},
        "candidate_predictions": candidate_predictions,
        "source": {
            "image": str(measure.source_image),
            "sha256": measure.source_sha256,
        },
        "truth_used": False,
    }


def replay_inference_row(
    inference_row: Mapping[str, Any],
    selector: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Replay baseline and every fixed recovery configuration for one row."""
    baseline = recovery.select_candidates(inference_row, selector)
    spacing = _staff_spacing(inference_row)
    x_radius = spacing * float(selector["nms_x_spaces"])
    results: dict[str, dict[str, Any]] = {
        "x_only": {
            "config_id": "x_only",
            "config_family": "x_only_baseline",
            "selected": baseline,
            "recovered": [],
            "onset_group_count": _onset_group_count(baseline, x_radius),
            "baseline_onset_group_count": None,
            "recovery_invariants": None,
        }
    }
    baseline_group_count = _onset_group_count(baseline, x_radius)
    for config in targeted_recovery_configs():
        recovered = recovery.recover_chord_candidates(
            inference_row,
            selector,
            baseline,
            minimum_y_gap_staff_spaces=float(config["minimum_y_gap_staff_spaces"]),
            maximum_y_gap_staff_spaces=float(config["maximum_y_gap_staff_spaces"]),
            minimum_score_ratio=float(config["minimum_score_ratio"]),
        )
        selected = [*baseline, *recovered]
        onset_group_count = _onset_group_count(selected, x_radius)
        recovered_group_ids = [item.get("recovery_group_index") for item in recovered]
        no_new_group = all(
            any(
                abs(float(item["center"]["x"]) - float(anchor["center"]["x"])) < x_radius
                for anchor in baseline
            )
            for item in recovered
        )
        invariants = {
            "onset_group_count_unchanged": onset_group_count == baseline_group_count,
            "no_new_onset_groups": no_new_group,
            "at_most_one_recovered_head_per_group": len(recovered_group_ids)
            == len(set(recovered_group_ids)),
        }
        if not all(invariants.values()):
            raise ValueError(f"Recovery invariants failed for {config['config_id']}")
        results[str(config["config_id"])] = {
            "config_id": config["config_id"],
            "config_family": "chord_recovery",
            "selected": selected,
            "recovered": recovered,
            "onset_group_count": onset_group_count,
            "baseline_onset_group_count": baseline_group_count,
            "recovery_invariants": invariants,
            "recovery_parameters": dict(config),
        }
    if "source" in inference_row or "source_image" in inference_row:
        stem_features, stem_metadata = recovery.candidate_local_stem_features(inference_row)
        for config in targeted_stem_recovery_configs():
            recovered = recovery.recover_stem_aware_chord_candidates(
                inference_row,
                selector,
                baseline,
                minimum_y_gap_staff_spaces=float(config["minimum_y_gap_staff_spaces"]),
                maximum_y_gap_staff_spaces=float(config["maximum_y_gap_staff_spaces"]),
                minimum_score_ratio=float(config["minimum_score_ratio"]),
                minimum_stem_score=float(config["minimum_stem_score"]),
                stem_features=stem_features,
            )
            selected = [*baseline, *recovered]
            onset_group_count = _onset_group_count(selected, x_radius)
            recovered_group_ids = [item.get("recovery_group_index") for item in recovered]
            invariants = {
                "onset_group_count_unchanged": onset_group_count == baseline_group_count,
                "no_new_onset_groups": all(
                    any(
                        abs(float(item["center"]["x"]) - float(anchor["center"]["x"])) < x_radius
                        for anchor in baseline
                    )
                    for item in recovered
                ),
                "at_most_one_recovered_head_per_group": len(recovered_group_ids)
                == len(set(recovered_group_ids)),
            }
            if not all(invariants.values()):
                raise ValueError(f"Stem recovery invariants failed for {config['config_id']}")
            results[str(config["config_id"])] = {
                "config_id": config["config_id"],
                "config_family": "chord_recovery_stem",
                "selected": selected,
                "recovered": recovered,
                "onset_group_count": onset_group_count,
                "baseline_onset_group_count": baseline_group_count,
                "recovery_invariants": invariants,
                "recovery_parameters": dict(config),
                "stem_feature_diagnostics": stem_features,
                "stem_feature_metadata": stem_metadata,
            }
    return results


def _example_result(example: Any, prediction: Mapping[str, Any]) -> dict[str, Any]:
    identity = dict(example.request.get("identity", {}))
    slug = str(identity.get("slug", "unknown"))
    system_index = identity.get("system_index")
    source_system = f"{slug}:system_{int(system_index):03d}" if system_index is not None else slug
    metrics = candidate_metrics(
        _candidate_ids(prediction["selected"]),
        example.matched_candidate_ids,
        recovered_ids=_candidate_ids(prediction["recovered"]),
    )
    metrics.update(
        {
            "example_key": str(example.key),
            "source_system": source_system,
            "identity": identity,
            "onset_group_count": prediction["onset_group_count"],
            "baseline_onset_group_count": prediction["baseline_onset_group_count"],
            "recovery_invariants": prediction["recovery_invariants"],
        }
    )
    if prediction.get("recovery_parameters") is not None:
        metrics["recovery_parameters"] = prediction["recovery_parameters"]
    if prediction.get("stem_feature_diagnostics") is not None:
        metrics["stem_feature_diagnostics"] = prediction["stem_feature_diagnostics"]
        metrics["stem_feature_metadata"] = prediction["stem_feature_metadata"]
    return metrics


def _configuration_result(
    examples: Sequence[Any],
    predictions: Mapping[str, Mapping[str, Any]],
    config_id: str,
) -> dict[str, Any]:
    example_rows = [_example_result(example, predictions[example.key]) for example in examples]
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in example_rows:
        by_source[str(row["source_system"])].append(row)
    return {
        "config_id": config_id,
        "config_family": str(predictions[examples[0].key]["config_family"]),
        "runtime_adoption_eligible": False,
        "parameters": predictions[examples[0].key].get("recovery_parameters"),
        "overall": aggregate_candidate_metrics(example_rows),
        "by_source_system": {
            source: aggregate_candidate_metrics(sorted(rows, key=lambda row: row["example_key"]))
            for source, rows in sorted(by_source.items())
        },
        "examples": sorted(example_rows, key=lambda row: row["example_key"]),
    }


def build_report(
    *,
    selector: Mapping[str, Any],
    model_audit: Mapping[str, Any],
    configurations: Mapping[str, Mapping[str, Any]],
    input_pins: Mapping[str, Any],
    output_dir: Path,
    example_count: int,
) -> dict[str, Any]:
    """Build the audit envelope; no truth-based configuration selection occurs."""
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "consumed_chord_recovery_regression_audit",
        "experiment_version": OUTPUT_VERSION,
        "status": "evaluated_consumed_in_sample_create_once",
        "protocol": {
            "dataset": "promoted consumed Aviador and Carrizal notehead examples",
            "example_count": example_count,
            "baseline": "exact x-only candidate selection replay",
            "targeted_grid": [
                *[dict(config) for config in targeted_recovery_configs()],
                *[dict(config) for config in targeted_stem_recovery_configs()],
            ],
            "truth_policy": (
                "Consumed matched_candidate_ids are read only for scoring. They never select, "
                "fit, or rank a configuration. No La Chata truth or MusicXML is read."
            ),
            "selection_rule": "none; every fixed configuration is reported",
        },
        "eligibility": {
            "held_out": False,
            "in_sample_consumed_evidence": True,
            "accuracy_claim": False,
            "runtime_adoption_eligible": False,
            "winner_selected_from_truth": False,
        },
        "model_reconstruction": dict(model_audit),
        "selector": dict(selector),
        "configurations": {
            key: configurations[key] for key in _configuration_order() if key in configurations
        },
        "provenance": {
            "inputs": dict(input_pins),
            "sealed_inputs_written": False,
            "original_predictions_rewritten": False,
            "existing_evidence_overwritten": False,
        },
        "artifacts": {
            "output_dir": str(output_dir.resolve()),
            "report_json": str((output_dir / "report.json").resolve()),
            "report_markdown": str((output_dir / "report.md").resolve()),
        },
    }


def _write_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Consumed Chord-Recovery Regression Audit",
        "",
        "This is consumed in-sample regression evidence, not held-out accuracy evidence.",
        "It is not runtime-adoption eligible and no configuration was selected from truth.",
        "",
        "| Configuration | TP | FP | FN | F1 | Exact-set | Selected | Recovered | Recovered FP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for config_id, config in report["configurations"].items():
        metrics = config["overall"]
        lines.append(
            f"| {config_id} | {metrics['tp']} | {metrics['fp']} | {metrics['fn']} | "
            f"{metrics['f1']:.6f} | {metrics['exact_set_rate']:.6f} | "
            f"{metrics['selected_count_total']} | {metrics['recovered_count_total']} | "
            f"{metrics['recovered_false_positive_count']} |"
        )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "- Held out: `false`",
            "- Accuracy claim: `false`",
            "- Runtime adoption eligible: `false`",
            "- Configuration selection: `none`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_create_once(output_dir: Path, report: Mapping[str, Any]) -> None:
    """Write JSON and Markdown atomically, refusing an existing directory."""
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite create-once output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "report.md").write_text(_write_markdown(report), encoding="utf-8")
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite create-once output: {output_dir}")
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _resolve_source_record(record: Mapping[str, Any]) -> dict[str, Any]:
    raw_path = Path(str(record["path"]))
    path = raw_path if raw_path.is_absolute() else REPO_ROOT / raw_path
    pin = _pin(path)
    if pin["sha256"] != str(record["sha256"]):
        raise ValueError(f"Consumed input hash drift: {path}")
    return pin


def _load_examples(
    out_dir: Path,
    *,
    reviews_dir: Path,
    carrizal_reviews: Path,
) -> tuple[list[Any], list[dict[str, Any]]]:
    aviador, aviador_sources = consumed._load_aviador_examples(
        out_dir,
        reviews_dir=reviews_dir,
    )
    carrizal_by_policy, carrizal_sources = consumed._load_carrizal_examples(
        out_dir,
        manifest_path=carrizal_reviews,
    )
    examples = sorted([*aviador, *carrizal_by_policy["C"]], key=lambda item: item.key)
    source_records = [
        _resolve_source_record(record) for record in [*aviador_sources, *carrizal_sources]
    ]
    return examples, sorted(source_records, key=lambda item: item["path"])


def run_audit(
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    model_path: Path = DEFAULT_MODEL,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    reviews_dir: Path = consumed.DEFAULT_REVIEWS_DIR,
    carrizal_reviews: Path = consumed.DEFAULT_CARRIZAL_REVIEWS,
) -> dict[str, Any]:
    """Run the bounded consumed-score audit and create its report once."""
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite create-once output: {output_dir}")
    out_dir = out_dir.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    reviews_dir = reviews_dir.expanduser().resolve()
    carrizal_reviews = carrizal_reviews.expanduser().resolve()

    model_payload, model_pin = _read_json_object(model_path, label="frozen model")
    model, _pitch_predictor, model_audit = heldout.reconstruct_model(model_payload)
    selector = recovery.selector_config_from_model(model_payload)
    examples, source_pins = _load_examples(
        out_dir,
        reviews_dir=reviews_dir,
        carrizal_reviews=carrizal_reviews,
    )
    predictions_by_config: dict[str, dict[str, Any]] = defaultdict(dict)
    for example in examples:
        inference_row = _inference_row_for_example(example, model)
        replayed = replay_inference_row(inference_row, selector)
        for config_id, prediction in replayed.items():
            predictions_by_config[config_id][example.key] = prediction
    configurations = {
        config_id: _configuration_result(examples, predictions_by_config[config_id], config_id)
        for config_id in _configuration_order()
    }
    report = build_report(
        selector=selector,
        model_audit=model_audit,
        configurations=configurations,
        input_pins={"model": model_pin, "consumed_sources": source_pins},
        output_dir=output_dir,
        example_count=len(examples),
    )
    write_create_once(output_dir, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reviews-dir", type=Path, default=consumed.DEFAULT_REVIEWS_DIR)
    parser.add_argument("--carrizal-reviews", type=Path, default=consumed.DEFAULT_CARRIZAL_REVIEWS)
    args = parser.parse_args(argv)
    try:
        report = run_audit(
            out_dir=args.out_dir,
            model_path=args.model,
            output_dir=args.output_dir,
            reviews_dir=args.reviews_dir,
            carrizal_reviews=args.carrizal_reviews,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(report["artifacts"]["report_json"])
    print(report["artifacts"]["report_markdown"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
