"""Combine dense 100%-recall proposals with review-trained patch templates.

The dense logistic and cap-24 patch experiments isolate complementary strengths:
dense proposals cover every reviewed head, while staff-suppressed patch models
classify the conservative candidates much more accurately. This arm combines
those components. Method and threshold selection use only S1+S7 review OOF
predictions; S8 predictions are frozen before evaluation and S3 is untouched.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import spike_cap24_review_augmented_selector as cap  # noqa: E402
from scripts.experiments import spike_composed_melody_chain as composed  # noqa: E402
from scripts.experiments import spike_notehead_patch_templates as patches  # noqa: E402
from scripts.experiments import spike_review_augmented_selector as dense  # noqa: E402

DEFAULT_OUT_DIR = REPO_ROOT / "out"
DEFAULT_SLUG = patches.DEFAULT_SLUG
DEFAULT_REVIEWS_DIR = REPO_ROOT / "tests/fixtures/vlm_melody/notehead_reviews"
OUTPUT_SUBDIR = "dense_patch_review_augmented_selector"
METHOD_PREFIX = "dense_patch_s1_s7"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--reviews-dir", type=Path, default=DEFAULT_REVIEWS_DIR)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        report = run_experiment(
            args.out_dir,
            slug=args.slug,
            reviews_dir=args.reviews_dir,
            output_dir=args.output_dir,
        )
    except (FileNotFoundError, KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(report["artifacts"]["report_json"])
    print(report["artifacts"]["report_markdown"])
    print(f"system-8 gate: {report['validation_gate']['status']}")
    return 0


def run_experiment(
    out_dir: Path,
    *,
    slug: str = DEFAULT_SLUG,
    reviews_dir: Path = DEFAULT_REVIEWS_DIR,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    benchmark_dir = out_dir / slug / "vlm_melody_event_benchmark"
    output_dir = output_dir or benchmark_dir / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    development_requests_path = benchmark_dir / "development/requests.jsonl"
    validation_requests_path = benchmark_dir / "validation/requests.jsonl"
    development_requests = dense._read_jsonl(development_requests_path)
    validation_requests = dense._read_jsonl(validation_requests_path)
    training_requests = dense._select_requests(
        [*development_requests, *validation_requests], dense.TRAINING_TARGETS
    )
    system8_requests = dense._select_requests(validation_requests, dense.VALIDATION_TARGETS)

    training_unlabeled = [
        dense._prepare_dense_measure(request, out_dir=out_dir) for request in training_requests
    ]
    validation_unlabeled = [
        dense._prepare_dense_measure(request, out_dir=out_dir) for request in system8_requests
    ]
    training = [
        dense._attach_review(
            request,
            measure,
            reviews_dir=reviews_dir,
            slug=slug,
        )
        for request, measure in zip(training_requests, training_unlabeled, strict=True)
    ]

    methods = []
    for patch_spec in patches.PATCH_SPECS:
        scorer_kind = "class_template"
        oof_scores = cap._out_of_fold_scores(
            training,
            patch_id=patch_spec.id,
            scorer_kind=scorer_kind,
        )
        selection = dense._select_training_configuration(training, oof_scores)
        methods.append(
            {
                "method_id": f"{scorer_kind}__{patch_spec.id}",
                "patch_id": patch_spec.id,
                "scorer_kind": scorer_kind,
                "selection": selection,
                "oof_scores": oof_scores,
            }
        )
    winner = max(
        methods,
        key=lambda method: (
            float(method["selection"]["metrics"]["f1"]),
            float(method["selection"]["metrics"]["recall"]),
            float(method["selection"]["metrics"]["precision"]),
            str(method["method_id"]),
        ),
    )
    method_id = f"{METHOD_PREFIX}__{winner['method_id']}"
    full_scorer = patches._fit_patch_scorer(
        [row for example in training for row in example.measure.rows],
        patch_id=str(winner["patch_id"]),
        scorer_kind=str(winner["scorer_kind"]),
    )
    selection = winner["selection"]
    model = dense.DenseSelectorModel(
        scorer=full_scorer,  # type: ignore[arg-type]
        learned_threshold=float(selection["threshold"]),
        threshold_training_metrics=dict(selection["metrics"]),
        training_keys=tuple(example.key for example in training),
        training_positive_count=sum(example.true_note_count for example in training),
        learned_count=int(statistics.median(example.true_note_count for example in training)),
        nms_x_spaces=float(selection["nms_x_spaces"]),
        minimum_selected_count=int(selection["minimum_selected_count"]),
        maximum_selected_count=int(selection["maximum_selected_count"]),
    )
    pitch_cv = dense._evaluate_pitch_methods(training)
    pitch_method = str(pitch_cv["selected_method"])
    accidental_model = (
        dense._fit_accidental_model(training) if pitch_method == "accidental_knn" else None
    )
    pitch_predictor = dense._build_pitch_predictor(accidental_model)

    training_snapshot_path = output_dir / "training_selection.json"
    training_snapshot = {
        "status": "selected_before_validation_prediction",
        "training_targets": [example.key for example in training],
        "review_hashes": [example.measure.review_sha256 for example in training],
        "proposal_recall": dense._proposal_recall(training),
        "winner": cap._serializable_method(winner),
        "method_search": [cap._serializable_method(method) for method in methods],
        "pitch_oof": pitch_cv,
        "validation_request_sha256": dense._sha256(validation_requests_path),
    }
    training_snapshot_path.write_text(
        json.dumps(training_snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    validation_composed = [
        dense._compose_with_meter_fallback(
            request,
            measure,
            model,
            out_dir=out_dir,
            selection_mode=dense.SELECTION_MODE,
            selector_method_id=method_id,
            pitch_predictor=pitch_predictor,
        )
        for request, measure in zip(system8_requests, validation_unlabeled, strict=True)
    ]
    artifacts = composed.freeze_split_predictions(
        split="validation",
        composed=validation_composed,
        output_dir=output_dir,
        requests_path=validation_requests_path,
        training=[example.measure for example in training],
        selector_mode="s1_s7_review_oof_dense_patch_template",
        selector_method_id=method_id,
    )
    validation_metrics = composed.evaluate_frozen_split(benchmark_dir, artifacts=artifacts)
    historical = dense._historical_system8_metrics(benchmark_dir)
    gate = dense._validation_gate(validation_metrics["summary"], historical)

    report_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    report = {
        "schema_version": 1,
        "kind": "dense_patch_review_augmented_selector_spike",
        "slug": slug,
        "protocol": {
            "training": "promoted S1M1-4 and S7M1-7 reviews only",
            "model_selection": "four dense patch-template methods plus OOF threshold/count/NMS",
            "validation": "freeze S8M1-7 before evaluation",
            "heldout": "not accessed; system 3 is consumed",
        },
        "training": {
            "proposal_recall": dense._proposal_recall(training),
            "winner": cap._serializable_method(winner),
            "method_search": [cap._serializable_method(method) for method in methods],
            "pitch_oof": pitch_cv,
        },
        "selector": {"method_id": method_id, "pitch_method": pitch_method},
        "leakage_audit": {
            "validation_truth_accessed_after_freeze": True,
            "prediction_freeze_sha256": artifacts.freeze_sha256,
            "system3_accessed": False,
            "interpretation": "System 8 is model-selection evidence, not fresh heldout.",
        },
        "validation": {
            "metrics": validation_metrics,
            "historical_threshold_selector_same_targets": historical,
            "per_measure_counts": composed._per_measure_counts(validation_composed),
            "artifacts": composed._artifact_summary(artifacts),
        },
        "validation_gate": gate,
        "artifacts": {
            "report_json": dense._display_path(report_path),
            "report_markdown": dense._display_path(markdown_path),
            "training_selection": dense._display_path(training_snapshot_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report, markdown_path)
    return report


def _write_markdown(report: Mapping[str, Any], path: Path) -> None:
    winner = report["training"]["winner"]
    summary = report["validation"]["metrics"]["summary"]
    historical = report["validation"]["historical_threshold_selector_same_targets"]
    lines = [
        "# Dense Patch Review-Augmented Selector",
        "",
        f"- Training proposal recall: {report['training']['proposal_recall']['recall']:.3f}",
        f"- OOF winner: {winner['method_id']}",
        f"- OOF candidate F1: {winner['selection']['metrics']['f1']:.3f}",
        f"- Pitch method: {report['selector']['pitch_method']}",
        f"- Frozen system-8 note F1: {summary['note_f1']:.3f}",
        f"- Frozen system-8 ordered pitch: {summary['ordered_pitch_accuracy']:.3f}",
        f"- Historical same-target note F1: {historical['summary']['note_f1']:.3f}",
        f"- Gate: {report['validation_gate']['status']}",
        "",
        "System 8 is model-selection evidence, not a fresh heldout claim. System 3 was not read.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
