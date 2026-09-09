"""Freeze a truth-blind second-score melody prediction.

This is the fresh-score gate for the VLM melody spike. It trains the selected
S1+S7 Aviador notehead/meter-gap configuration from promoted reviews, applies
it to a preselected system from a different score, and freezes every inference
artifact before canonical MusicXML exists.

The command intentionally has no ground-truth, MusicXML, or evaluation input.
An existing freeze is immutable and must not be overwritten.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_vlm_melody_event_benchmark as benchmark  # noqa: E402
from scripts.experiments import spike_cap24_review_augmented_selector as cap  # noqa: E402
from scripts.experiments import spike_composed_melody_chain as composed  # noqa: E402
from scripts.experiments import spike_meter_gap_resolver as meter_gap  # noqa: E402
from scripts.experiments import spike_notehead_patch_templates as patches  # noqa: E402
from scripts.experiments import spike_review_augmented_selector as dense  # noqa: E402

TRAINING_SLUG = "jaime-llanos_12_aviador_pasillo_fulgencio-garcia"
TARGET_SLUG = "jaime-llanos_19_carrizal_pasillo_emilio-murillo"
TARGET_SYSTEM = 4
TARGET_MEASURES = tuple(range(1, 8))
TARGET_CLEF = "treble"
TARGET_TIME_SIGNATURE = "3/4"
TARGET_KEY_HINT = "one flat: Bb"
SPLIT_NAME = "fresh_heldout"
OUTPUT_SUBDIR = "vlm_melody_fresh_heldout"
METHOD_PREFIX = "frozen_s1_s7_meter_gap"


@dataclass(frozen=True)
class FrozenModel:
    selector: meter_gap.GapAwareSelectorModel
    pitch_predictor: composed.PitchPredictor
    method_id: str
    training: tuple[dense.TrainingExample, ...]
    selection_payload: dict[str, Any]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", type=Path, default=Path("out"))
    parser.add_argument(
        "--reviews-dir",
        type=Path,
        default=REPO_ROOT / "tests/fixtures/vlm_melody/notehead_reviews",
    )
    args = parser.parse_args(argv)
    try:
        report = freeze_fresh_heldout(args.out_dir, reviews_dir=args.reviews_dir)
    except (FileNotFoundError, KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(report["artifacts"]["selection"])
    print(report["artifacts"]["predictions"])
    print(report["artifacts"]["freeze"])
    return 0


def freeze_fresh_heldout(
    out_dir: Path,
    *,
    reviews_dir: Path,
    training_slug: str = TRAINING_SLUG,
    target_slug: str = TARGET_SLUG,
    target_system: int = TARGET_SYSTEM,
    target_measures: Sequence[int] = TARGET_MEASURES,
    clef: str = TARGET_CLEF,
    time_signature: str = TARGET_TIME_SIGNATURE,
    key_hint: str | None = TARGET_KEY_HINT,
) -> dict[str, Any]:
    output_dir = out_dir / target_slug / OUTPUT_SUBDIR / f"system_{target_system:03d}"
    freeze_path = output_dir / SPLIT_NAME / "freeze.json"
    if freeze_path.exists():
        raise ValueError(
            f"Fresh heldout is already frozen and cannot be overwritten: {freeze_path}"
        )

    requests = prepare_target_requests(
        out_dir,
        slug=target_slug,
        system_index=target_system,
        measures=target_measures,
        clef=clef,
        time_signature=time_signature,
        key_hint=key_hint,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    requests_path = output_dir / "requests.jsonl"
    benchmark._write_jsonl(requests_path, requests)

    selection_path = output_dir / "selection.json"
    selection_payload = {
        "schema_version": 1,
        "status": "selected_before_prediction_and_before_truth",
        "truth_status": "not_created",
        "target": {
            "slug": target_slug,
            "system_index": target_system,
            "system_measure_indices": list(target_measures),
            "manifest_global_measure_indices": [
                int(row["mapping"]["manifest_global_measure_index"]) for row in requests
            ],
        },
        "selection_basis": [
            "different score and composer from selector training",
            "interior complete system with seven visually clean physical measures",
            "measure crops inspected for segmentation only",
            "no canonical events or MusicXML available during selection",
        ],
        "allowed_context": {
            "clef": clef,
            "time_signature": time_signature,
            "key_hint": key_hint,
            "source": "printed score header only",
        },
        "requests": {
            "path": composed._display_path(requests_path),
            "sha256": composed._sha256(requests_path),
        },
    }
    _write_json(selection_path, selection_payload)

    model = fit_frozen_model(
        out_dir,
        training_slug=training_slug,
        reviews_dir=reviews_dir,
    )
    model_path = output_dir / "model_selection.json"
    _write_json(model_path, model.selection_payload)

    target_unlabeled = [
        dense._prepare_dense_measure(request, out_dir=out_dir) for request in requests
    ]
    target_predictions = [
        meter_gap._compose_with_meter_gap_resolver(
            request,
            measure,
            model.selector,
            out_dir=out_dir,
            selector_method_id=model.method_id,
            pitch_predictor=model.pitch_predictor,
        )
        for request, measure in zip(requests, target_unlabeled, strict=True)
    ]
    artifacts = composed.freeze_split_predictions(
        split=SPLIT_NAME,
        composed=target_predictions,
        output_dir=output_dir,
        requests_path=requests_path,
        training=[example.measure for example in model.training],
        selector_mode="frozen_s1_s7_review_oof_meter_gap_resolver",
        selector_method_id=model.method_id,
        training_review_fields=(
            "final_noteheads[].center",
            "final_noteheads[].pitch",
            "final_noteheads[].source.kind",
        ),
    )

    report_path = output_dir / "sealed_manifest.json"
    report = {
        "schema_version": 1,
        "kind": "fresh_second_score_heldout_freeze",
        "status": "frozen_awaiting_canonical_musicxml",
        "truth_accessed": False,
        "selection_sha256": composed._sha256(selection_path),
        "model_selection_sha256": composed._sha256(model_path),
        "prediction_freeze_sha256": artifacts.freeze_sha256,
        "target": selection_payload["target"],
        "artifacts": {
            "selection": composed._display_path(selection_path),
            "model_selection": composed._display_path(model_path),
            "requests": composed._display_path(requests_path),
            "predictions": composed._display_path(artifacts.prediction_path),
            "inference": composed._display_path(artifacts.inference_path),
            "freeze": composed._display_path(artifacts.freeze_path),
            "overlays": [composed._display_path(path) for path in artifacts.overlay_paths],
            "sealed_manifest": composed._display_path(report_path),
        },
    }
    _write_json(report_path, report)
    return report


def prepare_target_requests(
    out_dir: Path,
    *,
    slug: str,
    system_index: int,
    measures: Sequence[int],
    clef: str,
    time_signature: str,
    key_hint: str | None,
) -> list[dict[str, Any]]:
    requested = tuple(int(measure) for measure in measures)
    if not requested or any(measure <= 0 for measure in requested):
        raise ValueError("Target measures must be positive and non-empty")
    if len(set(requested)) != len(requested):
        raise ValueError("Target measures must be unique")

    records = benchmark._load_measure_records(out_dir, slug)
    selected = {
        int(row["system_measure_index"]): row
        for row in records
        if int(row["system_index"]) == system_index
        and int(row["system_measure_index"]) in requested
    }
    missing = [measure for measure in requested if measure not in selected]
    if missing:
        raise ValueError(
            f"Missing target melody-input records for system {system_index}: {missing}"
        )
    targets = tuple(
        benchmark.BenchmarkTarget(
            system_index=system_index,
            system_measure_index=measure,
            global_measure_index=int(selected[measure]["global_measure_index"]),
            manifest_global_measure_index=int(selected[measure]["global_measure_index"]),
        )
        for measure in requested
    )
    return benchmark.prepare_requests(
        out_dir,
        slug=slug,
        targets=targets,
        split_name=SPLIT_NAME,
        clef=clef,
        time_signature=time_signature,
        key_hint=key_hint,
    )


def fit_frozen_model(
    out_dir: Path,
    *,
    training_slug: str,
    reviews_dir: Path,
) -> FrozenModel:
    benchmark_dir = out_dir / training_slug / "vlm_melody_event_benchmark"
    development_path = benchmark_dir / "development/requests.jsonl"
    validation_path = benchmark_dir / "validation/requests.jsonl"
    training_requests = dense._select_requests(
        [*dense._read_jsonl(development_path), *dense._read_jsonl(validation_path)],
        dense.TRAINING_TARGETS,
    )

    training_unlabeled = [
        dense._prepare_dense_measure(request, out_dir=out_dir) for request in training_requests
    ]
    training = tuple(
        dense._attach_review(
            request,
            measure,
            reviews_dir=reviews_dir,
            slug=training_slug,
        )
        for request, measure in zip(training_requests, training_unlabeled, strict=True)
    )

    methods = []
    for patch_spec in patches.PATCH_SPECS:
        oof_scores = cap._out_of_fold_scores(
            training,
            patch_id=patch_spec.id,
            scorer_kind="class_template",
        )
        selection = dense._select_training_configuration(training, oof_scores)
        recovery = meter_gap._select_recovery_configuration(
            training,
            oof_scores,
            selection=selection,
        )
        methods.append(
            {
                "method_id": f"class_template__{patch_spec.id}",
                "patch_id": patch_spec.id,
                "scorer_kind": "class_template",
                "selection": selection,
                "recovery": recovery,
            }
        )
    winner = max(
        methods,
        key=lambda method: (
            float(method["recovery"].metrics["f1"]),
            float(method["recovery"].metrics["recall"]),
            float(method["recovery"].metrics["precision"]),
            int(method["recovery"].metrics["exact_measures"]),
            str(method["method_id"]),
        ),
    )
    full_scorer = patches._fit_patch_scorer(
        [row for example in training for row in example.measure.rows],
        patch_id=str(winner["patch_id"]),
        scorer_kind=str(winner["scorer_kind"]),
    )
    selection = winner["selection"]
    base_model = dense.DenseSelectorModel(
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
    selector = meter_gap.GapAwareSelectorModel(
        base=base_model,
        recovery=winner["recovery"],
    )
    pitch_cv = dense._evaluate_pitch_methods(training)
    pitch_method = str(pitch_cv["selected_method"])
    accidental_model = (
        dense._fit_accidental_model(training) if pitch_method == "accidental_knn" else None
    )
    pitch_predictor = dense._build_pitch_predictor(accidental_model)
    method_id = f"{METHOD_PREFIX}__{winner['method_id']}"
    selection_payload = {
        "schema_version": 1,
        "status": "fit_from_promoted_training_reviews_before_target_prediction",
        "training_slug": training_slug,
        "training_targets": [example.key for example in training],
        "training_review_hashes": [example.measure.review_sha256 for example in training],
        "canonical_training_rhythm_used": False,
        "winner": meter_gap._serializable_method(winner),
        "pitch_oof": pitch_cv,
        "selector": {"method_id": method_id, "pitch_method": pitch_method},
    }
    return FrozenModel(
        selector=selector,
        pitch_predictor=pitch_predictor,
        method_id=method_id,
        training=training,
        selection_payload=selection_payload,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
