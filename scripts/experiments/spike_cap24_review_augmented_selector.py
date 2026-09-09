"""Retrain the cap-24 patch selector on promoted S1+S7 reviews.

This companion to ``spike_review_augmented_selector.py`` tests whether the
original patch representation benefits from the seven new reviewed measures.
Patch/scorer/NMS/count choices are selected only from measure-level out-of-fold
training predictions. System 8 is frozen before evaluation, and consumed
system 3 is never accessed.

Example:
    uv run python scripts/experiments/spike_cap24_review_augmented_selector.py out
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import spike_composed_melody_chain as composed  # noqa: E402
from scripts.experiments import spike_notehead_patch_templates as patches  # noqa: E402
from scripts.experiments import spike_review_augmented_selector as dense  # noqa: E402

DEFAULT_OUT_DIR = REPO_ROOT / "out"
DEFAULT_SLUG = patches.DEFAULT_SLUG
DEFAULT_REVIEWS_DIR = REPO_ROOT / "tests/fixtures/vlm_melody/notehead_reviews"
OUTPUT_SUBDIR = "cap24_review_augmented_selector"
METHOD_PREFIX = "cap24_s1_s7"


@dataclass(frozen=True)
class CapTrainingExample:
    key: str
    request: dict[str, Any]
    measure: patches.MeasureData
    matched_candidate_ids: frozenset[str]
    true_note_count: int
    unmatched_note_ids: tuple[str, ...]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--reviews-dir", type=Path, default=DEFAULT_REVIEWS_DIR)
    parser.add_argument("--output-dir", type=Path)
    return parser


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

    # Regenerate every cap-24 patch set before opening review fixtures.
    training_unlabeled = [
        _prepare_cap24(request, out_dir=out_dir, slug=slug) for request in training_requests
    ]
    validation_unlabeled = [
        _prepare_cap24(request, out_dir=out_dir, slug=slug) for request in system8_requests
    ]
    training = [
        _attach_review(
            request,
            measure,
            reviews_dir=reviews_dir,
            slug=slug,
        )
        for request, measure in zip(training_requests, training_unlabeled, strict=True)
    ]

    methods = []
    for patch_spec in patches.PATCH_SPECS:
        for scorer_kind in patches.SCORER_KINDS:
            oof_scores = _out_of_fold_scores(
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

    training_snapshot_path = output_dir / "training_selection.json"
    training_snapshot = {
        "status": "selected_before_validation_prediction",
        "training_targets": [example.key for example in training],
        "review_hashes": [example.measure.review_sha256 for example in training],
        "proposal_recall": _proposal_recall(training),
        "winner": _serializable_method(winner),
        "method_search": [_serializable_method(method) for method in methods],
        "validation_request_sha256": _sha256(validation_requests_path),
        "validation_targets": [
            dense._identity_key(request["identity"]) for request in system8_requests
        ],
    }
    training_snapshot_path.write_text(
        json.dumps(training_snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    pitch_predictor = dense._build_pitch_predictor(None)
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
        selector_mode="s1_s7_review_oof_cap24_patch_threshold",
        selector_method_id=method_id,
    )
    validation_metrics = composed.evaluate_frozen_split(benchmark_dir, artifacts=artifacts)
    historical = dense._historical_system8_metrics(benchmark_dir)
    validation_gate = dense._validation_gate(validation_metrics["summary"], historical)

    report_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    report = {
        "schema_version": 1,
        "kind": "cap24_review_augmented_selector_spike",
        "slug": slug,
        "protocol": {
            "training": "promoted S1M1-4 and S7M1-7 reviews only",
            "model_selection": "eight patch/scorer methods plus NMS/count/threshold on OOF labels",
            "validation": "freeze S8M1-7 before evaluation",
            "heldout": "not accessed; system 3 is consumed",
        },
        "training": {
            "targets": [example.key for example in training],
            "proposal_recall": _proposal_recall(training),
            "winner": _serializable_method(winner),
            "method_search": [_serializable_method(method) for method in methods],
        },
        "selector": {
            "method_id": method_id,
            "pitch_method": "key_signature_only",
        },
        "leakage_audit": {
            "training_review_fields": ["candidates[].label", "final_noteheads count"],
            "validation_expected_counts_used": False,
            "validation_truth_accessed_after_freeze": True,
            "prediction_freeze_sha256": artifacts.freeze_sha256,
            "system3_accessed": False,
            "interpretation": (
                "System 8 is model-selection evidence and was opened by prior arms; this is "
                "not a fresh heldout claim."
            ),
        },
        "validation": {
            "metrics": validation_metrics,
            "historical_threshold_selector_same_targets": historical,
            "per_measure_counts": composed._per_measure_counts(validation_composed),
            "artifacts": composed._artifact_summary(artifacts),
        },
        "validation_gate": validation_gate,
        "artifacts": {
            "report_json": dense._display_path(report_path),
            "report_markdown": dense._display_path(markdown_path),
            "training_selection": dense._display_path(training_snapshot_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report, markdown_path)
    return report


def _prepare_cap24(
    request: Mapping[str, Any],
    *,
    out_dir: Path,
    slug: str,
) -> patches.UnlabeledMeasure:
    identity = request["identity"]
    measure = patches._load_unlabeled_measure(
        out_dir,
        slug=slug,
        system_index=int(identity["system_index"]),
        measure=int(identity["system_measure_index"]),
        max_candidates=patches.DEFAULT_MAX_CANDIDATES,
    )
    composed._validate_unlabeled_identity(request, measure)
    return measure


def _attach_review(
    request: Mapping[str, Any],
    measure: patches.UnlabeledMeasure,
    *,
    reviews_dir: Path,
    slug: str,
) -> CapTrainingExample:
    identity = request["identity"]
    system_index = int(identity["system_index"])
    measure_index = int(identity["system_measure_index"])
    key = dense._target_key(system_index, measure_index)
    review_path = reviews_dir / (
        f"{slug}_system_{system_index:03d}_measure_{measure_index:03d}.json"
    )
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    review_identity = payload.get("identity", {})
    if (
        int(review_identity.get("system_index", -1)) != system_index
        or int(review_identity.get("system_measure_index", -1)) != measure_index
    ):
        raise ValueError(f"Review identity mismatch: {review_path}")
    source = payload.get("source", {})
    if source.get("image_sha256") != measure.source_sha256:
        raise ValueError(f"Review image hash mismatch: {review_path}")
    if int(source.get("candidate_cap", -1)) != len(measure.candidates):
        raise ValueError(f"Review candidate cap mismatch: {review_path}")
    decisions = payload.get("candidates")
    final_noteheads = payload.get("final_noteheads")
    if not isinstance(decisions, list) or not isinstance(final_noteheads, list):
        raise ValueError(f"Incomplete review fixture: {review_path}")
    by_id = {str(item["id"]): item for item in decisions}
    expected_ids = {candidate.id for candidate in measure.candidates}
    if set(by_id) != expected_ids:
        raise ValueError(f"Review candidate IDs mismatch: {review_path}")
    rows = []
    accepted_ids = set()
    for candidate in measure.candidates:
        decision = by_id[candidate.id]
        label = str(decision.get("label"))
        if label not in ("accepted", "rejected"):
            raise ValueError(f"Invalid candidate label {label!r}: {review_path}")
        accepted = label == "accepted"
        if accepted:
            accepted_ids.add(candidate.id)
        rows.append(patches.LabeledCandidate(candidate=candidate, label=int(accepted)))
    labeled = patches.MeasureData(
        measure=measure.measure,
        source_image=measure.source_image,
        source_sha256=measure.source_sha256,
        review_path=review_path,
        review_sha256=_sha256(review_path),
        staff_lines=measure.staff_lines,
        staff_spacing=measure.staff_spacing,
        rows=tuple(rows),
    )
    unmatched = tuple(
        f"n{index:03d}"
        for index, notehead in enumerate(final_noteheads, start=1)
        if str(notehead.get("source", {}).get("kind")) == "manual"
    )
    if len(accepted_ids) + len(unmatched) != len(final_noteheads):
        raise ValueError(f"Candidate/manual notehead count mismatch: {review_path}")
    return CapTrainingExample(
        key=key,
        request=dict(request),
        measure=labeled,
        matched_candidate_ids=frozenset(accepted_ids),
        true_note_count=len(final_noteheads),
        unmatched_note_ids=unmatched,
    )


def _out_of_fold_scores(
    examples: Sequence[CapTrainingExample],
    *,
    patch_id: str,
    scorer_kind: str,
) -> dict[tuple[str, str], float]:
    scores = {}
    for heldout in examples:
        training_rows = [
            row
            for example in examples
            if example.key != heldout.key
            for row in example.measure.rows
        ]
        scorer = patches._fit_patch_scorer(
            training_rows,
            patch_id=patch_id,
            scorer_kind=scorer_kind,
        )
        for row in heldout.measure.rows:
            scores[(heldout.key, row.id)] = scorer.score(row.candidate)
    return scores


def _proposal_recall(examples: Sequence[CapTrainingExample]) -> dict[str, Any]:
    matched = sum(len(example.matched_candidate_ids) for example in examples)
    truth = sum(example.true_note_count for example in examples)
    return {"matched": matched, "truth": truth, "recall": dense._ratio(matched, truth)}


def _serializable_method(method: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method_id": method["method_id"],
        "patch_id": method["patch_id"],
        "scorer_kind": method["scorer_kind"],
        "selection": method["selection"],
    }


def _write_markdown(report: Mapping[str, Any], path: Path) -> None:
    winner = report["training"]["winner"]
    summary = report["validation"]["metrics"]["summary"]
    historical = report["validation"]["historical_threshold_selector_same_targets"]
    lines = [
        "# Cap-24 Review-Augmented Selector",
        "",
        f"- Training proposal recall: {report['training']['proposal_recall']['recall']:.3f}",
        f"- OOF winner: {winner['method_id']}",
        f"- OOF candidate F1: {winner['selection']['metrics']['f1']:.3f}",
        f"- Frozen system-8 note F1: {summary['note_f1']:.3f}",
        f"- Frozen system-8 ordered pitch: {summary['ordered_pitch_accuracy']:.3f}",
        f"- Historical same-target note F1: {historical['summary']['note_f1']:.3f}",
        f"- Gate: {report['validation_gate']['status']}",
        "",
        "System 8 is model-selection evidence, not a fresh heldout claim. System 3 was not read.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
