"""Bounded truth-blind onset-group precision experiment.

This spike filters only the x-clusters already admitted by the frozen notehead
selector.  It calibrates on consumed Aviador and Carrizal reviews, reports
work-disjoint folds, and never uses truth during feature construction or
prediction.  La Chata can be scored after predictions are materialized; its
current frozen truth has onset divisions but no pixel x-coordinates, so that
held-out report is count-only.

Example:
    uv run python scripts/experiments/spike_consumed_onset_group_selector.py \
        --out-dir out --overlays --output-dir /tmp/onset-group-selector
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import run_third_score_heldout_inference as heldout  # noqa: E402
from scripts.experiments import spike_consumed_polyphonic_pitch_repair as recovery  # noqa: E402
from scripts.experiments import spike_cross_score_consumed_retraining as cross_score  # noqa: E402
from scripts.experiments import spike_notehead_patch_templates as patches  # noqa: E402
from scripts.experiments import spike_review_augmented_selector as dense  # noqa: E402

SCHEMA_VERSION = 1
OUTPUT_VERSION = "consumed_onset_group_selector_v1"
DEFAULT_OUT_DIR = REPO_ROOT / "out"
DEFAULT_MODEL = (
    DEFAULT_OUT_DIR
    / "vlm_melody_consumed_training"
    / "cross_score_notehead_v1_replay_v2"
    / "model.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_OUT_DIR / "vlm_melody_consumed_training" / OUTPUT_VERSION
DEFAULT_LA_CHATA_INFERENCE = (
    DEFAULT_OUT_DIR
    / "jaime-llanos_64_la-chata_pasillo_luis-a-calvo"
    / "vlm_melody_third_score_heldout"
    / "v2"
    / "system_007"
    / "inference_v2"
    / "inference.jsonl"
)
DEFAULT_LA_CHATA_REQUESTS = DEFAULT_LA_CHATA_INFERENCE.parent / "requests.jsonl"
DEFAULT_LA_CHATA_TRUTH = (
    DEFAULT_OUT_DIR
    / "jaime-llanos_64_la-chata_pasillo_luis-a-calvo"
    / "vlm_melody_third_score_heldout"
    / "v2"
    / "system_007"
    / "evaluation_v1"
    / "truth.jsonl"
)
EPSILON = 1e-9

# These are all observable from an inference row or its image-derived
# candidate patches.  No feature encodes a target count, truth pitch, or work.
GROUP_FEATURES = (
    "max_score",
    "mean_score",
    "score_margin",
    "max_stem_score",
    "mean_stem_score",
    "cluster_size",
    "vertical_spread_spaces",
    "left_gap_spaces",
    "right_gap_spaces",
    "nearest_neighbor_gap_spaces",
    "x_fraction",
    "dense_ink_density_mean",
    "dense_ink_density_max",
    "dense_core_density_mean",
    "dense_core_density_max",
    "dense_line_dominance_mean",
    "dense_line_dominance_max",
    "dense_stem_evidence_mean",
    "dense_stem_evidence_max",
    "dense_patch_center_density_mean",
    "dense_patch_center_density_max",
)


@dataclass(frozen=True)
class GroupObservation:
    """One baseline x-cluster with truth-free features."""

    group_id: str
    example_key: str
    work_id: str
    x_center: float
    candidate_ids: tuple[str, ...]
    features: tuple[float, ...]

    def feature_map(self) -> dict[str, float]:
        return dict(zip(GROUP_FEATURES, self.features, strict=True))


@dataclass(frozen=True)
class LabeledGroup:
    observation: GroupObservation
    label: int


@dataclass(frozen=True)
class GroupFilterModel:
    feature_names: tuple[str, ...]
    midpoint: tuple[float, ...]
    scales: tuple[float, ...]
    weights: tuple[float, ...]
    threshold: float
    training_group_count: int
    training_positive_count: int

    def score(self, observation: GroupObservation) -> float:
        if tuple(self.feature_names) != GROUP_FEATURES:
            raise ValueError("Group filter feature order drift")
        normalized = [
            (value - midpoint) / scale
            for value, midpoint, scale in zip(
                observation.features, self.midpoint, self.scales, strict=True
            )
        ]
        return sum(weight * value for weight, value in zip(self.weights, normalized, strict=True))


@dataclass(frozen=True)
class ExampleReplay:
    key: str
    work_id: str
    system_index: int
    measure_index: int
    inference_row: dict[str, Any]
    selector: dict[str, Any]
    baseline_groups: tuple[dict[str, Any], ...]
    observations: tuple[GroupObservation, ...]
    truth_centers: tuple[float, ...]
    x_radius_px: float
    source_image: Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.resolve().read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{label} row {line_number} must be an object: {path}")
        rows.append(value)
    if not rows:
        raise ValueError(f"{label} JSONL is empty: {path}")
    return rows


def _display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _file_pin(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": _display_path(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


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


def _identity(row: Mapping[str, Any]) -> tuple[str, int, int]:
    identity = row.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("Inference row has no identity")
    slug = str(identity.get("slug", ""))
    system = int(identity.get("system_index", 0))
    measure = int(identity.get("system_measure_index", identity.get("automatic_measure_index", 0)))
    if not slug or system <= 0 or measure <= 0:
        raise ValueError("Inference row identity is incomplete")
    return slug, system, measure


def _candidate_id(candidate: Mapping[str, Any]) -> str:
    value = candidate.get("candidate_id", candidate.get("id"))
    if value is None:
        raise ValueError("Candidate has no candidate_id")
    return str(value)


def _candidate_center(candidate: Mapping[str, Any]) -> tuple[float, float]:
    center = candidate.get("center")
    if not isinstance(center, Mapping):
        raise ValueError(f"Candidate has no center: {_candidate_id(candidate)}")
    return float(center["x"]), float(center["y"])


def _candidate_score(candidate: Mapping[str, Any]) -> float:
    value = candidate.get("score", candidate.get("probability"))
    if value is None:
        raise ValueError(f"Candidate has no score: {_candidate_id(candidate)}")
    return float(value)


def cluster_selected_candidates(
    selected: Sequence[Mapping[str, Any]], x_radius_px: float
) -> list[dict[str, Any]]:
    """Cluster already-selected candidates by x, without using labels."""
    if x_radius_px <= 0 or not math.isfinite(x_radius_px):
        raise ValueError("x_radius_px must be finite and positive")
    ordered = sorted(
        selected,
        key=lambda candidate: (
            _candidate_center(candidate)[0],
            _candidate_center(candidate)[1],
            _candidate_id(candidate),
        ),
    )
    groups: list[list[Mapping[str, Any]]] = []
    for candidate in ordered:
        if not groups:
            groups.append([candidate])
            continue
        previous_x = _candidate_center(groups[-1][-1])[0]
        current_x = _candidate_center(candidate)[0]
        if current_x - previous_x < x_radius_px:
            groups[-1].append(candidate)
        else:
            groups.append([candidate])
    result = []
    for index, group in enumerate(groups, start=1):
        xs = [_candidate_center(item)[0] for item in group]
        ys = [_candidate_center(item)[1] for item in group]
        result.append(
            {
                "group_id": f"g{index:03d}",
                "candidate_ids": tuple(_candidate_id(item) for item in group),
                "candidates": tuple(group),
                "x_center": sum(xs) / len(xs),
                "y_min": min(ys),
                "y_max": max(ys),
            }
        )
    return result


def _candidate_dense_features(candidate: Mapping[str, Any]) -> Mapping[str, float]:
    value = candidate.get("dense_features", {})
    return value if isinstance(value, Mapping) else {}


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _group_features(
    group: Mapping[str, Any],
    *,
    candidate_features: Mapping[str, Mapping[str, Any]],
    spacing: float,
    left_x: float | None,
    right_x: float | None,
    image_width: float,
) -> tuple[float, ...]:
    candidates = group["candidates"]
    details = [candidate_features.get(_candidate_id(candidate), {}) for candidate in candidates]
    scores = [_candidate_score(candidate) for candidate in candidates]
    stem_scores = [float(detail.get("stem_score", 0.0)) for detail in details]
    dense_values: dict[str, list[float]] = defaultdict(list)
    for detail in details:
        for name in (
            "ink_density",
            "core_density",
            "line_dominance",
            "stem_evidence",
            "patch_center_density",
        ):
            value = detail.get("dense_features", {}).get(name, 0.0)
            dense_values[name].append(float(value))
    ordered_scores = sorted(scores, reverse=True)
    score_margin = (
        ordered_scores[0] - ordered_scores[1] if len(ordered_scores) > 1 else ordered_scores[0]
    )
    vertical_spread = (float(group["y_max"]) - float(group["y_min"])) / spacing
    left_gap = (float(group["x_center"]) - left_x) / spacing if left_x is not None else 99.0
    right_gap = (right_x - float(group["x_center"])) / spacing if right_x is not None else 99.0
    dense_mean = {name: _mean(values) for name, values in dense_values.items()}
    dense_max = {name: max(values) if values else 0.0 for name, values in dense_values.items()}
    values = {
        "max_score": max(scores),
        "mean_score": _mean(scores),
        "score_margin": score_margin,
        "max_stem_score": max(stem_scores) if stem_scores else 0.0,
        "mean_stem_score": _mean(stem_scores),
        "cluster_size": float(len(candidates)),
        "vertical_spread_spaces": vertical_spread,
        "left_gap_spaces": left_gap,
        "right_gap_spaces": right_gap,
        "nearest_neighbor_gap_spaces": min(left_gap, right_gap),
        "x_fraction": _ratio(float(group["x_center"]), max(1.0, image_width)),
        "dense_ink_density_mean": dense_mean["ink_density"],
        "dense_ink_density_max": dense_max["ink_density"],
        "dense_core_density_mean": dense_mean["core_density"],
        "dense_core_density_max": dense_max["core_density"],
        "dense_line_dominance_mean": dense_mean["line_dominance"],
        "dense_line_dominance_max": dense_max["line_dominance"],
        "dense_stem_evidence_mean": dense_mean["stem_evidence"],
        "dense_stem_evidence_max": dense_max["stem_evidence"],
        "dense_patch_center_density_mean": dense_mean["patch_center_density"],
        "dense_patch_center_density_max": dense_max["patch_center_density"],
    }
    return tuple(float(values[name]) for name in GROUP_FEATURES)


def build_group_observations(
    selected: Sequence[Mapping[str, Any]],
    *,
    x_radius_px: float,
    spacing: float,
    candidate_features: Mapping[str, Mapping[str, Any]] | None = None,
    image_width: float = 1.0,
    example_key: str = "synthetic",
    work_id: str = "synthetic",
) -> tuple[GroupObservation, ...]:
    """Create features from a selected baseline, with no truth parameter."""
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    feature_map = candidate_features or {}
    groups = cluster_selected_candidates(selected, x_radius_px)
    observations = []
    for index, group in enumerate(groups):
        left_x = float(groups[index - 1]["x_center"]) if index else None
        right_x = float(groups[index + 1]["x_center"]) if index + 1 < len(groups) else None
        observations.append(
            GroupObservation(
                group_id=str(group["group_id"]),
                example_key=example_key,
                work_id=work_id,
                x_center=float(group["x_center"]),
                candidate_ids=tuple(str(value) for value in group["candidate_ids"]),
                features=_group_features(
                    group,
                    candidate_features=feature_map,
                    spacing=spacing,
                    left_x=left_x,
                    right_x=right_x,
                    image_width=image_width,
                ),
            )
        )
    return tuple(observations)


def _cluster_x_centers(xs: Sequence[float], x_radius_px: float) -> tuple[float, ...]:
    if not xs:
        return ()
    ordered = sorted(float(value) for value in xs)
    clusters: list[list[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        if value - clusters[-1][-1] < x_radius_px:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return tuple(_mean(cluster) for cluster in clusters)


def truth_group_centers(note_xs: Sequence[float], x_radius_px: float) -> tuple[float, ...]:
    """Derive truth onset locations only for post-prediction scoring/calibration."""
    return _cluster_x_centers(note_xs, x_radius_px)


def match_group_centers(
    predicted_xs: Sequence[float], truth_xs: Sequence[float], tolerance_px: float
) -> list[tuple[int, int]]:
    """Deterministically match predicted and truth x-clusters one-to-one."""
    unmatched = set(range(len(truth_xs)))
    matches = []
    for predicted_index, predicted_x in sorted(enumerate(predicted_xs), key=lambda item: item[1]):
        options = [
            (abs(float(predicted_x) - float(truth_xs[index])), index)
            for index in unmatched
            if abs(float(predicted_x) - float(truth_xs[index])) <= tolerance_px
        ]
        if not options:
            continue
        _, truth_index = min(options, key=lambda item: (item[0], item[1]))
        unmatched.remove(truth_index)
        matches.append((predicted_index, truth_index))
    return matches


def group_metrics(
    predicted_xs: Sequence[float], truth_xs: Sequence[float], tolerance_px: float
) -> dict[str, Any]:
    matches = match_group_centers(predicted_xs, truth_xs, tolerance_px)
    tp = len(matches)
    fp = len(predicted_xs) - tp
    fn = len(truth_xs) - tp
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "predicted_group_count": len(predicted_xs),
        "truth_group_count": len(truth_xs),
        "matched_pairs": [[int(left), int(right)] for left, right in matches],
    }


def _standard_scale(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 1.0
    scale = statistics.pstdev(values)
    return scale if math.isfinite(scale) and scale > EPSILON else 1.0


def _score_threshold(scores: Sequence[float], labels: Sequence[int]) -> float:
    positive = [score for score, label in zip(scores, labels, strict=True) if label]
    if not positive:
        raise ValueError("Group filter training needs at least one positive group")
    candidates = sorted(
        {score for score in scores if score <= min(positive) + EPSILON}, reverse=True
    )
    return float(candidates[0] if candidates else min(positive))


def fit_group_filter(groups: Sequence[LabeledGroup]) -> GroupFilterModel:
    """Fit a deterministic standardized centroid scorer and recall-first threshold."""
    if not groups:
        raise ValueError("Cannot fit an empty group filter")
    labels = [int(group.label) for group in groups]
    if any(label not in {0, 1} for label in labels):
        raise ValueError("Group labels must be 0 or 1")
    if not any(labels):
        raise ValueError("Group filter training needs at least one positive group")
    if all(labels):
        # A work-disjoint fold with no observed false groups cannot justify a
        # rejection boundary. Keep every group and make that limitation visible
        # through the zero-weight model rather than fabricating negatives.
        return GroupFilterModel(
            feature_names=GROUP_FEATURES,
            midpoint=tuple(0.0 for _ in GROUP_FEATURES),
            scales=tuple(1.0 for _ in GROUP_FEATURES),
            weights=tuple(0.0 for _ in GROUP_FEATURES),
            threshold=0.0,
            training_group_count=len(groups),
            training_positive_count=len(groups),
        )
    positives = [group.observation.features for group in groups if group.label]
    negatives = [group.observation.features for group in groups if not group.label]
    midpoint = []
    scales = []
    weights = []
    for index in range(len(GROUP_FEATURES)):
        pos_values = [row[index] for row in positives]
        neg_values = [row[index] for row in negatives]
        pos_mean = _mean(pos_values)
        neg_mean = _mean(neg_values)
        scale = _standard_scale([row[index] for row in [*positives, *negatives]])
        midpoint.append((pos_mean + neg_mean) / 2.0)
        scales.append(scale)
        weights.append((pos_mean - neg_mean) / scale)
    model = GroupFilterModel(
        feature_names=GROUP_FEATURES,
        midpoint=tuple(midpoint),
        scales=tuple(scales),
        weights=tuple(weights),
        threshold=0.0,
        training_group_count=len(groups),
        training_positive_count=sum(labels),
    )
    scores = [model.score(group.observation) for group in groups]
    return GroupFilterModel(
        feature_names=model.feature_names,
        midpoint=model.midpoint,
        scales=model.scales,
        weights=model.weights,
        threshold=_score_threshold(scores, labels),
        training_group_count=model.training_group_count,
        training_positive_count=model.training_positive_count,
    )


def predict_groups(
    model: GroupFilterModel, observations: Sequence[GroupObservation]
) -> tuple[GroupObservation, ...]:
    """Apply the fitted filter; labels are intentionally not accepted."""
    return tuple(
        observation
        for observation in observations
        if model.score(observation) + EPSILON >= model.threshold
    )


def _model_payload(model: GroupFilterModel) -> dict[str, Any]:
    return {
        "feature_names": list(model.feature_names),
        "midpoint": [round(value, 12) for value in model.midpoint],
        "scales": [round(value, 12) for value in model.scales],
        "weights": [round(value, 12) for value in model.weights],
        "threshold": round(model.threshold, 12),
        "training_group_count": model.training_group_count,
        "training_positive_count": model.training_positive_count,
    }


def _candidate_feature_map(
    inference_row: Mapping[str, Any],
    candidate_objects: Mapping[str, Any],
    stem_features: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result = {}
    for candidate in inference_row.get("candidate_predictions", []):
        candidate_id = _candidate_id(candidate)
        patch = candidate_objects.get(candidate_id)
        dense_features = {}
        if patch is not None:
            vector = patch.patches.get("dense_features", ())
            dense_features = dict(zip(dense.DENSE_FEATURES, vector, strict=True))
        result[candidate_id] = {
            "dense_features": dense_features,
            "stem_score": float(stem_features.get(candidate_id, {}).get("score", 0.0)),
        }
    return result


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
    predictions, _ = model.rank(unlabeled, selection_mode=dense.SELECTION_MODE)
    identity = dict(example.request.get("identity", {}))
    identity["automatic_measure_index"] = int(measure.measure)
    row = {
        "identity": identity,
        "staff_geometry": {"raw_staff_lines_y_px": list(measure.staff_lines)},
        "candidate_predictions": predictions,
        "source": {"image": str(measure.source_image), "sha256": measure.source_sha256},
        "truth_used": False,
    }
    return row


def _example_replay(example: Any, model: Any, selector: Mapping[str, Any]) -> ExampleReplay:
    inference_row = _inference_row_for_example(example, model)
    baseline = recovery.select_candidates(inference_row, selector)
    spacing = _staff_spacing(inference_row)
    x_radius = spacing * float(selector["nms_x_spaces"])
    stem_features, _ = recovery.candidate_local_stem_features(inference_row)
    candidate_objects = {row.candidate.id: row.candidate for row in example.measure.rows}
    feature_map = _candidate_feature_map(inference_row, candidate_objects, stem_features)
    image_width = float(Image.open(example.measure.source_image).size[0])
    observations = build_group_observations(
        baseline,
        x_radius_px=x_radius,
        spacing=spacing,
        candidate_features=feature_map,
        image_width=image_width,
        example_key=example.key,
        work_id=str(example.request["identity"]["slug"]),
    )
    truth_xs = truth_group_centers([note.x for note in example.notes], x_radius)
    slug, system, measure = _identity(inference_row)
    return ExampleReplay(
        key=example.key,
        work_id=slug,
        system_index=system,
        measure_index=measure,
        inference_row=inference_row,
        selector=dict(selector),
        baseline_groups=tuple(cluster_selected_candidates(baseline, x_radius)),
        observations=observations,
        truth_centers=truth_xs,
        x_radius_px=x_radius,
        source_image=example.measure.source_image,
    )


def _labels_for_replay(replay: ExampleReplay) -> list[LabeledGroup]:
    labels = {
        index
        for index, _ in match_group_centers(
            [item.x_center for item in replay.observations],
            replay.truth_centers,
            replay.x_radius_px,
        )
    }
    return [
        LabeledGroup(observation, int(index in labels))
        for index, observation in enumerate(replay.observations)
    ]


def _predicted_metrics(
    replay: ExampleReplay, observations: Sequence[GroupObservation]
) -> dict[str, Any]:
    return group_metrics(
        [observation.x_center for observation in observations],
        replay.truth_centers,
        replay.x_radius_px,
    )


def _fold_result(
    heldout_work: str,
    training: Sequence[ExampleReplay],
    evaluation: Sequence[ExampleReplay],
) -> dict[str, Any]:
    labeled = [group for replay in training for group in _labels_for_replay(replay)]
    model = fit_group_filter(labeled)
    baseline_rows = []
    proposed_rows = []
    for replay in evaluation:
        baseline = _predicted_metrics(replay, replay.observations)
        proposed_observations = predict_groups(model, replay.observations)
        proposed = _predicted_metrics(replay, proposed_observations)
        baseline_rows.append({"key": replay.key, "metrics": baseline})
        proposed_rows.append(
            {
                "key": replay.key,
                "metrics": proposed,
                "accepted_group_ids": [item.group_id for item in proposed_observations],
                "group_scores": {
                    item.group_id: round(model.score(item), 9) for item in replay.observations
                },
            }
        )
    return {
        "heldout_work": heldout_work,
        "training_work": sorted({replay.work_id for replay in training}),
        "training_group_count": model.training_group_count,
        "training_positive_count": model.training_positive_count,
        "calibration_model": _model_payload(model),
        "threshold": round(model.threshold, 9),
        "baseline": _aggregate_metrics([row["metrics"] for row in baseline_rows]),
        "proposed": _aggregate_metrics([row["metrics"] for row in proposed_rows]),
        "per_measure": {
            "baseline": baseline_rows,
            "proposed": proposed_rows,
        },
    }


def _aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    example_count = sum(int(row["example_count"]) if "example_count" in row else 1 for row in rows)
    return {
        "example_count": example_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "predicted_group_count": sum(int(row["predicted_group_count"]) for row in rows),
        "truth_group_count": sum(int(row["truth_group_count"]) for row in rows),
    }


def _adoption_gate(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baseline = _aggregate_metrics([fold["baseline"] for fold in folds])
    proposed = _aggregate_metrics([fold["proposed"] for fold in folds])
    no_work_recall_loss = all(
        int(fold["proposed"]["tp"]) >= int(fold["baseline"]["tp"])
        and int(fold["proposed"]["fn"]) <= int(fold["baseline"]["fn"])
        for fold in folds
    )
    improved = (
        proposed["precision"] > baseline["precision"] + EPSILON
        and proposed["f1"] > baseline["f1"] + EPSILON
    )
    return {
        "status": "adopt" if no_work_recall_loss and improved else "reject",
        "runtime_adoption_eligible": False,
        "no_true_group_loss_per_work": no_work_recall_loss,
        "precision_improved": proposed["precision"] > baseline["precision"] + EPSILON,
        "f1_improved": proposed["f1"] > baseline["f1"] + EPSILON,
        "baseline": baseline,
        "proposed": proposed,
        "reason": (
            "Proposed filter preserves true groups in both work-disjoint folds and improves "
            "both aggregate precision and F1."
            if no_work_recall_loss and improved
            else "Adoption rejected: the strict no-loss and precision/F1 improvement gate failed."
        ),
    }


def _truth_group_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(len({int(note["onset_divisions"]) for note in row.get("notes", [])}) for row in rows)


def _holdout_replays(
    inference_rows: Sequence[Mapping[str, Any]],
    *,
    request_rows: Sequence[Mapping[str, Any]] | None,
    out_dir: Path,
    model: Any,
    selector: Mapping[str, Any],
) -> list[ExampleReplay]:
    requests_by_measure = {}
    if request_rows:
        requests_by_measure = {
            int(row["identity"]["system_measure_index"]): row for row in request_rows
        }
    replays = []
    for row in inference_rows:
        slug, system, measure = _identity(row)
        spacing = _staff_spacing(row)
        x_radius = spacing * float(selector["nms_x_spaces"])
        baseline = recovery.select_candidates(row, selector)
        candidate_objects: dict[str, Any] = {}
        request = requests_by_measure.get(measure)
        if request is not None:
            prepared = dense._prepare_dense_measure(request, out_dir=out_dir)
            candidate_objects = {candidate.id: candidate for candidate in prepared.candidates}
        stem_features, _ = recovery.candidate_local_stem_features(row)
        feature_map = _candidate_feature_map(row, candidate_objects, stem_features)
        source = (
            row.get("source", {}).get("image") if isinstance(row.get("source"), Mapping) else None
        )
        source_path = Path(str(source)) if source else Path(".")
        if not source_path.is_absolute():
            source_path = REPO_ROOT / source_path
        with Image.open(source_path.resolve()) as opened:
            width = float(opened.width)
        observations = build_group_observations(
            baseline,
            x_radius_px=x_radius,
            spacing=spacing,
            candidate_features=feature_map,
            image_width=width,
            example_key=f"{slug}:S{system:02d}M{measure:02d}",
            work_id=slug,
        )
        replays.append(
            ExampleReplay(
                key=f"{slug}:S{system:02d}M{measure:02d}",
                work_id=slug,
                system_index=system,
                measure_index=measure,
                inference_row=dict(row),
                selector=dict(selector),
                baseline_groups=tuple(cluster_selected_candidates(baseline, x_radius)),
                observations=observations,
                truth_centers=(),
                x_radius_px=x_radius,
                source_image=source_path.resolve(),
            )
        )
    return replays


def _write_overlay(
    replay: ExampleReplay,
    accepted: Sequence[GroupObservation],
    output_path: Path,
) -> None:
    with Image.open(replay.source_image) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    accepted_ids = {item.group_id for item in accepted}
    baseline = recovery.select_candidates(replay.inference_row, replay.selector)
    for group in cluster_selected_candidates(baseline, replay.x_radius_px):
        x = int(round(float(group["x_center"])))
        accepted_group = group["group_id"] in accepted_ids
        color = (0, 150, 0) if accepted_group else (200, 40, 40)
        draw.line((x, 0, x, image.height - 1), fill=color, width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _write_markdown(report: Mapping[str, Any]) -> str:
    gate = report["adoption_gate"]
    lines = [
        "# Consumed Onset-Group Selector",
        "",
        "This is a bounded spike over consumed Aviador/Carrizal candidate reviews.",
        "The filter only removes groups already admitted by the frozen selector.",
        "",
        f"- Adoption gate: **{gate['status']}**",
        f"- Runtime adoption eligible: `{str(gate['runtime_adoption_eligible']).lower()}`",
        f"- Gate reason: {gate['reason']}",
        "",
        "## Work-disjoint folds",
        "",
        "| Held-out work | Baseline P/R/F1 | Proposed P/R/F1 | Baseline groups | Proposed groups |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for fold in report["folds"]:
        baseline = fold["baseline"]
        proposed = fold["proposed"]
        lines.append(
            f"| `{fold['heldout_work']}` | "
            f"{baseline['precision']:.3f}/{baseline['recall']:.3f}/{baseline['f1']:.3f} | "
            f"{proposed['precision']:.3f}/{proposed['recall']:.3f}/{proposed['f1']:.3f} | "
            f"{baseline['predicted_group_count']} | {proposed['predicted_group_count']} |"
        )
    baseline = gate["baseline"]
    proposed = gate["proposed"]
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"Baseline: TP={baseline['tp']} FP={baseline['fp']} FN={baseline['fn']} "
            f"P={baseline['precision']:.3f} R={baseline['recall']:.3f} F1={baseline['f1']:.3f}",
            f"Proposed: TP={proposed['tp']} FP={proposed['fp']} FN={proposed['fn']} "
            f"P={proposed['precision']:.3f} R={proposed['recall']:.3f} F1={proposed['f1']:.3f}",
            "",
            "## La Chata",
            "",
            f"Status: `{report['la_chata']['status']}`.",
            "",
        ]
    )
    if report["la_chata"].get("status") == "evaluated_count_only_truth_has_no_x":
        lines.append(
            "The frozen truth was opened only after predictions were materialized. "
            "It provides onset divisions, not pixel x-coordinates, so precision/recall are "
            "not claimed."
        )
    return "\n".join(lines) + "\n"


def run_experiment(
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    model_path: Path = DEFAULT_MODEL,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    la_chata_inference: Path | None = DEFAULT_LA_CHATA_INFERENCE,
    la_chata_requests: Path | None = DEFAULT_LA_CHATA_REQUESTS,
    la_chata_truth: Path | None = DEFAULT_LA_CHATA_TRUTH,
    overlays: bool = False,
) -> dict[str, Any]:
    out_dir = out_dir.resolve()
    model_path = model_path.resolve()
    output_dir = output_dir.resolve()
    model_payload = _read_json(model_path, label="frozen model")
    model, _pitch_predictor, model_audit = heldout.reconstruct_model(model_payload)
    selector = recovery.selector_config_from_model(model_payload)

    aviador, aviador_sources = cross_score._load_aviador_examples(
        out_dir, reviews_dir=cross_score.DEFAULT_REVIEWS_DIR
    )
    carrizal_by_policy, carrizal_sources = cross_score._load_carrizal_examples(
        out_dir, manifest_path=cross_score.DEFAULT_CARRIZAL_REVIEWS
    )
    consumed_examples = [*aviador, *carrizal_by_policy["C"]]
    replays = [_example_replay(example, model, selector) for example in consumed_examples]
    by_work: dict[str, list[ExampleReplay]] = defaultdict(list)
    for replay in replays:
        by_work[replay.work_id].append(replay)
    folds = []
    for heldout_work in sorted(by_work):
        training = [
            replay for work, rows in by_work.items() if work != heldout_work for replay in rows
        ]
        evaluation = by_work[heldout_work]
        folds.append(_fold_result(heldout_work, training, evaluation))
    gate = _adoption_gate(folds)

    final_labeled = [group for replay in replays for group in _labels_for_replay(replay)]
    final_model = fit_group_filter(final_labeled)

    la_chata_report: dict[str, Any] = {"status": "not_run_no_la_chata_paths"}
    holdout_replays: list[ExampleReplay] = []
    if la_chata_inference is not None:
        inference_rows = _read_jsonl(la_chata_inference, label="La Chata inference")
        request_rows = (
            _read_jsonl(la_chata_requests, label="La Chata requests")
            if la_chata_requests is not None
            else None
        )
        # This is the complete proposed prediction materialization boundary.
        holdout_replays = _holdout_replays(
            inference_rows,
            request_rows=request_rows,
            out_dir=out_dir,
            model=model,
            selector=selector,
        )
        holdout_predictions = [
            {
                "key": replay.key,
                "baseline_group_count": len(replay.observations),
                "proposed_group_count": len(predict_groups(final_model, replay.observations)),
            }
            for replay in holdout_replays
        ]
        if la_chata_truth is not None:
            truth_rows = _read_jsonl(la_chata_truth, label="La Chata truth")
            truth_count = _truth_group_count(truth_rows)
            la_chata_report = {
                "status": "evaluated_count_only_truth_has_no_x",
                "truth_group_count": truth_count,
                "predictions_materialized_before_truth_read": True,
                "precision": None,
                "recall": None,
                "per_measure": holdout_predictions,
            }
        else:
            la_chata_report = {
                "status": "predicted_no_truth_path",
                "predictions_materialized_before_truth_read": True,
                "per_measure": holdout_predictions,
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    if overlays:
        for replay in [*replays, *holdout_replays]:
            accepted = predict_groups(final_model, replay.observations)
            name = f"{replay.work_id}_s{replay.system_index:02d}_m{replay.measure_index:02d}.png"
            _write_overlay(replay, accepted, output_dir / "overlays" / name)
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "consumed_onset_group_selector_spike",
        "output_version": OUTPUT_VERSION,
        "protocol": {
            "candidate_selection": (
                "replay frozen threshold selector, then cluster selected candidates by "
                "frozen x-NMS radius"
            ),
            "features": list(GROUP_FEATURES),
            "truth_blind_inference": True,
            "calibration": "work-disjoint consumed Aviador/Carrizal folds",
            "count_or_score_constraints": (
                "no expected note/group count and no MusicXML used for predictions"
            ),
            "filter_scope": "remove-only; no new onset groups are created",
        },
        "inputs": {
            "model": _file_pin(model_path),
            "consumed_sources": sorted([*aviador_sources, *carrizal_sources], key=str),
            "la_chata_inference": _file_pin(la_chata_inference) if la_chata_inference else None,
            "la_chata_requests": _file_pin(la_chata_requests) if la_chata_requests else None,
            "la_chata_truth": _file_pin(la_chata_truth) if la_chata_truth else None,
        },
        "model_audit": model_audit,
        "folds": folds,
        "adoption_gate": gate,
        "final_fit": {
            "group_count": final_model.training_group_count,
            "positive_group_count": final_model.training_positive_count,
            "calibration_model": _model_payload(final_model),
        },
        "la_chata": la_chata_report,
        "artifacts": {
            "output_dir": _display_path(output_dir),
            "report_json": _display_path(output_dir / "report.json"),
            "report_markdown": _display_path(output_dir / "report.md"),
            "overlays": _display_path(output_dir / "overlays") if overlays else None,
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(_write_markdown(report), encoding="utf-8")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--la-chata-inference", type=Path, default=DEFAULT_LA_CHATA_INFERENCE)
    parser.add_argument("--la-chata-requests", type=Path, default=DEFAULT_LA_CHATA_REQUESTS)
    parser.add_argument("--la-chata-truth", type=Path, default=DEFAULT_LA_CHATA_TRUTH)
    parser.add_argument("--no-la-chata", action="store_true")
    parser.add_argument("--overlays", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = run_experiment(
            args.out_dir,
            model_path=args.model,
            output_dir=args.output_dir,
            la_chata_inference=None if args.no_la_chata else args.la_chata_inference,
            la_chata_requests=None if args.no_la_chata else args.la_chata_requests,
            la_chata_truth=None if args.no_la_chata else args.la_chata_truth,
            overlays=args.overlays,
        )
    except (FileNotFoundError, KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(report["artifacts"]["report_json"])
    print(report["artifacts"]["report_markdown"])
    print(f"adoption gate: {report['adoption_gate']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
