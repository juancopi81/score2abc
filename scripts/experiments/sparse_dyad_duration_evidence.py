"""Truth-blind duration evidence derived from the fixed sparse-dyad rule."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA_VERSION = 1
EVIDENCE_KIND = "sparse_shared_stem_dotted_half"
DOTTED_HALF_BEATS = 3.0


def derive_dotted_half_duration_evidence(
    decision: Mapping[str, Any],
    candidate_lane: Sequence[Mapping[str, Any]],
    *,
    expected_measure_beats: float | int | None,
) -> dict[str, Any]:
    """Recognize a full-measure dotted-half dyad without consulting truth."""
    _reject_truth_inputs(decision)
    _reject_truth_inputs(candidate_lane)
    if decision.get("accepted") is not True:
        return _not_applied("sparse_repair_not_accepted")
    if decision.get("reason") != "accepted":
        raise ValueError("Accepted sparse repair must have reason='accepted'")
    if expected_measure_beats is None or not math.isclose(
        float(expected_measure_beats), DOTTED_HALF_BEATS
    ):
        return _not_applied("expected_meter_is_not_three_beats")

    chosen = decision.get("chosen_pair")
    if not isinstance(chosen, Mapping):
        raise ValueError("Accepted sparse repair has no chosen_pair")
    chosen_ids = _two_unique_ids(chosen.get("candidate_ids"), "chosen pair")
    dot_pairs = chosen.get("augmentation_dot_pairs")
    if not isinstance(dot_pairs, list) or not dot_pairs:
        raise ValueError("Accepted sparse repair has no augmentation-dot pair")
    normalized_dot_pairs = [
        (
            _two_unique_ids(pair.get("candidate_ids"), "augmentation-dot pair")
            if isinstance(pair, Mapping)
            else _raise_invalid_dot_pair()
        )
        for pair in dot_pairs
    ]

    if len(candidate_lane) != 2:
        raise ValueError("Accepted sparse dotted dyad must contain exactly two candidates")
    lane_ids = [str(row.get("candidate_id")) for row in candidate_lane]
    if len(set(lane_ids)) != 2 or set(lane_ids) != set(chosen_ids):
        raise ValueError("Sparse dotted-dyad lane does not match the chosen pair")
    onset_groups = {int(row.get("onset_group_index", 0)) for row in candidate_lane}
    if len(onset_groups) != 1 or next(iter(onset_groups)) <= 0:
        raise ValueError("Sparse dotted-dyad candidates must share one positive onset group")

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": EVIDENCE_KIND,
        "applied": True,
        "reason": "accepted_visual_dotted_half_dyad",
        "candidate_ids": lane_ids,
        "onset_group_index": next(iter(onset_groups)),
        "augmentation_dot_pairs": normalized_dot_pairs,
        "duration_beats": DOTTED_HALF_BEATS,
        "expected_measure_beats": DOTTED_HALF_BEATS,
        "suppress_residual_rest_hypotheses": True,
        "truth_used": False,
    }


def _not_applied(reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": EVIDENCE_KIND,
        "applied": False,
        "reason": reason,
        "truth_used": False,
    }


def _two_unique_ids(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two candidate IDs")
    candidate_ids = [str(item) for item in value]
    if len(set(candidate_ids)) != 2 or any(not item for item in candidate_ids):
        raise ValueError(f"{label} candidate IDs must be unique and non-empty")
    return candidate_ids


def _raise_invalid_dot_pair() -> list[str]:
    raise ValueError("Augmentation-dot pair must be an object")


def _reject_truth_inputs(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"truth", "ground_truth", "expected_notes", "expected_duration"}:
                raise ValueError(f"Forbidden truth field in duration evidence: {key}")
            if key in {"truth_used", "truth_accessed"} and child is not False:
                raise ValueError(f"Duration evidence is not truth-blind: {key}")
            _reject_truth_inputs(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _reject_truth_inputs(child)
