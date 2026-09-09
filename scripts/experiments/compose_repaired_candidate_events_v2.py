"""Compose repaired candidates with bounded sparse-dyad duration evidence.

Ordinary measures replay the immutable v1 compositor. Only the previously
validated full-measure dotted-half dyad pattern can replace its rhythm result.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.experiments import compose_repaired_candidate_events as v1
from scripts.experiments import sparse_dyad_duration_evidence as duration_evidence
from scripts.experiments import spike_anchored_rhythm_parser as rhythm

SCHEMA_VERSION = 1
COMPOSITOR_VERSION = "repaired-candidate-events-v2"
DECODER_STATUS = "sparse_dyad_duration_evidence_v1:full_measure_dotted_half"


def compose_repaired_candidate_events_v2(
    request: Mapping[str, Any],
    inference_row: Mapping[str, Any],
    candidate_lane: Sequence[Mapping[str, Any]],
    sparse_repair_decision: Mapping[str, Any],
    *,
    pitch_predictor: Any,
    out_dir: Path,
    selector_method_id: str,
) -> dict[str, Any]:
    """Replay v1 and apply only the accepted full-measure dotted dyad."""
    base = v1.compose_repaired_candidate_events(
        request,
        inference_row,
        candidate_lane,
        pitch_predictor=pitch_predictor,
        out_dir=out_dir,
        selector_method_id=selector_method_id,
    )
    evidence = duration_evidence.derive_dotted_half_duration_evidence(
        sparse_repair_decision,
        candidate_lane,
        expected_measure_beats=request["allowed_context"].get("expected_measure_beats"),
    )
    if evidence["applied"] is not True:
        result = deepcopy(base)
        result["duration_evidence"] = evidence
        return result
    return _apply_dotted_half_evidence(base, evidence)


def _apply_dotted_half_evidence(
    base: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    groups = base.get("groups")
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError("Dotted-half duration evidence requires exactly one onset group")
    group = groups[0]
    if not isinstance(group, Mapping):
        raise ValueError("Dotted-half onset group is invalid")
    anchors = group.get("anchors")
    pitches = group.get("pitches")
    if not isinstance(anchors, list) or len(anchors) != 2:
        raise ValueError("Dotted-half duration evidence requires exactly two anchors")
    if not isinstance(pitches, list) or len(pitches) != 2:
        raise ValueError("Dotted-half duration evidence requires exactly two pitches")
    anchor_ids = [str(anchor["source"]["candidate_id"]) for anchor in anchors]
    if set(anchor_ids) != set(str(value) for value in evidence["candidate_ids"]):
        raise ValueError("Dotted-half duration evidence candidate identities drifted")

    duration = float(evidence["duration_beats"])
    symbol = {
        "kind": "note",
        "x": float(group["center_x"]),
        "onset_beats": 0.0,
        "duration_beats": duration,
        "pitches": [str(pitch) for pitch in pitches],
        "group_id": str(group["group_id"]),
        "evidence": duration_evidence.EVIDENCE_KIND,
        "duration_costs": {str(duration): 0.0},
    }
    prediction = rhythm.symbols_to_hypothesis(
        [symbol],
        identity=base["identity"],
        decoder_status=DECODER_STATUS,
    )
    provenance = deepcopy(base.get("inference_provenance", {}))
    provenance.update(
        {
            "rhythm_decoding_applied": True,
            "duration_evidence": dict(evidence),
            "leading_rest_repair": {"applied": False, "truth_used": False},
            "meter_fallback": {
                "applied": False,
                "expected_beats": duration,
                "truth_used": False,
            },
        }
    )
    prediction["inference_provenance"] = provenance
    result = deepcopy(base)
    result.update(
        {
            "schema_version": SCHEMA_VERSION,
            "status": v1.STATUS_MATERIALIZED,
            "prediction": prediction,
            "notes": prediction["notes"],
            "onsets": [0.0, 0.0],
            "durations": [duration, duration],
            "rests": [],
            "decoder_status": DECODER_STATUS,
            "expected_measure_beats": duration,
            "observed_extent_beats": duration,
            "decoded_extent_beats": duration,
            "meter_valid": True,
            "residual_rest_features": [],
            "visual_symbols": [symbol],
            "decoded_symbols": [symbol],
            "inference_provenance": provenance,
            "duration_evidence": dict(evidence),
            "v1_base_composition": {
                "status": base.get("status"),
                "decoder_status": base.get("decoder_status"),
                "observed_extent_beats": base.get("observed_extent_beats"),
                "decoded_extent_beats": base.get("decoded_extent_beats"),
                "meter_valid": base.get("meter_valid"),
            },
        }
    )
    return result
