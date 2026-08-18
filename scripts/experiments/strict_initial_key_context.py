"""Bounded adapter for independently supported initial/system-entry key reads."""

from __future__ import annotations

from typing import Any, Mapping

SCHEMA_VERSION = 1
MODE_INITIAL = "initial"
STATUS_CONFIRMED = "confirmed_explicit"
STATUS_INHERITED = "inherited"
STATUS_UNKNOWN = "unknown"
KIND = "strict_initial_or_system_entry_visual_key_state"
SUPPORTED_FIFTHS = frozenset({-1, 2})
SUPPORTED_METHODS = frozenset({"standard_position_sequence"})
FAMILY_BY_FIFTHS = {-1: "flat", 2: "sharp"}


def strict_initial_key_state(
    prediction: Mapping[str, Any],
    *,
    source_system_index: int,
    inherited_fifths: int | None = None,
) -> dict[str, Any]:
    """Accept only the one-flat and two-sharp truth-blind initial patterns."""
    reasons: list[str] = []
    if prediction.get("mode") != MODE_INITIAL:
        reasons.append("detector_mode_is_not_initial")
    if prediction.get("truth_used_for_prediction") is not False:
        reasons.append("prediction_is_not_truth_blind")
    if prediction.get("gate_passed") is not True:
        reasons.append("detector_structural_gate_failed")

    fifths = prediction.get("fifths")
    if isinstance(fifths, bool) or not isinstance(fifths, int):
        reasons.append("detector_fifths_is_not_an_integer")
        fifths = None
    elif fifths not in SUPPORTED_FIFTHS:
        reasons.append("detector_fifths_is_not_independently_supported")

    expected_family = FAMILY_BY_FIFTHS.get(fifths)
    if (
        expected_family is not None
        and prediction.get("predicted_signature_family") != expected_family
    ):
        reasons.append("detector_signature_family_is_inconsistent")
    if prediction.get("selection_method") not in SUPPORTED_METHODS:
        reasons.append("detector_selection_method_is_not_supported")

    accidental_count = prediction.get("accidental_count")
    selected_ids = prediction.get("selected_glyph_ids")
    if fifths in SUPPORTED_FIFTHS:
        if accidental_count != abs(fifths):
            reasons.append("accidental_count_does_not_match_fifths")
        if not isinstance(selected_ids, list) or len(selected_ids) != abs(fifths):
            reasons.append("selected_glyph_count_does_not_match_fifths")

    search_region = prediction.get("search_region")
    boundary: int | None = None
    if not isinstance(search_region, Mapping):
        reasons.append("detector_search_region_is_missing")
    else:
        right_px = search_region.get("right_px")
        if isinstance(right_px, bool) or not isinstance(right_px, int) or right_px < 0:
            reasons.append("detector_search_region_right_is_invalid")
        else:
            boundary = right_px

    if reasons:
        return unknown_key_state(
            source_system_index=source_system_index,
            reasons=reasons,
            inherited_fifths=inherited_fifths,
            detector_prediction=prediction,
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": STATUS_CONFIRMED,
        "fifths": fifths,
        "previous_fifths": inherited_fifths,
        "source_system_index": source_system_index,
        "applies_after_system_x_px": boundary,
        "detector": _detector_provenance(prediction),
        "rejection_reasons": [],
    }


def unknown_key_state(
    *,
    source_system_index: int,
    reasons: list[str],
    inherited_fifths: int | None = None,
    detector_prediction: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    inherited = inherited_fifths in SUPPORTED_FIFTHS
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": STATUS_INHERITED if inherited else STATUS_UNKNOWN,
        "fifths": inherited_fifths if inherited else None,
        "previous_fifths": inherited_fifths if inherited else None,
        "source_system_index": source_system_index,
        "applies_after_system_x_px": None,
        "detector": (
            _detector_provenance(detector_prediction) if detector_prediction is not None else None
        ),
        "rejection_reasons": list(reasons),
    }


def key_hint_for_fifths(fifths: int) -> str:
    if fifths == -1:
        return "1 flat(s): Bb"
    if fifths == 2:
        return "2 sharp(s): F#, C#"
    raise ValueError(f"Unsupported strict visual key fifths: {fifths}")


def key_hint_for_candidate(
    request: Mapping[str, Any],
    *,
    candidate_x_px: float,
) -> str | None:
    """Return candidate-local visual context, preserving legacy metadata behavior."""
    allowed = request.get("allowed_context")
    if not isinstance(allowed, Mapping):
        return None
    state = allowed.get("visual_key_state")
    if not isinstance(state, Mapping):
        value = allowed.get("key_hint")
        return str(value) if value is not None else None
    fifths = effective_fifths_for_candidate(
        state,
        candidate_x_px=candidate_x_px,
        crop_left_px=_request_crop_left_px(request),
    )
    return key_hint_for_fifths(fifths) if fifths is not None else None


def effective_fifths_for_candidate(
    state: Mapping[str, Any],
    *,
    candidate_x_px: float,
    crop_left_px: int | None,
) -> int | None:
    if state.get("kind") != KIND:
        return None
    status = state.get("status")
    fifths = _supported_fifths(state.get("fifths"))
    if status == STATUS_INHERITED:
        return fifths
    if status != STATUS_CONFIRMED or fifths is None:
        return None
    boundary = state.get("applies_after_system_x_px")
    previous = _supported_fifths(state.get("previous_fifths"))
    if isinstance(boundary, bool) or not isinstance(boundary, int) or crop_left_px is None:
        return None
    return fifths if crop_left_px + float(candidate_x_px) >= boundary else previous


def _request_crop_left_px(request: Mapping[str, Any]) -> int | None:
    prepared = request.get("prepared_provenance")
    if isinstance(prepared, Mapping):
        bbox = prepared.get("bbox_px")
        if isinstance(bbox, list) and bbox:
            value = bbox[0]
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    bounds = request.get("x_bounds_px")
    if isinstance(bounds, Mapping):
        value = bounds.get("left")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _supported_fifths(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value in SUPPORTED_FIFTHS:
        return value
    return None


def _detector_provenance(prediction: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mode": prediction.get("mode"),
        "gate_passed": prediction.get("gate_passed"),
        "truth_used_for_prediction": prediction.get("truth_used_for_prediction"),
        "selection_method": prediction.get("selection_method"),
        "predicted_signature_family": prediction.get("predicted_signature_family"),
        "accidental_count": prediction.get("accidental_count"),
        "selected_glyph_ids": list(prediction.get("selected_glyph_ids") or []),
        "input": (
            dict(prediction["input"]) if isinstance(prediction.get("input"), Mapping) else None
        ),
        "search_region": (
            dict(prediction["search_region"])
            if isinstance(prediction.get("search_region"), Mapping)
            else None
        ),
    }
