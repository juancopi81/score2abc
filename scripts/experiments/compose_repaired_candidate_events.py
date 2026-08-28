"""Compose a fixed repaired-candidate lane into truth-blind melody events.

This module does not select, add, or remove candidates. It validates a repaired
lane against deterministic dense proposals, preserves its onset groups, and
then reuses the bounded pitch and rhythm helpers from the melody spike.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from scripts.experiments import spike_anchored_rhythm_parser as rhythm
from scripts.experiments import spike_composed_melody_chain as composed
from scripts.experiments import spike_meter_gap_resolver as meter_gap
from scripts.experiments import spike_review_augmented_selector as dense
from scripts.experiments import strict_initial_key_context as visual_key_context

SCHEMA_VERSION = 1
STATUS_MATERIALIZED = "materialized"
STATUS_MISSING_METER = "not_materialized_missing_expected_measure_beats"
STATUS_REQUEST_METER_FALLBACK = "not_materialized_request_only_meter_fallback"


def compose_repaired_candidate_events(
    request: Mapping[str, Any],
    inference_row: Mapping[str, Any],
    candidate_lane: Sequence[Mapping[str, Any]],
    *,
    pitch_predictor: composed.PitchPredictor,
    out_dir: Path,
    selector_method_id: str,
) -> dict[str, Any]:
    """Materialize events from a fixed repaired lane without opening truth."""
    _reject_forbidden_inputs(request, inference_row, candidate_lane)
    _validate_inference_boundary(request, inference_row, out_dir=out_dir)
    measure = dense._prepare_dense_measure(request, out_dir=out_dir)
    composed._validate_unlabeled_identity(request, measure)
    candidates = _validate_candidate_lane(candidate_lane, measure.candidates)

    image_path = composed._resolve_request_image(request, out_dir)
    if image_path.resolve() != measure.source_image.resolve():
        raise ValueError("Dense candidate source image drift")
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")

    staff_lines = [int(value) for value in request["staff_geometry"]["raw_staff_lines_y_px"]]
    spacing = rhythm.staff_spacing(staff_lines)
    anchors = _build_anchors(
        candidates,
        request=request,
        image=image,
        pitch_predictor=pitch_predictor,
        selector_method_id=selector_method_id,
    )
    groups = _build_fixed_groups(anchors)
    key_context = _automatic_key_context(request, candidates)
    common = {
        "schema_version": SCHEMA_VERSION,
        "identity": dict(request["identity"]),
        "truth_used": False,
        "selector_method_id": selector_method_id,
        "candidate_lane": [dict(row) for row in candidate_lane],
        "anchors": anchors,
        "groups": groups,
        "automatic_key_context": key_context,
    }

    expected_raw = request["allowed_context"].get("expected_measure_beats")
    if expected_raw is None:
        return {
            **common,
            "status": STATUS_MISSING_METER,
            "prediction": None,
            "notes": _unmaterialized_notes(anchors),
            "onsets": None,
            "durations": None,
            "rests": None,
            "decoder_status": "not_applied_missing_expected_measure_beats",
            "expected_measure_beats": None,
            "observed_extent_beats": None,
            "decoded_extent_beats": None,
            "meter_valid": None,
            "anchor_features": [],
            "residual_rest_features": [],
            "visual_symbols": [],
            "decoded_symbols": [],
            "inference_provenance": _provenance(
                selector_method_id=selector_method_id,
                anchor_count=len(anchors),
                key_context=key_context,
                rhythm_decoding_applied=False,
            ),
        }

    expected_beats = _positive_number(expected_raw, "expected_measure_beats")
    allow_pickup = bool(request["allowed_context"].get("allow_pickup", False))
    anchor_features = rhythm.extract_anchor_features(image, anchors, staff_lines)
    rest_features = rhythm.extract_residual_rest_features(image, groups, staff_lines)
    visual_symbols = rhythm.build_visual_symbols(groups, anchor_features, rest_features)
    observed_extent = _symbol_extent(visual_symbols)

    rest_features, visual_symbols, leading_rest_applied = _apply_bounded_leading_rest(
        groups=groups,
        anchors=anchors,
        anchor_features=anchor_features,
        rest_features=rest_features,
        visual_symbols=visual_symbols,
        staff_lines=staff_lines,
        staff_spacing=spacing,
        expected_beats=expected_beats,
        allow_pickup=allow_pickup,
    )
    decoded_symbols, decoder_status, meter_fallback_applied = _decode_with_meter_fallback(
        visual_symbols,
        expected_beats=expected_beats,
        allow_pickup=allow_pickup,
        staff_spacing=spacing,
    )
    decoded_extent = _symbol_extent(decoded_symbols)
    if meter_fallback_applied:
        provenance = _provenance(
            selector_method_id=selector_method_id,
            anchor_count=len(anchors),
            key_context=key_context,
            rhythm_decoding_applied=False,
        )
        provenance["leading_rest_repair"] = {
            "applied": leading_rest_applied,
            "truth_used": False,
        }
        provenance["meter_fallback"] = {
            "applied": True,
            "accepted_as_transcription": False,
            "expected_beats": expected_beats,
            "truth_used": False,
        }
        return {
            **common,
            "status": STATUS_REQUEST_METER_FALLBACK,
            "prediction": None,
            "notes": _unmaterialized_notes(anchors),
            "onsets": None,
            "durations": None,
            "rests": None,
            "decoder_status": decoder_status,
            "expected_measure_beats": expected_beats,
            "observed_extent_beats": observed_extent,
            "decoded_extent_beats": decoded_extent,
            "meter_valid": False,
            "anchor_features": anchor_features,
            "residual_rest_features": rest_features,
            "visual_symbols": visual_symbols,
            "decoded_symbols": decoded_symbols,
            "inference_provenance": provenance,
        }
    meter_valid = math.isclose(decoded_extent, expected_beats) or (
        allow_pickup and decoded_extent <= expected_beats
    )
    prediction = rhythm.symbols_to_hypothesis(
        decoded_symbols,
        identity=request["identity"],
        decoder_status=decoder_status,
    )
    provenance = _provenance(
        selector_method_id=selector_method_id,
        anchor_count=len(anchors),
        key_context=key_context,
        rhythm_decoding_applied=True,
    )
    provenance["leading_rest_repair"] = {
        "applied": leading_rest_applied,
        "truth_used": False,
    }
    provenance["meter_fallback"] = {
        "applied": meter_fallback_applied,
        "expected_beats": expected_beats,
        "truth_used": False,
    }
    prediction["inference_provenance"] = provenance
    note_tokens = [token for token in prediction["rhythm_tokens"] if token["kind"] == "note"]
    return {
        **common,
        "status": STATUS_MATERIALIZED,
        "prediction": prediction,
        "notes": prediction["notes"],
        "onsets": [token["onset_beats"] for token in note_tokens],
        "durations": [token["duration_beats"] for token in note_tokens],
        "rests": prediction["rests"],
        "decoder_status": decoder_status,
        "expected_measure_beats": expected_beats,
        "observed_extent_beats": observed_extent,
        "decoded_extent_beats": decoded_extent,
        "meter_valid": meter_valid,
        "anchor_features": anchor_features,
        "residual_rest_features": rest_features,
        "visual_symbols": visual_symbols,
        "decoded_symbols": decoded_symbols,
        "inference_provenance": provenance,
    }


def _validate_inference_boundary(
    request: Mapping[str, Any],
    inference_row: Mapping[str, Any],
    *,
    out_dir: Path,
) -> None:
    if inference_row.get("truth_used") is not False:
        raise ValueError("Inference row is not explicitly truth-blind")
    for field in ("identity", "allowed_context", "allowed_context_provenance", "staff_geometry"):
        if inference_row.get(field) != request.get(field):
            raise ValueError(f"Inference/request {field} drift")

    source = inference_row.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("Inference source is missing")
    expected_path = composed._resolve_request_image(request, out_dir)
    actual_path = _resolve_inference_source(source.get("image"), out_dir=out_dir)
    if actual_path != expected_path:
        raise ValueError("Inference/request source path drift")
    expected_hash = str(request["images"]["raw"]["sha256"])
    if source.get("sha256") != expected_hash:
        raise ValueError("Inference/request source hash drift")


def _resolve_inference_source(value: Any, *, out_dir: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Inference source image is invalid")
    raw = Path(value)
    if raw.is_absolute():
        return raw.resolve()
    candidates = [composed.REPO_ROOT / raw, out_dir / raw]
    existing = {path.resolve() for path in candidates if path.exists()}
    if len(existing) != 1:
        raise ValueError("Inference source image cannot be resolved unambiguously")
    return existing.pop()


def _validate_candidate_lane(
    lane: Sequence[Mapping[str, Any]],
    dense_candidates: Sequence[Any],
) -> list[tuple[Mapping[str, Any], Any]]:
    if not lane:
        raise ValueError("Candidate lane is empty")
    by_id = {str(candidate.id): candidate for candidate in dense_candidates}
    if len(by_id) != len(dense_candidates):
        raise ValueError("Dense candidate identities are duplicated")

    seen: set[str] = set()
    validated: list[tuple[Mapping[str, Any], Any]] = []
    previous_x = -math.inf
    previous_group = 0
    groups: set[int] = set()
    for row in lane:
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id or candidate_id in seen:
            raise ValueError("Candidate lane contains a duplicate candidate")
        seen.add(candidate_id)
        candidate = by_id.get(candidate_id)
        if candidate is None:
            raise ValueError(f"Candidate lane references unknown candidate: {candidate_id}")
        center = row.get("center")
        if not isinstance(center, Mapping):
            raise ValueError(f"Candidate lane center is missing: {candidate_id}")
        actual = (round(float(center["x"]), 3), round(float(center["y"]), 3))
        expected = (round(float(candidate.center_x), 3), round(float(candidate.center_y), 3))
        if actual != expected:
            raise ValueError(f"Candidate lane center drift: {candidate_id}")
        group = row.get("onset_group_index")
        if isinstance(group, bool) or not isinstance(group, int) or group <= 0:
            raise ValueError("Onset group indices must be positive integers")
        if actual[0] < previous_x or group < previous_group:
            raise ValueError("Candidate lane must have monotonic x and onset-group order")
        previous_x = actual[0]
        previous_group = group
        groups.add(group)
        validated.append((row, candidate))
    expected_groups = set(range(1, max(groups) + 1))
    if groups != expected_groups:
        raise ValueError("Onset group indices must be contiguous from one")
    return validated


def _build_anchors(
    candidates: Sequence[tuple[Mapping[str, Any], Any]],
    *,
    request: Mapping[str, Any],
    image: Image.Image,
    pitch_predictor: composed.PitchPredictor,
    selector_method_id: str,
) -> list[dict[str, Any]]:
    anchors = []
    for order, (lane_row, candidate) in enumerate(candidates, start=1):
        anchors.append(
            {
                "order": order,
                "pitch": pitch_predictor(candidate, request, image),
                "center": {
                    "x": round(float(candidate.center_x), 3),
                    "y": round(float(candidate.center_y), 3),
                },
                "source": {
                    "kind": "repaired_automatic_candidate",
                    "candidate_id": str(candidate.id),
                    "selector_method": selector_method_id,
                    "onset_group_index": int(lane_row["onset_group_index"]),
                },
            }
        )
    return anchors


def _build_fixed_groups(anchors: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for anchor in anchors:
        group = int(anchor["source"]["onset_group_index"])
        grouped.setdefault(group, []).append(dict(anchor))
    output = []
    for group_index in sorted(grouped):
        members = grouped[group_index]
        center_x = sum(float(anchor["center"]["x"]) for anchor in members) / len(members)
        output.append(
            {
                "group_id": f"g{group_index:03d}",
                "onset_group_index": group_index,
                "center_x": round(center_x, 3),
                "anchors": members,
                "pitches": [str(anchor["pitch"]) for anchor in members],
            }
        )
    return output


def _automatic_key_context(
    request: Mapping[str, Any],
    candidates: Sequence[tuple[Mapping[str, Any], Any]],
) -> dict[str, Any]:
    allowed = request["allowed_context"]
    provenance = request.get("allowed_context_provenance")
    return {
        "key_hint": allowed.get("key_hint"),
        "visual_key_state": allowed.get("visual_key_state"),
        "provenance": provenance.get("key_hint") if isinstance(provenance, Mapping) else None,
        "candidate_key_hints": [
            {
                "candidate_id": str(candidate.id),
                "key_hint": visual_key_context.key_hint_for_candidate(
                    request,
                    candidate_x_px=float(candidate.center_x),
                ),
            }
            for _, candidate in candidates
        ],
        "truth_used": False,
    }


def _apply_bounded_leading_rest(
    *,
    groups: Sequence[Mapping[str, Any]],
    anchors: Sequence[Mapping[str, Any]],
    anchor_features: Sequence[Mapping[str, Any]],
    rest_features: list[dict[str, Any]],
    visual_symbols: list[dict[str, Any]],
    staff_lines: Sequence[int],
    staff_spacing: float,
    expected_beats: float,
    allow_pickup: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    note_extent = sum(
        float(symbol["duration_beats"]) for symbol in visual_symbols if symbol["kind"] == "note"
    )
    first_gap = float(anchors[0]["center"]["x"]) / staff_spacing if anchors else 0.0
    should_insert = meter_gap._should_insert_leading_rest(
        expected_beats=expected_beats,
        visual_note_extent_beats=note_extent,
        first_anchor_gap_spaces=first_gap,
        allow_pickup=allow_pickup,
        has_anchors=bool(anchors),
    )
    if not should_insert:
        return rest_features, visual_symbols, False
    first_x = float(anchors[0]["center"]["x"])
    leading_gap_right = max(staff_spacing, first_x - 0.6 * staff_spacing)
    center_x = max(staff_spacing, leading_gap_right / 2.0)
    synthetic = {
        "role": "leading",
        "center_x": round(center_x, 3),
        "duration_beats": meter_gap.HALF_BEAT,
        "bbox": {
            "left": round(center_x - 0.35 * staff_spacing),
            "top": round(staff_lines[1] - 0.45 * staff_spacing),
            "right": round(center_x + 0.35 * staff_spacing),
            "bottom": round(staff_lines[3] + 0.45 * staff_spacing),
        },
        "area_staff_squared": 0.0,
        "height_staff_spacing": 0.0,
        "width_staff_spacing": 0.0,
        "evidence": "request_meter_plus_leading_gap",
        "visual_component_detected": False,
    }
    repaired_rests = [synthetic]
    repaired_symbols = rhythm.build_visual_symbols(groups, anchor_features, repaired_rests)
    return repaired_rests, repaired_symbols, True


def _decode_with_meter_fallback(
    visual_symbols: Sequence[Mapping[str, Any]],
    *,
    expected_beats: float,
    allow_pickup: bool,
    staff_spacing: float,
) -> tuple[list[dict[str, Any]], str, bool]:
    decoded, status = rhythm.decode_meter(
        visual_symbols,
        expected_beats=expected_beats,
        allow_pickup=allow_pickup,
    )
    extent = _symbol_extent(decoded)
    if math.isclose(extent, expected_beats) or (allow_pickup and extent <= expected_beats):
        return decoded, f"meter_gap_resolver:{status}", False

    symbols = [dict(symbol) for symbol in visual_symbols[: dense.MAX_RHYTHM_EVENTS_PER_MEASURE]]
    target_units = round(expected_beats * 2)
    maximum_units = sum(3 if symbol["kind"] == "rest" else 4 for symbol in symbols)
    next_x = float(symbols[-1]["x"] + staff_spacing) if symbols else 0.0
    while maximum_units < target_units:
        symbols.append(
            {
                "kind": "rest",
                "x": next_x,
                "duration_beats": 0.5,
                "duration_costs": {"0.5": 1.0, "1.0": 0.8, "1.5": 0.9},
                "evidence": "request_meter_fallback",
            }
        )
        next_x += staff_spacing
        maximum_units += 3
    decoded, status = rhythm.decode_meter(
        symbols,
        expected_beats=expected_beats,
        allow_pickup=allow_pickup,
    )
    extent = _symbol_extent(decoded)
    if not math.isclose(extent, expected_beats):
        raise ValueError(f"Request-only meter fallback failed: {extent} != {expected_beats}")
    return decoded, f"request_meter_fallback:{status}", True


def _provenance(
    *,
    selector_method_id: str,
    anchor_count: int,
    key_context: Mapping[str, Any],
    rhythm_decoding_applied: bool,
) -> dict[str, Any]:
    return {
        "notehead_selector": selector_method_id,
        "selection_mode": "fixed_repaired_candidate_lane",
        "automatic_anchor_count": anchor_count,
        "review_anchors_used": False,
        "truth_used": False,
        "candidate_reranking_applied": False,
        "onset_groups_from_repaired_lane": True,
        "rhythm_decoding_applied": rhythm_decoding_applied,
        "automatic_key_context": dict(key_context),
    }


def _unmaterialized_notes(anchors: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "order": int(anchor["order"]),
            "pitch": str(anchor["pitch"]),
            "pitch_midi": rhythm.pitch_to_midi(str(anchor["pitch"])),
            "candidate_id": str(anchor["source"]["candidate_id"]),
            "onset_group_index": int(anchor["source"]["onset_group_index"]),
            "center": dict(anchor["center"]),
            "onset_beats": None,
            "duration_beats": None,
        }
        for anchor in anchors
    ]


def _symbol_extent(symbols: Sequence[Mapping[str, Any]]) -> float:
    return round(
        sum(float(symbol["duration_beats"]) for symbol in symbols),
        6,
    )


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive number")
    return number


def _reject_forbidden_inputs(*values: Any) -> None:
    forbidden = {"truth", "truth_path", "ground_truth", "label", "labels", "musicxml"}

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for raw_key, item in value.items():
                key = str(raw_key).lower()
                if key in forbidden or "musicxml" in key:
                    raise ValueError(f"Forbidden truth or MusicXML input field: {raw_key}")
                if key in {"truth_used", "truth_accessed"} and item is not False:
                    raise ValueError(f"Truth access is not allowed: {raw_key}")
                visit(item)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                visit(item)

    for value in values:
        visit(value)
