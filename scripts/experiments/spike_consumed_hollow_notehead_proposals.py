"""Evaluate GT-blind hollow-notehead centers derived from candidate rim pairs.

This consumed-data spike proposes a center only when two strong candidate
windows occupy adjacent pitch rows, are diagonally separated, and surround a
ring-like ink pattern with a comparatively light center. Ground truth is loaded
only by the evaluation phase; the proposal function receives the image and
candidate artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from score2abc.utils.imaging import estimate_ink_threshold  # noqa: E402

REPORT_KIND = "vlm_melody_consumed_hollow_notehead_proposal_report"
PROPOSAL_KIND = "vlm_melody_hollow_notehead_proposals"
DEFAULT_OUTPUT = "out/vlm_melody_consumed_training/hollow_notehead_proposals_v1"
DEFAULT_FIXTURE_MANIFEST = "tests/fixtures/vlm_melody/hollow_notehead_inputs/manifest.json"
MATCH_TOLERANCE_SPACING = 0.45
MIN_CANDIDATE_SCORE = 0.64
MIN_DX_SPACING = 0.35
MAX_DX_SPACING = 0.90
MIN_DY_SPACING = 0.84
MAX_DY_SPACING = 1.16
MIN_RING_DENSITY = 0.19
MIN_RING_COVERAGE = 0.80
MIN_HOLE_AREA_SPACING_SQUARED = 0.035
MAX_HOLE_CENTER_DISTANCE_SPACING = 0.55
MIN_HOLE_SUPPORT_ALIGNMENT = 0.90
MAX_HOLE_AXIS_RATIO = 0.65
MIN_OPEN_RING_DENSITY = 0.30
MAX_OPEN_INNER_DENSITY = 0.22
CONTOUR_CLOSE_RADIUS_PX = 1


@dataclass(frozen=True)
class ReviewedMeasure:
    identity: dict[str, Any]
    lane: str
    image_path: Path
    candidates_path: Path
    truth_source_path: Path
    truth_source_kind: str
    truth_measure_index: int | None


@dataclass(frozen=True)
class HollowProposal:
    support_candidate_ids: tuple[str, str]
    center: tuple[float, float]
    bbox: tuple[int, int, int, int]
    contour_kind: str
    score: float
    features: dict[str, float]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root containing out/ and tests/fixtures/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help="Create-once report directory.",
    )
    parser.add_argument(
        "--fixture-manifest",
        type=Path,
        default=None,
        help="Hash-pinned consumed fixture manifest (defaults inside repo).",
    )
    args = parser.parse_args(argv)
    try:
        result = run_consumed_hollow_notehead_spike(
            repo_root=args.repo_root,
            output_dir=args.output_dir,
            fixture_manifest=args.fixture_manifest,
        )
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(result)
    return 0


def run_consumed_hollow_notehead_spike(
    *,
    repo_root: Path,
    output_dir: Path,
    fixture_manifest: Path | None = None,
) -> Path:
    """Freeze GT-blind proposals, then score them against consumed reviews."""
    root = repo_root.resolve()
    destination = output_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite hollow-head spike: {destination}")
    temp_dir = destination.with_name(f".{destination.name}.tmp")
    if temp_dir.exists():
        raise FileExistsError(f"Refusing stale hollow-head temp directory: {temp_dir}")

    manifest_path = (
        fixture_manifest.resolve()
        if fixture_manifest is not None
        else (root / DEFAULT_FIXTURE_MANIFEST).resolve()
    )
    reviewed = load_reviewed_measures(root, manifest_path=manifest_path)
    if not reviewed:
        raise ValueError("No reviewed measures were found")

    frozen_rows = []
    for measure in reviewed:
        candidate_payload = _load_object(measure.candidates_path, "Candidate artifact")
        with Image.open(measure.image_path) as opened:
            image = opened.convert("RGB")
        proposals, considered_pairs = propose_hollow_notehead_centers(
            image,
            candidate_payload,
        )
        frozen_rows.append(
            {
                "identity": measure.identity,
                "lane": measure.lane,
                "source": {
                    "image": _file_record(measure.image_path, root),
                    "candidates": _file_record(measure.candidates_path, root),
                },
                "proposal_artifact": {
                    "schema_version": 1,
                    "kind": PROPOSAL_KIND,
                    "ground_truth_coordinates_used": [],
                    "considered_pair_count": len(considered_pairs),
                    "considered_pairs": considered_pairs,
                    "proposal_count": len(proposals),
                    "proposals": [_proposal_json(item) for item in proposals],
                },
            }
        )

    # Truth files are opened only after every proposal is fixed.
    truth_centers = [_load_truth_centers(measure) for measure in reviewed]
    evaluated_rows = [
        _evaluate_frozen_row(row, measure, centers)
        for row, measure, centers in zip(
            frozen_rows,
            reviewed,
            truth_centers,
            strict=True,
        )
    ]
    summary = _summarize(evaluated_rows)
    report = {
        "schema_version": 1,
        "kind": REPORT_KIND,
        "split_status": "consumed_postmortem",
        "eligible_for_heldout_claim": False,
        "detector": {
            "geometry": {
                "candidate_score_min": MIN_CANDIDATE_SCORE,
                "dx_spacing": [MIN_DX_SPACING, MAX_DX_SPACING],
                "dy_spacing": [MIN_DY_SPACING, MAX_DY_SPACING],
            },
            "pixel_gate": {
                "min_ring_density": MIN_RING_DENSITY,
                "min_ring_coverage": MIN_RING_COVERAGE,
                "min_hole_area_spacing_squared": MIN_HOLE_AREA_SPACING_SQUARED,
                "max_hole_center_distance_spacing": MAX_HOLE_CENTER_DISTANCE_SPACING,
                "min_hole_support_alignment": MIN_HOLE_SUPPORT_ALIGNMENT,
                "max_hole_axis_ratio": MAX_HOLE_AXIS_RATIO,
                "open_contour_min_ring_density": MIN_OPEN_RING_DENSITY,
                "open_contour_max_inner_density": MAX_OPEN_INNER_DENSITY,
                "contour_close_radius_px": CONTOUR_CLOSE_RADIUS_PX,
                "staff_lines_suppressed": True,
            },
            "evaluation_match_tolerance_spacing": MATCH_TOLERANCE_SPACING,
        },
        "summary": summary,
        "measures": evaluated_rows,
        "provenance": {
            "scope": "spike_only_consumed_evaluation",
            "proposal_generation_is_gt_blind": True,
            "proposal_function_inputs": ["source_image", "candidate_artifact"],
            "truth_access_policy": (
                "hash-pinned truth files are opened only after all proposals are frozen"
            ),
            "fixture_manifest": _file_record(manifest_path, root),
            "script": _file_record(Path(__file__).resolve(), root),
        },
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir()
    try:
        overlays_root = temp_dir / "overlays"
        for row, measure in zip(evaluated_rows, reviewed, strict=True):
            overlay_path = _overlay_path(overlays_root, measure.identity)
            _render_overlay(measure, row, overlay_path)
            row["artifacts"] = {
                "overlay": _local_record(overlay_path, temp_dir),
            }
        _write_json(temp_dir / "report.json", report)
        _write_json(
            temp_dir / "manifest.json",
            {
                "schema_version": 1,
                "kind": f"{REPORT_KIND}_manifest",
                "create_once": True,
                "report": _local_record(temp_dir / "report.json", temp_dir),
                "measure_count": len(evaluated_rows),
                "summary": summary,
            },
        )
        temp_dir.rename(destination)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return destination


def propose_hollow_notehead_centers(
    image: Image.Image,
    candidate_payload: Mapping[str, Any],
) -> tuple[list[HollowProposal], list[dict[str, Any]]]:
    """Return accepted proposals and diagnostics for every geometry-qualified pair."""
    staff_lines = _staff_lines(candidate_payload)
    spacing = _staff_spacing(candidate_payload, staff_lines)
    candidates = _candidate_rows(candidate_payload)
    gray = ImageOps.grayscale(image)
    threshold = estimate_ink_threshold(gray)
    raw_mask = [
        [gray.getpixel((x, y)) < threshold for x in range(gray.width)] for y in range(gray.height)
    ]
    closed_raw_mask = _close_ink_mask(raw_mask, radius=CONTOUR_CLOSE_RADIUS_PX)
    mask = _staff_suppressed_mask(
        gray,
        threshold=threshold,
        staff_lines=staff_lines,
        spacing=spacing,
    )
    accepted = []
    considered = []
    for index, first in enumerate(candidates):
        for second in candidates[index + 1 :]:
            geometry = _pair_geometry(first, second, spacing=spacing, staff_lines=staff_lines)
            if geometry is None:
                continue
            features = _hollow_pixel_features(
                mask,
                raw_mask=closed_raw_mask,
                center=geometry["center"],
                support_vector=geometry["support_vector"],
                spacing=spacing,
            )
            has_closed_hollow_shape = (
                features["ring_density"] >= MIN_RING_DENSITY
                and features["ring_coverage"] >= MIN_RING_COVERAGE
                and features["hole_area_spacing_squared"] >= MIN_HOLE_AREA_SPACING_SQUARED
                and features["hole_center_distance_spacing"] <= MAX_HOLE_CENTER_DISTANCE_SPACING
                and features["hole_support_alignment"] >= MIN_HOLE_SUPPORT_ALIGNMENT
                and features["hole_axis_ratio"] <= MAX_HOLE_AXIS_RATIO
            )
            has_open_hollow_shape = (
                features["hole_area_spacing_squared"] == 0.0
                and features["ring_density"] >= MIN_OPEN_RING_DENSITY
                and features["ring_coverage"] >= MIN_RING_COVERAGE
                and features["inner_density"] <= MAX_OPEN_INNER_DENSITY
            )
            contour_kind = (
                "closed" if has_closed_hollow_shape else "open" if has_open_hollow_shape else None
            )
            is_accepted = geometry["features"]["rises_to_right"] == 1.0 and contour_kind is not None
            score = _proposal_score(geometry, features)
            record = {
                "support_candidate_ids": [first["id"], second["id"]],
                "center": _point_json(geometry["center"]),
                "contour_kind": contour_kind,
                "score": score,
                "accepted": is_accepted,
                "features": {**geometry["features"], **features},
            }
            considered.append(record)
            if not is_accepted:
                continue
            accepted.append(
                HollowProposal(
                    support_candidate_ids=(str(first["id"]), str(second["id"])),
                    center=geometry["center"],
                    bbox=_bbox_union(first["bbox"], second["bbox"]),
                    contour_kind=contour_kind,
                    score=score,
                    features=record["features"],
                )
            )
    return _suppress_overlapping_proposals(accepted, spacing=spacing), considered


def load_reviewed_measures(
    repo_root: Path,
    *,
    manifest_path: Path | None = None,
) -> list[ReviewedMeasure]:
    """Load and hash-check fixture sources without opening review truth."""
    root = repo_root.resolve()
    source_manifest = (
        manifest_path.resolve()
        if manifest_path is not None
        else (root / DEFAULT_FIXTURE_MANIFEST).resolve()
    )
    manifest = _load_object(source_manifest, "Hollow-notehead fixture manifest")
    if manifest.get("kind") != "vlm_melody_hollow_notehead_fixture_manifest":
        raise ValueError("Unexpected hollow-notehead fixture manifest kind")
    if manifest.get("split_status") != "consumed_postmortem":
        raise ValueError("Hollow-notehead fixture manifest must be consumed")
    if manifest.get("eligible_for_heldout_claim") is not False:
        raise ValueError("Hollow-notehead fixture manifest cannot support heldout claims")
    rows = manifest.get("measures")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Hollow-notehead fixture manifest has no measures")
    if manifest.get("measure_count") != len(rows):
        raise ValueError("Hollow-notehead fixture manifest count mismatch")

    result = []
    seen = set()
    allowed_lanes = {
        "human_promoted_aviador",
        "agent_adjudicated_carrizal",
        "human_adjudicated_coqueteos",
    }
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise ValueError(f"Fixture measure {index} must be an object")
        identity = _identity(_required_object(raw, "identity", f"Fixture measure {index}"))
        identity_key = (
            identity["slug"],
            identity["system_index"],
            identity["system_measure_index"],
        )
        if identity_key in seen:
            raise ValueError(f"Duplicate fixture measure identity: {identity_key}")
        seen.add(identity_key)
        lane = str(raw.get("lane", ""))
        if lane not in allowed_lanes:
            raise ValueError(f"Unsupported fixture review lane: {lane!r}")
        truth_source = _required_object(
            raw,
            "truth_source",
            f"Fixture measure {index}",
        )
        truth_kind = str(truth_source.get("kind", ""))
        expected_truth_kind = {
            "human_promoted_aviador": "promoted_notehead_review",
            "agent_adjudicated_carrizal": "carrizal_adjudication",
            "human_adjudicated_coqueteos": "human_candidate_decision",
        }[lane]
        if truth_kind != expected_truth_kind:
            raise ValueError(f"Fixture truth kind {truth_kind!r} does not match lane {lane!r}")
        truth_measure_index = truth_source.get("review_measure_index")
        if truth_kind == "carrizal_adjudication":
            if (
                not isinstance(truth_measure_index, int)
                or isinstance(truth_measure_index, bool)
                or truth_measure_index <= 0
            ):
                raise ValueError("Carrizal fixture requires a positive review measure index")
        elif truth_measure_index is not None:
            raise ValueError("Only Carrizal fixtures use review_measure_index")
        image_path = _validate_pinned_file_record(
            _required_object(raw, "image", f"Fixture measure {index}"),
            root=root,
            label=f"Fixture measure {index} image",
        )
        candidates_path = _validate_pinned_file_record(
            _required_object(raw, "candidates", f"Fixture measure {index}"),
            root=root,
            label=f"Fixture measure {index} candidates",
        )
        candidate_payload = _load_object(candidates_path, "Fixture candidate artifact")
        for field in ("slug", "system_index", "system_measure_index"):
            if candidate_payload.get(field) != identity[field]:
                raise ValueError(f"Fixture candidate artifact {field} mismatch")
        recorded_image = _resolve(
            root=root,
            value=candidate_payload.get("source_image_path"),
        )
        if recorded_image != image_path:
            raise ValueError("Fixture candidate artifact points at a different image")
        result.append(
            ReviewedMeasure(
                identity=identity,
                lane=lane,
                image_path=image_path,
                candidates_path=candidates_path,
                truth_source_path=_validate_pinned_file_record(
                    truth_source,
                    root=root,
                    label=f"Fixture measure {index} truth",
                ),
                truth_source_kind=truth_kind,
                truth_measure_index=truth_measure_index,
            )
        )
    return sorted(
        result,
        key=lambda item: (
            item.identity["slug"],
            item.identity["system_index"],
            item.identity["system_measure_index"],
        ),
    )


def _load_truth_centers(
    measure: ReviewedMeasure,
) -> tuple[tuple[float, float], ...]:
    review = _load_object(measure.truth_source_path, "Reviewed truth source")
    candidate_payload = _load_object(measure.candidates_path, "Candidate artifact")
    candidate_by_id = {str(item["id"]): item for item in candidate_payload.get("candidates", [])}
    if measure.truth_source_kind == "promoted_notehead_review":
        if review.get("kind") != "vlm_melody_notehead_candidate_review":
            raise ValueError("Unexpected promoted notehead review kind")
        if _identity(_required_object(review, "identity", "Promoted review")) != measure.identity:
            raise ValueError("Promoted review identity mismatch")
        return tuple(
            _point(_required_object(head, "center", "Promoted reviewed head"))
            for head in review.get("final_noteheads", [])
        )
    if measure.truth_source_kind == "carrizal_adjudication":
        review_identity = _required_object(review, "identity", "Carrizal review")
        if (
            str(review_identity.get("slug")) != measure.identity["slug"]
            or int(review_identity.get("system_index", 0)) != measure.identity["system_index"]
        ):
            raise ValueError("Carrizal review identity mismatch")
        rows = review.get("measures")
        index = int(measure.truth_measure_index or 0)
        if not isinstance(rows, list) or not 1 <= index <= len(rows):
            raise ValueError("Carrizal review measure index is out of range")
        truth = []
        for head in rows[index - 1].get("heads", []):
            selection = _required_object(head, "selection", "Carrizal reviewed head")
            if selection.get("kind") == "candidate":
                candidate_id = str(selection.get("candidate_id", ""))
                if candidate_id not in candidate_by_id:
                    raise ValueError(f"Unknown Carrizal candidate id: {candidate_id}")
                truth.append(_point(candidate_by_id[candidate_id]["center"]))
            elif selection.get("kind") == "manual":
                truth.append(_point(_required_object(selection, "center", "Carrizal manual head")))
            else:
                raise ValueError(f"Unsupported Carrizal selection: {selection!r}")
        return tuple(truth)
    if measure.truth_source_kind == "human_candidate_decision":
        if review.get("kind") != "vlm_melody_consumed_human_candidate_review_decision":
            raise ValueError("Unexpected human candidate decision kind")
        if _identity(_required_object(review, "identity", "Human decision")) != measure.identity:
            raise ValueError("Human candidate decision identity mismatch")
        truth = []
        for head in review.get("accepted_heads", []):
            if "candidate_id" in head:
                candidate_id = str(head["candidate_id"])
                if candidate_id not in candidate_by_id:
                    raise ValueError(f"Unknown human-reviewed candidate id: {candidate_id}")
                truth.append(_point(candidate_by_id[candidate_id]["center"]))
                continue
            derived = _required_object(head, "derived_from", "Derived reviewed head")
            support_ids = derived.get("support_candidate_ids")
            if not isinstance(support_ids, list) or len(support_ids) < 2:
                raise ValueError("Derived reviewed head requires at least two support ids")
            support = []
            for candidate_id_raw in support_ids:
                candidate_id = str(candidate_id_raw)
                if candidate_id not in candidate_by_id:
                    raise ValueError(f"Unknown derived support candidate id: {candidate_id}")
                support.append(_point(candidate_by_id[candidate_id]["center"]))
            truth.append(
                (
                    sum(point[0] for point in support) / len(support),
                    sum(point[1] for point in support) / len(support),
                )
            )
        return tuple(truth)
    raise ValueError(f"Unsupported truth source kind: {measure.truth_source_kind!r}")


def _pair_geometry(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    spacing: float,
    staff_lines: Sequence[float],
) -> dict[str, Any] | None:
    if min(float(first.get("score", 0.0)), float(second.get("score", 0.0))) < MIN_CANDIDATE_SCORE:
        return None
    first_center = _point(first["center"])
    second_center = _point(second["center"])
    signed_dx = second_center[0] - first_center[0]
    signed_dy = second_center[1] - first_center[1]
    dx_spacing = abs(signed_dx) / spacing
    dy_spacing = abs(signed_dy) / spacing
    if not MIN_DX_SPACING <= dx_spacing <= MAX_DX_SPACING:
        return None
    if not MIN_DY_SPACING <= dy_spacing <= MAX_DY_SPACING:
        return None
    center = (
        (first_center[0] + second_center[0]) / 2.0,
        (first_center[1] + second_center[1]) / 2.0,
    )
    if not staff_lines[0] - 1.5 * spacing <= center[1] <= staff_lines[-1] + 1.5 * spacing:
        return None
    distance_spacing = math.hypot(signed_dx, signed_dy) / spacing
    return {
        "center": center,
        "support_vector": (signed_dx, signed_dy),
        "features": {
            "dx_spacing": round(dx_spacing, 6),
            "dy_spacing": round(dy_spacing, 6),
            "distance_spacing": round(distance_spacing, 6),
            "minimum_candidate_score": round(
                min(float(first["score"]), float(second["score"])),
                6,
            ),
            "rises_to_right": float(signed_dx * signed_dy < 0),
        },
    }


def _hollow_pixel_features(
    mask: Sequence[Sequence[bool]],
    *,
    raw_mask: Sequence[Sequence[bool]],
    center: tuple[float, float],
    support_vector: tuple[float, float],
    spacing: float,
) -> dict[str, float]:
    height = len(mask)
    width = len(mask[0]) if mask else 0
    distance = math.hypot(*support_vector)
    major_x = support_vector[0] / distance
    major_y = support_vector[1] / distance
    minor_x = -major_y
    minor_y = major_x
    major_radius = max(distance * 0.68, spacing * 0.68)
    minor_radius = spacing * 0.56
    radius = math.ceil(max(major_radius, minor_radius) * 1.2)
    inner_ink = inner_pixels = ring_ink = ring_pixels = 0
    sector_ink = [0] * 12
    sector_pixels = [0] * 12
    for y in range(
        max(0, math.floor(center[1] - radius)), min(height, math.ceil(center[1] + radius + 1))
    ):
        for x in range(
            max(0, math.floor(center[0] - radius)),
            min(width, math.ceil(center[0] + radius + 1)),
        ):
            delta_x = x - center[0]
            delta_y = y - center[1]
            u = (delta_x * major_x + delta_y * major_y) / major_radius
            v = (delta_x * minor_x + delta_y * minor_y) / minor_radius
            radial = math.hypot(u, v)
            if radial <= 0.38:
                inner_pixels += 1
                inner_ink += int(mask[y][x])
            if 0.56 <= radial <= 1.16:
                ring_pixels += 1
                value = int(mask[y][x])
                ring_ink += value
                angle = (math.atan2(v, u) + math.pi) / (2.0 * math.pi)
                sector = min(11, int(angle * 12))
                sector_pixels[sector] += 1
                sector_ink[sector] += value
    covered = sum(
        ink >= max(1, round(pixels * 0.05))
        for ink, pixels in zip(sector_ink, sector_pixels, strict=True)
        if pixels
    )
    available = sum(bool(pixels) for pixels in sector_pixels)
    return {
        "inner_density": round(inner_ink / inner_pixels, 6) if inner_pixels else 1.0,
        "ring_density": round(ring_ink / ring_pixels, 6) if ring_pixels else 0.0,
        "ring_coverage": round(covered / available, 6) if available else 0.0,
        **_enclosed_hole_features(
            raw_mask,
            center=center,
            major_radius=major_radius,
            minor_radius=minor_radius,
            support_vector=support_vector,
            spacing=spacing,
        ),
    }


def _enclosed_hole_features(
    ink_mask: Sequence[Sequence[bool]],
    *,
    center: tuple[float, float],
    major_radius: float,
    minor_radius: float,
    support_vector: tuple[float, float],
    spacing: float,
) -> dict[str, float]:
    height = len(ink_mask)
    width = len(ink_mask[0]) if ink_mask else 0
    radius_x = math.ceil(max(major_radius, minor_radius) * 1.25)
    radius_y = radius_x
    left = max(0, math.floor(center[0] - radius_x))
    top = max(0, math.floor(center[1] - radius_y))
    right = min(width, math.ceil(center[0] + radius_x + 1))
    bottom = min(height, math.ceil(center[1] + radius_y + 1))
    if left >= right or top >= bottom:
        return _missing_hole_features()

    patch_width = right - left
    patch_height = bottom - top
    exterior = set()
    stack = [
        (x, y)
        for x in range(patch_width)
        for y in (0, patch_height - 1)
        if not ink_mask[top + y][left + x]
    ]
    stack.extend(
        (x, y)
        for y in range(1, patch_height - 1)
        for x in (0, patch_width - 1)
        if not ink_mask[top + y][left + x]
    )
    while stack:
        point = stack.pop()
        if point in exterior:
            continue
        x, y = point
        if ink_mask[top + y][left + x]:
            continue
        exterior.add(point)
        for neighbor_x, neighbor_y in (
            (x - 1, y),
            (x + 1, y),
            (x, y - 1),
            (x, y + 1),
        ):
            if 0 <= neighbor_x < patch_width and 0 <= neighbor_y < patch_height:
                stack.append((neighbor_x, neighbor_y))

    unvisited = {
        (x, y)
        for y in range(patch_height)
        for x in range(patch_width)
        if not ink_mask[top + y][left + x] and (x, y) not in exterior
    }
    holes = []
    while unvisited:
        component = set()
        stack = [unvisited.pop()]
        while stack:
            point = stack.pop()
            if point in component:
                continue
            component.add(point)
            x, y = point
            for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    stack.append(neighbor)
        if component:
            centroid = (
                left + sum(point[0] for point in component) / len(component),
                top + sum(point[1] for point in component) / len(component),
            )
            holes.append((component, centroid))
    if not holes:
        return _missing_hole_features()
    component, centroid = min(
        holes,
        key=lambda item: (
            math.dist(item[1], center) / spacing,
            -len(item[0]),
        ),
    )
    area = len(component)
    min_x = min(point[0] for point in component)
    max_x = max(point[0] for point in component)
    min_y = min(point[1] for point in component)
    max_y = max(point[1] for point in component)
    bbox_area = (max_x - min_x + 1) * (max_y - min_y + 1)
    component_center = (
        sum(point[0] for point in component) / area,
        sum(point[1] for point in component) / area,
    )
    variance_x = sum((point[0] - component_center[0]) ** 2 for point in component) / area
    variance_y = sum((point[1] - component_center[1]) ** 2 for point in component) / area
    covariance = (
        sum(
            (point[0] - component_center[0]) * (point[1] - component_center[1])
            for point in component
        )
        / area
    )
    trace = variance_x + variance_y
    discriminant = math.sqrt(max(0.0, (variance_x - variance_y) ** 2 + 4 * covariance**2))
    major_eigenvalue = (trace + discriminant) / 2.0
    minor_eigenvalue = max(0.0, (trace - discriminant) / 2.0)
    if abs(covariance) > 1e-9:
        major_axis = (major_eigenvalue - variance_y, covariance)
    elif variance_x >= variance_y:
        major_axis = (1.0, 0.0)
    else:
        major_axis = (0.0, 1.0)
    major_axis_length = math.hypot(*major_axis)
    support_length = math.hypot(*support_vector)
    orientation_alignment = (
        abs(major_axis[0] * support_vector[0] + major_axis[1] * support_vector[1])
        / (major_axis_length * support_length)
        if major_axis_length and support_length
        else 0.0
    )
    return {
        "hole_area_spacing_squared": round(area / (spacing * spacing), 6),
        "hole_center_distance_spacing": round(
            math.dist(centroid, center) / spacing,
            6,
        ),
        "hole_bbox_fill": round(area / bbox_area, 6),
        "hole_bbox_width_spacing": round((max_x - min_x + 1) / spacing, 6),
        "hole_bbox_height_spacing": round((max_y - min_y + 1) / spacing, 6),
        "hole_axis_ratio": round(
            math.sqrt(minor_eigenvalue / major_eigenvalue) if major_eigenvalue else 1.0,
            6,
        ),
        "hole_support_alignment": round(orientation_alignment, 6),
    }


def _missing_hole_features() -> dict[str, float]:
    return {
        "hole_area_spacing_squared": 0.0,
        "hole_center_distance_spacing": 999.0,
        "hole_bbox_fill": 0.0,
        "hole_bbox_width_spacing": 0.0,
        "hole_bbox_height_spacing": 0.0,
        "hole_axis_ratio": 1.0,
        "hole_support_alignment": 0.0,
    }


def _close_ink_mask(
    mask: Sequence[Sequence[bool]],
    *,
    radius: int,
) -> list[list[bool]]:
    """Close small contour gaps without changing the source image."""
    height = len(mask)
    width = len(mask[0]) if mask else 0
    if radius <= 0 or not width:
        return [list(row) for row in mask]
    dilated = [
        [
            any(
                mask[neighbor_y][neighbor_x]
                for neighbor_y in range(max(0, y - radius), min(height, y + radius + 1))
                for neighbor_x in range(max(0, x - radius), min(width, x + radius + 1))
            )
            for x in range(width)
        ]
        for y in range(height)
    ]
    return [
        [
            all(
                0 <= neighbor_x < width
                and 0 <= neighbor_y < height
                and dilated[neighbor_y][neighbor_x]
                for neighbor_y in range(y - radius, y + radius + 1)
                for neighbor_x in range(x - radius, x + radius + 1)
            )
            for x in range(width)
        ]
        for y in range(height)
    ]


def _staff_suppressed_mask(
    gray: Image.Image,
    *,
    threshold: int,
    staff_lines: Sequence[float],
    spacing: float,
) -> list[list[bool]]:
    width, height = gray.size
    suppression_radius = max(1, round(spacing * 0.05))
    suppressed_rows = {
        y
        for line in staff_lines
        for y in range(
            max(0, round(line) - suppression_radius),
            min(height, round(line) + suppression_radius + 1),
        )
    }
    return [
        [y not in suppressed_rows and gray.getpixel((x, y)) < threshold for x in range(width)]
        for y in range(height)
    ]


def _proposal_score(
    geometry: Mapping[str, Any],
    pixels: Mapping[str, float],
) -> float:
    geometry_features = geometry["features"]
    score = (
        0.25 * float(geometry_features["minimum_candidate_score"])
        + 0.30 * float(pixels["ring_coverage"])
        + 0.25 * min(1.0, float(pixels["ring_density"]) / 0.30)
        + 0.20
        * min(
            1.0,
            float(pixels["hole_area_spacing_squared"]) / (MIN_HOLE_AREA_SPACING_SQUARED * 3.0),
        )
    )
    return round(score, 6)


def _suppress_overlapping_proposals(
    proposals: Sequence[HollowProposal],
    *,
    spacing: float,
) -> list[HollowProposal]:
    selected = []
    for proposal in sorted(proposals, key=lambda item: (-item.score, item.center)):
        if any(
            math.dist(proposal.center, existing.center) < spacing * 0.55 for existing in selected
        ):
            continue
        selected.append(proposal)
    return sorted(selected, key=lambda item: item.center)


def _evaluate_frozen_row(
    row: Mapping[str, Any],
    measure: ReviewedMeasure,
    truth_centers: Sequence[tuple[float, float]],
) -> dict[str, Any]:
    candidate_payload = _load_object(measure.candidates_path, "Candidate artifact")
    candidates = candidate_payload["candidates"]
    spacing = _staff_spacing(candidate_payload, _staff_lines(candidate_payload))
    tolerance = spacing * MATCH_TOLERANCE_SPACING
    baseline_centers = [_point(item["center"]) for item in candidates]
    proposal_records = row["proposal_artifact"]["proposals"]
    proposal_centers = [_point(item["center"]) for item in proposal_records]
    baseline_matches = _match_points(
        baseline_centers,
        truth_centers,
        tolerance=tolerance,
    )
    baseline_truth = {truth_index for _, truth_index in baseline_matches}
    available_truth_indices = [
        index for index in range(len(truth_centers)) if index not in baseline_truth
    ]
    proposal_matches = _match_points(
        proposal_centers,
        [truth_centers[index] for index in available_truth_indices],
        tolerance=tolerance,
    )
    recovered_by_proposal = {
        proposal_index: available_truth_indices[available_index]
        for proposal_index, available_index in proposal_matches
    }
    augmented_truth = baseline_truth | set(recovered_by_proposal.values())
    proposal_assessment = []
    for index, proposal in enumerate(proposal_records):
        center = proposal_centers[index]
        distances = [math.dist(center, truth_center) for truth_center in truth_centers]
        nearest_index = min(range(len(distances)), key=distances.__getitem__) if distances else None
        if index in recovered_by_proposal:
            outcome = "recovered_truth"
            matched_truth_index = recovered_by_proposal[index]
        elif nearest_index is not None and distances[nearest_index] <= tolerance:
            outcome = "duplicate_truth"
            matched_truth_index = nearest_index
        else:
            outcome = "false_proposal"
            matched_truth_index = None
        proposal_assessment.append(
            {
                **proposal,
                "outcome": outcome,
                "nearest_truth_index": nearest_index,
                "nearest_truth_distance_px": (
                    round(distances[nearest_index], 6) if nearest_index is not None else None
                ),
                "matched_truth_index": matched_truth_index,
            }
        )
    return {
        **row,
        "proposal_artifact": {
            **row["proposal_artifact"],
            "proposals": proposal_assessment,
        },
        "evaluation": {
            "truth_notehead_count": len(truth_centers),
            "match_tolerance_px": round(tolerance, 6),
            "baseline_matched_truth": len(baseline_truth),
            "augmented_matched_truth": len(augmented_truth),
            "recovered_truth_count": len(recovered_by_proposal),
            "proposal_outcomes": _counts(item["outcome"] for item in proposal_assessment),
        },
    }


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lanes = {}
    for row in rows:
        lane = str(row["lane"])
        lane_rows = lanes.setdefault(lane, [])
        lane_rows.append(row)
    return {
        "measure_count": len(rows),
        "proposal_count": sum(row["proposal_artifact"]["proposal_count"] for row in rows),
        "recovered_truth_count": sum(row["evaluation"]["recovered_truth_count"] for row in rows),
        "proposal_outcomes": _counts(
            proposal["outcome"]
            for row in rows
            for proposal in row["proposal_artifact"]["proposals"]
        ),
        "lanes": {
            lane: {
                "measure_count": len(lane_rows),
                "proposal_count": sum(
                    row["proposal_artifact"]["proposal_count"] for row in lane_rows
                ),
                "recovered_truth_count": sum(
                    row["evaluation"]["recovered_truth_count"] for row in lane_rows
                ),
                "proposal_outcomes": _counts(
                    proposal["outcome"]
                    for row in lane_rows
                    for proposal in row["proposal_artifact"]["proposals"]
                ),
            }
            for lane, lane_rows in sorted(lanes.items())
        },
    }


def _render_overlay(
    measure: ReviewedMeasure,
    row: Mapping[str, Any],
    destination: Path,
) -> None:
    with Image.open(measure.image_path) as opened:
        raw = opened.convert("RGB")
    scale = max(1, (520 + raw.width - 1) // raw.width)
    if scale > 1:
        raw = raw.resize(
            (raw.width * scale, raw.height * scale),
            Image.Resampling.NEAREST,
        )
    header = 62
    canvas = Image.new("RGB", (raw.width, raw.height + header), "white")
    canvas.paste(raw, (0, header))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    identity = measure.identity
    draw.text(
        (6, 5),
        (
            f"{identity['slug']} s{identity['system_index']:03d} "
            f"m{identity['system_measure_index']:03d}"
        ),
        fill="black",
        font=font,
    )
    draw.text(
        (6, 24),
        "blue=candidate pair  green=recovered  orange=duplicate  red=false",
        fill="black",
        font=font,
    )
    candidate_payload = _load_object(measure.candidates_path, "Candidate artifact")
    candidate_by_id = {str(item["id"]): item for item in candidate_payload["candidates"]}
    for proposal in row["proposal_artifact"]["proposals"]:
        outcome = proposal["outcome"]
        color = {
            "recovered_truth": (0, 145, 70),
            "duplicate_truth": (225, 125, 0),
            "false_proposal": (215, 45, 45),
        }[outcome]
        for candidate_id in proposal["support_candidate_ids"]:
            candidate = candidate_by_id[candidate_id]
            bbox = candidate["bbox"]
            draw.rectangle(
                (
                    int(bbox["left"]) * scale,
                    int(bbox["top"]) * scale + header,
                    int(bbox["right"]) * scale,
                    int(bbox["bottom"]) * scale + header,
                ),
                outline=(70, 90, 210),
                width=2,
            )
        center = _point(proposal["center"])
        x = round(center[0] * scale)
        y = round(center[1] * scale) + header
        radius = max(6, 4 * scale)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
        draw.line((x - radius - 3, y, x + radius + 3, y), fill=color, width=2)
        draw.line((x, y - radius - 3, x, y + radius + 3), fill=color, width=2)
        draw.text(
            (x + radius + 2, y - 10),
            f"{'+'.join(proposal['support_candidate_ids'])} {outcome}",
            fill=color,
            font=font,
            stroke_width=1,
            stroke_fill="white",
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def _match_points(
    candidates: Sequence[tuple[float, float]],
    truth: Sequence[tuple[float, float]],
    *,
    tolerance: float,
) -> list[tuple[int, int]]:
    pairs = sorted(
        (
            (math.dist(candidate, target), candidate_index, truth_index)
            for candidate_index, candidate in enumerate(candidates)
            for truth_index, target in enumerate(truth)
            if math.dist(candidate, target) <= tolerance
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    used_candidates = set()
    used_truth = set()
    matches = []
    for _, candidate_index, truth_index in pairs:
        if candidate_index in used_candidates or truth_index in used_truth:
            continue
        used_candidates.add(candidate_index)
        used_truth.add(truth_index)
        matches.append((candidate_index, truth_index))
    return matches


def _candidate_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("candidates")
    if not isinstance(rows, list):
        raise ValueError("Candidate artifact candidates must be a list")
    return [dict(row) for row in rows if isinstance(row, dict)]


def _staff_lines(payload: Mapping[str, Any]) -> tuple[float, ...]:
    values = payload.get("staff_lines_y_px")
    if not isinstance(values, list) or len(values) != 5:
        raise ValueError("Candidate artifact must contain five staff lines")
    return tuple(float(value) for value in values)


def _staff_spacing(
    payload: Mapping[str, Any],
    staff_lines: Sequence[float],
) -> float:
    value = payload.get("staff_spacing_px")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    differences = [
        second - first for first, second in zip(staff_lines, staff_lines[1:], strict=True)
    ]
    spacing = sum(differences) / len(differences)
    if spacing <= 0:
        raise ValueError("Invalid staff spacing")
    return spacing


def _bbox_union(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> tuple[int, int, int, int]:
    return (
        min(int(first["left"]), int(second["left"])),
        min(int(first["top"]), int(second["top"])),
        max(int(first["right"]), int(second["right"])),
        max(int(first["bottom"]), int(second["bottom"])),
    )


def _proposal_json(proposal: HollowProposal) -> dict[str, Any]:
    return {
        "support_candidate_ids": list(proposal.support_candidate_ids),
        "center": _point_json(proposal.center),
        "contour_kind": proposal.contour_kind,
        "bbox": {
            "left": proposal.bbox[0],
            "top": proposal.bbox[1],
            "right": proposal.bbox[2],
            "bottom": proposal.bbox[3],
        },
        "score": proposal.score,
        "features": proposal.features,
    }


def _identity(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "slug": str(value["slug"]),
        "system_index": int(value["system_index"]),
        "system_measure_index": int(value["system_measure_index"]),
    }


def _point(value: Mapping[str, Any]) -> tuple[float, float]:
    return float(value["x"]), float(value["y"])


def _point_json(value: tuple[float, float]) -> dict[str, float]:
    return {"x": round(value[0], 6), "y": round(value[1], 6)}


def _resolve(*, root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _validate_pinned_file_record(
    record: Mapping[str, Any],
    *,
    root: Path,
    label: str,
) -> Path:
    path = _resolve(root=root, value=record.get("path"))
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    expected_hash = record.get("sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"{label} must provide a SHA256")
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f"{label} hash drift: expected {expected_hash}, got {actual_hash}")
    return path


def _counts(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def _overlay_path(root: Path, identity: Mapping[str, Any]) -> Path:
    return (
        root
        / str(identity["slug"])
        / f"system_{int(identity['system_index']):03d}"
        / f"measure_{int(identity['system_measure_index']):03d}.png"
    )


def _required_object(value: Mapping[str, Any], key: str, label: str) -> dict[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"{label} {key} must be an object")
    return result


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_record(path: Path, repo_root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display_path = resolved.relative_to(repo_root).as_posix()
    except ValueError:
        display_path = str(resolved)
    return {
        "path": display_path,
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "bytes": resolved.stat().st_size,
    }


def _local_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


if __name__ == "__main__":
    raise SystemExit(main())
